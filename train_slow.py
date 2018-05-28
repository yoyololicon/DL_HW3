import matplotlib

matplotlib.use('Agg')

from models import decoder, encoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.utils import shuffle

data_dir = '/home/0316223/Datasets/faces'
model_name = 'vae'
scale_height = scale_width = 96
# parameters for training
batch_size = 256
epochs = 20
latent_size = 512
init_learning_rate = 0.0001
weight_decay = 0.


def read_img(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_float = tf.image.resize_images(image_float, [scale_height, scale_width])
    return image_float


print("Image preprocessing...")
# typical dataset usage
train_files = [f for f in os.listdir(data_dir)]
train_num = len(train_files)
train_files = tf.constant([os.path.join(data_dir, f) for f in train_files])
train_data = tf.data.Dataset.from_tensor_slices(train_files)
train_data = train_data.map(read_img)
train_data = train_data.shuffle(1000).repeat()
train_data = train_data.batch(batch_size)
steps_per_epoch = train_num // batch_size + 1

features = train_data.make_one_shot_iterator().get_next()
#X = tf.placeholder(dtype=tf.float32, shape=[None, scale_height, scale_width, 3], name='input_data')
print(train_data.output_shapes)
def main(_):
    with tf.Session() as sess:

        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

        mean, var = encoder(features, latent_size, regularizer)
        std = tf.sqrt(var, name='z_std')
        epsilon = tf.random_normal(tf.shape(var))
        sample_z = mean + epsilon * std
        decoded_x = decoder(sample_z, regularizer)

        # Add training ops into graph.
        with tf.variable_scope('train'):
            img_loss = tf.reduce_sum(tf.squared_difference(decoded_x, features), axis=[1, 2, 3])
            latent_loss = 0.5 * tf.reduce_sum(var + tf.square(mean) - 1 - tf.log(var), 1)
            loss = tf.reduce_mean(img_loss + latent_loss, name='loss_op')
            loss += tf.losses.get_regularization_loss()

            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

        sess.run(tf.global_variables_initializer())
        # Assign the required tensors to do the operation

        saver = tf.train.Saver()

        global_step_tensor = sess.graph.get_tensor_by_name('train/global_step:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')

        # Start training
        print('Start training ...')
        a = datetime.now().replace(microsecond=0)
        loss_history = []
        for i in range(epochs):
            total_loss = 0
            for j in range(steps_per_epoch):
                _, loss_value, loss1, loss2 = sess.run([train_op, loss_tensor, img_loss, latent_loss])
                total_loss += loss_value
                #print(loss1.mean(), loss2.mean())
                if global_step_tensor.eval() % 1000 == 0:
                    saver.save(sess, './' + model_name, global_step=global_step_tensor)

            total_loss /= steps_per_epoch
            print("Iter: {}, Global step: {}, loss: {:.4f}".format(i, global_step_tensor.eval(), total_loss))
            loss_history.append(total_loss)

        b = datetime.now().replace(microsecond=0)

        print("Time cost:", b - a)
        saver.save(sess, './' + model_name, global_step=global_step_tensor)

        plt.plot(loss_history, label='training loss')
        plt.xlabel("epochs")
        plt.ylabel("Square loss")
        plt.title("Training curve")
        plt.savefig("Training curve.png", dpi=100)

        plt.gcf().clear()


if __name__ == '__main__':
    tf.app.run()
