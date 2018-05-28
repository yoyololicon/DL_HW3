import matplotlib

matplotlib.use('Agg')

from models import decoder, encoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.04

data_dir = '/home/0316223/Datasets/faces'
scale_height = scale_width = 96
# parameters for training
batch_size = 32
epochs = 30
latent_size = 50
init_learning_rate = 0.001
weight_decay = 0.0001
is_training = tf.placeholder_with_default(True, shape=(), name='dropout_control')


def read_img(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image_float


print("Image preprocessing...")
# typical dataset usage
train_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
train_num = len(train_files)
train_files = tf.constant(train_files)
train_data = tf.data.Dataset.from_tensor_slices(train_files)
train_data = train_data.map(read_img)
steps_per_epoch = train_num // batch_size + 1

features = train_data.make_one_shot_iterator().get_next()
X = tf.placeholder(dtype=tf.float32, shape=[None, scale_height, scale_width, 3], name='input_data')


def main(_):
    with tf.Session(config=config) as sess:
        train_x = []

        print("Start reading", train_num, "of training files ...")
        a = datetime.now().replace(microsecond=0)
        while True:
            try:
                imgs = sess.run(features)
                train_x.append(imgs)
            except tf.errors.OutOfRangeError:
                break
        b = datetime.now().replace(microsecond=0)
        print("Complete reading training files.")
        print("Time cost:", b - a)

        train_x = np.array(train_x)

        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)

        mean, var = encoder(X, latent_size, regularizer, is_training)
        std = tf.sqrt(var, name='z_std')
        epsilon = tf.random_normal(tf.shape(var), name='random_prob')
        sample_z = mean + epsilon * std
        sample_z = tf.identity(sample_z, name='input_z')
        decoded_x = decoder(sample_z, regularizer, is_training)

        # Add training ops into graph.
        with tf.variable_scope('train'):
            img_loss = tf.reduce_sum(tf.squared_difference(decoded_x, X), axis=[1, 2, 3])
            latent_loss = 0.5 * tf.reduce_sum(var + tf.square(mean) - 1 - tf.log(var), 1)
            loss = tf.reduce_mean(img_loss + latent_loss, name='loss_op')
            loss += tf.losses.get_regularization_loss()

            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

        sess.run(tf.global_variables_initializer())

        global_step_tensor = sess.graph.get_tensor_by_name('train/global_step:0')
        train_op = sess.graph.get_operation_by_name('train/train_op')
        loss_tensor = sess.graph.get_tensor_by_name('train/loss_op:0')
        z_tensor = sess.graph.get_tensor_by_name('input_z:0')
        prob_tensor = sess.graph.get_tensor_by_name('random_prob:0')

        # Start training
        print('Start training ...')
        a = datetime.now().replace(microsecond=0)
        loss_history = []
        for i in range(epochs):
            total_loss = 0
            np.random.shuffle(train_x)
            for j in range(steps_per_epoch):
                pos = j * batch_size
                nums = min(train_num, pos + batch_size) - pos
                _, loss_value = sess.run([train_op, loss_tensor], feed_dict={X: train_x[pos:pos + nums]})
                total_loss += loss_value * nums

            total_loss /= train_num
            print("Iter: {}, Global step: {}, loss: {:.4f}".format(i + 1, global_step_tensor.eval(), total_loss))
            loss_history.append(total_loss)

        b = datetime.now().replace(microsecond=0)

        print("Time cost:", b - a)

        plt.plot(loss_history, label='training loss')
        plt.xlabel("epochs")
        plt.ylabel("Totla loss")
        plt.title("Training curve")
        plt.savefig("batch_"+str(batch_size)+"_latent_"+str(latent_size)+ "_training_curve.png", dpi=100)
        plt.gcf().clear()

        # plot some images
        decoded_img, rand_prob = sess.run([decoded_x, prob_tensor], feed_dict={X: train_x[:64], is_training: False})
        test_img = train_x[:64].reshape([8, 8, 96, 96, 3])
        test_img = np.column_stack(test_img)
        test_img = np.column_stack(test_img)
        decoded_img = np.column_stack(decoded_img.reshape([8, 8, 96, 96, 3]))
        decoded_img = np.column_stack(decoded_img)


        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(test_img)
        ax[0].set_title("Before encode")
        ax[1].imshow(decoded_img)
        ax[1].set_title("After decode")
        fig.suptitle("batch size: "+str(batch_size)+", latent size: "+str(latent_size))
        plt.savefig("batch_"+str(batch_size)+"_latent_"+str(latent_size)+ "_images_comparison.png", dpi=150)         
        plt.gcf().clear()

        generate_img = sess.run(decoded_x, feed_dict={z_tensor: rand_prob, is_training: False})
        generate_img = np.column_stack(generate_img.reshape([8, 8, 96, 96, 3]))
        generate_img = np.column_stack(generate_img)
        plt.imshow(generate_img)
        plt.title("Random images")
        plt.savefig("batch_"+str(batch_size)+"_latent_"+str(latent_size)+ "_generation.png", dpi=100)

if __name__ == '__main__':
    tf.app.run()
