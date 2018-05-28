import matplotlib

matplotlib.use('Agg')

from models import generator, discriminator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1

data_dir = '/home/0316223/Datasets/faces'
scale_height = scale_width = 64
# parameters for training
batch_size = 128
epochs = 150
latent_size = 100
d_learning_rate = 0.002
g_learning_rate = 0.002
beta1 = 0.5
weight_decay = 0.
is_training = tf.placeholder_with_default(True, shape=(), name='dropout_control')


def origin_img(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resize = tf.image.resize_images(image_float, [scale_height, scale_width])
    return image_resize * 2 - 1


def flip_img(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_flip = tf.image.flip_left_right(image_decoded)
    image_float = tf.image.convert_image_dtype(image_flip, tf.float32)
    image_resize = tf.image.resize_images(image_float, [72, 72])
    image_crop = tf.random_crop(image_resize, [scale_height, scale_width, 3])
    return image_crop * 2 - 1


print("Image preprocessing...")
# typical dataset usage
train_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
train_num = len(train_files)
train_files = tf.constant(train_files)
clean_data = tf.data.Dataset.from_tensor_slices(train_files)
clean_data = clean_data.map(origin_img)

aug_data = tf.data.Dataset.from_tensor_slices(train_files)
aug_data = aug_data.map(flip_img)

train_data = clean_data.concatenate(aug_data)
train_num *= 2
steps_per_epoch = train_num // batch_size + 1

features = train_data.make_one_shot_iterator().get_next()
X = tf.placeholder(dtype=tf.float32, shape=[None, scale_height, scale_width, 3], name='input_data')
Z = tf.placeholder(dtype=tf.float32, shape=[None, latent_size], name='input_z')


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

        fake_X = generator(Z, regularizer, is_training=is_training)
        fake_logits = discriminator(fake_X, regularizer, is_training=is_training)
        true_logits = discriminator(X, regularizer, is_training=is_training, reuse=True)

        # Add training ops into graph.
        with tf.variable_scope('train'):
            d_loss_true = tf.reduce_mean(tf.nn.l2_loss(true_logits-1))
            d_loss_fake = tf.reduce_mean(tf.nn.l2_loss(fake_logits))
            d_loss = d_loss_fake + d_loss_true
            d_loss = tf.identity(d_loss, name='d_loss')
            g_loss = tf.reduce_mean(tf.nn.l2_loss(fake_logits-1, name='g_loss'))

            d_loss += tf.losses.get_regularization_loss(scope='discriminator')
            g_loss += tf.losses.get_regularization_loss(scope='generator')
            d_vars = tf.trainable_variables(scope='discriminator')
            g_vars = tf.trainable_variables(scope='generator')

            global_step = tf.Variable(0, name='global_step', trainable=False,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

            d_optimizer = tf.train.AdamOptimizer(learning_rate=d_learning_rate, beta1=beta1)
            g_optimizer = tf.train.AdamOptimizer(learning_rate=g_learning_rate, beta1=beta1)
            d_train = d_optimizer.minimize(d_loss, global_step=global_step, var_list=d_vars, name='d_train_op')
            g_train = g_optimizer.minimize(g_loss, global_step=global_step, var_list=g_vars, name='g_train_op')

        sess.run(tf.global_variables_initializer())
        # Assign the required tensors to do the operation

        global_step_tensor = sess.graph.get_tensor_by_name('train/global_step:0')
        d_train_op = sess.graph.get_operation_by_name('train/d_train_op')
        g_train_op = sess.graph.get_operation_by_name('train/g_train_op')
        d_loss_tensor = sess.graph.get_tensor_by_name('train/d_loss:0')
        g_loss_tensor = sess.graph.get_tensor_by_name('train/g_loss:0')
        z_tensor = sess.graph.get_tensor_by_name('input_z:0')

        # Start training
        print('Start training ...')
        a = datetime.now().replace(microsecond=0)
        loss_history = []
        for i in range(epochs):
            total_d_loss = total_g_loss = 0
            np.random.shuffle(train_x)
            for j in range(steps_per_epoch):
                pos = j * batch_size
                nums = min(train_num, pos + batch_size) - pos
                noise = np.random.normal(size=(nums, latent_size))
                _, loss_value = sess.run([d_train_op, d_loss_tensor],
                                         feed_dict={X: train_x[pos:pos + nums], z_tensor: noise})
                total_d_loss += loss_value * nums

                _, loss_value = sess.run([g_train_op, g_loss_tensor], feed_dict={z_tensor: noise})
                total_g_loss += loss_value

            total_d_loss /= train_num
            total_g_loss /= train_num
            print("Iter: {}, Global step: {}, discriminator loss: {:.4f}, generator loss: {:.4f}".format(i + 1,
                                                                                                         global_step_tensor.eval(),
                                                                                                         total_d_loss,
                                                                                                         total_g_loss))
            loss_history.append((total_g_loss, total_d_loss))

        b = datetime.now().replace(microsecond=0)
        print("Time cost:", b - a)

        loss_history = np.array(loss_history).T
        plt.plot(loss_history[0], label='Generator')
        plt.plot(loss_history[1], label='Discriminator')
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.title("Training curve")
        plt.legend()
        plt.savefig("DCGAN_training_curve.png", dpi=100)
        plt.gcf().clear()

        # plot some images
        noise = np.random.normal(size=(64, latent_size))
        generate_img = sess.run(fake_X, feed_dict={z_tensor: noise, is_training: False})
        generate_img = (generate_img + 1) / 2
        generate_img = np.column_stack(generate_img.reshape([8, 8, scale_height, scale_width, 3]))
        generate_img = np.column_stack(generate_img)
        plt.imshow(generate_img)
        plt.title("Samples of DCGAN")
        plt.savefig("dcgan_generation.png", dpi=100)


if __name__ == '__main__':
    tf.app.run()
