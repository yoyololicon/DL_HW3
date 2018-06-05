import tensorflow as tf


def encoder(features, latent_size, regularizer, is_training):
    # input 96 x 96 x 3

    with tf.variable_scope('encoder'):
        conv1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=5, strides=2, padding='same', name='conv1',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=regularizer)  # 48 x 48
        conv1 = tf.layers.dropout(conv1, 0.2, training=is_training)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, padding='same', name='pool1')  # 24 x 24

        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=3, strides=2, padding='same', name='conv2',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=regularizer)  # 12 x 12
        conv2 = tf.layers.dropout(conv2, 0.2, training=is_training)

        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=3, strides=2, padding='same', name='conv3',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=regularizer)  # 6 x 6
        conv3 = tf.layers.dropout(conv3, 0.2, training=is_training)

        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=3, strides=2, padding='same', name='conv4',
                                 activation=tf.nn.leaky_relu, kernel_regularizer=regularizer)  # 3 x 3
        conv4 = tf.layers.dropout(conv4, 0.2, training=is_training)

        # flatten = tf.layers.flatten(conv4, name='flatten')  # 9 * 256
        flatten = tf.reduce_mean(conv4, axis=[1, 2])
        mean = tf.layers.dense(flatten, latent_size, kernel_regularizer=regularizer, name='z_mean')
        var = tf.exp(tf.layers.dense(flatten, latent_size, kernel_regularizer=regularizer), name='z_var')
        return mean, var


def decoder(sample_z, regularizer, is_training):
    with tf.variable_scope('decoder'):
        dense = tf.layers.dense(sample_z, 256, kernel_regularizer=regularizer, name='dense')
        reshape = tf.reshape(dense, shape=[-1, 1, 1, 256])

        deconv1 = tf.layers.conv2d_transpose(reshape, filters=256, kernel_size=3, strides=3, padding='same',
                                             name='deconv1', activation=tf.nn.leaky_relu,
                                             kernel_regularizer=regularizer)
        deconv1 = tf.layers.dropout(deconv1, 0.2, training=is_training)

        deconv2 = tf.layers.conv2d_transpose(deconv1, filters=128, kernel_size=3, strides=2, padding='same',
                                             name='deconv2', activation=tf.nn.leaky_relu,
                                             kernel_regularizer=regularizer)
        deconv2 = tf.layers.dropout(deconv2, 0.2, training=is_training)

        deconv3 = tf.layers.conv2d_transpose(deconv2, filters=64, kernel_size=3, strides=2, padding='same',
                                             name='deconv3', activation=tf.nn.leaky_relu,
                                             kernel_regularizer=regularizer)
        deconv3 = tf.layers.dropout(deconv3, 0.2, training=is_training)

        deconv4 = tf.layers.conv2d_transpose(deconv3, filters=64, kernel_size=3, strides=2, padding='same',
                                             name='deconv4', activation=tf.nn.leaky_relu,
                                             kernel_regularizer=regularizer)
        deconv4 = tf.layers.dropout(deconv4, 0.2, training=is_training)

        deconv5 = tf.layers.conv2d_transpose(deconv4, filters=32, kernel_size=5, strides=2, padding='same',
                                             name='deconv5', activation=tf.nn.leaky_relu,
                                             kernel_regularizer=regularizer)
        deconv5 = tf.layers.dropout(deconv5, 0.2, training=is_training)

        deconv6 = tf.layers.conv2d_transpose(deconv5, filters=3, kernel_size=5, strides=2, padding='same',
                                             name='deconv6', activation=tf.nn.sigmoid, kernel_regularizer=regularizer)

        return deconv6


def generator(z, regularizer, is_training=False):
    with tf.variable_scope('generator'):
        dense = tf.layers.dense(z, 4 * 4 * 256, kernel_regularizer=regularizer, name='dense')
        reshape = tf.reshape(dense, shape=[-1, 4, 4, 256])
        reshape = tf.layers.batch_normalization(reshape, training=is_training)
        reshape = tf.nn.relu(reshape)

        deconv1 = tf.layers.conv2d_transpose(reshape, filters=128, kernel_size=5, strides=2, padding='same',
                                             name='deconv1', kernel_regularizer=regularizer)
        deconv1 = tf.layers.batch_normalization(deconv1, training=is_training)
        deconv1 = tf.nn.relu(deconv1)

        deconv2 = tf.layers.conv2d_transpose(deconv1, filters=64, kernel_size=5, strides=2, padding='same',
                                             name='deconv2', kernel_regularizer=regularizer)
        deconv2 = tf.layers.batch_normalization(deconv2, training=is_training)
        deconv2 = tf.nn.relu(deconv2)

        deconv3 = tf.layers.conv2d_transpose(deconv2, filters=32, kernel_size=5, strides=2, padding='same',
                                             name='deconv3', kernel_regularizer=regularizer)
        deconv3 = tf.layers.batch_normalization(deconv3, training=is_training)
        deconv3 = tf.nn.relu(deconv3)

        deconv4 = tf.layers.conv2d_transpose(deconv3, filters=3, kernel_size=5, strides=2, padding='same',
                                             name='deconv4', activation=tf.nn.tanh, kernel_regularizer=regularizer)

        return deconv4


def discriminator(img, regularizer, is_training=False, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(inputs=img, filters=32, kernel_size=5, strides=2, padding='same', name='conv1',
                                 kernel_regularizer=regularizer)
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
        conv1 = tf.nn.leaky_relu(conv1)
        conv1 = tf.layers.dropout(conv1, 0.2, training=True)

        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=5, strides=2, padding='same', name='conv2',
                                 kernel_regularizer=regularizer)
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
        conv2 = tf.nn.leaky_relu(conv2)
        conv2 = tf.layers.dropout(conv2, 0.2, training=True)

        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=5, strides=2, padding='same', name='conv3',
                                 kernel_regularizer=regularizer)
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
        conv3 = tf.nn.leaky_relu(conv3)
        conv3 = tf.layers.dropout(conv3, 0.2, training=True)

        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=5, strides=2, padding='same', name='conv4',
                                 kernel_regularizer=regularizer)
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
        conv4 = tf.nn.leaky_relu(conv4)
        conv4 = tf.layers.dropout(conv4, 0.2, training=True)

        flatten = tf.layers.flatten(conv4, name='flatten')
        logits = tf.layers.dense(flatten, 1, kernel_regularizer=regularizer, name='logits')
        return logits


def generator_no_bn(z, regularizer):
    with tf.variable_scope('generator'):
        dense = tf.layers.dense(z, 4 * 4 * 256, kernel_regularizer=regularizer, activation=tf.nn.relu, name='dense')
        reshape = tf.reshape(dense, shape=[-1, 4, 4, 256])

        deconv1 = tf.layers.conv2d_transpose(reshape, filters=128, kernel_size=5, strides=2, padding='same',
                                             name='deconv1', kernel_regularizer=regularizer, activation=tf.nn.relu)

        deconv2 = tf.layers.conv2d_transpose(deconv1, filters=64, kernel_size=5, strides=2, padding='same',
                                             name='deconv2', kernel_regularizer=regularizer, activation=tf.nn.relu)

        deconv3 = tf.layers.conv2d_transpose(deconv2, filters=32, kernel_size=5, strides=2, padding='same',
                                             name='deconv3', kernel_regularizer=regularizer, activation=tf.nn.relu)

        deconv4 = tf.layers.conv2d_transpose(deconv3, filters=3, kernel_size=5, strides=2, padding='same',
                                             name='deconv4', activation=tf.nn.tanh, kernel_regularizer=regularizer)

        return deconv4


def discriminator_no_bn(img, regularizer, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(inputs=img, filters=32, kernel_size=5, strides=2, padding='same', name='conv1',
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        conv1 = tf.layers.dropout(conv1, 0.2, training=True)

        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=5, strides=2, padding='same', name='conv2',
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        conv2 = tf.layers.dropout(conv2, 0.2, training=True)

        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=5, strides=2, padding='same', name='conv3',
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        conv3 = tf.layers.dropout(conv3, 0.2, training=True)

        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=5, strides=2, padding='same', name='conv4',
                                 kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
        conv4 = tf.layers.dropout(conv4, 0.2, training=True)

        flatten = tf.layers.flatten(conv4, name='flatten')
        logits = tf.layers.dense(flatten, 1, kernel_regularizer=regularizer, name='logits')
        return logits