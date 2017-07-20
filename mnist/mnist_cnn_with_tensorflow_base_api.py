import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def start_train(train_dir):
    mnist = input_data.read_data_sets(train_dir, one_hot=True)

    images_input = tf.placeholder(dtype=tf.float32, shape=[None, 784])

    images = tf.reshape(images_input, shape=[-1, 28, 28, 1])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    with tf.name_scope("conv_01_layer"):
        weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], dtype=tf.float32), name="weights")
        biases = tf.Variable(tf.zeros(shape=[32]))

        conv2d = tf.nn.conv2d(images, filter=weights, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")
        h_conv1 = tf.nn.relu(conv2d + biases)
        max_pool = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.name_scope("conv_02_layer"):
        weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 128], dtype=tf.float32, stddev= 1 / tf.sqrt(28.0)), name="weights")
        biases = tf.Variable(tf.zeros(shape=[128]))

        conv2d = tf.nn.conv2d(max_pool, filter=weights, strides=[1, 1, 1, 1], padding="SAME", name="conv2d")
        h_conv2 = tf.nn.relu(conv2d + biases)

        max_pool = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.name_scope("full_connect_layer_01"):
        weights = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 128, 1024], stddev=1 / tf.sqrt(28.0)), name="weights")
        biases = tf.Variable(tf.zeros(shape=[1024]))

        logits = tf.nn.xw_plus_b(tf.reshape(max_pool, [-1, 7 * 7 * 128]), weights, biases)
        full_layer_output = tf.nn.relu(logits)

    with tf.name_scope("full_connect_layer_02"):
        weights = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=1 / tf.sqrt(28.0)), name="weights")
        biases = tf.Variable(tf.zeros(shape=[10]))

        logits = tf.nn.xw_plus_b(full_layer_output, weights, biases)

    cross_enctopy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_enctopy)

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)

    init_op = tf.global_variables_initializer()

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(init_op)

        for index in range(1000):
            batch_images, batch_labels = mnist.train.next_batch(100)
            _, loss_result, accuracy_result = sess.run([train_op, loss, accuracy], feed_dict={images_input: batch_images, labels: batch_labels})

            if index % 100 == 0:
                print("current step:{}, loss_result:{}, accuracy_result:{}".format(index, loss_result, accuracy_result))

        _, loss_result, accuracy_result = sess.run([train_op, loss, accuracy],
                                                   feed_dict={images_input: mnist.test.images, labels: mnist.test.labels})
        print("final test, loss_result:{}, accuracy_result:{}".format(loss_result, accuracy_result))



if __name__ == "__main__":
    start_train("MNIST_data")