import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

def start_train(train_dir):
    mnist = input_data.read_data_sets(train_dir, one_hot=True)

    images = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    with tf.name_scope("hidden_layer_01"):
        weights = tf.Variable(tf.truncated_normal(shape=[784, 32], stddev= 1 / tf.sqrt(28.0)), name="weights")
        biases = tf.Variable(tf.zeros(shape=[32]), name="biases")

        logits = tf.nn.xw_plus_b(images, weights, biases)
        hidden_layer_01_output = tf.nn.relu(logits)

    with tf.name_scope("hidden_layer_02"):
        weights = tf.Variable(tf.truncated_normal(shape=[32, 128], stddev= 1 / tf.sqrt(28.0)), name="weights")
        biases = tf.Variable(tf.zeros(shape=[128]), name="biases")

        logits = tf.nn.xw_plus_b(hidden_layer_01_output, weights, biases)
        hidden_layer_02_output = tf.nn.relu(logits)

    with tf.name_scope("softmax_linear_layer"):
        weights = tf.Variable(tf.truncated_normal(shape=[128, 10], stddev= 1 / tf.sqrt(28.0)), name="weights")
        biases = tf.Variable(tf.zeros(shape=[10]), name="biases")

        logits = tf.nn.xw_plus_b(hidden_layer_02_output, weights, biases)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1)), dtype=tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for index in range(10000):
            batch_images, batch_labels = mnist.train.next_batch(100)
            feed_dict = {images: batch_images, labels: batch_labels}
            _, loss_result, accuracy_result = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

            if (index % 100 == 0):
                print ("current step:{}, loss_result:{}, accuracy_result:{}".format(index, loss_result, accuracy_result))


        _, loss_result, accuracy_result = sess.run([train_op, loss, accuracy], feed_dict={images: mnist.test.images, labels: mnist.test.labels})
        print ("final test, loss_result:{}, accuracy_result{}".format(loss_result, accuracy_result))


if __name__ == "__main__":
    train_dir = "MNIST_data"
    start_train(train_dir)