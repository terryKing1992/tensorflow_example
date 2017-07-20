import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def start_train(train_dir):
    mnist = input_data.read_data_sets(train_dir)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[32, 128, 10],
                                                n_classes=10,
                                                model_dir=None)

    def get_train_inputs():
        batch_images = mnist.train.images
        batch_labels = mnist.train.labels
        x = tf.constant(batch_images)
        y = tf.cast(tf.constant(batch_labels), dtype=tf.int32)

        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=1000)

    def get_test_inputs():
        x = tf.constant(mnist.test.images)
        y = tf.cast(tf.constant(mnist.test.labels), dtype=tf.int32)

        return x, y

    accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


if __name__ == "__main__":
    start_train("MNIST_data")