from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model import myModel
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

model = myModel()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_function(labels, images)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def test_step(images, labels):
    predictions = model(images)
    loss = loss_function(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)



# epochs = 5
#
# for epoch in range(epochs):
#     train_loss.reset_state()
#     train_accuracy.reset_state()
#     test_loss.reset_state()
#     test_accuracy.reset_state()
#
#     for images, labels in train_ds:
#         train_step(images, labels)
#
#     for test_images, test_labels in test_ds:
#         test_step(images, labels)
#
#     template = 'Epoch:{}, Loss:{}, Accuracy:{}, Test Loss:{}, Test Accuracy:{}'
#     print(template.format(epoch + 1,
#                           train_loss.result(),
#                           train_accuracy.result() * 100,
#                           test_loss.result(),
#                           test_accuracy.result() * 100))
#
#
#
#
#
#
# #
# # if __name__ == '__main__':
# #     main()
