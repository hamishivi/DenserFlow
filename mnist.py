"""
An example of how to use this library.
Replace the filenames as need be.
"""
import logging
import random
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from DenserFlow.model import Model
from DenserFlow.layer import Dropout, DenseLayer, BatchNormLayer
from DenserFlow.error import CrossEntropyWithSoftmax
from DenserFlow.activation import leaky_relu
from DenserFlow.util import h5_to_np, np_to_h5, label_to_one_hot, StandardScalar

logger = logging.getLogger("prediction")

# make sure we can reproduce our results
np.random.seed(100)
random.seed(100)

# set logging level
logging.basicConfig(level=logging.INFO)
# load data
mnist = fetch_openml('mnist_784')
data = mnist.data
label = mnist.target.astype(int)
# train/test split - use default settings for now
data_train, data_test, label_train, label_test = train_test_split(data, label, random_state=100)

one_hot_output_train = label_to_one_hot(label_train, 10)

print('---')
print(one_hot_output_train.shape)
print(data_train.shape)
print('---')

# make our model
model = Model(CrossEntropyWithSoftmax())

model.add_layer(DenseLayer(784, 512, activation=leaky_relu()))
model.add_layer(BatchNormLayer())
model.add_layer(Dropout(0.1))

model.add_layer(DenseLayer(512, 256, activation=leaky_relu()))
model.add_layer(BatchNormLayer())
model.add_layer(Dropout(0.1))

model.add_layer(DenseLayer(256, 512, activation=leaky_relu()))
model.add_layer(BatchNormLayer())
model.add_layer(Dropout(0.1))

model.add_layer(DenseLayer(512, 10))

# fit to our data
model.SGD(
    data_train, one_hot_output_train, learning_rate=0.001, minibatch_size=32, epochs=4, adam=True
)

# make our predictions with our fitted model
predictions = model.predict(data_test)
pred_labels = np.argmax(predictions, axis=1)

# do a full evaluation
print(classification_report(label_test, pred_labels))
