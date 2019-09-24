"""
An example of how to use this library.
Replace the filenames as need be.
"""
import logging
import random
import numpy as np

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
data = h5_to_np("DATA_FILE", "data")
label = h5_to_np("LABEL_FILE", "label")
test_data = h5_to_np("TEST_DATA", "data")

one_hot_output = label_to_one_hot(label, 10)

# normalise input
data = StandardScalar(data)

# make our model
model = Model(CrossEntropyWithSoftmax())

model.add_layer(DenseLayer(128, 256, activation=leaky_relu()))
model.add_layer(BatchNormLayer())
model.add_layer(Dropout(0.1))

model.add_layer(DenseLayer(256, 512, activation=leaky_relu()))
model.add_layer(BatchNormLayer())
model.add_layer(Dropout(0.1))

model.add_layer(DenseLayer(512, 10))

# fit to our data
model.SGD(
    data, one_hot_output, learning_rate=0.001, minibatch_size=32, epochs=4, adam=True
)

predictions = model.predict(test_data)
pred_labels = np.argmax(predictions, axis=1)

np_to_h5("PREDICTED_OUTPUT_LABELS", "label", pred_labels)
