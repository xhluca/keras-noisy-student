Note: this is not currently working due to a bug in the keras efficientnet implementation

code modified/retrieved from:

* https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
* https://github.com/qubvel/efficientnet

please see respective license for more details.

Here's how you can get the weights:
* [Kaggle](https://www.kaggle.com/xhlulu/efficientnetl2-tfkeras-weights)
* [Github release](https://github.com/xhlulu/keras-efficientnet-l2/releases/tag/data)

First, make sure to have the library and download the weights:
```
pip install efficientnet
wget https://github.com/xhlulu/keras-efficientnet-l2/releases/download/data/efficientnet-l2_noisy-student_notop.h5
```

Then run this inside python:
```python
import efficientnet.keras as efn 

model_path = "./efficientnet-l2_noisy-student_notop.h5"
model = efn.EfficientNetL2(weights=model_path)
```
