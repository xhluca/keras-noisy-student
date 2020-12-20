code modified/retrieved from:

* https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
* https://github.com/qubvel/efficientnet

please see respective license for more details.

Here's how you can get the weights:
* [Kaggle](https://www.kaggle.com/xhlulu/efficientnetl2-tfkeras-weights)
* [Github release](https://github.com/xhlulu/keras-efficientnet-l2/releases/download/data/efficientnet-l2_noisy-student_notop.h5)

Here's an example (after you `pip install efficientnet`):
```python
import efficientnet.keras as efn 

model_url = "https://github.com/xhlulu/keras-efficientnet-l2/releases/download/data/efficientnet-l2_noisy-student_notop.h5"
model = efn.EfficientNetL2(weights=model_url)
```
