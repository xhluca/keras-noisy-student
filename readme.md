# Acknowledgment
Code modified/retrieved from:

* https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
* https://github.com/qubvel/efficientnet

Please see respective license for more details.

## Download

Here's how you can get the weights:
* [Kaggle](https://www.kaggle.com/xhlulu/efn-l2)
* [Github release](https://github.com/xhlulu/keras-efficientnet-l2/releases/tag/data)


## Instructions
First, make sure to have the library and download the weights:
```
pip install efficientnet
wget https://github.com/xhlulu/keras-efficientnet-l2/releases/download/data/efficientnet-l2_noisy-student.h5
wget https://github.com/xhlulu/keras-efficientnet-l2/releases/download/data/efficientnet-l2_noisy-student_notop.h5
```

For `tensorflow>=2.4.0`:
```python
import efficientnet.keras as efn 

model = efn.EfficientNetL2(weights="./efficientnet-l2_noisy-student_notop.h5", include_top=False)
# or
model = efn.EfficientNetL2(weights="./efficientnet-l2_noisy-student.h5", include_top=True)
```

For `tensorflow<=2.3.1`, there's a bug that would cause the L2 model to not load correctly. To use it, apply the following hack:
```
model = efn.EfficientNetL2(
  weights="./efficientnet-l2_noisy-student_notop.h5", 
  include_top=False,
  drop_connect_rate=0  # the hack
)
```

However, this will modify the behavior of the model so you will need to be careful when using this.
