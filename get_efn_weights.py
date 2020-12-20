#!/usr/bin/env bash
# =============================================================================
# Copyright 2019 Pavel Yakubovskiy, Sasha Illarionov. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import sys

import numpy as np

import tensorflow.compat.v1 as tf
import efficientnet.tfkeras
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from efficientnet_tf import eval_ckpt_main
from tqdm.auto import tqdm

# def load_weights(model, weights):
#     """Load weights to Conv2D, BatchNorm, Dense layers of model sequentially"""
#     layer_index = 0
#     groupped_weights = group_weights(weights)
#     for layer in model.layers:
#         if isinstance(layer, (Conv2D, BatchNormalization, Dense)):
#             print(layer)
#             layer.set_weights(groupped_weights[layer_index])
#             layer_index += 1


model_name = "efficientnet-b0"
model_ckpt = "./tf_weights/noisy_student_efficientnet-b0"
output_file = ""
example_img = "misc/panda.jpg"
weights_only = True


image_files = [example_img]
eval_ckpt_driver = eval_ckpt_main.EvalCkptDriver(model_name)
with tf.Graph().as_default(), tf.Session() as sess:
    images, _ = eval_ckpt_driver.build_dataset(
        image_files, [0] * len(image_files), False
    )
    eval_ckpt_driver.build_model(images, is_training=False)
    sess.run(tf.global_variables_initializer())
    eval_ckpt_driver.restore_model(sess, model_ckpt)
    global_variables = tf.global_variables()
    weights = dict()
    print("Starting!")
    for variable in tqdm(global_variables):
        try:
            weights[variable.name] = variable.eval()
        except:
            print(f"Skipping variable {variable.name}, an exception occurred")
# model = _get_model_by_name(
#     model_name, include_top=True, input_shape=None, weights=None, classes=1000
# )
# load_weights(model, weights)
# output_file = f"{output_file}.h5"
# if weights_only:
#     model.save_weights(output_file)
# else:
#     model.save(output_file)
