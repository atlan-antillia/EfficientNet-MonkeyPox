# Copyright 2021 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2022/07/29 Copyright (C) antillia.com

# EfficientNetV2Evaluator.py

from operator import ge
import os
import sys
sys.path.append("../../")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import time
import numpy as np
import glob

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import preprocessing

sys.path.append("../../")
from TestDataset import TestDataset
from FineTuningModel import FineTuningModel
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

FLAGS = flags.FLAGS


def define_flags():
  """Define all flags for binary run."""
  flags.DEFINE_string('mode', 'eval', 'Running mode.')
  flags.DEFINE_string('image_path', None, 'Location of test image.')
  flags.DEFINE_string('label_map',  './label_map.txt', 'Label map txt file')
  flags.DEFINE_integer('image_size', None, 'Image size.')
  flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
  flags.DEFINE_string('model_name', 'efficientnetv2-s', 'Model name to use.')
  flags.DEFINE_string('dataset_cfg', 'Imagenet', 'dataset config name.')
  flags.DEFINE_string('hparam_str', '', 'k=v,x=y pairs or yaml file.')
  flags.DEFINE_bool('debug', False, 'If true, run in eager for debug.')
  flags.DEFINE_string('export_dir', None, 'Export or saved model directory')
  flags.DEFINE_string('trace_file', '/tmp/a.trace', 'If set, dump trace file.')

  # 2022/07/20
  flags.DEFINE_integer('eval_image_size', None, 'Image size.')
  flags.DEFINE_string('data_dir', './Testing', 'Testing data directory.')
  
  flags.DEFINE_string('strategy', 'gpu', 'Strategy: tpu, gpus, gpu.')
  flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
  flags.DEFINE_string('best_model_name', 'best_model.5h', 'Best model name.')
  flags.DEFINE_bool('mixed_precision', True, 'If True, use mixed precision.')
  flags.DEFINE_bool('fine_tuning', True,  'Fine tuning flag')
  flags.DEFINE_float('trainable_layers_ratio',  0.3, 'Trainable layers ratio')
  flags.DEFINE_string('test_dir',  "./test", 'Directoroy to save test results.')
  flags.DEFINE_bool('channels_first', False, 'Channel first flag.')
  flags.DEFINE_string('ckpt_dir', "", 'Pretrained checkpoint dir.')


class EfficientNetV2Evaluator:
  # Constructor
  def __init__(self):

    self.classes = []

    with open(FLAGS.label_map, "r") as f:
       lines = f.readlines()
       for line in lines:
        if len(line) >0:
          self.classes.append(line.strip())
    print("--- classes {}".format(self.classes))


    tf.keras.backend.clear_session()

    tf.config.run_functions_eagerly(FLAGS.debug)
  
    tf.keras.backend.clear_session()

    model_name  = FLAGS.model_name
    image_size  = FLAGS.image_size
    num_classes = FLAGS.num_classes
    fine_tuning = FLAGS.fine_tuning
    trainable_layers_ratio = FLAGS.trainable_layers_ratio
    
    
    finetuning_model = FineTuningModel(model_name, None, FLAGS.debug)

    self.model = finetuning_model.build(image_size, 
                                        num_classes, 
                                        fine_tuning, 
                                        trainable_layers_ratio = trainable_layers_ratio)
    if FLAGS.debug:
      self.model.summary()
 
    best_model = FLAGS.model_dir + "/best_model.h5"

    if not os.path.exists(best_model):
      raise Exception("Not found best_model " + best_model)
    self.model.compile()
    self.model.load_weights(best_model) #, by_name=False)
    print("--- loaded weights {}".format(best_model))
  

  def run(self ):
    print("--- EfficientNetV2Evaluator.run() ")
    test_dir   = FLAGS.test_dir
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)
    test_results_file = os.path.join(test_dir, "test.csv")
    test_dataset = TestDataset()
    test_gen     = test_dataset.create(FLAGS)
    print("--- call model.evaluate ")
    y_pred = self.model.predict(test_gen, verbose=1) 
    print("{}".format(pred))
    predictions = np.array(list(map(lambda x: np.argmax(x), y_pred)))
    print("--- prediction {}".format(prediction))
def main(_):
  
  tester = EfficientNetV2Evaluator()
  tester.run()


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  define_flags()
  app.run(main)
