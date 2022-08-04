# Copyright 2022 Antillia.com. All Rights Reserved.
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
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
from sklearn.metrics import classification_report

#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append("../../")
from TestDataset import TestDataset
from FineTuningModel import FineTuningModel

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
  flags.DEFINE_string('evaluation_dir',  "./evaluation", 'Directoroy to save test results.')
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
    #self.model.compile()
    self.model.load_weights(best_model, by_name=True)
    print("--- loaded weights {}".format(best_model))
    self.evaluation_dir = FLAGS.evaluation_dir
    if not os.path.exists(self.evaluation_dir):
      os.makedirs(self.evaluation_dir)


  def run(self ):
    print("--- EfficientNetV2Evaluator.run() ")
    test_dataset = TestDataset()
    test_gen     = test_dataset.create(FLAGS)
    print("--- call model.evaluate ")
    y_pred = self.model.predict(test_gen, verbose=1) 
    print("{}".format(y_pred))
    predictions = np.array(list(map(lambda x: np.argmax(x), y_pred)))
    print("--- predictions:\n{}".format(predictions))

    y_true=test_gen.classes
    print("--- y_true:\n{}".format(y_true))
    self.compute_classification_score(y_true, predictions)
    
    self.create_classification_report(y_true, predictions, self.classes)
    
    self.create_confusion_matrix(y_true, predictions, self.classes, self.evaluation_dir, fig_size=(8, 6))


  def compute_classification_score(self, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    accuracy = round(accuracy, 4)
    print("--- accuracy {}".format(accuracy))
    precision = precision_score(y_true, y_pred)
    precision = round(precision, 4)
    print("--- precision {}".format(precision))

    recall    = recall_score(y_true, y_pred)
    recall    = round(recall, 4)
    print("--- recall {}".format(recall))

    f1        = f1_score(y_true, y_pred)
    f1        = round(f1, 4)
    print("--- f1 {}".format(f1))
    evaluation_csv_file = os.path.join(self.evaluation_dir, "evaluation.csv")
    NL = "\n"
    SEP = ","
    with open(evaluation_csv_file, "w") as f:
      f.write("accuray, precison, recall, f1" + NL)
      line = str(accuracy) + SEP + str(precision) + SEP + str(recall) + SEP + str(f1) 
      f.write(line + NL)


  def create_classification_report(self, y_true, predictions, classes): 
    cls_report = classification_report( 
                               y_true       = y_true,
                               y_pred       = predictions,
                               target_names = classes, 
                               output_dict  = True)
    print("--- classification report:\n")
    pprint.pprint(cls_report)

    report_df = pd.DataFrame(cls_report).T
    cls_report_csv_file = os.path.join(self.evaluation_dir, "classification_report.csv")

    report_df.to_csv(cls_report_csv_file)


  def create_confusion_matrix(self, y_true, predictions, classes, save_dir, fig_size=(8, 6)):
    labels = sorted(list(set(y_true)))
    cmatrix = confusion_matrix(y_true, predictions, labels= labels) 
    print("--- confusion matrix:\n{}".format(cmatrix))
    plt.figure(figsize=fig_size) 
    ax = sns.heatmap(cmatrix, annot = True, xticklabels=classes, yticklabels=classes, cmap = 'Blues')
    ax.set_title('Confusion Matrix',fontsize = 14, weight = 'bold' ,pad=20)

    ax.set_xlabel('Predicted',fontsize = 12, weight='bold')
    #ax.set_xticklabels(ax.get_xticklabels(), rotation =0)
    ax.set_ylabel('Actual',fontsize = 12, weight='bold') 
    #ax.set_yticklabels(ax.get_yticklabels(), rotation =0)

    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    confusion_matrix_file = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_file)    
    #plt.show()

def main(_):
  
  tester = EfficientNetV2Evaluator()
  tester.run()


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  define_flags()
  app.run(main)
