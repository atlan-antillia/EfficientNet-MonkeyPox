# Copyright 2022 antillia.com All Rights Reserved.
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

# 2022/08/03 Copyright (C) antillia.com

# ConfusionMatrix.py

import os
import sys

from sklearn.metrics import confusion_matrix

import seaborn as sns
#import pandas as pd 
import matplotlib.pyplot as plt

class ConfusionMatrix:
  def __init__(self):
    pass

  def create(self, y_true, predictions, classes, save_dir, fig_size=(8, 6)):
  

    labels = sorted(list(set(y_true)))
    print("---- y_true {}".format(y_true))
    cmatrix = confusion_matrix(y_true, predictions, labels= labels) 
    print("---confusion matrix {}".format(cmatrix))
    plt.figure(figsize=fig_size) 
    ax = sns.heatmap(cmatrix, annot = True, xticklabels=classes, yticklabels=classes ) #, cmap = 'Blues')
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
