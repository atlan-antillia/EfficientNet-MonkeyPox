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


import os
import sys

import numpy as np
import tensorflow as tf


# Based on the following colab
# https://colab.research.google.com/github/google/automl/blob/master/efficientnetv2/tfhub.ipynb#scrollTo=lus9bIA-bQgj
class CustomDataset:

  def __init__(self):
    pass


  def create(self, FLAGS):
    data_dir    = FLAGS.data_dir
    image_size  = FLAGS.image_size
    eval_image_size = FLAGS.eval_image_size
    target_size = (image_size, image_size)
    eval_size   = (eval_image_size, eval_image_size)
    #eval_size   = (image_size, image_size)

    batch_size  = FLAGS.batch_size
    """    
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,  
      validation_split = 0.2) 
    
    train_generator = image_generator.flow_from_directory(data_dir, 
                                                          target_size = target_size, 
                                                          subset      = "training" )
    valid_generator = image_generator.flow_from_directory(data_dir, 
                                                          target_size = eval_size, 
                                                          subset      = "validation" )
    """

    data_augmentation = FLAGS.data_augmentation # 
    if data_augmentation:
       
       print("---- Do data_augumentation")
       train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = 0.2,
    
          rotation_range     = 8,
          horizontal_flip    = True,
       
          width_shift_range  = 0.9, 
          height_shift_range = 0.9,
          shear_range        = 0.1, 
          #zoom_range         = [0.8, 1.2],
       )
       train_generator = train_datagen.flow_from_directory(
             data_dir, 
             target_size   = target_size, 
             batch_size    = batch_size,
             interpolation = "bilinear",
             subset        = "training", 
             shuffle       = True)

       valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = 0.2,
    
          rotation_range     = 8,
          horizontal_flip    = True,
       
          width_shift_range  = 0.9, 
          height_shift_range = 0.9,
          shear_range        = 0.1, 
          #zoom_range         = [0.8, 1.2],
       )
       valid_generator = valid_datagen.flow_from_directory(
             data_dir, 
             #target_size   = target_size,
             target_size   = eval_size,
             batch_size    = 1, #batch_size, 
             interpolation = "bilinear",
             subset        = "validation", 
             shuffle       = False)

    else:
       print("---- No data_augumentation ")
       train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = 0.2,
    
       )
       train_generator = train_datagen.flow_from_directory(
             data_dir, 
             target_size   = target_size, 
             batch_size    = batch_size,
             interpolation = "bilinear",
             subset        = "training", 
             shuffle       = True)

       valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rescale            = 1/255,  
          validation_split   = 0.2,
       )
       valid_generator = valid_datagen.flow_from_directory(
             data_dir, 
             #target_size   = target_size, 
             target_size   = eval_size,
             batch_size    = 1, #batch_size, 
             interpolation = "bilinear",
             subset        = "validation", 
             shuffle       = False)

    return (train_generator, valid_generator)

