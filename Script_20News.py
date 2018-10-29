#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:43:34 2018

@author: BenjaminSalem
"""

from Datas_20News import *
from Model_Training_20News import *

data_class = Input_Data('data/20news-bydate 2/matlab/')
x_train,y_train, vocab= data_class.training_input('train.data','train.label',1, dataset_voc)
x_test, x_val, y_test, y_val = data_class.training_input('test.data','test.label',0, dataset_voc)


model = Net(300,20)
model_test = training_model(0.001, 3000, model)

model_test.main(x_train, x_val, x_test, y_train, y_val, y_test)

