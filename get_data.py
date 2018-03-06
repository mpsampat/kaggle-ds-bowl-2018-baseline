#!/usr/bin/env python
import os
import sys

os.system('rm -rf stage1_train/')
os.system('!rm -rf kaggle-dsbowl-2018-dataset-fixes')
if os.path.exists('stage1_train')==False:
  os.system('git clone https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes')
  os.system('mv kaggle-dsbowl-2018-dataset-fixes/stage1_train/ .')
  os.system('!ls -d s*')
if os.path.exists('stage1_test')==False:
  os.system('wget https://raw.githubusercontent.com/AakashSudhakar/2018-data-science-bowl/master/compressed_files/stage1_test.zip -c')
  #!wget https://raw.githubusercontent.com/AakashSudhakar/2018-data-science-bowl/master/compressed_files/stage1_train.zip -c
  os.system('mkdir stage1_test')
  #!unzip stage1_train.zip -d stage1_train/
  os.system('!unzip stage1_test.zip -d stage1_test/')
