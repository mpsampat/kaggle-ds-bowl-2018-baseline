#!/usr/bin/env python
import model as modellib
import pandas as pd
import cv2
import os
import sys
import numpy as np
from tqdm import tqdm
from inference_config import inference_config
from bowl_dataset import BowlDataset
from utils import rle_encode, rle_decode, rle_to_string
import functions as f
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
ep = sys.argv[1]
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()[1]
all_models = model.find_last()
print(all_models)
model_path = '/home/mpsampat/work/dsb2018/kaggle-ds-bowl-2018-baseline/logs/bowl20180308T0721/mask_rcnn_bowl_00' + ep + '.h5'
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_test = BowlDataset()
dataset_test.load_bowl('stage1_test')
dataset_test.prepare()

output = []
sample_submission = pd.read_csv('stage1_sample_submission.csv')
ImageId = []
EncodedPixels = []
for image_id in tqdm(sample_submission.ImageId):
    image_path = os.path.join('stage1_test', image_id, 'images', image_id + '.png')
    
    original_image = cv2.imread(image_path)
    if original_image.ndim != 3:
    	original_image = skimage.color.gray2rgb(image)
        
    results = model.detect([original_image], verbose=0)
    r = results[0]
    
    masks = r['masks']
    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap2(masks, image_id, r['scores'])
    ImageId += ImageId_batch
    EncodedPixels += EncodedPixels_batch


f.write2csv('submission_v2_'+ ep + '_epochs.csv', ImageId, EncodedPixels)
