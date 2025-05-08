#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyyaml==5.1')
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html')


# In[4]:


import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
print(torch.__version__, torch.cuda.is_available())


# In[7]:


get_ipython().system('wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg')
im = cv2.imread("./input.jpg")
plt.imshow(im)


# In[8]:


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)


# In[9]:


print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)


# In[11]:


v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(out.get_image()[:, :, ::-1])


# In[ ]:


get_ipython().system('wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip')


# In[ ]:


get_ipython().system('unzip balloon_dataset.zip > /dev/null')

