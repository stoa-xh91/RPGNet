# RPGNet: Relation based Pose Graph Network for Crowded Scenes Pose Estimation

# Introduction
This is the implementation of RPGNet: Relation based Pose Graph Network for Crowded Scenes Pose Estimation. In this work, we focus on two main problems: 1) how to remove the joints-of-interference from a given person proposal; and 2) how to infer the ambiguous joints. To tackle these problems, we propose a new pipeline named Relation based Pose Graph Network (RPGNet). Unlike existing works that directly predict joints-of-target by labeling joints-ofinterference as false positive, we encourage all joints to be predicted and model their relation through a multi-joints relation parser (MRP) for joints-of-interference removal. This new pipeline will largely relieve the confusion of the joints estimation model when seeing identical joints with totally distinct labels (e.g., the identical hand exists in two bounding boxes). Furthermore, human beings can well estimate the joints with ambiguity by looking at the surrounding regions. For example, human can easily infer the location of ‘neck’ after seeing ‘head’ and ‘shoulder’. Inspired by this, we propose a joints refinement machine (JRM) with commonsense knowledge to refine pose estimation results for handling ambiguous joints. 
![](https://github.com/stoa-xh91/RPGNet/blob/master/visualization/RPGNet.jpg)
# Main Results on CrowdPose test set
![](https://github.com/stoa-xh91/RPGNet/blob/master/visualization/main_results.png)
# Environment
The code is developed based on the [HRNet project](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA RTX GPU cards. Other platforms or GPU cards are not fully tested.
# Installation

- Install requirments
```
  pip install -r requirements.txt
```
- Make libs
```
  cd ./lib
  make
``` 
- COCO API
```
  # COCOAPI=/path/to/clone/cocoapi
  git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
  cd $COCOAPI/PythonAPI
  # Install into global site-packages
  make install
  # Alternatively, if you do not have permissions or prefer
  # not to install the COCO API into global site-packages
  python3 setup.py install --user 
```
- CrowdPose API
```  
  Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
  Reverse the bug stated in https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
```
# Data Preparation
**COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under ./data.

**CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training and validation.
Download and extract them under ./data.

# Training and Testing
- Download pretrained models ([Baidu](https://pan.baidu.com/s/1OyuimZ4Xd6rtC3iD4SbyZQ).)
- Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/rl_train.py \
--cfg experiments/crowdpose/hrnet/rpgnet_w32_256x192_adam_lr1e-3.yaml \
```
- Testing on CrowdPose dataset using provided models([Baidu](https://pan.baidu.com/s/1OyuimZ4Xd6rtC3iD4SbyZQ).)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/rl_test.py \
--cfg experiments/crowdpose/hrnet/rpgnet_w32_256x192_adam_lr1e-3.yaml \
TEST.MODEL_FILE rpgnet_w32.pth
```
