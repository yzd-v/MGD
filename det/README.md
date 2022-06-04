# Object Detection
## Install
  - Our codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please follow the installation of MMDetection and make sure you can run it successfully.
  - This repo uses mmcv-full==1.3.17 and mmdet==2.19.0
  - If you want to use lower mmcv-full version, you may have to change the optimizer in apis/train.py and build_distiller in tools/train.py.
  - For lower mmcv-full, you may refer [FGD](https://github.com/yzd-v/FGD) to change model.init_weights() in [train.py](https://github.com/yzd-v/MGD/tree/master/det/tools/train.py) and self.student.init_weights() in [distiller.py](https://github.com/yzd-v/MGD/tree/master/det/mmdet/distillation/distillers/detection_distiller.py).
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmdetectin's codes.
  - Add the mmdet/distillation/. in our codes to the mmdet/ in mmdetectin's codes.
  - Replace the mmdet/apis/train.py and tools/train.py in mmdetection's codes with mmdet/apis/train.py and tools/train.py in our codes.
  - Add pth_transfer.py to mmdetection's codes.
  - Unzip COCO dataset into data/coco/
## Train

```
#single GPU
python tools/train.py configs/distillers/mgd/retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py

#multi GPU
bash tools/dist_train.sh configs/distillers/mgd/retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py 8
```

## Transfer
```
# Tansfer the MGD model into mmdet model
python pth_transfer.py --mgd_path $mgd_ckpt --output_path $new_mmdet_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/retinanet/retinanet_r50_fpn_2x_coco.py $new_mmdet_ckpt --eval bbox

#multi GPU
bash tools/dist_test.sh configs/retinanet/retinanet_r50_fpn_2x_coco.py $new_mmdet_ckpt 8 --eval bbox
```
## Results
|    Model    |  Backbone  | Baseline(mAP) | +MGD(mAP) |                            config                            |                          log                          | weight |
| :---------: | :--------: | :-----------: | :-------: | :----------------------------------------------------------: | :------------------------------------------------------: | :--: |
|  RetinaNet  | ResNet-50  |     37.4      |   41.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) | [baidu](https://pan.baidu.com/s/1sBxgi110KtZLSB8NDr7G-g?pwd=n83s) | [baidu](https://pan.baidu.com/s/1Bqv2XNa_TAGZJFUd177WWA?pwd=gu2x) |
| Faster RCNN | ResNet-50  |     38.4      |   42.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py) | [baidu](https://pan.baidu.com/s/1xrLcE2e9e5qT1nomX4TqTg?pwd=aheq) | [baidu](https://pan.baidu.com/s/1vuZuq06wg3X9SJPNWQSgrw?pwd=2x8z) |
|  RepPoints  | ResNet-50  |     38.6      |   42.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_2x_coco.py) | [baidu](https://pan.baidu.com/s/103unzbTgqjIBdYzH8zliEg?pwd=aucd) | [baidu](https://pan.baidu.com/s/1HfqvLoMU57y9NXPq5TNhow?pwd=g79p) |

|  Model   | Backbone  | Baseline(Mask mAP) | +MGD(Mask mAP) |                            config                            |                          log                          | weight |
| :------: | :-------: | :----------------: | :------------: | :----------------------------------------------------------: | :------------------------------------------------------: | :--: |
|   SOLO   | ResNet-50 |        33.1        |      36.2      | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/solo/solo_r50_fpn_1x_coco.py) | [baidu](https://pan.baidu.com/s/1kl7bSSkToN7atGZdWp9Ntw?pwd=wdpt) | [baidu](https://pan.baidu.com/s/1xZmIj_wP7SXsSxfXxa_4Ww?pwd=ksr1) |
| MaskRCNN | ResNet-50 |        35.4        |      38.1      | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py) | [baidu](https://pan.baidu.com/s/1uN8Q5Ew57oKUjzh65_TCVw?pwd=nykm) | [baidu](https://pan.baidu.com/s/1B4Bcw6S_sy882SMK2bp7uw?pwd=a7xf) |

## Citation
```
@article{yang2022masked,
  title={Masked Generative Distillation},
  author={Yang, Zhendong and Li, Zhe and Shao, Mingqi and Shi, Dachuan and Yuan, Zehuan and Yuan, Chun},
  journal={arXiv preprint arXiv:2205.01529},
  year={2022}
}
```

## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).