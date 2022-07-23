# Semantic Segmentation
## Install
  - Our codes are based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the installation of MMSegmentation and make sure you can run it successfully.
  - This repo uses mmcv-full==1.4.2 and mmseg==0.20.2
  - If you want to use lower mmcv-full version, you may have to change the optimizer in apis/train.py and build_distiller in tools/train.py.
  - For lower mmcv-full, you may refer [FGD](https://github.com/yzd-v/FGD) to change model.init_weights() in [train.py](https://github.com/yzd-v/MGD/tree/master/seg/tools/train.py) and self.student.init_weights() in [distiller.py](https://github.com/yzd-v/MGD/tree/master/seg/mmseg/distillation/distillers/segmentation_distiller.py).
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmsegmentation's codes.
  - Add the mmseg/distillation/. in our codes to the mmseg/ in mmsegmentation's codes.
  - Replace the mmseg/apis/train.py and tools/train.py in mmsegmentation's codes with mmseg/apis/train.py and tools/train.py in our codes.
  - Add pth_transfer.py to mmsegmentation's codes.
  - Unzip CityScapes dataset into data/cityscape/
## Train

```
#single GPU
python tools/train.py configs/distillers/mgd/psp_r101_distill_psp_r18_40k_512x512_city.py

#multi GPU
bash tools/dist_train.sh configs/distillers/mgd/psp_r101_distill_psp_r18_40k_512x512_city.py 8
```

## Transfer
```
# Tansfer the MGD model into mmseg model
python pth_transfer.py --mgd_path $mgd_ckpt --output_path $new_mmseg_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py $new_mmseg_ckpt --eval mIoU

#multi GPU
bash tools/dist_test.sh configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py $new_seg_ckpt 8 --eval mIoU
```
## Results
|  Model   | Backbone  | Baseline(mIoU) | +MGD(mIoU) |                            config                            |                          log                          | weight |
| :------: | :-------: | :----------------: | :------------: | :----------------------------------------------------------: | :------------------------------------------------------: | :--: |
|   PspNet   | ResNet-18 |        69.85        |      73.63      | [config](https://github.com/yzd-v/MGD/tree/master/seg/configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py) | [baidu](https://pan.baidu.com/s/15mLdMez1yf4-lrR0u5XUag?pwd=7vqd) | [baidu](https://pan.baidu.com/s/1a2DgN70r-jxl4bpC07NXQQ?pwd=u5ii) |
| DeepLabV3 | ResNet-18 |        73.20        |      76.31      | [config](https://github.com/yzd-v/MGD/tree/master/seg/configs/deeplabv3/deeplabv3_r18-d8_512x512_40k_cityscapes.py) | [baidu](https://pan.baidu.com/s/1xSXxQuIJ52ZihP0g3-0_pw?pwd=h9aw) | [baidu](https://pan.baidu.com/s/1Q8fOKhJWTHOSaEQVIIg4aw?pwd=1m9s) |

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

Our code is based on the project [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).