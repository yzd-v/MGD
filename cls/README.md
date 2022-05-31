# Image Classification
## Install
  - Our codes are based on [MMClassification](https://github.com/open-mmlab/mmclassification). Please follow the installation of MMClassification and make sure you can run it successfully.
  - This repo uses mmcv-full==1.3.17 and mmcls = 0.19.0
  - If you want to use lower mmcv-full version, you may have to change the optimizer in apis/train.py and build_distiller in tools/train.py.
  - For lower mmcv-full, you may refer [FGD](https://github.com/yzd-v/FGD) to change model.init_weights() in [train.py](https://github.com/yzd-v/MGD/tree/master/cls/tools/train.py) and self.student.init_weights() in [distiller.py](https://github.com/yzd-v/MGD/tree/master/cls/mmcls/distillation/distillers/classification_distiller.py).
## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmclassification's codes.
  - Add the mmcls/. in our codes to the mmcls/ in mmclassification's codes.
  - Replace the mmcls/apis/train.py and tools/train.py in mmclassification's codes with mmcls/apis/train.py and tools/train.py in our codes.
  - Add pth_transfer.py to mmclassification's codes.
  - Unzip ImageNet dataset into data/imagenet/

## Train

```
#single GPU
python tools/train.py configs/distillers/res34_distill_res18_img.py

#multi GPU
bash tools/dist_train.sh configs/distillers/res34_distill_res18_img.py 8
```

## Transfer
```
# Tansfer the MGD model into mmcls model
python pth_transfer.py --mgd_path $mgd_ckpt --output_path $new_mmcls_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/resnet/resnet18_8xb32_in1k.py $new_mmcls_ckpt --metrics accuracy

#multi GPU
bash tools/dist_test.sh configs/resnet/resnet18_8xb32_in1k.py $new_mmcls_ckpt 8 --metrics accuracy
```

## Results
|  Model   | Teacher  | Baseline(Top-1 Acc) | +MGD(Top-1 Acc) |                            config                            |                          log                          | weight |
| :------: | :-------: | :----------------: | :------------: | :----------------------------------------------------------: | :------------------------------------------------------: | :--: |
|   ResNet18   | ResNet34 |        69.90        |      71.69      | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | [baidu](https://pan.baidu.com/s/1PoJeqOzlEb6MKBEQMSvYsw?pwd=27wc) | [baidu](https://pan.baidu.com/s/1VtjqCvFHGh-qUR7wOvojYw?pwd=ehnn) ||                                                          |      |
| MobileNet | ResNet50 |        69.21        |      72.49      | [config](https://github.com/yzd-v/MGD/tree/master/cls/configs/mobilenet_v1/mobilenet_v1.py) | [baidu](https://pan.baidu.com/s/1m5yuPATnpnfBB1izZc0I3g?pwd=piu8) | [baidu](https://pan.baidu.com/s/1NdoHf0KA3MiIUKC9_gH3ng?pwd=fnii) |

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

Our code is based on the project [MMClassification](https://github.com/open-mmlab/mmclassification).