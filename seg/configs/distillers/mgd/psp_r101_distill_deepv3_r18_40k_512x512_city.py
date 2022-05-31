_base_ = [
    '../../deeplabv3/deeplabv3_r18-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.75
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
    init_student = False,
    use_logit = True,
    distill_cfg = [ dict(methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fea',
                                       student_channels = 512,
                                       teacher_channels = 2048,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd,
                                       )
                                ]
                        ),
                   ]
    )

student_cfg = 'configs/deeplabv3/deeplabv3_r18-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py'
