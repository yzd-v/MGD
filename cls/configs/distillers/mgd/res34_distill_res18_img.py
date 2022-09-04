_base_ = [
    '../../resnet/resnet18_b32x8_imagenet.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth',
    distill_cfg = [ 
                    dict(methods=[dict(type='MGDLoss',
                                       name='loss_mgd',
                                       student_channels = 512,
                                       teacher_channels = 512,
                                       alpha_mgd=0.00007,
                                       lambda_mgd=0.15,
                                       )
                                ]
                        ),
                   ]
    )

student_cfg = 'configs/resnet/resnet18_b32x8_imagenet.py'
teacher_cfg = 'configs/resnet/resnet34_b32x8_imagenet.py'
