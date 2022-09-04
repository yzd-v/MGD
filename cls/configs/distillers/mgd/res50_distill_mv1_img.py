_base_ = [
    '../../mobilenet_v1/mobilenet_v1.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='ClassificationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
    distill_cfg = [ 
                    dict(methods=[dict(type='MGDLoss',
                                       name='loss_mgd',
                                       student_channels = 1024,
                                       teacher_channels = 2048,
                                       alpha_mgd=0.00007,
                                       lambda_mgd=0.15,
                                       )
                                ]
                        ),
                   ]
    )

student_cfg = 'configs/mobilenet_v1/mobilenet_v1.py'
teacher_cfg = 'configs/resnet/resnet50_b32x8_imagenet.py'
optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=5.0))