import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.models import build_classifier
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict


@DISTILLER.register_module()
class ClassificationDistiller(BaseClassifier):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None):

        super(ClassificationDistiller, self).__init__()
        
        self.teacher = build_classifier(teacher_cfg.model)
        if teacher_pretrained:
            self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_classifier(student_cfg.model)
        self.student.init_weights()
            
        self.distill_cfg = distill_cfg   
        self.distill_losses = nn.ModuleDict()
        if self.distill_cfg is not None:  
            for item_loc in distill_cfg:
                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)
    
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')

    def forward_train(self, 
                      img, 
                      gt_label, 
                      **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        if self.student.augments is not None:
            img, gt_label = self.student.augments(img, gt_label)

        fea_s = self.student.extract_feat(img, stage='backbone')
        x = fea_s
        if self.student.with_neck:
            x = self.student.neck(x)
        if self.student.with_head and hasattr(self.student.head, 'pre_logits'):
            x = self.student.head.pre_logits(x)
        
        logit_s = self.student.head.fc(x)
        loss = self.student.head.loss(logit_s, gt_label)

        student_loss = dict()
        for key in loss.keys():
            student_loss['ori_'+key] = loss[key]

        with torch.no_grad():
            fea_t = self.teacher.extract_feat(img, stage='backbone')

        loss_name = 'loss_mgd'
        student_loss[loss_name] = self.distill_losses[loss_name](fea_s[-1], fea_t[-1]) 

        return student_loss
    
    def simple_test(self, img, img_metas=None, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)

    def extract_feat(self, imgs, stage='neck'):
        """Extract features from images.
          'backbone', 'neck', 'pre_logits'
        """
        return self.student.extract_feat(imgs, stage)


