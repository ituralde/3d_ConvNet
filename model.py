import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d, Unit3D
from weight_init import weight_init


class TemporalActionLocalization(nn.Module):
    def __init__(self, num_classes, I3D_trained_model):
        super(TemporalActionLocalization, self).__init__()
        self.num_classes = num_classes
        
        self.I3D = InceptionI3d(157, in_channels=3)
        
        I3D_checkpoint = torch.load(I3D_trained_model)
        self.I3D.load_state_dict(I3D_checkpoint)
        for param in self.I3D.parameters():
            param.requires_grad = False
        
        self.predictor = nn.Sequential(
            Unit3D(in_channels=2*(384 + 384 + 128 + 128), output_channels=256,
                   kernel_shape=[1, 1, 1],
                   name='layer1'),
            Unit3D(in_channels=256, output_channels=self.num_classes + 4,
                   kernel_shape=[1, 1, 1],
                   activation_fn=None,
                   use_batch_norm=False,
                   use_bias=True,
                   name='layer2'),

        )
        self.sigmoid = nn.Sigmoid()        
        self.predictor.apply(weight_init)

    def forward(self, face_clips, cabin_clips, labels=None, weight=0.25):
        feat1 = self.I3D.extract_features(face_clips)
        feat2 = self.I3D.extract_features(cabin_clips)
        feat = torch.cat((feat1, feat2), 1)
        preds = self.predictor(feat)
        preds = preds.squeeze(3).squeeze(3)
        preds = torch.mean(preds, dim=2)
        # shape(B,num_classes+4)
        class_scores = preds[:, :self.num_classes]
        start_scores = preds[:, self.num_classes].sigmoid()
        end_scores = preds[:, self.num_classes+1].sigmoid()
        rst_scores = preds[:, self.num_classes+2]
        rend_scores = preds[:, -1]
        if labels is not None:
            class_labels = labels['event_id']
            start_labels = labels['start']
            end_labels = labels['end']
            rst_labels = labels['rst']
            rend_labels = labels['rend']
            class_loss = F.cross_entropy(class_scores, class_labels)
            chunk_inclusion_loss = 1/2*(F.binary_cross_entropy(start_scores, start_labels, reduction='none')
                                        + F.binary_cross_entropy(end_scores, end_labels, reduction='none'))
            loss1 = F.smooth_l1_loss(rst_scores, rst_labels, reduction='none')
            loss2 = F.smooth_l1_loss(rend_scores, rend_labels, reduction='none')
            regression_loss = start_labels*loss1 + end_labels*loss2
            indicator = (class_labels > 0).to(class_loss.dtype)
            
            total_loss = class_loss + weight*torch.mean(indicator*(chunk_inclusion_loss+regression_loss))
            return total_loss, class_loss, class_scores, start_scores, end_scores, rst_scores, rend_scores
        else:
            return class_scores, start_scores, end_scores, rst_scores, rend_scores