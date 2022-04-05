import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import videotransforms
from dataset import IVBSSDataset, collate_fn
from model import TemporalActionLocalization
import time


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_video_dir', type=str, help='face video directory')
    parser.add_argument('--cabin_video_dir', type=str, help='cabin video directory')
    parser.add_argument('--test_data_path', type=str, help='path to the test data')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--ckp_path', type=str, help='path to the loaded checkpoint')
    parser.add_argument('--pretrained_I3D_model', type=str, help='path to the pretrained I3D model')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    args = parser.parse_args()
    return args


def load_ckp(ckp_path, model):
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model'])
    return model


def test():
    args = get_parse()
    face_video_dir = args.face_video_dir
    cabin_video_dir = args.cabin_video_dir
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    pretrained_I3D_model = args.pretrained_I3D_model
    num_classes = args.num_classes
        
    print('Start to load data')
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToTensor()
                                         ])
    test_dataset = IVBSSDataset(face_video_dir,
                                cabin_video_dir,
                                test_data_path,
                                test_transforms
                               )
    print('Total number of test samples is {0}'.format(len(test_dataset)))
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 sampler=SequentialSampler(test_dataset),
                                 collate_fn=collate_fn
                                )
    model = TemporalActionLocalization(num_classes, pretrained_I3D_model) 
    print('Load checkpoint')
    model = load_ckp(args.ckp_path, model)

    model.cuda()
    model.eval()

    print('Start to test')
    test_loss = 0.0
    class_accuracy = 0.0
    start_accuracy = 0.0
    end_accuracy = 0.0
    mse_rst = 0.0
    mse_rend = 0.0
    test_steps = 0
    eps = 1e-10
    
    start_time = time.time()
    for i, (face_imgs, cabin_imgs, labels) in enumerate(test_dataloader):
        face_imgs = face_imgs.cuda()
        cabin_imgs = cabin_imgs.cuda()
        for k, v in labels.items():
            labels[k] = v.cuda()
        class_labels = labels['event_id']
        start_labels = labels['start']
        end_labels = labels['end']
        rst_labels = labels['rst']
        rend_labels = labels['rend']
        
        loss, class_loss, class_scores, start_scores, end_scores, rst_scores, rend_scores = model(face_imgs, cabin_imgs, labels)
        test_loss += loss.item()
        
        class_pred = torch.argmax(class_scores, dim=1)
        class_accuracy += torch.sum((class_pred==class_labels).float())/class_labels.shape[0]
        
        mask1 = (class_labels > 0).float()
        start_pred = (start_scores > 0.5).float()
        start_accuracy += torch.sum(mask1 * (start_pred==start_labels).float())/(torch.sum(mask1)+eps)
        end_pred = (end_scores > 0.5).float()
        end_accuracy += torch.sum(mask1 * (end_pred==end_labels).float())/(torch.sum(mask1)+eps)
        
        mask2 = (start_labels == 1).float()
        mask3 = mask1 * mask2
        mse_rst += torch.sum(mask3 * (rst_scores - rst_labels)**2)/(torch.sum(mask1)+eps)
        mask4= (end_labels == 1).float()
        mask5 = mask1 * mask4
        mse_rend += torch.sum(mask5 * (rend_scores - rend_labels)**2)/(torch.sum(mask1)+eps)  
        
        test_steps += 1
    
    avg_test_loss = test_loss/test_steps
    avg_class_accuracy = class_accuracy/test_steps
    avg_start_accuracy = start_accuracy/test_steps
    avg_end_accuracy = end_accuracy/test_steps
    avg_mse_rst = mse_rst/test_steps
    avg_mse_rend = mse_rend/test_steps
    end_time = time.time()
    total_time = end_time-start_time
    avg_time = total_time/(test_steps*batch_size)
    
    print('avg_test_loss:{}, avg_class_accuracy:{}, avg_start_accuracy:{}, avg_end_accuracy:{}, avg_mse_rst:{}, avg_mse_rend:{}, avg_time:{}'.format(avg_test_loss, avg_class_accuracy, avg_start_accuracy, avg_end_accuracy, avg_mse_rst, avg_mse_rend, avg_time))   


if __name__ == '__main__':
    test()

