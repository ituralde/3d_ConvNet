import argparse
import os
import numpy as np
import cv2
import collections
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
import videotransforms
from dataset import IVBSSDataset
from model import TemporalActionLocalization
import time


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_video_path', type=str, help='path to face video')
    parser.add_argument('--cabin_video_path', type=str, help='path to cabin video')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--pretrained_I3D_model', type=str, help='path to the pretrained I3D model')
    parser.add_argument('--clip_length', default=64, type=int, help='Number of frames in each clip')
    parser.add_argument('--clip_stride', default=16, type=int, help='Number of frames between the starts of two clips')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    args = parser.parse_args()
    return args


def clip_generation(video_path, clip_length, clip_stride):
    frames = os.listdir(video_path)
    frames.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    L = len(frames)
    indices = np.arange(start=0, stop=len(frames) - clip_length, step=clip_stride)
    indices_in_clips = [list(range(_idx, _idx + clip_length)) for _idx in indices]
    clips = []
    for indices_in_clip in indices_in_clips:
        clip = [frames[i] for i in indices_in_clip]
        clips.append(clip)
    return clips


def load_rgb_frames(video_path, clip):
    imgs = []
    for frame in clip:
        frame_path = os.path.join(video_path, frame)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, c = img.shape
        sc = float(256)/min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs


def main():
    args = get_parse()
    face_video_path = args.face_video_path
    cabin_video_path = args.cabin_video_path
    checkpoint = args.checkpoint
    pretrained_I3D_model = args.pretrained_I3D_model
    clip_length = args.clip_length
    clip_stride = args.clip_stride
    num_classes = args.num_classes

    face_frame_list = os.listdir(face_video_path)
    cabin_frame_list = os.listdir(cabin_video_path)
    if len(face_frame_list) < len(cabin_frame_list):
        video_path = face_video_path
    else:
        video_path = cabin_video_path

    clips = clip_generation(video_path, clip_length, clip_stride)
    model = TemporalActionLocalization(num_classes, pretrained_I3D_model)
    ckp = torch.load(checkpoint)
    model.load_state_dict(ckp['model'])
    model.eval()
    
    clip_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                          videotransforms.ToTensor()
                                          ])
    all_predict_classes = []
    all_start_scores = []
    all_end_scores = []
    all_rst_scores = []
    all_rend_scores = []
    
    total_time = 0.0
    for clip in clips:
        start_time = time.time()
        face_video_frames = load_rgb_frames(face_video_path, clip)
        cabin_video_frames = load_rgb_frames(cabin_video_path, clip)
        face_video_frames = clip_transforms(face_video_frames)
        cabin_video_frames = clip_transforms(cabin_video_frames)
        face_video_frames = face_video_frames.unsqueeze(0)
        cabin_video_frames = cabin_video_frames.unsqueeze(0)
        class_scores, start_scores, end_scores, rst_scores, rend_scores = model(face_video_frames, cabin_video_frames)
        pred_class = torch.argmax(class_scores, dim=1)
        end_time = time.time()
        all_predict_classes.append(pred_class.item())
        all_start_scores.append(start_scores.item())
        all_end_scores.append(end_scores.item())
        all_rst_scores.append(rst_scores.item())
        all_rend_scores.append(rend_scores.item())
        total_time += end_time-start_time
    # chunk aggregation
#     all_merged_clips = collections.defaultdict(list)
#     merged_clips = []
#     L = len(all_predict_classes)
#     for i in range(L):
#         predict_class = all_predict_classes[i]
#         if predict_class != 0:
#             merged_clips.append(i)
#             if i == (L-1):
#                 all_merged_clips[predict_class].append(merged_clips)
                
#             elif all_predict_classes[i] != all_predict_classes[i+1]:
#                 all_merged_clips[predict_class].append(merged_clips)
#                 merged_clips = []               
#     print(all_merged_clips)
    avg_time = total_time/len(clips)
    print(avg_time)
                
if __name__ == '__main__':
    main()
