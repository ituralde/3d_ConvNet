import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def load_rgb_frames(video_dir, video_id, frame_names):
    frame_dir = os.path.join(video_dir,video_id)
    imgs = []
    for frame_name in frame_names:
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, c = img.shape
        sc = float(256)/min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        imgs.append(img)
    imgs = np.asarray(imgs)
    return imgs


class IVBSSDataset(Dataset):
    def __init__(self, face_video_dir, cabin_video_dir, clip_info_path, transforms=None):
        """
        Args:
        face_video_dir(string): Directory with all the face video frames.
        cabin_video_dir(string): Directory with all the cabin video frames.
        clip_info_path(string): Path to the json file containing information for clips.
        transform(callable, optional): Optional transform to be applied on a sample.
        """
        self.face_video_dir = face_video_dir
        self.cabin_video_dir = cabin_video_dir
        with open(clip_info_path,'r') as f:
            self.all_clip_info = json.load(f)
        self.transforms = transforms

    def __len__(self):
        return len(self.all_clip_info)

    def __getitem__(self, idx):
        clip_info = self.all_clip_info[idx]
        event_id = clip_info['class']
        if event_id == 3:
            event_id = 2
        elif event_id == 5:
            event_id = 3
#         elif event_id == 7:
#             event_id = 4
        start = clip_info['start']
        rst = clip_info['rst']
        end = clip_info['end']
        rend = clip_info['rend']
        video_id = clip_info['video_id']
        frame_names = clip_info['frames']
        face_video_id = 'Face' + video_id
        cabin_video_id = 'Cabin' + video_id
        face_imgs = load_rgb_frames(self.face_video_dir, face_video_id, frame_names)
        cabin_imgs = load_rgb_frames(self.cabin_video_dir, cabin_video_id, frame_names)

        if self.transforms is not None:
            face_imgs = self.transforms(face_imgs)
            cabin_imgs = self.transforms(cabin_imgs)
        return face_imgs, cabin_imgs, event_id, start, rst, end, rend
#         return face_imgs, cabin_imgs, torch.LongTensor(event_id), torch.LongTensor(start), torch.FloatTensor(rst), torch.LongTensor(end), torch.FloatTensor(rend)


def collate_fn(batch):
    face_imgs, cabin_imgs, event_id, start, rst, end, rend = zip(*batch)
    face_imgs = torch.stack(face_imgs)
    cabin_imgs = torch.stack(cabin_imgs)
    event_id = torch.tensor(event_id, dtype=torch.long)
    start = torch.tensor(start, dtype=torch.float)
    rst = torch.tensor(rst, dtype=torch.float)
    end = torch.tensor(end, dtype=torch.float)
    rend = torch.tensor(rend, dtype=torch.float)
#     event_id = torch.stack(event_id)
#     start = torch.stack(start)
#     rst = torch.stack(rst)
#     end = torch.stack(end)
#     rend = torch.stack(rend)
    labels = {}
    labels['event_id'] = event_id
    labels['start'] = start
    labels['rst'] = rst
    labels['end'] = end
    labels['rend'] = rend
    return face_imgs, cabin_imgs, labels
    














