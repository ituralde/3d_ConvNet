import pandas as pd
import os
import numpy as np
import argparse
import glob
import re
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Parameters to generate clips')
    parser.add_argument('--face_video_dir', help='Face View Video directory')
    parser.add_argument('--cabin_video_dir', help='Cabin View Video directory')
    parser.add_argument('--face_control_video_dir', help='Face View Control Video directory')
    parser.add_argument('--cabin_control_video_dir', help='Cabin View Control Video directory')
    
    parser.add_argument('--label_file_path', help='Path to the label file')
    parser.add_argument('--clip_length', default=64, type=int, help='Number of frames in each clip')
    parser.add_argument('--clip_stride', default=16, type=int, help='Number of frames between the starts of two clips')
    parser.add_argument('--save_path', help='Path to save the preprocessed clips')
    args = parser.parse_args()
    return args


def extract_clips(args, frame_dir, df):
    items = os.path.basename(frame_dir).split('_')
    driver_id = int(items[1].lstrip('0'))
    trip_id = int(items[2].lstrip('0'))
    time = int(items[3]) / 100

    frame_list = glob.glob(os.path.join(frame_dir, '*.jpg'))
    frame_list = [os.path.basename(frame) for frame in frame_list]
    frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    total_duration = len(frame_list) / 10

    df1 = df.loc[lambda df: df['driver'] == driver_id]
    df2 = df1.loc[lambda df1: df1['trip'] == trip_id]
    if df2.size > 1:
        df2 = df2.loc[lambda df2: df2['starttime'] >= time]
        df2 = df2.loc[lambda df2: df2['endtime'] <= time + total_duration + 10]
    start_ids = (df2.values[:, 2] - time) * 10
    end_ids = (df2.values[:, 3] - time) * 10
    start_labels = df2.values[:, 4]
    end_labels = df2.values[:, 5]
    indices = np.arange(start=0, stop=len(frame_list) - args.clip_length, step=args.clip_stride)
    indices_in_clips = [list(range(_idx, _idx + args.clip_length)) for _idx in indices]

    clips_with_start_or_end = []
    clips_without_start_or_end = []
    
    # TODO: extract info for each clip
    i = 0
    j = 0
    length = args.clip_length
    while i < len(indices_in_clips):
        clip_info = {}
        indices_in_clip = indices_in_clips[i]
        first_frame_id = indices_in_clip[0]
        last_frame_id = indices_in_clip[-1]
        start_id = start_ids[j]
        end_id = end_ids[j]
        flag = 0
        if start_labels[j] == 7:
            if last_frame_id <= end_id:
                clip_info['class'] = 5
                clip_info['start'] = 0
                clip_info['rst'] = -1
                clip_info['end'] = 0
                clip_info['rend'] = 1
                i += 1
                flag = 1
            else:
                if first_frame_id < end_id:
                    if (end_id - first_frame_id) >= 0.55 * length:
                        clip_info['class'] = 5
                        clip_info['start'] = 0
                        clip_info['rst'] = -1
                        clip_info['end'] = 1
                        clip_info['rend'] = round((end_id - (first_frame_id + length / 2)) / float(length / 2), 3)
                        i += 1
                        flag = 1
                    else:
                        i += 1
                else:
                    if j < start_ids.shape[0] - 1:
                        j += 1
                    else:
                        i += 1

        elif end_labels[j] == 8:
            if last_frame_id <= start_id:
                i += 1
            elif first_frame_id <= start_id:
                if (last_frame_id - start_id) >= 0.55 * length:
                    clip_info['class'] = 5
                    clip_info['start'] = 1
                    clip_info['rst'] = round((start_id - (first_frame_id + length / 2)) / float(length / 2), 3)
                    clip_info['end'] = 0
                    clip_info['rend'] = 1
                    i += 1
                    flag = 1
                else:
                    i += 1
            else:
                clip_info['class'] = 5
                clip_info['start'] = 0
                clip_info['rst'] = -1
                clip_info['end'] = 0
                clip_info['rend'] = 1
                i += 1
                flag = 1

        else:
            if last_frame_id <= start_id:
                i += 1
            elif first_frame_id < start_id:
                if (last_frame_id - start_id) >= 0.55 * length:
                    clip_info['class'] = start_labels[j]
                    clip_info['start'] = 1
                    clip_info['rst'] = round((start_id - (first_frame_id + length / 2)) / float(length / 2), 3)
                    clip_info['end'] = 0
                    clip_info['rend'] = 1
                    i += 1
                    flag = 1
                else:
                    i += 1
            elif first_frame_id >= start_id:
                if last_frame_id <= end_id:
                    clip_info['class'] = start_labels[j]
                    clip_info['start'] = 0
                    clip_info['rst'] = -1
                    clip_info['end'] = 0
                    clip_info['rend'] = 1
                    i += 1
                    flag = 1
                elif first_frame_id < end_id:
                    if (end_id - first_frame_id) >= 0.55 * length:
                        clip_info['class'] = start_labels[j]
                        clip_info['start'] = 0
                        clip_info['rst'] = -1
                        clip_info['end'] = 1
                        clip_info['rend'] = round((end_id - (first_frame_id + length / 2)) / float(length / 2), 3)
                        i += 1
                        flag = 1
                    else:
                        i += 1
                elif first_frame_id >= end_id:
                    if j < start_ids.shape[0] - 1:
                        j += 1
                    else:
                        i += 1
        if flag == 1:
            _, driver_id1, trip_id1, time1 = items
            clip_info['video_id'] = '_' + driver_id1 + '_' + trip_id1 + '_' + time1
            clip_info['frames'] = [frame_list[k] for k in indices_in_clip]
            if clip_info['start'] == 0 and clip_info['end'] == 0:
                clips_without_start_or_end.append(clip_info)
            else:
                clips_with_start_or_end.append(clip_info)
    clips = {
        'clips_without_start_or_end': clips_without_start_or_end,
        'clips_with_start_or_end': clips_with_start_or_end
    }   
    return clips


def extract_control_clip(args, frame_dir):
    items = os.path.basename(frame_dir).split('_')
    frame_list = glob.glob(os.path.join(frame_dir, '*.jpg'))
    frame_list = [os.path.basename(frame) for frame in frame_list]
    frame_list.sort(key=lambda f: int(re.sub('\D', '', f)))
    indices = np.arange(start=0, stop=len(frame_list) - args.clip_length, step=args.clip_stride)
    indices_in_clips = [list(range(_idx, _idx + args.clip_length)) for _idx in indices]
    clips = []
    for indices_in_clip in indices_in_clips:
        clip_info = {
            'class': 0,
            'start': 0,
            'rst': 0,
            'end': 0,
            'rend': 0,
        }
        _, driver_id1, trip_id1, time1 = items
        clip_info['video_id'] = '_' + driver_id1 + '_' + trip_id1 + '_' + time1
        clip_info['frames'] = [frame_list[k] for k in indices_in_clip]
        clips.append(clip_info)
    return clips


def main():
    args = parse_args()
    face_video_dir = args.face_video_dir
    cabin_video_dir = args.cabin_video_dir
    face_control_video_dir = args.face_control_video_dir
    cabin_control_video_dir = args.cabin_control_video_dir
    
    label_file_path = args.label_file_path
    save_path = args.save_path
    df = pd.read_excel(label_file_path)
    starttime = df.values[:, 2]
    starttime = starttime / 100
    endtime = df.values[:, 3]
    endtime = endtime / 100
    df.loc[:, 'starttime'] = starttime
    df.loc[:, 'endtime'] = endtime

    face_video_list = os.listdir(face_video_dir)
    face_control_video_list = os.listdir(face_control_video_dir)
    
    driver_ids = []
    for video_name in face_video_list:
        items = video_name.split('_')
        driver_id = items[1]
        if driver_id not in driver_ids:
            driver_ids.append(driver_id) 
    L = len(driver_ids)
    indices = np.random.permutation(L)
    num_train_driver_ids = int(L*0.7)
    num_val_driver_ids = int(L*0.2)
    num_test_driver_ids = L - num_train_driver_ids - num_val_driver_ids 
    train_driver_ids = [driver_ids[i] for i in indices[:num_train_driver_ids]]
    val_driver_ids = [driver_ids[i] for i in indices[num_train_driver_ids:(num_train_driver_ids+num_val_driver_ids)]]
    test_driver_ids = [driver_ids[i] for i in indices[(num_train_driver_ids+num_val_driver_ids):]]
    train_video_list = []
    val_video_list = []
    test_video_list = []
    for video_name in face_video_list:
        if video_name.split('_')[1] in train_driver_ids:
            train_video_list.append(video_name)
        elif video_name.split('_')[1] in val_driver_ids:
            val_video_list.append(video_name)
        elif video_name.split('_')[1] in test_driver_ids:
            test_video_list.append(video_name)
    
    train_control_video_list = []
    val_control_video_list = []
    test_control_video_list = []
    for video_name in face_control_video_list:
        if video_name.split('_')[1] in train_driver_ids:
            train_control_video_list.append(video_name)
        elif video_name.split('_')[1] in val_driver_ids:
            val_control_video_list.append(video_name)
        elif video_name.split('_')[1] in test_driver_ids:
            test_control_video_list.append(video_name)
        
    train_video_clips = []
    for i in range(len(train_video_list)):
        face_video = train_video_list[i]
        ids = face_video.split('_', 1)[1]
        cabin_video = 'Cabin_' + ids
        frames1 = os.listdir(os.path.join(face_video_dir, face_video))
        frames2 = os.listdir(os.path.join(cabin_video_dir, cabin_video))
        if len(frames1) <= len(frames2):
            frame_dir = os.path.join(face_video_dir, face_video)
        else:
            frame_dir = os.path.join(cabin_video_dir, cabin_video)
        clips = extract_clips(args, frame_dir, df)
        train_video_clips.append(clips)
 
    train_control_video_clips = []
    for i in range(len(train_control_video_list)):
        face_video = train_control_video_list[i]
        ids = face_video.split('_', 1)[1]
        cabin_video = 'Cabin_' + ids
        frames1 = os.listdir(os.path.join(face_control_video_dir, face_video))
        frames2 = os.listdir(os.path.join(cabin_control_video_dir, cabin_video))
        if len(frames1) <= len(frames2):
            frame_dir = os.path.join(face_control_video_dir, face_video)
        else:
            frame_dir = os.path.join(cabin_control_video_dir, cabin_video)
        clips = extract_control_clip(args, frame_dir)
        train_control_video_clips.append(clips)
    
    val_video_clips = []
    for i in range(len(val_video_list)):
        face_video = val_video_list[i]
        ids = face_video.split('_', 1)[1]
        cabin_video = 'Cabin_' + ids
        frames1 = os.listdir(os.path.join(face_video_dir, face_video))
        frames2 = os.listdir(os.path.join(cabin_video_dir, cabin_video))
        if len(frames1) <= len(frames2):
            frame_dir = os.path.join(face_video_dir, face_video)
        else:
            frame_dir = os.path.join(cabin_video_dir, cabin_video)
        clips = extract_clips(args, frame_dir, df)
        val_video_clips.append(clips)
    
    val_control_video_clips = []
    for i in range(len(val_control_video_list)):
        face_video = val_control_video_list[i]
        ids = face_video.split('_', 1)[1]
        cabin_video = 'Cabin_' + ids
        frames1 = os.listdir(os.path.join(face_control_video_dir, face_video))
        frames2 = os.listdir(os.path.join(cabin_control_video_dir, cabin_video))
        if len(frames1) <= len(frames2):
            frame_dir = os.path.join(face_control_video_dir, face_video)
        else:
            frame_dir = os.path.join(cabin_control_video_dir, cabin_video)
        clips = extract_control_clip(args, frame_dir)
        val_control_video_clips.append(clips)
        
        
    test_video_clips = []
    for i in range(len(test_video_list)):
        face_video = test_video_list[i]
        ids = face_video.split('_', 1)[1]
        cabin_video = 'Cabin_' + ids
        frames1 = os.listdir(os.path.join(face_video_dir, face_video))
        frames2 = os.listdir(os.path.join(cabin_video_dir, cabin_video))
        if len(frames1) <= len(frames2):
            frame_dir = os.path.join(face_video_dir, face_video)
        else:
            frame_dir = os.path.join(cabin_video_dir, cabin_video)
        clips = extract_clips(args, frame_dir, df)
        test_video_clips.append(clips)
    
    test_control_video_clips = []
    for i in range(len(test_control_video_list)):
        face_video = test_control_video_list[i]
        ids = face_video.split('_', 1)[1]
        cabin_video = 'Cabin_' + ids
        frames1 = os.listdir(os.path.join(face_control_video_dir, face_video))
        frames2 = os.listdir(os.path.join(cabin_control_video_dir, cabin_video))
        if len(frames1) <= len(frames2):
            frame_dir = os.path.join(face_control_video_dir, face_video)
        else:
            frame_dir = os.path.join(cabin_control_video_dir, cabin_video)
        clips = extract_control_clip(args, frame_dir)
        test_control_video_clips.append(clips)
    
    video_clips = {
        'train_video_clips':train_video_clips,
        'train_control_video_clips': train_control_video_clips,
        'val_video_clips':val_video_clips,
        'val_control_video_clips': val_control_video_clips,
        'test_video_clips':test_video_clips,
        'test_control_video_clips': test_control_video_clips,
        'train_video_list':train_video_list,
        'val_video_list':val_video_list,
        'test_video_list':test_video_list,
        'train_control_video_list':train_control_video_list,
        'val_control_video_list':val_control_video_list,
        'test_control_video_list':test_control_video_list
    }
    
   
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'video_clips.json'), 'w') as f:
        f.write(json.dumps(video_clips, indent=4))


if __name__ == "__main__":
    main()

