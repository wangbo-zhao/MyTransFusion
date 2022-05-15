from calendar import c
from distutils.util import convert_path
import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np

waymo_tfrecords_dir = "data/waymo/waymo_format/validation"
convert_save_path = "data/waymo/waymo_format/convert_validation"
os.makedirs(convert_save_path, exist_ok=True)
prefix = "1"
files_list = os.listdir(waymo_tfrecords_dir)
T_ref_to_front_cam = np.array([[0.0, 0.0, 1.0, 0.0],
                               [-1.0, 0.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])


for file_idx, _ in enumerate(files_list):
    file_pathname = os.path.join(waymo_tfrecords_dir, _)
    print(file_pathname)

    file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')

    for frame_num, frame_data in enumerate(file_data):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(frame_data.numpy()))

        filename = f'{prefix}{file_idx:03d}{frame_num:03d}'
        print(filename)

        for camera in frame.context.camera_calibrations:
                # FRONT = 1, see dataset.proto for details
            if camera.name == 1:
                T_front_cam_to_vehicle = np.array(
                        camera.extrinsic.transform).reshape(4, 4)

        T_k2w = T_front_cam_to_vehicle @ T_ref_to_front_cam
        T_k2w = list(T_k2w.reshape(16))
        T_k2w = [f'{i:e}' for i in T_k2w]

        context_name = frame.context.name
        frame_timestamp_micros = frame.timestamp_micros

        save_context = "T_k2w: " + " ".join(T_k2w) + "\n"
        save_context += "context_name: " + context_name + "\n"
        save_context += "frame_timestamp_micros: " + str(frame_timestamp_micros)

        save_path = os.path.join(convert_save_path, filename + '.txt')

        with open(save_path,'w+') as fp_calib:
            fp_calib.write(save_context)
            fp_calib.close()
