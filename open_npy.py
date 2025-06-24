import numpy as np
import tensorflow as tf

# raw_dataset = tf.data.TFRecordDataset("/mnt/new_usb/jupyter-altis5526/hc3_extra/data/MICCAI_long_tail_test.tfrecords")

raw_dataset = tf.data.TFRecordDataset("/mnt/new_usb/jupyter-altis5526/hc3_extra/data/MICCAI_Resample_test_seed0.tfrecords")

chosen = []

for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy()) 
    gender = example.features.feature['gender'].bytes_list.value[0].decode('utf-8')
    label = example.features.feature["Tortuous Aorta"].float_list.value[0]
    subject = example.features.feature["subject_id"].int64_list.value[0]
    # if gender == 'F' and label == 1:
    chosen.append(subject)
    # "Tortuous Aorta"

print(len(chosen))
