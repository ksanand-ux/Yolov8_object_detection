# Computer Vision Engineer
#
# This project incorporates components from the Apache 2.0 licensed project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ******************************************************************************
# DISCLAIMER:
#
# This script is designed to download images and annotations from the Google Images
# Dataset V7. It is important to note that the images and annotations in the
# Google Images Dataset V7 are subject to specific licenses and usage terms. Users
# of this script are strongly advised to refer to the Google Open Images website
# (https://storage.googleapis.com/openimages/web/index.html) to verify and comply
# with the licensing terms associated with both the images and annotations that
# will be downloaded using this script.
#
# By using this script, you acknowledge and agree to adhere to the terms and
# conditions set forth by the creators of the Google Images Dataset V7 for the
# usage of both images and annotations. Any unauthorized use or violation of the
# licensing terms is the sole responsibility of the user.
# ******************************************************************************

import ast
import os
import shutil
import argparse
import sys

import requests

import pandas as pd


def process(classes, data_out_dir, yolov8_format, max_number_images_per_class):

    if max_number_images_per_class is None:
        max_number_images_per_class = sys.maxsize

    train_data_url = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
    val_data_url = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'
    test_data_url = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'

    downloader_url = 'https://raw.githubusercontent.com/openimages/dataset/master/downloader.py'

    class_names_all_url = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv'

    for url in [train_data_url, val_data_url, test_data_url, class_names_all_url, downloader_url]:
        if not os.path.exists(url.split('/')[-1]):
            print('downloading {}...'.format(url.split('/')[-1]))
            r = requests.get(url)
            with open(url.split('/')[-1], 'wb') as f:
                f.write(r.content)

    class_ids = []

    classes_all = pd.read_csv(class_names_all_url.split('/')[-1])

    for class_ in classes:
        if class_ not in list(classes_all['DisplayName']) or class_ not in list(classes_all['DisplayName']):
            raise Exception('Class name not found: {}'.format(class_))
        class_index = list(classes_all['DisplayName']).index(class_)
        class_ids.append(classes_all['LabelName'].iloc[class_index])

    image_list_file_path = os.path.join('.', 'image_list_file')
    if os.path.exists(image_list_file_path):
        os.remove(image_list_file_path)

    image_list_file_list = []
    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        image_list_file_per_class = [[] for j in class_ids]
        filename = url.split('/')[-1]
        with (open(filename, 'r') as f):
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
                if class_name in class_ids and id not in image_list_file_list \
                    and len(image_list_file_per_class[class_ids.index(class_name)]) < max_number_images_per_class:
                    image_list_file_list.append(id)
                    image_list_file_per_class[class_ids.index(class_name)].append(id)
                    with open(image_list_file_path, 'a') as fw:
                        fw.write('{}/{}\n'.format(['train', 'validation', 'test'][j], id))
                line = f.readline()

            f.close()

    out_dir = './.out'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.system('python downloader.py {} --download_folder={}'.format(image_list_file_path, out_dir))

    DATA_ALL_DIR = out_dir

    for set_ in ['train', 'val', 'test']:
        for dir_ in [os.path.join(data_out_dir, set_),
                     os.path.join(data_out_dir, set_, 'imgs'),
                     os.path.join(data_out_dir, set_, 'anns')]:
            if os.path.exists(dir_):
                shutil.rmtree(dir_)
            os.makedirs(dir_)

    for j, url in enumerate([train_data_url, val_data_url, test_data_url]):
        filename = url.split('/')[-1]
        set_ = ['train', 'val', 'test'][j]
        print(filename)
        with open(filename, 'r') as f:
            line = f.readline()
            while len(line) != 0:
                id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]
                if class_name in class_ids:
                    if os.path.exists(os.path.join(DATA_ALL_DIR, '{}.jpg'.format(id))):
                        if not os.path.exists(os.path.join(data_out_dir, set_, 'imgs', '{}.jpg'.format(id))):
                            shutil.copy(os.path.join(DATA_ALL_DIR, '{}.jpg'.format(id)),
                                        os.path.join(data_out_dir, set_, 'imgs', '{}.jpg'.format(id)))
                        with open(os.path.join(data_out_dir, set_, 'anns', '{}.txt'.format(id)), 'a') as f_ann:
                            # class_id, xc, yx, w, h
                            x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                            xc = (x1 + x2) / 2
                            yc = (y1 + y2) / 2
                            w = x2 - x1
                            h = y2 - y1

                            f_ann.write('{} {} {} {} {}\n'.format(int(class_ids.index(class_name)), xc, yc, w, h))
                            f_ann.close()

                line = f.readline()

    shutil.rmtree(out_dir, ignore_errors=True)

    if yolov8_format:
        for set_ in ['train', 'val', 'test']:
            for dir_ in [os.path.join(data_out_dir, 'images', set_),
                         os.path.join(data_out_dir, 'labels', set_)]:
                if os.path.exists(dir_):
                    shutil.rmtree(dir_)
                os.makedirs(dir_)

            for filename in os.listdir(os.path.join(data_out_dir, set_, 'imgs')):
                shutil.copy(os.path.join(data_out_dir, set_, 'imgs', filename), os.path.join(data_out_dir, 'images', set_, filename))
            for filename in os.listdir(os.path.join(data_out_dir, set_, 'anns')):
                shutil.copy(os.path.join(data_out_dir, set_, 'anns', filename), os.path.join(data_out_dir, 'labels', set_, filename))

            shutil.rmtree(os.path.join(data_out_dir, set_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', default=['Duck', 'Dog'])
    parser.add_argument('--out-dir', default='./data')
    parser.add_argument('--yolov8-format', default=True)
    parser.add_argument('--max-number-images-per-class', default=100)
    args = parser.parse_args()

    classes = args.classes
    if type(classes) is str:
        classes = ast.literal_eval(classes)

    out_dir = args.out_dir

    yolov8_format = True if args.yolov8_format in ['T', 'True', 1, '1'] else False

    max_number_images_per_class = int(args.max_number_images_per_class) \
        if args.max_number_images_per_class is not None else None

    process(classes, out_dir, yolov8_format, max_number_images_per_class)
