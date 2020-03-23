"""
Copyright 2020 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import code
import sys
sys.path.append("..")
import cv2
import numpy as np
from glob import glob
import json
from tqdm import tqdm

import keras
from keras.applications import imagenet_utils
from keras.layers import Dense,Dropout,Input,concatenate
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint

import params
from utilities.misc_utils import get_paths
from utilities.misc_utils import load_tiff
from utilities.misc_utils import get_batch_inds
from utilities.misc_utils import wrangle_data

import keras.backend as K
import tensorflow as tf

from classification_models.keras import Classifiers

from lib.CLR.clr_callback import CyclicLR

architecture = "resnet50"
ResNet, preprocess_input = Classifiers.get(architecture)

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    HueSaturationValue,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    JpegCompression
)

def load_batch(chip_coords, all_density, all_gsd, image, args):
    images = np.zeros((len(chip_coords), args.model_image_size[0], args.model_image_size[1], 3))
    densities = np.zeros((len(chip_coords),1))
    gsds = np.zeros((len(chip_coords),2))
    
    for i in range(len(chip_coords)):
        x1,y1,x2,y2 = chip_coords[i]
        density = all_density[i]
        gsd = all_gsd[i]
        sub_image = image[y1:y2,x1:x2,:3]
        sub_image = cv2.resize(sub_image, args.model_image_size)
        
        aug = Compose([ VerticalFlip(p=0.5),
                    RandomRotate90(p=0.5),
                    HorizontalFlip(p=0.5),
                    Transpose(p=0.5),
                    CLAHE(p=0.2),
                    RandomBrightness(limit=0.2, p=0.2),
                    RandomGamma(p=0.2),
                    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=10,
                        val_shift_limit=10, p=0.2),
                    RandomContrast(limit=0.4, p=0.3)])

        sub_image = aug(image=sub_image)['image']
        
        images[i,:,:,:] = sub_image

        densities[i,0] = density
        gsds[i,0] = gsd[0]
        gsds[i,1] = gsd[1]

    images = imagenet_utils.preprocess_input(images) / 255.0
    
    
    return images,densities,gsds


def image_generator(train_data, args):

    while True:
        prev_image_path = ""
        
        sample_num = np.int(np.ceil(args.sample_size / args.batch_size))
        
        batch_data = []
        for key,val in train_data.items():
            idx = np.random.permutation(len(val))
            for ind in idx[:sample_num]:
                batch_data.append(val[ind])
        
        for data in batch_data:
            image_path,chip_coords,density,gsd = data
            if prev_image_path != image_path:
                basename = os.path.basename(image_path).replace(".tif", "")
                image,tr = load_tiff(image_path)
                if args.add_osm:
                    osm_path = os.path.join(params.osm_dir, basename+".png")
                    osm_mask = cv2.imread(osm_path, 0)
                    osm_mask = cv2.resize(osm_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    image[osm_mask==0] = 0
                prev_image_path = image_path
            images,densities,gsds = load_batch(chip_coords, density, gsd, image, args)
            if args.add_metadata:
                yield ([images, gsds],densities)
            else:
                yield (images,densities)

def get_train_data(image_paths, args):
    batch_data = []

    min_density,max_density = np.inf,-np.inf

    for image_path in image_paths:
        basename = os.path.basename(image_path).replace(".tif", "")
        chip_data_file = os.path.join(args.chip_pop_dir, basename+".json")
        data = json.load(open(chip_data_file, "r"))
        idx = np.random.permutation(len(data["chip_coords"]))
        batch_inds = get_batch_inds(idx, args.batch_size)
        for inds in batch_inds:
            batch_chip_coords = []
            batch_density = []
            batch_gsd= []
#            batch_area = []
            for ind in inds:
                batch_chip_coords.append(data["chip_coords"][ind])
                density = data["all_gt"][ind][0] / data["all_gt"][ind][1]
                
                batch_density.append(density)
                batch_gsd.append(np.abs(data["all_gt"][ind][2]).tolist())
                
            batch_data.append([image_path,batch_chip_coords, batch_density, batch_gsd])


    return batch_data

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    squared_loss = 0.5 * K.square(error)
    linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)
    huber_loss = tf.where(cond, squared_loss, linear_loss)
    return K.mean(huber_loss)

def train(args):

    train_paths, _ = get_paths()
    
    if args.wrangle_data:
        wrangle_data(train_paths, args)

    batch_data = get_train_data(train_paths, args)
    
    train_data = {}
    for data in batch_data:
        if data[0] not in train_data.keys():
            train_data[data[0]] = []
        train_data[data[0]].append(data)

    base_model = ResNet(input_shape=(args.model_image_size[0],args.model_image_size[1],3), weights="imagenet", include_top=False)

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    
    if args.add_metadata:
        input_gsd = Input(shape=(2,))
        x = concatenate([x, input_gsd])
        out = Dense(1, name="out")(x)
        model = Model(inputs=[base_model.input,input_gsd], outputs=out)
    else:
        out = Dense(1, name="out")(x)
        model = Model(inputs=base_model.input, outputs=out)
        
    model.compile(optimizer="Adam", loss=huber_loss_mean)
    
    print(model.summary())

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
            
    train_datagen = image_generator(train_data, args)

    checkpoint_filepath = os.path.join(args.checkpoint_dir, "weights.{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="loss", verbose=0, save_best_only=False,
                                 save_weights_only=False, mode="auto", period=args.save_period)

    triangular_clr = CyclicLR(mode='triangular')
    callbacks_list = [checkpoint, triangular_clr]

    model.fit_generator(generator=train_datagen,
                        steps_per_epoch=((args.sample_size * len(train_paths)) / args.batch_size + 1),
                        epochs=args.num_epochs, callbacks=callbacks_list)


def test(args):

    _, test_paths = get_paths()
    if args.wrangle_data:
        wrangle_data(test_paths, args, is_test=True)
        
    weights_path = os.path.join(args.checkpoint_dir, "weights.15.hdf5")

    model = load_model(weights_path, custom_objects={"huber_loss_mean":huber_loss_mean})

    predicted_pops,gt_pops = [],[]
    predicted_tots,gt_tots = [],[]
    
    gt_dict = json.load(open("gt.json", "r"))

    for test_path in tqdm(test_paths):
        basename = os.path.basename(test_path).replace(".tif", "")
        gt_pop = gt_dict[basename]
        image,tr = load_tiff(test_path)
        if args.add_osm:
            osm_path = os.path.join(params.osm_dir, basename+".png")
            osm_mask = cv2.imread(osm_path, 0)
            osm_mask = cv2.resize(osm_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            image[osm_mask==0] = 0
        gt_tot,pred_tot = 0,0
        for x1 in tqdm(range(0, image.shape[1], args.test_block_size)):
            x2 = min(image.shape[1], x1 + args.test_block_size)
            for y1 in range(0, image.shape[0], args.test_block_size):
                y2 = min(image.shape[0], y1 + args.test_block_size)
                ratio = min((x2-x1) / args.test_block_size, (y2-y1) / args.test_block_size)
                if ratio < 0.5:
                    continue

                sub_image = np.zeros((args.test_block_size, args.test_block_size, 3))
                sub_image[:(y2-y1),:(x2-x1),:] = image[y1:y2,x1:x2,:3]
                
                if sub_image.sum() == 0:
                    continue
                
                if sub_image.shape[:2] != args.model_image_size: 
                    sub_image = cv2.resize(sub_image, args.model_image_size)
                
                xgsd,ygsd = tr[1],tr[-1]
                
                gsd = np.zeros((1,2))
                gsd[0,0] = abs(xgsd)
                gsd[0,1] = abs(ygsd)
                area = np.float(abs(xgsd*ygsd)*abs(y2-y1)*abs(x2-x1))
                    
                sub_image = imagenet_utils.preprocess_input(sub_image) / 255.0
                sub_image = np.expand_dims(sub_image, axis=0)
                if args.add_metadata:
                    pred = model.predict([sub_image,gsd])[0][0]
                else:
                    pred = model.predict(sub_image)[0][0]
                predicted_pop = pred*area
                pred_tot += predicted_pop
        
        predicted_tots.append(pred_tot)
        gt_tots.append(gt_pop)
        curr_mape = np.mean(100 * np.divide(np.abs(np.array(predicted_tots)-np.array(gt_tots)), gt_tots))
        print(curr_mape)
        
    mape = np.mean(100 * np.divide(np.abs(np.array(predicted_tots)-np.array(gt_tots)), gt_tots))
    code.interact(local=locals())





