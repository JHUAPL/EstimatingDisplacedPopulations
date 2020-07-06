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
from utilities.misc_utils import wrangle_test_data

import keras.backend as K
import tensorflow as tf

from classification_models.keras import Classifiers

from libs.CLR.clr_callback import CyclicLR

architecture = "resnet50"
ResNet, preprocess_input = Classifiers.get(architecture)

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Transpose,
    RandomRotate90,
    HueSaturationValue,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
)

def load_batch(chip_coords, all_density, all_gsd, image, gsd_train_mean, gsd_train_std,  args):
    images = np.zeros((len(chip_coords), args.model_image_size[0], args.model_image_size[1], 3))
    densities = np.zeros((len(chip_coords), 1))
    gsds = np.zeros((len(chip_coords), 2))

    for i in range(len(chip_coords)):
        x1, y1, x2, y2 = chip_coords[i]
        density = all_density[i]
        gsd = all_gsd[i]
        sub_image = image[y1:y2, x1:x2, :3]
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
        gsds[i, 0] = (abs(gsd[0]) - gsd_train_mean[0]) / gsd_train_std[0]
        gsds[i, 1] = (abs(gsd[1]) - gsd_train_mean[1]) / gsd_train_std[1]

    images = imagenet_utils.preprocess_input(images) / 255.0


    return images, densities, gsds


def image_generator(train_data, args):
    norm_params_path = os.path.join(params.normalization_params_dir, 'normalization_parameters.json')
    norm_data = json.load(open(norm_params_path, 'r'))
    gsd_train_mean = [norm_data["mean"][i] for i in range(2)]
    gsd_train_std = [norm_data["var"][i] ** 0.5 for i in range(2)]

    while True:
        prev_image_path = ""
        sample_num = np.int(np.ceil(args.sample_size / args.batch_size))

        batch_data = []
        for key, val in train_data.items():
            idx = np.random.permutation(len(val))
            for ind in idx[:sample_num]:
                batch_data.append(val[ind])

        for data in batch_data:
            image_path, chip_coords, density, gsd = data
            if prev_image_path != image_path:
                basename = os.path.basename(image_path).replace(".tif", "")
                image, tr = load_tiff(image_path)
                if args.add_osm:
                    osm_path = os.path.join(params.osm_dir, basename + ".png")
                    osm_mask = cv2.imread(osm_path, 0)
                    osm_mask = cv2.resize(osm_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    image[osm_mask==0] = 0
                prev_image_path = image_path
            images, densities, gsds = load_batch(chip_coords, density, gsd, image, gsd_train_mean, gsd_train_std, args)
            if args.add_metadata:
                yield ([images, gsds], densities)
            else:
                yield (images, densities)

def get_train_data(image_paths, args):
    batch_data = []

    min_density, max_density = np.inf, -np.inf

    for image_path in image_paths:
        basename = os.path.basename(image_path).replace(".tif", "")
        chip_data_file = os.path.join(args.chip_pop_dir, basename + ".json")
        data = json.load(open(chip_data_file, "r"))
        idx = np.random.permutation(len(data["chip_coords"]))
        batch_inds = get_batch_inds(idx, args.batch_size)
        for inds in batch_inds:
            batch_chip_coords = []
            batch_density = []
            batch_gsd= []
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

    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    train_datagen = image_generator(train_data, args)

    initial_epoch = 0
    if args.load_best_train_weights:
        weights_paths = glob(os.path.join(args.checkpoint_dir, "*.hdf5"))
        nums = [np.int(weights_path.split(".")[-2]) for weights_path in weights_paths]
        idx = np.argsort(nums)[::-1]
        weights_path = weights_paths[idx[0]]
        initial_epoch = np.int(weights_path.split(".")[-2])
        model = load_model(weights_path, custom_objects={"huber_loss_mean":huber_loss_mean})
    else:
        base_model = ResNet(input_shape=(args.model_image_size[0], args.model_image_size[1], 3), weights="imagenet", include_top=False)

        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.7)(x)
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

    checkpoint_filepath = os.path.join(args.checkpoint_dir, "weights.{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor="loss", verbose=0, save_best_only=False,
                                 save_weights_only=False, mode="auto", period=args.save_period)

    steps_per_epoch = ((args.sample_size * len(train_paths)) / args.batch_size + 1)
    triangular_clr = CyclicLR(base_lr=args.lr,
                            max_lr=args.max_lr,
                            mode='triangular2',
                            step_size= np.int(0.5 * args.save_period * steps_per_epoch))
    callbacks_list = [checkpoint, triangular_clr]

    model.fit_generator(generator=train_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=args.num_epochs,
                        callbacks=callbacks_list,
                        initial_epoch=initial_epoch)


def test(args):
    _, test_paths = get_paths()
    if args.wrangle_data:
        wrangle_test_data(test_paths, args)
    weights_paths = glob(os.path.join(args.checkpoint_dir, "*.hdf5"))
    nums = [np.int(weights_path.split(".")[-2]) for weights_path in weights_paths]
    idx = np.argsort(nums)[::-1]
    weights_path = weights_paths[idx[0]]

    print(f"Loading weights from: {weights_path}", flush=True)

    model = load_model(weights_path, custom_objects={"huber_loss_mean":huber_loss_mean})

    predicted_pops, gt_pops = [], []
    predicted_tots, gt_tots = [], []

    norm_params_path = os.path.join(params.normalization_params_dir, 'normalization_parameters.json')
    norm_data = json.load(open(norm_params_path, 'r'))
    gsd_train_mean = [norm_data["mean"][i] for i in range(2)]
    gsd_train_std = [norm_data["var"][i] ** 0.5 for i in range(2)]

    for test_path in tqdm(test_paths):
        basename = os.path.basename(test_path).replace(".tif", "")
        print(basename)

        data_file = os.path.join(args.test_chip_pop_dir, f"{basename}.json")
        data = json.load(open(data_file, "r"))
        gt_pop = data["total_gt"]
        pred_tot = 0

        image, tr = load_tiff(test_path)

        if args.add_osm:
            osm_path = os.path.join(params.osm_dir, basename + ".png")
            osm_mask = cv2.imread(osm_path, 0)
            osm_mask = cv2.resize(osm_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            image[osm_mask==0] = 0

        for i, coords in enumerate(tqdm(data["chip_coords"])):
            x1, y1, x2, y2 = coords

            sub_image_size = (min((y2-y1), (image.shape[0] - y1)), min((x2-x1), (image.shape[1] - x1)), 3)
            sub_image = np.zeros(sub_image_size)
            sub_image[:(y2-y1), :(x2-x1), :] = image[y1:y2, x1:x2, :3]

            if sub_image.shape[:2] != args.model_image_size:
                sub_image = cv2.resize(sub_image, args.model_image_size)

            xgsd, ygsd = tr[1], tr[-1]

            area = np.float(abs(xgsd) * abs(x2 - x1) * abs(ygsd) * abs(y2 - y1))
            gsd = np.zeros((1, 2))
            gsd[0, 0] = (abs(xgsd) - gsd_train_mean[0]) / gsd_train_std[0]
            gsd[0, 1] = (abs(ygsd) - gsd_train_mean[1]) / gsd_train_std[1]

            sub_image = sub_image.astype(np.uint8)

            if sub_image.sum() == 0:
                continue

            sub_image = sub_image.astype(np.float64)
            sub_image = imagenet_utils.preprocess_input(sub_image) / 255.0
            sub_image = np.expand_dims(sub_image, axis=0)
            if args.add_metadata:
                pred = model.predict([sub_image, gsd])[0][0]
            else:
                pred = model.predict(sub_image)[0][0]

            pred = abs(pred * (pred > 0.0)) # Remove negative predictions
            predicted_pop = pred * area

            pred_tot += predicted_pop

        predicted_tots.append(pred_tot)
        gt_tots.append(gt_pop)
        curr_mape = np.mean(100 * np.divide(np.abs(np.array(pred_tot) - np.array(gt_pop)), gt_pop))
        print(f"Current image pred: {pred_tot} gt: {gt_pop} and tested images MAPE: {curr_mape}%", flush=True)

    mape = np.mean(100 * np.divide(np.abs(np.array(predicted_tots) - np.array(gt_tots)), gt_tots))
    print(f"Test MAPE: {mape}%", flush=True)
