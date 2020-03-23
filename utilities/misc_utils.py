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

import gdal
import numpy as np
import json
import os
import code
import string
import cv2
import utm
import multiprocessing
from tqdm import tqdm
from datetime import datetime
from shapely.geometry import Polygon, LineString

from utilities.pop_utils import population_lookup
from utilities.pop_utils import get_pop_polygons

import params


pop_polygons = get_pop_polygons(params.shape_dir)

def _wrangle_helper(item):
    
    image_path, block_sizes, overlap, chip_pop_dir = item
    
    image,tr = load_tiff(image_path, size_only=False)

    basename = os.path.basename(image_path).replace(".tif", "")
    chip_data_file = os.path.join(chip_pop_dir, basename+".json")

    chip_coords_orig = get_chip_coords(image.shape[:2], block_sizes, overlap)
    all_gt_orig = get_pop_gt(chip_coords_orig, tr, pop_polygons, image, image_path)
    chip_coords,all_gt = [],[]
    for i in range(len(all_gt_orig)):
        if all_gt_orig[i] is not None:
            all_gt.append(all_gt_orig[i])
            chip_coords.append(chip_coords_orig[i])

    data = {"all_gt":all_gt, "chip_coords":chip_coords}
    json.dump(data, open(chip_data_file, "w"))

    return None
    
def wrangle_data(image_paths, args):
    
    print("Data wrangling")
    
    items = [(image_path, args.block_sizes, args.overlap, args.chip_pop_dir) for image_path in image_paths]
    
    if args.multiprocessing:
        pool = multiprocessing.Pool()
        results = list(tqdm(pool.imap_unordered(_wrangle_helper, items), total=len(items)))
        pool.close()
        pool.join()
    else:
        for item in tqdm(items):
            result = _wrangle_helper(item)
    
    
def get_latlng(x, y, tr):
    xutm = tr[0] + (x * tr[1]) + (y * tr[2])
    yutm = tr[3] + (x * tr[4]) + (y * tr[5])
    
    lat,lng = [],[]
    
    for i in range(len(xutm)):
        currlat,currlng = utm.to_latlon(xutm[i], yutm[i], params.utm_zone[0], params.utm_zone[1])
        lat.append(currlat)
        lng.append(currlng)
    
    return lat,lng

def get_pop_gt(chip_coords, tr, pop_polygons, image, image_path):
    valid_mask = np.sum(image, axis=2)>0
    
    base_datetime = get_datetime(os.path.basename(image_path))
    
    all_gt = []
    
    for coord in chip_coords:
        x1,y1,x2,y2 = coord
    
        sub_mask = valid_mask[y1:y2, x1:x2]
        
        ratio = np.float(sub_mask.sum()) / np.float(sub_mask.shape[0]*sub_mask.shape[1])
        
        if ratio < 0.1:
            all_gt.append(None)
            continue
        

        _,contours, hierarchy = cv2.findContours(sub_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pop = np.float(0)
        for contour in contours:
            curr_contour = np.squeeze(contour)
            if curr_contour.shape[0]<3:
                continue
                
            curr_contour[:,0] += x1
            curr_contour[:,1] += y1

            lat,lng = get_latlng(curr_contour[:,0], curr_contour[:,1], tr)

            curr_coords = [(lng[i],lat[i]) for i in range(len(lat))]
            
            if len(curr_coords)<3:
                continue

            line = LineString(np.array(curr_coords))
            simplified_line = line.simplify(params.tolerance, preserve_topology=True)
            lng = np.array(simplified_line.coords.xy[0].tolist())
            lat = np.array(simplified_line.coords.xy[1].tolist())

            curr_coords = [(lat[i],lng[i]) for i in range(len(lat))]

            if len(curr_coords)<3:
                continue

            bbox_polygon = Polygon(curr_coords)
            pop += population_lookup(bbox_polygon, base_datetime, pop_polygons)
            
        xgsd = tr[1]
        ygsd = tr[-1]
               
        area = np.float(abs(xgsd*ygsd)*abs(y2-y1)*abs(x2-x1))
                
        all_gt.append((pop,area,(xgsd,ygsd)))
        
    return all_gt

def get_chip_coords(image_size, block_sizes, overlap):            
    
    coords = []
    for block_size in block_sizes:
        yend, xend = np.subtract(image_size, (block_size,block_size))
        x = np.linspace(0, xend, np.ceil(xend / np.float(block_size - overlap)) + 1, endpoint=True).astype('int')
        y = np.linspace(0, yend, np.ceil(yend / np.float(block_size - overlap)) + 1, endpoint=True).astype('int')
        for x1 in x:
            x2 = x1 + block_size
            for y1 in y:
                y2 = y1 + block_size
                coords.append(np.array([x1,y1,x2,y2]).astype(np.int).tolist())      
    return coords

def get_batch_inds(idx, batch_size):
    n = len(idx)
    batch_inds = []
    idx0 = 0
    to_process = True
    while to_process:
        idx1 = idx0 + batch_size
        if idx1 > n:
            idx1 = n
            idx0 = idx1 - batch_size
            to_process = False
        batch_inds.append(idx[idx0:idx1])
        idx0 = idx1
    return batch_inds

def get_paths():
    basenames = json.load(open(params.image_basenames_path, "r"))
    train_paths,test_paths = [],[]
    for basename in basenames:
        image_path = os.path.join(params.image_dir, basename+".tif")
        if not os.path.isfile(image_path):
            continue
        if get_subcamp_name(basename) in params.test_camps:
            test_paths.append(image_path)
        else:
            train_paths.append(image_path)
            
    return train_paths,test_paths

def load_tiff(image_path, size_only=False):
    image_data = gdal.Open(image_path)
    
    image = image_data.ReadAsArray()
    if len(image.shape)>2:
        image = np.transpose(image, [1,2,0])
    tr = image_data.GetGeoTransform()
    return image,tr

def get_subcamp_name(filename):
    fname = filename.lower()
    fname = fname.split('_')[1:]
    #strip punctuation
    fname = [''.join(c for c in seg if c not in string.punctuation) for seg in fname]
    for segment in fname:
        if 'kutupalong' in segment:
            return 'kutupalongrc'
        if 'choukhali' in segment:
            return 'choukhlai'
        if '01e' in segment or 'camp1e' in segment:
            return 'camp_01e'
        if '01w' in segment or 'camp1w' in segment:
            return 'camp_01w'
        if '02e' in segment or 'camp2e' in segment:
            return 'camp_02e'
        if '02w' in segment or 'camp2w' in segment:
            return 'camp_02w'
        if '03' in segment or 'camp3' in segment:
            return 'camp_03'
        if '04ext' in segment  or 'camp4ext' in segment:
            return 'camp_04ext'
        if '04' in segment or 'camp4' in segment:
            return 'camp_04'
        if '05' in segment or 'camp5' in segment:
            return 'camp_05'
        if '06' in segment or 'camp6' in segment:
            return 'camp_06'
        if '07' in segment or 'camp7' in segment:
            return 'camp_07'
        if '08e' in segment or 'camp8e' in segment:
            return 'camp_08e'
        if '08w' in segment or 'camp8w' in segment:
            return 'camp_08w'
        if '09' in segment or 'camp9' in segment:
            return 'camp_09'
        if '10' in segment or 'camp10' in segment:
            return 'camp_10'
        if '11' in segment:
            return 'camp_11'
        if '12' in segment:
            return 'camp_12'
        if '13' in segment:
            return 'camp_13'
        if '14' in segment:
            return 'camp_14'
        if '15' in segment:
            return 'camp_15'
        if '16' in segment:
            return 'camp_16'
        if '17' in segment:
            return 'camp_17'
        if '19' in segment:
            return 'camp_19'
        if '20ext' in segment:
            return 'camp_20ext'
        if 'chakmarkul' in segment or '21' in segment:
            return 'camp_21_chakmarkul'
        if 'unchiprang' in segment or '22' in segment:
            return 'camp_22_unchiprang'
        if 'shamlapur' in segment or '23' in segment:
            return 'camp_23_shamlapur'
        if 'leda' in segment or '24' in segment:
            return 'camp_24_leda'
        if 'khali' in segment or '25' in segment:
            return 'camp_25_ali_khali'
        if 'nayapara' in segment or '26' in segment:
            return 'camp_26_nayapara'
        if 'jadimura' in segment or '27' in segment:
            return 'camp_27_jadimura'
        if '18' in segment:
            return 'camp_18'
        if '20' in segment:
            return 'camp_20'
        
    return filename

def get_datetime(basename):
    datestr = basename.split('_')[0].lower()
    if datestr == 'april2018':
        return datetime(2018, 4, 17)
    elif datestr == 'may2018':
        return datetime(2018, 5, 20)
    elif datestr == 'june2018':
        return datetime(2018, 6, 14)
    elif datestr == 'july2018':
        return datetime(2018, 7, 22)
    elif datestr == 'august2018':
        return datetime(2018, 9, 4)
    elif datestr == 'septemper-october2018':
        return datetime(2018, 10, 10)
    elif datestr == 'november2018':
        return datetime(2018, 11, 20)
    elif datestr == 'january2019':
        return datetime(2018, 12, 19)
    elif datestr == 'march2019':
        return datetime(2019, 2, 13)
    else:
        print(datestr)