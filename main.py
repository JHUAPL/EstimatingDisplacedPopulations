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
import argparse

from utilities.ml_utils import train,test
import params

def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="")
    parser.add_argument("--test", action="store_true", help="")
    parser.add_argument("--augmentation", action="store_true", help="")
    parser.add_argument("--add-metadata", action="store_true", help="")
    parser.add_argument("--multiprocessing", action="store_true", help="")
    parser.add_argument("--add-osm", action="store_true", help="")
    parser.add_argument("--wrangle-data", action="store_true", help="")
    parser.add_argument("--gpu", type=str, help="index of gpu device to use", default="0")
    parser.add_argument("--batch-size", type=int, help="", default=40)
    parser.add_argument("--num-epochs", type=int, help="", default=100)
    parser.add_argument("--save-period", type=int, help="", default=5)
    parser.add_argument("--sample-size", type=int, help="", default=200)
    parser.add_argument("--test-block-size", type=int, help="", default=512)
    parser.add_argument("--overlap", type=int, help="", default=0)
    parser.add_argument("--block-sizes", type=int, nargs="+", help="", default=[1024])
    parser.add_argument("--model-image-size", type=int, nargs="+", help="", default=(224,224))
    parser.add_argument("--checkpoint-dir", type=str, help="",
                        default=os.path.join(params.sandbox_dir, 'checkpoints'))
    parser.add_argument("--chip-pop-dir", type=str, help="",
                        default=os.path.join(params.sandbox_dir, 'tmp_chip_pop_data'))
    
    return parser

if __name__=="__main__":

    parser = create_arg_parser()
    args = parser.parse_args()
    
    if not os.path.isdir(args.chip_pop_dir):
        os.makedirs(args.chip_pop_dir)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.train:
        train(args)
    elif args.test:
        test(args)