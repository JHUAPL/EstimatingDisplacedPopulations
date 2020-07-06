import os
import code
import argparse
import params

from utilities.ml_utils import train, test

def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="")
    parser.add_argument("--test", action="store_true", help="")
    parser.add_argument("--augmentation", action="store_true", help="")
    parser.add_argument("--add-metadata", action="store_true", help="")
    parser.add_argument("--multiprocessing", action="store_true", help="")
    parser.add_argument("--add-osm", action="store_true", help="")
    parser.add_argument("--load-best-train-weights", action="store_true", help="")
    parser.add_argument("--wrangle-data", action="store_true", help="")
    parser.add_argument("--gpu", type=str, help="index of gpu device to use", default="0")
    parser.add_argument("--batch-size", type=int, help="", default=40)
    parser.add_argument("--lr", type=float, help="", default=1.0e-4)
    parser.add_argument("--max_lr", type=float, help="", default=6.0e-3)
    parser.add_argument("--num-epochs", type=int, help="", default=60)
    parser.add_argument("--save-period", type=int, help="", default=6)
    parser.add_argument("--sample-size", type=int, help="", default=200)
    parser.add_argument("--test-block-size", type=int, help="", default=224)
    parser.add_argument("--overlap", type=int, help="", default=0)
    parser.add_argument("--block-sizes", type=int, nargs="+", help="", default=[1024, 720, 640, 480, 320, 224])
    parser.add_argument("--model-image-size", type=int, nargs="+", help="", default=(224,224))
    parser.add_argument("--checkpoint-dir", type=str, help="",
                        default=os.path.join(os.getcwd(), "checkpoints"))
    parser.add_argument("--chip-pop-dir", type=str, help="",
                        default=os.path.join(params.sandbox_dir, "tmp", "train_chips"))
    parser.add_argument("--test-chip-pop-dir", type=str, help="",
                        default=os.path.join(os.getcwd(), "dataset", "test_chips"))

    return parser

if __name__=="__main__":

    parser = create_arg_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.chip_pop_dir):
        os.makedirs(args.chip_pop_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.isdir(args.test_chip_pop_dir):
        os.makedirs(args.test_chip_pop_dir)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.train:
        train(args)
    elif args.test:
        test(args)
