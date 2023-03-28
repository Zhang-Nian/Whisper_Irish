import os
import argparse
import yaml

from finetune.Finetune_Irish import FinetuneIrish


def get_args():
    parser = argparse.ArgumentParser(description=("Finetune whisper model by using Irish speech!"))

    parser.add_argument(
        "--inputfile",
        required=False,
        default="./train.list",
        type=str,
        help="dataset file including wav and text",
    )

    parser.add_argument(
        "--outputdir",
        type=str,
        required=False,
        default="./whisper_finetune",
        help="directory to dump feature files.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        required=False,
        default="0", # select first gpu for training
        help="CUDA_VISIBLE_DEVICES"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./config/finetune_irish.yaml",
        help="yaml format configuration file.",
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]

    if config["dataset"] == "Irish":
        print("start to finetune Irish")
        
        trainer = FinetuneIrish(config)


if __name__ == "__main__":
    main()

