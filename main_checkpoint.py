import os
import argparse
import yaml

import whisper

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from finetune_checkpoint.Utils import load_irish_data
from finetune_checkpoint.Model import WhisperModelModule


def get_args():
    parser = argparse.ArgumentParser(description=("Finetune whisper model by using Irish speech!"))

    parser.add_argument(
        "--trainfile",
        required=False,
        default="./train.list",
        type=str,
        help="train dataset file including wav and text",
    )

    parser.add_argument(
        "--testfile",
        required=False,
        default="./test.list",
        type=str,
        help="test dataset file including wav and text",
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
        default="./config/finetune_checkpoint_irish.yaml",
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

        # load train and test dataset
        train_list = load_irish_data(config["trainfile"])

        test_list = load_irish_data(config["testfile"])

        print("train num is :", len(train_list))
        print("test num is :", len(test_list))

        log_output_dir = os.path.join(config["output_dir"], "logs")
        check_output_dir = os.path.join(config["output_dir"], "checkpoints")

        os.makedirs(log_output_dir, exist_ok=True)
        os.makedirs(check_output_dir, exist_ok=True)

        tflogger = TensorBoardLogger(save_dir=log_output_dir, name="whisper-Irish", version="00001")

        checkpoint_callback = ModelCheckpoint(dirpath=check_output_dir, filename="checkpoint-{epoch:04d}", save_top_k=-1)

        callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
        model = WhisperModelModule(config, train_list, test_list)

        trainer = Trainer(
            precision=config["precision"],
            accelerator="gpu",
            max_epochs=config["num_train_epochs"],
            accumulate_grad_batches=config["gradient_accumulation_steps"],
            logger=tflogger,
            callbacks=callback_list
        )

        trainer.fit(model)

        print("training finish")


if __name__ == "__main__":
    main()

