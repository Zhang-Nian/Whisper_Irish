import os
import argparse
import yaml

from sklearn.model_selection import train_test_split

from finetune.Utils import load_irish_data, make_special_format_like_commonvoice
from finetune.Dataset import IrishDataSet
from finetune.DataCollator import DataCollatorSpeechSeq2SeqWithPadding
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
        default="./config/finetune_transformer_irish.yaml",
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
        
        # load irish data
        wav_list, text_list = load_irish_data()

        # split into train and test
        train_wav, test_wav, train_text, test_text = train_test_split(wav_list, text_list, test_size=0.2)
        print("train size is :", len(train_wav))
        print("test size is :", len(test_wav))

        # generate special format in order to train
        irish_common_voice = make_special_format_like_commonvoice(train_wav, test_wav, train_text, test_text)

        # generate pytorch dataset
        train_dataset = IrishDataSet(config, irish_common_voice["train"])
        test_dataset = IrishDataSet(config, irish_common_voice["test"])
        
        # define data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(config)

        trainer = FinetuneIrish(config)

        trainer.train(config, train_dataset, test_dataset, data_collator)


if __name__ == "__main__":
    main()

