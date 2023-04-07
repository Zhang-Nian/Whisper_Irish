import os
import argparse
import yaml

from finetune_transformer.Utils import load_irish_data, make_special_format_like_commonvoice
from finetune_transformer.Dataset import IrishDataSet
from finetune_transformer.DataCollator import DataCollatorSpeechSeq2SeqWithPadding
from finetune_transformer.Finetune_Irish import FinetuneIrish


def get_args():
    parser = argparse.ArgumentParser(description=("Finetune whisper model by using Irish speech!"))

    parser.add_argument(
        "--trainfile",
        required=False,
        default="./train.list",
        type=str,
        help="dataset file including wav and text",
    )

    parser.add_argument(
        "--testfile",
        required=False,
        default="./test.list",
        type=str,
        help="test dataset file including wav and text",
    )

    parser.add_argument(
        "--outputdir",
        type=str,
        required=False,
        default="./whisper_transformer_finetune",
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

        # modify out dir name, it depends on the config 
        config["outputdir"] = config["outputdir"] + "_" + config["model_size"] + "_" + config["tokenizer_language"]

        # load irish data
        train_wav, train_text = load_irish_data(config["trainfile"])
        test_wav, test_text = load_irish_data(config["testfile"])
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

