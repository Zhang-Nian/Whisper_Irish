# Whisper

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.


## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. All of these tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing for a single model to replace many different stages of a traditional speech processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.



## Setup

We used Python 3.8 and [PyTorch](https://pytorch.org/) (gpu) to finetune models

    pip install datasets
    pip install transformers
    pip install librosa
    pip install evaluate
    pip install jiwer
    pip install gradio


## Finetune

[[Key link1]](https://huggingface.co/blog/fine-tune-whisper)
[[Key link2]](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event#python-script)
[[Key link3]](https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862)




## Ideas of finetune

    Step 1: Getting familiar with the flow of the finetune whisper model by using the commonvoice corpus

    Step 2: Rewrite the entire finetune module (mainly pre-processing of the Irish corpus and how to put it into pytorch, mainly building Dataset subclasses)

    Step 3: Build a mini dataset and divide the train dataset and test dataset to check the correctness of the whole finetune process

    Step 4: How to do a finetune training on the tokenizer, mainly because the existing tokenizer in whisper does not support Irish

        [[link1]](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)

    Step 5: How to find the most suitable training parameters for the Irish speech

    Step 6: The order of finetune, which parameter should be adjusted first, can it be adjusted separately?

    Step 7: Consider optimizing Adam or Adafactor in training

    Step 8: Use Data Enhancement to Boost Results

    Step 9: Analyze the sentences that identify errors and count the different types of errors before considering other options to improve the model


