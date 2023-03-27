

第一步: 熟悉 finetune whisper 模型的流程，通过commonvoice 语料熟悉

第二步：改写整个finetune 模块 （主要是爱尔兰语料的预处理以及如何放到 pytorch当中，主要是构建Dataset子类）

第三步：构建一个 mini dataset ，划分出训练集和测试集来检验整个finetune 流程的正确性

第四步：如何验证爱尔兰语和哪个语言的特性最像（主要是文本的表示，文本表示层面要大于语义层面）

第五步： 如何对tokenizer做一个finetune训练，主要是现有的 whisper 中的tokenizer 不支持爱尔兰语， （https://blog.csdn.net/qq_35812205/article/details/120002522#21_tokenizer_251）

第六步： 如何对爱尔兰语， 找到最合适的训练参数

第七步： 分析识别错误的句子，统计出不同类型的错误后，再考虑其他方案提升模型

