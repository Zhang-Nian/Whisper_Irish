import os

import whisper

from transformers import WhisperTokenizer

#wtokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language="chinese", task="transcribe")

#aa = len(whisper.tokenizer.get_tokenizer('english').tokenizer)
#print("aa is :", aa)

english_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="english", task="transcribe")
spanish_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="spanish", task="transcribe")

english_text = "Today is a beautiful day"
irish_text = "níorbh é a saol siúd a shaol féin"

text = irish_text

#labels = wtokenizer.encode(text)
#final_text = wtokenizer.decode(labels, skip_special_tokens=True)
#final_text = wtokenizer.decode(labels)

print("text :", text)
#print("labels :", labels)
#print("final text :", final_text)

english_ids = english_tokenizer([text]).input_ids
spanish_ids = spanish_tokenizer([text]).input_ids

english_decoded_str = english_tokenizer.decode(english_ids[0], skip_special_tokens=True)
spanish_decoded_str = spanish_tokenizer.decode(spanish_ids[0], skip_special_tokens=True)

print("english_ids :", english_ids)
print("spanish_ids :", spanish_ids)

print("english_decoded_str :", english_decoded_str)
print("spanish_decoded_str :", spanish_decoded_str)


"""
def get_token_text(text):
    labels = tokenizer([text]).input_ids
    decoded_str = tokenizer.decode(labels[0], skip_special_tokens=True)

    return decoded_str

def main():
    file_path = "test.list"

    diff_num = 0

    f = open(file_path)
    readlines = f.readlines()

    total_num = len(readlines)

    for line in readlines:
        line = line.rstrip("\n")
        data = line.split("\t")
        cur_text = data[1]

        decode_str = get_token_text(cur_text)

        if cur_text != decode_str:
            w_data = cur_text + "\t" + decode_str
            print("w_data is :", w_data)

            diff_num += 1


    rate = float(diff_num / total_num)

    print("diff num is :", diff_num)
    print("total num is :", total_num)
    print("rate is :", rate)


#if __name__ == "__main__":
#    main()
"""
