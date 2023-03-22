import os 
import re
from urllib.parse import urlparse


def write_data(file_name, data):
    with open(file_name, 'a') as file_object:
        file_object.write(data)
        file_object.write("\n")


def main():
    os.makedirs("./wav", exist_ok=True)

    file_path = "../irish_asr_data/corpus.txt"

    f = open(file_path)
    readlines = f.readlines()

    for line in readlines:
        line = line.rstrip("\n")
        data = line.split("\t")

        wav_id = data[0]
        wav_url = data[3]

        wav_text = data[4]
        wav_text = re.sub(r'[.,"\'-?:!;]', '', wav_text)
        wav_text = wav_text.strip()

        file_name = os.path.basename(urlparse(wav_url).path)
        w_data = "./wav/" + file_name + "\t" + wav_text

        write_data("train.list", w_data)

        download_cmd = "wget -P ./wav " + wav_url
        os.system(download_cmd)


if __name__ == "__main__":
    main()

