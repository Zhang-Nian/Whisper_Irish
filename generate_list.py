import os
import sys

import contextlib
import wave


def write_data(file_name, data):
    with open(file_name, 'a') as file_object:
        file_object.write(data)
        file_object.write("\n")


def get_wav_duration(wav_path):
    wav_length = 0
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = frames / float(rate)

    return wav_length


def load_scp_map(file_path):
    scp_map = {}
    
    f = open(file_path)
    readlines = f.readlines()
    for line in readlines:
        line = line.rstrip("\n")
        data = line.split(" ")
        
        wav_id = data[0]
        wav_path = data[1]
        scp_map[wav_id] = wav_path

    return scp_map


def main():
    max_input_length = 30
    min_input_length = 0

    wav_scp_file = sys.argv[1]
    text_file = sys.argv[2]
    output_file = sys.argv[3]

    scp_map = load_scp_map(wav_scp_file)

    f = open(text_file)
    readlines = f.readlines()
    for line in readlines:
        line = line.rstrip("\n")
        data = line.split(" ")

        wav_id = data[0]
        wav_text = " ".join(data[1:])

        wav_path = scp_map[wav_id]
        wav_path = "/media/storage" + wav_path

        # add filtering
        cur_wav_duration = get_wav_duration(wav_path)
        if cur_wav_duration == min_input_length or cur_wav_duration > max_input_length:
            print(wav_id, " wav length is not normal to finetune the model")
            continue

        w_data = wav_path + "\t" + wav_text
        write_data(output_file, w_data)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_list.py wav.scp text output_file")
        exit()
    main()

