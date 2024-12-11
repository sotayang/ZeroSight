import json
import time
import traceback

import sys, os
from glob import glob

from tqdm import tqdm
from pathlib import Path

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--bert_relativeCaptions_files_path", type=str, help="Folder path of files containing candidate relative captions after self Bert filtering", required=True)
parser.add_argument("--relativeCaptions_files_path", type=str, help="Folder path of files containing final relative captions", required=True)

args = parser.parse_args()

prompt_format = """
    Given several declarative sentences without subjects but with similar meanings, you need to generate a declarative sentence in the same format. 
    However, the generated declarative sentence should be able to summarize all the given declarative sentences.
    I will start with \"Given declarative sentences:\" to provide you with the declarative sentences.
    When generating, strictly follow the given example, and do not generate any other characters. The generated summary declarative sentence must be simple enough and generated in one line.

    Input example:
    [\"shows a person standing alone in a room with patterned wallpaper and no visible candles.\",
    \"shows a single person standing in front of a patterned wall.\",
    \"shows a lone individual standing indoors with a different background and lighting.\",
    \"shows a different woman in a dimly lit area with fewer visible light sources.\",
    \"shows a person alone in dim lighting.\"]


    Generated example:
    shows a person standing alone.

    Given declarative sentences:
    {}
"""

for root, dirs, files in os.walk(args.bert_relativeCaptions_files_path):
    files.sort()
    for file in tqdm(files):
        file_path = os.path.join(root,file)
        file_name, _ = os.path.splitext(file)
        with open(file_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)
        output = []
        for i in range(len(datas)):
            if len(datas[i]) != 0:
                allCaps = []
                referenceImg = datas[i]['pair']['referenceImg']
                smilarImgs = []
                smilarImgs.append(datas[i]['pair']['smilarImg'])
                allCaps.append(datas[i]['pair']['relativeCaption'])
                for similar_pair in datas[i]['similar_pairs']:
                    smilarImgs.append(similar_pair['pair']['smilarImg'])
                    allCaps.append(similar_pair['pair']['relativeCaption'])
                
                prompt = prompt_format.format(allCaps)
                gpt_client = GPT()
                answer = gpt_client.chat(prompt, max_cycle = 20)
                while answer is None:
                    answer = gpt_client.chat(prompt, max_cycle = 20)
                output_data = {
                    'referenceImg':referenceImg,
                    'smilarImgs':smilarImgs,
                    'relativeCap':answer
                }
                output.append(output_data)

        json_filename = args.relativeCaptions_files_path + '/' + file_name + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)

        print(f"Data has been written to {json_filename}")