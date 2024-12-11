# -*- coding: utf-8 -*-
import os, sys
from glob import glob
import json
import math
import random
import re

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--video_frame_folders_path", type=str, help="Folder path of folders containing video frames", required=True)
parser.add_argument("--gpt4o_referImgs_files_path", type=str, help="Folder path of files containing candidate reference images", required=True)

args = parser.parse_args()

prompt_noFormer = """
        Task Description:
        You will be given several images extracted from video frames in sequence. You need to select one image that meets the following requirements.

        Requirements:
        1. The selected image must be aesthetically pleasing, conforming to human aesthetics, and must be clear without any blurry areas.
        2. The selected image should be sufficiently distinct from the other images, and the difference should be describable in one sentence.
        3. The number of people and objects in the image should be moderate, neither too many nor too few.
        4. The overall arrangement of elements in the image should not appear too monotonous or too chaotic.
        5. Images with only English words are not allowed to be selected.
        6. Only use Arabic numerals to output the sequence number of the selected image, and do not output any other characters.
        7. If there is no suitable image, output the number 0 using Arabic numerals, and do not output any other characters.

        Input image sequence:

        """

prompt_withFormer = """
        Task Description:
        You will be given several images extracted from video frames in sequence. The first image will be used as a reference image, and you need to select one image from the remaining images that meets the following requirements.

        Requirements:
        1. The selected image must not resemble the reference image and must have sufficient differences from it.
        2. The selected image must be aesthetically pleasing, conforming to human aesthetics, and must be clear without any blurry areas.
        3. The selected image should be sufficiently distinct from the other images (excluding the reference image), and the difference should be describable in one sentence.
        4. The number of people and objects in the image should be moderate, neither too many nor too few.
        5. The overall arrangement of elements in the image should not appear too monotonous or too chaotic.
        6. Images with only English words are not allowed to be selected.
        7. Only use Arabic numerals to output the sequence number of the selected image, ranging from 2 to 10, and do not output any other characters.
        8. If there is no suitable image, output the number 1 using Arabic numerals, and do not output any other characters.

        Input image sequence:

        """

for root, dirs, files in os.walk(args.video_frame_folders_path):
    dirs.sort()
    for dir in dirs:
        imagesAll = []
        referImgs = []
        imagesPath = os.path.join(root,dir)
        for subRoot, subDirs, subFiles in os.walk(imagesPath):
            subFiles.sort()
            for name in subFiles:
                imagesAll.append(os.path.join(subRoot,name))

        noImg = True
        referImgLast = ''

        while len(imagesAll)>0:
            if noImg:
                imagesFetched = imagesAll[:10]
                gpt4o_client = GPT()
                answer = gpt4o_client.vision(prompt_noFormer, imagesFetched, max_cycle = 20)
                while answer is None:
                    answer = gpt4o_client.vision(prompt_noFormer, imagesFetched, max_cycle = 20)
                row_number = re.findall(r'\d+', answer)
                while int(row_number[0]) > len(imagesFetched) or int(row_number[0]) < 0:
                    answer = gpt4v_client.vision(prompt_noFormer, imagesFetched, max_cycle = 20)
                    while answer is None:
                        answer = gpt4v_client.vision(prompt_noFormer, imagesFetched, max_cycle = 20)
                    row_number = re.findall(r'\d+', answer)
                if int(row_number[0]) > 0:
                    referImgLast = imagesFetched[int(row_number[0])-1]
                    referImgs.append(referImgLast)
                    noImg = False
                else:
                    noImg = True
                temp = imagesAll[10:]
                imagesAll = temp
            else:
                imagesFetched_temp = imagesAll[:9]
                imagesFetched = [referImgLast]
                imagesFetched.extend(imagesFetched_temp)
                gpt4o_client = GPT()
                answer = gpt4o_client.vision(prompt_withFormer, imagesFetched, max_cycle = 20)
                while answer is None:
                    answer = gpt4o_client.vision(prompt_withFormer, imagesFetched, max_cycle = 20)
                row_number = re.findall(r'\d+', answer)
                while int(row_number[0]) > len(imagesFetched) or int(row_number[0]) < 1:
                    answer = gpt4v_client.vision(prompt_withFormer, imagesFetched, max_cycle = 20)
                    while answer is None:
                        answer = gpt4v_client.vision(prompt_withFormer, imagesFetched, max_cycle = 20)
                    row_number = re.findall(r'\d+', answer)
                if int(row_number[0]) > 1:
                    referImgLast = imagesFetched[int(row_number[0])-1]
                    referImgs.append(referImgLast)
                    noImg = False
                else:
                    noImg = True

                temp = imagesAll[9:]
                imagesAll = temp

        output = {
            'referenceImages':referImgs
        }
        json_filename = args.candidate_referImgs_folders_path + dir + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)

        print(f"Data has been written to {json_filename}")