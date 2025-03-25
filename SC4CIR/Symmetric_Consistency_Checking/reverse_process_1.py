import os, sys
sys.path.append(".")
from glob import glob
import json
import re

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--circo_test_set_path", type=str, help="Folder path of circo test set", required=True)
parser.add_argument("--model_results_path", type=str, help="Folder path of your model results", required=True)
parser.add_argument("--output_anti_query_path", type=str, help="Folder path saving anti-query of your model results", required=True)

args = parser.parse_args()

test_set_path = args.circo_test_set_path
results_path = args.model_results_path

prompt_format = """
    Task Description:
        You are given an image and a text description. The text description does not match the content of the image. You need to modify the text description and output it so that it matches the content of the image.

    Requirements:
        1. The modified text description must be simple enough to be output in one sentence and must match the format of the given text description.
        2. Only output the modified text description, and do not output any other characters.

    Input Text Description:
    {Text}

    Input Image:
"""

with open(test_set_path, 'r', encoding='utf-8') as file:
    test_datas = json.load(file)

with open(results_path, 'r', encoding='utf-8') as file:
    results_datas = json.load(file)

results_anti_query = []

for data in test_datas:
    refer_Img_Id = str(data["reference_img_id"]).zfill(12)
    relative_Cap = data["relative_caption"]
    query_Id = str(data["id"])
    candidate_Imgs_Id = results_datas[query_Id]
    imagesFetched = [f'CIRCO/COCO2017_unlabeled/unlabeled2017/{refer_Img_Id}.jpg']
    prompt = prompt_format.format(Text = relative_Cap)

    ####################################################################################
    ####################################################################################
    ###############                                                      ###############
    ###############                  GPT4o Answer.                       ###############
    ###############                                                      ###############     
    ####################################################################################
    ####################################################################################

    for candidate in candidate_Imgs_Id:
        output_data = {
            "reference_img_id":int(candidate),
            "relative_caption":answer,
            "shared_concept":data["shared_concept"],
            "ori_reference_img_id":data["reference_img_id"],
            "ori_id":data["id"],
            "id":len(results_anti_query)
        }
        results_anti_query.append(output_data)

        with open(args.output_anti_query_path, "a") as f:
            f.write(json.dumps(output_data, indent=4) + '\n')