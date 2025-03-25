import os, sys
sys.path.append(".")
from glob import glob
import json
import re
from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--circo_test_set_path", type=str, help="Folder path of circo test set", required=True)
parser.add_argument("--model_results_path", type=str, help="Folder path of your model results", required=True)
parser.add_argument("--output_anti_caption_path", type=str, help="Folder path saving anti-caption of your model results", required=True)

args = parser.parse_args()

test_set_path = args.circo_test_set_path
results_path = args.model_results_path


prompt = """
        Task Description:
        You will be given two images in sequence. You need to generate a declarative sentence that, when combined with the content of the first image, will enable a search engine to accurately retrieve the second image.

        Output Example:
        "add more people in a bright room."

        Requirement:
        1. The output sentence should describe the modifications needed to change the first image into the second image.
        2. The subject of the output sentence should be "the first image," and the predicate can be a series of verbs such as "increase", "enlarge", "reduce", "show", "zoom", etc. However, the output must follow the given output example, and the subject must be omitted.
        3. Only this sentence can be output, and no other characters are allowed.
        4. The declarative sentence must be output in English.
        5. The modifications described in the declarative sentence should focus more on the elements within the images (people, objects, colors, numbers, environments, etc.).

        Input Image Sequence:

        """


with open(test_set_path, 'r', encoding='utf-8') as file:
    test_datas = json.load(file)

with open(results_path, 'r', encoding='utf-8') as file:
    results_datas = json.load(file)

for data in tqdm(test_datas):
    refer_Img_Id = str(data["reference_img_id"]).zfill(12)
    relative_Cap = data["relative_caption"]
    query_Id = str(data["id"])
    candidate_Imgs_Id = results_datas[query_Id]

    anti_captions = []

    for candidate in candidate_Imgs_Id:
        candidate_Img_Id = candidate.zfill(12)
        imagesFetched = [f'CIRCO/COCO2017_unlabeled/unlabeled2017/{refer_Img_Id}.jpg',
            f'CIRCO/COCO2017_unlabeled/unlabeled2017/{candidate_Img_Id}.jpg']
        
        ####################################################################################
        ####################################################################################
        ###############                                                      ###############
        ###############                  GPT4o Answer.                       ###############
        ###############                                                      ###############     
        ####################################################################################
        ####################################################################################

        anti_captions.append(answer)

    output_data = {
        "reference_img_id":data["reference_img_id"],
        "relative_caption":data["relative_caption"],
        "shared_concept":data["shared_concept"],
        "candidate_Imgs_Id":candidate_Imgs_Id,
        "anti_captions":anti_captions,
        "id":data["id"]
    }

    with open(args.output_anti_caption_path, "a") as f:
        f.write(json.dumps(output_data, indent=4) + '\n')
