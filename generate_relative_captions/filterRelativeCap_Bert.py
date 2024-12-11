import sys, os
import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics.pairwise import cosine_similarity

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--gpt4o_relativeCaptions_files_path", type=str, help="Folder path of files containing candidate relative captions after GPT4o generation", required=True)
parser.add_argument("--bert_relativeCaptions_files_path", type=str, help="Folder path of files containing candidate relative captions after self Bert filtering", required=True)

args = parser.parse_args()

model_name = 'stsb-bert-large'  
model = SentenceTransformer(model_name)

for root, dirs, files in os.walk(args.gpt4o_relativeCaptions_files_path):
    files.sort()
    for file in tqdm(files):
        file_path = os.path.join(root,file)
        file_name, _ = os.path.splitext(file)
        with open(file_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)

        all_batches = []

        for data in datas:
            batches_output = []
            for i in range(len(data['pairs'])):
                pair_sim = []
                for j in range(len(data['pairs'])):
                    text1 = data['pairs'][i]['relativeCaption']
                    text2 = data['pairs'][j]['relativeCaption']

                    embedding1 = model.encode(text1, convert_to_tensor=True)
                    embedding2 = model.encode(text2, convert_to_tensor=True)

                    similarity = util.pytorch_cos_sim(embedding1, embedding2)
                    if similarity.item() >= 0.59 and i != j:
                        output_data = {
                            'pair':data['pairs'][j],
                            'similarity':similarity.item()
                        }
                        pair_sim.append(output_data)
                output_batch_pair = {
                    'pair':data['pairs'][i],
                    'similar_pairs':pair_sim
                }
                batches_output.append(output_batch_pair)

            all_batches.append(batches_output)

        output = []

        for pairs in all_batches:
            selected_pair = -1
            selected_sim = 0
            selected_num = 0
            for i in range(len(pairs)):
                pairs_num = len(pairs[i]['similar_pairs'])
                average_sim = 0
                if pairs_num == 0:
                    continue
                sum_sim = 0
                for sim_pair in pairs[i]['similar_pairs']:
                    sum_sim += sim_pair['similarity']
                average_sim = round(sum_sim/pairs_num,16)
                if average_sim > selected_sim:
                    selected_pair = i
                    selected_sim = average_sim
                    selected_num = pairs_num
                elif average_sim == selected_sim:
                    if pairs_num > selected_num:
                        selected_pair = i
                        selected_num = pairs_num
                else:
                    pass
            if selected_pair == -1:
                output_data = {}
            else:
                output_data = pairs[selected_pair]
            output.append(output_data)

        json_filename = args.bert_relativeCaptions_files_path + '/' + file_name + '.json'
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(output, json_file, ensure_ascii=False, indent=4)

        print(f"Data has been written to {json_filename}")
