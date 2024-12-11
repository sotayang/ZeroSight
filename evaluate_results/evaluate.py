import sys, os
import json

import numpy as np
import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--all_queries_file_path", type=str, help="File path containing all queries in ZeroSight or test set of ZeroSight", required=True)
parser.add_argument("--results_file_path", type=str, help="File path containing top-50 retrieval results of ZS-CIR or CIR", required=True)

args = parser.parse_args()

with open(args.all_queries_file_path, 'r', encoding='utf-8') as file:
    datas_ori = json.load(file)

datas_ori_dic = {}
for data in datas_ori:
    output = {
        'groundtruths': data['groundtruths'],
        'negativeInstances': data['negativeInstances']
    }
    datas_ori_dic[str(data['id'])] = output

ap_at5 = []
ap_at10 = []
ap_at25 = []
ap_at50 = []


pnr_ap_at5 = []
pnr_ap_at10 = []
pnr_ap_at25 = []
pnr_ap_at50 = []

with open(args.results_file_path, 'r', encoding='utf-8') as file:
    datas_retrieval = json.load(file)


for key in datas_retrieval:
    results = datas_retrieval[key]
    gts = datas_ori_dic[key]['groundtruths']
    negs = datas_ori_dic[key]['negativeInstances']

    sorted_index_names = np.array(results)
    gt_img_ids = np.array(gts)
    neg_img_ids = np.array(negs)


    map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
    precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
    precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

    neg_labels = torch.tensor(np.isin(sorted_index_names, neg_img_ids), dtype=torch.uint8)

    weights = torch.zeros_like(neg_labels, dtype=torch.float32)
    for i in range(1, len(neg_labels)):
        if i == 0:
            weights[i] = 1.0
        else:
            indices_neg = torch.nonzero(neg_labels[:i]).squeeze()
            if indices_neg.numel() > 0:
                weights[i] = torch.mean(indices_neg.float() / i+1)
            else:
                weights[i] = 1.0
    PNR_precisons = precisions * weights

    ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
    pnr_ap_at5.append(float(torch.sum(PNR_precisons[:5]) / min(len(gt_img_ids), 5)))


    ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
    pnr_ap_at10.append(float(torch.sum(PNR_precisons[:10]) / min(len(gt_img_ids), 10)))


    ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
    pnr_ap_at25.append(float(torch.sum(PNR_precisons[:25]) / min(len(gt_img_ids), 25)))


    ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))
    pnr_ap_at50.append(float(torch.sum(PNR_precisons[:50]) / min(len(gt_img_ids), 50)))


map_at5 = np.mean(ap_at5) * 100
map_at10 = np.mean(ap_at10) * 100
map_at25 = np.mean(ap_at25) * 100
map_at50 = np.mean(ap_at50) * 100

pnr_map_at5 = np.mean(pnr_ap_at5) * 100
pnr_map_at10 = np.mean(pnr_ap_at10) * 100
pnr_map_at25 = np.mean(pnr_ap_at25) * 100
pnr_map_at50 = np.mean(pnr_ap_at50) * 100

res = {
    'map_at5': map_at5,
    'map_at10': map_at10,
    'map_at25': map_at25,
    'map_at50': map_at50,
    'pnr_map_at5': pnr_map_at5,
    'pnr_map_at10': pnr_map_at10,
    'pnr_map_at25': pnr_map_at25,
    'pnr_map_at50': pnr_map_at50
}

for k, v in res.items():
    print(f"{k} = {v:.2f}")