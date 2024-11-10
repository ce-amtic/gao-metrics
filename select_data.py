import argparse
import pickle, time
import os, os.path as osp
import numpy as np
import random
import json 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='eval/testin')
    parser.add_argument('--output_dir', type=str, default='eval')
    parser.add_argument('--select_num', type=int, default=100)
    args = parser.parse_args()

    fn_list = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".dat")])
    
    # random shuffle 
    random.shuffle(fn_list)

    # select at most target_file_num fn
    fn_list = fn_list[:args.select_num]

    # save selected fn as json
    json_dict = {
        "fn_list": fn_list,
        "select_num": len(fn_list)
    }
    with open(osp.join(args.output_dir, 'selected_files.json'), 'w') as f:
        json.dump(json_dict, f)
