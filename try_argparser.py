# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:00:16 2022

@author: test1
"""

import argparse
import json
def parse():
    parser = argparse.ArgumentParser("This is a trying on argparse")
    parser.add_argument('--name', type=str)
    parser.add_argument('--age', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    print(args.__dict__)
    
    
    def save_parser(args, address):
        with open(address, "w", encoding="utf8") as f:
            json.dump(args.__dict__, f, ensure_ascii=False)
