import argparse 
import os
import json
import sys
import pandas as pd

def parse():
    parser = argparse.ArgumentParser(description='Process arguments passed to script')
    parser.add_argument('--input')

    (args, extra_args) = parser.parse_known_args()
    print(args)
    print(extra_args)
    return args
    

def main(args):
    df=pd.read_parquet(args.input)
    print(df)
    return

if __name__ == '__main__':
    args = parse()
    main(args)
