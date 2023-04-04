import argparse
import json

import pandas as pd
from pathlib import Path
import os
from glob import glob

def prep_data(file_path, label):

    df = pd.read_csv(file_path)
    df = df.drop(columns='Unnamed: 0')
    df = df.drop(columns='count')
    #df = df.drop(columns='temperature')
    #df = df.drop(columns='humidity')
    df.index.name = 'index'
    df['class'] = label

    return df

def parse_raw_file(raw_file_path):

    metadata = None

    with open(raw_file_path) as topo_file:
        for line in topo_file:
            line = line.strip()
            if line.startswith('#'):
                if line.startswith('#measurement'):
                    metadata = line.replace('#measurement,','')
                    metadata = metadata[1:-(len(metadata) - metadata.rindex('"'))]
                    metadata = metadata.replace('""','"')
                    metadata = json.loads(metadata)


    if metadata is not None:
        name = metadata['data']['name'].replace('_',' ').lower()
        name = ''.join([i for i in name if not i.isdigit()])
        name = name.strip()

        df = pd.read_csv(raw_file_path)
        df = df.iloc[3:]
        df.columns = df.iloc[0]
        df = df[1:]
        df.rename(columns={'#header:timestamp': 'timestamp'}, inplace=True)
        df.rename(columns={'3': 'index'}, inplace=True)
        #df.index.names = ['index']
        df['class'] = name
        df.reset_index(drop=True, inplace=True)
        df.index.names = ['index']
        return df

def create_dataset(args):

    print('Creating combined smell dataset')
    #combine all raw data into a single df
    raw_dfs = []
    raw_file_list = [y for x in os.walk(args.raw_dataset_path) for y in glob(os.path.join(x[0], '*.csv'))]
    print('\tFound', len(raw_file_list), 'raw smell data files')
    for raw_file_path in raw_file_list:
        raw_df = parse_raw_file(raw_file_path)
        if raw_df is not None:
            raw_dfs.append(raw_df)

    channel_map = [999, 999, 69, 69, 69, 41, 41, 41, 40, 40, 40, 33, 33, 33, 999, 999, 999, 999, 999, 999, 999, 61, 61,
                   61, 47, 47,
                   47, 43, 43, 43, 999, 999, 999, 999, 90, 90, 90, 67, 67, 67, 53, 53, 53, 42, 42, 42, 999, 999, 999,
                   999, 94, 94, 94,
                   89, 89, 89, 85, 85, 85, 59, 59, 59, 999, 999, 1, 1]

    df = pd.concat(raw_dfs, axis=0)

    print('Removing unused channels from the dataset.')
    #drop channels that are not active
    drop_list = []
    for idx, x in enumerate(channel_map):
        if x == 999:
            colname = df.columns[idx]
            drop_list.append(colname)

    for colname in drop_list:
        df = df.drop(columns=colname)

    print('Encoding class label names as numbers')
    #fix classes
    df['class_name'] = df['class'].astype('category')
    df['class'] = df['class_name'].cat.codes

    class_map = dict()

    for index, row in df.iterrows():
        class_name = row['class_name']
        class_id = row['class']
        if class_id not in class_map:
            class_map[class_id] = class_name

    df = df.drop('class_name', axis=1)

    print('Cleaning up indexes')
    #clean up indexes
    df = df.reset_index()
    df = df.drop(columns='index')
    df.index.name = 'index'

    print('\tDataset contains',len(class_map), 'classes with',len(df.index), 'input records')
    print('Saving smell dataset to:', args.output_dataset_path)
    df.to_csv(args.output_dataset_path)

    print('Saving smell dataset metadata to:', args.output_dataset_metadata_path)
    with open(args.output_dataset_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Smell Dataset Parser')

    # general args
    parser.add_argument('--project_name', type=str, default='smell_dataset_parser', help='name of project')
    parser.add_argument('--raw_dataset_path', type=str, default='raw_data', help='location of dataset')
    parser.add_argument('--output_dataset_path', type=str, default='smell_dataset.csv', help='location of dataset')
    parser.add_argument('--output_dataset_metadata_path', type=str, default='smell_dataset_metadata.json', help='location of dataset')

    args = parser.parse_args()

    create_dataset(args)