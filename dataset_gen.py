import argparse
import json

import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split

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
    name = None
    df = None

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

    return df, name

def create_dataset(args):

    print('Creating combined smell dataset')
    #combine all raw data into a single df
    raw_dfs = []

    raw_file_list = [y for x in os.walk(args.raw_dataset_path) for y in glob(os.path.join(x[0], '*.csv'))]
    print('\tFound', len(raw_file_list), 'raw smell data files')
    for raw_file_path in raw_file_list:
        raw_df, class_name = parse_raw_file(raw_file_path)

        if raw_df is not None:
            raw_dfs.append(raw_df)

    #raw_dfs_keys = [item for item in range(0, len(raw_dfs))]
    #Add 0 to begining to include timestamp + add 0 to end to include class
    channel_map = [999, 999, 999, 69, 69, 69, 41, 41, 41, 40, 40, 40, 33, 33, 33, 999, 999, 999, 999, 999, 999, 999, 61, 61,
                   61, 47, 47,
                   47, 43, 43, 43, 999, 999, 999, 999, 90, 90, 90, 67, 67, 67, 53, 53, 53, 42, 42, 42, 999, 999, 999,
                   999, 94, 94, 94,
                   89, 89, 89, 85, 85, 85, 59, 59, 59, 999, 999, 1, 1, 0]

    df = pd.concat(raw_dfs, axis=0)

    print('Removing unused channels from the dataset.')
    #drop channels that are not active
    drop_list = []
    for idx, x in enumerate(channel_map):
        if x == 999:
            colname = df.columns[idx]
            print(colname)
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
    df.rename(columns={'index': 'timepoints'}, inplace=True)

    #more index cleanup
    df = df.drop(columns='timepoints')
    df.index.name = 'index'

    print('\tDataset contains', len(class_map), 'classes with', len(df.index), 'input records')

    print('Saving smell dataset to:', args.output_dataset_path)
    df.to_csv(args.output_dataset_path)

    print('Saving smell dataset metadata to:', args.output_dataset_metadata_path)
    with open(args.output_dataset_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=4)

    class_list = dict()

    demo_df = pd.DataFrame(columns=df.columns)
    demo_df.index.name = 'index'

    # create demo dataset
    for i, row in df.iterrows():
        if row['class'] not in class_list:
            class_list[row['class']] = 0
        class_list[row['class']] += 1
        if class_list[row['class']] <= args.demo_max_records:
            demo_df.loc[len(demo_df)] = row

    demo_df.to_csv(args.output_demo_dataset_path)


def process_timeseries_train_test(args, raw_dfs, min_sample_size, train=True):

    new_raw_dfs = []

    mindex = 0
    for df in raw_dfs:

        df['mindex'] = mindex
        mindex += 1

        n = len(df) - min_sample_size
        if n > 0:
            df = df.drop(index=df.index[:n])
            df = df.reset_index()

        new_raw_dfs.append(df)

    channel_map = [999, 999, 69, 69, 69, 41, 41, 41, 40, 40, 40, 33, 33, 33, 999, 999, 999, 999, 999, 999, 999, 61, 61,
                   61, 47, 47,
                   47, 43, 43, 43, 999, 999, 999, 999, 90, 90, 90, 67, 67, 67, 53, 53, 53, 42, 42, 42, 999, 999, 999,
                   999, 94, 94, 94,
                   89, 89, 89, 85, 85, 85, 59, 59, 59, 999, 999, 1, 1]

    df = pd.concat(new_raw_dfs, axis=0)

    print('Removing unused channels from the dataset.')
    # drop channels that are not active
    drop_list = []
    for idx, x in enumerate(channel_map):
        if x == 999:
            colname = df.columns[idx]
            drop_list.append(colname)

    for colname in drop_list:
        df = df.drop(columns=colname)

    print('Encoding class label names as numbers')
    # fix classes
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
    # clean up indexes
    df = df.reset_index()
    df.rename(columns={'index': 'timepoints'}, inplace=True)
    df.rename(columns={'class': 'class_val'}, inplace=True)

    file_type = 'train'
    save_path = args.output_timeseries_train_dataset_path
    if train is False:
        file_type = 'test'
        save_path = args.output_timeseries_test_dataset_path


    print('\t',file_type,'dataset contains', len(class_map), 'classes with', len(df.index), 'input records')

    print('Saving smell timeseries dataset to:', save_path)

    column_save_list = ['timepoints', 'class_val', 'mindex']
    new_column_map = dict()
    df_columns = list(df.columns.values)
    remap_count = 0
    for column_name in df_columns:
        if column_name not in column_save_list:
            new_column_map[column_name] = 'dim_' + str(remap_count)
            remap_count += 1

    df = df.rename(columns=new_column_map)
    # df = df.rename(columns={'':'timepoints'})

    df.index.name = 'index'

    first_column = df.pop('mindex')
    df.insert(0, 'mindex', first_column)

    df.rename(columns={'mindex': ''}, inplace=True)

    df.to_csv(save_path, index=False)

    print('Saving timeseries smell dataset metadata to:', args.output_timeseries_dataset_metadata_path)
    with open(args.output_timeseries_dataset_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, ensure_ascii=False, indent=4)


def create_timeseries_dataset(args):

    print('Creating combined timeseries smell dataset')
    #combine all raw data into a single df
    raw_dfs = []
    X_index = []
    X_index_count = 0
    y = []

    name_map = dict()

    raw_file_list = [y for x in os.walk(args.raw_dataset_path) for y in glob(os.path.join(x[0], '*.csv'))]
    print('\tFound', len(raw_file_list), 'raw smell data files')
    for raw_file_path in raw_file_list:
        raw_df, class_name = parse_raw_file(raw_file_path)
        if raw_df is not None:
            raw_dfs.append(raw_df)
            X_index.append(X_index_count)
            y.append(class_name)
            X_index_count += 1
            if class_name in name_map:
                name_map[class_name] += 1
            else:
                name_map[class_name] = 1

    for class_name, dataset_count in name_map.items():

        if dataset_count % 2:
            print('Removing one dataset from class [', class_name, '] found [', dataset_count,'] classes, sets must be even.')
            drop_index = y.index(class_name)
            raw_dfs.pop(drop_index)
            X_index.pop(drop_index)
            y.pop(drop_index)


    X_index = pd.DataFrame(X_index, columns=['X_index'])

    train_X, test_X, train_y, test_y = train_test_split(X_index, y, test_size=args.test_size, stratify=y,
                                                        random_state=args.random_state)

    sample_sizes = []
    for df in raw_dfs:
        sample_sizes.append(len(df))

    min_sample_size = min(sample_sizes)


    raw_dfs_train = []
    raw_dfs_test = []

    for index, row in train_X.iterrows():
        raw_dfs_train.append(raw_dfs[index])

    for index, row in test_X.iterrows():
        raw_dfs_test.append(raw_dfs[index])

    process_timeseries_train_test(args, raw_dfs_train, min_sample_size, train=True)
    process_timeseries_train_test(args, raw_dfs_test, min_sample_size, train=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Smell Dataset Parser')

    # general args
    parser.add_argument('--project_name', type=str, default='smell_dataset_parser', help='name of project')
    parser.add_argument('--raw_dataset_path', type=str, default='raw_data/fw_3_0_1', help='location of dataset')
    parser.add_argument('--output_dataset_path', type=str, default='smell_dataset.csv', help='location of dataset')
    parser.add_argument('--output_demo_dataset_path', type=str, default='demo_smell_dataset.csv', help='location of dataset')
    parser.add_argument('--demo_max_records', type=int, default=15,help='location of dataset')
    parser.add_argument('--output_dataset_metadata_path', type=str, default='smell_dataset_metadata.json', help='location of dataset')
    parser.add_argument('--output_timeseries_train_dataset_path', type=str, default='smell_timeseries_train_dataset.csv', help='location of dataset')
    parser.add_argument('--output_timeseries_test_dataset_path', type=str, default='smell_timeseries_test_dataset.csv',
                        help='location of dataset')
    parser.add_argument('--output_timeseries_dataset_metadata_path', type=str,
                        default='smell_timeseries_dataset_test_metadata.json',
                        help='location of dataset')

    parser.add_argument('--test_size', type=float, default=0.5, help='size of the testing split')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Pass an int for reproducible output across multiple function calls')


    args = parser.parse_args()

    create_dataset(args)
    create_timeseries_dataset(args)