import argparse
import json
from pathlib import Path

import pandas as pd


def convert_ehr_csv_to_dict(csv_dir_path, hv_dir_path, split_index):
    database = {}
    for file_path in csv_dir_path.iterdir():
        filename = file_path.name
        if 'split{}'.format(split_index) not in filename:
            continue
            
        data = pd.read_csv(csv_dir_path / filename, index_col=0, header=0)
        keys = []
        subsets = []
        labels = []
        for hv_file in hv_dir_path.iterdir():
            pickle_name = hv_file.name
            mri_id = int(pickle_name.split('_')[0])
            try:
                subset_index = data.loc[data["MRI_ID"]==mri_id]["subset"].values[0]
                label = data.loc[data["MRI_ID"]==mri_id]["Adverse_Outcome"].values[0]
    
                if subset_index == 0:
                    continue
                elif subset_index == 1:
                    subset = 'training'
                elif subset_index == 2:
                    subset = 'validation'

                keys.append(pickle_name.split('.')[0])
                subsets.append(subset)
                labels.append(label)
            except:
                continue

        # for i in range(data.shape[0]):
        #     row = data.iloc[i, :]
        #     if row[1] == 0:
        #         continue
        #     elif row[1] == 1:
        #         subset = 'training'
        #     elif row[1] == 2:
        #         subset = 'validation'

        #     keys.append(row[0].split('.')[0])
        #     subsets.append(subset)

        for i in range(len(keys)):
            key = keys[i]
            database[key] = {}
            database[key]['subset'] = subsets[i]
            database[key]['annotations'] = {'label': str(labels[i])}

    return database


def get_labels(csv_dir_path):
    labels = []
    for file_path in csv_dir_path.iterdir():
        labels.append('_'.join(file_path.name.split('_')[:-2]))
    return sorted(list(set(labels)))


def convert_ehr_csv_to_json(csv_dir_path, split_index, hv_dir_path,
                               dst_json_path):
    # labels = get_labels(csv_dir_path)
    database = convert_ehr_csv_to_dict(csv_dir_path, hv_dir_path, split_index)

    dst_data = {}
    dst_data['labels'] = ['0', '1']
    dst_data['database'] = {}
    dst_data['database'].update(database)

    # for k, v in dst_data['database'].items():
    #     if v['annotations'] is not None:
    #         label = v['annotations']['label']
    #     else:
    #         label = 'test'

    #     video_path = hv_dir_path / k
    #     n_frames = get_n_frames(video_path)
    #     v['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('EHR_path',
                        default=None,
                        type=Path,
                        help='Directory path of EHR files with outcomes.')
    parser.add_argument('hv_path',
                        default=None,
                        type=Path,
                        help=('Path of heart volume files (pickle).'))
    parser.add_argument('dst_dir_path',
                        default=None,
                        type=Path,
                        help='Directory path of dst json file.')

    argv = ["/mnt/host/c/Users/Changxin/Documents/datasets/HCM_DATA_Organized/HCM_EHR/", 
            "/mnt/host/c/Users/Changxin/Documents/datasets/HCM_DATA_Organized/hv_dict_standard_LGE/",
            "/mnt/host/c/Users/Changxin/Documents/datasets/HCM_DATA_Organized/"]
    args = parser.parse_args(argv)

    for split_index in range(1, 2):
        dst_json_path = args.dst_dir_path / 'HCM_{}.json'.format(split_index)
        convert_ehr_csv_to_json(args.EHR_path, split_index, args.hv_path,
                                dst_json_path)
