# Definitions to be used in this HCM_Project folder
import os

# Main directory in which everything is stored
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = '/mnt/host/c/Users/Changxin/Documents/datasets/HCM_DATA_Organized'
DATA_DIR_WIN = 'c:/Users/Changxin/Documents/datasets/HCM_DATA_Organized'
# Directory where original hv_dict are stored
hv_dict_path_original = os.path.join(DATA_DIR,'hv_dict_original_LGE')
hv_dict_path_original_win = os.path.join(DATA_DIR_WIN,'hv_dict_original_LGE')


# Directory where predicted hv_dict are stored
hv_dict_path_predicted = os.path.join(DATA_DIR,'hv_dict_predicted_LGE')
hv_dict_path_predicted_win = os.path.join(DATA_DIR_WIN,'hv_dict_predicted_LGE')

# Directory where standardized hv_dict are stored
hv_dict_path_standard = os.path.join(DATA_DIR,'hv_dict_standard_LGE')
hv_dict_path_standard_win = os.path.join(DATA_DIR_WIN,'hv_dict_standard_LGE')

# Directory where weights for segmentation DNN weights are stored
dnn_seg_weights_path = os.path.join(ROOT_DIR,'SegDNN')

# ROI Specific parameters
roi_img_size = 192
roi_minimum_area = 30
