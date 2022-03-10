from config import ROOT_DIR, hv_dict_path_original, hv_dict_path_predicted
import glob
import heart_utils as hu
import numpy as np
import os
import pickle
import pandas as pd

def count():
    base_path = os.path.join(ROOT_DIR, 'DATA','LGE_Images')
    c = 0
    # Iterate through each patient
    for folder in sorted(glob.glob(os.path.join(base_path,'*/'))):
        dcm_file_names = os.listdir(folder)
        if len(dcm_file_names) == 0:
            c = c + 1
            continue
    return c

def beepul_hcm2hvdict(write=False):
    # Establish base path
    base_path = os.path.join(ROOT_DIR, 'DATA','LGE_Images')

    # Load csv table to identify SAX 3D + t cine images
    # Table = pd.read_csv(os.path.join(ROOT_DIR,'SAX_ID_and_Image_Names_Updated.csv'))

    # Check to see if we have a 'hv_dict_original' folder
    if not os.path.isdir(hv_dict_path_original):
        os.mkdir(hv_dict_path_original)

    # Iterate through each patient
    for folder in sorted(glob.glob(os.path.join(base_path,'*/'))):
        dcm_file_names = os.listdir(folder)
        if len(dcm_file_names) == 0:
            continue

        dcm_files = np.array([os.path.join(folder,file) for file in dcm_file_names])

        # Get patient name from the first dicom image
        pt_name = hu.read_dcm_file(dcm_files[0]).PatientName

        # Arbitrarily assume the first slice is the z-direction reference point
        ref_dcm = hu.read_dcm_file(dcm_files[0])
        sa_direction = hu._compute_sa_normal(ref_dcm)

        # Read the z locations and overall data for every dcm file in the dcm_files list
        dcm_data = np.array([hu.read_dcm_file(dcm_files[i]) for i in range(len(dcm_files))])

        slice_locs = []
        for i in range(len(dcm_data)):
            dcm = dcm_data[i]
            pixel_array = dcm.pixel_array
            i_0 = pixel_array.shape[1] // 2
            j_0 = pixel_array.shape[0] // 2
            pixel_spacing = np.asarray(dcm.PixelSpacing)

            if 'ImageOrientationPatient' in dcm and 'ImagePositionPatient' in dcm:
                slice_location = np.dot(sa_direction, hu.pixel_to_real_coordinates(
                    i_0,
                    j_0,
                    np.asarray(dcm.ImagePositionPatient),
                    np.asarray(dcm.ImageOrientationPatient),
                    pixel_spacing
                )[:-1])
            elif 'ImagePositionPatient' in dcm:
                # Assume that the SA direction is in the direction of increasing z
                this_pos = np.asarray(dcm.ImagePositionPatient)
                ref_pos = np.asarray(ref_dcm.ImagePositionPatient)
                slice_location = np.sign(this_pos[2] - ref_pos[2]) * np.linalg.norm(this_pos - ref_pos)
            else:
                raise ValueError('Cannot find necessary tags to compute slice_location')

            slice_locs.append(slice_location)

        # Separate images into 2D + t image 'batches'
        sorted_locs = sorted(np.unique(slice_locs))
        sorted_locs = sorted_locs[int(np.round(len(sorted_locs) * .25)):int((len(sorted_locs) - np.round(len(sorted_locs) * 0.25)))]
        batches = []
        for loc in sorted_locs:
            dcm_files_at_loc = dcm_files[slice_locs == loc]
            dcm_data_at_loc = dcm_data[slice_locs == loc]
            instances_at_loc = np.array([dcm_data_at_loc[i].InstanceNumber for i in range(len(dcm_data_at_loc))])
            sorted_files_at_loc = dcm_files_at_loc[np.argsort(instances_at_loc)]
            batches.append(sorted_files_at_loc)

        # Make the patient folder in hv_dict_original
        if not os.path.isdir(os.path.join(hv_dict_path_original,str(pt_name))):
            os.mkdir(os.path.join(hv_dict_path_original, str(pt_name)))

        # For each 2D+t image turn that image into a hv_dict
        for dcm_batch in batches:
            # turn into hv_obj
            unique_id = hu.generate_unique_id(os.path.join(hv_dict_path_original,
                                                           str(pt_name)))
            hvdict = hu.dcm_to_hvdict(
                unique_id,
                dcm_batch,
                patient_name=str(pt_name),
                mri_type='cine',
                segmentation_info=None,
                seg_label_info=None,
            )
            if write:
                file_name = os.path.join(hv_dict_path_original,str(pt_name),
                                         str(pt_name) + '_' + unique_id + '.pickle')
                with open(file_name, 'wb') as pickle_file:
                    pickle.dump(hvdict, pickle_file)
    return None

def jhh_hcm_data2hv_2(write = False):
    base_path = '/home/beepul/avaeseg-master/data/Clinician_Data/no_hv_dict_created'
    for pt_folder in sorted(glob.glob(os.path.join(base_path,'*/'))):
        try:
            dcm_list = glob.glob(os.path.join(pt_folder, 'MRI_0', '*.dcm'))
            first_dcm = hu.read_dcm_file(dcm_list[0])
            pt_name = first_dcm.PatientID
            unique_id = hu.generate_unique_id('/home/beepul/avaeseg-master/data/hv_dict_original_2')
            hvdict = hu.dcm_to_hvdict(
                unique_id,
                dcm_list,
                patient_name=pt_name,
                mri_type='lge',
                segmentation_info=None,
                seg_label_info=None,
            )
            if write:
                file_name = os.path.join('/home/beepul/avaeseg-master/data/hv_dict_predicted_2', unique_id + '.pickle')
                with open(file_name, 'wb') as pickle_file:
                    pickle.dump(hvdict, pickle_file)
        except:
            continue


if __name__ == '__main__':
    # print(count())
    beepul_hcm2hvdict(write=True)
    # jhh_hcm_data2hv_2(write=False)
