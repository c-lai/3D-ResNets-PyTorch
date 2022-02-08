from functional import compose, partial
from functools import reduce
from scipy.ndimage import rotate as scipy_rotate
import copy
import glob
import heart_volume.constants as const
import nibabel as nib
import numpy as np
import os
import pickle
import pydicom
import scipy.interpolate as interp
import scipy.stats as stats
import warnings
import medpy.io as mpy
from skimage.transform import rescale
import config


required_attributes_volume = ['children', 'segmentation_slices', 'pixel_intensity', 'patient_name']

required_attributes_slice = ['pixel_array', 'slice_position', 'length', 'width',
                             'pixel_spacing', 'parent']


def generate_unique_id(data_path):
    pkl_files = glob.glob(os.path.join(data_path, "*.pickle"))
    unique_id = 'P0000'

    if pkl_files:
        ids = []
        for pkl in pkl_files:
            if os.path.getsize(pkl):
                with open(pkl, 'rb') as pickle_file:
                    ids.append(int(pickle.load(pickle_file)['unique_id'].replace('P', '')))
        unique_id = 'P%04d' % (max(ids) + 1)

    return unique_id


def read_dcm_file(file_name: str):
    """
    Read a DICOM file
    :param file_name: string with the full path to the file
    :return: DICOM data set
    """
    data_set = pydicom.dcmread(file_name, force=True)
    # noinspection PyUnresolvedReferences
    # data_set.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    return data_set


def _nii_is_default(nii_header):
    return not nii_header['sform_code'] and not nii_header['qform_code']


def nii_to_hvdict(
    unique_id,
    vol_filename,
    sort_type,
    patient_name='',
    seg_filename='',
    seg_label_info=None
):
    """
    Given a nifti file name, build a heart volume dictionary. Assumes vol and seg
    are sorted in the same order and have the same shape. The volume needs to be sorted.
    It is assumed that header information is consistent across segmentation and image.
    For consistency with DICOM files, we assume the LPS+ coordinate system and store
    the transpose of the image in pixel_array. Slices with blank segmentations are discarded.
    'None' segmentations are kept.
    :param unique_id: generate a unique id by calling generate_unique_id()
    :param vol_filename: file path for volume
    :param sort_type: "apex_to_base" or "base_to_apex"
    :param patient_name: name of this patient
    :param seg_filename: optional segmentation file
    :param seg_label_info: segmentation encoding (e.g., 0 = background)
    :return: dict
    """
    nimg = nib.load(vol_filename)
    voxel_array = nimg.get_fdata()

    # Extract pixel spacing in x, y, z
    pixel_spacing = nimg.header['pixdim']
    dx = pixel_spacing[1]  # pixel spacing in the x-direction
    dy = pixel_spacing[2]  # pixel spacing in the y-direction

    transforms = []
    if _nii_is_default(nimg.header):
        #  Cannot rely on affine - assume LPS+ orientation
        dz = 10  # pixel spacing in the z-direction
    else:
        # If affine is available, first convert to to RAS+, then to LPS+
        dz = pixel_spacing[3]  # pixel spacing in the z-direction

        # First, convert to RAS+ using affine
        for i in [0, 1]:
            if np.sign(nimg.affine[i, i]) == -1:
                transforms.append(partial(np.flip, axis=i))

        # Now, convert from RAS+ to LPS+
        transforms.append(partial(np.flip, axis=(0, 1)))

    # Ensure images are sorted apex to base
    if sort_type == 'base_to_apex':
        transforms.append(partial(np.flip, axis=2))

    # DICOM stores transposed versions of the image
    transforms.append(partial(np.swapaxes, axis1=0, axis2=1))

    # def multi_compose(*functions):
    #     return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

    # noinspection PyTypeChecker
    final_transform_fn = partial(reduce, compose)(transforms[::-1])

    voxel_array = final_transform_fn(voxel_array)

    # Handle segmentation
    seg_voxel_array = None
    if seg_filename:
        seg_voxel_array = nib.load(seg_filename).get_fdata()
        seg_voxel_array = final_transform_fn(seg_voxel_array)
        # Make segmentation values consistent
        if seg_label_info:
            reference_voxel_array = seg_voxel_array.copy()
            for key, value in seg_label_info.items():
                seg_voxel_array[reference_voxel_array == value] = const.GT_LABELS[key]

    dict_list = []
    for i in range(voxel_array.shape[-1]):  # go through slice by slice
        pixel_array = voxel_array[:, :, i]
        seg = seg_voxel_array[:, :, i] if seg_voxel_array is not None else None

        # all or None: get rid of slices that have no segmentation
        if seg is not None and not np.any(seg):
            warnings.warn(
                'Discarding slice with no segmentation for HeartVolume %s: %s' %
                (unique_id, i)
            )
            continue

        dict_list.append({
            'pixel_array': pixel_array,
            'slice_location': i * dz,  # assume sorted
            'pixel_spacing': np.array([dx, dy]),
            'parent': unique_id,
            'segmentation': seg,
            'slice_unique_id': i,
            'rotation_angle': 0,
            'visualization_parameters': {
                'center': (np.max(pixel_array) + np.min(pixel_array)) / 2,
                'width': np.max(pixel_array) - np.min(pixel_array),
                'slope': 1,
                'intercept': 0
            }
        })

    patient_hv = {
        'unique_id': unique_id,
        'children': dict_list,
        'name': patient_name,
    }

    return patient_hv


def check_attributes(provided_dict, slices=True):
    if slices:
        attrs = required_attributes_slice
    else:
        attrs = required_attributes_volume

    return all([attrs[i] in provided_dict.keys() for i in range(len(provided_dict.keys()))])


def _compute_sa_normal(dcm):

    # Cannot get sa normal from single dcm without 'ImageOrientationPatient'
    if 'ImageOrientationPatient' in dcm:
        ref_slice_orientation = np.asarray(dcm.ImageOrientationPatient)

        # The values from the row (X) direction cosine of Image Orientation (Patient) (0020,0037).
        x = np.asarray(ref_slice_orientation[:3])
        # The values from the column (Y) direction cosine of Image Orientation (Patient) (0020,0037).
        y = np.asarray(ref_slice_orientation[3:])

        sa_direction = np.cross(x, y) / np.linalg.norm(np.cross(x, y))
    else:
        sa_direction = np.zeros((3,))

    return sa_direction


def pixel_to_real_coordinates(i, j, position, orientation, pixel_spacing):

    # 1. Compute affine matrix for slice
    # The 3 values of Image Position (Patient) (0020,0032). It is the location in mm from the origin of the RCS.
    s = np.asarray(position)

    # The values from the row (X) direction cosine of Image Orientation (Patient) (0020,0037).
    x = np.asarray(orientation[:3])

    # The values from the column (Y) direction cosine of Image Orientation (Patient) (0020,0037).
    y = np.asarray(orientation[3:])

    # Column pixel resolution of Pixel Spacing (0028,0030) in units of mm.
    di = pixel_spacing[1]

    # Row pixel resolution of Pixel Spacing (0028,0030) in units of mm.
    dj = pixel_spacing[0]

    # Coordinate transform matrix
    m = np.stack((x * di, y * dj, np.zeros(3), s), axis=1)
    m = np.concatenate((m, [[0, 0, 0, 1]]))

    # 2. Convert pixel coordinates to real coordinates: M.(i,j)
    pixel_coord = np.asarray([i, j, 0, 1], dtype=float)
    real_coord = np.dot(m, pixel_coord)

    return real_coord


def dcm_rotation_angle(dcm):
    # Reference vector
    u = const.ROTATION_REFERENCE
    angle = 0
    if 'ImageOrientationPatient' in dcm and 'PixelSpacing':
        # The values from the column (Y) direction cosine of Image Orientation (Patient) (0020,0037).
        y = np.asarray(dcm.ImageOrientationPatient)[3:]

        # Row pixel resolution of Pixel Spacing (0028,0030) in units of mm.
        dj = np.asarray(dcm.PixelSpacing)[0]

        # Unit vector from (0, 0) pixel of image to (0, 1) in RCS coordinates
        v = y * dj

        # Calculate angle between vectors
        angle = np.rad2deg(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))

        # Decide in which direction to perform the rotation
        if np.cross(v, u)[-1] < 0:
            angle *= -1

    return angle


def _dcm_extract_visualization_parameters(dcm_dataset, visualization_parameters_override=None):
    """
    Use window center/width and potentially other adjustments for pixel array.
    :param dcm_dataset: pydicom.dataset
    :param visualization_parameters_override: what to use for WindowCenter and WindowWith to adjust image
    :return: pixel array normalized according to dcm tags
    """
    raw_pixel_array = dcm_dataset.pixel_array

    default_params = {
        'center': (np.max(raw_pixel_array) + np.min(raw_pixel_array)) / 2,
        'width': np.max(raw_pixel_array) - np.min(raw_pixel_array),
        'slope': 1,
        'intercept': 0
    }

    stored_params = {}
    if 'WindowCenter' in dcm_dataset and dcm_dataset.WindowCenter != '':
        stored_params['center'] = float(dcm_dataset.WindowCenter)

    if 'WindowWidth' in dcm_dataset and dcm_dataset.WindowWidth != '':
        stored_params['width'] = float(dcm_dataset.WindowWidth)

    if 'RescaleSlope' in dcm_dataset and dcm_dataset.RescaleSlope != '':
        stored_params['slope'] = float(dcm_dataset.RescaleSlope)

    if 'RescaleIntercept' in dcm_dataset and dcm_dataset.RescaleIntercept != '':
        stored_params['intercept'] = float(dcm_dataset.RescaleIntercept)

    if visualization_parameters_override is None:
        visualization_parameters_override = {}

    final_params = {**default_params, **stored_params, **visualization_parameters_override}

    return final_params


def dcm_to_hvdict(
    unique_id,
    slice_paths,
    patient_name='',
    mri_type = '',
    segmentation_info=None,
    seg_label_info=None,
    visualization_parameters_override=None
):
    """
    Given a list of dcm file names, creates a dehydrated version of an hv object as a dict.
    Slices with blank segmentations are discarded. 'None' segmentations are kept.
    :param unique_id: generate a unique id by calling generate_unique_id()
    :param slice_paths: list (or similar) of file paths for each slice
    :param patient_name: name of this patient
    :param segmentation_info: dictionary with segmentation keyed by slice path
    :param seg_label_info: segmentation encoding (e.g., 0 = background)
    :param visualization_parameters_override: center and width of intensity values
    :return: dict
    """
    # Arbitrarily assume the first slice is the z-direction reference point
    ref_dcm = read_dcm_file(slice_paths[0])
    sa_direction = _compute_sa_normal(ref_dcm)

    # Calculate angle relative to reference point. See const.ROTATION_REFERENCE
    rotation_angle = dcm_rotation_angle(ref_dcm)

    dict_list = []
    frame_number = []
    for (i, fname) in enumerate(slice_paths):
        dcm = read_dcm_file(fname)

        # all or None: get rid of slices that have no segmentation
        this_seg = None
        if segmentation_info:
            if fname in segmentation_info:
                this_seg = segmentation_info[fname]
                if seg_label_info:
                    for key, value in seg_label_info.items():
                        this_seg[this_seg == value] = const.GT_LABELS[key]
            else:
                warnings.warn(
                    'Discarding slice with no segmentation for HeartVolume %s: %s' %
                    (unique_id, fname)
                )
                continue

        visualization_parameters = _dcm_extract_visualization_parameters(dcm, visualization_parameters_override)
        # Define slice midpoint in pixel coordinates
        pixel_array = dcm.pixel_array
        pixel_array = crop_to_square(pixel_array, config.roi_img_size)
        
        i_0 = pixel_array.shape[1] // 2
        j_0 = pixel_array.shape[0] // 2

        pixel_spacing = np.asarray(dcm.PixelSpacing)

        if 'ImageOrientationPatient' in dcm and 'ImagePositionPatient' in dcm:
            slice_location = np.dot(sa_direction, pixel_to_real_coordinates(
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

        frame_number.append(dcm.InstanceNumber)

        dict_list.append({
            'pixel_array': pixel_array,
            'slice_location': slice_location,
            'pixel_spacing': pixel_spacing,
            'parent': unique_id,
            'segmentation': this_seg,
            'slice_unique_id': dcm.InstanceNumber,
            'rotation_angle': rotation_angle,
            'visualization_parameters': visualization_parameters,
        })

    if mri_type == 'cine':
        frame_array = np.asarray(frame_number)
        sorted_dict_list = list(np.array(dict_list)[np.argsort(frame_array)])
    else:
        _, uniq_indices = np.unique(
            [round(x['slice_location'] / const.SLICE_SPACING_TOL) for x in dict_list],
            return_index=True
        )
        sorted_dict_list = [dict_list[i] for i in uniq_indices]
        # print(len(sorted_dict_list))

        duplicates = set(range(len(dict_list))) - set(uniq_indices)
        if len(duplicates):
            warnings.warn(
                'Discarding non-unique slice(s) in HeartVolume %s: %s' %
                (unique_id, [dict_list[z]['slice_unique_id'] for z in duplicates])
                )

    patient_hv = {
        'unique_id': unique_id,
        'children': sorted_dict_list,
        'name': patient_name,
    }

    return patient_hv


def align_and_scale_seg(img_fname, seg_fname):
    """
    this is specific to mha files. it aligns an image with its segmentation from 3D volumes that contain offset
    and spacing information
    """
    img_image, img_header = mpy.load(img_fname)
    seg_image, seg_header = mpy.load(seg_fname)

    # get z coords to match up
    seg_z_offset = seg_header.offset[-1]
    seg_z_spacing = seg_header.spacing[-1]
    img_z_offset = img_header.offset[-1]
    img_z_spacing = img_header.spacing[-1]

    img_z_locs = [(img_z_offset + img_z_spacing * i) for i in range(img_image.shape[-1])]
    seg_z_locs = [(seg_z_offset + seg_z_spacing * i) for i in range(seg_image.shape[-1])]

    matching_seg_locs = [min(seg_z_locs, key=lambda x: abs(x - sl)) for sl in img_z_locs]
    inds = [seg_z_locs.index(loc) for loc in matching_seg_locs]

    # only keep z coords that matter
    impt_segs = seg_image[:, :, inds]

    # adjust x and y
    x_shift = int((img_header.offset[1] - seg_header.offset[1]) / seg_header.spacing[1])
    y_shift = int((img_header.offset[0] - seg_header.offset[0]) / seg_header.spacing[0])
    new_addition_x = np.zeros((impt_segs.shape[0], np.abs(x_shift), impt_segs.shape[2]))
    new_addition_y = np.zeros((np.abs(y_shift), impt_segs.shape[1] + np.abs(x_shift), impt_segs.shape[2]))
    if x_shift < 0:
        impt_segs = np.concatenate((new_addition_x, impt_segs), axis=1)
    elif x_shift > 0:
        impt_segs = np.concatenate((impt_segs, new_addition_x), axis=1)
    if y_shift > 0:
        impt_segs = np.concatenate((new_addition_y, impt_segs), axis=0)
    elif y_shift < 0:
        impt_segs = np.concatenate((impt_segs, new_addition_y), axis=0)

    segs_rescaled = rescale(impt_segs,
                            (img_image.shape[0] / impt_segs.shape[0], img_image.shape[1] / impt_segs.shape[1], 1),
                            anti_aliasing=False)
    segs_rescaled = np.where(segs_rescaled > 32, 255, 0)

    return segs_rescaled


def mha_to_hvdict(
    unique_id,
    vol_filename,
    sort_type=None,
    patient_name='',
    seg_filename='',
    seg_label_info=None
):
    """
    ***Note: this doesn't take into account rotations. It is an oversimplified version of what it should be currently***
    Given a mha file name, build a heart volume dictionary. Assumes vol and seg
    are sorted in the same order and have the same shape. The volume is assumed to be sorted.
    It is assumed that header information is consistent across segmentation and image.
    # :param unique_id: generate a unique id by calling generate_unique_id()
    # :param vol_filename: file path for volume
    # :param sort_type: "apex_to_base" or "base_to_apex"
    # :param patient_name: name of this patient
    # :param seg_filename: optional segmentation file
    # :param seg_label_info: segmentation encoding (e.g., 0 = background)
    :return: dict
    """
    voxel_array, img_header = mpy.load(vol_filename)
    dy, dx, dz = img_header.spacing
    seg_voxel_array = None
    if seg_filename:
        seg_voxel_array = align_and_scale_seg(vol_filename, seg_filename)

        if seg_label_info:
            for key, value in seg_label_info.items():
                seg_voxel_array[seg_voxel_array == value] = const.GT_LABELS[key]

    print('seg size: ', seg_voxel_array.shape)
    print('img size: ', voxel_array.shape)
    dict_list = []
    for i in range(voxel_array.shape[-1]):  # go through slice by slice
        pixel_array = voxel_array[:, :, i]
        seg = seg_voxel_array[:, :, i] if seg_voxel_array is not None else None

        # all or None: get rid of slices that have no segmentation
        if seg is not None and not np.any(seg):
            warnings.warn(
                'Discarding slice with no segmentation for HeartVolume %s: %s' %
                (unique_id, i)
            )
            continue

        dict_list.append({
            'pixel_array': pixel_array,
            'slice_location': i * dz,  # assume sorted
            'pixel_spacing': np.array([dx, dy]),
            'parent': unique_id,
            'segmentation': seg,
            'slice_unique_id': i,
            'rotation_angle': 0,
            'visualization_parameters': {
                'center': (np.max(pixel_array) + np.min(pixel_array)) / 2,
                'width': np.max(pixel_array) - np.min(pixel_array),
                'slope': 1,
                'intercept': 0
            }
        })

    patient_hv = {
        'unique_id': unique_id,
        'children': dict_list,
        'name': patient_name,
    }

    return patient_hv


def __np_to_mat(np_arr: np.ndarray):
    import matlab
    # noinspection PyUnresolvedReferences
    return matlab.double(np_arr.tolist())


def matlab_interp3(old_grid, values, new_grid, method):
    """
    Wrapper around Matlab's interp3 function
    :param old_grid: old meshgrid
    :param values: values to interpolate
    :param new_grid: new meshgrid
    :param method: either 'variational_implicit' or 'spline'
    :return: interpolated values as np.ndarray
    """
    import matlab.engine as mat

    x, y, z = old_grid
    xq, yq, zq = new_grid

    eng = mat.start_matlab()

    try:
        if method == 'image':
            values_interpolated = eng.interp3(
                __np_to_mat(x), __np_to_mat(y), __np_to_mat(z),
                __np_to_mat(values),
                __np_to_mat(xq), __np_to_mat(yq), __np_to_mat(zq),
                'spline',  # interpolation method
                0  # extrapolation values
            )

        elif method == 'mask':
            #  using nearest neighbor
            values_interpolated = eng.interp3(
                __np_to_mat(x), __np_to_mat(y), __np_to_mat(z),
                __np_to_mat(values),
                __np_to_mat(xq), __np_to_mat(yq), __np_to_mat(zq),
                'nearest',  # interpolation method
                0  # extrapolation values
            )
        else:
            raise ValueError('Unrecognized interpolation method: %s' % method)
    finally:
        eng.quit()

    rv = np.asarray(values_interpolated, dtype=np.float32)

    return rv


def modify_meshgrid(curr_meshgrid, pixel_resolution, spatial_resolution):
    """
    Construct a meshgrid using pixel_resolution and spatial_resolution. Current meshgrid will be used
    to infer missing resolution specifications.
    :param curr_meshgrid: numpy meshgrid used for reference
    :param pixel_resolution: truple with requested resolution. None means use curr_meshgrid for axis
    :param spatial_resolution: truple with requestedspatial  resolution. None means use curr_meshgrid for axis
    :return: np.meshgrid
    """

    assert len(pixel_resolution) == 3, 'Pixel resolution must have length 3'
    assert len(spatial_resolution) == 3, 'Spatial resolution must have length 3'

    axes = []
    for i in range(3):
        pxl_res = pixel_resolution[i]
        spt_res = spatial_resolution[i]

        if pxl_res is None and spt_res is None:
            #  preserving the current axis
            axis = np.swapaxes(curr_meshgrid[i], i - 1, i)[0, :, 0]
        elif pxl_res is None and spt_res is not None:
            curr_axis = np.swapaxes(curr_meshgrid[i], i - 1, i)[0, :, 0]
            axis = np.arange(len(curr_axis)) * spt_res
        elif pxl_res is not None and spt_res is None:
            curr_axis = np.swapaxes(curr_meshgrid[i], i - 1, i)[0, :, 0]
            axis = np.linspace(min(curr_axis), max(curr_axis), num=pxl_res, retstep=False)
        else:
            axis = np.arange(pixel_resolution[i]) * spatial_resolution[i]

        if i == 2:
            axis = [x - np.min(axis) for x in axis]  # start at 0 in z
        else:
            axis = [x - (np.min(axis) + np.max(axis)) / 2 for x in axis]  # keep things centered in x-y
        axes.append(axis)

    meshgrid_new = np.meshgrid(*axes)

    return meshgrid_new


def normalize_intensity(img_np, max_skew=2, max_kurtosis=2):
    """
    Adjust dynamic range with limits on the skew and kurtosis
    :param img_np:
    :param max_skew: maximum sample skewness of remaining intensities
    :param max_kurtosis: maximum sample kurtosis of remaining intensities
    :return: changed image
    """
    skew = stats.skew(img_np.ravel())
    kurtosis = stats.kurtosis(img_np.ravel())

    if skew > max_skew or kurtosis > max_kurtosis:
        return np.clip(img_np.copy(), 0, np.percentile(img_np.ravel(), 99))
    else:
        return img_np


def rotate(vol, angle, strict, order):
    """
    Takes numpy array img and converts it to a square by trimming
    :param vol: np.array representing image
    :param angle: angle by which to rotate
    :param strict: whether rotation should be increments of 90 deg (strict=0)
    :param order: order of interpolation
    :return: np.array
    """

    rot_imgs = []
    if strict:
        angle_adj = 0
        for z in range(vol.shape[-1]):
            img = vol[:, :, z]
            img_rot = scipy_rotate(img, angle, reshape=False, order=order)
            img_rot = np.clip(img_rot, np.min(img), np.max(img))
            rot_imgs.append(img_rot)
    else:
        no_rotations = [2, 3, 0, 1, 2]
        allowable_rotations = np.linspace(-180, 180, 5)
        k = interp.interp1d(allowable_rotations, no_rotations, kind='nearest')(angle)
        angle_adj = angle - (int(k) * 90) % 90
        for z in range(vol.shape[-1]):
            img = vol[:, :, z]
            img_rot = np.rot90(img, k=int(k))
            rot_imgs.append(img_rot)

    vol_rot = np.stack(rot_imgs, axis=-1)

    return vol_rot, angle_adj


def crop_to_square(img, target_size=None):
    """
    Takes numpy array img and converts it to a square by trimming
    :param img: np.array representing image
    :param target_size: optionally specify target size. If None, will return min(l, w) x min(l, w)
    :return: np.array
    """
    l, w = img.shape
    img_copy = img.copy()

    if l > w:
        delta = l - w
        cropped_img = img_copy[delta // 2: -delta + delta // 2, :]
    elif l < w:
        delta = w - l
        cropped_img = img_copy[:, delta // 2: -delta + delta // 2]
    else:
        cropped_img = img_copy

    if target_size:
        current_size = cropped_img.shape[0]  # should be a square

        center = max(target_size, current_size) // 2
        offset_min = center - min(target_size, current_size) // 2
        offset_max = offset_min + min(target_size, current_size)

        if target_size > current_size:
            new_image = np.zeros((target_size, target_size))
            new_image[offset_min:offset_max, offset_min:offset_max] = cropped_img
            cropped_img = new_image.copy()
        else:
            cropped_img = cropped_img[offset_min:offset_max, offset_min:offset_max]

    return np.asarray(cropped_img, dtype=np.float32)


def convert_seg_label_to_intensity(seg_lb):
    seg_i = seg_lb.copy()

    for (lb, lb_val) in const.GT_LABELS.items():
        seg_i[seg_i == lb_val] = const.GT_INTENSITIES[lb]

    return seg_i


def convert_intensity_to_seg_label(seg_i):
    seg_lb = seg_i.copy()

    for (lb, lb_val) in const.GT_INTENSITIES.items():
        seg_lb[seg_lb == lb_val] = const.GT_LABELS[lb]

    return seg_lb


def pixel_array_for_visualization(hv_obj):
    hv_obj = copy.deepcopy(hv_obj)
    pixel_array = hv_obj.pixel_array.copy()

    for (z, child) in enumerate(hv_obj.children):
        slice_pixel_array = pixel_array[:, :, z].copy()
        slice_pixel_array *= child.visualization_parameters['slope']
        slice_pixel_array += child.visualization_parameters['intercept']

        min_val = child.visualization_parameters['center'] - child.visualization_parameters['width'] / 2
        max_val = child.visualization_parameters['center'] + child.visualization_parameters['width'] / 2
        pixel_array[:, :, z] = np.clip(slice_pixel_array, min_val, max_val)

    hv_obj.pixel_array = pixel_array

    return hv_obj


def sort_order(hv_obj):
    return np.array([c.slice_unique_id for c in hv_obj.children])
