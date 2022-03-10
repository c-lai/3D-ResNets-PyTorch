"""
We need some description about the heart volume objects. Children are assumed to be stored apex to base.
"""

import heart_volume.constants as const
import heart_volume.heart_utils as utils
import numpy as np
import warnings


class HeartVolume:
    def __init__(
            self,
            unique_id='',
            name='',
            children=None,
            _pixel_array=None,
            _segmentation=None,
            _meshgrid=None,
            _rotation_angle=None,
    ):
        self.unique_id = unique_id
        self.name = name

        children_to_add = []
        if children is not None:
            for child in children:
                child_obj = HeartSlice(**child)
                child_obj.parent = self
                children_to_add.append(child_obj)

        self.children = children_to_add

        self._pixel_array = _pixel_array
        self._segmentation = _segmentation
        self._meshgrid = _meshgrid
        self._rotation_angle = _rotation_angle

        # self.validate_object()

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def to_stored_dict(self):
        stored_dict = {}
        for key, val in self:
            if key == 'children':
                stored_dict[key] = [x.to_stored_dict() for x in self.children]
            else:
                stored_dict[key] = val

        return stored_dict

    def __eq__(self, other):
        return (
                all([self.children[i] == other.children[i] for i in range(len(self.children))]) and
                self.unique_id == other.unique_id and self.name == other.name
        )

    def validate_object(self):
        # 1. is sorted
        locations_current = [x.slice_location for x in self.children]
        new_locations = sorted(locations_current)
        assert locations_current == new_locations, 'Incorrect sort order'

        # 2. are there duplicates?
        _, uniq_indices = np.unique([round(x / const.SLICE_SPACING_TOL) for x in locations_current], return_index=True)
        duplicates = set(range(len(locations_current))) - set(uniq_indices)

        assert not len(duplicates), 'There are duplicate slices in the volume'
        
        # 3. are there inconsistent pixel array sizes?
        assert len(set([x.pixel_array.shape for x in self.children])) == 1, 'Slice pixel_array shapes are inconsistent'

    def reset_top_level(self):
        """
        There are certain operations which will trigger a re-calculation of top level quantities.
        :return:
        """
        warnings.warn('Top-level heart volume information will be reset for %s' % self.unique_id)
        self._meshgrid = None
        self._segmentation = None
        self._pixel_array = None
        self._rotation_angle = None

    @property
    def pixel_array(self):
        """
        Returns a 3-D numpy array of each child's numpy array
        :return: np.array
        """
        if self._pixel_array is None:
            self._pixel_array = np.stack([child.pixel_array for child in self.children], axis=-1).astype(np.float32)

        return self._pixel_array

    @pixel_array.setter
    def pixel_array(self, pixel_array):
        self._pixel_array = pixel_array

    @property
    def segmentation(self):
        """
        Returns a 3-D numpy array of each child's numpy array
        :return: np.array
        """
        # if segmentation stored  is not set, extract from children
        if self._segmentation is None:
            children_segs = [child.segmentation for child in self.children]
            # if any child segmentations is missing, cannot return anything
            if all([x is not None for x in children_segs]):
                self._segmentation = np.stack(children_segs, axis=-1).astype(np.float32)

        return self._segmentation

    @segmentation.setter
    def segmentation(self, segmentation):
        self._segmentation = np.asarray(segmentation, dtype=np.float32)

    def _first_nonzero_slice_location(self):
        """
        Computes the slice location of the first non-zero apical slice
        :return: float
        """
        for z in range(len(self.children)):
            if self.segmentation is not None:
                if np.any(self.segmentation[:, :, z]):
                    return self.children[z].slice_location
        return np.min([z.slice_location for z in self.children])

    @property
    def meshgrid(self):
        """
        Returns a mesh grid corresponding to pixel array
        :return: np.array
        """
        if self._meshgrid is None:
            #  Calculate coordinates in the short-axis direction
            #  We assume that all slices have the same pixel spacing
            x_axis = np.arange(self.pixel_array.shape[0]) * self.children[0].pixel_spacing[0]
            x_axis = [x - (np.min(x_axis) + np.max(x_axis)) / 2 for x in x_axis]
            y_axis = np.arange(self.pixel_array.shape[1]) * self.children[0].pixel_spacing[1]
            y_axis = [y - (np.min(y_axis) + np.max(y_axis)) / 2 for y in y_axis]
            z_axis = [z.slice_location for z in self.children]
            z_axis = [z - self._first_nonzero_slice_location() for z in z_axis]  # start at 0 in z-direction

            _meshgrid = np.meshgrid(y_axis, x_axis, z_axis)  # meshgrid flips x and y (as per Matlab)!
            self._meshgrid = _meshgrid

        return self._meshgrid

    @meshgrid.setter
    def meshgrid(self, meshgrid):
        self.pixel_array = utils.matlab_interp3(self.meshgrid, self.pixel_array, meshgrid, 'mask')

        if self.segmentation is not None:
            # Perform interpolation on intensities not labels
            seg_lbl = self.segmentation
            seg_int = utils.convert_seg_label_to_intensity(seg_lbl)
            seg_int_interp = utils.matlab_interp3(self.meshgrid, seg_int, meshgrid, 'mask')
            self.segmentation = utils.convert_intensity_to_seg_label(seg_int_interp)

        self._meshgrid = meshgrid

    @property
    def rotation_angle(self):
        """
        Returns a rotation angle
        :return: float
        """
        if self._rotation_angle is None:
            child_angles = [x.rotation_angle for x in self.children]
            ref_angle = child_angles[0]

            if not np.all(np.isclose(child_angles, ref_angle)):
                warnings.warn('Inconsistent rotation angles between children for %s' % self.unique_id)

            self._rotation_angle = ref_angle

        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, angle_info):
        angle, strict = angle_info
        self.pixel_array, angle_adj = utils.rotate(self.pixel_array, self.rotation_angle - angle, strict, order=3)

        if self.segmentation is not None:
            # Perform interpolation on intensities not labels
            seg_lbl = self.segmentation
            seg_int = utils.convert_seg_label_to_intensity(seg_lbl)
            seg_int_rot, _ = utils.rotate(seg_int, self.rotation_angle - angle, strict, order=0)
            self.segmentation = utils.convert_intensity_to_seg_label(seg_int_rot)

        self._rotation_angle = angle_adj


class HeartSlice:
    def __init__(
            self,
            pixel_array,
            pixel_spacing,
            slice_location,
            parent,
            segmentation,
            slice_unique_id,
            rotation_angle,
            visualization_parameters,
    ):

        self.slice_location = slice_location
        self.pixel_spacing = pixel_spacing
        self.parent = parent
        self.slice_unique_id = slice_unique_id
        self.rotation_angle = rotation_angle
        self.visualization_parameters = visualization_parameters
        self._set_normalized_img(pixel_array, segmentation)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __eq__(self, other):
        return (
                self.parent == other.parent and
                self.pixel_array.all() == other.pixel_array.all()
        )

    def _set_normalized_img(self, pixel_array, segmentation):
        pixel_array = utils.crop_to_square(pixel_array)

        if segmentation is not None:
            segmentation = utils.crop_to_square(segmentation)

        self.pixel_array = pixel_array
        self.segmentation = segmentation

    def to_stored_dict(self):
        return {
            'pixel_array': self.pixel_array,
            'slice_location': self.slice_location,
            'pixel_spacing': self.pixel_spacing,
            'parent': self.parent.unique_id,
            'segmentation': self.segmentation,
            'slice_unique_id': self.slice_unique_id,
            'rotation_angle': self.rotation_angle,
            'visualization_parameters': self.visualization_parameters,
        }
