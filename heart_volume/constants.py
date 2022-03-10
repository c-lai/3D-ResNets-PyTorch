"""
File with constants needed in heart volume
"""
import numpy as np

# GT-labels
GT_LABELS = {
    'nonroi': 0,
    'lvbp': 2 ** 0,
    'rvbp': 2 ** 1,
    'lvmyo': 2 ** 2,
    'rvmyo': 2 ** 3,
}

GT_INTENSITIES = {
    'nonroi': 0,
    'lvbp': np.uint8(np.clip((np.log2(GT_LABELS['lvbp']) + 1) * 64, 0, 255)),
    'rvbp': np.uint8(np.clip((np.log2(GT_LABELS['rvbp']) + 1) * 64, 0, 255)),
    'lvmyo': np.uint8(np.clip((np.log2(GT_LABELS['lvmyo']) + 1) * 64, 0, 255)),
    'rvmyo': np.uint8(np.clip((np.log2(GT_LABELS['rvmyo']) + 1) * 64, 0, 255)),
}


SEG_TYPE_LV = 'lv_segmentation'
SEG_TYPE_LVRV = 'lvrv_segmentation'

SLICE_SPACING_TOL = 1e-6

ROTATION_REFERENCE = 1.0e+02 * np.array([
    -0.004923968222400,
    0.006070312924000,
    -0.013529233073600,
])
