from functools import wraps
from warnings import warn
# Import helper functions shared with vox2track
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
import numpy as np

def _with_initialize(generator):
    """Allows one to write a generator with initialization code.

    All code up to the first yield is run as soon as the generator function is
    called and the first yield value is ignored.
    """
    @wraps(generator)
    def helper(*args, **kwargs):
        gen = generator(*args, **kwargs)
        next(gen)
        return gen

    return helper


@_with_initialize
def target(streamlines, affine, target_mask, include=True):
    """Filters streamlines based on whether or not they pass through an ROI.

    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    target_mask : array-like
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    include : bool, default True
        If True, streamlines passing through `target_mask` are kept. If False,
        the streamlines not passing through `target_mask` are kept.

    Returns
    -------
    streamlines : generator
        A sequence of streamlines that pass through `target_mask`.

    Raises
    ------
    ValueError
        When the points of the streamlines lie outside of the `target_mask`.

    See Also
    --------
    density_map
    """
    target_mask = np.array(target_mask, dtype=bool, copy=True)
    lin_T, offset = _mapping_to_voxel(affine)
    yield
    # End of initialization

    for l,sl in enumerate(streamlines):
        try:
            ind = _to_voxel_coordinates(sl, lin_T, offset)
            i, j, k = ind.T
            state = target_mask[i, j, k]
        except IndexError:
            raise ValueError("streamlines points are outside of target_mask")
        if state.any() == include:
            yield l,sl