
import multiprocessing
from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent.absolute()
sys.path.append(str(PROJECT_DIR))
from src.ConnectivityAnalysis import ConnectivityAnalysis

def lins(l,s):
    for ll in l:
        if ll in s:
            return True
    return False

def add_parser():
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument('--dout', type=str, required=True,
                        help="Output directory")
    parser.add_argument('--fsrc_mask', type=str, required=True,
                        help="Source mask file path")
    parser.add_argument('--fbrainmask', type=str, required=True,
                        help="Brain mask file path")
    parser.add_argument('--ftrg_seg', type=str, required=True,
                        help="Target segmentation file path")

    # Complemetary arguments (either A or B)
    parser.add_argument('--fdimg', type=str, default=None,
                        help="Input diffusion image file path")
    parser.add_argument('--fbvec', type=str, default=None,
                        help="b-vector file path")
    parser.add_argument('--fbval', type=str, default=None,
                        help="b-value file path")

    parser.add_argument('--fFOD', type=str, default=None,
                        help="FOD image file path")

    # Optional arguments
    parser.add_argument('--exp_mode', action='store_true',
                        help="Experiment mode")
    parser.add_argument('--tr_select', type=int, default=100000,
                        help="number of streamlines")
    parser.add_argument('--tr_alg', type=str, default="iFOD2",
                        choices=["iFOD2","iFOD1","SD_STREAM","Tensor_Det","Tensor_Prob"],
                        help="tracing algorithm")
    parser.add_argument('--normalize_by_area', type=int, default=1,
                        help="whether normalize counts by area of target region")
    parser.add_argument('--bvec_flip', type=str, default=None,
                        help="Flip b-vector in x,y,z direction", choices=["x","y","z"])
    parser.add_argument('--down_res', type=float, default=None,
                        help="the target of downsampled resolution. This option only works when input is raw diffusion image")
    parser.add_argument('--tr_dilate_src', type=int, default=0,
                        help="Dilate source mask? If so, by how many voxels")
    parser.add_argument('--tr_dilate_trg', type=int, default=0,
                        help="Dilate target mask? If so, by how many voxels")

    return parser

from easydict import  EasyDict
from pathlib import Path

if __name__ == "__main__":
    parser = add_parser()
    args = parser.parse_args()

    # validate input arguments
    if args.fdimg is None and args.fFOD is None:
        raise ValueError("Either --fdimg or --fFOD should be specified")
    # check if all the required file exist
    for fpath in [args.fsrc_mask, args.fbrainmask, args.ftrg_seg, args.fdimg, args.fbvec, args.fbval, args.fFOD]:
        if fpath is not None:
            if not Path(fpath).exists():
                raise ValueError(f"File path {fpath} does not exist")
    # run
    # try:
    ca = ConnectivityAnalysis(args)
    ca.run()
    # except Exception as Argument:
    #     with open("debug.log", "a") as f:
    #         f.write(f"Error happens when {args} \n: \n" + str(Argument))



