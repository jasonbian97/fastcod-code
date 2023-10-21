import nibabel as nib
import numpy as np
from pathlib import Path
import sys

CURRENTPATH = Path(__file__).resolve().parent
PROJECT_DIR = CURRENTPATH.parent
sys.path.append(str(PROJECT_DIR))
import src.slant_helper as slhelper
from argparse import ArgumentParser, Namespace

def add_parser():
    parser = ArgumentParser()
    parser.add_argument('--seg_type', type=str, required=True, metavar='slant/freesurfer',
                        help="Type of segmentation", choices=["slant","freesurfer"])
    # input file
    parser.add_argument('--fseg', type=str, required=True, metavar='PATH',
                        help="Segmentation file path")
    # dout
    parser.add_argument('--dout', type=str, required=True, metavar='DIR',
                        help="Output directory")
    return parser


if __name__ == "__main__":
    parser = add_parser()
    args = parser.parse_args()

    fseg = args.fseg
    seg_type = args.seg_type
    dout = Path(args.dout)

    if not dout.exists():
        dout.mkdir(parents=True,exist_ok=True)

    if seg_type == "slant":
        slhelper.prepare_cortical_label(fseg, dout)
        slhelper.get_csf_mask(fseg, dout)
        slhelper.get_tha_mask(fseg, dout)

















