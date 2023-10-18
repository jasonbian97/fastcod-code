
import os
from os.path import join
from glob import glob
import multiprocessing
from argparse import ArgumentParser, Namespace
import logging
import sys
sys.path.append("/mnt/ssd2/Projects/ThaParc")
sys.path.append("/mnt/ssd2/Projects/ThaParc/Unsupervised")
sys.path.append("/iacl/pg22/zhangxing/Projects/ThaParc")
sys.path.append("/iacl/pg22/zhangxing/Projects/ThaParc/Unsupervised")
sys.path.append("/home/zhangxing/Projects/ThaParc")
sys.path.append("/home/zhangxing/Projects/ThaParc/Unsupervised")
from Unsupervised.lib.ConnectivityAnalysis import ConnectivityAnalysis

def lins(l,s):
    for ll in l:
        if ll in s:
            return True
    return False

def run_one_case(args):
    try:
        ca = ConnectivityAnalysis(args)
        ca.run()
    except Exception as Argument:
        with open("debug.log", "a") as f:
            f.write(f"Error happens when {args} \n: \n" + str(Argument))

def add_parser():
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default="fill_exp_name_here",
                        help="the experiment name, which is used to differetiate each run")
    parser.add_argument('--dataset', type=str, default="mtbi",
                        help="the dataset identifier")
    parser.add_argument('--debug', type=int, default=0,
                        help="local debug mode")
    parser.add_argument('--skip1', type=int, default=0,
                        help="skip running local modeling, assume there's already local modeling file availble (e.g., wmfod.mif)")

    parser.add_argument('--tr_select', type=int, default=100000,
                        help="number of streamlines")
    parser.add_argument('--tr_seedtha', type=int, default=1,
                        help="1-seed inside thalamus. 0-seed in white matter")
    parser.add_argument('--tr_alg', type=str, default="iFOD2",
                        choices=["iFOD2","iFOD1","SD_STREAM","Tensor_Det","Tensor_Prob"],
                        help="tracing algorithm")
    parser.add_argument('--con_normalize', type=int, default=1,
                        help="whether normalize counts by area of cortical region")
    parser.add_argument('--tr_dilate_thamask', type=int, default=0,
                        help="dilate thalamus mask for seeding? if so, by how many voxels")
    parser.add_argument('--tr_bbox_thamask', type=int, default=0,
                        help="use the bbox as the seed region? also see --margin")
    parser.add_argument('--margin', type=int, default=64,
                        help="use the bbox as the seed region?")

    parser.add_argument('--tr_brainmask', type=str, default="slant", choices=["slant","robex"],
                        help="slant or robex")

    parser.add_argument('--tr_excsf', type=int, default=1,
                        help="exclude csf region during tracts?"
                        )
    parser.add_argument('--tr_dilate_targetmask', type=int, default=0,
                        help="dilate cortical mask? if so, by how many voxels")

    parser.add_argument('--cerebellum', type=int, default=0,
                        help="Want to include cerebellum for thalamus parcellation? 0-No, 1-Yes")

    parser.add_argument('--rm_exist', type=int, default=1,
                        help="if outdir is exsiting, do we want to remove the existing folder and make a new one.")
    parser.add_argument('--save_all', type=int, default=1,
                        help="Besides parc results, this will also save probility maps. Turn this off when parc is the only your interest")

    parser.add_argument('--file_list', type=str, default=None,
                        help="if assgined, it should be a .txt file which contains the case path at each row. If not assgined, all searchable case will be processed.")
    parser.add_argument('--skip_list', type=str, default=None,
                        help="if assgined, it should be a .txt file which contains keyword of cases you want to skip")
    parser.add_argument('--thread', type=int, default=3,
                        help="number of thread")
    return parser

from easydict import  EasyDict

if __name__ == "__main__":
    parser = add_parser()
    args = parser.parse_args()
    args = EasyDict(vars(args))
    if args.debug:
        args.output_folder = "/mnt/ssd2/Projects/ThaParc/Unsupervised/demo"
        args.input_folder = ""
        ca = ConnectivityAnalysis(args)
        ca.run()

    if args.dataset == "mtbi":
        CASES_DIR = "/iacl/pg20/muhan/MTBI_project/data/mtbi"

    if args.dataset == "mmti":
        CASES_DIR = "/iacl/pg22/muhan/MTBI_project/data/mmti/alldata_withslant/data"

    if args.file_list:
        with open(args.file_list, 'r') as f:
            lines = f.readlines()
            fcases = [join(CASES_DIR,os.path.basename(line.strip())) for line in lines]
    else:
        # fcases = sorted(glob(join(CASES_DIR, "MR*")))
        fcases = [join(CASES_DIR,subj) for subj in os.listdir(CASES_DIR)] # more general

    skip_cases = []
    if args.skip_list:
        with open(args.skip_list, 'r') as f:
            skip_cases = f.readlines()
            print(f"====> skip files:", "\n", "\n".join(skip_cases))

    fcases = [fcase for fcase in fcases if not lins(skip_cases,fcase)] # maybe skip some cases

    err_cases = []
    largs = []
    for fcase in fcases:
        print("===> add to queue: ", fcase)
        case_id = os.path.basename(fcase)
        dargs = {}
        dargs.update(args)
        dargs["input_folder"] = fcase
        dargs["output_folder"] = f"/iacl/pg22/zhangxing/AllDatasets/{args.dataset}/{case_id}/ConnectivityAnalysis_{args.name}"
        largs.append(dargs)

    pool = multiprocessing.Pool(args.thread)
    pool.map(run_one_case,largs)
    # run_one_case(largs[0])



