from dipy.tracking import utils
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.tracking.utils import _mapping_to_voxel, _to_voxel_coordinates

from nilearn.image import resample_img

import nibabel as nib
import numpy as np
import os
from os.path import join
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.tracking.streamline import set_number_of_points

import random
import subprocess
from glob import glob
from easydict import EasyDict
import pandas as pd
import Unsupervised.lib.dti_utils as dtils
from path import Path
import scipy.ndimage as ndi
import shutil
import warnings
from FsLut import FsLut
from FOD import FOD
"""
1. read and set neccessary file path
2. group and prepare the labels (seg) for target region
3. do tractography using mrtrix
4. compute connectivity given the streamlines and target labels
"""
import logging
from timeit import default_timer as timer
import sys
from typing import List, Tuple, Dict, Optional, Union


class ConnectivityAnalysisHCP(object):
    def __init__(self, args):
        self.args = args

        self.dout = args.io.dout
        self.dcache = join(self.dout, "cache")
        self.din = args.io.din # the HCP subject folder

        self.subid = os.path.basename(self.din)

        self.fslut = FsLut()
        self.lut =  None

        self.fdimg = ""
        self.fbvec = ""
        self.fbval = ""
        self.fseg = ""
        self.fbrainmask = ""

        self.setup_folder()

    def setup_folder(self):
        if not os.path.exists(self.dout):
            os.makedirs(self.dout)
            os.makedirs(self.dcache)

    def set_paths_HCP(self):
        print(self.subid)
        self.fdimg = join(self.din,"T1w/Diffusion/data.nii.gz")
        self.fbvec = join(self.din,"T1w/Diffusion/bvecs")
        self.fbval = join(self.din, "T1w/Diffusion/bvals")
        self.fbrainmask = join(self.din, "T1w/brainmask_fs.nii.gz")

        # The aparc+aseg.mgz uses the Desikan-Killiany atlas. To see the Destrieux atlas, you would load fsaverage/mri/aparc.a2009s+aseg.mgz
        if self.args.atlas.cparc == "Desikan":
            self.fseg = join(self.din, "T1w/aparc+aseg.nii.gz")
        elif self.args.atlas.cparc == "Destrieux":
            self.fseg = join(self.din, "T1w/aparc.a2009s+aseg.nii.gz")
        else:
            raise ValueError(f"Can't find {self.args.atlas.cparc} ")

        # For HCP dataset, here's no need to flip the sign of bvec. However, for other dataset, it's might be necessary.

        print(f"using {self.fdimg}")
        print(f"using {self.fbvec}")
        print(f"using {self.fbval}")


    def prepare_cortical_label(self, return_fpath = False, return_bimask = False):
        seg_data, _, seg = load_nifti(self.fseg, return_img=True)
        seg_shape = seg.shape

        # TODO: left or right should be shown differently in csv file. Now it is not.
        if self.args.atlas.cparc == "Desikan":
            if self.args.alg.sep_lr:
                target_labels = [1000 + i for i in range(1,36)] + [2000 + i for i in range(1,36)]
                self.lut = self.fslut.lut.loc[(self.fslut.lut.No>=1000) & ((self.fslut.lut.No<=1035))] # note that include 1000 which has label as "unknown"
                self.lut.append(self.fslut.lut.loc[(self.fslut.lut.No>=2001) & ((self.fslut.lut.No<=2035))])
                self.lut["Ind"] = list(range(71))
            else:
                target_labels = [[1000 + i, 2000+i] for i in range(1, 36)]
                self.lut = self.fslut.lut.loc[(self.fslut.lut.No >= 1000) & ((self.fslut.lut.No <= 1035))]
                self.lut["Ind"] = list(range(35+1))

        elif self.args.atlas.cparc == "Destrieux":
            if self.args.alg.sep_lr:
                target_labels = [11100 + i for i in range(1,76)] + [12100 + i for i in range(1,76)]
                self.lut = self.fslut.lut.loc[(self.fslut.lut.No >= 11100) & ((self.fslut.lut.No <= 11175))]
                self.lut.append(self.fslut.lut.loc[(self.fslut.lut.No >= 12100) & ((self.fslut.lut.No <= 12175))])
                self.lut["Ind"] = list(range(75*2+1))
            else:
                target_labels = [[11100 + i, 12100 + i] for i in range(1, 76)]
                self.lut = self.fslut.lut.loc[(self.fslut.lut.No >= 11100) & ((self.fslut.lut.No <= 11175))]
                self.lut["Ind"] = list(range(75+1))
        else:
            raise ValueError(f"Can't find {self.args.atlas.cparc} ")

        n_target = len(target_labels)
        print("number of targeted regions: ", n_target)

        # Create set of target masks
        target_mask = np.zeros(seg_shape)
        for j in range(n_target):
            j_mask = np.zeros(seg_shape)
            if isinstance(target_labels[j], list):
                for label in target_labels[j]:
                    j_mask = np.logical_or(j_mask, (seg_data == label))
            else:
                j_mask = np.logical_or(j_mask, (seg_data == target_labels[j]))

            target_mask[j_mask] = j + 1

        out_name = join(self.dout, f"{self.args.atlas.cparc}_target_parc.nii.gz")
        nib.Nifti1Image(target_mask.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
        out_name = join(self.dout, "cortical_bimask.nii.gz")
        nib.Nifti1Image((target_mask > 0).astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

        return (
            [join(self.dout, "cortical_bimask.nii.gz"),
             join(self.dout, "target_mask.nii.gz")],
            [(target_mask > 0).astype(np.uint32),
             target_mask],
            n_target
        )

    def get_tha_mask(self, return_fpath = False, return_bimask = False):
        seg_data, _, seg = load_nifti(self.fseg, return_img=True)
        mask = 1 * (seg_data == 10) + 2 * (seg_data == 49)
        out_name = join(self.dout, "tha_mask.nii.gz")
        nib.Nifti1Image(mask.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
        out_name = join(self.dout, "tha_bimask.nii.gz")
        nib.Nifti1Image((mask > 0).astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

        if return_fpath:
            if return_bimask:
                return join(self.dout, "tha_bimask.nii.gz")
            else:
                return join(self.dout, "tha_mask.nii.gz")
        else:
            if return_bimask:
                return (mask > 0).astype(np.uint32)
            else:
                return mask

    def dilate_mask(self, fmask, n_dilate):
        outfile = fmask.split(".")[0] + f"_dilate{n_dilate}.nii.gz"
        subprocess.run(
            f"maskfilter -npass {n_dilate} {fmask} dilate {outfile} -force".split())
        return outfile

    # def bbox_tha_mask(self, margin = 64):
    #     in_name = join(self.dout, "tha_bimask.nii.gz")
    #     seg_data, _, seg = load_nifti(in_name, return_img=True)
    #     dilated_data = np.zeros_like(seg_data)
    #     c = np.array([120, 147, 107])
    #     x0, y0, z0 = c - margin
    #     x1, y1, z1 = c + margin
    #
    #     dilated_data[x0:x1,y0:y1,z0:z1] = 1
    #     out_name = join(self.dout, f"tha_bimask_bbox_{margin}.nii.gz")
    #     nib.Nifti1Image(dilated_data.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
    #     return dilated_data, out_name


    def resample(self, fimg, suffix="default", res=None, tempimg=None, interp="linear"):
        """use mrgrid to perform resampling. During downsampling, mrgrid will handle aliasing problem"""
        if res and tempimg is None:
            outfile = fimg.split(".")[0] + f"{suffix}.nii.gz"
            subprocess.run(
                f"mrgrid {fimg} regrid -voxel {res} -interp {interp} {outfile} -force".split())
        else:
            outfile = fimg.split(".")[0] + f"{suffix}.nii.gz"
            subprocess.run(
                f"mrgrid {fimg} regrid -template {tempimg} -interp {interp} {outfile} -force".split())

        return outfile

    def parc_from_stack_of_maps(self, dms, template, tha_mask, kw = "keyword"):
        ind2label = {row["Ind"]: row["No"] for _, row in self.lut.iterrows()}
        dms = np.concatenate([np.zeros(template.shape)[..., np.newaxis], dms], axis=-1)
        parc = np.argmax(dms, axis=-1)
        parc[tha_mask == 0] = 0
        fsparc = np.vectorize(ind2label.get)(parc)  # map value through ind2label
        rgbimg = self.fslut.parc2rbg_3d(fsparc)
        
        affine = template.affine
        header = template.header
        cparc = self.args.atlas.cparc
        out_name = join(self.dout, f"{cparc}_{kw}_parc.nii.gz")
        nib.Nifti1Image(parc.astype(np.uint32), affine, header).to_filename(out_name)
        out_name = join(self.dout, f"{cparc}_{kw}_fsparc.nii.gz")
        nib.Nifti1Image(fsparc.astype(np.uint32), affine, header).to_filename(out_name)
        out_name = join(self.dout, f"{cparc}_{kw}_fsparc_rgb.nii.gz")
        nib.Nifti1Image(rgbimg.astype(np.uint32), affine, header).to_filename(out_name)

    def resample_strms(self, strms, factor):
        up_strms = []
        for strm in strms:
            N = strm.shape[0]
            up_strm = set_number_of_points(strm, int(N*factor))
            up_strms.append(up_strm)
        return up_strms

    def run(self):
        """ when everything is prepared, then run!

        """
        logger = logging.getLogger('root')
        # check output dir and see if it's already exist
        if os.path.exists(f"{self.dout}/tracks.tck"):
            warnings.warn("The output dir contains the output file, skip re-run this case.")
            return

        self.set_paths_HCP()
        data, _, dmri = load_nifti(self.fdimg, return_img=True)

        # keep resolution of dMRI and resample brain_mask and seg to match it.
        res, sr = self.args.alg.res, self.args.alg.sr
        # dif_res = dmri.header.get_zooms()[:3] # get diffusion voxel spacing
        # dif_res_str = f"{dif_res[0]},{dif_res[1]},{dif_res[2]}"
        # self.fdimg, dmri = self.resample(dmri, f"data_res{res}.nii.gz", interp = 'linear') # update reference
        self.fbrainmask = self.resample(self.fbrainmask, suffix="_regrid", tempimg=self.fdimg , interp = 'nearest')
        if sr:
            self.fseg = self.resample(self.fseg, suffix = f"_{sr}", res = sr, interp='nearest')
        else: # downsampling the seg, so that the parcellation map will be the same res as drmi
            self.fseg = self.resample(self.fseg, suffix = f"_regrid", tempimg=self.fdimg, interp = 'nearest')
        labels, _, seg = load_nifti(self.fseg, return_img=True) # update

        timer1 = timer()
        # get thalamus mask from seg
        ftha_mask = self.get_tha_mask(return_fpath=True, return_bimask=True)
        if self.args.alg.dilate_roi>0:
            ftha_mask = self.dilate_mask(ftha_mask, n_dilate = self.args.alg.dilate_roi)
        tha_mask, _ = load_nifti(ftha_mask)

        print(f"Number of thalamic voxels: {np.sum(tha_mask.flatten())}")
        logger.info(f"Number of thalamic voxels: {np.sum(tha_mask.flatten())}")

        # get cortical mask from seg
        ([fcortical_bimask, ftarget_parc], [cortical_bimask, target_parc], n_target) \
            = self.prepare_cortical_label()

        # run mrtrix3 tractography
        print("===> running tractography...")
        pre = self.dcache
        subprocess.run(f"mrconvert {self.fdimg} -fslgrad {self.fbvec} {self.fbval} {pre}/dwi_corrected.mif -force".split())

        if not os.path.exists(f"{pre}/wmfod.mif"):
            subprocess.run(f"dwi2response dhollander {pre}/dwi_corrected.mif {pre}/wm.txt {pre}/gm.txt {pre}/csf.txt -force".split())
            subprocess.run(f"dwi2fod msmt_csd {pre}/dwi_corrected.mif -mask {self.fbrainmask} \
                            {pre}/wm.txt {pre}/wmfod.mif {pre}/gm.txt {pre}/gmfod.mif {pre}/csf.txt {pre}/csffod.mif -force".split())
        fwmfod = f"{pre}/wmfod.mif"

        finput = fwmfod
        excludes = ""
        subprocess.run(f"tckgen -select {self.args.alg.tr.select} -algorithm iFOD2 -seed_image {ftha_mask} \
                    -angle 22.5 -maxlen 250 -minlen 10 -include {ftha_mask} -include {fcortical_bimask} \
                    {excludes} -mask {self.fbrainmask} {finput} \
                    -output_seeds {pre}/survived_seeds.txt {self.dout}/tracts.tck -force".split())

        timer2 = timer()
        print("===> finish tractography")

        # compute density map
        streamlines = load_tractogram(join(self.dout,"tracts.tck"), seg)
        fseeds = f"{pre}/survived_seeds.txt"

        streamlines = streamlines.streamlines

        seedinfo = pd.read_csv(fseeds, skiprows=1)
        seedinfo = seedinfo.iloc[:, :5]
        seedinfo = seedinfo.to_numpy()

        for target_name, target_mask in zip([self.args.atlas.cparc,],[target_parc, ]):
            dms = []
            seedfroms = []
            passthros = []
            print(f"Using {target_name} to count fibers...")
            for i in range(n_target):
                cortical_mask_i = target_mask == (i + 1)
                if self.args.alg.dilate_trg > 0:
                    cortical_mask_i = ndi.binary_dilation(cortical_mask_i, iterations = self.args.alg.dilate_trg)
                # stm_ind, stm_i = dtils.target(streamlines, seg.affine, cortical_mask_i)
                r = dtils.target(streamlines, seg.affine, cortical_mask_i)
                r = list(r)
                stm_ind = [rr[0] for rr in r]
                stm_ind = np.array(stm_ind)
                stm_i = [rr[1] for rr in r]

                print(f"{i+1:02d}/{n_target}; #streamlines = {len(stm_i)}", )
                if len(stm_i)>0:
                    stm_i = self.resample_strms(stm_i, self.args.alg.up_factor) if self.args.alg.up_factor > 1 else stm_i
                    dm_i,seedfrom_i = self.two_type_density_map(stm_i, seedinfo[stm_ind,:], seg)
                    dm_i, seedfrom_i = dm_i.astype(np.float32),seedfrom_i.astype(np.float32)
                else:
                    dm_i, seedfrom_i = np.zeros((seg.shape),dtype="float32"), np.zeros((seg.shape),dtype="float32")

                if self.args.alg.conn_normalize:
                    area = np.sum(cortical_mask_i.flatten())/1000.
                    dm_i /= area
                    seedfrom_i /= area

                dms.append(dm_i)
                seedfroms.append(seedfrom_i)
                passthros.append(dm_i-seedfrom_i)

            timer3 = timer()

            dms = np.stack(dms, axis=-1)
            seedfroms = np.stack(seedfroms, axis=-1)
            passthros = np.stack(passthros, axis=-1)

            if self.args.io.save_all:
                print("writing to ", self.dout)
                out_name = join(self.dout, f"{target_name}_denmap.nii.gz")
                nib.Nifti1Image(dms.astype(np.float32), seg.affine, seg.header).to_filename(out_name)
                out_name = join(self.dout, f"{target_name}_seedfrommap.nii.gz")
                nib.Nifti1Image(seedfroms.astype(np.float32), seg.affine, seg.header).to_filename(out_name)
                out_name = join(self.dout, f"{target_name}_passthromap.nii.gz")
                nib.Nifti1Image(passthros.astype(np.float32), seg.affine, seg.header).to_filename(out_name)

            self.parc_from_stack_of_maps(dms, template=seg, tha_mask=tha_mask, kw="denmap")
            self.parc_from_stack_of_maps(seedfroms, template=seg, tha_mask=tha_mask, kw="seedfrommap")
            self.parc_from_stack_of_maps(passthros, template=seg, tha_mask=tha_mask, kw="passthromap")

        timer3 = timer()
        logger.info(f"total running elapse: {timer3-timer1}")
        logger.info(f"tractography elapse: {timer2 - timer1}")
        logger.info(f"compute connectivity elapse: {timer3 - timer2}")
        self.lut.to_csv(f"{self.dout}/{self.args.atlas.cparc}_lut.csv")

    def two_type_density_map(self,streamlines,seedinfo,seg):
        affine = seg.affine
        lin_T, offset = _mapping_to_voxel(affine)

        seed_coord = seedinfo[:,2:]

        inds = _to_voxel_coordinates(seed_coord, lin_T, offset)
        seed_from_count = np.zeros(seg.shape, 'int')
        for coord in inds:
            i,j,k = coord[0],coord[1],coord[2]
            seed_from_count[i, j, k] += 1

        counts = np.zeros(seg.shape, 'int')
        for sl in streamlines:
            inds = _to_voxel_coordinates(sl, lin_T, offset)
            i, j, k = inds.T
            # this takes advantage of the fact that numpy's += operator only
            # acts once even if there are repeats in inds
            counts[i, j, k] += 1

        return counts, seed_from_count

from argparse import ArgumentParser, Namespace

def add_parser():
    parser = ArgumentParser()
    # parser.add_argument('--name', type=str, default="fill_exp_name_here",
    #                     help="the experiment name, which is used to differetiate each run")
    parser.add_argument('--din', type=str, default="",
                        help="input case folder: xx/subjid")
    parser.add_argument('--dout', type=str, default="",
                        help="where to save output files")
    parser.add_argument('--sr', type=float, default=0,
                        help="compute features on higher resolution image gird?")

    parser.add_argument('--seg,cparc', type=str, default="Desikan", choices=["Desikan", "Destrieux"],
                        help="which cortical parcellation is used.")

    parser.add_argument('--sep_lr', type=int, default=0,
                        help="if labels in seg labeled left and right hemisphere seperately, do we want to keep them seperated?")

    parser.add_argument('--res', type=float, default=2.0,
                        help="resample image to the res")

    parser.add_argument('--tr_select', type=int, default=100000,
                        help="number of streamlines")
    parser.add_argument('--tr_seedtha', type=int, default=1,
                        help="1-seed inside thalamus. 0-seed in white matter")
    parser.add_argument('--con_normalize', type=int, default=0,
                        help="whether normalize counts by area of cortical region")
    # parser.add_argument('--tr_dilate_thamask', type=int, default=0,
    #                     help="dilate thalamus mask for seeding? if so, by how many voxels")
    # parser.add_argument('--tr_bbox_thamask', type=int, default=0,
    #                     help="use the bbox as the seed region? also see --margin")
    # parser.add_argument('--margin', type=int, default=64,
    #                     help="use the bbox as the seed region?")

    parser.add_argument('--up_factor', type=float, default=1.0,
                        help="upsampling the streamlines for higher resolution grid")

    parser.add_argument('--tr_dilate_targetmask', type=int, default=0,
                        help="dilate cortical mask? if so, by how many voxels")
    parser.add_argument('--rm_exist', type=int, default=1,
                        help="if outdir is exsiting, do we want to remove the existing folder and make a new one.")
    parser.add_argument('--save_all', type=int, default=1,
                        help="Besides parc results, this will also save probility maps. Turn this off when parc is the only your interest")
    return parser

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # args = EasyDict()
    # args["din"] = "/mnt/ssd2/AllDatasets/HCP102/991267"
    # args["dout"] = "/mnt/ssd2/Projects/ThaParc/data/cache/991267/default"
    # args["tr_select"] = 1000
    # args["tr_seedtha"] = True
    # args["con_normalize"] = False
    # args["sr"] = 1.25
    # args["rm_exist"] = False
    # args["seg"] = "Desikan"
    # args["sep_lr"] = False
    # args["res"] = 2.0
    # args["save_all"] = 1
    # args["tr_dilate_targetmask"] = 0
    # args["up_factor"] = 1.5

    # parser = add_parser()
    # args = parser.parse_args()
    # args = EasyDict(vars(args))
    print(OmegaConf.to_yaml(cfg))
    ca = ConnectivityAnalysisHCP(cfg)

    file_handler = logging.FileHandler(filename=f'{ca.dout}/debug.log')
    stdout_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(fmt='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    # handlers = [file_handler, stdout_handler]
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    ca.run()

"""python xx.py io.din=/mnt/ssd2/AllDatasets/HCP102P/991267 io.dout=/mnt/ssd2/Projects/ThaParc/data/cache/991267/default6"""
if __name__ == "__main__":
    main()

