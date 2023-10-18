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
import random
import subprocess
from glob import glob
from easydict import EasyDict
import pandas as pd
import dti_utils as dtils
import scipy.ndimage as ndi
import shutil
import warnings
from pathlib import Path
import tempfile
from slant_helper import prepare_cortical_label, get_tha_mask, get_csf_mask

"""
1. read and set neccessary file path
2. group and prepare the labels (seg) for target region
3. do tractography using mrtrix
4. compute connectivity given the streamlines and target labels
"""


class ConnectivityAnalysis(object):
    def __init__(self, args):
        self.args = EasyDict(args)

        # Required arguments
        self.dout = Path(self.args["output_folder"])
        if not self.dout.exists():
            self.dout.mkdir(parents=True, exist_ok=True)
        self.dcache = Path(tempfile.mkdtemp(prefix=str(self.dout) + "/0"))

        self.fdimg = args.fdimg
        self.fbvec = args.fbvec
        self.fbval = args.fbval
        self.fFOD = args.fFOD
        self.fsrc_mask = args.fsrc_mask
        self.fbrainmask = args.fbrainmask

        # optional
        self.fslant = args.fslant
        self.ftrg_seg = args.ftrg_seg
        self.tr_select = self.args.tr_select
        self.wm_mask = ""
        self.bvec_flip = args.bvec_flip
        self.down_res = args.down_res
        self.tr_alg = args.tr_alg
        self.tr_dilate_src = args.tr_dilate_src
        self.tr_dilate_trg = args.tr_dilate_trg
        self.tr_bbox = args.tr_bbox
        self.tr_bbox_margin = args.tr_bbox_margin


        # check input arguments
        if (self.ftrg_seg is None) and (self.fslant is None):
            raise ValueError("Either fslant or ftrg_seg should be provided")



    def set_paths_mmti(self):

        self.wm_mask = f"/iacl/pg22/muhan/MTBI_project/data/mmti/alldata_withslant/{self.subid}/{self.subid}_01_MPRAGE_T1w_norm_slant_filledwm_mask.nii.gz"

        # special operation
        bvecs = np.loadtxt(self.fbvec)
        # bvecs[0, :] *= -1  # flip x
        bvecs[1, :] *= -1  # flip y
        np.savetxt(f"{self.dout}/flippedy.bvecs", bvecs, fmt="%.6f")
        self.fbvec = f"{self.dout}/flippedy.bvecs"

        print(f"using {self.fdimg}")
        print(f"using {self.fbvec}")
        print(f"using {self.fbval}")



    def set_paths_mtbi(self):
        print(self.subid)
        self.fdimg = glob(f"/iacl/pg22/zhangxing/AllDatasets/mtbi/{self.subid}/ConnectivityAnalysis/*_DWI_DrBUDDY_acqres.nii.gz")[0]
        if not os.path.exists(self.fdimg):
            self.fdimg = glob(join(self.din,"01","proc","*_01_tortoise","*_01_DWI_AP_proc_DRBUDDI_proc", "*_01_DWI_AP_proc_DRBUDDI_up_final.nii"))[0]
        self.fbvec = glob(join(self.din,"01","proc","*_01_tortoise","*_01_DWI_AP_proc_DRBUDDI_proc", "*_01_DWI_AP_proc_DRBUDDI_up_final.bvecs"))[0]
        self.fbval = glob(join(self.din,"01","proc","*_01_tortoise","*_01_DWI_AP_proc_DRBUDDI_proc", "*_01_DWI_AP_proc_DRBUDDI_up_final.bvals"))[0]
        self.fseg = glob(join(self.din,"01","proc", "*_01_MPRAGEPre_norm_slant_macruise.nii.gz"))[0]
        self.fbrainmask = glob(join(self.din,"01","proc", "*_01_MPRAGEPre_norm_robex.nii.gz"))[0]

        # special paths
        self.wm_mask = glob(join(self.din,"01","proc", "**_01_MPRAGEPre_norm_slant_filledwm_mask.nii.gz"))[0]

        # special operation
        bvecs = np.loadtxt(self.fbvec)
        bvecs[0, :] *= -1 # flip x
        bvecs[1, :] *= -1 # flip y
        np.savetxt(f"{self.dout}/flippedxy.bvecs", bvecs, fmt="%.6f")
        self.fbvec = f"{self.dout}/flippedxy.bvecs"

    def set_paths_demo(self):

        self.fbval = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/DWI_tortoise/MRCON2002_01_DWI_AP_proc_DRBUDDI_up_final.bvals"
        self.fbvec = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/DWI_tortoise/MRCON2002_01_DWI_AP_proc_DRBUDDI_up_final_flipped.bvecs"
        # self.fdimg = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/DWI_tortoise/MRCON2002_01_DWI_AP_proc_DRBUDDI_up_final.nii.gz"
        self.fdimg = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/DWI_tortoise/lowres_tortoise_MNI.nii.gz"
        self.fbrainmask = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/MRCON2002_01_MPRAGEPre_norm_robex.nii.gz"
        self.fseg = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/MRCON2002_01_MPRAGEPre_norm_slant.nii.gz"
        self.wm_mask = "/mnt/ssd2/Projects/DTI/MTBI_subset/MRCON2002/MRCON2002_01_MPRAGEPre_norm_slant_filledwm_mask.nii.gz"





    def dilate_tha_mask(self, iter = 1):
        in_name = join(self.dout, "tha_bimask.nii.gz")
        seg_data, _, seg = load_nifti(in_name, return_img=True)
        dilated_data = ndi.binary_dilation(seg_data, iterations = iter)
        out_name = join(self.dout, "tha_bimask_dilated.nii.gz")
        nib.Nifti1Image(dilated_data.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
        return dilated_data, out_name

    def dilate_target_mask(self,iter=1):
        in_name = join(self.dout, "cortical_bimask.nii.gz")
        seg_data, _, seg = load_nifti(in_name, return_img=True)
        dilated_data = ndi.binary_dilation(seg_data, iterations=iter)
        out_name = join(self.dout, "cortical_bimask_dilated.nii.gz")
        nib.Nifti1Image(dilated_data.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
        return dilated_data, out_name

    def bbox_tha_mask(self, margin = 64):
        in_name = join(self.dout, "tha_bimask.nii.gz")
        seg_data, _, seg = load_nifti(in_name, return_img=True)
        dilated_data = np.zeros_like(seg_data)
        c = np.array([120, 147, 107])
        x0, y0, z0 = c - margin
        x1, y1, z1 = c + margin

        dilated_data[x0:x1,y0:y1,z0:z1] = 1
        out_name = join(self.dout, f"tha_bimask_bbox_{margin}.nii.gz")
        nib.Nifti1Image(dilated_data.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
        return dilated_data, out_name

    def run(self):
        """ when everything is prepared, then run!
        """

        data, _, dmri = load_nifti(self.fdimg, return_img=True)
        labels, _, seg = load_nifti(self.fseg, return_img=True)
        src_mask, _, src = load_nifti(self.fsrc_mask, return_img=True)
        trg_seg, _, trg = load_nifti(self.ftrg_seg, return_img=True)

        bvals, bvecs = read_bvals_bvecs(self.fbval, self.fbvec)
        gtab = gradient_table(bvals, bvecs)

        # if self.fslant is not None:
        #     # ftha_mask, _, tha_mask, _ = get_tha_mask(self.fslant, self.dcache)
        #     fcsf_mask = get_csf_mask(self.fslant, self.dcache, return_fpath=True, return_bimask=True)
        #     ([fcortical_bimask,ftarget_6_mask,ftarget_98_mask], [cortical_bimask,target_6_mask,target_98_mask])\
        #         = prepare_cortical_label(self.fslant, self.dcache)
        #     print(f"Number of src voxels: {src_mask.flatten().sum()}")

        # downsample dMRI to accelerate process
        if self.down_res:
            print("===> downsampling resolution of dMR and brain_mask to (2,2,2)...")
            target_affine = float(self.down_res) * np.eye(3) * np.sign(np.diag(dmri.affine[:3,:3]))
            dmri = resample_img(dmri, target_affine=target_affine, interpolation='linear')
            self.fdimg = f"{self.dout}/dwi_down_res_{self.down_res}.nii.gz"
            nib.save(dmri, self.fdimg)

        # downsample the brainmask to match the resolution of dMRI
        _, _, brain_mask = load_nifti(self.fbrainmask, return_img=True)
        brain_mask_lowres = resample_img(brain_mask, target_affine = dmri.affine, target_shape=dmri.shape[:3],interpolation='nearest')
        nib.save(brain_mask_lowres,f"{self.dout}/brain_mask_lowres.nii.gz")
        nib.save(brain_mask, f"{self.dout}/brain_mask.nii.gz")

        # run mrtrix FOD
        print("===> running FOD...")
        pre = self.dcache
        subprocess.run(f"mrconvert {self.fdimg} -fslgrad {self.fbvec} {self.fbval} {pre}/dwi.mif".split())
        subprocess.run(f"dwi2response dhollander {pre}/dwi.mif {pre}/wm.txt {pre}/gm.txt {pre}/csf.txt", shell=True)
        subprocess.run(f"dwi2fod msmt_csd {pre}/dwi.mif -mask {self.dout}/brain_mask_lowres.nii.gz \
                        {pre}/wm.txt {pre}/wmfod.mif {pre}/gm.txt {pre}/gmfod.mif {pre}/csf.txt {pre}/csffod.mif", shell=True)
        fwmfod = f"{pre}/wmfod.mif"

        fseed_mask = self.fsrc_mask

        if self.tr_alg in ["iFOD2","iFOD1","SD_STREAM"]: # input is FOD
            finput = fwmfod
        elif self.tr_alg in ["Tensor_Prob","Tensor_Det"]: # input is dwi
            finput = f"{pre}/dwi_corrected.mif"
        else:
            raise ValueError(f"{self.tr_alg} alg is not found!")

        if self.tr_dilate_src>0: # seed in dilated src region and not exclude surrounding streamlines
            tha_mask_dilated, ftha_mask_dilated = self.dilate_tha_mask(iter=self.tr_dilate_src)
            fseed_mask = ftha_mask_dilated
            ftha_mask = ftha_mask_dilated
            tha_mask = tha_mask_dilated

        if self.tr_bbox:
            tha_mask_dilated, ftha_mask_dilated = self.bbox_tha_mask(margin = self.tr_bbox_margin)
            fseed_mask = ftha_mask_dilated
            ftha_mask = ftha_mask_dilated
            tha_mask = tha_mask_dilated

        # if self.args.tr_excsf:
        #     excludes = f"-exclude {fcsf_mask}"
        # else:
        #     excludes = ""
        excludes = ""

        if self.tr_dilate_trg>0:
            cortical_bimask, fcortical_bimask = self.dilate_target_mask(iter=self.tr_dilate_trg)

        subprocess.run(f"tckgen -select {self.tr_select} -algorithm {self.tr_alg} -seed_image {fseed_mask} \
                        -angle 22.5 -maxlen 250 -minlen 10 -include {fsrc_bimask} -include {ftrg_bimask} \
                        {excludes} -mask {self.fbrainmask} {finput} \
                        -output_seeds {pre}/survived_seeds.txt {self.dout}/tracts.tck".split())

        if fseed_mask == self.wm_mask:
            subprocess.run(f"tckedit {self.dout}/tracts.tck {self.dout}/tracts_filt_tha.tck -include {ftha_mask}".split())

        print("===> finish tractography")


        # compute density map
        # if self.args["debug"]:
        #     streamlines = load_tractogram("/mnt/ssd2/Projects/DTI/Connectivity_DIPY/demo/ConnectivityAnalysis_971120/tracks_100k.tck",seg)
        #     fseeds = "/mnt/ssd2/Projects/DTI/Connectivity_DIPY/demo/ConnectivityAnalysis/cache_971120/survived_seeds.txt"
        # else:
        streamlines = load_tractogram(join(self.dout,"tracts.tck"), seg)
        fseeds = f"{pre}/survived_seeds.txt"

        streamlines = streamlines.streamlines

        seedinfo = pd.read_csv(fseeds, skiprows=1)
        seedinfo = seedinfo.iloc[:, :5]
        seedinfo = seedinfo.to_numpy()

        for target_name, target_mask in zip(["target_6", "target_98"],[target_6_mask, target_98_mask]):
            num_classes = len(np.unique(target_mask)) - 1
            dms = []
            seedfroms = []
            passthros = []
            print(f"Using {target_name} to count fibers...")
            for i in range(num_classes):
                cortical_mask_i = target_mask == (i + 1)
                if self.args.tr_dilate_targetmask>0:
                    cortical_mask_i = ndi.binary_dilation(cortical_mask_i, iterations = self.args.tr_dilate_targetmask)
                # stm_ind, stm_i = dtils.target(streamlines, seg.affine, cortical_mask_i)
                r = dtils.target(streamlines, seg.affine, cortical_mask_i)
                r = list(r)
                stm_ind = [rr[0] for rr in r]
                stm_ind = np.array(stm_ind)
                stm_i = [rr[1] for rr in r]

                print(f"{i:02d}/{num_classes}; #streamlines = {len(stm_i)}", )
                if len(stm_i)>0:
                    dm_i,seedfrom_i = self.two_type_density_map(stm_i, seedinfo[stm_ind,:], seg)
                    dm_i, seedfrom_i = dm_i.astype(np.float32),seedfrom_i.astype(np.float32)
                else:
                    dm_i, seedfrom_i = np.zeros((seg.shape),dtype="float32"), np.zeros((seg.shape),dtype="float32")

                if self.args.con_normalize:
                    area = np.sum(cortical_mask_i.flatten())/1000.
                    dm_i /= area
                    seedfrom_i /= area

                dms.append(dm_i)
                seedfroms.append(seedfrom_i)
                passthros.append(dm_i-seedfrom_i)

            dms = np.stack(dms, axis=-1)
            seedfroms = np.stack(seedfroms, axis=-1)
            passthros = np.stack(passthros, axis=-1)

            if self.args.save_all:
                print("writing to ", self.dout)
                out_name = join(self.dout, f"{target_name}_denmap.nii.gz")
                nib.Nifti1Image(dms.astype(np.float32), seg.affine, seg.header).to_filename(out_name)
                out_name = join(self.dout, f"{target_name}_seedfrommap.nii.gz")
                nib.Nifti1Image(seedfroms.astype(np.float32), seg.affine, seg.header).to_filename(out_name)
                out_name = join(self.dout, f"{target_name}_passthromap.nii.gz")
                nib.Nifti1Image(passthros.astype(np.float32), seg.affine, seg.header).to_filename(out_name)

            # label 0 is bg, label 1 is the voxel that has zero count, starting from label2 are connectivity info.
            a = np.concatenate([np.zeros(seg.shape)[...,np.newaxis],dms],axis=-1)
            parc = np.argmax(a, axis=-1) + 1
            parc[tha_mask == 0] = 0
            out_name = join(self.dout, f"{target_name}_denmap_parc.nii.gz")
            nib.Nifti1Image(parc.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

            a = np.concatenate([np.zeros(seg.shape)[..., np.newaxis], seedfroms], axis=-1)
            parc = np.argmax(a, axis=-1) + 1
            parc[tha_mask == 0] = 0
            out_name = join(self.dout, f"{target_name}_seedfrommap_parc.nii.gz")
            nib.Nifti1Image(parc.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

            a = np.concatenate([np.zeros(seg.shape)[..., np.newaxis], passthros], axis=-1)
            parc = np.argmax(a, axis=-1) + 1
            parc[tha_mask == 0] = 0
            out_name = join(self.dout, f"{target_name}_passthromap_parc.nii.gz")
            nib.Nifti1Image(parc.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

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

if __name__ == "__main__":
    args = {}
    args["input_folder"] = ""
    args["dataset"] = "demo"
    args["output_folder"] = "/mnt/ssd2/Projects/ThaParc/Unsupervised/demo"
    args["tr_select"] = 10000
    args["tr_seedtha"] = True
    args["con_normalize"] = False
    args["debug"] = False

    ca = ConnectivityAnalysis(args)
    ca.run()

