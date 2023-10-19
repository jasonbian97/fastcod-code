import subprocess
import tempfile
from os.path import join
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import _mapping_to_voxel, _to_voxel_coordinates
from easydict import EasyDict
from nilearn.image import resample_img

import dti_utils as dtils

"""
1. read and set neccessary file path
2. group and prepare the labels (seg) for target region
3. do tractography using mrtrix
4. compute connectivity given the streamlines and target labels
"""


class ConnectivityAnalysis(object):
    def __init__(self, args):
        self.args = EasyDict(vars(args))

        # Required arguments
        self.dout = Path(args.dout)
        if not self.dout.exists():
            self.dout.mkdir(parents=True, exist_ok=True)
        self.dcache = Path(tempfile.mkdtemp(dir=str(self.dout)))

        self.fdimg = args.fdimg
        self.fbvec = args.fbvec
        self.fbval = args.fbval
        self.fFOD = args.fFOD
        self.fsrc_mask = args.fsrc_mask
        self.fbrainmask = args.fbrainmask
        self.ftrg_seg = args.ftrg_seg

        # optional
        self.exp_mode = args.exp_mode
        self.tr_select = args.tr_select
        self.bvec_flip = args.bvec_flip
        self.down_res = args.down_res
        self.tr_alg = args.tr_alg
        self.tr_dilate_src = args.tr_dilate_src
        self.tr_dilate_trg = args.tr_dilate_trg
        # self.tr_bbox = args.tr_bbox
        # self.tr_bbox_margin = args.tr_bbox_margin
        self.normalize_by_area = args.normalize_by_area

    @staticmethod
    def dilate_mask(fmask, iter=1):
        seg_data, _, seg = load_nifti(fmask, return_img=True)
        dilated_data = ndi.binary_dilation(seg_data, iterations=iter)
        out_name = fmask.replace(".nii.gz", f"_dilated.nii.gz")
        nib.Nifti1Image(dilated_data.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
        return dilated_data, out_name

    @staticmethod
    def flip_bvec(fbvec, flipaxis, fout):
        bvecs = np.loadtxt(fbvec)
        if flipaxis == "x":
            bvecs[0, :] *= -1  # flip x
        elif flipaxis == "y":
            bvecs[1, :] *= -1  # flip y
        elif flipaxis == "z":
            bvecs[2, :] *= -1
        np.savetxt(f"{fout}", bvecs, fmt="%.6f")

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

    def run(self):
        """ when everything is prepared, then run!
        """
        src_mask, _, srcimg = load_nifti(self.fsrc_mask, return_img=True)
        trg_seg, _, trgimg = load_nifti(self.ftrg_seg, return_img=True)
        pre = self.dcache

        if self.fdimg is not None: # raw diffusion image input
            _, _, dmri = load_nifti(self.fdimg, return_img=True)
            fdimg = self.fdimg

            # flip bvec
            fbvec = self.fbvec
            fbval = self.fbval
            if self.bvec_flip:
                print("===> flipping bvec...")
                self.flip_bvec(self.fbvec, self.bvec_flip, f"{self.dcache}/bvecs_flipped.bvec")
                fbvec = f"{self.dcache}/bvecs_flipped_{self.bvec_flip}.bvec"

            # downsample dMRI to accelerate process
            if self.down_res:
                print(f"===> downsampling resolution of dMR to {self.down_res} isotropic")
                target_affine = float(self.down_res) * np.eye(3) * np.sign(np.diag(dmri.affine[:3, :3]))
                dmri = resample_img(dmri, target_affine=target_affine, interpolation='linear')
                fdimg = f"{self.dout}/dwi_down_res_{self.down_res}.nii.gz"
                nib.save(dmri, fdimg)

            # downsample the brainmask to match the resolution of dMRI/FOD
            _, _, brain_mask = load_nifti(self.fbrainmask, return_img=True)
            brain_mask_lowres = resample_img(brain_mask, target_affine=dmri.affine, target_shape=dmri.shape[:3],
                                             interpolation='nearest')
            nib.save(brain_mask_lowres, f"{self.dout}/brain_mask_lowres.nii.gz")
            nib.save(brain_mask, f"{self.dout}/brain_mask.nii.gz")

            # run mrtrix FOD
            # note that "cwd = pre" is kinda key here because mrtirx by default will write temporary file in __file__
            # directory which does not have writting permission in singularity container, so we need to change the working directory.
            print("===> running FOD...")
            subprocess.run(f"mrconvert {fdimg} -fslgrad {fbvec} {fbval} {pre}/dwi.mif", shell=True)
            subprocess.run(f"dwi2response dhollander -scratch {self.dout} {pre}/dwi.mif {pre}/wm.txt {pre}/gm.txt {pre}/csf.txt",
                           shell=True, cwd=self.dout)
            subprocess.run(f"dwi2fod msmt_csd {pre}/dwi.mif -mask {self.dout}/brain_mask_lowres.nii.gz \
                                {pre}/wm.txt {pre}/wmfod.mif {pre}/gm.txt {pre}/gmfod.mif {pre}/csf.txt {pre}/csffod.mif",
                           shell=True)
            subprocess.run(f"mrconvert {pre}/wmfod.mif {self.dout}/wmfod.nii.gz -force", shell=True)

            fFOD = f"{self.dout}/wmfod.nii.gz"

        elif self.fFOD is not None: # fod image input
            fFOD = self.fFOD

        else:
            raise ValueError("either fdimg or fFOD should be provided!")

        # _, _, fod = load_nifti(fFOD, return_img=True)

        # fsrc_bimask
        src_bimask = (src_mask > 0).astype(np.uint32)
        fsrc_bimask = f"{pre}/src_bimask.nii.gz"
        nib.Nifti1Image(src_bimask, srcimg.affine, srcimg.header).to_filename(fsrc_bimask)

        # ftrg_bimask
        trg_bimask = (trg_seg > 0).astype(np.uint32)
        ftrg_bimask = f"{pre}/trg_bimask.nii.gz"
        nib.Nifti1Image(trg_bimask, trgimg.affine, trgimg.header).to_filename(ftrg_bimask)

        # input image for tractography FOD or dwi
        if self.tr_alg in ["iFOD2", "iFOD1", "SD_STREAM"]:  # input is FOD
            finput = fFOD
        elif self.tr_alg in ["Tensor_Prob", "Tensor_Det"]:  # input is dwi
            finput = f"{pre}/dwi.mif"
        else:
            raise ValueError(f"{self.tr_alg} alg is not found!")

        # set seed mask
        fseed_mask = fsrc_bimask

        if self.tr_dilate_src > 0:  # seed in dilated src region and not exclude surrounding streamlines
            src_mask_dilated, fsrc_mask_dilated = self.dilate_mask(fseed_mask, iter=self.tr_dilate_src)
            fseed_mask = fsrc_mask_dilated

        if self.tr_dilate_trg > 0:
            trg_bimask, ftrg_bimask = self.dilate_mask(ftrg_bimask, iter=self.tr_dilate_trg)

        # if self.tr_bbox:
        #     tha_mask_dilated, ftha_mask_dilated = self.bbox_tha_mask(margin = self.tr_bbox_margin)
        #     fseed_mask = ftha_mask_dilated
        #     # ftha_mask = ftha_mask_dilated
        #     tha_mask = tha_mask_dilated

        # if self.args.tr_excsf:
        #     excludes = f"-exclude {fcsf_mask}"
        # else:
        #     excludes = ""

        # run tractography
        excludes = ""
        subprocess.run(f"tckgen -select {self.tr_select} -algorithm {self.tr_alg} -seed_image {fseed_mask} \
                        -angle 22.5 -maxlen 250 -minlen 10 -include {fsrc_bimask} -include {ftrg_bimask} \
                        {excludes} -mask {self.fbrainmask} {finput} \
                        -output_seeds {pre}/survived_seeds.txt {pre}/tracts.tck", shell=True, cwd = self.dout)
        subprocess.run(f"mv {pre}/tracts.tck {self.dout}/tracts.tck", shell=True, cwd=self.dout)
        # Note that seed image and the -include image can be different due to dilation option.

        # if fseed_mask == self.wm_mask:
        #     subprocess.run(f"tckedit {self.dout}/tracts.tck {self.dout}/tracts_filt_tha.tck -include {ftha_mask}".split())

        print("===> finish tractography")
        # compute connectivity
        streamlines = load_tractogram(join(self.dout, "tracts.tck"), trgimg)
        streamlines = streamlines.streamlines

        fseeds = f"{pre}/survived_seeds.txt"
        seedinfo = pd.read_csv(fseeds, skiprows=1)
        seedinfo = seedinfo.iloc[:, :5]
        seedinfo = seedinfo.to_numpy()

        num_classes = len(np.unique(trg_seg)) - 1
        dms = []
        seedfroms = []
        passthros = []
        print(f"Using {self.ftrg_seg} to count fibers...")
        for i in range(num_classes):
            cortical_mask_i = trg_seg == (i + 1)
            if self.tr_dilate_trg > 0:
                cortical_mask_i = ndi.binary_dilation(cortical_mask_i, iterations=self.tr_dilate_trg)
            # stm_ind, stm_i = dtils.target(streamlines, seg.affine, cortical_mask_i)
            r = dtils.target(streamlines, trgimg.affine, cortical_mask_i)
            r = list(r)
            stm_ind = [rr[0] for rr in r]
            stm_ind = np.array(stm_ind)
            stm_i = [rr[1] for rr in r]

            print(f"{i:02d}/{num_classes}; #streamlines = {len(stm_i)}", )
            if len(stm_i) > 0:
                dm_i, seedfrom_i = self.two_type_density_map(stm_i, seedinfo[stm_ind, :], trgimg)
                dm_i, seedfrom_i = dm_i.astype(np.float32), seedfrom_i.astype(np.float32)
            else:
                dm_i, seedfrom_i = np.zeros((trgimg.shape), dtype="float32"), np.zeros((trgimg.shape), dtype="float32")

            if self.normalize_by_area:
                area = np.sum(cortical_mask_i.flatten())
                dm_i /= area
                seedfrom_i /= area

            dms.append(dm_i)
            seedfroms.append(seedfrom_i)
            passthros.append(dm_i - seedfrom_i)

        dms = np.stack(dms, axis=-1)
        seedfroms = np.stack(seedfroms, axis=-1)
        passthros = np.stack(passthros, axis=-1)

        print("writing to ", self.dout)
        if self.exp_mode:
            out_name = self.dout / Path(self.ftrg_seg).name.replace(".nii.gz", f"_denmap.nii.gz")
            nib.Nifti1Image(dms.astype(np.float32), trgimg.affine, trgimg.header).to_filename(out_name)
            out_name = self.dout / Path(self.ftrg_seg).name.replace(".nii.gz", f"_seedfrommap.nii.gz")
            nib.Nifti1Image(seedfroms.astype(np.float32), trgimg.affine, trgimg.header).to_filename(out_name)

        out_name = self.dout / Path(self.ftrg_seg).name.replace(".nii.gz", f"_passthromap.nii.gz")
        nib.Nifti1Image(passthros.astype(np.float32), trgimg.affine, trgimg.header).to_filename(out_name)

        if self.exp_mode:
            # label 0 is bg, label 1 is the voxel that has zero count, starting from label2 are connectivity info.
            a = np.concatenate([np.zeros(trgimg.shape)[..., np.newaxis], dms], axis=-1)
            parc = np.argmax(a, axis=-1) + 1
            parc[src_mask == 0] = 0
            out_name = self.dout / Path(self.ftrg_seg).name.replace(".nii.gz", f"_denmap_parc.nii.gz")
            nib.Nifti1Image(parc.astype(np.uint32), trgimg.affine, trgimg.header).to_filename(out_name)

            a = np.concatenate([np.zeros(trgimg.shape)[..., np.newaxis], seedfroms], axis=-1)
            parc = np.argmax(a, axis=-1) + 1
            parc[src_mask == 0] = 0
            out_name = self.dout / Path(self.ftrg_seg).name.replace(".nii.gz", f"_seedfrommap_parc.nii.gz")
            nib.Nifti1Image(parc.astype(np.uint32), trgimg.affine, trgimg.header).to_filename(out_name)

        a = np.concatenate([np.zeros(trgimg.shape)[..., np.newaxis], passthros], axis=-1)
        parc = np.argmax(a, axis=-1) + 1
        parc[src_mask == 0] = 0
        out_name = self.dout / Path(self.ftrg_seg).name.replace(".nii.gz", f"_passthromap_parc.nii.gz")
        nib.Nifti1Image(parc.astype(np.uint32), trgimg.affine, trgimg.header).to_filename(out_name)

    def two_type_density_map(self, streamlines, seedinfo, seg):
        affine = seg.affine
        lin_T, offset = _mapping_to_voxel(affine)

        seed_coord = seedinfo[:, 2:]

        inds = _to_voxel_coordinates(seed_coord, lin_T, offset)
        seed_from_count = np.zeros(seg.shape, 'int')
        for coord in inds:
            i, j, k = coord[0], coord[1], coord[2]
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
