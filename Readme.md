# FastCod

This repo is the official implementation of paper: [FastCod: Fast Brain Connectivity in Diffusion Imaging](https://arxiv-export3.library.cornell.edu/abs/2302.09247)
This paper won the best student paper runner-up award in SPIE-MI 2023!

FastCod can efficiently compute the diffusion-based connectivity feature between brain regions. It is a open-source tool and written in python. If you use diffusion-weighted MRI and tractography in your research or clinical practice, this tool can possibly save you lots of time!

Key features:

- Over 30x speedup than traditional method on computing connectivity
- Run from command line, server-friendly
- Easy to work with MRtrix3, fsl, and freesurfer
- Flexible pipeline design: you can flexibly run one, several, or whole pipline: dMRI preprocessing, tractography, align with anatomical images, computing connectivity
- Super-resolve ability: you can get high-res connectity features even if you are dealt with low resolution dMRI (e.g., 2mm)
- Multiple visualization tools for QA

[Under Construction] For tutorial, blogs, and more information, please visit our [project page](https://jasonbian97.github.io/fastcod/)

# How to use

There are currently two modes that you can run FastCod, MODE 1 assumes you organize your data in a specific
structure,
and MODE 2 is more flexible but requires you to specify the input images (such as FOD, source mask, target mask,
brain mask, etc).

## MODE 3

1. set up IO path. There are two options (either one is fine):
   1. modify the `config.yaml` file in the `conf` folder.
   2. assign the path in the command line. For example, see step 2.

your input folder `din` should have the following structure, which is similiar to Human Connectome Project dataset:
```shell
├── 991267
│   ├── T1w
│   │   ├── 991267 # freesurfer's recon-all output folder
│   │   ├── aparc.a2009s+aseg.nii.gz # Destrieux atlas-based cortical parcellation
│   │   ├── aparc+aseg.nii.gz # Desikan-Killiany atlas-based cortical parcellation
│   │   ├── brainmask_fs.nii.gz
│   │   ├── Diffusion # diffusion image folder
│   │   │   ├── bvals # bval file
│   │   │   ├── bvecs # bvec file
│   │   │   ├── data.nii.gz # dwi data (should already be eddy and distortion corrected)
│   │   │   └── nodif_brain_mask.nii.gz

```
Note: It's okay to have extra irrelavant files here and there than specified above. The code will just ignore them.

2. run the command in the terminal:
```shell
# din = data input, dout = data output, subid = subject id
python src/ConnectivityAnalysisHCP.py io.din=HCP/991267 io.dout=data/991267/run1 io.subid=991267
```
3. check the results in the `io.dout` folder. If you run this successfully, you should be able to see following files:
```shell
data/
└── 991267
    └── run1
        ├── cache
        ├── cortical_bimask.nii.gz
        ├── debug.log
        ├── Desikan_lut.csv
        ├── Desikan_passthromap_fsparc.nii.gz # can visualize with Freesurfer:freeview and load Freesurfer's lookup table: FreeSurferColorLUT.txt
        ├── Desikan_passthromap_fsparc_rgb.nii.gz
        ├── Desikan_passthromap.nii.gz # fiber density map
        ├── Desikan_passthromap_parc.nii.gz # the parcellation results
        ├── Desikan_target_parc.nii.gz # the target cortical regions for parcellation
        ├── tha_bimask.nii.gz
        ├── tha_mask.nii.gz
        └── tracts.tck # tractography results

```

## MODE 1

Step 1 Prepare src ROI and target ROI.
```shell
python src/prepare_src_trg.py --seg_type slant --fseg data/mtbi_demo/slant.nii.gz --dout data/mtbi_demo/conn
```
Step 2 run connectivity analysis.
```shell
python src/run_ConnectivityAnalysis.py \
--fdimg data/mtbi_demo/DWI_proc.nii \
  --fbvec data/mtbi_demo/DWI_proc.bvecs \
  --fbval data/mtbi_demo/DWI_proc.bvals \
  --dout data/mtbi_demo/conn \
  --fsrc_mask data/mtbi_demo/conn/tha_mask.nii.gz \
  --ftrg_seg data/mtbi_demo/conn/slant6_trg_mask.nii.gz \
  --fbrainmask data/mtbi_demo/robex.nii.gz
  [options]
```

Or you have already has your FOD image, you can do
```shell 
python src/run_ConnectivityAnalysis.py \
  --fFOD data/mtbi_demo/conn/wmfod.nii.gz \
  --dout data/mtbi_demo/conn \
  --fsrc_mask data/mtbi_demo/conn/tha_mask.nii.gz \
  --ftrg_seg data/mtbi_demo/conn/slant6_trg_mask.nii.gz \
  --fbrainmask data/mtbi_demo/robex.nii.gz
  [options]
```



## MODE 2 [under construction]
This mode assumes you have already has your own:
- FOD image, which will be used for tractography
- source mask, which is the mask of the source region where seeds will be placed
- target parcellation, which is the parcellation of the target region (e.g. Cortex) where fibers will be terminated.

I haven't been able to work on this mode yet. But it should not be very hard to customize your own. For example, take a look at the code 
[snippet](https://github.com/jasonbian97/fastcod-code/blob/f26c2ebabaa2344f4490d707ab1c050c21608dc3/src/ConnectivityAnalysis.py#L254)
where I start to compute connectivity.

# How to cite
If you find this is useful for your research, please cite our paper:

```bibtex
@article{bian2023fastcod,
  title={FastCod: Fast Brain Connectivity in Diffusion Imaging},
  author={Bian, Zhangxing and Shao, Muhan and Zhuo, Jiachen and Gullapalli, Rao P and Carass, Aaron and Prince, Jerry L},
  journal={arXiv preprint arXiv:2302.09247},
  year={2023}
}
```

# Contact
If you have any questions, please contact me at jasonbian.zx@gmail.com