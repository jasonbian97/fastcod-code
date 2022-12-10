# FastCod

# Quick Start
1. set up IO path. There are two ways to do that:
   1. modify the `config.yaml` file in the `conf` folder.
   2. assign the path in the command line. For example, see step 2.

your input folder `din` should have following structure, which is similiar to Human Connectome Project dataset:
```shell
├── 991267
│   ├── T1w
│   │   ├── 991267 # freesurfer's recon-all output folder
│   │   ├── aparc.a2009s+aseg.nii.gz # Destrieux atlas-based cortical parcellation
│   │   ├── aparc+aseg.nii.gz # Desikan-Killiany atlas-based cortical parcellation
│   │   ├── brainmask_fs.nii.gz
          │   ├── Diffusion # diffusion image folder
          │   │   ├── bvals # bval file
          │   │   ├── bvecs # bvec file
          │   │   ├── data.nii.gz # dwi data (should already be eddy and distortion corrected)
          │   │   └── nodif_brain_mask.nii.gz

```
Note: It's okay to have extra irrelavant files here and there than specified above. The code will just ignore them.

2. run the command in the terminal:
```shell
# din = data input, dout = data output, subid = subject id
python src/ConnectivityAnalysis.py io.din=HCP/991267 io.dout=data/991267/run1 io.subid=991267
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