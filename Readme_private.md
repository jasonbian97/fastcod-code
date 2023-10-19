```shell


singularity run -e -B /mnt fastcod.sif prepare_src_trg.py --seg_type slant --fseg /mnt/ssd3/Projects/fastcod/data/mtbi_demo/slant.nii.gz --dout /mnt/ssd3/Projects/fastcod/data/mtbi_demo/conn

singularity run -e -B /mnt fastcod.sif run_ConnectivityAnalysis.py   --fdimg /mnt/ssd3/Projects/fastcod/data/mtbi_demo/DWI_proc.nii  --fbvec /mnt/ssd3/Projects/fastcod/data/mtbi_demo/DWI_proc.bvecs   --fbval /mnt/ssd3/Projects/fastcod/data/mtbi_demo/DWI_proc.bvals   --dout /mnt/ssd3/Projects/fastcod/data/mtbi_demo/conn   --fsrc_mask /mnt/ssd3/Projects/fastcod/data/mtbi_demo/conn/tha_mask.nii.gz   --ftrg_seg /mnt/ssd3/Projects/fastcod/data/mtbi_demo/conn/slant6_trg_mask.nii.gz   --fbrainmask /mnt/ssd3/Projects/fastcod/data/mtbi_demo/robex.nii.gz --tr_select 10000 --down_res 3.0 --bvec_flip z




```