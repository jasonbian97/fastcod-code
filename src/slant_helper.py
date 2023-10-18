from dipy.io.image import load_nifti, load_nifti_data, save_nifti
import numpy as np
import nibabel as nib
from os.path import join

def prepare_cortical_label(fslant, dout, return_fpath = False, return_bimask = False, include_cerebellum = False):
    # preparing cortical slant label

    seg_data, _, seg = load_nifti(fslant, return_img=True)

    # seg_data = seg.get_fdata().astype(np.float32)
    # seg.set_data_dtype(np.uint32)
    seg_shape = seg_data.shape
    frontal_lobe = [100, 101, 102, 103, 104, 105, 112, 113, 118, 119, 120, 121, 124, 125, 136, 137, 138, 139, 140,
                    141, 142, 143, 146, 147, 152, 153, 162, 163, 164, 165, 172, 173, 178, 179, 186, 187, 190, 191,
                    192, 193, 204, 205]
    # 100 Right-ACgG--anterior-cingulate-gyrus    102 178 75  0
    # 101 Left-ACgG--anterior-cingulate-gyrus     179 255 152 0
    # 102 Right-AIns--anterior-insula             106 0   0   0
    # 103 Left-AIns--anterior-insula              182 76  76  0
    # 104 Right-AOrG--anterior-orbital-gyrus      0   129 178 0
    # 105 Left-AOrG--anterior-orbital-gyrus
    # 112 Right-CO----central-operculum           178 31  0   0
    # 113 Left-CO----central-operculum
    # 118 Right-FO----frontal-operculum           0   68  178 0
    # 119 Left-FO----frontal-operculum            76  144 255 0
    # 120 Right-FRP---frontal-pole                0   0   170 0
    # 121 Left-FRP---frontal-pole


    temporal_lobe = [116, 117, 122, 123, 132, 133, 154, 155, 166, 167, 170, 171, 180, 181, 184, 185, 200, 201, 202,
                     203,
                     206, 207]
    occipital_lobe = [108, 109, 114, 115, 128, 129, 134, 135, 144, 145, 156, 157, 160, 161, 196, 197]
    parietal_lobe = [106, 107, 168, 169, 174, 175, 194, 195, 198, 199]
    # 106 Right-AnG---angular-gyrus
    # 107 Left-AnG---angular-gyrus
    # 168 Right-PCu---precuneus
    # 169 Left-PCu---precuneus
    # 174 Right-PO----parietal-operculum
    # 175 Left-PO----parietal-operculum
    # 194 Right-SMG---supramarginal-gyrus
    # 195 Left-SMG---supramarginal-gyrus
    # Right-SPL---superior-parietal-lobule
    # 199 Left-SPL---superior-parietal-lobule

    precentral_gyrus = [150, 151, 182, 183]
    # 150 Right-MPrG--precentral-gyrus
    # 151 Left-MPrG--precentral-gyrus
    # 182 Right-PrG---precentral-gyrus
    # 183 Left-PrG---precentral-gyrus

    postcentral_gyrus = [148, 149, 176, 177]
    # 148 Right-MPoG--postcentral-gyrus
    # 149 Left-MPoG--postcentral-gyrus
    # 176 Right-PoG---postcentral-gyrus
    # 177 Left-PoG---postcentral-gyrus

    csf_labels = [4, 11, 46, 49, 50, 51, 52]
    # 51  Right-Lateral-Ventricle
    # 52  Left-Lateral-Ventricle
    # 4   3rd-Ventricle
    # 11  4th-Ventricle
    # 49  Right-Inf-Lat-Vent
    # 50  Left-Inf-Lat-Vent

    R_cerebellum_labels = [38 ,40]
    L_cerebellum_labels = [39 ,41]
    ## cerebellum
    # 38  Right-Cerebellum-Exterior
    # 39  Left-Cerebellum-Exterior
    # 40  Right-Cerebellum-White-Matter
    # 41  Left-Cerebellum-White-Matter

    ## 35  Brain-Stem

    target_labels = [frontal_lobe, temporal_lobe, occipital_lobe, parietal_lobe,
                     precentral_gyrus, postcentral_gyrus]
    all_labels = frontal_lobe + temporal_lobe + occipital_lobe + parietal_lobe + precentral_gyrus + postcentral_gyrus

    if include_cerebellum:
        target_labels.append(R_cerebellum_labels)
        target_labels.append(L_cerebellum_labels)
        all_labels += R_cerebellum_labels
        all_labels += L_cerebellum_labels

    num_groups = len(target_labels)
    # Create set of target masks
    target_6_mask = np.zeros(seg_shape)
    for j in range(num_groups):
        target_mask = np.zeros(seg_shape)
        if isinstance(target_labels[j], list):
            for label in target_labels[j]:
                target_mask = np.logical_or(target_mask, (seg_data == label))
        else:
            target_mask = np.logical_or(target_mask, (seg_data == target_labels[j]))

        target_6_mask[target_mask] = j + 1

    out_name = join(dout, "target_6_mask.nii.gz")
    nib.Nifti1Image(target_6_mask.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
    out_name = join(dout, "cortical_bimask.nii.gz")
    nib.Nifti1Image((target_6_mask > 0).astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

    target_98_mask = np.zeros(seg_shape)
    for i, label in enumerate(all_labels):
        target_mask = np.zeros(seg_shape)
        target_mask = np.logical_or(target_mask, (seg_data == label))
        target_98_mask[target_mask] = i + 1

    out_name = join(dout, "target_98_mask.nii.gz")
    nib.Nifti1Image(target_98_mask.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
    # ([fcortical_bimask,ftarget_6_mask,ftarget_98_mask], [cortical_bimask,target_6_mask,target_98_mask])
    return (
        [join(dout, "cortical_bimask.nii.gz"),
         join(dout, "target_6_mask.nii.gz"),
         join(dout, "target_98_mask.nii.gz")],
        [(target_6_mask > 0).astype(np.uint32),
         target_6_mask,
         target_98_mask]
    )


def get_csf_mask(fslant, dout, return_fpath = False, return_bimask = False):
    seg_data, _, seg = load_nifti(fslant, return_img=True)
    seg_shape = seg_data.shape

    csf_labels = [4, 11, 46, 49, 50, 51, 52]

    mask = np.zeros(seg_shape)
    for i, label in enumerate(csf_labels):
        target_mask = np.zeros(seg_shape)
        target_mask = np.logical_or(target_mask, (seg_data == label))
        mask[target_mask] = i + 1

    out_name = join(dout, "mask.nii.gz")
    nib.Nifti1Image(mask.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
    out_name = join(dout, "csf_bimask.nii.gz")
    nib.Nifti1Image((mask > 0).astype(np.uint32), seg.affine, seg.header).to_filename(out_name)

    if return_fpath:
        if return_bimask:
            return join(dout, "csf_bimask.nii.gz")
        else:
            return join(dout, "csf_mask.nii.gz")
    else:
        if return_bimask:
            return (mask > 0).astype(np.uint32)
        else:
            return mask

def get_tha_mask(fslant, dout, return_fpath = False, return_bimask = False):
    seg_data, _, seg = load_nifti(fslant, return_img=True)
    mask = 1 * (seg_data == 59) + 2 * (seg_data == 60)
    out_name = join(dout, "tha_mask.nii.gz")
    nib.Nifti1Image(mask.astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
    out_name = join(dout, "tha_bimask.nii.gz")
    nib.Nifti1Image((mask > 0).astype(np.uint32), seg.affine, seg.header).to_filename(out_name)
    return join(dout, "tha_bimask.nii.gz"), join(dout, "tha_mask.nii.gz"), (mask > 0).astype(np.uint32), mask
    # if return_fpath:
    #     if return_bimask:
    #         return join(dout, "tha_bimask.nii.gz")
    #     else:
    #         return join(dout, "tha_mask.nii.gz")
    # else:
    #     if return_bimask:
    #         return (mask > 0).astype(np.uint32)
    #     else:
    #         return mask