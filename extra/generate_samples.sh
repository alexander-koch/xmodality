#!/bin/bash
src="../modality_data/registered/040_Warped.nii.gz"

mkdir -p extra/samples
#python3 resampled_inference.py --input $src --output extra/samples/sample_unet.nii.gz --load weights/unet.pkl --arch unet --disable_diffusion --bfloat16
python3 resampled_inference.py --input $src --output extra/samples/sample_adm.nii.gz --load weights/adm.pkl --arch adm --batch_size 32 --bfloat16
python3 resampled_inference.py --input $src --output extra/samples/sample_uvit.nii.gz --load weights/uvit.pkl --arch uvit --bfloat16
python3 resampled_inference.py --input $src --output extra/samples/sample_dit.nii.gz --load weights/dit.pkl --arch dit --bfloat16
