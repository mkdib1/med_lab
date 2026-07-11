from monai.transforms import *
import random

def build_augmented_train_transforms():
    
    augmentation_grid = {

        # CropForegroundd
        "margin": [0, 5, 10],

        # RandCropByPosNegLabeld
        "pos_neg": [(1, 1),(2, 1), (3, 1)],
        "num_samples": [2, 4, 6],

        # RandFlipd (stessa probabilità per entrambi gli assi)
        "flip_prob": [0.2, 0.4, 0.6],

        # RandAffined
        "affine_prob": [0.2,0.3, 0.4],
        "rotate_range": [(0.02, 0.02, 0.02),(0.05, 0.05, 0.05),(0.08, 0.08, 0.08)],
        "scale_range": [(0.02, 0.02, 0.02),(0.05, 0.05, 0.05), (0.08, 0.08, 0.08)],
        "translate_range": [(2, 2, 1), (5, 5, 3), (8, 8, 5)],
        
        # RandGaussianNoise
        "noise_prob": [0.2,0.3, 0.4],
        "noise_std": [0.01, 0.03, 0.05],

        # RandAdjustContrast
        "contrast_prob": [0.2,0.3, 0.4],
        "gamma": [(0.95, 1.05), (0.90, 1.10), (0.85, 1.15)]
    }

    params = {
    key: random.choice(values)
    for key, values in augmentation_grid.items()
    }

    pos, neg = params["pos_neg"]

    return params, Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear","nearest")),
        CropForegroundd(keys=["image","label"], source_key="image", margin=params["margin"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 64)),
        RandCropByPosNegLabeld(keys=["image","label"], label_key="label", spatial_size=(128,128,64), 
                               pos=pos, neg=neg,  
                               num_samples=params["num_samples"]),
        RandFlipd(keys=["image","label"], prob=params["flip_prob"], spatial_axis=0),
        RandFlipd(keys=["image","label"], prob=params["flip_prob"], spatial_axis=1),
        RandAffined(keys=["image","label"], prob=params["affine_prob"], 
                    rotate_range=params["rotate_range"], scale_range=params["scale_range"], 
                    translate_range=params["translate_range"], mode=("bilinear","nearest")),
        RandGaussianNoised(keys=["image"], prob=params["noise_prob"], mean=0.0, std=params["noise_std"]),
        RandAdjustContrastd(keys=["image"], prob=params["contrast_prob"], gamma=params["gamma"]),
        EnsureTyped(keys=["image","label"])
    ])