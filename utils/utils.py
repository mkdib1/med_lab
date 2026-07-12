from utils.augmentation import build_augmented_train_transforms
from collections import Counter
import numpy as np
from pathlib import Path
import nibabel as nib
from typing import Sequence, Union
from matplotlib import pyplot as plt
from monai.data import DataLoader, SmartCacheDataset
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Compose
import pandas as pd
from sklearn.model_selection import train_test_split
import torch



def t1_masks_path_extractor(
    data_dir: Path
    ) -> tuple[list[str], list[str]]:

    image_paths = []
    mask_paths = []

    for t1_file in data_dir.rglob("*T1w.nii.gz"):
        mask_file = next(
            t1_file.parent.glob("*lesion_mask.nii.gz"),
            None
        )

        if mask_file is not None:
            image_paths.append(str(t1_file))
            mask_paths.append(str(mask_file))

    if len(image_paths) > 0 and len(mask_paths) > 0:
        print("'image paths' and 'mask paths' lists properly created")

        if len(image_paths) != len(mask_paths):
            print("Warning: different number between images and masks")

    else:
        print("No file was found!")

    return image_paths, mask_paths


def reorder_imgs_to_canonical_space(
    *,
    image_paths: Sequence[str],
    mask_paths: Sequence[str],
) -> tuple[list[nib.nifti1.Nifti1Image], 
           list[nib.nifti1.Nifti1Image]]:
    
    images = []
    masks = []

    for img_path, msk_path in zip(image_paths, mask_paths):

        img = nib.load(img_path)
        msk = nib.load(msk_path)

        if nib.aff2axcodes(img.affine) != ("R", "A", "S"):
            img = nib.as_closest_canonical(img)

        if nib.aff2axcodes(msk.affine) != ("R", "A", "S"):
            msk = nib.as_closest_canonical(msk)

        images.append(img)
        masks.append(msk)

    return images, masks

def plot_orient_distr(image_paths: Sequence[str]):
    
    affine_lst = []
    for path in image_paths:
        affine_lst.append(nib.aff2axcodes(nib.load(path).affine))

    counts = Counter(affine_lst)
    counts = pd.Series(affine_lst).value_counts()

    counts.plot(kind='bar')

    plt.xlabel("Orientation")
    plt.ylabel("Number of brain volumes")
    plt.title("MRI orientation distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return counts.rename_axis("orientation").reset_index(name="count")


def shape_mismatch_indeces(images: Sequence[nib.Nifti1Image],
                           masks: Sequence[nib.Nifti1Image]
) -> list[int]:
    
    def check_shape_match(img: nib.Nifti1Image, 
                      mask: nib.Nifti1Image) -> bool:
        return img.shape == mask.shape

    idx_list = []
    for i in range(len(images)-1):
        if not check_shape_match(images[i], masks[i]):
            print(f"img/mask couple of index {i} has different shapes:\n img shape: {images[i].shape} \t mask shape: {masks[i].shape}")
            idx_list.append(i)

    return idx_list

def show_overlay(
    t1ImmData: Union[np.ndarray, MetaTensor],
    maskData: Union[np.ndarray, MetaTensor],
    slice_idx=None
):

    if slice_idx is None:
        slice_idx = t1ImmData.shape[2] // 2

    plt.figure(figsize=(6,6))

    plt.imshow(t1ImmData[:,slice_idx,:], cmap='Greys_r', origin="lower")
    plt.imshow(maskData[:,slice_idx,:], cmap="autumn", alpha=0.5, origin="lower")

    plt.title("T1 + Lesion overlay")
    plt.axis("off")
    plt.show()  


def build_train_val_test_loaders(data: list[dict[str, str]],
                                 base_transforms: Compose,
                                 train_augm_transforms: Compose = None):

    train_files, temp_files = train_test_split(data, test_size=0.20, random_state=42,shuffle=True)
    val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42, shuffle=True)

    params = None
    if train_augm_transforms is None:
        params, train_augm_transforms = build_augmented_train_transforms()

    train_data = SmartCacheDataset(data=train_files, transform=train_augm_transforms, replace_rate=0.8)
    val_data = SmartCacheDataset(data=val_files, transform=base_transforms, replace_rate=0.8)
    test_data = SmartCacheDataset(data=test_files, transform=base_transforms, replace_rate=0.8)



    return (
         params,
         DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()),
         DataLoader(val_data, batch_size=1, shuffle=False,num_workers=0, pin_memory=torch.cuda.is_available()),
         DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
   )


def show_loss_eval_metric_history(history_path: str):

    history = torch.load(history_path)

    train_losses = history["train_losses"]
    val_losses = history["val_losses"]

    train_dices = history["train_dices"]
    val_dices = history["val_dices"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss plot
    axes[0].plot(
        range(1, len(train_losses)+1),
        train_losses,
        label="Train Loss"
    )

    axes[0].plot(
        range(1, len(val_losses)+1),
        val_losses,
        label="Validation Loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss history")
    axes[0].legend()
    axes[0].grid(True)


    # Dice plot
    axes[1].plot(
        range(1, len(train_dices)+1),
        train_dices,
        label="Train DiceCE"
    )

    axes[1].plot(
        range(1, len(val_dices)+1),
        val_dices,
        label="Validation DiceCE"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("DiceCE score")
    axes[1].set_title("DiceCE history")
    axes[1].legend()
    axes[1].grid(True)


    plt.tight_layout()
    plt.show()

        