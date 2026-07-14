from monai.data import DataLoader
from monai.networks.nets import UNet, SegResNet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import MetricReduction
import os
from pathlib import Path
from monai.inferers import sliding_window_inference
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
CHECKPOINTS_PATH = Path(r"C:\Users\mirko\med_lab\checkpoints")
MODELS_PATH = Path(r"C:\Users\mirko\med_lab\models")

def unet_model(dice_focal_loss=False):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512), 
        strides=(2, 2, 2, 2),
        num_res_units=4,
        dropout=0.2
    )

    model = model.to(DEVICE)

    
    if dice_focal_loss:
        loss_fn = DiceFocalLoss(
                    sigmoid=True,
                    lambda_dice=1.0,
                    lambda_focal=1.0
                )
    else:
        loss_fn = DiceCELoss(sigmoid=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )
    return model, loss_fn, optimizer


def segresnet_model(dice_focal_loss=False):

    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        init_filters=16,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2
    )


    model = model.to(DEVICE)

    if dice_focal_loss:

        loss_fn = DiceFocalLoss(
            sigmoid=True,
            lambda_dice=1.0,
            lambda_focal=1.0
        )

    else:

        loss_fn = DiceCELoss(sigmoid=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    return model, loss_fn, optimizer



def training_eval_loop(*,
                       train_loader: DataLoader, 
                       val_loader: DataLoader, 
                       test_loader: DataLoader,
                       model: UNet,
                       loss_fn: DiceCELoss,
                       optimizer: torch.optim.AdamW,
                       NUM_EPOCHS=50,
                       resume_checkpoint=None
                       ):

    dice_metric = DiceMetric(
        include_background=False,
        reduction=MetricReduction.MEAN
    )

    # =========================
    # BEST TRACKING
    # =========================
    best_val_dice = -1.0

    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []

    start_epoch = 0


    # =========================
    # RESUME TRAINING
    # =========================
    if resume_checkpoint is not None:

        checkpoint = torch.load(
            resume_checkpoint,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint["best_val_dice"]

        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        val_dices = checkpoint["val_dices"]

        print("\n====================")
        print("Checkpoint caricato")
        print(f"Riprendo da epoch {start_epoch}")
        print(f"Best Dice precedente: {best_val_dice:.4f}")
        print("====================\n")


    # =========================
    # TRAINING LOOP
    # =========================

    for epoch in range(start_epoch, NUM_EPOCHS):

        # =====================
        # TRAIN
        # =====================

        model.train()
        train_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(
                outputs,
                labels
            )


            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            dice_metric(y_pred=preds, y=labels)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        train_dices.append(train_dice)


        # =====================
        # VALIDATION
        # =====================

        model.eval()

        val_loss = 0.0
        dice_metric.reset()

        with torch.no_grad():

            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = sliding_window_inference(
                                    inputs=images,
                                    roi_size=(128,128,64),
                                    sw_batch_size=4,
                                    predictor=model
                                )
                loss = loss_fn(outputs,labels)

                val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                dice_metric(y_pred=preds, y=labels)



        val_loss /= len(val_loader)
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        val_losses.append(val_loss)
        val_dices.append(val_dice)



        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Dice:   {val_dice:.4f}")



        # =====================
        # SAVE BEST MODEL
        # =====================

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "train_losses": train_losses,
                    "train_dices": train_dices,
                    "val_losses": val_losses,
                    "val_dices": val_dices,
                },
                os.path.join(MODELS_PATH, "best_model.pth")
            )

            print("💾 Best model salvato!")

        # =====================
        # SAVE LAST CHECKPOINT
        # =====================

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_val_dice":
                    best_val_dice,
                "train_losses":
                    train_losses,
                "train_dices":
                    train_dices,
                "val_losses":
                    val_losses,
                "val_dices":
                    val_dices,

            },
            os.path.join(CHECKPOINTS_PATH, "last_checkpoint.pth")
        )

        print("💾 Checkpoint aggiornato!")

    # ======================================================
    # TEST
    # ======================================================

    checkpoint = torch.load(
        os.path.join(MODELS_PATH, "best_model.pth"),
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()
    test_loss = 0.0
    dice_metric.reset()

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = sliding_window_inference(
                        inputs=images,
                        roi_size=(128,128,64),
                        sw_batch_size=4,
                        predictor=model
            )


            loss = loss_fn(
                outputs,
                labels
            )

            test_loss += loss.item()


            preds = torch.sigmoid(outputs)
            preds = (
                preds > 0.5
            ).float()


            dice_metric(
                y_pred=preds,
                y=labels
            )


    test_loss /= len(test_loader)
    test_dice = dice_metric.aggregate().item()
    dice_metric.reset()



    print("\n====================")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test Dice:            {test_dice:.4f}")
    print("====================")
    
    return test_dice