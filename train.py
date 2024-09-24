# Import necessary libraries and modules
from DataIterator import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

import argparse
import pickle
import time
import torch.nn.functional as F
import torch.nn as nn
import os
import shutil

class IoULoss(nn.Module):
    """
    Custom loss function that calculates the Intersection over Union (IoU) loss.
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, labels):
        outputs = torch.sigmoid(outputs)  # Ensure outputs are in [0,1] range
        labels = labels.float()  # Convert labels to float for consistent operations

        # Calculate intersection and union
        intersection = (outputs * labels).sum()
        total = (outputs + labels).sum()
        union = total - intersection

        # Compute IoU loss
        IoU = (intersection + self.smooth) / (union + self.smooth)
        return 1 - IoU  # Return IoU loss

def calculate_iou(preds, labels):
    """
    Calculates the mean Intersection over Union (IoU) for a batch of predictions and labels.
    """
    # Convert predictions to binary (0 or 1) using a threshold (e.g., 0.5)
    preds = torch.sigmoid(preds) > 0.5
    labels = labels > 0.5

    # Remove channel dimension if present
    preds = preds.squeeze(1)
    labels = labels.squeeze(1)

    # Calculate intersection and union
    intersection = (preds & labels).float().sum((1, 2))  # Intersection points
    union = (preds | labels).float().sum((1, 2))         # Union points

    # Compute IoU for each image in the batch
    iou = (intersection + 1e-6) / (union + 1e-6)         # Add small epsilon to avoid division by zero
    return iou.mean()  # Return average IoU over batch

def train():
    """
    Trains the model and evaluates it on the validation set.
    """
    ious = []
    losses = []

    # Training loop
    for epoch in tqdm(range(opt.epoch), desc='Epoch Progress', leave=False):
        model.train()  # Set model to training mode
        train_loss = 0.0
        train_items = 0
        train_bar = tqdm(train_data_iter, desc=f'Epoch {epoch+1}/{opt.epoch} Training', leave=False)
        for images, masks in train_bar:
            if torch.cuda.is_available():
                images = images.cuda()
                masks = masks.cuda()

            # Forward pass
            if opt.architecture in ["unet_tiny", "unet_small", "unet_medium", "unet_large", "imagenet_pretrained",
                                    "opensentinelmap_pretrained"]:
                outputs = model(images.float())
            elif opt.architecture == "spacenet8_pretrained":
                feats = model.forward_once(images.float())
                outputs = model.building_seg_head(feats)

            loss = loss_fn(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item() * images.size(0)
            train_items += images.size(0)
            current_train_loss = train_loss / train_items

            # Update progress bar description
            train_bar.set_description(f'Epoch {epoch+1}/{opt.epoch} Training Loss: {current_train_loss:.4f}')
            train_bar.refresh()

        # Calculate average training loss
        train_loss /= len(train_data_iter.dataset)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_iou = 0.0
        val_items = 0
        val_bar = tqdm(valid_data_iter, desc=f'Epoch {epoch+1}/{opt.epoch} Validation', leave=False)
        with torch.no_grad():  # Disable gradient computation for validation
            for images, masks in val_bar:
                if torch.cuda.is_available():
                    images = images.cuda()
                    masks = masks.cuda()

                # Forward pass
                if opt.architecture in ["unet_tiny", "unet_small", "unet_medium", "unet_large", "imagenet_pretrained",
                                        "opensentinelmap_pretrained"]:
                    outputs = model(images.float())
                elif opt.architecture == "spacenet8_pretrained":
                    feats = model.forward_once(images.float())
                    outputs = model.building_seg_head(feats)

                # Calculate loss and IoU
                loss = loss_fn(outputs, masks)
                if opt.task == "train":
                    iou = calculate_iou(outputs, masks)

                # Accumulate loss and IoU
                val_loss += loss.item() * images.size(0)
                if opt.task == "train":
                    val_iou += iou.item() * images.size(0)
                val_items += images.size(0)
                current_val_loss = val_loss / val_items
                if opt.task == "train":
                    current_val_iou = val_iou / val_items

                # Update progress bar description
                description = f'Epoch {epoch+1}/{opt.epoch} Val Loss: {current_val_loss:.4f}'
                if opt.task == "train":
                    description += f', Val IOU: {current_val_iou:.4f}'
                val_bar.set_description(description)
                val_bar.refresh()

        # Calculate average validation loss and IoU
        val_loss /= len(valid_data_iter.dataset)
        val_iou /= len(valid_data_iter.dataset)
        losses.append(val_loss)
        ious.append(val_iou)

        # Update learning rate scheduler based on validation IoU
        if opt.task == "train":
            scheduler.step(val_iou)

        # Save checkpoint if condition is met (best IoU or lowest loss)
        if opt.task == "train":
            condition = val_iou == max(ious)
        elif opt.task == "pretrain":
            condition = val_loss == min(losses)
        if condition:
            torch.save(model.state_dict(), os.path.join(save_path_folder, str(epoch+1) + ".pt"))
            torch.save(model.state_dict(), os.path.join(save_path_folder, "best.pt"))

        # Print epoch summary
        if opt.task == "train":
            description += f', Max IOU: {max(ious):.4f}'
        print(description)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default="unet_medium",
                        help="unet_tiny|unet_small|unet_medium|unet_large|imagenet_pretrained|spacenet8_pretrained|opensentinelmap_pretrained")
    parser.add_argument("--spacenet8_pretrained_weight", type=str, default="models/best_building.pt",
                        help="Pretrained weights directory for spacenet8 (valid when architecture is 'spacenet8_pretrained')")
    parser.add_argument("--task", type=str, default="train",
                        help="train|pretrain (only needed for OpenSentinelMap pretraining)")
    parser.add_argument("--opensentinelmap_pretraining_dataset_path", type=str, default="../dataset",
                        help="The OpenSentinelMap dataset path for pretraining")
    parser.add_argument("--opensentinelmap_pretrained_weight", type=str, default="models/opensentinelmap_pretrained_checkpoint.pt",
                        help="The OpenSentinelMap pretraining checkpoint path (valid with 'opensentinelmap_pretrained' architecture)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--train_dataset_path", type=str, default="../dataset/dataset.pickle", help="Path to training dataset")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epochs to train")
    opt = parser.parse_args()

    # Initialize model based on the selected architecture
    if opt.architecture in ["unet_tiny", "unet_small", "unet_medium", "unet_large"]:
        from models.unet import *
        model = UNet(type=opt.architecture)
    elif opt.architecture == "imagenet_pretrained":
        import segmentation_models_pytorch as smp
        model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights="swsl")
    elif opt.architecture == "spacenet8_pretrained":
        from models.seg_hrnet_ocr import *
        from models.spacenet8_config import update_config

        cfg = update_config("models/base_spacenet.yaml")
        model = get_seg_model(cfg)
        # Load pretrained weights
        ckpt = torch.load(opt.spacenet8_pretrained_weight)
        model.load_state_dict(ckpt["model"].state_dict(), strict=False)
    elif opt.architecture == "opensentinelmap_pretrained":
        from models.unet import *
        model = UNet(type="unet_medium", n_classes=3)
        if opt.task == "train":
            # Load pretrained weights
            model.load_state_dict(torch.load(opt.opensentinelmap_pretrained_weight))
            model.outc = OutConv(64, 1)  # Adjust output layer for single-channel output

    # Load model to CUDA environment
    if torch.cuda.is_available():
        model = model.cuda()

    # Set up checkpoint directory
    save_path_folder = "checkpoints_" + opt.architecture + "_" + opt.task
    if os.path.isdir(save_path_folder):
        shutil.rmtree(save_path_folder)  # Remove existing directory
    os.mkdir(save_path_folder)

    # Initialize optimizer
    optimizer = Adam(model.parameters())

    if opt.task == "train":
        # Load training and validation data
        with open(opt.train_dataset_path, "rb") as f:
            data = pickle.load(f)
        train_data = data["train"]
        val_data = data["val"]

        # Prepare data loaders
        train_data_iter = SentinelDataset(train_data, architecture=opt.architecture)
        valid_data_iter = SentinelDataset(val_data, architecture=opt.architecture)

        # Define loss function as a combination of BCEWithLogitsLoss and IoULoss
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = IoULoss()
        loss_fn = lambda output, target: criterion1(output, target) + criterion2(output, target)

        # Prepare learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=20)
    elif opt.task == "pretrain":
        # Prepare data loaders for pretraining
        train_data_iter = OSMDataset(opt.opensentinelmap_pretraining_dataset_path, data_type="train")
        valid_data_iter = OSMDataset(opt.opensentinelmap_pretraining_dataset_path, data_type="val")

        # Define loss function for pretraining
        loss_fn = nn.MSELoss()

    # Wrap data iterators with DataLoader
    train_data_iter = DataLoader(train_data_iter, batch_size=opt.batch_size, shuffle=True)
    valid_data_iter = DataLoader(valid_data_iter, batch_size=1, shuffle=False)

    # Start training
    train()
