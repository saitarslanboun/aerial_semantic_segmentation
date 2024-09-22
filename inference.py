# Import necessary libraries
from models.unet import UNet
from DataIterator import *
from models.seg_hrnet_ocr import *
from models.spacenet8_config import update_config

import torch
import argparse
import tqdm
import pickle
import time
import segmentation_models_pytorch as smp

def calculate_iou(preds, labels): 
    # Convert predictions to binary (0 or 1) using a threshold (e.g., 0.5)
    preds = torch.sigmoid(preds) > 0.5 
    labels = labels > 0.5

    intersection = (preds & labels).float().sum((2, 3))  # Intersection points
    union = (preds | labels).float().sum((2, 3))         # Union points

    iou = (intersection + 1e-6) / (union + 1e-6)         # Add small epsilon to avoid division by zero
    return iou.mean()  # Average over batch

def infer():
    val_iou = 0.0
    val_items = 0
    total_time = 0.0  # For measuring latency

    val_bar = tqdm.tqdm(val_loader, desc='Processing batches', leave=False)
    with torch.no_grad():
        for images, masks in val_bar:
            start_time = time.time()

            # Forward pass
            if opt.architecture == "spacenet8_winner":
                feats = model.forward_once(images.float())
                outputs = model.building_seg_head(feats)
            else:   
                outputs = model(images.float())

            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time

            # Calculate IOU
            iou = calculate_iou(outputs, masks)

            val_iou += iou.item() * images.size(0)
            val_items += images.size(0)

    # Calculate average validation IOU
    val_iou /= len(val_loader.dataset)

    # Print results
    print(f'Validation IOU: {round(val_iou, 2):.2f}')
    print(f'Inference latency: {total_time:.2f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset.pickle", help="dataset pickle file")
    parser.add_argument("--checkpoint", type=str, default="unet_medium.pt", help="pretrained checkpoint state dictionary")
    parser.add_argument("--architecture", type=str, default="unet_medium",
                        help="unet_tiny|unet_small|unet_medium|unet_large|imagenet_tretrained|spacenet8_winner|opensentinelmap_pretrained")
    opt = parser.parse_args()

    # Load model architecture
    if opt.architecture=="spacenet8_winner":
        cfg = update_config("models/base_spacenet.yaml")
        model = get_seg_model(cfg)
    elif opt.architecture=="imagenet_pretrained":
        model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights="swsl")
    elif opt.architecture=="opensentinelmap_pretrained":
        model = UNet(type="unet_medium")
    else:   
        model = UNet(type=opt.architecture)

    # Load the trained weights
    model.load_state_dict(torch.load(opt.checkpoint, map_location=torch.device('cpu')))

    # Set the model to evaluation mode
    model.eval()

    # Load data
    with open(opt.dataset_path, 'rb') as f:
        data = pickle.load(f)
    train_data = data['train']
    val_data = data['val']

    # Prepare data iterator
    val_dataset = SentinelDataset(val_data, opt.architecture)

    # Data loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # Start inference
    infer()

