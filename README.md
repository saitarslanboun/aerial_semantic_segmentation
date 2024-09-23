# Building Segmentation from Aerial Images using Sentinel-2 Satellite Data
## Overview
This project aims to segment buildings from aerial images captured by the European Space Agency's (ESA) Sentinel-2 satellite, which offers images with a pixel resolution of 256x156 at a 10m image resolution. The primary challenge involves developing a semantic segmentation model capable of accurately identifying buildings in varying terrains and under the constraint of a relatively small dataset. The provided dataset has 232 training, and 75 validation samples.

| Dataset        | Samples |
| -------------- | ------- |
| Training       | 232     |
| Validation     | 75      |
## Challenges
1. Complex terrains pose significant difficulties in accurate building detection.
2. The limited size of the dataset challenges the model’s learning capability.
## Objective
The main objective is to enhance segmentation accuracy using pretrained models, focusing on the UNet architecture due to its proven effectiveness in similar tasks. Time constraints limited the exploration of alternative architectures, training schedules, and data augmentation techniques.
## Environment Setup
### Building and Activating Conda Environment
```bash
# Create a new conda environment
conda create --name hasans_task_submission
# Activate the new conda environment
conda activate hasans_task_submission

# Install necessary packages
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install yacs
conda install tqdm
pip install segmentation-models-pytorch
```
### Downloading Checkpoints
1. Download trained checkpoints from the [Google Drive Link](https://drive.google.com/file/d/1cCnV6z6prRFWQbPzUJkzHaNEZ2firc25/view?usp=drive_link). Decompress it, and allocate under the main `aerial_image_segmentation` folder. 
2. Download the pretrained imagenet checkpoint for HRNet from the [Google Drive Link](https://drive.google.com/file/d/1XL2Z4jEsAqAQpDsim_gyuMQ6MjYeUMyM/view?usp=drive_link). Allocate it under the `aerial_image_segmentation/models` folder.
### Running commands
```bash
python inference.py --dataset dataset.pickle --checkpoint checkpoints/unet_medium.pt # Baseline inference, refer to the 1st step
python inference.py --dataset dataset.pickle --checkpoint checkpoints/imagenet_pretrained.pt --architecture imagenet_pretrained # Refer to the 2nd step
python inference.py --dataset dataset.pickle --checkpoint checkpoints/spacenet8_pretrained.pt --architecture spacenet8_pretrained # Refer to the 3rd step
python inference.py --dataset dataset.pickle --checkpoint checkpoints/spacenet8_pretrained.pt --architecture opensentinelmap_pretrained # Refer to the 4th step

# Refer to the bonus step for the following commands
python inference.py --dataset dataset.pickle --checkpoint checkpoints/unet_tiny.pt --architecture unet_tiny # Tiny Unet Model
python inference.py --dataset dataset.pickle --checkpoint checkpoints/unet_small.pt --architecture unet_small # Small Unet Model
python inference.py --dataset dataset.pickle --checkpoint checkpoints/unet_medium.pt --architecture unet_medium # Medium Unet Model (Baseline)
python inference.py --dataset dataset.pickle --checkpoint checkpoints/unet_large.pt --architecture unet_large # Large Unet Model
```

## Thought Process and Experiments
| Experiment | Model  | Pretrained | Initial IOU | Latency (seconds/image) | Note                                         |
|------------|--------|------------|-------------|-------------------------|----------------------------------------------|
| 1          | UNet   | No         | 0.50        | 0.7                     | Baseline model                               |
| 2          | UNet++ | Yes        | 0.39        | 1.0                     | Pretrained on 900M ImageNet images           |
| 3          | HRNet  | Yes        | 0.39        | 0.7                     | Pretrained on SpaceNet challenge data        |
| 4          | UNet   | No         | 0.35        | -                       | Trained from scratch on Sentinel-2 images    |
### Initial Info
All models are trained for 300 epochs using the Adam optimizer and Binary Cross Entropy Loss, starting with a learning rate of 0.001. If no improvement is seen in 20 epochs, the learning rate is reduced by 0.1. Model latency was evaluated on a single core of Intel's 3rd generation 8 Core processors.
### 1st Step
To get started, we established a baseline using the trusty [U-Net](https://arxiv.org/pdf/1505.04597) model. Even though there are newer models out there, big contributions to the field like [OpenSentinelMap](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Johnson_OpenSentinelMap_A_Large-Scale_Land_Use_Dataset_Using_OpenStreetMap_and_Sentinel-2_CVPRW_2022_paper.pdf) and top solutions from competitions like [SpaceNet](https://spacenet.ai/challenges/) still utilizes models that are not really state-of-art. Given U-Net's solid performance and simplicity, we chose it as our starting point.

We used a U-Net with skip connections, which is great for segmenting complex scenes (it's popular in medical imaging too!). We normalized the input images by dividing pixel values by 10,000, following Sentinel-2's guidelines. The result? An Intersection over Union (IOU) of 0.50 and an average latency of about 0.7 seconds per image.
### 2nd Step
Next up, we explored how pretrained weights might boost our model's accuracy. We fine-tuned a [U-Net++](https://arxiv.org/pdf/1807.10165) model using pretrained weights from a model initially trained on 900 million mostly unlabeled ImageNet images [(more info here)](https://arxiv.org/pdf/1905.00546). We kept the normalization consistent across the dataset. This pretrained model served as the encoder in our segmentation framework. The outcome was an IOU of 0.39 and a processing latency of about 1 second per image. It gave us some insights into the trade-offs between using ImageNet pretraining and performance efficiency. Based on this single experiment in our limited time, we conclude that the usage of pretrained imagenet weights for this assignment task is not feasible.

We picked this experiment mainly because it was convenient—we used an existing [PyTorch library](https://segmentation-modelspytorch.readthedocs.io/en/latest/) with these pretrained weights. Our goal was to see how general-domain pretrained weights would affect our task.
### 3rd Step
Seeing that ImageNet pretrained weights didn't boost our IOU, we shifted gears to a pretrained HRNet model from the first-place solution of the [SpaceNet8](https://github.com/SpaceNetChallenge/SpaceNet8) challenge. We thought that using weights pretrained on data closer to our target—specifically [WorldView-2](https://earth.esa.int/eogateway/missions/worldview-2) and [WorldView-3](https://earth.esa.int/eogateway/missions/worldview-3) imagery—might help.

This HRNet model was originally trained on extensive SpaceNet datasets (phases 1-7), which included detailed building and land feature segmentations. We modified the model by adding two extra convolutional upsampling layers and fine-tuned it on our dataset, using the same normalization as before. This resulted in an IOU of 0.39 and a processing latency of about 0.7 seconds per image. The accuracy didn't improve.
### 4th Step
<figure>
  <img src="https://github.com/VisionSystemsInc/open-sentinel-map/blob/main/img/dataset_teaser.png" alt="Example Image">
  <figcaption><strong>The illustration of OpenSentinelMap Dataset (Adopted from OpenSentinelMap repository)</strong></figcaption>
</figure>
<p></p>
<p></p>
<p></p>
Since the previous approaches didn't bump up the IOU, I decided to build a U-Net model from scratch using only Sentinel-2 imagery. I wanted to see how using consistent aerial data types during training would affect the results.
I checked out several sources for Sentinel-2 images with annotations, like the [Google Open Buildings 2.5D](https://research.google/blog/open-buildings-25d-temporal-dataset-tracks-building-changes-across-the-global-south/) Temporal dataset and the new OpenSentinelMap. I chose OpenSentinelMap because it was easy to access and offered comprehensive ready-to-use pixelwise labels for features like roads, water, and buildings.

Given that 37% of the original dataset includes images of buildings, I decided to use all available labeled data. Instead of traditional semantic segmentation, I trained the model to create a label map with constant color values for each class in PNG format. This was to simplify data preprocessing and present a more complex task within the same domain.

For starters, I trained the model using a Mean Square Error loss function over 10,000 iterations. I wanted to keep it short at the beginning for experimental purposes. Then, I fine-tuned it using the same settings as before. I only wanted to see, whether pretraining on this setting would provide an incremental change. The result was an IOU of 0.35, showing that this approach didn't lead to an improvement either. If I would see an incremental change in improving IOU, I would keep pretraining for longer iterations.

## Conclusion and Future Work
Despite the various strategies employed in the initial four steps of the project, none led to an improvement over the baseline U-Net model. However, I firmly believe that the usage of the right pretrained weights with right training data is the key, and there are proven examples (such as [BEiT](https://arxiv.org/pdf/2106.08254), [MAE](https://arxiv.org/pdf/2111.06377), [DeiT](https://arxiv.org/pdf/2204.07118)) on many image domains but not on aerial images. Looking ahead, one potential avenue for future work is to explore the availability of pretrained semantic segmentation models that have been specifically trained with Sentinel-2 images from the Google Open Buildings dataset with potentially hundreds of millions of images that includes satisfactory amount of building shots in all different terrain environments ([such as](https://arxiv.org/pdf/2310.11622)). If such models are not readily available (provided by the authors), the next logical step would be to use the Google Open Buildings dataset itself to pretrain a robust foundational model focused on building segmentation tasks in plenty of different terrain settings. While experimenting with new state-of-the-art architectures is always an option, recent research in the field of transfer learning suggests that leveraging appropriate datasets might have a more substantial impact on performance than significant architectural modifications. 

## Bonus Step
As an additional bonus experiment, I examined the effect of varying the architectural complexity of the baseline U-Net model. Given the complex task of satellites, I believe, reducing the computational demands of processing this data without significantly sacrificing accuracy would provide a significant convenience for the satellite processing pipeline. To this end, I experimented with three scaled versions of the U-Net architecture by altering the dimensionality of the feature space, to measure how much we can reduce the computational need without a significant accuracy change:
| Architecture  | Feature Space Changes | IOU  | Latency (seconds/image) |
|---------------|-----------------------|------|-------------------------|
| UNet-tiny     | 4x smaller            | 0.48 | 0.06                    |
| UNet-small    | 2x smaller            | 0.50 (same!) | 0.20 (>2 times faster!)                    |
| UNet-large    | 2x larger             | 0.49 | 3.00                    |

1. UNet-tiny: This model has a quarter of the baseline's feature space, achieving an IOU of 0.48 in just about 0.06 seconds per image.
2. UNet-small: Halving the feature space of the baseline model, this variant maintains an IOU of 0.50 but processes images more than twice as fast as the baseline, at approximately 0.2 seconds per image.
3. UNet-large: Doubling the feature space of the baseline results in a slightly lower IOU of 0.49 but takes around 3 seconds per image to process.

These results suggest a promising direction for developing foundational models akin to BERT for the satellite imagery field, which could be adapted for various sub-tasks within satellite image analysis. The community's focus has traditionally been on maximizing accuracy, often at the expense of computational efficiency. However, our findings indicate that it is feasible to construct models that are both computationally efficient and highly accurate, provided that adequate data is available. As the necessary data is freely accessible, leveraging it effectively could lead to significant advancements in the field.
