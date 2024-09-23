import torch
import torchvision

class SentinelDataset(torch.utils.data.Dataset):
    def __init__(self, data, architecture):
        self.data = data
        self.architecture=architecture
        self.spacenet_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.249566, 0.318912, 0.21801], std=[0.12903, 0.11784, 0.10739])])
        self.imagenet_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        if self.architecture in ["spacenet8_pretrained", "imagenet_pretrained"]:
            image = image[:, :, ::-1]
        mask = self.data[idx]['mask']

        # Clip and normalize image values
        image = image / 10000.0

        if self.architecture == "spacenet8_pretrained":
            image = self.spacenet_transform(image)
        elif self.architecture == "imagenet_pretrained":
            image = self.imagenet_transform(image)
        else:
            image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
