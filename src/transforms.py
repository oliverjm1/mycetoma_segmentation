import torchvision.transforms.v2 as transforms
from PIL import Image

class ImageMaskTransforms:
    def __init__(self):
        # Define the transformations for both image and mask
        self.shared_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(600, 800), scale=(0.9, 1.0)),
        ])

        # Define color jitter for image
        self.image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

    def __call__(self, image, mask):
        # Apply the shared transformations
        image, mask = self.shared_transforms(image, mask)
        
        # Apply image-only transformations
        image = self.image_transforms(image)
        
        return image, mask