import os
from PIL import Image
from torch.utils.data import Dataset
from utils.config import CSCANConfig

class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for Masoud Nickparvar Brain Tumor MRI Dataset.
    Inherits from torch.utils.data.Dataset to load brain MRI slices on-the-fly.
    
    Why it is used:
    Provides direct mapping of class directories (glioma, meningioma, pituitary, notumor)
    to integer targets and executes our custom CLAHE-augmented transform pipeline.
    It logs class frequencies during initialization, offering research visibility
    into potential data imbalance.
    """
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Path to either Train or Test directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = CSCANConfig.CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Crawl the directory and build file list
        self._build_dataset()
        
        # Log dataset statistics
        self._print_stats()

    def _build_dataset(self):
        """Crawls dataset directory and maps image paths to class indices."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
            
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"[Warning] Class directory {class_dir} does not exist.")
                continue
                
            class_idx = self.class_to_idx[class_name]
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    file_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(file_path)
                    self.labels.append(class_idx)

    def _print_stats(self):
        """Logs directory summary to console to analyze dataset structure."""
        print("-" * 50)
        print(f"Dataset Path: {self.root_dir}")
        print(f"Total samples: {len(self.image_paths)}")
        
        # Calculate counts per class
        class_counts = {cls: 0 for cls in self.classes}
        for label in self.labels:
            class_counts[self.classes[label]] += 1
            
        for cls_name, count in class_counts.items():
            print(f"  Class '{cls_name}': {count} images")
        print("-" * 50)

    def __len__(self) -> int:
        """Returns total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Retrieves, loads, and preprocesses a single MRI sample.
        
        Address MRI limitation:
        Ensures all loaded images are forced to 3-channel RGB format, even if 
        saved as grayscale, maintaining input tensor dimension consistency.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image using PIL
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            # Force conversion to RGB (even if stored as grayscale or L)
            img = img.convert('RGB')
            
        # Apply CLAHE and/or data augmentations if defined
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
