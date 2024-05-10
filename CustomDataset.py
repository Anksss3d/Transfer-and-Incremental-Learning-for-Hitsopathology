import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from collections import Counter
import torch.nn.functional as F

imagenet_normalizer = transforms.Normalize(
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225])
)

medned_normalizer = transforms.Normalize(
    mean=torch.tensor([0.7480, 0.5932, 0.7370]),
    std=torch.tensor([0.1340, 0.1649, 0.1216])
)


class ImageDataset(Dataset):
    """
    :class:`ImageDataset` represents an image dataset that can be used for machine learning tasks. It is a subclass of :class:`torch.utils.data.Dataset`.

    **Attributes:**

    - `SIZE` (int): The size of the images after resizing.
    - `FEATURE_START_INDEX` (int): The index from which the features start in the dataset.
    - `FILE_NAME_INDEX` (int): The index of the file name column in the dataset.
    - `BRIGHTNESS` (float): The brightness factor for data augmentation.
    - `CONTRAST` (float): The contrast factor for data augmentation.
    - `SATURATION` (float): The saturation factor for data augmentation.
    - `HUE` (float): The hue factor for data augmentation.
    - `DFT` (torch.Tensor): Dummy attribute for demonstration purposes.

    **Methods:**

    - __init__(filepath, img_dir, label_index, is_val, only_dataset=None)
        Initializes the ImageDataset object.
        :param filepath: Path to the CSV file containing the dataset.
        :param img_dir: Directory containing the image files.
        :param label_index: Index of the label column in the dataset.
        :param is_val: Boolean value indicating if the dataset is a validation set.
        :param only_dataset: Optional parameter to filter the dataset by a specific value.

    - init_transforms()
        Initializes the transforms for training and validation data.
        :return: None

    - calculate_class_weights()
        Calculates class weights for imbalanced classification.
        :return: Normalized class weights.
        :rtype: torch.Tensor

    - __len__()
        Returns the length of the data stored in the object.
        :return: The length of the data.
        :rtype: int

    - __str__()
        Returns a formatted string representation of the object.
        :return: A string representing the number of rows in the dataset.

    - __getitem__(index)
        Returns the image and label corresponding to the given index.
        :param index: The index of the item to get from the dataset.
        :return: A tuple containing the image and label.

    - print_classwise_details()
        Prints the classwise details of the labels.
        :return: None

    - get_features_labels()
        Returns the labels associated with the dataset.
        :return: The labels.
        :rtype: pandas.DataFrame
    """
    SIZE = 224
    FEATURE_START_INDEX = 1
    FILE_NAME_INDEX = 0
    BRIGHTNESS = CONTRAST = SATURATION = HUE = 0.1
    DFT = torch.tensor([1])


    def __init__(self, filepath, img_dir, label_index, is_val, only_dataset=None, num_features = None):
        """
        Initialize the dataset.

        :param filepath: Path to the CSV file containing the dataset.
        :param img_dir: Directory containing the image files.
        :param label_index: Index of the label column in the dataset.
        :param is_val: Boolean value indicating if the dataset is a validation set.
        :param only_dataset: Optional parameter to filter the dataset by a specific value.
        """
        self.is_val = is_val
        self.LABEL_INDEX = label_index
        self.data = pd.read_csv(filepath)
        self.img_dir = img_dir
        if only_dataset:
            self.data = self.data[self.data.iloc[:, self.FILE_NAME_INDEX].str.contains(only_dataset)]

        self.file_names = self.data.iloc[:, self.FILE_NAME_INDEX].values.tolist()
        self.labels = self.data.iloc[:, self.LABEL_INDEX].values.tolist()
        self.init_transforms()
        self.class_count = len(set(self.labels))

        print(f"Dataset Initialized: Number of samples: {len(self.file_names)}\t Number of Classes: {self.class_count}")
        self.NUM_FEATURES = None
        if num_features:
            self.NUM_FEATURES=num_features
            self.features = self.data.iloc[:,
                            self.FEATURE_START_INDEX:self.FEATURE_START_INDEX + self.NUM_FEATURES].values.tolist()

    def init_transforms(self):
        """
        Initialize the transforms for training and validation data.

        :return: None
        """
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.SIZE),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=self.BRIGHTNESS, contrast=self.CONTRAST, saturation=self.SATURATION,
                                   hue=self.HUE),
            transforms.ToTensor(),
            medned_normalizer,
        ])
        self.validation_transform = transforms.Compose([
            transforms.CenterCrop(self.SIZE),
            transforms.ToTensor(),
            medned_normalizer
        ])


    def calculate_class_weights(self):
        """
        Calculate class weights for imbalanced classification.

        :return: Normalized class weights.
        :rtype: torch.Tensor
        """
        class_counts = Counter(self.labels)
        total_samples = sum(class_counts.values())
        weights = [total_samples / class_counts[i] for i in range(self.class_count)]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        normalized_weights = weight_tensor / weight_tensor.sum()
        return normalized_weights


    def __len__(self):
        """
        Returns the length of the data stored in the object.

        :return: The length of the data.
        :rtype: int
        """
        return len(self.data)

    def __str__(self):
        """
        Returns a formatted string representation of the object.

        :return: A string representing the number of rows in the dataset.
        """
        return f'Dataset Rows: {len(self.data)}'

    def __getitem__(self, index):
        """
        :param index: The index of the item to get from the dataset.
        :return: A tuple containing the image and label corresponding to the given index.
        """
        img = Image.open(self.file_names[index])
        img = self.validation_transform(img) if self.is_val else self.train_transform(img)
        original_label = self.labels[index]
        label = F.one_hot(torch.tensor(original_label, dtype=torch.long), num_classes=self.class_count)
        if self.NUM_FEATURES:
            features = torch.Tensor(self.features[index])
            return img, features, label.to(dtype=torch.float32)
        else:
            return img, label.to(dtype=torch.float32)

    def print_classwise_details(self):
        """
        Prints the classwise details of the labels.

        :return: None
        """
        counter = Counter(self.labels)
        print("classwise details: ", counter)
        print("Total Samples: ", len(self.labels))


    def get_features_labels(self):
        """
        Get the labels as a DataFrame.

        :return: The labels as a DataFrame.
        :rtype: pandas.DataFrame
        """
        labels = pd.DataFrame(self.labels)
        features = torch.Tensor(self.features)
        return features, labels
