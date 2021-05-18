import os
from tkinter import Image
import Dataset
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.distributions import transforms

path = 'D:\\ДИПЛОМ\Проект_1\\Снимки экрана\\'
img_path = path + 'image'


train_df = pd.read_csv(path + 'img_pixels.csv')
test_df = pd.read_csv(path + 'img_pixels_low.csv')
# sample = pd.read_csv(path + 'sample_submission.csv')

train_df['img_path'] = train_df['image_id'] + '.png'
test_df['img_path'] = test_df['image_id'] + '.png'
train_df.head()

# UNPIVOT / MELT THE LABELS
train_label = train_df.melt(id_vars=['image_id', 'img_path'])
# FILTER THE DATA
train_label = train_label[train_label['value'] == 1]
# GET THE IMAGE ID NUMBER
train_label['id'] = [int(i[1]) for i in train_label['image_id'].str.split('_')]
# RESET THE INDEX
train_label = train_label.sort_values('id').reset_index()
# ADD THE LABEL TO THE DATASET
train_df['label'] = train_label['variable']
# REFORMAT THE DATASET
train_df = train_df[train_df.columns[[0, 5, 1, 2, 3, 4, 6]]]
print(train_label.shape)
train_df.head()


from sklearn.preprocessing import LabelEncoder
# Encode the label
le = LabelEncoder()
label_encoded = le.fit_transform(train_df['label'])
train_df['label_encoded'] = label_encoded
# Taking the class name
label_names = label_encoded.classes_
train_df.head()


class PathologyPlantsDataset(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """

    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, -1]

        if self.transform:
            image = self.transform(image)

        return (image, label)


# INSTANTIATE THE OBJECT
pathology_train = PathologyPlantsDataset(
    data_frame=train_part,
    root_dir=path + 'images',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)