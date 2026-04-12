from torch.utils.data.dataset import Dataset
import csv
from PIL import Image
import local_configuration


class ImageDataset(Dataset):
    def __init__(self, transform, num_images=-1):
        super().__init__()
        csv_path = local_configuration.ILSVRC2020_val_VT_csv_path
        img_path = local_configuration.ILSVRC2020_val_VT_images_path
        rdr = csv.reader(open(csv_path, "r"))
        self.img_names = []
        self.label = []
        i = 0
        for filename, label in rdr:
            i += 1
            if i == 1:
                continue
            self.img_names.append((img_path / filename).as_posix())
            self.label.append(label)
        self.transform = transform
        self.num_images = num_images

    def __len__(self):
        return len(self.label) if self.num_images==-1 else self.num_images

    def __getitem__(self, index):
        img = Image.open(self.img_names[index])
        label = self.label[index]
        img_T = self.transform(img)
        return img_T, (int(label) - 1)  # label index of csv file is [1, 1000]

    def get_item_by_label(self, label):
        index = self.label.index(str(label+1))
        return self.__getitem__(index)

def create_dataset(transform, num_images=-1):
    return ImageDataset(transform, num_images)
