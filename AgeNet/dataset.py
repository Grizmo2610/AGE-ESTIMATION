import os
import shutil
import kagglehub
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from torch.utils.data import Dataset

class IMDBDataset(Dataset):
    """
    Dataset for Pretrain Stage
    """
    
    def __init__(self, 
                 root: str, 
                 data_list: pd.DataFrame, 
                 transform=None, 
                 has_label: bool = True):
        self.IMAGE_ROOT = root
        self.data = data_list
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        name = row["filename"]
        img_path = os.path.join(self.IMAGE_ROOT, name)

        image: Image.Image = Image.open(img_path).convert("RGB")

        bbox_cols: set[str] = {"x_min", "y_min", "x_max", "y_max"}
        if bbox_cols.issubset(self.data.columns):
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])

            w, h = image.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max > x_min and y_max > y_min:
                image = image.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            image = self.transform(image)

        if self.has_label:
            age = int(row["age"])

            # Ordinal label (0-119)
            label = np.zeros(120, dtype=np.float32)
            label[:age] = 1.0

            # Gender: M/F -> 0/1
            if "gender" in row:
                gender = 0 if row["gender"] == "M" else 1
            else:
                gender = int(row["filename"].split("_")[1])

            return {"image":image, "label":label, "gender": gender}

        return {"image":image}

class UTKFacesDataset(Dataset):
    """
    Dataset for test stage
    """
    
    def __init__(self, root: str, 
                 data_list: pd.DataFrame, 
                 transform=None, 
                 has_label: bool = True):
        self.IMAGE_ROOT = root
        self.data: pd.DataFrame = data_list
        self.data.columns = self.data.columns.str.lower()
        self.transform = transform
        self.has_label: bool = has_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename: str = row["filename"]
        img_path = os.path.join(self.IMAGE_ROOT, filename)

        image: Image.Image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        # ----- PREPROCESSING -----
        image = cv2.medianBlur(image, 3)

        blur = cv2.GaussianBlur(image, (0, 0), 3)
        image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        image = Image.fromarray(image)
        # --------------------------

        bbox_cols: set[str] = {"x_min", "y_min", "x_max", "y_max"}
        if bbox_cols.issubset(self.data.columns):
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])

            w, h = image.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max > x_min and y_max > y_min:
                image = image.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            image = self.transform(image)

        if self.has_label:
            age = int(row["age"])

            label = np.zeros(120, dtype=np.float32)
            label[:age] = 1.0
            gender = int(row["gender"])

            return {"image": image, "label": label, "gender": gender, "ID": row["id"] }
        if "id" in self.data.columns:
            return {"image": image, "ID": row["id"]}
        return {"image": image}

def get_data(
    source: str,
    source_type: str = "dataset",  # "dataset" | "competition"
    target_dir: str = "./data"
):
    """
    source: kaggle id
        dataset: owner/dataset-name
        competition: competition-name
    """

    if source_type == "dataset":
        path = kagglehub.dataset_download(source)
    elif source_type == "competition":
        path = kagglehub.competition_download(source)
    else:
        raise ValueError("source_type must be 'dataset' or 'competition'")

    os.makedirs(target_dir, exist_ok=True)

    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(target_dir, item)

        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    return target_dir

if __name__ == "__main__":
    DATA_FOLDER = "./data"
    
    print("Get IMDB dataset for pretrain stage")
    get_data(
        source="yuulind/imdb-clean",
        source_type="dataset",
        target_dir="./data/imdb-clean"
    )

    print("Get UTK Face dataset for Evaluate stage")
    get_data(
        source="moritzm00/utkface-cropped",
        source_type="dataset",
        target_dir="./data/utkface-cropped"
    )

    
    print("Get competition dataset for final finetune Stage")
    get_data(
        source="hamic-new-year-2026-cv-task",
        source_type="competition",
        target_dir="./data/hamic-new-year-2026-cv-task"
    )