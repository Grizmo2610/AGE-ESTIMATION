import os
import time

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch import Tensor

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.model_selection import train_test_split

from AgeNet import *

def fit(
    model: nn.Module, optimizer: torch.optim.Adam,
    device: torch.device, epochs: int,
    train_loader: DataLoader, val_loader: DataLoader, criterion: dict,
    gamma: float = 0.5, patience=5,
    save_paths: dict = {
        "loss":"best_loss_model.pth",
        "rmse":"best_rmse_model.pth"
        },
    history: dict[str, dict[str, list]] = init_history(),
    roots: dict[str, str] = {"model": "models", "sample": "sample"},
    best_metrics: dict = {"loss": float("inf")},
) -> dict[str, dict[str, list]]:
    start_epoch = get_latest_epoch(roots["model"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=gamma, patience=patience // 2
    )

    early_stop_counter = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        start_time: float = time.time()
        print('=' * 25 + f'Epoch {epoch + 1:02d}/{start_epoch + epochs:02d}' + '=' * 25)

        train_result: dict[str, float] = train(model, train_loader, optimizer, device, criterion)
        val_result: dict[str, float] = evaluate(model, val_loader, device, criterion)

        # ---- train history ----
        history['train']['loss'].append(train_result['loss'])
        history['train']['rmse'].append(train_result['rmse'])
        history['train']['ordinal_loss'].append(train_result['ordinal_loss'])
        history['train']['ordinal_rmse'].append(train_result['ordinal_rmse'])
        history['train']['cls_loss'].append(train_result['cls_loss'])
        history['train']['cls_rmse'].append(train_result['cls_rmse'])
        history['train']['gender_loss'].append(train_result['gender_loss'])
        history['train']['gender_acc'].append(train_result['gender_acc'])

        # ---- Validation history ----
        history['val']['loss'].append(val_result['loss'])
        history['val']['rmse'].append(val_result['rmse'])
        history['val']['ordinal_loss'].append(val_result['ordinal_loss'])
        history['val']['ordinal_rmse'].append(val_result['ordinal_rmse'])
        history['val']['cls_loss'].append(val_result['cls_loss'])
        history['val']['cls_rmse'].append(val_result['cls_rmse'])
        history['val']['gender_loss'].append(val_result['gender_loss'])
        history['val']['gender_acc'].append(val_result['gender_acc'])

        # ---- save best ----
        if save_best_models(model, val_result, epoch, save_paths, roots["model"], best_metrics):
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f'Stopping counter {early_stop_counter}/{patience}')

        save_epoch_model(model, epoch, roots["model"])

        minutes, seconds = divmod(time.time() - start_time, 60)
        print(f"Epoch time: {int(minutes):02d}:{int(seconds):02d}")

        current_lr = optimizer.param_groups[0]['lr']

        # ===== LOSS =====
        print(
            # Overal Loss
            f"Train Loss: {train_result['loss']:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | "

            # Ordinal Head loss
            f"Train Ord Loss: {train_result['ordinal_loss']:.4f} | "
            f"Val Ord Loss: {val_result['ordinal_loss']:.4f} | "

            # Classification Head loss
            f"Train Cls Loss: {train_result['cls_loss']:.4f} | "
            f"Val Cls Loss: {val_result['cls_loss']:.4f} | "

            # Gender loss
            f"Train gender loss: {train_result['gender_loss']:.4f} | "
            f"Val gender loss: {val_result['gender_loss']:.4f} | "
        )

        # ===== METRICS =====
        print(
            # Overal RMSE
            f"Train RMSE: {train_result['rmse']:.4f} | "
            f"Val RMSE: {val_result['rmse']:.4f} | "

            # Ordinal Head RMSE
            f"Train Ord RMSE: {train_result['ordinal_rmse']:.4f} | "
            f"Val Ord RMSE: {val_result['ordinal_rmse']:.4f} | "

            # Classification Head RMSE
            f"Train Cls RMSE: {train_result['cls_rmse']:.4f} | "
            f"Val Cls RMSE: {val_result['cls_rmse']:.4f} | "

            # Gender Accuracy
            f"Train Gender Acc: {train_result['gender_acc']:.4f} | "
            f"Val Gender Acc: {val_result['gender_acc']:.4f} | "
        )

        print(f"LR: {current_lr:.6f}")

        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

        scheduler.step(val_result['rmse'])

    return history

def setup_environment(config):
    seed_everything(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = os.cpu_count()

    roots = {
        "imdb_root": config["PRETRAIN_DATASET_ROOT"],
        "utk_root": config["VAL_DATASET_ROOT"],
        "competitions": config["COMPETITIONS_ROOT"],
        "model": config["MODEL_FOLDER"],
        "sample": config["SAMPLE_FOLDER"],
    }

    os.makedirs(roots["model"], exist_ok=True)
    os.makedirs(roots["sample"], exist_ok=True)

    return device, num_workers, roots

def build_model(config, device):
    model = AgeModel(config["max_age"]).to(device)

    optimizer = torch.optim.AdamW([
            {"params": model.gender_head.parameters(), "lr": 3e-4},
            {"params": model.range_head.parameters(), "lr": 3e-4},
            {"params": model.ordinal_head.parameters(), "lr": 3e-4},
            {"params": model.class_head.parameters(), "lr": 3e-4},
            {"params": model.gender_embedding.parameters(), "lr": 3e-4},
            {"params": model.range_embedding.parameters(), "lr": 3e-4},
    ], weight_decay=0.01)

    criterion = {
        "ordinal_loss": F.binary_cross_entropy_with_logits,
        "class_loss": F.cross_entropy,
        "gender_loss": F.binary_cross_entropy_with_logits
    }

    return model, optimizer, criterion

def build_transforms(model):
    config = resolve_data_config({}, model=model)

    train_transform = create_transform(**config, is_training=True)
    val_transform = create_transform(**config, is_training=False)

    return train_transform, val_transform

def build_utk_dataframe(root):
    rows = []

    for fname in os.listdir(root):
        if not fname.lower().endswith(".jpg"):
            continue

        parts = fname.split("_")
        if len(parts) < 2:
            continue

        age = int(parts[0])
        gender = "M" if int(parts[1]) == 0 else "F"

        rows.append({
            "filename": os.path.join(root, fname),
            "age": age,
            "gender": gender
        })

    return pd.DataFrame(rows)

def load_imdb_data(roots, transform):
    train_csv = os.path.join(roots["imdb_root"], "imdb_train_new_1024.csv")
    val_csv = os.path.join(roots["imdb_root"], "imdb_valid_new_1024.csv")
    test_csv = os.path.join(roots["imdb_root"], "imdb_test_new_1024.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    root = os.path.join(roots["imdb_root"], "imdb-clean-1024/imdb-clean-1024")

    train_dataset = PretrainingPhaseDataset(root, df_all, transform)
    # val_dataset = IMDBDataset(root, df_val, val_transform)
    # test_dataset = IMDBDataset(root, df_test, val_transform)

    return train_dataset

def load_utk_data(roots, transform):
    df_val = build_utk_dataframe(roots["utk_root"])
    val_dataset = PretrainingPhaseDataset(roots["utk_root"], df_val, transform)
    return val_dataset

def load_competition_data(roots, train_transform, val_transform, config, model, device, num_workers):
    train_csv = os.path.join(roots["competitions"], "dataset/train.csv")
    test_csv = os.path.join(roots["competitions"], "dataset/test.csv")

    root = os.path.join(roots["competitions"], "dataset/images")
    df_train: pd.DataFrame = pd.read_csv(train_csv)
    df_test: pd.DataFrame = pd.read_csv(test_csv)
    df_train = infer_gender_for_df(df_train, root, val_transform, config, model, device, num_workers)
    df_test = infer_gender_for_df(df_test, root, val_transform, config, model, device, num_workers)
    df_train, df_val = train_test_split(df_train, test_size=config["val_ratio"], random_state=config["seed"])


    train_dataset = CompetitionDataset(root, df_train, train_transform, has_label=True)
    val_dataset = CompetitionDataset(root, df_val, val_transform, has_label=True)
    test_dataset = CompetitionDataset(root, df_test, val_transform, has_label=False)
    return train_dataset, val_dataset, test_dataset

def build_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def generate_submission(model, test_loader):
    results = []

    for load in test_loader:
        images = load["image"]
        filenames = load["filename"]

        preds = model.predict(images)

        genders = preds["gender"]
        ages = preds["age"]

        for i in range(len(filenames)):
            results.append({
                "filename": filenames[i],
                "age": ages[i]
            })

    df = pd.DataFrame(results)
    df.to_csv("submission.csv", index=False)

def infer_gender_for_df(
    df: pd.DataFrame, 
    image_root: str, 
    transform, 
    config, 
    model: nn.Module, 
    device,
    num_workers,
): 
    
    df = df.reset_index(drop=True)
    dataset = CompetitionDataset(image_root, df, transform, has_label=False ) 
    loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=num_workers) 
    model.eval() 
    all_genders = [] 
    with torch.no_grad(): 
        for images in tqdm(loader): 
            images: Tensor = images.to(device) 
            outputs = model(images) 
            gender_logits = outputs["gender_logits"] 
            gender_pred: Tensor = (torch.sigmoid(gender_logits) > 0.5).long() 
            all_genders.extend(gender_pred.cpu().numpy().flatten()) 
        df["gender"] = all_genders 
    return df

if __name__ == "__main__":
    config = load_config("config.json")

    device, num_workers, roots = setup_environment(config)

    model, optimizer, criterion = build_model(config, device)

    train_transform, val_transform = build_transforms(model)

    train_dataset_pretrain = load_imdb_data(roots, train_transform)
    val_dataset_pretrain = load_utk_data(roots, val_transform)

    train_loader_pretrain: DataLoader = build_loader(train_dataset_pretrain, config["BATCH_SIZE"], shuffle=True,num_workers=num_workers)
    valid_loader_pretrain: DataLoader = build_loader(val_dataset_pretrain, config["BATCH_SIZE"], shuffle= False,num_workers=num_workers)

    best_metrics = {
        "loss": float("inf"),
        "rmse": float("inf")
    }
    history = fit(
        model=model,
        optimizer=optimizer,
        device=device,
        epochs=config["EPOCHS"],
        train_loader=train_loader_pretrain,
        val_loader=valid_loader_pretrain,
        criterion=criterion,
        best_metrics=best_metrics,
    )

    plot_history(history, {"plot_image": config["plot_image_path"], "history": config["history"]}, root=roots["sample"])

    train_dataset_contest, val_dataset_contest, test_dataset_contest = load_competition_data(
        roots,
        train_transform=train_transform,
        val_transform=val_transform,
        config=config,
        model=model, 
        device=device,
        num_workers=num_workers
    )

    train_loader_contest = build_loader(train_dataset_contest, config["BATCH_SIZE"], shuffle=True,num_workers=num_workers)
    valid_loader_contest = build_loader(val_dataset_contest, config["BATCH_SIZE"], shuffle=False,num_workers=num_workers)
    test_loader_contest = build_loader(test_dataset_contest, config["BATCH_SIZE"], shuffle=False,num_workers=num_workers)

    checkpoint = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW([
        {"params": model.gender_head.parameters(), "lr": 3e-4},
        {"params": model.range_head.parameters(), "lr": 3e-4},
        {"params": model.ordinal_head.parameters(), "lr": 3e-4},
        {"params": model.class_head.parameters(), "lr": 3e-4},
        {"params": model.gender_embedding.parameters(), "lr": 3e-4},
        {"params": model.range_embedding.parameters(), "lr": 3e-4},
    ], weight_decay=0.01)
    
    history_contest = fit(
        model=model, 
        optimizer=optimizer, 
        device=device, 
        epochs=5,
        criterion=criterion, 
        train_loader=train_loader_contest, 
        val_loader=valid_loader_contest
    )
    
    for p in model.backbone.blocks[-3:].parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.blocks[-3:].parameters(), "lr": 1e-5},

        {"params": model.gender_head.parameters(), "lr": 3e-4},
        {"params": model.range_head.parameters(), "lr": 3e-4},
        {"params": model.ordinal_head.parameters(), "lr": 3e-4},
        {"params": model.class_head.parameters(), "lr": 3e-4},
        {"params": model.gender_embedding.parameters(), "lr": 3e-4},
        {"params": model.range_embedding.parameters(), "lr": 3e-4},
    ], weight_decay=0.01)
    
    history_contest = fit(
        model=model, optimizer=optimizer, 
        device=device, epochs=20,criterion=criterion, 
        train_loader=train_loader_contest, val_loader=valid_loader_contest, 
        best_metrics=best_metrics, history=history_contest
    )


    merged_dataset = ConcatDataset([train_dataset_contest, val_dataset_contest])
    train_loader_contest = build_loader(merged_dataset, config["BATCH_SIZE"], shuffle=True,num_workers=num_workers)
    

    for p in model.backbone.parameters():
        p.requires_grad = True

    history_contest = fit(
        model=model,
        optimizer=optimizer, 
        device=device, 
        epochs=20,
        criterion=criterion, 
        train_loader=train_loader_contest, 
        val_loader=valid_loader_pretrain, 
        best_metrics=best_metrics, 
        history=history_contest
    )
    
    plot_history(history_contest, {"plot_image": "Contest_history.png", "history": config["history_path"]}, root=roots["sample"])
    
    generate_submission(model, test_loader_contest)