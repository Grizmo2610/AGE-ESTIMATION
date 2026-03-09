from typing import Callable
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


def train(
    model: nn.Module, 
    train_loader: DataLoader, 
    optimizer: torch.optim.Adam, 
    device: torch.device, 
    criterion: dict[str, Callable[..., Tensor]]
) -> dict[str, float]:
    
    model.train()

    total_loss = 0
    total_ordinal_loss = 0
    total_cls_loss = 0
    total_gender_loss = 0

    total_se = 0
    total_se_ord = 0
    total_se_cls = 0
    total_samples = 0

    total_gender_correct = 0

    ordinal_loss_fn: Callable[..., Tensor] = criterion["ordinal_loss"]
    class_loss_fn: Callable[..., Tensor] = criterion["class_loss"]
    gender_loss_fn: Callable[..., Tensor] = criterion["gender_loss"]

    loops:tqdm[Tensor] = tqdm(train_loader, desc="Training", leave=False)

    for loop in loops:
        images: Tensor = loop["image"].to(device)
        labels: Tensor = loop["label"].float().to(device)
        gender: Tensor = loop["gender"].float().unsqueeze(1).to(device)

        age_true:Tensor = labels.sum(dim=1)

        optimizer.zero_grad()

        outputs = model(images)
        gender_logits: Tensor = outputs["gender_logits"]
        ordinal_logits:Tensor = outputs["ordinal_logits"]
        class_logits: Tensor = outputs["class_logits"]

        loss_ord:Tensor = ordinal_loss_fn(ordinal_logits, labels)
        loss_cls:Tensor = class_loss_fn(class_logits, age_true.long())
        loss_gen:Tensor = gender_loss_fn(gender_logits, gender)

        loss: Tensor = 0.7 * loss_ord + 0.2 * loss_cls + 0.1 * loss_gen
        loss.backward()
        optimizer.step()

        batch_size: int = images.size(0)

        total_loss += loss.item() * batch_size
        total_ordinal_loss += loss_ord.item() * batch_size
        total_cls_loss += loss_cls.item() * batch_size
        total_gender_loss += loss_gen.item() * batch_size

        with torch.no_grad():
            probs = torch.sigmoid(ordinal_logits)
            age_pred_ord = probs.sum(dim=1)
            age_pred_cls = torch.argmax(class_logits, dim=1).float()
            age_pred_overal = 0.8 * age_pred_ord + 0.2 * age_pred_cls

            total_se += torch.sum((age_pred_overal - age_true) ** 2).item()
            total_se_ord += torch.sum((age_pred_ord - age_true) ** 2).item()
            total_se_cls += torch.sum((age_pred_cls - age_true) ** 2).item()

            gender_pred = (torch.sigmoid(gender_logits) > 0.5).float()
            total_gender_correct += (gender_pred == gender).sum().item()

            total_samples += batch_size

        loops.set_postfix(
            loss=loss.item(),
            ord_loss=loss_ord.item(),
            cls_loss=loss_cls.item(),
            gen_loss=loss_gen.item(),
        )

    n = len(train_loader.dataset)

    avg_loss: float = total_loss / n
    avg_ordinal_loss = total_ordinal_loss / n
    avg_cls_loss = total_cls_loss / n
    avg_gender_loss = total_gender_loss / n

    rmse = (total_se / total_samples) ** 0.5
    ordinal_rmse = (total_se_ord / total_samples) ** 0.5
    cls_rmse = (total_se_cls / total_samples) ** 0.5
    gender_acc = total_gender_correct / total_samples

    return {
        "loss": avg_loss,
        "rmse": rmse,
        "ordinal_loss": avg_ordinal_loss,
        "ordinal_rmse": ordinal_rmse,
        "cls_loss": avg_cls_loss,
        "cls_rmse": cls_rmse,
        "gender_loss": avg_gender_loss,
        "gender_acc": gender_acc,
    }

if __name__ == "__main__":
    ""
    # TODO Update train sample script
    
    # train_result = train(model, train_loader, optimizer, device, criterion)