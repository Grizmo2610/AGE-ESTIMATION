import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import numpy as np
from PIL import Image
from typing import Literal

# ---------------------------
# CORAL Head
# ---------------------------
class CoralHead(nn.Module):
    def __init__(self, in_features, K):
        super().__init__()
        self.weight = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(K))

    def forward(self, x):
        # (B, 1) + (K,) -> broadcast -> (B, K)
        return self.weight(x) + self.bias

# ---------------------------
# Full Model
# Backbone -> Gender
# Backbone -> Range
# Backbone + GenderEmbed + RangeEmbed -> Ordinal + BCE
# ---------------------------
class AgeModel(nn.Module):
    def __init__(self, max_age=120, num_ranges=9):
        super().__init__()

        # ViT backbone
        self.backbone: nn.Module = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )

        feat_dim = self.backbone.num_features

        # ---- Gender Head ----
        self.gender_head = nn.Linear(feat_dim, 1)
        self.gender_embedding = nn.Embedding(2, 16)

        # ---- Range Head ----
        self.range_head = nn.Linear(feat_dim, num_ranges)
        self.range_embedding = nn.Embedding(num_ranges, 32)

        # ---- Combined feature dim ----
        combined_dim = feat_dim + 16 + 32

        # ---- Age Heads (conditioned) ----
        self.ordinal_head = CoralHead(combined_dim, max_age)
        self.class_head = nn.Linear(combined_dim, max_age)

    def forward(self, x):
        # Backbone feature
        feat = self.backbone(x)  # (B, D)

        # ---- Gender prediction ----
        gender_logits = self.gender_head(feat)  # (B, 1)
        gender_pred = torch.argmax(
            torch.cat([1 - torch.sigmoid(gender_logits),
                       torch.sigmoid(gender_logits)], dim=1),
            dim=1
        )
        gender_embed = self.gender_embedding(gender_pred)  # (B,16)

        # ---- Range prediction ----
        range_logits = self.range_head(feat)  # (B, num_ranges)
        range_pred = torch.argmax(range_logits, dim=1)
        range_embed = self.range_embedding(range_pred)  # (B,32)

        # ---- Conditioning ----
        conditioned_feat = torch.cat(
            [feat, gender_embed, range_embed],
            dim=1
        )

        # ---- Age Heads ----
        ordinal_logits = self.ordinal_head(conditioned_feat)
        class_logits = self.class_head(conditioned_feat)

        return {
            "gender_logits": gender_logits,
            "range_logits": range_logits,
            "ordinal_logits": ordinal_logits,
            "class_logits": class_logits,
        }


class AgeNet:
    def __init__(self, 
                 model_path: str = "" # TODO: Update model Link
                 ):
        self.device: Literal['cuda'] | Literal['cpu'] =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model: AgeModel = AgeNet.load_model(model_path=model_path)
        self.model.eval()
        
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config, is_training=False)

    @torch.no_grad()
    def predict(self, src):
        
        self.model.eval()

        if isinstance(src, list):
            imgs = [self._load_img(x) for x in src]
            x = torch.stack([self.transform(i) for i in imgs]).to(self.device)
        else:
            img = self._load_img(src)
            x = self.transform(img).unsqueeze(0).to(self.device)

        out = self.model(x)
        gender_logits = out["gender_logits"]
        ordinal_logits = out["ordinal_logits"]
        class_logits = out["class_logits"]
        
        probs = torch.sigmoid(ordinal_logits)
        age_ord = probs.sum(dim=1)

        age_cls = torch.argmax(class_logits, dim=1).float()
        age = 0.8 * age_ord + 0.2 * age_cls

        gender = (torch.sigmoid(gender_logits) > 0.5).long()

        return {
            "gender": gender.cpu().numpy(),
            "age": age.cpu().numpy()
        }


    def _load_img(self, src: str | np.ndarray | Image.Image | torch.Tensor):
        if isinstance(src, str):
            return Image.open(src).convert("RGB")

        if isinstance(src, Image.Image):
            return src.convert("RGB")

        if isinstance(src, np.ndarray):
            if src.ndim == 3 and src.shape[2] == 3:
                return Image.fromarray(src.astype(np.uint8))
            raise ValueError("Invalid numpy image")

        if isinstance(src, torch.Tensor):
            if src.ndim == 3 and src.shape[0] in (1, 3):
                src = src.detach().cpu()
                if src.dtype != torch.uint8:
                    src = (src * 255).clamp(0, 255).byte()
                return Image.fromarray(src.permute(1, 2, 0).numpy())
            if src.ndim == 3 and src.shape[2] == 3:
                src = src.detach().cpu().byte()
                return Image.fromarray(src.numpy())
            raise ValueError("Invalid tensor image")

        raise TypeError("Unsupported input type")
    
    @staticmethod
    def load_model(
        pretrained:bool = True,
        model_path = "", # TODO Update model link
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ):
        MAX_AGE = 120
        model = AgeModel(MAX_AGE).to(device)

        if pretrained:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            
        return model