import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32", num_classes=6, dropout=0.3):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_name)
        proj_dim = self.clip.config.projection_dim

        self.norm = nn.LayerNorm(proj_dim * 4)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.freeze_backbone()

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

        image_feat = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_feat = F.normalize(outputs.text_embeds, p=2, dim=-1)

        abs_diff = torch.abs(image_feat - text_feat)
        elem_mul = image_feat * text_feat

        fused = torch.cat([image_feat, text_feat, abs_diff, elem_mul], dim=-1)
        fused = self.norm(fused)
        logits = self.classifier(fused)

        return logits

    def freeze_backbone(self):
        for p in self.clip.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_top_layers(self, vision_last_n=2, text_last_n=2):
        self.freeze_backbone()

        for layer in self.clip.vision_model.encoder.layers[-vision_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        for layer in self.clip.text_model.encoder.layers[-text_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True