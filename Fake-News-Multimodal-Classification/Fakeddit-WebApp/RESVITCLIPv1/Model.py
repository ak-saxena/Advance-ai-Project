import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
from transformers import BertModel, BertTokenizer
from transformers import ViTModel, CLIPModel
import torch.nn.functional as F
class BERTResNetClassifier(nn.Module):
    def __init__(self, num_classes=6):

        super(BERTResNetClassifier, self).__init__()

        self.num_classes = num_classes

        # ResNet)
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Fully Connected Layer
        self.fc_image = nn.Linear(in_features=1000, out_features=num_classes, bias=True)

        # Dropout layer
        self.drop = nn.Dropout(p=0.3)

        # Text processing using the 768-dimensional BERT arrays
        self.text_model = BertModel.from_pretrained("bert-base-uncased")

        # Text processing Fully Connected Layer
        self.fc_text = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=num_classes, bias=True)

        # Fusion and classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, text_input_ids, text_attention_mask,):
        # Image branch
        x_img = self.image_model(image)
        x_img = self.drop(x_img)
        x_img = self.fc_image(x_img)

        # Text branch
        x_text_last_hidden_states = self.text_model(
            input_ids = text_input_ids,
            attention_mask = text_attention_mask,
            return_dict=False
        )
        x_text_pooled_output = x_text_last_hidden_states[0][:, 0, :]
        x_text = self.drop(x_text_pooled_output)
        x_text = self.fc_text(x_text_pooled_output)

        # Fusion and max merge
        x = torch.max(x_text, x_img)

        # Classification
        #x = self.softmax(x) #-> already applied in crossentropy loss

        return x

class BERTViTClassifierV22(nn.Module):
    def __init__(
        self,
        num_classes=6,
        bert_name="bert-base-uncased",
        vit_name="google/vit-base-patch16-224-in21k",
        dropout=0.4
    ):
        super().__init__()

        self.text_model = BertModel.from_pretrained(bert_name)
        self.image_model = ViTModel.from_pretrained(vit_name)

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_model.config.hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout)
        )

        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.config.hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_outputs.last_hidden_state[:, 0, :]

        image_outputs = self.image_model(
            pixel_values=pixel_values
        )
        image_feat = image_outputs.last_hidden_state[:, 0, :]

        text_feat = self.text_proj(text_feat)
        image_feat = self.image_proj(image_feat)

        fused = torch.cat([text_feat, image_feat], dim=1)
        logits = self.classifier(fused)
        return logits

class CLIPMultimodalClassifier(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32", num_classes=6, dropout=0.2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        proj_dim = self.clip.config.projection_dim

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

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
        logits = self.classifier(fused)
        return logits