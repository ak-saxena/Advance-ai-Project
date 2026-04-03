import streamlit as st
import torch
from collections import Counter
from PIL import Image
from torchvision.transforms import v2
from transformers import BertTokenizer, ViTImageProcessor, CLIPProcessor

from Model import BERTResNetClassifier, BERTViTClassifierV22, CLIPMultimodalClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = [
    "TRUE",
    "SATIRE",
    "FALSE CONNECTION",
    "IMPOSTER CONTENT",
    "MANIPULATED CONTENT",
    "MISLEADING CONTENT"
]

CLASS_DESCRIPTIONS = {
    "TRUE": "Content that is factually accurate and based on verified information.",
    "SATIRE": "Content created for humorous or satirical purposes, often not meant to be taken seriously.",
    "FALSE CONNECTION": "Headlines or visuals that mislead or do not accurately reflect the content of the article.",
    "IMPOSTER CONTENT": "Content that impersonates genuine sources, such as fake news articles mimicking legitimate media outlets.",
    "MANIPULATED CONTENT": "Content that has been altered, edited, or manipulated to mislead or distort facts.",
    "MISLEADING CONTENT": "Content that uses selective information or framing to mislead the audience, often by omitting context."
}

BERT_MODEL_NAME = "bert-base-uncased"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def get_bert_inputs(text, tokenizer):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        max_length=80,
        truncation=True,
        padding="max_length"
    )
    return inputs["input_ids"], inputs["attention_mask"]


def get_resnet_transform():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])


def extract_logits(output):
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        return next(iter(output.values()))
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def decode_prediction(logits):
    logits = extract_logits(logits)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(logits, dim=1).item()
    pred_label = CLASS_LABELS[pred_idx]
    confidence = probs[0, pred_idx].item()

    return {
        "index": pred_idx,
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probs[0].detach().cpu()
    }


def load_checkpoint_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_resources():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    bert_resnet_model = load_checkpoint_weights(
        BERTResNetClassifier(),
        "m1.pth"
    )

    bert_vit_model = load_checkpoint_weights(
        BERTViTClassifierV22(),
        "bert_vit_v2_2_best_score.pth"
    )

    clip_model = load_checkpoint_weights(
        CLIPMultimodalClassifier(),
        "clip_multimodal_bestv1.pth"
    )

    return tokenizer, vit_processor, clip_processor, bert_resnet_model, bert_vit_model, clip_model


def predict_bert_resnet(model, image, text, tokenizer, transform, device):
    input_ids, attention_mask = get_bert_inputs(text, tokenizer)

    img_tensor = transform(image).unsqueeze(0).to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(
            image=img_tensor,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask
        )

    return decode_prediction(logits)


def predict_bert_vit(model, image, text, tokenizer, vit_processor, device):
    input_ids, attention_mask = get_bert_inputs(text, tokenizer)
    vit_inputs = vit_processor(images=image, return_tensors="pt")

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = vit_inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )

    return decode_prediction(logits)


def predict_clip(model, image, text, clip_processor, device):
    clip_inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}

    with torch.no_grad():
        logits = model(
            input_ids=clip_inputs["input_ids"],
            attention_mask=clip_inputs["attention_mask"],
            pixel_values=clip_inputs["pixel_values"]
        )

    return decode_prediction(logits)


def majority_vote(pred_indices):
    counts = Counter(pred_indices)
    winner, votes = counts.most_common(1)[0]

    if votes >= 2:
        return winner, votes
    return None, 0


def show_result_block(title, result):
    st.subheader(title)
    st.write(f"**Predicted Category:** {result['label']}")
    st.write(f"**Confidence:** {result['confidence'] * 100:.2f}%")
    st.write(CLASS_DESCRIPTIONS[result["label"]])

    with st.expander(f"{title} class probabilities"):
        for i, label in enumerate(CLASS_LABELS):
            st.write(f"{label}: {result['probabilities'][i].item() * 100:.2f}%")


st.set_page_config(page_title="Fakeddit App", layout="wide")
st.title("Fakeddit App")

uploaded_title = st.text_input("Article Headline", "Lorem Ipsum")
uploaded_file = st.file_uploader(
    "Choose an accompanying image for the article...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    uploaded_file.seek(0)
    preview_image = Image.open(uploaded_file).convert("RGB")
    st.image(preview_image, caption="Uploaded Image Preview", use_container_width=True)

if st.button("Predict"):
    if not uploaded_title.strip():
        st.warning("Please enter an article headline.")
        st.stop()

    if uploaded_file is None:
        st.warning("Please upload an image.")
        st.stop()

    with st.spinner("Loading models and running inference..."):
        tokenizer, vit_processor, clip_processor, bert_resnet_model, bert_vit_model, clip_model = load_resources()

        uploaded_file.seek(0)
        input_image = Image.open(uploaded_file).convert("RGB")

        resnet_result = predict_bert_resnet(
            bert_resnet_model,
            input_image,
            uploaded_title,
            tokenizer,
            get_resnet_transform(),
            DEVICE
        )

        vit_result = predict_bert_vit(
            bert_vit_model,
            input_image,
            uploaded_title,
            tokenizer,
            vit_processor,
            DEVICE
        )

        clip_result = predict_clip(
            clip_model,
            input_image,
            uploaded_title,
            clip_processor,
            DEVICE
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_result_block("BERT + ResNet50", resnet_result)

    with col2:
        show_result_block("BERT + ViT", vit_result)

    with col3:
        show_result_block("CLIP", clip_result)

    st.divider()
    st.subheader("Project-level Majority Vote")

    voted_idx, votes = majority_vote([
        resnet_result["index"],
        vit_result["index"],
        clip_result["index"]
    ])

    if voted_idx is None:
        st.warning("No majority vote. All three models predicted different classes.")
    else:
        voted_label = CLASS_LABELS[voted_idx]
        st.success(f"Majority Vote Result: {voted_label} ({votes}/3 models)")
        st.write(CLASS_DESCRIPTIONS[voted_label])