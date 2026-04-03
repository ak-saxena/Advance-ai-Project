import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor
from Model import CLIPMultimodalClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_NAME = "openai/clip-vit-base-patch32"
MODEL_PATH = "clip_multimodal_best.pth"

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


@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(CLIP_NAME)

    model = CLIPMultimodalClassifier(
        clip_name=CLIP_NAME,
        num_classes=6,
        dropout=0.3
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return processor, model


def predict(headline, image, processor, model):
    inputs = processor(
        text=[headline],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"]
        )

        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred_idx].item()

    return pred_idx, conf, probs[0].cpu()


st.set_page_config(page_title="Fakeddit CLIP App", layout="centered")
st.title("Fakeddit App")

headline = st.text_input("Article Headline", "Lorem Ipsum")
uploaded_file = st.file_uploader(
    "Choose an accompanying image for the article...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image Preview", use_container_width=True)

if st.button("Predict"):
    if not headline.strip():
        st.error("Please enter an article headline.")
        st.stop()

    if uploaded_file is None:
        st.error("Please upload an image.")
        st.stop()

    try:
        processor, model = load_model_and_processor()

        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        pred_idx, conf, probs = predict(headline, image, processor, model)
        pred_label = CLASS_LABELS[pred_idx]

        st.subheader("Predicted Category")
        st.success(f"{pred_label} ({conf * 100:.2f}%)")
        st.write(CLASS_DESCRIPTIONS[pred_label])

        with st.expander("Class probabilities"):
            for i, label in enumerate(CLASS_LABELS):
                st.write(f"{label}: {probs[i].item() * 100:.2f}%")

    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
    except RuntimeError as e:
        st.error("Model loading failed. This usually means your Model.py architecture does not match training.")
        st.code(str(e))
    except Exception as e:
        st.error("Unexpected error during prediction.")
        st.code(str(e))