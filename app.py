import streamlit as st
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
import torch
import os
import clip
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from PIL import Image

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

Base = declarative_base()
DB_PATH = "sqlite:///history.db"
engine = create_engine(DB_PATH)
Session = sessionmaker(bind=engine)
db_session = Session()

class History(Base):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    prompt = Column(String)
    image_path = Column(String)
    similarity_score = Column(Float)
    model_name = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# Save image & evaluate
def save_image_locally(image, prompt, model_name):
    folder = "generated_images"
    os.makedirs(folder, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(folder, filename)
    image.save(path)
    score = evaluate_image(prompt, path)
    entry = History(prompt=prompt, image_path=path, similarity_score=score, model_name=model_name)
    db_session.add(entry)
    db_session.commit()

# def evaluate_image(prompt, image_path):
#     image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#     text = clip.tokenize([prompt]).to(device)
#     with torch.no_grad():
#         image_features = clip_model.encode_image(image)
#         text_features = clip_model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     similarity = (image_features @ text_features.T).item()
#     return round(similarity * 100, 2)

def evaluate_image(prompt, image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item()
    score = round((similarity + 1) / 2 * 100, 2)  # Rescale [-1,1] to [0,100]
    return score


# Image generation
def generate_image(prompt, model_id):
    try:
        if "xl" in model_id.lower():  # Handle SDXL models separately
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir="./models"
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir="./models"
            )
        pipe.to(device)
    except (TypeError, ValueError, OSError):
        if "xl" in model_id.lower():
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                cache_dir="./models"
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                cache_dir="./models"
            )
        pipe.to(device)

    return pipe(prompt=prompt).images[0]


# Load gallery
def load_gallery(page, per_page=6):
    total_entries = db_session.query(History).count()
    total_pages = max((total_entries + per_page - 1) // per_page, 1)

    st.subheader(f"History Gallery (Page {page} / {total_pages})")

    offset = (page - 1) * per_page
    entries = db_session.query(History).order_by(History.timestamp.desc()).offset(offset).limit(per_page).all()

    cols = st.columns(3)
    for idx, entry in enumerate(entries):
        with cols[idx % 3]:
            st.image(entry.image_path, use_container_width=True)
            st.markdown(f"""
            **Prompt:** {entry.prompt}  
            **Model:** `{entry.model_name}`  
            **Similarity:** {entry.similarity_score}%
            """, unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1:
            if st.button("⬅️ Previous", key="prev_btn"):
                st.session_state.page -= 1
                st.rerun()
    with col3:
        if page < total_pages:
            if st.button("Next ➡️", key="next_btn"):
                st.session_state.page += 1
                st.rerun()


# Sidebar
def load_sidebar_history():
    entries = db_session.query(History).order_by(History.timestamp.desc()).limit(10).all()
    for entry in entries:
        st.sidebar.image(entry.image_path, caption=entry.prompt[:30], use_container_width=True)

if st.sidebar.button("Clear History"):
    db_session.query(History).delete()
    db_session.commit()
    st.sidebar.success("History cleared!")
    st.session_state.page = 1
    st.rerun()

load_sidebar_history()

# State defaults
if 'page' not in st.session_state:
    st.session_state.page = 1

# Main UI
st.title("Image Generation with Stable Diffusion")

current_prompt = st.text_area("Prompt", st.session_state.get('current_prompt', 'A cute kitten'))

model = st.selectbox(
    "Select the model",
    (
        "Lykon/dreamshaper-8",
        "Fictiverse/Stable_Diffusion_PaperCut_Model",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
        "stablediffusionapi/anything-v5",
        "prompthero/openjourney-v4"
    ),
)

if st.button("Generate"):
    with st.spinner("Generating image..."):
        try:
            image = generate_image(current_prompt, model)
            st.image(image, caption=current_prompt)
            save_image_locally(image, current_prompt, model_name=model)
            st.session_state.refresh_gallery = True
            st.session_state.refresh_sidebar = True
        except Exception as e:
            st.error(f"Image generation failed: {e}")
    st.session_state.page = 1  # Reset to first page on new image
    st.rerun()

# Always load gallery
load_gallery(page=st.session_state.page)
print(st.__version__)
