import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="AI Room Style Transfer", page_icon="🎨", layout="centered")
st.title("🎨 AI Room Style Transfer")
st.markdown("Transform your room into a work of art using neural style transfer.")
st.divider()

STYLES = {
    "Van Gogh — Starry Night": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    "Hokusai — The Great Wave": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/1280px-Tsunami_by_hokusai_19th_century.jpg",
    "Picasso — Les Demoiselles": "https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Les_Demoiselles_d%27Avignon.jpg/600px-Les_Demoiselles_d%27Avignon.jpg",
}

@st.cache_resource
def load_vgg():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    content_layers = ['block5_conv2']
    style_extractor = tf.keras.Model([vgg.input], [vgg.get_layer(n).output for n in style_layers])
    content_extractor = tf.keras.Model([vgg.input], [vgg.get_layer(n).output for n in content_layers])
    return style_extractor, content_extractor, style_layers, content_layers

def load_img(img, max_dim=400):
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img)).convert('RGB')
    elif isinstance(img, Image.Image):
        img = img.convert('RGB')
    scale = max_dim / max(img.size)
    new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
    img = img.resize(new_size, Image.LANCZOS)
    img = np.array(img, dtype=np.float32) / 255.0
    return tf.constant(img[np.newaxis, :])

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor[0])

def gram_matrix(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    return result / tf.cast(x.shape[1]*x.shape[2], tf.float32)

def get_style(extractor, img, style_layers):
    img = tf.keras.applications.vgg19.preprocess_input(img * 255)
    return [gram_matrix(o) for o in extractor(img)]

def get_content(extractor, img):
    img = tf.keras.applications.vgg19.preprocess_input(img * 255)
    return extractor(img)

def run_style_transfer(content_img, style_img, steps=600, style_weight=1e-1, content_weight=1e4):
    style_extractor, content_extractor, style_layers, _ = load_vgg()
    
    target_style = get_style(style_extractor, style_img, style_layers)
    target_content = get_content(content_extractor, content_img)
    
    generated = tf.Variable(content_img)
    opt = tf.optimizers.Adam(learning_rate=0.01)
    
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            gen_style = get_style(style_extractor, generated, style_layers)
            gen_content = get_content(content_extractor, generated)
            s_loss = tf.add_n([tf.reduce_mean((gs-ts)**2) for gs,ts in zip(gen_style, target_style)])
            c_loss = tf.reduce_mean((gen_content[0] - target_content[0])**2)
            loss = style_weight * s_loss + content_weight * c_loss
        grad = tape.gradient(loss, generated)
        opt.apply_gradients([(grad, generated)])
        generated.assign(tf.clip_by_value(generated, 0.0, 1.0))
        return loss
    
    progress = st.progress(0, text="Starting style transfer...")
    for i in range(steps):
        step()
        if i % 60 == 0:
            progress.progress(int(i/steps*100), text=f"Applying style... {int(i/steps*100)}%")
    progress.progress(100, text="Done!")
    return tensor_to_image(generated)

# ── UI ──
content_file = st.file_uploader("Upload your room photo", type=["jpg","jpeg","png"])

st.subheader("Choose a style")
style_choice = st.radio("Preset styles", list(STYLES.keys()))
custom_style = st.file_uploader("Or upload your own style image", type=["jpg","jpeg","png"])

if content_file and st.button("✨ Apply Style Transfer", type="primary"):
    content_img = Image.open(content_file)
    
    if custom_style:
        style_img_pil = Image.open(custom_style)
    else:
        import requests
        r = requests.get(STYLES[style_choice], headers={'User-Agent': 'Mozilla/5.0'})
        style_img_pil = Image.open(io.BytesIO(r.content))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(content_img, caption="Your Room", use_container_width=True)
    with col2:
        st.image(style_img_pil, caption="Style Reference", use_container_width=True)
    
    st.divider()
    
    content_tensor = load_img(content_img)
    style_tensor = load_img(style_img_pil)
    
    with st.spinner("Loading model..."):
        load_vgg()
    
    result = run_style_transfer(content_tensor, style_tensor)
    
    st.image(result, caption="Styled Room", use_container_width=True)
    
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button("Download Result", buf.getvalue(), "styled_room.png", "image/png")
