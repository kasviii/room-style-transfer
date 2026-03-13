# 🎨 AI Room Style Transfer (VGG19)

A neural style transfer app that transforms room photos into artwork using VGG19.

## How It Works
Upload a room photo, pick an artistic style (Van Gogh, Hokusai, Picasso), and the app applies the style's textures and patterns to your room while preserving the original layout and structure.

## Model
- Architecture: VGG19 (pretrained on ImageNet)
- Method: Neural Style Transfer (Gatys et al., 2015)
- Style layers: block1–block5 conv1
- Content layer: block5_conv2
- Optimizer: Adam (lr=0.01, 600 steps)

## Styles Available
- Van Gogh — Starry Night
- Hokusai — The Great Wave
- Picasso — Les Demoiselles d'Avignon
- Custom upload — bring your own style image

## Deployment Note
This app is **not currently deployed** since Streamlit Community Cloud has a memory limit of 1GB in the free-tier. VGG19 + TensorFlow exceeds this limit at inference time. A lightweight MobileNet version is available at: (https://github.com/kasviii/room-style-transfer-lite)

## Stack
- Python, TensorFlow, Streamlit
- VGG19 pretrained weights (ImageNet)

## Sample Result
Neural style transfer working correctly — tested locally and on Google Colab (T4 GPU).
