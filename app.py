import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# ============================================================================
# MODEL DEFINITION
# ============================================================================


class ResNetEmotionDetector(nn.Module):
    """ResNet50-based emotion detector"""

    def __init__(self, num_emotions=7, pretrained=False):
        super(ResNetEmotionDetector, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_emotions)

    def forward(self, x):
        return self.resnet(x)


# ============================================================================
# GRAD-CAM FUNCTIONS
# ============================================================================


def setup_grad_cam(model, target_layer):
    """Initialize Grad-CAM"""
    grad_cam = GradCAM(model=model, target_layers=[target_layer])
    return grad_cam


def generate_gradcam(model, image, target_layer, emotion_names):
    """Generate Grad-CAM visualization"""
    # Preprocess image
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(image).unsqueeze(0)

    # Get device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    img_tensor = img_tensor.to(device)
    model = model.to(device)
    model.eval()

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        pred_emotion = emotion_names[pred_idx]
        confidence = probs[pred_idx].item()

    # Generate Grad-CAM
    grad_cam = setup_grad_cam(model, target_layer)
    targets = [ClassifierOutputTarget(pred_idx)]

    try:
        cam = grad_cam(input_tensor=img_tensor, targets=targets)
        cam = cam[0, :]
    finally:
        if hasattr(grad_cam, "activations_and_grads"):
            grad_cam.activations_and_grads.release()
        del grad_cam

    # Prepare visualization
    img_array = np.array(image.resize((256, 256))) / 255.0
    visualization = show_cam_on_image(img_array, cam, use_rgb=True)

    # Get all emotion probabilities
    all_probs = {emotion_names[i]: probs[i].item() for i in range(len(emotion_names))}

    return pred_emotion, confidence, cam, visualization, all_probs


# ============================================================================
# EMOTION INSIGHTS (Your commentary for each emotion)
# ============================================================================

EMOTION_INSIGHTS = {
    "happy": {
        "title": "😊 Happy",
        "color": "#FFD700",
        "description": """
        When detecting **happy** emotions, the model consistently focuses on the **mouth region**, 
        particularly the smile. The Grad-CAM heatmaps show strong activation around the lips and 
        sometimes the cheeks (where smile lines form).
        
        **Key observations:**
        - ✓ Primary focus: Mouth and smile
        - ✓ Secondary focus: Eyes (crinkles indicating genuine smile)
        - ✓ High confidence when both mouth and eye regions show happiness
        
        This aligns with how humans recognize happiness - we naturally look at smiles!
        """,
    },
    "sad": {
        "title": "😢 Sad",
        "color": "#4169E1",
        "description": """
        For **sad** emotions, the model distributes attention between the **mouth** (downturned) 
        and **eyes** (drooping or tearful). The heatmaps often show activation on both regions simultaneously.
        
        **Key observations:**
        - ✓ Dual focus: Downturned mouth corners + drooping eyes
        - ✓ Sometimes focuses on eyebrows (furrowed in sadness)
        - ✓ More diffuse attention than happiness (sadness is subtle)
        
        Sadness is one of the harder emotions to detect due to its subtlety.
        """,
    },
    "angry": {
        "title": "😠 Angry",
        "color": "#DC143C",
        "description": """
        The model detects **anger** primarily through the **eyebrows and eye region**. 
        Furrowed brows and intense eye expressions create strong activation patterns.
        
        **Key observations:**
        - ✓ Primary focus: Furrowed eyebrows and eyes
        - ✓ Secondary focus: Tense mouth/jaw
        - ✓ Strong, concentrated heatmap on upper face
        
        The model learned that anger manifests most clearly in the eye region - matching human perception.
        """,
    },
    "surprised": {
        "title": "😮 Surprised",
        "color": "#FF69B4",
        "description": """
        **Surprise** detection shows the clearest pattern - the model focuses almost exclusively on 
        **wide-open eyes** and **raised eyebrows**. The mouth (often open in surprise) receives secondary attention.
        
        **Key observations:**
        - ✓ Very strong focus: Wide eyes and raised eyebrows
        - ✓ Secondary focus: Open mouth (O-shape)
        - ✓ Most consistent attention pattern across examples
        
        This is one of the easiest emotions for the model to detect due to distinctive facial features.
        """,
    },
    "fearful": {
        "title": "😨 Fearful",
        "color": "#9370DB",
        "description": """
        **Fear** shows similar patterns to surprise (wide eyes) but with additional tension around 
        the mouth and overall face. The model looks at both **eyes** and **mouth** simultaneously.
        
        **Key observations:**
        - ✓ Wide eyes (similar to surprise)
        - ✓ Tense mouth and jaw
        - ✓ Sometimes confused with surprise due to similar eye patterns
        
        Fear and surprise share facial features, making them harder to distinguish.
        """,
    },
    "disgusted": {
        "title": "🤢 Disgusted",
        "color": "#228B22",
        "description": """
        The model detects **disgust** primarily through the **nose and mouth region** - 
        the wrinkled nose and raised upper lip are distinctive features.
        
        **Key observations:**
        - ✓ Primary focus: Nose wrinkle and upper lip
        - ✓ Secondary focus: Narrowed eyes
        - ✓ Concentrated attention on mid-face region
        
        Disgust has a unique facial signature that the model learned to recognize.
        """,
    },
    "neutral": {
        "title": "😐 Neutral",
        "color": "#808080",
        "description": """
        **Neutral** expressions show the most **distributed attention** - no single feature dominates. 
        The model looks across the entire face without strong focus on any particular region.
        
        **Key observations:**
        - ✓ Diffuse, spread-out attention pattern
        - ✓ No strong activation on any single feature
        - ✓ Lower confidence than other emotions
        
        Neutral is challenging because it's defined by the *absence* of emotional features rather than presence.
        """,
    },
}

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================


def show_home_page(model, target_layer, emotion_names):
    """Main homepage with emotion buttons and upload"""

    st.title("🔍 Explaining Facial Emotion Recognition with Grad-CAM")
    st.markdown(
        """
    This interactive app demonstrates how **Grad-CAM** (Gradient-weighted Class Activation Mapping) 
    reveals which facial features a ResNet50 model uses to predict emotions.
    """
    )

    # Emotion buttons right under subtitle
    st.markdown("**Explore model behavior by emotion:**")

    # Create 4 columns for buttons (first row)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("😊 Happy", use_container_width=True):
            st.session_state.page = "happy"
            st.rerun()

    with col2:
        if st.button("😢 Sad", use_container_width=True):
            st.session_state.page = "sad"
            st.rerun()

    with col3:
        if st.button("😠 Angry", use_container_width=True):
            st.session_state.page = "angry"
            st.rerun()

    with col4:
        if st.button("😮 Surprised", use_container_width=True):
            st.session_state.page = "surprised"
            st.rerun()

    # Second row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if st.button("😨 Fearful", use_container_width=True):
            st.session_state.page = "fearful"
            st.rerun()

    with col6:
        if st.button("🤢 Disgusted", use_container_width=True):
            st.session_state.page = "disgusted"
            st.rerun()

    with col7:
        if st.button("😐 Neutral", use_container_width=True):
            st.session_state.page = "neutral"
            st.rerun()

    # Continue with original layout
    st.markdown(
        """
    **Upload a face image** to see:
    - 🎯 Predicted emotion and confidence
    - 🔥 Grad-CAM heatmap showing important regions
    - 📊 Probability distribution across all emotions
    """
    )

    # Original two-column layout for upload and results
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a facial image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear frontal face image",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # Add cropping instructions
            st.info(
                "💡 **Tip:** Crop the image to focus on the face for better results!"
            )

            # Use Streamlit's image editor with crop tool
            edited_image = st.image_editor(
                image, width=400, crop_shape="rect", key="image_editor"
            )

            # Convert edited image back to PIL Image if it was cropped
            if edited_image is not None:
                display_image = Image.fromarray(edited_image)
            else:
                display_image = image

            # Generate button
            if st.button(
                "🔍 Analyze Emotion", type="primary", use_container_width=True
            ):
                with st.spinner("Analyzing facial features..."):
                    pred_emotion, confidence, cam, visualization, all_probs = (
                        generate_gradcam(
                            model, display_image, target_layer, emotion_names
                        )
                    )

                    # Store results in session state
                    st.session_state["upload_results"] = {
                        "pred_emotion": pred_emotion,
                        "confidence": confidence,
                        "cam": cam,
                        "visualization": visualization,
                        "all_probs": all_probs,
                        "original_image": display_image,
                    }

    with col_right:
        st.header("📊 Results")

        if "upload_results" in st.session_state:
            results = st.session_state["upload_results"]

            st.subheader("Predicted Emotion")
            st.markdown(f"## **{results['pred_emotion'].upper()}**")

            # Color code confidence
            if results["confidence"] > 0.8:
                color = "🟢"
            elif results["confidence"] > 0.5:
                color = "🟡"
            else:
                color = "🔴"

            st.markdown(f"### Confidence: {color} **{results['confidence']:.1%}**")

            # Probability distribution
            st.subheader("Probability Distribution")
            sorted_probs = sorted(
                results["all_probs"].items(), key=lambda x: x[1], reverse=True
            )

            for emotion, prob in sorted_probs:
                st.progress(prob, text=f"{emotion}: {prob:.1%}")
        else:
            st.info("👆 Upload an image and click 'Analyze Emotion' to see results")

    # Show visualization if results exist
    if "upload_results" in st.session_state:
        st.markdown("---")
        st.header("🔥 Grad-CAM Visualization")
        st.markdown(
            """
        The heatmap shows which regions of the face the model focused on to make its prediction.
        - **Red/Yellow**: High importance (model looked here)
        - **Blue**: Low importance (model ignored this)
        """
        )

        results = st.session_state["upload_results"]

        vis_col1, vis_col2, vis_col3 = st.columns(3)

        with vis_col1:
            st.image(
                results["original_image"],
                caption="Original Image",
                use_container_width=True,
            )

        with vis_col2:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(results["cam"], cmap="jet")
            ax.axis("off")
            ax.set_title("Grad-CAM Heatmap", fontsize=14, pad=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close()

        with vis_col3:
            st.image(
                results["visualization"], caption="Overlay", use_container_width=True
            )

        # Interpretation section
        st.markdown("---")
        st.header("💡 Interpretation")

        with st.expander("ℹ️ What does this mean?", expanded=True):
            st.markdown(
                f"""
            The model predicted **{results['pred_emotion']}** with **{results['confidence']:.1%}** confidence.
            
            **Key observations:**
            - The red/yellow regions show where the model "looked" to make this decision
            - For emotions like **happy**, the model typically focuses on the mouth (smile)
            - For emotions like **surprised**, the model focuses on the eyes (wide open)
            - Scattered attention (no clear focus) often indicates low confidence or potential misclassification
            
            **Why this matters for XAI:**
            - Reveals whether the model uses sensible features (like humans do)
            - Helps identify when the model might be wrong (scattered attention)
            - Builds trust by showing the reasoning process
            """
            )


def show_emotion_page(emotion_name, model, target_layer, emotion_names):
    """Individual emotion exploration page"""

    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # Emotion header with color
    emotion_info = EMOTION_INSIGHTS[emotion_name]
    st.markdown(
        f"<h1 style='color: {emotion_info['color']}'>{emotion_info['title']}</h1>",
        unsafe_allow_html=True,
    )

    # Commentary section
    st.markdown("### 📝 Model Behavior Analysis")
    st.markdown(emotion_info["description"])

    st.markdown("---")

    # Load and display examples
    examples_dir = os.path.join("viz_examples", emotion_name)

    if not os.path.exists(examples_dir):
        st.warning(
            f"""
        No examples found for **{emotion_name}**.
        
        Please create a `viz_examples/{emotion_name}/` folder and add example images.
        """
        )
        return

    image_files = [
        f
        for f in os.listdir(examples_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        st.warning(f"No images found in viz_examples/{emotion_name}/")
        return

    st.subheader(f"📸 Examples from Dataset")
    st.markdown(
        f"Showing **{len(image_files)}** example(s) of **{emotion_name}** emotion"
    )

    # Display each example
    for i, img_file in enumerate(image_files, 1):
        st.markdown(f"### Example {i}")

        img_path = os.path.join(examples_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        # Generate Grad-CAM
        with st.spinner(f"Generating visualization {i}..."):
            pred_emotion, confidence, cam, visualization, all_probs = generate_gradcam(
                model, image, target_layer, emotion_names
            )

        # Display results in 4 columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image(image, caption="Original", use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cam, cmap="jet")
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=80)
            buf.seek(0)
            st.image(buf, caption="Heatmap", use_container_width=True)
            plt.close()

        with col3:
            st.image(visualization, caption="Overlay", use_container_width=True)

        with col4:
            # Color code confidence
            if confidence > 0.8:
                color = "🟢"
            elif confidence > 0.5:
                color = "🟡"
            else:
                color = "🔴"

            st.metric("Predicted", pred_emotion.capitalize())
            st.metric("Confidence", f"{color} {confidence:.1%}")

            if pred_emotion == emotion_name:
                st.success("✓ Correct")
            else:
                st.error(f"✗ Predicted: {pred_emotion}")

        st.markdown("---")


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.set_page_config(
        page_title="Emotion Recognition Explainability", page_icon="😊", layout="wide"
    )

    # Emotion mapping
    emotion_names = [
        "surprised",
        "fearful",
        "disgusted",
        "happy",
        "sad",
        "angry",
        "neutral",
    ]

    # Sidebar - Model loading
    st.sidebar.header("⚙️ Model Configuration")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="resnet50_emotions.pth",
        help="Path to your trained model weights (.pth file)",
    )

    # Load model
    @st.cache_resource
    def load_model(model_path):
        model = ResNetEmotionDetector(num_emotions=7, pretrained=False)

        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                st.sidebar.success("✅ Model loaded!")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
        else:
            st.sidebar.warning("⚠️ Model file not found")

        return model

    model = load_model(model_path)
    target_layer = model.resnet.layer4[-1]

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Route to appropriate page
    if st.session_state.page == "home":
        show_home_page(model, target_layer, emotion_names)
    elif st.session_state.page in emotion_names:
        show_emotion_page(st.session_state.page, model, target_layer, emotion_names)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### 📚 About
    **Explainable AI for Emotion Recognition**
    
    - Model: ResNet50 with transfer learning
    - Dataset: RAF-DB
    - Technique: Grad-CAM
    
    Created for AIPI590 Explainable AI.
    """
    )


if __name__ == "__main__":
    main()
