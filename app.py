# Claude Sonnet 4.5 was queried on 11/23/2025 to generate this code. I started with a basic Stremlit app and then iterated to add more features. Commentary and text sections were written by me.
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
        - Primary focus: Mouth and smile
        - Secondary focus: Eyes (possibly looking for eye crinkles indicating smile)
        
        This aligns with how humans recognize happiness, we look at smiles! I think its nice that the model recognizes smile lines as well.
        """,
    },
    "sad": {
        "title": "😢 Sad",
        "color": "#4169E1",
        "description": """
        For **sad** emotions, the model distributes attention between the **mouth** (downturned) 
        and **eyes** (drooping or tearful). The heatmaps often show activation on both regions simultaneously.
        
        **Key observations:**
        - Dual focus: Downturned mouth corners + drooping eyes
        - Sometimes focuses on eyebrows (furrowed in sadness)
        - More diffuse attention than happiness
        
        Sadness is one of the harder emotions to detect due to its subtlety. How we express sadness also varies as we grow older, with babies and children often crying more visibly than adults.
        """,
    },
    "angry": {
        "title": "😠 Angry",
        "color": "#DC143C",
        "description": """
        The model detects **anger** primarily through the **eyebrows and eye region, and scrunched noses**. 
        Furrowed brows and intense eye expressions create strong activation patterns.
        
        **Key observations:**
        - Primary focus: Furrowed eyebrows and eyes
        - Secondary focus: Tense mouth/jaw or open mouth
        - Strong, concentrated heatmap on upper face
        
        The model learned that anger manifests most clearly in the eyebrow area.
        """,
    },
    "surprised": {
        "title": "😮 Surprised",
        "color": "#FF69B4",
        "description": """
        **Surprise** detection shows the clearest pattern - the model focuses almost exclusively on 
        **wide-open eyes** and **raised eyebrows**. The mouth (often open in surprise) receives secondary attention.
        
        **Key observations:**
        - Very strong focus: Wide eyes and raised eyebrows
        - Secondary focus: Open mouth (O-shape)
        - Most consistent attention pattern across examples
        
        This is one of the easiest emotions for the model to detect due to distinctive facial features of wide eyes and an open mouth.
        """,
    },
    "fearful": {
        "title": "😨 Fearful",
        "color": "#9370DB",
        "description": """
        **Fear** shows similar patterns to surprise (wide eyes) but with additional tension around 
        the mouth and overall face. The model looks at both **eyes** and **mouth** simultaneously.
        
        **Key observations:**
        - Wide eyes (similar to surprise)
        - Tense mouth and jaw
        - Sometimes confused with surprise due to similar eye patterns
        
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
        - Primary focus: Nose wrinkle and upper lip
        - Secondary focus: Narrowed eyes
        - Concentrated attention on mid-face region
        
        Disgust has a unique facial signature of a scrunched nose that the model learned to recognize.
        """,
    },
    "neutral": {
        "title": "😐 Neutral",
        "color": "#808080",
        "description": """
        **Neutral** expressions show the most **distributed attention** - no single feature dominates. 
        The model looks across the entire face without strong focus on any particular region.
        
        **Key observations:**
        - Diffuse, spread-out attention pattern
        - No strong activation on any single feature
        - Lower confidence than other emotions
        
        Neutral is difficult because it's defined by the *absence* of emotional features rather than presence, so we would expect not incredibly strong focus in the model.
        """,
    },
}

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================


def show_home_page(model, target_layer, emotion_names):
    """Main homepage with emotion buttons and upload"""

    st.title("🔍 Explaining Facial Emotion Recognition with Grad-CAM 🔍")

    # About Model button right under title

    if st.button("Project Details", use_container_width=True, type="secondary"):
        st.session_state.page = "about"
        st.rerun()

    st.markdown(
        """
    This interactive app demonstrates how **Grad-CAM** reveals which facial features a ResNet50 model uses to predict emotions.
    """
    )
    # Emotion buttons right under subtitle
    st.markdown("**Explore model behavior by emotion:**")

    # Create 4 columns for buttons (first row)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("😊\nHappy", use_container_width=True):
            st.session_state.page = "happy"
            st.rerun()

    with col2:
        if st.button("😢\nSad", use_container_width=True):
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

    st.markdown("---")

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

            st.info(
                "💡 **Tip:** For best results, crop your image to focus on the face before uploading!"
            )
            st.image(image, caption="Uploaded Image", use_container_width=True)

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
            - The blue areas show where the model paid little attention
            - Scattered attention (no clear focus) often indicates low confidence or potential misclassification
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


def show_about_page():
    """About Model & Dataset page"""

    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    # Project info
    st.markdown("---")
    st.title("ℹ️ About This Project")

    st.markdown(
        """
    This webapp was developed as a final project for AIPI590, Explainable AI, with Dr. Bent, at Duke University.
    
    **Project Goals:**
    Emotion recognition AI is increasingly deployed in hiring, healthcare, education, and law enforcement—high-stakes domains where mistakes have serious consequences. Yet most models are "black boxes" that provide no explanation for their predictions. Even state-of-the-art systems like Meta's DeepFace achieve impressive accuracy, but their internal decision-making processes remain opaque. When a model predicts someone is "angry" or "fearful," we have no insight into whether it's focusing on relevant facial features.

    This project uses **Grad-CAM** to visualize what a ResNet50 emotion classifier is actually looking at when making decisions. By revealing the model's reasoning process, we can:
    - **Build trust** through transparency
    - **Detect biases** by seeing what features the model relies on
    - **Debug failures** by identifying when and why predictions go wrong
    - **Ensure fairness** by auditing whether the model uses appropriate facial features
    
    **Future Work:**
    Originally, I had intended the focus of this project to be on the differences in accuracy of emotion detection models across different demographic groups (racial, gender, age), inspired in part by my reading of "Unmasking AI" by Joy Buolamwini this semester. However, as I dove deeper into this project, I encountered challenges in finding datasets with the kind of demographic annotations needed to perform the analysis I was interested in. Many popular emotion recognition datasets I considered lack detailed demographic information (FER-2013, RAF-DB), or the demographic datasets I found lacked emotion labeling (UTKFace, FairFace). While completing this project, I considered doing the manual labeling of demographics on the RAF-DB dataset, but I thought this would be innaccurate as race and gender was often hard for me to discern and could lead to me injecting my own implicit biases into the labeling. In the future, expanding this project will require accessing or building a dataset with reliable demographic annotations, enabling a more rigorous investigation into model performance disparities.
    """
    )
    st.markdown("---")
    st.title("📊 Model & Dataset Information")

    # Two column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🏛️ Model Architecture")
        st.markdown(
            """
        ### ResNet50 with Transfer Learning
        
        **Base Architecture:**
        - ResNet50 (50 layers deep)
        - Pre-trained on ImageNet (1.2M images)
        - 23.5 million parameters
        
        **Modifications:**
        - Replaced final classification layer
        - 7 output classes (emotions)
        - Fine-tuned on RAF-DB dataset
        
        **Training Details:**
        - Optimizer: Adam
        - Learning Rate: 0.0001 (reduced for fine-tuning)
        - Epochs: 30
        - Batch Size: 32
        - Input Size: 256×256 (upscaled from 100×100)
        
        **Why ResNet50?**
        - Deep architecture captures subtle facial features
        - Transfer learning from ImageNet provides strong feature extraction
        - I experimented with other models, including developing my own CNN from scratch, but found ResNet50 yielded the strongest performance as it already had robust feature extraction capabilities for things like eyes and noses, whereas my CNN struggled and took much longer to learn these features meaningfully. Models like Meta's DeepFace are state of the art in this space, but its architecture is not publicly available, making applying easily interpretable methods like GradCAM difficult.
        """
        )

    with col2:
        st.header("📚 Dataset: RAF-DB")
        st.markdown(
            """
        ### Real-world Affective Faces Database
        
        **Dataset Characteristics:**
        - **Training Set:** 12,271 images
        - **Test Set:** ~3,000 images
        - **Image Size:** 100×100 pixels (cropped to faces)
        - **Source:** In-the-wild images from internet
        
        **Emotion Categories (7):**
        1. 😊 Happy
        2. 😢 Sad
        3. 😠 Angry
        4. 😮 Surprised
        5. 😨 Fearful
        6. 🤢 Disgusted
        7. 😐 Neutral
        
        **Why This Dataset Was Selected:**
        - Each image labeled by ~40 independent annotators, so crowdsourced emotion labeling ensures a level of reliability.
        - Captures real-world expression variability: lighting, poses, obstructions (glasses, hair, hands), wide range of gender, ages and races.
        - Images are in color which I thought would be necessary for future steps of the project where I might explore skin tone biases.
        
        **Citation:**
        Li, S., Deng, W. (2019). *Reliable Crowdsourcing and Deep Locality-Preserving Learning for Unconstrained Facial Expression Recognition.* IEEE Transactions on Image Processing.
        """
        )

    # Performance section
    st.markdown("---")
    st.header("📈 Model Performance")

    # Check if stats file exists
    import json

    if os.path.exists("model_stats.json"):
        with open("model_stats.json", "r") as f:
            stats = json.load(f)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Test Accuracy",
                f"{stats['overall_accuracy']:.1f}%",
                help="Accuracy on unseen test set",
            )

        with col2:
            best = stats["best_emotions"]
            st.metric(
                "Best Performance",
                f"{best[0].capitalize()}",
                f"{stats['per_emotion'][best[0]]:.1f}%",
            )

        with col3:
            worst = stats["worst_emotions"]
            st.metric(
                "Most Challenging",
                f"{worst[0].capitalize()}",
                f"{stats['per_emotion'][worst[0]]:.1f}%",
            )

        # Per-emotion breakdown
        st.subheader("Performance by Emotion")

        # Sort emotions by accuracy
        sorted_emotions = sorted(
            stats["per_emotion"].items(), key=lambda x: x[1], reverse=True
        )

        for emotion, accuracy in sorted_emotions:
            # Color code based on accuracy
            if accuracy > 80:
                color = "🟢"
            elif accuracy > 60:
                color = "🟡"
            else:
                color = "🔴"

            st.progress(
                accuracy / 100, text=f"{color} {emotion.capitalize()}: {accuracy:.1f}%"
            )
    else:
        st.info(
            """
        📊 Performance statistics not yet generated.
        
        Run the evaluation script in your notebook to generate `model_stats.json`,
        then restart the app to see detailed performance metrics.
        """
        )

    # XAI Technique
    st.markdown("---")
    st.header("🔍 Explainability: Grad-CAM")

    st.markdown(
        """
    ### Gradient-weighted Class Activation Mapping
    
    **What is Grad-CAM?**
    
    Grad-CAM is a visualization technique that highlights which regions of an image 
    were important for a model's prediction. It was one of my favorite techniques we learned this semester, and I knew I wanted to incorporate it into my final project. 
    
    **Why Use Grad-CAM for Emotion Recognition?**
    
    - Reveals which facial features drive predictions (eyes, mouth, eyebrows)
    - Can clearly diagnose diagnose model failures (scattered attention = uncertainty)
    - Validates that model uses human-interpretable features
    - Builds trust by making "black box" decisions transparent
    

    """
    )


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
    elif st.session_state.page == "about":
        show_about_page()
    elif st.session_state.page in emotion_names:
        show_emotion_page(st.session_state.page, model, target_layer, emotion_names)


if __name__ == "__main__":
    main()
