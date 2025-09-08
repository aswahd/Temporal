# Time-Contrastive Pretraining for In-Context Image and Video Segmentation

## Abstract

In-context learning (ICL) has shown promise for generalizing to new visual tasks using a few examples, but current methods are limited. They typically rely on a rigid gridding strategy that restricts the number and resolution of context images. We propose **Temporal**, a novel approach that overcomes these limitations by reformulating visual ICL as a video object segmentation (VOS) problem. This VOS-based approach naturally handles a variable number of full-resolution context images. To automatically select the most relevant context for a given query, we introduce a prompt retriever pretrained on videos using a time-contrastive objective. This objective learns from the temporal coherence of videos, using adjacent frames as positive examples (i.e., useful context images) and distant frames as negatives. For image segmentation, our retriever builds a pseudo-video by prepending the retrieved context images to the query image, which is then processed by the VOS model. For video segmentation, the retriever identifies keyframes, our ICL pipeline generates their masks, and these masks are propagated through the video. On the MICCAI FLARE 2022 challenge, Temporal significantly outperforms baselines, achieving a Dice score of 90.95% for image segmentation (+10.64%) and 92.45% for video segmentation (+14.88%).

## Quick Start

Download demo dataset from [here](https://drive.google.com/drive/folders/1XPEijJrCzLskw7i49zMXZ-u2545bqi-l?usp=sharing), or see how to structure your dataset below.

For (absolutely) training-free start to check if everything is working fine, use `dinov2` as the similarity function and a non-finetuned SAM as the foundation model.

### Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install SAM2 with CUDA support (if available):

```bash
cd temporal/models/sam2
pip install -e .
cd ../../..
```

3. Install the main package:

```bash
pip install -e .
```

## Dataset Structure

The dataset should be organized in one of the following formats:

### Format 1: Separate Images and Masks with Class Subdirectories

This format is suitable when masks for different classes are stored in separate subdirectories.

```

root/
├── imgs/
│ ├── img1.png
│ └── ...
└── gts/
│ ├── img1/
│ ├── class1/
│ └── mask.png
│ └── class2/
│ └── mask.png
│ ...

```

### Format 2: Paired Images and Masks

This format pairs each image with its corresponding mask directly.

```

root/
├── imgs/
│ ├── img1.png
│ └── ...
└── gts/
│ ├── img1.png
│ └── ...

```

### Running Gradio Apps

Add the path to your training images in [`temporal/inference_configs/app_image.yaml`](temporal/inference_configs/app_image.yaml) or [`temporal/inference_configs/app_video.yaml`](temporal/inference_configs/app_video.yaml).

Launch Gradio applications for image or video segmentation.

```bash
# Launch a Gradio app for image segmentation
python -m temporal.app.app_image
# Launch a Gradio app for video segmentation
python -m temporal.app.app_video

```

Or you can run the following commands from the terminal:

```bash
ti # For temporal image (ti) segmentation
tv # For temporal video (tv) segmentation
```

## Fine-tuning SAM2 for Improved Performance

Fine-tuning SAM2 involves creating "synthetic" videos from 2D images in the training set.

### Creating Synthetic Videos

1. **Similarity Calculation:** For each image in the training set, use a similarity function (e.g., `dinov2`) to find the top-k most similar images.
2. **Video Construction:** Concatenate these similar images to form a synthetic video. Use the [`temporal/scripts/make_video_dataset.py`](temporal/scripts/make_video_dataset.py) script as a reference.

The resulting dataset should have the following structure:

```
root/
    JPEGImages/
        query_image1_name/
            00000.jpg  # Least similar image
            00001.jpg  # Second least similar
            ...
            000XX.jpg  # Original query image
        query_image2_name/
            00000.jpg
            ...
    Annotations/
        query_image1_name/
            00000.png  # Mask for least similar
            00001.png  # Mask for second least similar
            ...
            000XX.png  # Mask for query image
        query_image2_name/
            00000.png
            ...
```

### Fine-tuning the Model

Once the synthetic video dataset is prepared, fine-tune the SAM2 model:

```bash
# Navigate to the SAM2 directory
cd temporal/model/sam2

# Start the training process using the provided configuration file
python -m training.train -c configs/sam2.1_training/config_file.yaml
```
