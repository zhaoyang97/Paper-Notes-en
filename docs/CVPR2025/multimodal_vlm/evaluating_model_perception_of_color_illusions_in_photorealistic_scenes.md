---
title: >-
  [Paper Note] Evaluating Model Perception of Color Illusions in Photorealistic Scenes
description: >-
  [CVPR 2025][Multimodal VLM][Color Illusion] This paper proposes an automated framework to generate the RCID dataset containing 19,000 photorealistic color illusion images, systematically revealing for the first time that VLMs indeed exhibit human-like color perception biases, and employs a mixed-training approach to enable models to simultaneously understand both human perception and ground-truth pixel values.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Color Illusion"
  - "Visual Perception"
  - "VLM Evaluation"
  - "Human Visual Bias"
  - "Perception Alignment"
date: 2026-05-08
content_hash: 89e0a7fcdd946d02
---

# Evaluating Model Perception of Color Illusions in Photorealistic Scenes

**Conference**: CVPR 2025  
**arXiv**: [2412.06184](https://arxiv.org/abs/2412.06184)  
**Code**: [https://github.com/mao1207/RCID](https://github.com/mao1207/RCID)  
**Area**: Multimodal VLM  
**Keywords**: Color Illusion, Visual Perception, VLM Evaluation, Human Visual Bias, Perception Alignment

## TL;DR

This paper proposes an automated framework to generate the RCID dataset containing 19,000 photorealistic color illusion images, systematically revealing for the first time that VLMs indeed exhibit human-like color perception biases, and employs a mixed-training approach to enable models to simultaneously understand both human perception and ground-truth pixel values.

## Background & Motivation

**Background**: Color illusions are widely studied phenomena in the human visual system—our perceived color is influenced by factors like the surrounding environment, lighting conditions, and stripe patterns, making it inconsistent with actual pixel values. As VLMs show increasingly human-like behavior on various visual tasks, a natural question arises: Are VLMs also tricked by color illusions like humans?

**Limitations of Prior Work**: Existing studies (such as IllusionVQA, GVIL) have attempted to answer this question but suffer from key limitations: (1) The test images are collected from the internet, which are limited in quantity and mostly consist of classic illusion diagrams (60%+ are well-known examples), allowing VLMs to potentially memorize the "correct" answers through training rather than truly perceiving the illusions; (2) Images with simple geometric shapes do not reflect illusions in real-world scenes; (3) The sample size is insufficient to support an in-depth analysis of influencing factors.

**Key Challenge**: There is a need for large-scale, photorealistic, and unseen illusion images for fair testing, but manually creating such images is extremely costly.

**Goal**: (1) How to automatically generate large-scale photorealistic illusion images? (2) Do VLMs actually get deceived by color illusions, or are they just guessing? (3) What are the causes of the illusions—do they stem from the visual system, prior knowledge, or both? (4) Can models be trained to understand both human perception and pixel ground truths?

**Key Insight**: The authors utilize ControlNet to transform simple geometric illusion patterns into photorealistic scene images, maintaining the illusion effects while achieving a realistic appearance. By combining procedural generation and human validation, a scalable, controllable evaluation dataset is constructed.

**Core Idea**: Map procedurally generated simple illusion images to photorealistic scenes using ControlNet, construct the 19K-scale RCID dataset, systematically reveal the color perception biases of VLMs, and propose a mixed-training method.

## Method

### Overall Architecture

The overall pipeline is divided into three steps: (1) Image Generation—procedurally generating simple illusion diagrams $\rightarrow$ converting them into photorealistic images using ControlNet (for contrast and stripe illusions) or directly applying color filters to COCO images (for filter illusions); (2) Question Generation—GPT-4o generates color judgment questions based on the image content; (3) Human Verification—collecting feedback from 241 participants via the Prolific platform to verify whether the images actually elicit illusions.

### Key Designs

1. **ControlNet-Driven Photorealistic Illusion Image Generation**:

    - **Function**: Transform simple geometric illusion diagrams into color illusions in photorealistic scenes.
    - **Mechanism**: First, ControlNet is trained on COCO 2017 to learn the mapping from $10 \times 10$ color grids to photorealistic images. During training, images are quantized into color grids $G_{x,y} = \frac{1}{|R_{x,y}|}\sum_{p \in R_{x,y}} I(p)$, which are used as conditioning inputs to train the diffusion model. During inference, procedural functions generate simple color-block diagrams with specific luminance differences (contrast illusion: different background brightness + foreground blocks of the same color) or alternating color stripes (stripe illusion), which are then transformed into photorealistic images via the trained ControlNet.
    - **Design Motivation**: Directly embedding controlled color illusions into real images is extremely difficult, and ControlNet allows precise control over color distribution while generating high-quality realistic scenes. Procedural generation ensures controllability and scalability.

2. **Systematic Construction of Three Types of Color Illusions**:

    - **Function**: Cover three major color illusion mechanisms, each with a control group.
    - **Mechanism**: **Contrast Illusion**—placing identical foreground objects on backgrounds of different brightness, causing humans to perceive the foreground colors differently. Luminance is controlled using $p(C, \mu) = (r \cdot \mu, g \cdot \mu, b \cdot \mu)$. **Stripe Illusion**—alternating colored and black stripes, which influences the perception of the background color. **Filter Illusion**—applying contrasting color filters to COCO images (shifting hue in HSV space), making humans perceive a certain color even though no pixels of that color actually exist. A control group (no-illusion version) is generated for each illusion category for baseline comparison.
    - **Design Motivation**: A single type of illusion cannot comprehensively evaluate the color perception of VLMs. These three types test three distinct perceptual mechanisms: luminance contrast, spatial frequency interference, and color constancy.

3. **Human Verification and Dataset Construction**:

    - **Function**: Ensure that the generated illusion images indeed deceive humans.
    - **Mechanism**: 241 participants were recruited on the Prolific platform, with 5 people assigned to judge each image. If ≥3 people are deceived, it is labeled as an illusion image; if all 5 people give the correct pixel-value answer, it is categorized into the control group. After filtering, Fleiss' kappa increased from 0.648 to 0.806. The final RCID dataset contains 19,000 images (8,000 contrast + 9,000 stripe + 2,000 filter), split into train/val/test with 9,500/4,750/4,750 respectively.
    - **Design Motivation**: The procedurally generated theoretical illusion images might not retain their illusion effects after photorealistic conversion. Human validation is a necessary step to ensure data quality.

### Loss & Training

The evaluation of VLMs follows a fine-tune $\rightarrow$ test paradigm: first, the model is fine-tuned on non-illusion images to improve basic color comparison capabilities (>75% accuracy), and then tested on illusion images. The mixed-training method blends "pixel-value-based" and "human-perception-based" QA pairs during training, enabling the model to learn to distinguish between the two response modes based on the prompts.

## Key Experimental Results

### Main Results

| Model | Non-Illusion Accuracy | Contrast Illusion Human-like Rate | Stripe Illusion Human-like Rate | Filter Illusion Human-like Rate |
|------|------------|----------------------|----------------------|----------------------|
| LLaVA-1.5 (7B) | >75% | 41.2% | 35.2% | 67.8% |
| InternVL2 | >75% | ~35% | ~30% | ~60% |
| CogVLM | >75% | ~38% | ~33% | ~65% |

All VLMs exhibit a significant drop in accuracy on illusion images, and a portion of their responses show human-like perceptual biases. However, there is also a large number of responses that align with neither pixel values nor human perception (N/A category), indicating that VLM color perception is not entirely human-like.

### Ablation Study

| Training Strategy | Contrast Illusion No-Illusion Rate (Pixel Prompt) | Contrast Illusion Human-like Rate (Perceptual Prompt) |
|---------|----------------------------------|--------------------------------|
| LLaVA baseline | 35.4% | 37.4% |
| Mixed Training (Ours) | **83.9%** | **79.0%** |

Mixed training enables the model to switch response modes based on different prompts: when asked "based on pixel values", the accuracy increases from 35.4% to 83.9%; when asked "based on human perception", the Human-like rate increases from 37.4% to 79.0%.

### Key Findings

- **All VLMs are affected by color illusions**, but their influence patterns are not entirely identical to humans—a large number of responses belong to the N/A category (neither pixel values nor human perception).
- **Larger models are more easily deceived**: In the OFA series, as the model scale increases, the Human-like rate rises and the No-Illusion rate decreases, suggesting that larger models learn more human-like perceptual biases.
- **Pure vision models are also affected**: The accuracy of ResNet, ViT, and VGG generally decreases on illusion images, indicating that perceptual bias partially originates from the visual encoder itself.
- **Prior knowledge amplifies the illusion effect**: Illusions in complex scenes are more likely to deceive models and humans than those in simple objects; the higher the diversity of color terms described in language, the stronger the model’s ability to differentiate that color.
- **The closer the foreground-background colors are, the stronger the contrast illusion is**; the more stripes there are, the stronger the stripe illusion is—these trends remain consistent between humans and VLMs.

## Highlights & Insights

- **Creative application of ControlNet as an illusion image generator**: The two-stage method of procedural control over color distribution + diffusion model generation of photorealistic appearance ensures experimental controllability while achieving unprecedented scale. This "controlled attribute $\rightarrow$ photorealistic image" paradigm can be transferred to other cognitive testing scenarios (depth illusions, motion illusions, etc.).
- **Revealing a dual-source origin of VLM perceptual bias**: Proving that biases partly come from the visual encoder (bottom-up) and partly from linguistic prior knowledge (top-down), which maps interestingly onto the "bottom-up vs top-down" framework of classical human vision theory.
- **High practical value of the mixed-training method**: Allowing models to flexibly switch between "understanding human perception" and "reporting pixel ground truth" based on user intent has direct application value in scenarios such as medical image analysis (requiring pixel accuracy) and human-computer interaction (requiring understanding of human perception).

## Limitations & Future Work

- **Only covers color illusions**: Other important types such as shape illusions (e.g., Müller-Lyer), motion illusions, and depth illusions are not covered, limiting the comprehensiveness of the evaluation.
- **ControlNet generation may introduce artifacts**: Subtle color drift may occur during the photorealistic conversion process, affecting the precise control of illusion effects.
- **Limited human validation sample size**: With only 5 human judges per image, classification of boundary cases close to the threshold may be unstable.
- **Insufficient testing on closed-source models**: GPT-4o and Gemini are only briefly reported in the appendix, with the main conclusions based on open-source models.

## Related Work & Insights

- **vs IllusionVQA (Shahgir et al., 2024)**: IllusionVQA uses classic illusion images collected from the web, 60% of which are well-known, so VLMs may have memorized the answers. This paper resolves data contamination through automated generation of entirely new images, with a scale more than 10 times larger.
- **vs GVIL (2024)**: GVIL also studies VLM perception of visual illusions but uses combinations of simple shapes as test images. The photorealistic images in this work better reflect perceptual challenges in real-world scenes.
- **vs Gomez-Villa et al. (2019, 2020)**: Prior work demonstrated that CNNs exhibit perceptual biases on simple illusion images, but was limited to reconstruction tasks and simple images. This work extends this research to the VLM era and photorealistic scenes.

## Rating

- Novelty: ⭐⭐⭐⭐ The first large-scale photorealistic color illusion dataset and systematic evaluation of VLM perception.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering multiple VLMs and analyzing multiple influencing factors (model size, image structure, prompting methods, linguistic differences).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and in-depth analysis.
- Value: ⭐⭐⭐⭐ Provides crucial tools and insights for understanding the perceptual characteristics of VLMs, with cautionary implications for safety-critical fields (medicine, autonomous driving).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Do VLMs Perceive or Recall? Probing Visual Perception vs. Memory with Classic Visual Illusions](../../CVPR2026/multimodal_vlm/do_vlms_perceive_or_recall_probing_visual_perception_vs_memory_with_classic_visu.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](../../NeurIPS2025/multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[CVPR 2025\] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](homesafe-bench_evaluating_vision-language_models_on_unsafe_action_detection_for_.md)
- [\[CVPR 2025\] Revisiting Model Stitching in the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model_era.md)
- [\[CVPR 2025\] CodePercept: Code-Grounded Visual STEM Perception for MLLMs](codepercept_code-grounded_visual_stem_perception_for_mllms.md)

</div>

<!-- RELATED:END -->
