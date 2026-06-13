---
title: >-
  [Paper Note] AI-Generated Video Detection via Perceptual Straightening
description: >-
  [NeurIPS 2025][Model Compression][AI-generated video detection] This paper proposes ReStraV, a method grounded in the perceptual straightening hypothesis—which posits that real videos form straighter trajectories in neur…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "AI-generated video detection"
  - "perceptual straightening"
  - "DINOv2"
  - "temporal curvature"
  - "representation geometry"
date: 2026-05-08
content_hash: f288eeca16657513
---

# AI-Generated Video Detection via Perceptual Straightening

**Conference**: NeurIPS 2025
**arXiv**: [2507.00583](https://arxiv.org/abs/2507.00583)  
**Code**: [GitHub](https://github.com/ChristianInterno/ReStraV)  
**Area**: Model Compression
**Keywords**: AI-generated video detection, perceptual straightening, DINOv2, temporal curvature, representation geometry

## TL;DR
This paper proposes ReStraV, a method grounded in the perceptual straightening hypothesis—which posits that real videos form straighter trajectories in neural representation space—to detect AI-generated videos. Using temporal curvature and step-size statistics extracted from DINOv2 feature space, a lightweight classifier is trained to distinguish real from generated content, achieving 97.17% accuracy and 98.63% AUROC on VidProM with only ~48ms inference time.

## Background & Motivation

**Background**: AI video generation systems (Sora, Pika, VideoCrafter, etc.) are advancing rapidly, producing increasingly photorealistic content, making robust detection urgently necessary. Existing detection approaches include image-based methods (CNNSpot, UnivFD) and video-based methods (I3D, SlowFast, VideoSwin); the former disregard temporal information, while the latter require extensive training and generalize poorly.

**Limitations of Prior Work**: (a) Image-level detectors fail to capture temporal inconsistencies; (b) video-level detectors require large-scale generator-specific training and generalize poorly to unseen generators; (c) watermarking schemes depend on generator cooperation and can be circumvented.

**Key Challenge**: A generalizable detection method is needed that does not rely on generator-specific artifacts and can capture anomalies along the temporal dimension.

**Goal**
- Does a fundamental geometric difference exist between real and AI-generated videos in neural representation space?
- Can such a difference support efficient and generalizable detection?

**Key Insight**: Motivated by the neuroscientific perceptual straightening hypothesis—which holds that the visual system straightens the temporal trajectories of natural videos to facilitate predictive coding—this work hypothesizes that pre-trained vision models (DINOv2) selectively straighten real videos but not AI-generated ones, yielding discriminative curvature differences.

**Core Idea**: Real videos trace straighter trajectories (lower curvature) in DINOv2 representation space, whereas AI-generated videos follow more curved trajectories—this geometric discrepancy constitutes a reliable detection signal.

## Method

### Overall Architecture
ReStraV operates in three stages: (1) uniformly sample 24 frames from the input video and extract per-frame features using a frozen DINOv2 ViT-S/14 by concatenating CLS and patch tokens into $z_i \in \mathbb{R}^{75648}$; (2) compute inter-frame step sizes $d_i = \|z_{i+1} - z_i\|$ and curvature angles $\theta_i = \arccos(\frac{\Delta z_i \cdot \Delta z_{i+1}}{\|\Delta z_i\| \|\Delta z_{i+1}\|})$; (3) extract statistical descriptors (mean/min/max/var) and train a lightweight classifier (MLP/GB/RF) for binary discrimination.

### Key Designs

1. **Differential Effect of Perceptual Straightening**

    - **Function**: Identifies that DINOv2 straightens real and AI-generated videos to different degrees.
    - **Mechanism**: A systematic comparison across 14 visual encoders reveals that HVS-inspired models (Gabor, LGN-V1) straighten all videos equally ($\Delta\theta < 0$, no discriminative power), whereas the self-supervised model DINOv2 selectively straightens real videos that conform to its training distribution while leaving AI-generated videos curved ($\Delta\theta = 45.46°$, strong discriminative signal). Crucially, absolute straightening ability is uncorrelated with detection performance ($\rho=-0.13, p=0.64$); differential straightening is the key factor.
    - **Design Motivation**: DINOv2 is self-supervised on large-scale real-world data, internalizing the statistical regularities of natural content. Trajectories of real videos that conform to this prior are straightened; those of AI-generated videos, which violate the prior, remain curved.

2. **Temporal Curvature and Step Size as Detection Features**

    - **Function**: Quantifies the geometric properties of trajectories in representation space.
    - **Mechanism**: Curvature $\theta_i = \arccos(\frac{\Delta z_i \cdot \Delta z_{i+1}}{d_i \cdot d_{i+1}})$ measures directional change between successive displacements (path bending); step size $d_i = \|\Delta z_i\|$ measures the magnitude of inter-frame change. Real videos exhibit low mean curvature ($\mu_\theta$ small) and high curvature variance ($\sigma_\theta^2$ large), whereas AI-generated videos exhibit high mean curvature and low variance—i.e., they bend uniformly. An 8-dimensional statistical feature vector is sufficient for discrimination.
    - **Design Motivation**: These simple geometric quantities (angles and distances) are physically intuitive, interpretable, and computationally inexpensive.

3. **Lightweight Classifier**

    - **Function**: Performs binary classification from a 21-dimensional feature vector using off-the-shelf classifiers.
    - **Mechanism**: The feature vector comprises 7 step-size values + 6 curvature values + 8 statistical descriptors = 21 dimensions. Six classifiers (LR/GNB/RF/GB/SVM/MLP) are evaluated; MLP (64→32) achieves the best performance (97.17% accuracy). No pixel-level processing or DINOv2 fine-tuning is required.
    - **Design Motivation**: The lightweight classifier ensures transparency and interpretability while enabling extremely fast inference—DINOv2 forward pass 43.6ms + classification <5ms = ~48ms end-to-end.

### Loss & Training
- DINOv2 is fully frozen (pre-trained weights used as-is).
- The classifier is trained with standard cross-entropy loss.
- No data augmentation or feature engineering is applied; only 3-fold grid search is performed.

## Key Experimental Results

### Main Results: VidProM Benchmark

| Method | Type | Accuracy↑ | AUROC↑ |
|--------|------|-----------|--------|
| CNNSpot (image) | Supervised | 52.66 | 55.47 |
| UnivFD (image) | Supervised | 68.71 | 66.11 |
| I3D (video) | Supervised | 91.76 | 95.18 |
| VideoSwin (video) | Supervised | 94.47 | 97.95 |
| **ReStraV-MLP (Ours)** | Lightweight | **97.17** | **98.63** |

### Cross-Benchmark Generalization

| Benchmark | ReStraV Accuracy | Best Baseline |
|-----------|-----------------|---------------|
| VidProM | **97.17%** | 94.47% (VideoSwin) |
| GenVidBench | **SOTA** | — |
| Physics-IQ | **SOTA** | — |

### Ablation Study: Encoder Selection

| Encoder | $\Delta\theta$ (Curvature Gap) | Detection Ability |
|---------|-------------------------------|-------------------|
| DINOv2 ViT-S/14 | **+45.46°** | **Best** |
| CLIP | +25° | Second |
| SimCLR | +15° | Moderate |
| Gabor (HVS) | −5° | Ineffective |

### Key Findings
- **Differential straightening by DINOv2 is the key**: Absolute straightening ability does not equal detection ability; differential straightening—straighter for real videos, not straightened for AI videos—is the detection signal.
- **"Uniform bending" signature of AI-generated videos**: AI-generated videos exhibit high mean curvature but low variance, indicating that temporal inconsistencies introduced by generative models are systematic rather than random.
- **Pixel space provides no discriminative power**: In raw pixel space, curvature and step-size distributions of real and AI-generated videos overlap heavily; the difference is only visible in learned representation space.
- **Extreme efficiency**: ~48ms end-to-end inference, orders of magnitude faster than VideoSwin and comparable methods.
- **Cross-generator generalization**: The method is effective across diverse generators including Sora, Pika, and VideoCrafter.

## Highlights & Insights
- **Cross-domain inspiration from neuroscience to AI safety** is particularly elegant: the perceptual straightening hypothesis was originally developed to explain biological visual systems, and this work ingeniously repurposes it for AI detection—real videos are straightened even within a "digital neural system."
- **A 21-dimensional feature vector outperforms deep video models**, demonstrating that correct feature design matters far more than model complexity.
- **The discovery of differential vs. absolute straightening** constitutes the core contribution: HVS models straighten everything equally (no discriminative power), whereas SSL models straighten selectively (only within-distribution content is straightened, yielding discriminative power).
- **Strong interpretability**: Curvature and step size carry clear physical meaning—bending corresponds to temporal inconsistency, and large step sizes correspond to abrupt transitions.

## Limitations & Future Work
- **Dependence on DINOv2's specific training distribution**: If a generative model's training data substantially overlaps with DINOv2's, the differential straightening effect may diminish.
- **Only short videos (2–5 seconds) are evaluated**: Effectiveness on longer videos remains unverified.
- **Adversarial robustness**: An adversary aware of ReStraV's operating principle could potentially post-process AI-generated videos to artificially straighten their trajectories.
- **Fixed frame sampling strategy**: Uniform sampling of 24 frames may not be optimal across all scenarios.
- **Potential failure on static scenes**: When video content changes minimally, the curvature signal may be insufficiently strong.

## Related Work & Insights
- **vs. CNNSpot / UnivFD (image detectors)**: Frame-by-frame detection ignores the temporal dimension, yielding performance far below ReStraV.
- **vs. I3D / VideoSwin (video detectors)**: End-to-end video models require extensive training and heavy computation; ReStraV surpasses them using frozen features and a lightweight classifier.
- **vs. watermarking**: Watermarking requires generator cooperation; ReStraV does not—it operates as a purely post-hoc detector.
- **Transfer implications**: Geometric analysis of representation space may generalize to detection of other forms of generated content (AI images, AI audio, AI text).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The cross-domain leap from perceptual straightening to AI detection is highly original; the discovery of differential straightening is a genuinely novel insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison across 14 encoders, 50K training/test samples, multiple benchmarks, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative arc from hypothesis to validation to method to experiments is exceptionally coherent, with high-quality visualizations.
- Value: ⭐⭐⭐⭐⭐ 48ms inference, 97% accuracy, and cross-generator generalization make this directly deployable for content authentication.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[ICML 2026\] Images as Tables: In-Context Learning with TabPFN for Low-Data Detection of AI-Generated Images](../../ICML2026/model_compression/images_as_tables_in-context_learning_with_tabpfn_for_low-data_detection_of_ai-ge.md)
- [\[NeurIPS 2025\] Knowledge Distillation Detection for Open-weights Models](knowledge_distillation_detection_for_open-weights_models.md)
- [\[NeurIPS 2025\] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)
- [\[NeurIPS 2025\] Smooth Regularization for Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)

</div>

<!-- RELATED:END -->
