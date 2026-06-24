---
title: >-
  [Paper Note] HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation
description: >-
  [CVPR 2025][Segmentation][SAM Adapter] HFP-SAM proposes a hierarchical frequency-prompted SAM framework. By injecting marine scene information through a Frequency Guided Adapter (FGA), automatically generating high-quality point prompts via Frequency-aware Point Selection (FPS), and decoding efficiently with Full-view Mamba (FVM), it achieves SOTA performance on four marine animal segmentation datasets.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "SAM Adapter"
  - "Frequency Domain Prior"
  - "Point Prompt Generation"
  - "Mamba Decoder"
  - "Marine Animal Segmentation"
date: 2026-05-08
content_hash: 51bd58a3ed2df0e2
---

# HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.12708](https://arxiv.org/abs/2603.12708)  
**Code**: [GitHub](https://github.com/Drchip61/TIP-HFP-SAM)  
**Area**: Segmentation / Marine Animal  
**Keywords**: SAM Adapter, Frequency Domain Prior, Point Prompt Generation, Mamba Decoder, Marine Animal Segmentation

## TL;DR
HFP-SAM proposes a hierarchical frequency-prompted SAM framework. By injecting marine scene information through a Frequency Guided Adapter (FGA), automatically generating high-quality point prompts via Frequency-aware Point Selection (FPS), and decoding efficiently with Full-view Mamba (FVM), it achieves SOTA performance on four marine animal segmentation datasets.

## Background & Motivation
**Background**: Marine Animal Segmentation (MAS) is highly challenging due to poor underwater visibility, variable lighting, and particulate interference. CNN-based methods are limited by local receptive fields, while Transformers require large-scale data training. Although SAM is generalizable, it lacks fine-grained and frequency-domain awareness.

**Limitations of Prior Work**: (1) Existing SAM adaptation methods (e.g., Dual-SAM, MAS-SAM) only modify the encoder/decoder, neglecting the importance of **prompt design**; (2) Marine scenes suffer from severe high-frequency noise, to which SAM is highly sensitive, leading to segmentation artifacts; (3) Simple point/box prompts perform poorly on marine organisms with complex structures.

**Key Challenge**: How to automatically generate high-quality point prompts in noise-heavy underwater scenes, and effectively leverage frequency domain information to improve the segmentation accuracy of SAM?

**Key Insight**: Leveraging the frequency-domain prior of the wavelet transform to simultaneously guide feature adaptation and prompt generation, as the frequency domain naturally possesses the capability to suppress noise and highlight boundaries.

## Method

### Overall Architecture
Input marine image $\rightarrow$ Frozen SAM backbone + FGA to inject frequency domain information $\rightarrow$ SAM generates a coarse segmentation mask $M^c$ $\rightarrow$ FPS combines frequency prior and $M^c$ to generate point prompts $\rightarrow$ Point prompts + coarse mask are input to SAM prompt encoder $\rightarrow$ FVM decoder outputs the final fine segmentation mask.

### Key Designs

1. **Frequency Guided Adapter (FGA)**:

    - **Function**: Injects frequency-domain priors into each Transformer block of the frozen SAM backbone.
    - **Mechanism**: Performs Discrete Haar Wavelet Transform (DHWT) on the input image to obtain low-frequency $I^{ll}$ and three high-frequency subbands $I^{lh}, I^{hl}, I^{hh}$, taking the average of the three high-frequency subbands as the frequency map $M^h$. A sliding window is used to select the top-$k$ windows with the highest responses as the frequency prior region $P$. After downsampling to align with the feature map, element-wise multiplication is performed to obtain the frequency-guided feature $\hat{X}_i^f$.
    - **Dual-path injection**: The frequency-guided feature $\hat{X}_i^f$ and the original spatial feature $\hat{X}_i$ are independently projected via down-up linear layers before being added back to the residual connections.
    - **Design Motivation**: The frequency-domain prior mask acts as a modulation signal (rather than directly encoding frequency features), narrowing the alignment gap between frequency cues and SAM's pre-trained spatial representations.

2. **Frequency-aware Point Selection (FPS)**:

    - **Function**: Automatically generates positive/negative point prompts without requiring external networks.
    - **Mechanism**: Keypoints are sampled within the high-response windows of the frequency map $M^h$. For each window, the $t$ points with the highest and lowest frequency values (totaling $2t$ points) are selected, and then their positive/negative attributes are determined by the binarization of the coarse segmentation mask $M^c$—points falling inside the foreground region are labeled as positive prompts $p^+$, and those in the background as negative prompts $p^-$.
    - **Design Motivation**: High-response frequency areas correspond to image boundaries (the transition between target and background). Points sampled from these areas are more informative than random or maximum-distance sampling, and require no external prompt generator networks.

3. **Full-view Mamba (FVM)**:

    - **Function**: Replaces the simple SAM decoder to simultaneously model long-range dependencies in both spatial and channel dimensions.
    - **Mechanism**: Utilizes State Space Models (SSMs) as a linear-complexity alternative for global modeling. It scans along the spatial dimension to capture long-range spatial context, and performs bidirectional scanning along the channel dimension to capture global channel correlations.
    - **Design Motivation**: SAM's original decoder is too simplistic, leading to loss of detail, whereas Transformer decoders are computationally heavy. The Mamba structure maintains global modeling capability with linear complexity.

## Key Experimental Results

### Datasets and Evaluation
- **MAS3K**: 3,103 marine animal images (1,769 for training / 1,141 for testing).
- **RMAS**: 3,014 images (2,514 for training / 500 for testing).
- **UFO-120**: 1,620 diverse underwater scenes (1,500 for training / 120 for testing).
- **RUWI**: 700 images (525 for training / 175 for testing).
- Evaluation metrics: mIoU, $S_\alpha$, $F_\beta^w$, $mE_\phi$, MAE.
- Training hardware: Single RTX 3090 GPU, input size 512×512, batch size 6, 50 epochs, AdamW (lr=0.001).

### Main Results (MAS3K / RMAS / UFO120 / RUWI)

| Method | Backbone | MAS3K mIoU | MAS3K MAE↓ | RMAS mIoU | UFO120 mIoU | RUWI mIoU |
|------|----------|-----------|-----------|----------|------------|-----------|
| Dual-SAM | ViT-B | 0.789 | 0.023 | 0.735 | 0.810 | 0.900 |
| MAS-SAM | ViT-B | 0.788 | 0.025 | 0.742 | 0.807 | 0.902 |
| SAM2-Adapter | Hiera-L | 0.778 | 0.027 | 0.650 | 0.755 | 0.883 |
| **HFP-SAM** | ViT-B | **0.797** | **0.024** | **0.745** | **0.803** | **0.904** |
| **HFP-SAM2** | Hiera-L | **0.807** | **0.022** | **0.758** | **0.813** | **0.913** |

- HFP-SAM2 achieves comprehensive SOTA performance on all four datasets, reaching 0.807 mIoU on MAS3K and 0.913 mIoU on RUWI.
- Compared to the original SAM (ViT-B, mIoU 0.566), HFP-SAM achieves a gain of +23.1%.
- SAM2-Adapter achieves only 0.650 on RMAS, falling far behind HFP-SAM2's 0.758, demonstrating that general adaptation is inferior to domain specialization.

### Ablation Study (MAS3K)

| Config | Adapter | FGA | FPS | FVM | mIoU | MAE↓ |
|------|---------|-----|-----|-----|------|------|
| (A) SAM baseline | ✕ | ✕ | ✕ | ✕ | 0.566 | 0.059 |
| (B) +Standard Adapter | ✓ | ✕ | ✕ | ✕ | 0.739 | 0.031 |
| (C) +FGA | ✓ | ✓ | ✕ | ✕ | 0.754 | 0.030 |
| (D) +FPS | ✓ | ✓ | ✓ | ✕ | 0.771 | 0.028 |
| (E) +FVM | ✓ | ✓ | ✓ | ✓ | 0.792 | 0.026 |
| (F) +Auxiliary Loss | ✓ | ✓ | ✓ | ✓ | 0.797 | 0.024 |

- FGA: +1.5% mIoU (frequency prior mask modulation vs. standard spatial adapter).
- FPS: +1.7% mIoU (frequency-aware point sampling vs. no prompt); compared to random sampling (0.760) and global sampling (0.764), FPS reaches 0.771 with only 9.3ms of overhead.
- FVM: +2.1% mIoU (spatial + channel bidirectional SSM decoding).
- FPS hyperparameters: Number of windows = 10, window size = 32, sampled points per window = 2 is the optimal configuration.
- Combined positive + negative prompts (mIoU 0.797) outperforms positive-only prompts (0.789) or negative-only prompts (0.782).

### Key Findings
- DHWT frequency domain analysis effectively filters out high-frequency noise in marine scenes, allowing the model to focus on target boundaries.
- The location and quality of point prompts are crucial to SAM's segmentation performance; frequency-guided point selection significantly outperforms heuristic methods.
- The Mamba structure provides global context in the decoding stage while maintaining linear complexity.
- Domain shift among the four datasets was measured using W1 and MMD-RBF, showing significant gaps between datasets.

## Highlights & Insights
- **Frequency Domain Triple-Play**: FGA (Encoder) → FPS (Prompt) → FVM (Decoder) all leverage frequency domain information, establishing a complete frequency-aware pipeline.
- **Zero-External-Network Prompt Generation**: FPS introduces no learnable parameters, purely generating point prompts based on frequency analysis and coarse masks, making it lightweight and efficient.
- **Prior Mask Modulation instead of Direct Frequency Feature Encoding**: FGA weights spatial features using a frequency-prior mask, avoiding the alignment problem between direct frequency encoding and SAM's spatial representation.

## Limitations & Future Work
- Validated only on the marine animal segmentation task; generalization to other underwater tasks (e.g., coral/seagrass segmentation) or general scenes remains unknown.
- The window size and top-$k$ parameters of FPS require manual configuration and may need tuning for different datasets.
- Although FVM offers linear complexity, it increases the parameters and latency of the decoder; the paper lacks a complete comparison of inference speed and parameter count.
- The frequency prior may fail in extreme camouflage scenarios where the target and background patterns are highly similar.
- The design of the loss functions (weighted BCE + weighted IoU) is relatively standard, without exploring more advanced boundary supervision strategies.

## Related Work & Insights
- **vs. Dual-SAM**: Dual-SAM mainly modifies the encoder and decoder, whereas HFP-SAM additionally focuses on prompt design, making them complementary.
- **vs. MAS-SAM**: MAS-SAM uses a hypermap to fuse multi-layer encoder features, while HFP-SAM provides a more direct guidance through the frequency domain.
- **vs. SAM2-Adapter**: SAM2-Adapter only achieves 0.650 mIoU on RMAS, which is significantly lower than HFP-SAM2's 0.758, illustrating that general adaptation is less effective than domain-specific adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐ Frequency-domain driven prompt generation is a novel and reasonable design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated with four datasets, 20+ comparison methods, and dual versions of SAM/SAM2.
- Writing Quality: ⭐⭐⭐ Equations are detailed, but some descriptions are verbose.
- Value: ⭐⭐⭐ Beneficial for the marine animal segmentation field; the frequency prompting idea possesses certain generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[NeurIPS 2025\] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning](../../NeurIPS2025/segmentation/sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)
- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [\[ICLR 2026\] SAM-Veteran: An MLLM-based Human-like SAM Agent for Reasoning Segmentation](../../ICLR2026/segmentation/sam-veteran_an_mllm-based_human-like_sam_agent_for_reasoning_segmentation.md)
- [\[CVPR 2025\] Frequency Dynamic Convolution for Dense Image Prediction](frequency_dynamic_convolution_for_dense_image_prediction.md)

</div>

<!-- RELATED:END -->
