---
title: >-
  [Paper Note] PiTe: Pixel-Temporal Alignment for Large Video-Language Model
description: >-
  [ECCV 2024][Video Understanding][Large Video-Language Model] The PiTe model is proposed to achieve spatiotemporal video-language alignment at the pixel level using object trajectories. The PiTe-143k dataset is constructed, and the method significantly outperforms existing approaches in zero-shot QA, temporal localization, and dense captioning tasks.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Large Video-Language Model"
  - "trajectory alignment"
  - "pixel-level"
  - "instruction tuning"
date: 2026-05-08
content_hash: ff4c6f8c5ced1741
---

# PiTe: Pixel-Temporal Alignment for Large Video-Language Model

**Conference**: ECCV 2024  
**arXiv**: [2409.07239](https://arxiv.org/abs/2409.07239)  
**Code**: [https://github.com/yliu-cs/PiTe](https://github.com/yliu-cs/PiTe)  
**Area**: Video Understanding / Vision-Language  
**Keywords**: Large Video-Language Model, trajectory alignment, pixel-level, instruction tuning, video understanding

## TL;DR
The PiTe model is proposed to achieve spatiotemporal video-language alignment at the pixel level using object trajectories. The PiTe-143k dataset is constructed, and the method significantly outperforms existing approaches in zero-shot QA, temporal localization, and dense captioning tasks.

## Background & Motivation
**Background**: Large Language Models (LLMs) have driven the advancement of Large Vision-Language Models (LVLMs), shifting the research focus from image to video understanding. Existing Large Video-Language Models (LVidLMs, e.g., VideoChat, Video-LLaMA, Video-ChatGPT) align visual and language features through instruction tuning.

**Limitations of Prior Work**: Traditional QA training paradigms primarily assist LLMs in understanding visual data from a spatial perspective, which struggles to effectively capture temporal dynamics and spatial consistency. Relying solely on instruction tuning is insufficient for comprehensive video understanding.

**Key Challenge**: Videos contain complex spatiotemporal structures, whereas existing solutions lack fine-grained multimodal alignment across both spatial and temporal dimensions.

**Goal**: To achieve fine-grained video-language alignment at the pixel level simultaneously across spatial and temporal dimensions.

**Key Insight**: Utilizing object trajectories as a bridge between video and language, enabling the model to predict the motion trajectories of objects mentioned in the text, thereby learning fine-grained text-to-pixel alignment.

**Core Idea**: Implementing trajectory-guided pixel-temporal alignment, where the LVidLM is trained to predict the motion trajectory of each object, achieving fine-grained alignment in both spatial and temporal dimensions.

## Method

### Overall Architecture
PiTe comprises four core components: (1) a ViT visual encoder (CLIP ViT-L/14) to extract frame features; (2) a linear projection layer (Visual Adapter) to map visual features into the LLM semantic space; (3) Vicuna v1.5 as the LLM; and (4) a localization projector/trajectory projector to map LLM hidden states into the coordinate space. A three-stage training strategy is adopted to progressively enhance model capabilities.

### Key Designs

1. **PiTe-143k Automatic Annotation Dataset**

    - **Function**: Constructing a large-scale video-language dataset containing object motion trajectories.
    - **Mechanism**: Generated via a two-stage automatic annotation pipeline based on InternVid-10M-FLT. Stage 1 uses SuPar to extract noun phrases and GLaMM to generate segmentation masks; Stage 2 employs DOT to track points to obtain trajectories, which are clustered into 3 keypoints using k-means++.
    - **Data Scale**: 143.64k videos, 343.93k event segments, 1.02M motion trajectories, with a total duration of 2086.44 hours.
    - **Design Motivation**: Existing video instruction datasets lack object motion trajectory annotations, impeding research on pixel-level alignment.

2. **Three-Stage Training Strategy**

    - **Function**: Progressively transitioning from image localization $\to$ video trajectory alignment $\to$ instruction following.
    - **Stage 1 — Referring Expression Localization**:
        - Training the visual adapter using the Localized Narratives dataset.
        - Adding an MLP localization projector $\varphi(\cdot)$ in parallel to the vocabulary mapping layer to map language features to 2D coordinates: $p_i = \varphi(h_i)$.
        - Loss: Cross-entropy + L1 regression: $\mathcal{L}_1 = \frac{1}{\ell}\sum_{i=1}^{\ell}(\text{CE}(\text{LLM}(\mathbf{z}, \mathbf{w}_{1:i-1}), w_i) + \lambda|\hat{p}_i - p_i|)$.
        - Fine-tuning the LLM using LoRA (r=64, α=128).
    - **Stage 2 — Pixel-Temporal Alignment**:
        - Aligning video and language using the PiTe-143k dataset via trajectories.
        - The trajectory projector $\rho(\cdot)$ outputs $P \times N$ 2D coordinates ($P$ tracked points $\times N$ frames): $\mathbf{p}_i = \rho(h_i)$.
        - Loss: $\mathcal{L}_2 = \frac{1}{\ell}\sum_{i=1}^{\ell}(\text{CE} + \frac{\lambda}{P \cdot N}\sum_{j=1}^{P}\sum_{k=1}^{N}|\hat{p}_{ijk} - p_{ijk}|)$.
        - Key choice: Initializing the trajectory projector with the weights of the Stage 1 localization projector, as formulated by: $\mathbf{m}_\varphi = \overbrace{\mathbf{m}_\rho \oplus \cdots \oplus \mathbf{m}_\rho}^{P \cdot N}$.
    - **Stage 3 — Video QA Instruction Tuning**:
        - Fine-tuning with high-quality dialogue data from Valley + Video-ChatGPT.
        - Using only standard cross-entropy autoregressive generation loss.

3. **Temporal Boundary Learning**

    - **Function**: Enabling the model to learn temporal boundaries of events.
    - **Mechanism**: Structuring temporal information within the generated text, using formats such as "..., from s to e" or "From s to e, ...", where s and e denote frame indices.
    - Object coordinates without trajectories are uniformly set to $(-1, -1)$ to denote absence.
    - **Design Motivation**: Enhancing the model's perception of temporal boundaries.

### Loss & Training
- Three distinct losses are utilized in the respective stages: Stage 1 (CE + L1), Stage 2 (CE + trajectory L1), and Stage 3 (only CE).
- For each stage, the LoRA weights from the previous stage are merged, and a new LoRA is introduced.
- Training configuration: AdamW optimizer, lr=0.0001, cosine decay, BFloat16 precision.
- Training a 7B model takes approximately 10 hours on a single node equipped with 8×A100 GPUs, while a 13B model requires about 17 hours.

## Key Experimental Results

### Main Results — Zero-shot Video QA

| Dataset | Metric | PiTe-7B | PiTe-13B | Video-ChatGPT | PG-Video-LLaVA | Gain (7B) |
|--------|------|---------|----------|---------------|-----------------|----------|
| MSVD-QA | Acc | 68.4 | 71.6 | 64.9 | 64.1 | +3.5 |
| MSRVTT-QA | Acc | 56.4 | 57.7 | 49.3 | 51.6 | +4.8 |
| ActivityNet-QA | Acc | 42.0 | 42.2 | 35.2 | 39.9 | +2.1 |

### Main Results — Temporal Localization & Dense Captioning (ActivityNet)

| Task | Metric | PiTe-7B | PiTe-13B | Video-ChatGPT |
|------|------|---------|----------|---------------|
| Temporal Localization | R@0.3 | 30.4 | 37.2 | 26.4 |
| Temporal Localization | R@0.5 | 17.8 | 23.7 | 13.6 |
| Temporal Localization | mIoU | 22.0 | 26.0 | 18.9 |
| Dense Captioning | CIDEr | 21.7 | 26.5 | 5.8 |
| Dense Captioning | METEOR | 5.8 | 6.6 | 2.1 |

### Ablation Study

| Configuration | MSVD Acc | R@0.3 | mIoU | CIDEr | Description |
|------|----------|-------|------|-------|------|
| PiTe (full) | 68.4 | 30.4 | 22.0 | 21.7 | Full model |
| w/o Initialization | 68.2 | 22.8 | 17.1 | 21.7 | Trajectory projector not initialized by localization projector |
| w/o Trajectory Alignment | 68.1 | 23.9 | 17.4 | 21.4 | Removing the entire trajectory alignment stage |

### Key Findings
- Trajectory alignment yields the most substantial gain for temporal localization (mIoU increases from 17.4 to 22.0), while its improvement on QA is limited.
- The initialization strategy is crucial; omitting initialization yields worse results than skipping trajectory training altogether (unstable parameters hinder temporal perception).
- The number of tracking points $P=3$ exhibits the most stable performance across multiple tasks.
- Dense captioning CIDEr skyrockets from 5.8 to 21.7, demonstrating that pixel-level alignment significantly enhances fine-grained generation capabilities.

## Highlights & Insights
- **Significant Dataset Contribution**: PiTe-143k fills the gap of lacking object trajectory annotations in video-language datasets, with an extensible automatic annotation pipeline. It features 1.02M motion trajectories, highlighting its large scale.
- **Projector Initialization as a Key Trick**: Initializing the trajectory projector by repeatedly concatenating the weights of the image localization projector elegantly addresses the dimension expansion problem from 2D coordinates to a $P \times N$ dimensional trajectory matrix. This design is highly intuitive yet remarkably effective.
- **Substantial Leads in Temporal Localization and Dense Captioning**: Compared to methods of equivalent LLM scale, the temporal localization mIoU increases by 3.1, and the dense captioning CIDEr improves by 15.9. This implies that the gain of trajectory alignment for temporal understanding far outweighs its gain for QA.

## Limitations & Future Work
- Only 100 frames are sampled, which provides insufficient coverage for ultra-long videos.
- The trajectory annotation pipeline relies on the quality of GLaMM and DOT; small objects (e.g., pens) are skipped directly if they are difficult to detect.
- Evaluated solely under zero-shot settings, lacking comparison with supervised methods on temporal localization.
- Trajectory alignment brings limited gains (~3.5%) to QA tasks, indicating that QA may depend more on high-level semantics rather than pixel-level alignment.

## Related Work & Insights
- **vs Video-ChatGPT**: PiTe achieves a 15.9 CIDEr improvement in dense captioning via trajectory alignment, demonstrating that fine-grained alignment significantly outperforms pure instruction tuning on generation tasks.
- **vs PixelLLM**: While PixelLLM associates modalities using word coordinates in images, PiTe extends this paradigm to the spatiotemporal dimensions of video.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of trajectory alignment is novel, and the dataset construction pipeline is thorough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on six datasets across three tasks under zero-shot settings, with comprehensive ablation analyses.
- Writing Quality: ⭐⭐⭐ The notation system is slightly disorganized, and some English expressions lack smoothness.
- Value: ⭐⭐⭐⭐ The dataset and alignment paradigm serve as valuable references for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](../../CVPR2025/video_understanding/on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[ECCV 2024\] VideoMamba: Spatio-Temporal Selective State Space Model](videomamba_spatio-temporal_selective_state_space_model.md)
- [\[CVPR 2026\] Towards Streaming Referring Video Segmentation via Large Language Model](../../CVPR2026/video_understanding/towards_streaming_referring_video_segmentation_via_large_language_model.md)
- [\[ECCV 2024\] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](../../NeurIPS2025/video_understanding/lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)

</div>

<!-- RELATED:END -->
