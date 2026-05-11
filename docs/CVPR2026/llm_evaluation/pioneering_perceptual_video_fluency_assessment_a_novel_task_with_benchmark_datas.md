---
title: >-
  [Paper Note] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline
description: >-
  [CVPR 2026][LLM Evaluation][Video Fluency Assessment] This paper formally separates Video Fluency Assessment (VFA) from conventional Video Quality Assessment (VQA) for the first time…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Video Fluency Assessment"
  - "Temporal Quality"
  - "Benchmark Dataset"
  - "Self-Attention"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: 4d743c130073accb
---

# Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline

**Conference**: CVPR 2026
**arXiv**: [2603.26055](https://arxiv.org/abs/2603.26055)
**Code**: [https://github.com/KeiChiTse/VFA](https://github.com/KeiChiTse/VFA)
**Area**: Video Understanding / Video Quality Assessment
**Keywords**: Video Fluency Assessment, Temporal Quality, Benchmark Dataset, Self-Attention, Self-Supervised Learning

## TL;DR

This paper formally separates Video Fluency Assessment (VFA) from conventional Video Quality Assessment (VQA) for the first time, introduces FluVid — the first fluency-oriented benchmark dataset (4,606 videos) — and proposes a baseline model FluNet that leverages Temporal Permuted Self-Attention (T-PSA) for efficient inter-frame interaction, achieving SRCC/PLCC of 0.816/0.821.

## Background & Motivation

**Background**: Video Quality Assessment (VQA) is the dominant paradigm for quantifying subjective video perception, with models such as Fast-VQA and DOVER being widely adopted. Existing VQA approaches jointly assess spatial quality (noise, color, etc.) and temporal quality (motion consistency, frame continuity, etc.) in a holistic manner.

**Limitations of Prior Work**: Through pilot experiments, the authors demonstrate that predictions from existing VQA models are strongly biased toward spatial quality, while their capacity to assess the temporal dimension — i.e., fluency — is severely lacking. Consequently, VQA scores fail to effectively guide temporally oriented downstream tasks such as adaptive frame-rate coding and frame interpolation.

**Key Challenge**: The spatial–temporal entanglement inherent in VQA models substantially dilutes the fluency signal. Although the human visual system is more sensitive to temporal distortions than to spatial ones, model outputs exhibit precisely the opposite behavior. Three fundamental factors underlie this problem: (1) the absence of an independent fluency scoring standard; (2) the lack of a large-scale fluency-annotated dataset; and (3) the absence of model architectures specifically designed for fluency assessment.

**Goal**: To formalize VFA as an independent perceptual task; to construct the first fluency scoring standard and dataset, FluVid; and to design a fluency-aware baseline model, FluNet.

**Key Insight**: Drawing on visual psychophysics and cognitive science, perceptual fluency is determined by three core video components — foreground, background, and camera motion. Meanwhile, the primary bottlenecks of existing methods are insufficient input frame count and inadequate inter-frame interaction.

**Core Idea**: A channel-compression combined with temporal-dimension permutation self-attention mechanism (T-PSA) substantially expands the temporal window while keeping computation tractable, coupled with a self-supervised ranking training strategy that enables the model to perceive fluency differences.

## Method

### Overall Architecture

FluNet consists of three components: a patch embedding layer $F_p$ (single convolutional layer), an encoder $F_e$ (a four-stage Transformer incorporating T-PSA blocks), and a VFA prediction head $F_h$ (two pointwise convolutional layers). An input video $V \in \mathbb{R}^{T \times H \times W \times 3}$ is first mapped to a feature map by $F_p$, then progressively encoded, and finally regressed to a fluency score via $F_h$. The overall architecture follows the hierarchical design of Swin Transformer, with (2, 2, 6, 2) T-PSA blocks distributed across four stages.

### Key Designs

1. **Temporal Permuted Self-Attention (T-PSA)**

    - **Function**: Expands the temporal receptive field while preserving computational efficiency.
    - **Mechanism**: In standard self-attention, $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ all have channel dimension $C$. T-PSA compresses the channel dimension of $\mathbf{K}$ and $\mathbf{V}$ to $C/\gamma$ ($\gamma=2$), then permutes temporal tokens into the channel dimension, transforming the attention window of $\mathbf{K}_p$ and $\mathbf{V}_p$ from $(D, S, S)$ to $(D/\gamma, S, S)$ while restoring the channel dimension to $C$, thereby enabling normal attention computation with $\mathbf{Q}$. This allows the temporal window to expand from 8 to 32 while GFLOPs actually decrease from 1114 to 308.
    - **Design Motivation**: Conventional methods with sparse 32-frame sampling cannot capture subtle fluency variations; directly increasing frame count causes computational explosion. T-PSA exclusively expands the temporal window $D$ while fixing the spatial window $S$, effectively implementing a "focus on fluency rather than spatial detail" paradigm.

2. **Self-Supervised Ranking Training Strategy**

    - **Function**: Enables the model to learn fluency level discrimination without fluency annotations.
    - **Mechanism**: 2,000 high-quality anchor videos are sampled from the HD-VILA dataset. For each anchor, $K=7$ videos at distinct fluency levels are synthesized via random frame dropping and frame duplication. The drop rate increases with level, and drop positions are randomly distributed across $M=5$ temporal intervals. The ranking loss is a margin ranking loss: $\mathcal{L}_{\text{rank}} = \frac{1}{K}\sum_{i=0}^{K-1}\max(0, \hat{y}_{i+1} - \hat{y}_i + \beta)$, where $\beta=0.4$.
    - **Design Motivation**: Fluency annotation requires expert annotators in controlled laboratory settings, making it extremely costly. Ranking learning over synthesized video pairs at varying fluency levels enables training the model's fluency ordering capability using unlabeled data.

3. **FluVid Dataset Construction**

    - **Function**: Provides the first benchmark dataset centered on fluency assessment.
    - **Mechanism**: Two design principles guide the construction: (1) videos are collected according to the three video components that govern fluency (foreground/background/camera); (2) content and parameter diversity are ensured. A total of 4,606 videos are selected from SSv2 and five UGC-VQA datasets, and 20 expert annotators assign fluency MOS scores using the 5-level ACR standard.
    - **Design Motivation**: Existing VQA datasets focus on overall quality and lack fluency-centric data and annotations, making them insufficient to support VFA model training and evaluation.

### Loss & Training

Training proceeds in three stages: (1) optional LSVQ pre-training to endow the model with quality perception capability (yielding FluNet++); (2) ranking learning using $\mathcal{L}_{\text{rank}}$ on 16,000 synthesized videos for 30 epochs; (3) fine-tuning using L1 loss $\mathcal{L}_{\text{ft}} = \|\hat{y}_b - y_b\|_1$ on 606 FluVid videos for 60 epochs.

## Key Experimental Results

### Main Results

| Method | Type | Frames | Window Size | GFLOPs | SRCC↑ | PLCC↑ |
|--------|------|--------|-------------|--------|-------|-------|
| Fast-VQA | VQA | 32 | (8,7,7) | 279 | 0.640 | 0.633 |
| DOVER | VQA | - | - | - | 0.638 | 0.614 |
| Qwen 2.5-VL | LMM | - | - | - | 0.598 | 0.584 |
| FineVQ | LMM | - | - | - | 0.622 | 0.609 |
| Fast-VQA+128 frames+Ranking+FT | VQA | 128 | (8,7,7) | 1114 | 0.725 | 0.716 |
| **FluNet (Ours)** | VFA | 128 | (32,7,7) | 308 | **0.774** | **0.770** |
| **FluNet++ (Ours)** | VFA | 128 | (32,7,7) | 308 | **0.816** | **0.821** |

### Ablation Study

| Configuration | SRCC↑ | PLCC↑ | Note |
|---------------|-------|-------|------|
| Ranking only | 0.722 | 0.718 | Ranking learning is effective |
| Fine-tuning only | 0.710 | 0.693 | Fine-tuning is also effective |
| Joint training | 0.753 | 0.748 | Joint training inferior to staged |
| Ranking → Fine-tuning | 0.774 | 0.770 | Staged training is optimal |
| Window (8,7,7) | 0.736 | 0.722 | Smaller window degrades performance |
| Window (16,7,7) | 0.758 | 0.749 | Larger window consistently better |
| Window (32,7,7) | 0.774 | 0.770 | Optimal window size |
| T-PSA in stages 1–3 | 0.779 | 0.766 | Best stage configuration |
| T-PSA in all 4 stages | 0.774 | 0.770 | Marginal effect of adding stage 4 |

### Key Findings

- FluNet surpasses Fast-VQA (+4.9% SRCC) at only 308 GFLOPs versus 1114 GFLOPs, demonstrating the computational efficiency of T-PSA.
- Increasing the number of input frames (32→128) benefits all methods, but T-PSA's distinctive advantage lies in simultaneously enabling a larger temporal window.
- The staged Ranking→Fine-tuning strategy outperforms joint training, indicating that learning to rank before calibrating scores constitutes a superior learning trajectory.
- VQA methods consistently outperform LMMs, suggesting that fine-grained quality perception is more critical than general semantic understanding; among LMMs, Qwen 2.5-VL performs best, benefiting from its high frame-rate processing capability.

## Highlights & Insights

- **The channel–temporal dimension permutation in T-PSA** is a particularly elegant design. By compressing the channel dimension of K/V and then permuting temporal tokens into the channel dimension, it achieves a 4× temporal window expansion without increasing computational cost. This idea is directly transferable to any video task requiring long-range temporal modeling.
- **The insight of decoupling VFA from VQA** carries substantial value in its own right. The authors quantitatively demonstrate that VQA models are biased toward spatial quality — a finding that is equally instructive for video generation evaluation, suggesting that current VQA-based metrics may systematically underestimate temporal artifacts.
- **The synthetic ranking training** elegantly addresses annotation scarcity; although the frame-dropping and frame-duplication synthesis is simple, it effectively simulates real-world stuttering artifacts.

## Limitations & Future Work

- FluVid contains only 4,606 videos, representing a limited scale, and the dataset is sourced predominantly from UGC content, with no coverage of AI-generated video.
- The synthetic ranking training data simulates only frame-drop stuttering and does not cover other fluency degradations such as motion blur or unstable frame rates.
- The channel compression ratio $\gamma$ in T-PSA is fixed at 2, and adaptive compression strategies remain unexplored.
- Joint VFA–VQA prediction, which may be more practically valuable, has not been investigated.

## Related Work & Insights

- **vs. Fast-VQA**: Fast-VQA employs sparse frame sampling with fixed-window attention. FluNet achieves a larger temporal window and higher frame count at equivalent computational cost via T-PSA, improving SRCC from 0.640 to 0.774.
- **vs. LMM methods (Qwen 2.5-VL, FineVQ)**: LMMs possess strong semantic understanding but lack fine-grained fluency perception. Quality-aware LMMs (e.g., FineVQ) outperform general-purpose LMMs yet still fall short of purpose-built VFA approaches.
- This work offers significant implications for video generation evaluation: fluency may be a severely neglected dimension in current generation quality metrics such as FVD.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to define the VFA task and build a complete ecosystem (standard + data + method); T-PSA design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive benchmark across 23 methods, thorough ablation studies, and multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, overall structure is well-organized, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐ Fills the gap in fluency assessment and provides practical guidance for video generation quality evaluation and video processing optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [\[CVPR 2026\] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](../../ICLR2026/llm_evaluation/anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](../../NeurIPS2025/llm_evaluation/houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)

</div>

<!-- RELATED:END -->
