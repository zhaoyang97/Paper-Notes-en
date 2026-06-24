---
title: >-
  [Paper Note] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models
description: >-
  [CVPR 2025][Video Understanding][Visual Token Pruning] Reformulates the visual token pruning problem as the **Max-Min Diversity Problem (MMDP)**. By solving it precisely to **maximize the minimum pair-wise distance** within the retained token set, a training-free and calibration-free plug-and-play pruning scheme is achieved. It yields SOTA performance on 16 multimodal benchmarks, significantly outperforming all baselines particularly under extreme pruning rates of $\ge 80\%$.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Visual Token Pruning"
  - "Large Multimodal Models"
  - "Diversity Maximization"
  - "Max-Min Diversity Problem"
  - "Training-Free"
date: 2026-05-08
content_hash: fea4fd394ac7bd7d
---

# DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models

**Conference**: CVPR 2025  
**arXiv**: [2503.02175](https://arxiv.org/abs/2503.02175)  
**Code**: [https://github.com/vbdi/divprune](https://github.com/vbdi/divprune)  
**Area**: Video Understanding  
**Keywords**: Visual Token Pruning, Large Multimodal Models, Diversity Maximization, Max-Min Diversity Problem, Training-Free

## TL;DR

Reformulates the visual token pruning problem as the **Max-Min Diversity Problem (MMDP)**. By solving it precisely to **maximize the minimum pair-wise distance** within the retained token set, a training-free and calibration-free plug-and-play pruning scheme is achieved. It yields SOTA performance on 16 multimodal benchmarks, significantly outperforming all baselines particularly under extreme pruning rates of $\ge 80\%$.

## Background & Motivation

Large Multimodal Models (LMMs) encode text and visual information into tokens and feed them into LLMs. The number of visual tokens usually far exceeds that of text tokens (e.g., LLaVA 1.5 defaults to 576 visual tokens), leading to a dramatic increase in LLM inference latency and memory consumption, as the complexity of the attention mechanism is quadratic to the input length.

**Limitations of Prior Work**:
- **Attention-based methods** (FastV, PruMerge): Rely on attention scores to measure token importance, but this has been proven suboptimal—certain crucial tokens are overlooked, and the retained tokens are **highly redundant** (high similarity), causing performance to plunge under high pruning rates.
- **Calibration/Fine-tuning-required methods** (FitPrune, VTW, $M^3$): Achieve higher accuracy but require extra calibration datasets or model fine-tuning, resulting in **high deployment costs** and a lack of generalization across different models.

**Key Challenge**: A vast amount of informational redundancy exists in visual tokens (prior studies indicate 50%-95% can be pruned), but existing methods fail to preserve sufficiently diverse tokens to represent the original information under high pruning rates.

**Core Idea**: Instead of focusing on the "importance" of individual tokens, the focus should be on the **diversity** of the retained set—selecting a subset of tokens that are mutually most dissimilar naturally represents the original token distribution best.

## Method

### Overall Architecture

DivPrune is applied after the visual encoder output (or LLM intermediate layers) in LMMs: it calculates the pair-wise cosine distance matrix among all visual tokens, solves the MMDP problem to select a subset of $\tilde{M}$ most diverse tokens, discards the rest, and feeds the selected tokens along with text tokens into the LLM. The entire process requires no training or calibration data, operating in a plug-and-play manner.

### Key Designs

1. **Max-Min Diversity Problem (MMDP) Modeling**
    - **Function**: Reformulates token pruning into a combinatorial optimization problem, mathematically ensuring the maximum diversity of the retained set.
    - **Mechanism**: Seeks a subset $\tilde{\mathbf{E}}_v$ that maximizes the minimum distance between any two of its elements: $\tilde{\mathbf{E}}_v = \text{arg max}[\min_{\gamma,\omega \in S} d(\gamma,\omega)]$, where the distance metric is cosine distance $d(\gamma,\omega) = 1 - \frac{\gamma \cdot \omega}{\|\gamma\|\|\omega\|}$.
    - **Design Motivation**: Maximizing diversity is equivalent to minimizing redundancy in the retained set. Attention-based methods tend to keep tokens that are highly correlated with the query but highly similar to each other, resulting in narrow information coverage; in contrast, the diversity objective ensures that tokens better "cover" the distribution space of the original tokens.

2. **Two-Stage Greedy Solver Algorithm**
    - **Function**: Solves the MMDP efficiently and accurately.
    - **Mechanism**:
     - **First Stage** (First Token Selection): Iterates through all tokens, calculates the minimum distance from each token to all other tokens, and selects the token with the largest minimum distance as the seed.
     - **Second Stage** (Iterative Addition): In each iteration, for each token in the candidate list, calculates its minimum distance to the already selected set, adds the token with the maximum value to the selected set, and repeats until the target quantity is met.
    - **Design Motivation**: Since the number of tokens is limited (e.g., 576), re-computation is avoided by pre-calculating the distance matrix. The extra GPU overhead is negligible compared to the LLM computation.

3. **Flexible Action Location**
    - **Function**: Supports applying pruning at either the visual encoder output layer or the LLM's intermediate decoding layers.
    - **Mechanism**: Can be applied either to the projected visual tokens (before the LLM) or to the portion of hidden states corresponding to visual tokens output by a specific LLM layer.
    - **Design Motivation**: Different models and tasks may yield varying pruning performance at different locations; this flexibility allows DivPrune to adapt to diverse LMM architectures.

## Key Experimental Results

### Main Results (LLaVA 1.5-7B, TFLOP ratio $\approx 15.6\%$, i.e., pruning ~84% visual tokens)

| Method | COCO (CIDEr) | GQA (EM) | MMBench (Acc) | OKVQA (EM) | POPE (F1) | SeedB (Acc) |
|------|-------------|---------|-------------|-----------|----------|-----------|
| Original Model | 1.10 | 61.96 | 64.09 | 53.39 | 85.84 | 66.17 |
| VTW | 0.05 | 38.94 | 21.31 | 18.64 | 25.35 | 36.13 |
| FastV | 0.06 | 38.73 | 20.62 | 18.32 | 32.84 | 35.69 |
| FitPrune (calibration required) | 0.90 | 52.39 | 57.65 | 42.53 | 60.89 | 54.84 |
| **DivPrune** | **0.96** | **56.85** | **59.19** | **46.98** | **86.02** | **59.47** |

- On POPE, DivPrune even outperforms the unpruned original model (86.02 vs 85.84).
- Under an ~85% pruning rate, DivPrune achieves massive performance gains compared to FastV/VTW (e.g., OKVQA: 47 vs 18).

### Video Understanding Experiments (LLaVA-NeXT-Video-7B)

DivPrune consistently outperforms FastV across 5 video benchmarks including MVBench, VideoMME, and MLVU, achieving an average improvement of around 2-5 percentage points.

### Efficiency Gains

| Metric | Original | DivPrune (50% Pruned) | DivPrune (75% Pruned) |
|------|---------|-------------------|-------------------|
| Latency (s) | Baseline | Reduced by ~30% | Reduced by ~50% |
| GPU Memory | Baseline | Reduced by ~20% | Reduced by ~40% |

### Ablation Study

- Distance Metric: Cosine Distance > Euclidean Distance > L1 Distance.
- Pruning Location: Pruning after the 2nd LLM layer yields the best results (LLaVA 1.5-7B).
- Minimum Distance Distribution: The minimum pair-wise distance of the token subset retained by DivPrune is significantly higher than that of FastV (histogram analysis), validating the diversity hypothesis.

### Key Findings

- t-SNE visualization demonstrates that tokens retained by FastV cluster in local areas of the feature space, whereas tokens retained by DivPrune cover the entire feature space uniformly.
- The higher the pruning rate ($\ge 80\%$), the more pronounced the advantages of DivPrune over baselines.
- Consistent performance on LLaVA 1.5-13B indicates the model-agnostic nature of the method.

## Highlights & Insights

- **Elegant Problem Formulation**: Models token pruning as the classic combinatorial optimization problem MMDP, offering a novel perspective with mathematical guarantees.
- **Plug-and-Play Engineering Value**: Requires no training or calibration datasets, generalizes to any LMM architecture, and is compatible with inference optimization techniques like KV caching.
- **Counter-Intuitive Discovery**: The "importance" of tokens (attention scores) is less critical than the "diversity" of tokens (mutual dissimilarity)—high-attention tokens are often highly redundant with one another.

## Limitations & Future Work

- Although the overhead of solving the MMDP is negligible with current token counts (~576), the computational cost may become non-negligible if the token count scales up dramatically (e.g., thousands of tokens in high-resolution LMMs).
- The diversity objective completely ignores the semantic importance of tokens, which might underperform compared to attention-based methods in tasks that require focusing on specific regions (such as fine-grained recognition).
- The method has only been verified on LMMs using CLIP visual encoders; its effectiveness on other visual encoders (e.g., SigLIP, InternViT) remains to be validated.

## Related Work & Insights

- **FastV** and **PruMerge** represent the mainstream attention-based paradigm; DivPrune exposes their fundamental flaw of "focusing on importance while ignoring redundancy".
- Although **FitPrune** and **M³** show promising performance, they require extra training/calibration, whereas DivPrune offers a training-free alternative.
- The concept of diversity optimization can be extended to other scenarios, such as **KV cache compression** (preserving diverse key-value pairs) and **video frame sampling** (selecting diverse keyframes).

## Rating

⭐⭐⭐⭐ — Elegantly modeling the token pruning problem as an MMDP is the core highlight of this paper. The method is simple, efficient, and thoroughly evaluated (16 datasets $\times$ 4 models). Its substantial advantages under high pruning rates and plug-and-play characteristics make it highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](../../CVPR2026/video_understanding/utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)

</div>

<!-- RELATED:END -->
