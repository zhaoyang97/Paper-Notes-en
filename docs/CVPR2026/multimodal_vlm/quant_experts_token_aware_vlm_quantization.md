---
title: >-
  [Paper Note] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization
description: >-
  [CVPR 2026][Multimodal VLM][post-training quantization] This paper proposes Quant Experts (QE), a token-aware adaptive quantization error reconstruction framework based on Mixture-of-Experts. It partitions important chan…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "post-training quantization"
  - "VLM"
  - "MoE"
  - "token-aware"
  - "low-rank adapter"
  - "channel importance"
date: 2026-05-08
content_hash: 360da294c993b4b0
---

# Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization

**Conference**: CVPR 2026
**arXiv**: [2602.24059](https://arxiv.org/abs/2602.24059)
**Code**: N/A
**Area**: Model Compression / VLM Quantization
**Keywords**: post-training quantization, VLM, MoE, token-aware, low-rank adapter, channel importance

## TL;DR

This paper proposes Quant Experts (QE), a token-aware adaptive quantization error reconstruction framework based on Mixture-of-Experts. It partitions important channels into token-independent (high-frequency, globally consistent) and token-dependent (low-frequency, locally dynamic) groups, compensating global and local quantization errors via low-rank adapters in shared and routed experts, respectively. QE consistently improves VLM performance across diverse quantization settings ranging from W4A6 to W3A16.

## Background & Motivation

**Background**: Post-Training Quantization (PTQ) is a key technique for reducing the computational and memory overhead of large vision-language models (VLMs). Existing approaches include channel smoothing (SmoothQuant, AWQ), mixed-precision quantization (SpQR), Hessian-based optimization (GPTQ), and low-rank reconstruction (LQER, ASER). In multimodal settings, MBQ reveals cross-modal channel sensitivity discrepancies and proposes modality-aware channel scaling.

**Limitations of Prior Work**: (1) Channel smoothing methods (SmoothQuant/AWQ) apply fixed scaling factors estimated from calibration data uniformly to all tokens, failing to capture token-level variations in channel importance. (2) Static low-rank reconstruction methods (LQER/ASER) employ a single global adapter for all important channels, ignoring the dynamic nature of channel importance. (3) Although MBQ differentiates cross-modal discrepancies, it still relies on static channel scaling without accounting for intra-modal channel importance fluctuations across different tokens.

**Key Challenge**: The locations of important channels are not static — they shift not only across modalities but, more critically, across different tokens within the same modality due to changes in activation distributions driven by token semantics and contextual information. Globally fixed channel identification and compensation strategies are fundamentally incapable of capturing this token-level dynamics.

**Goal**: To design a framework that simultaneously handles globally consistent (token-independent) and locally dynamic (token-dependent) quantization errors, precisely compensating for the distinct quantization losses faced by different tokens.

**Key Insight**: Important channels are partitioned by occurrence frequency into two groups — high-frequency token-independent channels are globally compensated via a shared expert, while low-frequency token-dependent channels are clustered by co-occurrence patterns and dynamically compensated via routed experts.

**Core Idea**: Drawing inspiration from MoE, a two-level "shared expert + routed experts" architecture is employed to compensate for global quantization errors and token-dependent local errors separately.

## Method

### Overall Architecture

The QE framework proceeds in three stages: (1) estimating the importance frequency distribution of each channel from calibration data and partitioning channels into token-independent and token-dependent groups; (2) applying a Shared Expert (SE) with a whitening SVD low-rank adapter to globally reconstruct quantization errors from token-independent channels while performing channel scaling to suppress activation outliers; (3) employing multiple Routed Experts (REs) trained via co-occurrence clustering and weighted SVD to locally compensate errors for different subgroups of token-dependent channels, with a lightweight router dynamically selecting the optimal expert at inference time.

### Key Designs

1. **Channel Importance Analysis and Partitioning**
   - **Function**: Partition important channels by occurrence frequency into globally consistent and locally dynamic groups.
   - **Mechanism**: For each token $x_t$, the important channel set is computed as $\mathcal{C}_t = \text{Top-}k(|x_t| \odot \mathbf{w})$, where $\mathbf{w} = \text{Mean}_{\text{row}}(|\mathbf{W}_f|)$. The frequency $f_c = k \times \frac{m_c}{\sum_i m_i}$ measures how often each channel is identified as important across all tokens. Channels are sorted by frequency in descending order: the top $k$ form the token-independent set $\mathcal{C}_s$, and the subsequent $N_r \times k$ form the token-dependent set $\mathcal{C}_r$.
   - **Design Motivation**: Empirical observations show that only a small number of channels appear consistently as important across most tokens (suitable for global compensation), while the majority of important channels exhibit strong input-dependent activation patterns (requiring dynamic compensation).

2. **Shared Expert (SE) for Global Compensation**
   - **Function**: Reconstruct global quantization errors caused by token-independent channels.
   - **Mechanism**: Token-independent channels are excluded from direct quantization and decomposed via whitening SVD into a low-rank adapter $(\mathbf{L}_{SA}^l, \mathbf{L}_{SB}^l)$ for reconstruction. Channel scaling is also applied to reduce activation outlier magnitudes (with corresponding weight scaling applied inversely), suppressing activation quantization errors. After SE processing, the residual error $\mathbf{E}_S^l = \mathbf{E}^l - \mathbf{L}_{SA}^l \mathbf{L}_{SB}^l$ is passed to the routed experts for further refinement.
   - **Design Motivation**: High-frequency channels are the dominant contributors to quantization error; accurate reconstruction via low-rank adapters follows the design rationale of LQER/ASER. Channel scaling simultaneously addresses quantization issues on both the weight and activation sides.

3. **Routed Experts (REs) for Dynamic Compensation**
   - **Function**: Dynamically select the optimal local error compensation strategy for different tokens.
   - **Mechanism**: A co-occurrence matrix $\mathcal{O}_{t,i}^l = \mathbf{1}(c_i \in \mathcal{C}_r^l \cap \mathcal{A}_t^l)$ is first constructed for token-dependent channels. Normalized Pointwise Mutual Information (NPMI) $\mathbf{S}_{i,j} = (\log\frac{p(i,j)}{p(i)p(j)}) / -\log p(i,j)$ is used to quantify inter-channel association strength, followed by spectral clustering (normalized Laplacian eigendecomposition + K-Means) to partition channels into $N_r$ subgroups. Each subgroup corresponds to one routed expert, reconstructing residual errors of its channels via weighted SVD. At inference, a lightweight router $\mathbf{R}^l$ predicts the residual magnitude of each expert given the input token and activates the expert with the smallest predicted residual.
   - **Design Motivation**: Ideally, a customized compensation strategy should be tailored to each token, but the combinatorial space is intractable. By clustering channels with similar activation patterns via co-occurrence, a finite number ($N_r = 8$) of experts approximately covers the local error characteristics across all tokens.

### Loss & Training

The core pipeline of QE (channel partitioning, SVD decomposition, clustering) is computed offline without end-to-end training. An optional lightweight refinement strategy trains only the routed experts $(\mathbf{L}_{RA}^l, \mathbf{L}_{RB}^l)$ and router $\mathbf{R}^l$ while freezing all other parameters. Refinement is performed layer-by-layer (not end-to-end): 16 epochs × 100 iterations, AdamW (lr=$1 \times 10^{-4}$), cosine annealing schedule. Calibration set: 128 image-text pairs randomly sampled from ShareGPT4V-augmented COCO Caption data. Total SVD rank is fixed at 64, split equally between shared and routed experts (32 each). $k=32$ important channels, $N_r=8$ routed experts.

## Key Experimental Results

### Main Results (Qwen2VL-2B, average accuracy across 11 multimodal benchmarks)

| Method | #W | #A | MMMU | OCRBench | ScienceQA | TextVQA | Avg. |
|--------|----|----|------|----------|-----------|---------|------|
| Full Precision | 16 | 16 | 39.89 | 74.90 | 76.96 | 77.72 | 62.97 |
| RTN | 4 | 6 | 34.00 | 59.80 | 64.70 | 67.58 | 53.62 |
| SmoothQuant | 4 | 6 | 30.44 | 59.60 | 65.25 | 65.88 | 50.27 |
| LQER | 4 | 6 | 33.00 | 65.80 | 68.32 | 69.37 | 55.92 |
| MBQ | 4 | 6 | 34.44 | 61.10 | 67.08 | 69.45 | 54.73 |
| **QE** | **4** | **6** | **33.78** | **68.20** | **71.84** | **73.18** | **58.74** |

Key results on Qwen2VL-72B:

| Setting | Method | MMMU | OCRBench | ScienceQA | TextVQA | VizWiz |
|---------|--------|------|----------|-----------|---------|--------|
| FP16 | - | 61.44 | 78.70 | 91.22 | 82.26 | 76.27 |
| W4A6 | MBQ | 52.67 | 69.70 | 86.32 | 76.08 | 67.99 |
| W4A6 | **QE** | **58.11** | **76.60** | **90.33** | **79.27** | **73.91** |

### Ablation Study

Contribution of each component (Qwen2VL-2B, W4A6):

| Setting | Components | MMMU↑ | ScienceQA↑ |
|---------|------------|-------|-----------|
| FP16 | - | 39.89 | 76.95 |
| W4A6 | REs only | 34.56 | 68.72 |
| W4A6 | SE only | 35.22 | 69.61 |
| W4A6 | SE + random routing | 35.89 | 70.00 |
| W4A6 | SE + random clustering | 35.33 | 69.71 |
| W4A6 | **QE (SE+REs)** | **36.89** | **70.85** |

### Key Findings

- Under the most challenging W4A6 setting, QE outperforms MBQ by 4.01% on Qwen2VL-2B, falling only 4.23% short of full precision.
- On Qwen2VL-72B W4A6, QE achieves a 5.09% accuracy gain over MBQ, nearly matching full-precision performance.
- Removing either type of expert degrades performance: the shared expert is more critical for global stability, while the routed experts are more essential for precise compensation of specific tokens.
- Random routing (35.89) vs. adaptive routing: random routing underperforms QE (36.89), validating the adaptive selection capability of the router.
- Random clustering (35.33) vs. co-occurrence clustering: random clustering underperforms QE (36.89), validating the effectiveness of NPMI-based spectral clustering for modeling inter-channel associations.
- MBQ's distribution reshaping offers limited improvement over AWQ in weight-only quantization (W3A16) due to excessively strong channel importance dynamics.
- On InternVL2-8B W4A6, QE achieves an average accuracy of 68.13, significantly outperforming LQER (65.29) and MBQ (65.00).

## Highlights & Insights

- **Observation-Driven Design**: Two key observations — that important channel locations vary across tokens and that their frequency distribution is highly uneven — directly motivate the method design. The shared expert corresponds to high-frequency global channels, and the routed experts to low-frequency local channels, forming a clear logical chain.
- **Novel Application of MoE to Quantization**: The "shared + routed" MoE paradigm is adapted for quantization error compensation. Unlike conventional MoE, which increases model capacity, this work employs MoE to adapt to the token-level dynamics of quantization error.
- **NPMI-Based Spectral Clustering**: Normalized Pointwise Mutual Information is used to quantify channel co-occurrence associations, capturing semantic correlation patterns among channels more effectively than naive clustering.
- **Consistent Effectiveness Across 2B to 72B**: The method consistently improves performance across four model scales (2B, 7B, 8B, 72B), demonstrating strong scalability.

## Limitations & Future Work

- The number of routed experts $N_r=8$ and the number of important channels $k=32$ are fixed hyperparameters that may require differentiated tuning across models and layers.
- The introduction of $N_r$ low-rank adapters and a router increases inference-time parameter count and computational overhead, even though the total rank budget remains unchanged.
- Clustering and SVD decomposition are performed independently per layer, without considering cross-layer channel importance correlations.
- The calibration set contains only 128 samples, which may yield insufficiently robust channel frequency estimates for very large models (e.g., 72B).
- The optional refinement strategy requires additional training time, and its isolated contribution is not separately reported in the ablation study.

## Related Work & Insights

- **LQER (ICML'24)**: A pioneer in low-rank quantization error reconstruction; QE builds upon this by introducing token-aware dynamic grouping.
- **MBQ (CVPR'25)**: Reveals cross-modal channel sensitivity discrepancies but still employs static scaling; QE further uncovers token-level dynamics within the same modality.
- **SmoothQuant (ICML'23)**: A classical channel scaling method; QE's shared expert augments this with low-rank reconstruction.
- **Insight**: The evolution of quantization error compensation — from "globally uniform" to "modality-aware" to "token-aware" — shows that each step of finer granularity yields substantial gains. The next frontier may be "position-aware" or "attention-aware" compensation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (Applying MoE to quantization error compensation is a novel perspective; the observation of token-level channel dynamics is valuable.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Five model scales, 11 benchmarks, three quantization settings, complete component ablations.)
- **Writing Quality**: ⭐⭐⭐⭐ (The observation → motivation → method logic chain is clear and well-supported by visualizations.)
- **Value**: ⭐⭐⭐⭐ (VLM quantization is a critical problem for practical deployment; the 5% gain on the 72B model carries significant engineering value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

</div>

<!-- RELATED:END -->
