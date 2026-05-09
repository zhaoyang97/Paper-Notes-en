---
title: >-
  [Paper Note] Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective
description: >-
  [ICLR 2026][Multimodal VLM][Diffusion language models] This work investigates the underlying mechanism behind the "Repetition Curse" in diffusion multimodal large language models (dMLLMs) when cache-based acceleration is applied, through an information flow perspective. It reveals that context tokens act as anchors that aggregate semantic information, and that caching disrupts this information flow pattern. The proposed CoTA method reduces repetition rates by up to 92%.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Diffusion language models
  - repetitive generation
  - information flow analysis
  - cache acceleration
  - attention mechanism
date: 2026-05-08
content_hash: 913350eddcdf2fc6
---

# Context Tokens are Anchors: Understanding the Repetition Curse in dMLLMs from an Information Flow Perspective

**Conference**: ICLR 2026
**arXiv**: [2601.20520](https://arxiv.org/abs/2601.20520)
**Code**: [GitHub](https://github.com/ErikZ719/CoTA)
**Area**: Multimodal VLM
**Keywords**: Diffusion language models, repetitive generation, information flow analysis, cache acceleration, attention mechanism

## TL;DR

This work investigates the underlying mechanism behind the "Repetition Curse" in diffusion multimodal large language models (dMLLMs) when cache-based acceleration is applied, through an information flow perspective. It reveals that context tokens act as anchors that aggregate semantic information, and that caching disrupts this information flow pattern. The proposed CoTA method reduces repetition rates by up to 92%.

## Background & Motivation

**Inference efficiency in dMLLMs**: Diffusion-based large language models (e.g., LLaDA) generate tokens in parallel through iterative denoising, requiring attention computation over the full sequence at each step. Due to the use of bidirectional attention, conventional KV-cache is inapplicable, resulting in high inference latency.

**Side effects of cache acceleration**: Methods such as dLLM-Cache exploit the observation that attention values for prefix and suffix tokens change little across iterations to enable cache reuse, effectively reducing latency but causing severe text repetition—referred to by the authors as the "Repeat Curse."

**Quantitative metrics**: Four complementary metrics are proposed to measure repetition: Adjacent Repetition Rate (ARR), Sample Repetition Rate (SRR), Maximum Repetition Length (MRL), and Average Repetition Length (ARL).

**Three key information flow findings**:
- Context tokens (the target token and its nearest neighbors) serve as anchors in dMLLMs, progressively aggregating semantic information across layers and absorbing attention.
- Under normal decoding, the information entropy of context tokens converges in deeper layers, reflecting increasing prediction certainty.
- Caching randomizes attention allocation, preventing context token entropy from converging in deep layers, which leads to repetition.

## Method

### Overall Architecture

CoTA is a plug-and-play, training-free method consisting of two complementary modules applied during inference.

### CTAE (Context-token Attention Enhancement)

A Gaussian decay term based on relative distance is multiplied element-wise with the attention matrix to enhance attention toward context tokens, preserving the original information flow pattern:

$$\mathcal{G}_{i,j} = \gamma_{\min} + (1-\gamma_{\min}) \exp\left(-\left(\frac{|i-j|}{\tau}\right)^2\right)$$

- Modified attention: $\tilde{A}_{i,j} = A_{i,j} \cdot \mathcal{G}_{i,j}$
- Temperature factor $\tau=5$; lower bound $\gamma_{\min} \in (0,1]$ ensures stability.

### CTEV (Context-token Entropy-Guided Voting)

The cumulative entropy of context tokens in deep layers (layers 26–30) is used as a penalty term incorporated into the confidence score:

$$\text{Score}(i) = c_{(i)} + \alpha \cdot E_{sum}^{ctx}(i)$$

where $E_{sum}^{ctx}(i) = \sum_{j \in \mathcal{C}(i)} \sum_{l=26}^{30} E^{(l)}(j)$, and $\mathcal{C}(i)$ denotes the target token and its two nearest neighbors.

### Collaborative Workflow

CTAE preserves the information flow pattern → CTEV penalizes voting scores for uncertain context tokens → jointly mitigating repetition.

## Key Experimental Results

### Main Results (LLaDA-V 8B, COCO Caption)

| Method | ARR↓(512) | MRL↓(512) | SRR↓(512) | ARR↓(64) |
|--------|-----------|-----------|-----------|----------|
| LLaDA-V Baseline | 0.2 | 2.0 | 6.9 | 0.1 |
| +Cache | 14.3 | 11.0 | 82.3 | 7.1 |
| +Cache+CTAE | 3.5 | 3.2 | 22.1 | 2.8 |
| +Cache+CTEV | 4.1 | 3.8 | 25.3 | 3.2 |
| **+Cache+CoTA** | **1.2** | **1.3** | **6.3** | **1.0** |

### Multimodal Benchmarks

| Benchmark | LLaDA-V | +Cache | +Cache+CoTA |
|-----------|---------|--------|-------------|
| DocVQA | 78.2 | 72.6 | 76.9 |
| MMStar | 55.1 | 49.3 | 54.2 |
| MME | 1892 | 1645 | 1856 |

### Ablation Study

| Configuration | ARR↓(512) | Notes |
|---------------|-----------|-------|
| CTAE only | 3.5 | Each module is individually effective |
| CTEV only | 4.1 | Each module is individually effective |
| **CTAE+CTEV** | **1.2** | Complementary synergy yields best results |
| CTAE τ=3 | 2.1 | Window too narrow |
| CTAE τ=5 | 1.2 | Optimal temperature |
| CTAE τ=10 | 1.8 | Window too wide |

### Key Findings

- CoTA reduces ARR by up to 92% (14.3→1.2), nearly recovering the no-cache baseline performance.
- Long sequences (512 tokens) are more susceptible to repetition than short ones (64 tokens), due to the cumulative effect of caching.
- CTAE and CTEV address complementary aspects: attention enhancement preserves the information flow pattern, while entropy-guided voting prevents incorrect token selection.
- CoTA recovers cache-induced performance degradation across 8 multimodal benchmarks.

## Highlights & Insights

- **First systematic analysis of cache-induced repetition in dMLLMs**: A clear causal chain is established from an information flow perspective.
- Context tokens are found to play a role analogous to "attention sinks" in autoregressive models under bidirectional attention—suggesting a universal phenomenon across model architectures.
- The method is elegant in its simplicity: **fully training-free, plug-and-play, and computationally lightweight**.
- The proposed quantitative metrics (ARR/SRR/MRL/ARL) provide a standardized evaluation framework for future research.

## Limitations & Future Work

- Validation is limited to LLaDA-V; generalizability to other diffusion LMs (e.g., dMDT, MDLM) remains to be examined.
- The temperature $\tau=5$ and the definition of deep layers (26–30) are empirically determined and may require adjustment for different architectures.
- Effectiveness on very long sequences (>1024 tokens) and multi-turn dialogue scenarios has not been sufficiently verified.
- The Gaussian decay in CTAE assumes stronger semantic relevance among neighboring tokens, which may not hold in scenarios with long-range references.

## Related Work & Insights

- **Attention sinks in autoregressive models**: Chen et al. and Wang et al. identify special tokens that aggregate attention in AR models; this work is the first to discover an analogous mechanism in dMLLMs.
- **dLLM-Cache / TinyCache**: These acceleration methods are effective in reducing latency, but CoTA identifies and remedies their side effects.
- **ADLM**: Discusses the semantic guidance role of anchors in diffusion language models, corroborating the findings of this work.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel information flow analysis perspective with clearly defined problem formulation
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations, multi-benchmark validation, and thorough quantitative metrics
- Writing Quality: ⭐⭐⭐⭐ Clear visualizations and well-articulated causal reasoning
- Value: ⭐⭐⭐⭐ Addresses a practical problem with a plug-and-play solution of high industrial relevance

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[AAAI 2026\] CAMU: Context Augmentation for Meme Understanding](../../AAAI2026/multimodal_vlm/trace_textual_relevance_augmentation_and_contextual_encoding_for_multimodal_hate.md)
- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](../../NeurIPS2025/multimodal_vlm/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](liveweb-ie_a_benchmark_for_online_web_information_extraction.md)

<!-- RELATED:END -->
