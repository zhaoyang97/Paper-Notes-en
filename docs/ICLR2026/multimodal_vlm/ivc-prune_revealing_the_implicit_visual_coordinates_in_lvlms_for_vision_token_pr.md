---
title: >-
  [Paper Note] IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning
description: >-
  [ICLR 2026][Multimodal VLM][vision token pruning] This paper reveals the implicit visual coordinate (IVC) system established by RoPE positional encoding within LVLMs, and proposes a training-free…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "vision token pruning"
  - "RoPE positional encoding"
  - "implicit visual coordinates"
  - "spatial reasoning"
  - "training-free"
date: 2026-05-08
content_hash: 348a95850624ecd0
---

# IVC-Prune: Revealing the Implicit Visual Coordinates in LVLMs for Vision Token Pruning

**Conference**: ICLR 2026
**arXiv**: [2602.03060](https://arxiv.org/abs/2602.03060)  
**Code**: [GitHub](https://github.com/FireRedTeam/IVC-Prune)  
**Area**: Multimodal VLM
**Keywords**: vision token pruning, RoPE positional encoding, implicit visual coordinates, spatial reasoning, training-free

## TL;DR

This paper reveals the implicit visual coordinate (IVC) system established by RoPE positional encoding within LVLMs, and proposes a training-free, prompt-aware vision token pruning strategy that preserves IVC tokens and semantic foreground tokens while pruning approximately 50% of visual tokens with ≥99% of original performance retained.

## Background & Motivation

High-resolution image inputs generate thousands of visual tokens for large vision-language models (LVLMs), leading to substantial memory consumption and inference latency. Existing visual token pruning methods primarily focus on semantic relevance, selecting tokens aligned with textual semantics via attention scores or similarity metrics. However, these methods suffer severe degradation on spatially sensitive tasks such as visual grounding and spatial reasoning, because they retain only semantically relevant "foreground" tokens while discarding positional reference tokens that are critical for spatial understanding.

The authors conduct an in-depth analysis of how LVLMs perform spatial reasoning through RoPE. The core finding is that RoPE's rotation matrices approximate an identity matrix (real-axis reference) or a 90° rotation matrix (imaginary-axis reference) at certain token positions, making those tokens serve as natural implicit visual coordinate anchors. Removing these tokens via pruning severely impairs the model's spatial reasoning capability.

## Method

### Overall Architecture

IVC-Prune is a training-free, prompt-aware pruning strategy that operates within the language decoder and preserves two categories of critical visual tokens:
1. **IVC tokens** (Implicit Visual Coordinate tokens): tokens essential for spatial reasoning.
2. **Foreground tokens**: visual tokens semantically aligned with the text prompt.

The final retained set is $\mathcal{I}_{\text{selected}} = \mathcal{I}_{\text{ivc}} \cup \mathcal{I}_{\text{fg}}$.

### Key Designs

#### 1. Theoretical Discovery of Implicit Visual Coordinates

RoPE partitions each $d$-dimensional feature vector into $d/2$ two-dimensional subspaces and applies a rotation at position $m$. The attention score can be written as:

$$\text{Score}(\mathbf{q}_n, \mathbf{k}_m) = \mathbf{x}_n^T \mathbf{W}_q^T \mathbf{R}_{m-n} \mathbf{W}_k \mathbf{x}_m$$

This shows that attention fundamentally depends on relative position $(m-n)$. Spatial reasoning, however, requires absolute positional information. The key insight is: when the rotation matrix $\mathbf{R}_m$ at a key token position $m$ approximates the identity matrix or a 90° rotation matrix, attention effectively isolates the absolute position component $\mathbf{R}_n$ of the query.

**Real-axis reference**: positions where $\mathbf{R}_m \approx \mathbf{I}$, equivalent to maximizing the cosine score:
$$V(m) = \sum_{k=0}^{d/2-1} \cos(m\theta_k)$$

**Imaginary-axis reference**: positions where $\mathbf{R}_m \approx \mathbf{J}$ (90° rotation), equivalent to maximizing the sine score:
$$U(m) = \sum_{k=0}^{d/2-1} \sin(m\theta_k)$$

Positions with the highest $V(m)$ and $U(m)$ values constitute the anchors of the implicit visual coordinate system.

#### 2. IVC Token Selection

For each token position $m$, $V(m)$ and $U(m)$ are computed, and the top-$k_c$ positions from each are merged:
$$\mathcal{I}_{\text{ivc}} = \arg\text{TopK}(\{V(m)\}, k_c) \cup \arg\text{TopK}(\{U(m)\}, k_c)$$

The default $k_c = 10\%$ covers approximately 10% of total tokens. This step is determined entirely by mathematical formulae and requires no model inference.

#### 3. Foreground Token Selection (Two-Stage)

Conventional methods select foreground tokens directly via attention scores, but attention scores are influenced by positional encoding, causing text tokens to preferentially attend to spatially proximate rather than semantically relevant visual tokens. To eliminate this positional bias, IVC-Prune uses **Value vectors** (which are unaffected by positional encoding) to compute similarity.

**Stage 1: Semantic Seed Identification.** The similarity between text Value vectors $\mathbf{V}_{\text{text}}$ and image Value vectors $\mathbf{V}_{\text{img}}$ is computed, and normalized attention over text tokens is averaged for each visual token:
$$\mathbf{s} = \text{Mean}(\text{Softmax}(\frac{\mathbf{V}_{\text{text}} \cdot \mathbf{V}_{\text{img}}^T}{\sqrt{D}}))$$
The top 1% of visual tokens are selected as the semantic seed set $\mathcal{I}_{\text{seed}}$.

**Stage 2: Contextual Foreground Refinement.** The semantic seed tokens are merged with all text tokens to form an extended query set $\mathbf{V}_{\text{query}}$, and similarity is recomputed to obtain refined scores $\mathbf{f}$. The top-$k_f$ tokens are selected as the final foreground set $\mathcal{I}_{\text{fg}}$. This ensures complete coverage of large or complex objects.

#### 4. Single-Shot Selection Pruning Strategy

Prior work suggests that LVLMs are sensitive to token pruning in early layers. The authors demonstrate experimentally that this sensitivity actually arises from the removal of IVC tokens rather than the pruning operation itself. They therefore propose: at a selected intermediate layer, the retained token set is determined **once**, with original position IDs preserved; this selection is then applied to prune KV caches of all earlier layers and is used in all subsequent layers.

This offers three benefits: (1) minimal overhead from a single-pass operation; (2) avoidance of poor early-layer selections corrupting downstream computation; (3) maximized KV cache compression for improved decoding efficiency.

### Loss & Training

IVC-Prune is entirely **training-free** and requires no additional training or fine-tuning. IVC token selection is determined directly by the mathematical properties of RoPE, and foreground token selection is performed via Value vector similarity computation. The selection layer $i$ is fixed empirically based on performance on a small validation set (a subset of RefCOCO testA).

## Key Experimental Results

### Main Results

Evaluation is conducted on 4 LVLMs (Qwen2.5-VL-7B, InternVL-2.5-8B, DeepSeek-VL2-16B, LLaVA-v1.5-7B) across 20 benchmarks.

**Visual Grounding (RefCOCO/RefCOCO+/RefCOCOg)**:

| Model | Method | Tokens Retained | Relative Avg. Performance |
|-------|--------|-----------------|--------------------------|
| Qwen2.5-VL-7B | Vanilla | 100% | 100% |
| Qwen2.5-VL-7B | FastV | 54% | 84.7% |
| Qwen2.5-VL-7B | IVC-Prune | 50% | **99.6%** |
| InternVL-2.5-8B | Vanilla | 100% | 100% |
| InternVL-2.5-8B | FastV | 53% | 90.1% |
| InternVL-2.5-8B | IVC-Prune | 50% | **99.5%** |
| DeepSeek-VL2-16B | FastV | 54% | 96.7% |
| DeepSeek-VL2-16B | IVC-Prune | 52% | **99.0%** |

**General VQA (average over 9 benchmarks)**:

| Model | Method | Tokens Retained | Relative Avg. Performance |
|-------|--------|-----------------|--------------------------|
| Qwen2.5-VL-7B | Vanilla | 100% | 100% |
| Qwen2.5-VL-7B | FastV | 54% | 98.4% |
| Qwen2.5-VL-7B | IVC-Prune | 50% | **100.6%** |
| InternVL-2.5-8B | IVC-Prune | 50% | **99.6%** |
| DeepSeek-VL2-16B | IVC-Prune | 52% | **100.1%** |
| LLaVA-v1.5-7B | IVC-Prune | 28% | **101.3%** |

### Ablation Study

**Effect of IVC Tokens (Qwen2.5-VL-7B)**:

| Method | Config | RefCOCO testA | RefCOCO+ testA | SEED | MMB |
|--------|--------|--------------|----------------|------|-----|
| Vanilla | Default | 92.2 | 88.0 | 76.7 | 82.4 |
| Vanilla | Remove IVC | 84.1 | 79.4 | 76.1 | 82.2 |
| FastV | Default | 74.4 | 68.9 | 72.9 | 80.5 |
| FastV | +IVC | 82.1 | 76.5 | 74.6 | 80.5 |
| PDrop | Default | 77.6 | 72.1 | 74.0 | 78.9 |
| PDrop | +IVC | 83.9 | 76.5 | 74.6 | 79.2 |

**Inference Efficiency (Qwen2.5-VL-7B, RefCOCO testA)**:

| Method | KV Cache | Prefill Time | Decode Latency | Total Time | Accuracy |
|--------|----------|-------------|----------------|------------|----------|
| Vanilla | 26.0MB (1.0×) | 408ms | 65.3ms/tok | 60'17 | 92.2 |
| FastV | 16.1MB (1.6×) | 297ms | 62.7ms/tok | 51'51 | 74.4 |
| IVC-Prune | 15.9MB (1.6×) | 322ms | 60.2ms/tok | **47'47** | **92.0** |

### Key Findings

1. **IVC tokens are critical for spatial reasoning**: Even in the unmodified Vanilla model, removing only IVC tokens (~10%) causes an 8.1-point drop on RefCOCO testA, while SEED and MMB are largely unaffected.
2. **IVC tokens are universally applicable**: Incorporating IVC tokens into FastV and PDrop yields significant improvements in grounding performance (FastV: +7.7, PDrop: +6.3).
3. **True cause of early-layer sensitivity**: The early-layer pruning sensitivity reported in prior work stems from the inadvertent removal of IVC tokens; preserving IVC tokens renders pruning at any layer harmless.
4. **Consistency across model scales**: Consistent results are observed on Qwen2.5-VL at 3B, 7B, and 32B parameter scales.

## Highlights & Insights

- **Deep theoretical contribution**: This work is the first to mathematically reveal how LVLMs implicitly establish a visual coordinate system through RoPE, explaining why existing pruning methods fail on spatial tasks and offering a new perspective on spatial reasoning in LVLMs.
- **Strong practicality**: Entirely training-free; IVC token selection is purely mathematical, and foreground token selection requires only a single Value vector similarity computation, making the method plug-and-play for existing LVLMs.
- **Insight-driven design**: Upon identifying the root cause of early-layer sensitivity, the authors propose a single-shot selection strategy that simplifies pruning from a multi-layer operation to a single-layer one while maximizing KV cache savings.
- **Value vector debiasing**: The method cleverly exploits the positional-encoding-invariant nature of Value vectors to eliminate the positional bias inherent in attention scores.

## Limitations & Future Work

1. The proportions of IVC tokens ($k_c=10\%$) and foreground tokens ($k_f=40\%$) are fixed hyperparameters that may not generalize optimally to all scenarios; dynamic ratio adjustment warrants exploration.
2. Determining the selection layer requires a validation set search, introducing a non-trivial setup cost.
3. Evaluation is primarily conducted on 2D static images; effectiveness on dynamic scenarios such as video understanding remains to be thoroughly assessed.
4. The theoretical analysis is based on standard 1D and 2D RoPE; generalizability to other positional encoding schemes requires further investigation.

## Related Work & Insights

Analogous to the ideas in StreamingLLM and SepLLM—which retain "attention sink" tokens and separator tokens in LLMs—IVC-Prune identifies and preserves "coordinate anchor" tokens in LVLMs. This suggests that distinct types of "special tokens" play critical roles in different model capabilities. The Value vector debiasing technique may also inspire other scenarios requiring unbiased feature selection.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to reveal the RoPE implicit coordinate system; outstanding theoretical contribution.
- **Technical Quality**: ⭐⭐⭐⭐⭐ — Rigorous mathematical analysis; elegant two-stage foreground selection design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 models × 20 benchmarks; comprehensive ablations.
- **Practicality**: ⭐⭐⭐⭐⭐ — Training-free plug-and-play with clear efficiency gains.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logical flow from theoretical discovery to method design.
- **Overall**: ⭐⭐⭐⭐⭐ (9.5/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](../../CVPR2026/multimodal_vlm/hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICLR 2026\] HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late Injection, Concave Pyramid Pruning, and Early Exit](hidrop_hierarchical_vision_token_reduction_in_mllms_via_late_injection_concave_p.md)
- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](../../CVPR2026/multimodal_vlm/when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)

</div>

<!-- RELATED:END -->
