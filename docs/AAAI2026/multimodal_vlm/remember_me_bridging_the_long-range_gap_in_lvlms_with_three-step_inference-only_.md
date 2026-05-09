---
title: >-
  [Paper Note] Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies
description: >-
  [AAAI 2026][Multimodal VLM][Large Vision-Language Models] This paper proposes T-DRS (Three-step Decay Resilience Strategies), a training-free inference-time framework that mitigates RoPE-induced long-range attention decay through three cooperative stages: semantics-driven enhancement, distance-aware control, and remote-distance re-reinforcement, achieving consistent performance gains across multiple LVLMs on VQA benchmarks.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Large Vision-Language Models
  - Positional Encoding
  - RoPE
  - Long-Range Attention Decay
  - Inference-Time Optimization
date: 2026-05-08
content_hash: 46e061adbf765e1c
---

# Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies

**Conference**: AAAI 2026
**arXiv**: [2511.09868](https://arxiv.org/abs/2511.09868)
**Code**: [https://github.com/labixiaoq-qq/Remember-me](https://github.com/labixiaoq-qq/Remember-me)
**Area**: Multimodal VLM
**Keywords**: Large Vision-Language Models, Positional Encoding, RoPE, Long-Range Attention Decay, Inference-Time Optimization

## TL;DR

This paper proposes T-DRS (Three-step Decay Resilience Strategies), a training-free inference-time framework that mitigates RoPE-induced long-range attention decay through three cooperative stages: semantics-driven enhancement, distance-aware control, and remote-distance re-reinforcement, achieving consistent performance gains across multiple LVLMs on VQA benchmarks.

## Background & Motivation

### Long-Range Decay in RoPE

Large Vision-Language Models (LVLMs) commonly adopt Rotary Position Embedding (RoPE) to encode relative positional relationships between tokens. RoPE introduces rotation matrices into the query-key dot product to achieve distance-aware attention computation:

$$A_{i,j} = \text{softmax}\left(\frac{Q_i^\top R_{j-i} K_j}{\sqrt{d}}\right)$$

**Problem**: As the relative distance $|j-i|$ between tokens increases, the rotational orthogonality induced by large-angle embeddings causes the attention score $A_{i,j}$ to gradually decay. While this local inductive bias is reasonable in pure language modeling (distant tokens are typically less relevant), it becomes problematic in multimodal tasks:

- Question tokens may need to attend to distant visual regions (image and text tokens are separated by a large number of tokens in the sequence)
- Critical cross-modal semantic alignment may be unjustly suppressed due to positional distance
- This results in a **mismatch between semantic importance and attention strength**

### Limitations of Prior Work

Existing solutions (e.g., position interpolation, memory extension techniques) mostly require retraining or fine-tuning, making them infeasible in resource-constrained settings.

### Design Intuition of T-DRS

The three strategies address the problem progressively:
1. SD-DRS: First restores suppressed long-range semantic associations (which may perturb local structure)
2. DC-DRS: Smooths local structure to mitigate perturbations introduced by the first step
3. reRD-DRS: Finally re-enhances pairs that remain suppressed after the first two steps despite high semantic relevance

## Method

### Overall Architecture

Given image and text inputs, the visual encoder and language model produce a concatenated token sequence $S = \{S_{\text{vision}}, S_{\text{instr}}\} \in \mathbb{R}^{(V+T) \times d}$. RoPE attention yields decayed attention logits $A$, to which the three DRS strategies are applied sequentially, producing the final attention:

$$A_{i,j}^{T-DRS} = A_{i,j} + A_{i,j}^{sd} + A_{i,j}^{dc} + A_{i,j}^{re}$$

The method operates entirely at inference time without modifying model parameters.

### Key Designs

#### 1. **Semantics-Driven DRS (SD-DRS)**

**Mechanism**: Injects semantic similarity biases into pre-softmax attention logits, enabling token pairs that are semantically aligned but positionally distant to recover attention.

- Computes cosine similarity (semantic affinity) between queries and keys:
$$\text{sem\_sim}_{i,j} = \frac{Q_i \cdot K_j}{\|Q_i\| \cdot \|K_j\|}$$

- Normalizes to $[0, 1]$:
$$\text{sem\_pos}_{i,j} = \frac{1}{2}(\text{sem\_sim}_{i,j} + 1)$$

- Adds as a bias to the original logits:
$$A_{i,j}^{sd} = A_{i,j} + \text{sem\_pos}_{i,j}$$

- Further computes a normalized scaling factor $\text{scale}_{i,j}$ for use by the subsequent two DRS stages

**Design Motivation**: RoPE assumes the dot product suffices to capture relevance, which does not hold for long-range semantic pairs. SD-DRS introduces a content-aware bias to complement the position-centric Transformer design.

#### 2. **Distance-aware Control DRS (DC-DRS)**

**Mechanism**: While SD-DRS restores long-range pairs, it may slightly disturb local structure. DC-DRS introduces a smooth, analytically-guaranteed distance decay function to protect local inductive bias.

**Design Criteria** (inspired by Bishop 2006):
- Monotonicity: $w(d)$ strictly decreases with distance
- Smoothness: continuously differentiable
- Lower-bound guarantee: non-zero minimum attention $w_{\min}^{dc}$ at maximum distance

**Decay Function** (Gaussian form):
$$w(d_{i,j}) = \exp\left(-\frac{1}{2}\left(\frac{d_{i,j}}{\sigma_0}\right)^2\right), \quad \sigma_0 = \frac{\max(d_{i,j})}{\sqrt{-2\ln w_{\min}^{dc}}}$$

**Semantics-Adaptive Modulation**: Uses the SD-DRS scaling factor to adjust effective distance:
$$\hat{d}_{i,j} = \frac{d_{i,j}}{\text{scale}_{i,j}}$$

When semantic alignment is high, $\text{scale}$ is large, the effective distance is shortened, and decay is weakened — allowing semantically important long-range pairs to maintain stronger attention.

Final form: $A_{i,j}^{dc} = \lambda_{dc} \cdot A_{i,j} \cdot r_{i,j}^{dc}$

#### 3. **Remote-Distance Re-reinforcement DRS (reRD-DRS)**

**Mechanism**: After the first two steps, some token pairs with high semantic relevance but extreme positional distance remain affected by cumulative decay. reRD-DRS applies a heavy-tailed kernel for selective reinforcement.

**Rational Quadratic Kernel** (inspired by Rasmussen 2006):
$$r_{i,j}^{re} = \left(1 + \frac{d_{i,j}^2}{2 \cdot (\sigma_{re} \cdot \text{scale}_{i,j})^2}\right)^{-\alpha}$$

**Design Properties**:
- Decays more slowly than the Gaussian kernel (heavy-tailed), enabling stronger reinforcement of long-range dependencies
- $\alpha$ is analytically determined by $w_{\min}^{re}$, requiring no manual tuning of decay sharpness
- Lower-bound constraint: $r_{i,j}^{re}|_{d=d_{\max}, \text{scale}=1} = w_{\min}^{re}$

Final form: $A_{i,j}^{re} = \lambda_{re} \cdot A_{i,j} \cdot r_{i,j}^{re}$

**Design Motivation**: The Gaussian decay of DC-DRS may over-suppress extreme-range pairs. The heavy-tailed property of the rational quadratic kernel ensures that token pairs at extreme distances are not completely ignored as long as semantic relevance is sufficiently high.

### Hyperparameter Design

Four hyperparameters $\{w_{\min}^{dc}, \lambda_{dc}, w_{\min}^{re}, \lambda_{re}\}$:
- DC-DRS parameters are determined prior to reRD-DRS
- $w_{\min}$ is typically set as a multiple of the attention map minimum $|A|_{\min}$
- Optimal settings: $w_{\min}^{dc} = 3|A|_{\min}$, $\lambda_{dc} = 1$, $w_{\min}^{re} = 2|A|_{\min}$, $\lambda_{re} = 0.8$ (ScienceQA) or $1$ (POPE)

## Key Experimental Results

### Main Results

T-DRS is applied to three LVLMs with different architectures: LLaVA1.5-7B, InternVL2-8B, and Qwen2.5-VL-7B.

| Method | ScienceQA | GQA | TextVQA | POPE Acc | POPE F1 |
|--------|-----------|-----|---------|----------|---------|
| LLaVA1.5-7B | 67.9 | 62.0 | 58.2 | 83.3 | 85.7 |
| InternVL2-8B | 96.6 | 62.6 | 79.1 | 88.0 | 87.0 |
| Qwen2.5-VL-7B | 79.4 | 57.9 | 84.5 | 87.7 | 86.4 |
| **LLaVA1.5 + T-DRS** | **69.2** (+1.3) | **63.1** (+1.1) | **59.0** (+0.8) | **83.7** | **86.1** |
| **InternVL2 + T-DRS** | **97.3** (+0.7) | **62.8** (+0.2) | **79.7** (+0.6) | **88.0** | **87.4** |
| **Qwen2.5 + T-DRS** | **80.7** (+1.3) | **58.3** (+0.4) | **85.0** (+0.5) | **88.5** | **87.3** |

### Ablation Study

| Configuration | ScienceQA (LLaVA) | ScienceQA (InternVL) | ScienceQA (Qwen) |
|---------------|-------------------|----------------------|------------------|
| Baseline | 67.9 | 96.6 | 79.4 |
| + SD-DRS | 68.1 (+0.2) | 96.9 (+0.3) | 79.8 (+0.4) |
| + SD-DRS + DC-DRS | 68.8 (+0.9) | 97.1 (+0.5) | 80.4 (+1.0) |
| + SD-DRS + DC-DRS + reRD-DRS (Full) | **69.2** (+1.3) | **97.3** (+0.7) | **80.7** (+1.3) |

POPE ablation (F1-score):

| Configuration | LLaVA | InternVL | Qwen |
|---------------|-------|----------|------|
| Baseline | 85.7 | 87.0 | 86.4 |
| + SD-DRS | 85.8 | 86.8 | 86.6 |
| + SD-DRS + DC-DRS | 86.0 | 87.2 | 86.9 |
| Full Model | **86.1** | **87.4** | **87.3** |

### Key Findings

1. **Consistent improvement**: T-DRS improves all three LVLMs across different architectures, confirming that RoPE long-range decay is a universal issue across models
2. **Complementary three-stage design**: SD-DRS introduces semantic awareness → DC-DRS protects local structure → reRD-DRS reinforces the heavy tail; each stage contributes positively
3. **LLaVA benefits most** (+1.3% ScienceQA), InternVL least (+0.7%), possibly because InternVL already incorporates better long-range modeling mechanisms
4. **Balance of $w_{\min}$**: Too large causes noisy tokens to receive undue attention; too small suppresses meaningful neighboring tokens
5. **Visualization evidence**: RoPE-only attention concentrates on image borders (local bias); after T-DRS, attention correctly focuses on central semantic regions (e.g., the face of a red kangaroo rather than background fur)

## Highlights & Insights

1. **Purely inference-time method**: No training, no additional parameters (only a few fixed hyperparameters) — an elegant plug-and-play solution
2. **Principled mathematical design**: Gaussian decay guarantees monotonicity, smoothness, and a lower bound; the rational quadratic kernel guarantees heavy-tailed behavior and a lower bound; hyperparameters can be determined analytically
3. **Progressive three-stage design**: Loosen → constrain → reinforce, with each step having a clear functional role and demonstrated necessity
4. **Cross-architecture generality**: Effective across three architecturally diverse LVLMs: LLaVA, InternVL, and Qwen
5. **Compelling visualization**: The attention shift from incorrect (painted stork) to correct (red kangaroo) answers clearly illustrates the mechanism of the proposed method

## Limitations & Future Work

1. **Limited absolute improvement**: Maximum gain is 1.3% (ScienceQA); gains on already strong baselines (e.g., InternVL at 96.6%) are smaller (0.7%)
2. **Hyperparameters require dataset-specific tuning**: $\lambda_{re}$ takes values of 0.8 and 1.0 on ScienceQA and POPE respectively
3. **Absence of long-context scenarios**: The sequence lengths of the four benchmarks may not be sufficiently extreme; validation on genuinely long-context settings such as Video-QA or multi-image reasoning is lacking
4. **Computational overhead not analyzed in detail**: Although inference-time only, the additional cosine similarity computation and distance matrix construction may introduce significant latency
5. **No comparison with RoPE alternatives**: Comparisons with different positional encoding schemes such as ALiBi and NoPE are absent
6. **Semantic similarity computed from Q/K themselves**: Rather than independent semantic representations, potentially introducing a circular dependency

## Related Work & Insights

- **RoPE** (Su 2024): The foundation of this work; reveals the inherent long-range decay flaw of RoPE
- **Position Interpolation** (Chen 2023): Extends sequence length by scaling rotational frequencies, but requires fine-tuning
- **VideoRoPE** (Wei 2025): Observes similar long-range decay issues with RoPE in video settings
- **MCA** (Zhao 2025) and **HOPE** (Li 2025): Other methods for mitigating attention decay, but require training

## Rating

- Novelty: ⭐⭐⭐⭐ (The three-stage inference-time framework design is novel, though the core idea is a relatively straightforward attention bias modification)
- Experimental Thoroughness: ⭐⭐⭐ (Three models and four datasets, but lacking long-context evaluation and efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐ (Detailed mathematical derivations, clear illustrations)
- Value: ⭐⭐⭐⭐ (High practical utility as a training-free plug-and-play solution, though gains require validation in more diverse scenarios)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Bridging the Copyright Gap: Do Large Vision-Language Models Recognize and Respect Copyrighted Content?](bridging_the_copyright_gap_do_large_vision-language_models_r.md)
- [\[CVPR 2026\] SciPostGen: Bridging the Gap between Scientific Papers and Poster Layouts](../../CVPR2026/multimodal_vlm/scipostgen_bridging_the_gap_between_scientific_papers_and_poster_layouts.md)
- [\[CVPR 2026\] Narrative Weaver: Towards Controllable Long-Range Visual Consistency with Multi-Modal Conditioning](../../CVPR2026/multimodal_vlm/narrative_weaver_towards_controllable_long-range_visual_consistency_with_multi-m.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](../../CVPR2026/multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](../../CVPR2026/multimodal_vlm/sope_spherical_positional_encoding_3d_lvlm.md)

<!-- RELATED:END -->
