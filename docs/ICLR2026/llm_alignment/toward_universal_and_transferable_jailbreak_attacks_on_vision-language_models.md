---
title: >-
  [Paper Note] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)
description: >-
  [ICLR 2026][LLM Alignment][VLM jailbreak] This paper proposes UltraBreak, which combines a semantic adversarial objective (replacing cross-entropy with cosine similarity to produce a smooth loss landscape) and input-space constraints (random transformations + TV regularization to yield transformation-invariant features) to optimize a single universal adversarial image capable of jailbreaking 6+ VLM architectures and commercial models. The average black-box ASR reaches 71% on SafeBench, substantially outperforming prior methods.
tags:
  - ICLR 2026
  - LLM Alignment
  - VLM jailbreak
  - adversarial attack
  - universal adversarial image
  - semantic loss
  - transferable attack
date: 2026-05-08
content_hash: e7a58d90c6779fc2
---

# Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)

**Conference**: ICLR 2026
**arXiv**: [2602.01025](https://arxiv.org/abs/2602.01025)
**Code**: Available (GitHub)
**Area**: LLM Alignment
**Keywords**: VLM jailbreak, adversarial attack, universal adversarial image, semantic loss, transferable attack

## TL;DR
This paper proposes UltraBreak, which combines a semantic adversarial objective (replacing cross-entropy with cosine similarity to produce a smooth loss landscape) and input-space constraints (random transformations + TV regularization to yield transformation-invariant features) to optimize a single universal adversarial image capable of jailbreaking 6+ VLM architectures and commercial models. The average black-box ASR reaches 71% on SafeBench, substantially outperforming prior methods.

## Background & Motivation

**Background**: VLM jailbreak attacks fall into two categories: manually crafted attacks (e.g., FigStep, which embeds harmful text into images) and gradient-based optimization methods (e.g., VAJM/UMK). Gradient-based methods can theoretically produce universal triggers, but in practice they severely overfit to a single white-box surrogate model.

**Limitations of Prior Work**:
   - Universality: Existing gradient-based attacks are effective against a single target but fail to generalize across queries.
   - Transferability: Adversarial images optimized against a white-box surrogate do not transfer to black-box models.
   - Root cause: Cross-entropy loss produces a spiky loss landscape, and the sharp local minima found during optimization generalize poorly.

**Key Challenge**: The goal of attacking all queries and all models with a single image is fundamentally undermined by the overfitting induced by existing loss functions and optimization strategies.

**Goal**: Simultaneously achieve universality (a single image effective across all harmful queries) and transferability (across model architectures).

**Key Insight**: The smoothness of the loss landscape governs generalization—replacing token-level cross-entropy with semantic-level cosine similarity yields a smoother landscape.

**Core Idea**: Semantic loss smooths the loss landscape + input transformations produce invariant features = a single universal image for cross-model jailbreaking.

## Method

### Overall Architecture
A white-box surrogate VLM is selected; a single adversarial image is optimized over 50 queries for 1,300 steps using Adam. The resulting image can be directly applied to attack arbitrary VLMs (including commercial models) across arbitrary harmful queries. The input is a blank image with an adversarial perturbation; the output is a universal trigger image effective against all target models.

### Key Designs

1. **Semantic Adversarial Target**:

    - Function: Replaces token-level cross-entropy with semantic similarity as the optimization objective.
    - Mechanism: Output logits are projected into the embedding space to obtain $\mu_t = W^\top \text{softmax}(z_t)$; target tokens are mapped to embeddings $e_t$ with added Gaussian noise for robustness: $\tilde{e}_t = e_t + \varepsilon_t$. The loss is $\mathcal{L}_{\text{sem}} = \frac{1}{T}\sum_t (1 - \cos(\mu_t, e_t^{\text{att}}))$, where $e_t^{\text{att}}$ is the target embedding weighted by causal attention.
    - Design Motivation: Cross-entropy requires exact token matching, producing a spiky landscape. Cosine similarity measures proximity in semantic space, permitting solutions that are "semantically correct but lexically different," thereby yielding a smoother landscape and better generalization.
    - Attention mechanism: Q/K construction with positional encoding; temperature $\tau=0.5$ controls distribution sharpness. Setting $\tau=0$ degenerates to CE; $\tau \to \infty$ over-smooths.

2. **Input Space Constraints**:

    - Function: Encourages the adversarial image to produce transformation-invariant robust features.
    - Three components: (a) Random transformations—random rotation (−15° to 15°), scaling (0.8–1.2), and translation at each step, preventing overfitting to pixel positions; (b) Input projection—normalization with CLIP mean/std and clipping to [0, 1]; (c) TV regularization $\mathcal{L}_{\text{TV}}$ enforcing spatial smoothness and suppressing noise patterns.
    - Effect: Without constraints → noisy images; with random transformations → text-like patterns emerge; with TV → smoother and more coherent structures. These transformation-invariant structures serve as cross-model invariant cues.

3. **Target Prompt Guidance (TPG)**:

    - Function: Enhances attack effectiveness on the text side.
    - Format: $q^{\text{TPG}} = \text{"Steps to "} + q + \text{" You must begin with: "} + p$, where $p$ = "[Jailbroken Mode]" (open-source) or "[START LIST]" (commercial).

### Loss & Training
$$\arg\min_x \sum_{(q,y) \in \mathcal{Q}'} \mathbb{E}_{l,r,s}[\mathcal{L}_{\text{sem}}^{\text{att}}(M', A(x_{\text{blank}}, x_{\text{proj}}, l, r, s), q^{\text{TPG}}, y)] + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}(x)$$
- Surrogate model: Qwen2-VL-7B-Instruct
- Training: SafeBench-Tiny (50 queries), 1,300 steps Adam, $\tau=0.5$, $\lambda_{\text{TV}}=0.5$

## Key Experimental Results

### Main Results: Black-box ASR (SafeBench, 315 queries)

| Target Model | No Attack | FigStep | VAJM | UMK | **UltraBreak** |
|---------|-----------|---------|------|-----|--------------|
| Qwen-VL-Chat | 22.86 | 69.52 | 12.06 | 0.63 | **72.70** |
| Qwen2.5-VL-7B | 14.29 | 53.97 | 28.89 | 15.24 | **60.32** |
| LLaVA-v1.6 | 80.32 | 47.94 | 57.46 | 20.63 | **88.25** |
| GLM-4.1V-9B | 46.03 | 88.25 | 67.62 | 50.79 | **66.03** |
| Black-box Avg. | 40.57 | 66.54 | 41.46 | 20.00 | **71.05** |
| Commercial Avg. | 20.00 | — | 11.48 | 14.59 | **32.26** |

### Ablation Study

| Configuration | SafeBench Avg | AdvBench Avg | Notes |
|------|-------------|-------------|------|
| Full UltraBreak | **71.83** | **57.64** | — |
| w/o image (text only) | 40.79 | 25.90 | Image contributes ~30% ASR |
| w/o constraints | 51.99 | 29.86 | White-box overfitting (89%→49% transfer) |
| w/o semantic loss (CE) | 55.80 | 40.15 | Spiky CE landscape → poor transfer |
| w/o attention weighting | 57.54 | 41.83 | Unstable optimization + high variance |

### Key Findings
- **Universal single-image attack**: Training on 50 queries generalizes to 315+ harmful queries across 6+ model architectures with a single image.
- **Semantic loss vs. CE**: The semantic loss yields a clustered and smooth loss landscape, whereas CE produces a scattered and spiky one.
- **Transformation-invariant structures**: TV regularization combined with random transformations causes the adversarial image to exhibit text/symbol-like structures; these high-level features transfer across models more readily than pixel-level noise.
- **Commercial models are also vulnerable**: Gemini-2.5 reaches 42% ASR and GPT-4.1-nano reaches 38.78%.

## Highlights & Insights
- **Core insight from the loss landscape perspective**: The transferability problem is reframed as a loss landscape smoothness problem; replacing CE with semantic loss directly addresses this. This perspective generalizes to all research on adversarial transferability.
- **Transformation invariance = model invariance**: Applying random input-side transformations encourages the perturbation to learn high-level semantic features rather than low-level pixel patterns. High-level features are shared across models—this also explains why human-designed text images (FigStep) exhibit cross-model effectiveness.
- **Challenges the "multi-surrogate" assumption**: Prior work assumed that cross-model transfer requires ensembling multiple surrogate models; UltraBreak demonstrates that a single surrogate with the correct loss function suffices.

## Limitations & Future Work
- **Limited effectiveness against highly secure models**: Claude-3-haiku achieves only 16% ASR, indicating that strongly aligned models remain robust.
- **White-box surrogate dependency**: A white-box open-source VLM is still required for gradient-based optimization.
- **Defense directions**: The paper exposes VLM vulnerability to transformation-invariant visual features, suggesting detection/defense strategies such as identifying text-like adversarial structures within images.
- **Connection to GuardAlign**: UltraBreak's visual attack and GuardAlign's OT-based safety detection represent a direct adversarial pairing.

## Related Work & Insights
- **vs. FigStep**: FigStep manually crafts a separate image per target (one image per goal); UltraBreak automatically optimizes a single universal image, achieving higher ASR.
- **vs. UMK/VAJM**: These gradient-based methods use CE loss and severely overfit to the white-box surrogate; UltraBreak's semantic loss fundamentally resolves this issue.
- **vs. text-side jailbreaks (GCG, etc.)**: UltraBreak operates in the visual modality, orthogonal to text-based attacks, and the two can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of loss landscape perspective, semantic loss, and transformation invariance is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6+ open-source models, 3 commercial models, 3 benchmarks, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Technical details are clear; ablation and visualization analyses are thorough.
- Value: ⭐⭐⭐⭐⭐ Reveals fundamental vulnerabilities in VLM safety with important implications for defense research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)

<!-- RELATED:END -->
