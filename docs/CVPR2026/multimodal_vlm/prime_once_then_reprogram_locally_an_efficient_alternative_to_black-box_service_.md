---
title: >-
  [Paper Note] Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation
description: >-
  [CVPR 2026][Multimodal VLM][Model-as-a-Service] This paper proposes AReS, which replaces the continuous API calls of conventional zeroth-order optimization (ZOO) with a single-round API query to prime a local encoder. AR…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Model-as-a-Service"
  - "black-box adaptation"
  - "visual reprogramming"
  - "zeroth-order optimization"
  - "API-efficient utilization"
date: 2026-05-08
content_hash: bc9cab5a2c68e0fa
---

# Prime Once, then Reprogram Locally: An Efficient Alternative to Black-Box Service Model Adaptation

**Conference**: CVPR 2026
**arXiv**: [2604.01474](https://arxiv.org/abs/2604.01474)
**Code**: [https://github.com/yunbeizhang/AReS](https://github.com/yunbeizhang/AReS)
**Area**: Multimodal VLM
**Keywords**: Model-as-a-Service, black-box adaptation, visual reprogramming, zeroth-order optimization, API-efficient utilization

## TL;DR

This paper proposes AReS, which replaces the continuous API calls of conventional zeroth-order optimization (ZOO) with a single-round API query to prime a local encoder. AReS achieves a +27.8% improvement on GPT-4o (where ZOO methods are nearly ineffective), while reducing API calls by over 99.99% and enabling zero-cost inference.

## Background & Motivation

1. **Background**: Model-as-a-Service (MaaS) is the dominant paradigm for deploying SOTA models, where users can only obtain input–output predictions via APIs. Closed-box visual reprogramming methods (e.g., BAR, BlackVIP) adapt API models by modifying input images through zeroth-order optimization.
2. **Limitations of Prior Work**: ZOO methods face a triple dilemma: (1) they require massive API calls (~$10^8$), incurring extremely high training and inference costs; (2) gradient estimation is unstable, making optimization slow and unreliable; (3) modern powerful APIs (e.g., GPT-4o) are robust to input perturbations, so the small perturbations that ZOO relies on are ignored by the model, yielding nearly no performance gain.
3. **Key Challenge**: The fundamental assumption of ZOO methods—that perturbing the input can influence model outputs—progressively fails as models become more powerful and robust to perturbations.
4. **Goal**: How to efficiently adapt service models under the strictest black-box setting (input–predicted probability access only), especially when ZOO methods are ineffective against modern APIs.
5. **Key Insight**: Rather than continuously performing costly zeroth-order optimization on the black-box model, perform a one-time API interaction to acquire knowledge and then conduct efficient white-box optimization on a local model.
6. **Core Idea**: Prime the local encoder with a single API query, then complete all visual reprogramming and inference locally, entirely eliminating subsequent API dependency.

## Method

### Overall Architecture

AReS operates in two stages: (1) **Prime Once**—the API is queried exactly once per training image to obtain predicted probabilities, which are used to train a lightweight linear layer on top of the local encoder, enabling the local model to "mimic" the service model's behavior; (2) **Reprogram Locally**—visual prompts are optimized on the primed local model using standard gradient descent in a fully white-box manner, requiring no further API access. At inference time, only the local model is used, achieving zero API cost.

### Key Designs

1. **Prime Once**:

    - **Function**: Transfer knowledge from the service model to the local encoder via a single API interaction.
    - **Mechanism**: The local encoder is frozen; only a single linear layer $\theta \in \mathbb{R}^{K^S \times (d_{enc}+1)}$ appended to its top is trained. For each training image $x_i$, the API is queried once to obtain predicted probabilities $p_S(x_i)$; the KL divergence between the local model and the service model outputs is then minimized: $\mathcal{L}_P(p_L, p_S) = -\sum_j p_{S,j} \log p_{L,j}$. Crucially, priming does not require label space alignment—even if the API outputs in the ImageNet label space while the target task is Flowers, priming remains effective, as its purpose is to "prepare" rather than to "distill."
    - **Design Motivation**: Unlike conventional knowledge distillation, the goal of priming is not to produce a high-performance model directly, but to make the local model more "sensitive" and "programmable" for subsequent reprogramming.

2. **Local White-Box Visual Reprogramming**:

    - **Function**: Efficiently learn visual prompts on the primed local model.
    - **Mechanism**: A learnable visual prompt $\mathbf{P}$ and an input transformation function $g_{in}$ are defined; optimization is performed using standard cross-entropy loss with exact gradients: $\mathbf{P}^* = \arg\min_{\mathbf{P}} \mathbb{E}_{(x,y)} [\ell(g_{out}(\mathcal{F}_L(g_{in}(x, \mathbf{P}); \theta^*)), y)]$. As a white-box operation, first-order optimizers (e.g., Adam) can be used, yielding more stable and faster convergence than the approximate gradients of ZOO.
    - **Design Motivation**: Transform a difficult black-box optimization problem into a straightforward white-box optimization problem, leveraging exact gradients for better and faster convergence.

3. **AReS-MS Model Selection Strategy**:

    - **Function**: Automatically determine when to use the local model and when to fall back to zero-shot API inference.
    - **Mechanism**: The priming stage is exploited as a low-cost diagnostic tool—if the local model's primed performance falls within a tolerance $\tau$ of the API zero-shot baseline, the efficient local path is taken; otherwise, the method falls back to zero-shot API. This makes AReS not merely an adaptation method, but an intelligent decision framework for cost–performance trade-offs.
    - **Design Motivation**: On datasets such as Food101 and Cars, all reprogramming methods (including white-box ones) underperform CLIP zero-shot, indicating an inherent limitation of input-level reprogramming for these domains. AReS-MS automatically identifies such scenarios and makes the optimal choice.

### Loss & Training

Priming stage: KL divergence loss, Adam optimizer with lr=0.001. Reprogramming stage: cross-entropy loss, Adam optimizer with lr=0.01, padding-based visual prompts. For VM (standard vision models), Bayesian Label Mapping (BLM) is additionally applied to bridge source/target label space discrepancies.

## Key Experimental Results

### Main Results (CLIP ViT-B/16 as service model, 16-shot)

| Method | Flowers | DTD | UCF | Food | GTSRB | EuroSAT | Pets | Cars | SUN | SVHN | Avg | API Calls (M) | Time (h) |
|--------|---------|-----|-----|------|-------|---------|------|------|-----|------|-----|--------------|---------|
| Zero-shot | 71.3 | 43.9 | 66.9 | 85.9 | 21.0 | 47.9 | 89.1 | 65.2 | 62.6 | 17.9 | 57.2 | 0.12 | 0 |
| BAR | 71.0 | 46.8 | 64.2 | 84.4 | 21.5 | 77.3 | 88.4 | 63.0 | 62.4 | 34.6 | 61.4 | 612.8 | 185.6 |
| BlackVIP | 70.6 | 45.3 | 68.7 | 85.9 | 21.3 | 73.3 | 89.1 | 65.4 | 64.5 | 44.4 | 62.9 | 754.2 | 197.5 |
| **AReS** | **86.6** | 48.2 | 67.1 | 68.8 | **39.4** | **85.7** | 88.9 | 43.2 | 62.8 | **63.2** | **65.4** | **0.02** | **3.7** |
| **AReS-MS** | **86.6** | 48.2 | 67.1 | **85.9** | **39.4** | **85.7** | 88.9 | **65.2** | 62.8 | **63.2** | **69.3** | 0.06 | 3.7 |

### Ablation Study (Real API Evaluation, EuroSAT 16-shot)

| Method | LLaVA Acc | GPT-4o Acc | GPT-4o Total Cost ($) | Clarifai Acc | Clarifai Total Cost ($) |
|--------|-----------|-----------|----------------------|-------------|------------------------|
| Zero-shot | 40.1 | 59.4 | 14.6 | - | - |
| BAR | 34.1 | 59.1 | 72.2 | 68.1 | 48.1 |
| BlackVIP | 39.4 | 60.1 | 101.0 | 72.1 | 67.3 |
| **AReS** | **73.1** | **87.2** | **0.3** | **83.2** | **0.2** |

### Key Findings

- **Substantial advantage on GPT-4o**: AReS achieves a +27.8% gain on GPT-4o (59.4→87.2), while BlackVIP improves by only +0.7%, directly confirming the claim that ZOO methods fail against robustly trained APIs.
- **Over 99.99% reduction in API calls**: AReS requires only 0.02M API calls (vs. 754M for BlackVIP), and training time is reduced from 197 hours to 3.7 hours.
- **Component analysis**: Priming only (45.6%) < Local VR only (70.6%) < Local LP only (73.8%) < Local VR+LP (80.1%) < Full AReS (85.7%), demonstrating a significant synergistic effect between priming and reprogramming.
- **Utilization of additional unlabeled data**: The priming stage can leverage unlabeled downstream data to further improve performance, an advantage that ZOO methods cannot achieve.

## Highlights & Insights

- **Paradigm shift**: The approach moves from "continuous optimization on a black-box model" to "one-time interaction followed by local optimization"—a fundamental conceptual change. The key insight is that rather than expending large resources to directly optimize a black-box model, it is more effective to use limited resources to make the local model more programmable.
- **Priming ≠ Distillation**: Priming is mechanistically similar to knowledge distillation but serves an entirely different purpose. Distillation pursues final performance and requires label space alignment; priming only "prepares" the local model and works even when the label spaces are completely different. This distinction is particularly elegant.
- **Theoretical guarantee**: By introducing the $\epsilon$-faithful priming assumption, a performance bound $\mathcal{R}_L(\mathcal{D}^T, \mathbf{P}^*) - \epsilon \leq \mathcal{R}_S(\mathcal{D}^T, \mathbf{Q}^*) \leq \mathcal{R}_L(\mathcal{D}^T, \mathbf{P}^*)$ is established, transforming the unstable ZOO optimization into a stable local optimization problem.

## Limitations & Future Work

- On Food101 and Cars, AReS (along with all reprogramming methods) underperforms zero-shot CLIP, indicating an inherent limitation of input-level visual reprogramming for certain data domains.
- The priming stage still requires one API query per training sample, which may incur non-trivial cost at scale.
- Evaluation is limited to image classification; applicability to tasks requiring dense prediction (e.g., detection, segmentation) remains unknown.
- The choice of local encoder substantially affects performance (ViT-B/16 vs. RN50 gap is considerable); automatic selection of the optimal local encoder warrants further investigation.

## Related Work & Insights

- **vs. BlackVIP**: BlackVIP estimates gradients via SPSA-GC and introduces a Coordinator network, but remains fundamentally a ZOO method. AReS completely avoids black-box optimization through priming. Both methods assume access to an identical local encoder.
- **vs. BAR**: BAR employs random gradient-free (RGF) optimization, incurring even more API calls (612M vs. 754M for BlackVIP) and performing worse on powerful APIs.
- **vs. Knowledge Distillation**: Conventional distillation requires label space alignment and aims for the student model to independently achieve high performance. AReS's priming tolerates label space mismatch, with a subsequent reprogramming step to bridge the remaining gap.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Paradigm-shift-level innovation, transitioning from continuous black-box optimization to one-time priming followed by local optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 datasets and multiple service models (CLIP/ViT/LLaVA/GPT-4o/Clarifai), with real-cost comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Narrative is fluent and experimental design is elegant, though notation is occasionally overloaded.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a real practical pain point (API cost), with particular relevance in the era of GPT-4o.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model](../../ICCV2025/multimodal_vlm/interpretable_zero-shot_learning_with_locally-aligned_vision-language_model.md)
- [\[CVPR 2026\] Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation](locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination.md)
- [\[CVPR 2026\] EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](ebmc_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] Revisiting Model Stitching in the Foundation Model Era](revisiting_model_stitching_in_the_foundation_model.md)
- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
