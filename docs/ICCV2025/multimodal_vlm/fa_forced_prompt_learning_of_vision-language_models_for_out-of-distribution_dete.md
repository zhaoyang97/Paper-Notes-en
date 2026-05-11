---
title: >-
  [Paper Note] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection
description: >-
  [ICCV 2025][Multimodal VLM][OOD detection] This paper proposes FA (Forced prompt leArning), which introduces a learnable "forced prompt" and trains it to produce higher ID-class matching scores than a frozen original pro…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "OOD detection"
  - "CLIP prompt learning"
  - "few-shot learning"
  - "out-of-distribution detection"
  - "forced prompt"
date: 2026-05-08
content_hash: 0ac5c93ce625c792
---

# FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection

**Conference**: ICCV 2025
**arXiv**: [2507.04511](https://arxiv.org/abs/2507.04511)
**Code**: [https://github.com/0xFAFA/FA](https://github.com/0xFAFA/FA)
**Area**: Multimodal VLM
**Keywords**: OOD detection, CLIP prompt learning, few-shot learning, out-of-distribution detection, forced prompt

## TL;DR
This paper proposes FA (Forced prompt leArning), which introduces a learnable "forced prompt" and trains it to produce higher ID-class matching scores than a frozen original prompt, compelling it to capture richer ID class descriptions beyond label text semantics. FA achieves significant improvements in CLIP-based few-shot OOD detection without external auxiliary data or additional parameters.

## Background & Motivation
OOD detection aims to determine whether a test sample belongs to the training distribution (ID), and is critical for AI reliability. CLIP-based methods excel in few-shot OOD detection due to their strong zero-shot generalization, yet existing approaches exhibit a fundamental methodological divide:

**Dominant paradigm — learning OOD-related knowledge**:
- LoCoOp/SCT: maximize entropy of ID-irrelevant regions to push OOD embeddings away
- ID-like: construct OOD samples from the neighborhood of ID samples to learn fine-grained differences
- NegPrompt/LSN: learn negative prompts representing the "opposite" of ID classes
- Methods relying on large-scale external auxiliary datasets

The common issue with these approaches: OOD data is **infinite and unknowable** in practice, making explicit modeling of OOD knowledge inherently limited. Generalizing from specific exposed OOD features to unknown OOD distributions is fundamentally difficult.

**This paper's new paradigm — fully exploiting ID knowledge**:
Inspired by open-set recognition research (where improving closed-set accuracy typically also improves open-set recognition), the authors argue that **rather than laboriously modeling infinite OOD patterns, the model should more deeply understand ID classes**. If the model possesses sufficiently rich and precise descriptions of ID classes, OOD samples will naturally fail to match them.

Core Idea: Learn a "forced prompt" whose text features produce ID-image similarities that **exceed** those of the original hand-crafted prompt, compelling it to acquire richer ID class descriptive information beyond the label name.

## Method

### Overall Architecture
The FA framework contains two prompts — a frozen original prompt (reference baseline) and a learnable forced prompt (learning target), both initialized identically ("a photo of a [class-c]"). During training, the original prompt is frozen and only the forced prompt is optimized, such that ID image similarity to the forced prompt exceeds similarity to the original prompt.

### Key Designs
1. **Forced Prompt**:

    - Function: Introduces a learnable copy of the original prompt with identical initialization
    - Mechanism: Prompt format $\mathbf{u}_c = [\mathbf{v}_1, \cdots, \mathbf{v}_L, \mathbf{w}_c]$, where $\mathbf{v}_i$ are class-shared learnable vectors (consistent with CoOp) and $\mathbf{w}_c$ is the frozen class name embedding
    - Design Motivation:
        - Hand-crafted template initialization (rather than random) preserves semantic information from CLIP priors, improving generalization
        - Shared learnable vectors (rather than class-specific) prevent overfitting from excessive parameters in few-shot settings
        - Freezing the original prompt as a reference baseline ensures the forced prompt must learn "more" rather than merely drifting

2. **Forced Cross-Entropy Loss (FCE Loss)**:

    - Function: Modifies the standard cross-entropy denominator to incorporate scores from the original prompt into the normalization
    - Core formula:
    $\mathcal{L}_{FCE} = \mathbb{E}_{(\mathbf{x}, y_c)} \left[-\log \frac{e^{s_c^f/\tau}}{\sum_{j=1}^C e^{s_j^f/\tau} + \sum_{j=1}^C e^{s_j^o/\tau}}\right]$
      where $s_j^f = \cos(\mathbf{z}, \mathbf{t}_j^f)$ and $s_j^o = \cos(\mathbf{z}, \mathbf{t}_j^o)$ denote image similarities to the forced and original prompts, respectively
    - Design Motivation: Including original prompt similarities in the denominator means matching the original prompt is insufficient — the forced prompt must produce **higher** similarities to minimize the loss, compelling it to learn richer and more discriminative ID class descriptions

3. **Forcing Coefficient $K$**:

    - Function: Controls the weight of the original prompt in the denominator, regulating the intensity of "forcing"
    - Core formula:
    $\mathcal{L}_{FCE-K} = \mathbb{E}_{(\mathbf{x}, y_c)} \left[-\log \frac{e^{s_c^f/\tau}}{\sum_{j=1}^C e^{s_j^f/\tau} + K\sum_{j=1}^C e^{s_j^o/\tau}}\right]$
    - Design Motivation: $K=0$ degenerates to CoOp (no forcing constraint); larger $K$ compels the forced prompt to learn more comprehensive ID descriptions. Experiments show $K=1$ already yields significant improvement, with the optimal value being dataset-dependent

4. **Inference Scoring Functions**:

    - MCM score: $S_{MCM}(\mathbf{x}) = \max_c \frac{e^{\cos(\mathbf{z}^g, \mathbf{t}_c^a)/\tau_0}}{\sum_j e^{\cos(\mathbf{z}^g, \mathbf{t}_j^f)/\tau_0} + K e^{\cos(\mathbf{z}^g, \mathbf{t}_j^o)/\tau_0}}$
    - GL-MCM score: jointly considers global and local feature matching
    - $\mathbf{t}_c^a$ is the concatenation of forced and original prompt features

### Loss & Training
- Only the shared token vectors of the forced prompt are trained, with parameter count identical to CoOp
- The original prompt and class name embeddings are frozen to preserve CLIP's generalization capability
- No external OOD data or auxiliary datasets are used

## Key Experimental Results

### Main Results (ImageNet-1k as ID, 1-shot setting)

| Method | iNaturalist FPR95↓ | SUN FPR95↓ | Places FPR95↓ | Textures FPR95↓ | Avg. FPR95↓ | Avg. AUROC↑ |
|------|-------------------|-----------|--------------|----------------|-----------|-----------|
| SCT_GL (SOTA) | 20.57 | 24.56 | 33.27 | 48.12 | 31.62 | 92.01 |
| LoCoOp_GL | 21.97 | 24.95 | 34.14 | 49.04 | 32.53 | 92.17 |
| IDLike | 17.73 | 48.17 | 50.43 | 29.12 | 36.36 | 91.93 |
| **FA_GL (Ours)** | **14.12** | **29.99** | **32.48** | **34.66** | **27.81** | **93.26** |

### Ablation Study / Additional OOD Benchmarks

| Configuration | Challenging OOD (16-shot) Avg. FPR95↓ | Avg. AUROC↑ | Notes |
|------|-------------------------------|-----------|------|
| CoOp_GL | 57.21 | 81.99 | Baseline |
| SCT_GL | 58.25 | 82.24 | Marginal improvement over CoOp |
| LoCoOp_GL | 59.03 | 82.23 | Similar |
| **FA_GL (Ours)** | **53.61** | **83.93** | Significant improvement |

| ID Accuracy (ImageNet Top-1) | 1-shot | 4-shot | 16-shot |
|------------------------|--------|--------|---------|
| CoOp | 67.44 | 69.71 | 70.99 |
| LoCoOp | 67.40 | 69.55 | 71.53 |
| SCT | 68.63 | 69.93 | 71.78 |
| **FA (Ours)** | **68.67** | **69.96** | 71.02 |

### Key Findings
- **FA outperforms all methods without any external data**: 1-shot FPR95 drops from 31.62% to 27.81%, AUROC improves from 92.01% to 93.26%
- **ID classification accuracy improves rather than degrades**: The richer descriptions learned by the forced prompt also benefit classification
- **Compatible with MCM/GL-MCM scoring functions**: FA can be flexibly combined with different OOD detection scores
- **Larger advantage on challenging OOD datasets**: On cleaner benchmarks such as OpenImage-O, NINCO, and ImageNet-O, FA_GL's FPR95 is 4.64 percentage points lower than SCT_GL
- **Zero parameter overhead**: The number of learnable parameters is identical to CoOp
- OOD-knowledge-learning methods (e.g., NegPrompt with FPR95 as high as 62.08%) perform poorly in certain settings

## Highlights & Insights
1. **Paradigm reversal**: Rather than learning complex OOD patterns, the model is made to better understand ID classes — "knowing oneself" rather than "knowing the enemy"
2. **Minimal design, strong results**: Only the loss denominator and a scalar coefficient $K$ are modified, with no additional modules or parameters, yet significant gains are achieved
3. **Elegance of the forcing mechanism**: By placing the frozen original prompt as a "competitor" in the denominator, the learnable prompt is naturally compelled to surpass the information content of semantic labels
4. **Theoretical intuition**: High matching of ID samples with the forced prompt implies a more compact ID distribution representation, making the differential between OOD samples' similarities to the two prompts more distinguishable

## Limitations & Future Work
1. The forcing coefficient $K$ requires tuning; although the authors claim low sensitivity, the optimal value still varies across datasets
2. When $K=0$ degenerates to CoOp, performance is already reasonable, suggesting FA's gains stem partly from the forcing mechanism and partly from initialization strategy — their individual contributions are not fully decoupled
3. Experiments are conducted only on ViT-B/16; larger backbones or other VLM architectures are not evaluated
4. At 16-shot, FA's ID accuracy is slightly lower than LoCoOp and SCT, suggesting that the forcing constraint may be less effective than direct learning when data is sufficient
5. The combination of FA with existing OOD-knowledge-learning methods is unexplored — are the two paradigms complementary?

## Related Work & Insights
- **Relationship to CoOp**: FA can be viewed as an improved CoOp — the same parameter budget, but learning better prompts via a frozen reference and a forcing loss
- **Relationship to LoCoOp/SCT**: The latter push OOD representations away via entropy maximization; FA pulls ID representations closer — opposite directions that are potentially complementary
- **Relationship to NegPrompt**: NegPrompt explicitly learns negative prompts (OOD knowledge); FA implicitly repels OOD samples by enhancing ID knowledge — the latter is more stable in few-shot settings
- **Insight**: In open-set/OOD problems, **improving the representational quality of known classes** may be more efficient and reliable than **explicitly modeling unknown classes** — consistent with the human cognitive intuition that deeper familiarity yields greater certainty

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm reversal is clever, and the forced prompt design is elegantly simple
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple OOD benchmarks, various shot settings, and challenging datasets, though only one model architecture is tested
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is articulated with exceptional clarity; method derivation flows naturally
- Value: ⭐⭐⭐⭐ Offers a new paradigm for OOD detection, though broader experiments are needed to validate generalizability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out_of_distribution_detection.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/multimodal_vlm/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[ICCV 2025\] FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)

</div>

<!-- RELATED:END -->
