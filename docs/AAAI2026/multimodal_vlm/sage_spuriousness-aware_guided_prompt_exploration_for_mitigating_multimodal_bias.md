---
title: >-
  [Paper Note] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias
description: >-
  [AAAI 2026][Multimodal VLM][Spurious correlations] This paper proposes SAGE, a training-free prompt selection method that requires no fine-tuning or external annotations. By computing inter-class separation scores for prompt templates, SAGE mitigates multimodal spurious bias in CLIP models, consistently improving Worst Group Accuracy (WGA) and Harmonic Mean (HM) across four benchmarks and five backbone models.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Spurious correlations
  - CLIP
  - zero-shot classification
  - prompt selection
  - robustness
date: 2026-05-08
content_hash: fbd860bdfdaf0ad5
---

# SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias

**Conference**: AAAI 2026
**arXiv**: [2511.13005](https://arxiv.org/abs/2511.13005)
**Code**: [https://github.com/wenqian-ye/spurious_vlm](https://github.com/wenqian-ye/spurious_vlm)
**Area**: Multimodal VLM
**Keywords**: Spurious correlations, CLIP, zero-shot classification, prompt selection, robustness

## TL;DR
This paper proposes SAGE, a training-free prompt selection method that requires no fine-tuning or external annotations. By computing inter-class separation scores for prompt templates, SAGE mitigates multimodal spurious bias in CLIP models, consistently improving Worst Group Accuracy (WGA) and Harmonic Mean (HM) across four benchmarks and five backbone models.

## Background & Motivation

### State of the Field
Pretrained vision-language models such as CLIP achieve strong zero-shot classification by aligning images and text in a shared embedding space. However, CLIP models frequently develop **multimodal spurious bias**, whereby predictions rely on non-essential features.

### Limitations of Prior Work

**Nature of spurious correlations**: In pretraining data, "landbirds" frequently co-occur with "land backgrounds," causing CLIP to align "land background" features with the textual representation of "landbird." When presented with a "waterbird on a land background," the model incorrectly predicts "landbird." Such cross-modal spurious associations severely degrade zero-shot generalization on out-of-distribution data.

### Limitations of Existing Approaches
- **Fine-tuning methods** (Yang et al. 2023; You et al. 2024): require downstream labeled data and cannot handle zero-shot scenarios.
- **ROBOSHOT** (Adila et al. 2024): requires an LLM to explicitly specify spurious attributes for each task.
- **TIE***: directly uses spurious attributes to generate pseudo-labels, also relying on prior knowledge.

**Key Challenge**: Existing methods either require training data or prior knowledge—both of which undermine CLIP's out-of-the-box zero-shot advantage.

### Starting Point
Can spurious bias be mitigated solely by **selecting appropriate text prompt templates**? SAGE's core insight is: **prompts with higher separation scores better capture the core class semantics, thereby reducing reliance on spurious features**. When a prompt produces large inter-class similarity differences, it focuses on essential features; conversely, similar scores across classes suggest that spurious features are "confusing" the classifier.

## Method

### Overall Architecture
SAGE follows a straightforward pipeline: (a) prepare $M$ candidate prompt templates; (b) for each test image, compute the inter-class separation score for each prompt; (c) select the top-$K$ prompts with the highest separation scores for ensemble zero-shot classification.

### Key Designs

1. **Theoretical Analysis of Multimodal Spurious Bias**:

    - Introduces the concept of latent spurious feature $\mathbf{u}_s$.
    - **Definition 1 (Multimodal Spurious Bias)**: When $p(\mathbf{u}_s|\mathbf{u}_1) \approx 1$ and $p(\mathbf{u}_s|\mathbf{v}) \approx 1$, the spurious feature is highly correlated with both the class label and the image representation.
    - **Theorem 1**: Proves that under spurious bias, the model is biased toward incorrectly predicting class $c_1$, since $\frac{p(\mathbf{u}_1|\mathbf{v})}{p(\mathbf{u}_2|\mathbf{v})} \approx \frac{p(\mathbf{u}_s|\mathbf{u}_1)p(\mathbf{u}_1)}{p(\mathbf{u}_2|\mathbf{v})} > 1$.
    - **Key Corollary**: Prompt templates directly control $p(\mathbf{u}_s|\mathbf{u}_1)$. Prompts strongly affected by spurious bias drive inter-class similarities closer together → lower separation; prompts less affected yield higher separation.

2. **Separation Score**:

    - For prompt template $T_j$ and image $x_n$, the score is computed as: $\sigma_j^n = \max_i \frac{\mathbf{v}_n^T\mathbf{u}_i^j}{\|\mathbf{v}_n\|_2\|\mathbf{u}_i^j\|_2} - \min_i \frac{\mathbf{v}_n^T\mathbf{u}_i^j}{\|\mathbf{v}_n\|_2\|\mathbf{u}_i^j\|_2}$
    - That is, the difference between the maximum and minimum class-wise cosine similarities under a given prompt.
    - Higher separation → the prompt better discriminates core semantics → less susceptible to spurious features.

3. **Template Selection and Ensemble Inference**:

    - For each image, all $M$ prompt templates are ranked by their separation scores.
    - The top-$K$ prompts are selected to construct $K$ zero-shot classifiers.
    - Ensemble prediction: $\hat{y_n} = \arg\max_i \frac{1}{K}\sum_{k=1}^K \frac{\mathbf{v}_n^T\mathbf{u}_i^k}{\|\mathbf{v}_n\|_2\|\mathbf{u}_i^k\|_2}$
    - Default setting: $K=1$ (i.e., single best prompt selected per image).

### Loss & Training
SAGE is entirely **training-free**. No parameter updates, fine-tuning, or external model calls are involved. Prompt selection is computed at inference time only.

## Key Experimental Results

### Main Results

Average results across four benchmark datasets (over 5 backbone models):

| Method | Waterbirds WGA | Waterbirds HM | CelebA WGA | CelebA HM | PACS WGA | VLCS WGA |
|--------|---------------|---------------|------------|-----------|----------|----------|
| ZS (baseline) | 36.7 | 51.1 | 75.3 | 78.1 | 75.5 | 23.0 |
| ROBOSHOT | 41.5 | 52.5 | 79.5 | 81.9 | 78.0 | 30.1 |
| TIE* | 38.4 | 52.8 | 69.8 | 73.1 | 77.2 | 31.7 |
| **SAGE** | **44.9** | **59.7** | **80.6** | **82.0** | **81.9** | **33.8** |

SAGE leads all datasets on both WGA and HM **without requiring any prior knowledge**.

### Ablation Study (vs. Random Selection and Full Ensemble)

| Strategy | Waterbirds WGA | Waterbirds HM | CelebA WGA | CelebA HM |
|----------|---------------|---------------|------------|-----------|
| Full ensemble ($K$=80) | 36.2 | 51.6 | 73.2 | 76.5 |
| Random selection | 40.1 | 55.1 | 74.9 | 77.3 |
| **SAGE** | **44.9** | **59.7** | **80.6** | **82.0** |

Key finding: **Full ensemble ($K$=80) performs worst**, as aggregating numerous spuriously biased prompts dilutes the effect of high-quality ones.

### Per-Model Results on Waterbirds WGA

| Model | ZS | SAGE | Gain |
|-------|----|------|------|
| CLIP-RN-50 | 41.0 | 41.3 | +0.3 |
| CLIP-ViT-B/32 | 27.5 | **46.0** | **+18.5** |
| CLIP-ViT-L/14 | 27.6 | **47.8** | **+20.2** |
| ALIGN | 50.0 | 47.0 | -3.0 |
| AltCLIP | 37.2 | **42.6** | **+5.4** |

### Key Findings

1. **Separation score is an effective proxy for spurious bias**: Theory and experiments consistently show that high-separation prompts focus on core semantics.
2. **Prompt selection critically matters**: Different prompt templates vary substantially in their sensitivity to spurious bias—selecting the right prompt yields large robustness gains.
3. **Full ensemble is harmful**: Ensembling 80 prompts underperforms a single optimal prompt, indicating that "more is better" does not hold in the presence of spurious bias.
4. **Larger models benefit more**: CLIP-ViT-L/14 WGA improves from 27.6 to 47.8 (+20.2), while ResNet-50 gains only +0.3.
5. **No prior knowledge required**: SAGE is the only method that requires no spurious attribute information, yet achieves the best performance.

## Highlights & Insights

- **Theory-driven method design**: The paper provides a complete and elegant theoretical derivation, from a formal definition of spurious bias to proving the relationship between separation score and bias.
- **Extreme simplicity**: The entire method reduces to computing differences in cosine similarities and ranking, with zero training cost.
- **Generality**: Applicable to any CLIP-based zero-shot vision-language model without modification.
- **Harmonic Mean (HM) as an evaluation metric**: Simultaneously captures average accuracy and worst-group accuracy, providing a more comprehensive assessment than either metric alone.
- Reveals a counterintuitive phenomenon: **ensembling more prompts reduces robustness**.

## Limitations & Future Work

- SAGE relies on a predefined candidate prompt set; prompt set quality may affect performance.
- WGA on Waterbirds slightly decreases for the ALIGN model (50.0→47.0), indicating that not all model–dataset combinations benefit.
- The default $K=1$ setting may not be optimal in all scenarios, requiring dataset-level hyperparameter tuning.
- Computational overhead scales linearly with the number of candidate prompts $M$ and the number of classes $C$, which may require optimization for large-scale classification.
- Validation is limited to classification tasks; applicability to other CLIP use cases such as retrieval and segmentation remains unexplored.

## Related Work & Insights

- **Fundamental distinction from ROBOSHOT and TIE***: The latter methods require knowledge of what constitutes spurious features, while SAGE requires none.
- **Connection to prompt engineering**: SAGE provides a theoretical basis for prompt selection from a robustness perspective.
- Spurious bias has been extensively studied in unimodal settings (DRO, reweighting, etc.); this paper extends the problem to multimodal zero-shot scenarios.
- **Inspiration**: The separation score could be applied to **automated prompt filtering** or as a **regularizer in prompt learning** in future work.

## Rating
- Novelty: ⭐⭐⭐⭐ (novel theoretical analysis with an exceptionally simple method)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets × 5 models × 3 baselines + ablation)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear theoretical derivation, complete motivation chain)
- Value: ⭐⭐⭐⭐ (direct practical value for zero-shot CLIP deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification](../../ICLR2026/multimodal_vlm/lets_think_in_two_steps_mitigating_agreement_bias_in_mllms_with_self-grounded_ve.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](../../CVPR2026/multimodal_vlm/capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](../../ICLR2026/multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)
- [\[AAAI 2026\] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models](biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)
- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/multimodal_vlm/flashcache_frequency_kv_cache_compression.md)

</div>

<!-- RELATED:END -->
