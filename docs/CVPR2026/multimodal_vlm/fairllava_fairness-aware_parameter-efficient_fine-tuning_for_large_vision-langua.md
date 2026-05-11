---
title: >-
  [Paper Note] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Fairness] This paper proposes FairLLaVA, a parameter-efficient fairness-aware fine-tuning method that eliminates demographic shortcuts in multimodal large language models by minimizing the mut…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Fairness"
  - "MLLM"
  - "Mutual Information"
  - "LoRA"
  - "Medical Image Analysis"
date: 2026-05-08
content_hash: 3b80bbdaae518480
---

# FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.26008](https://arxiv.org/abs/2603.26008)
**Code**: [github.com/bhosalems/FairLLaVA](https://github.com/bhosalems/FairLLaVA)
**Area**: Multimodal VLM
**Keywords**: Fairness, MLLM, Mutual Information, LoRA, Medical Image Analysis

## TL;DR

This paper proposes FairLLaVA, a parameter-efficient fairness-aware fine-tuning method that eliminates demographic shortcuts in multimodal large language models by minimizing the mutual information between hidden states and demographic attributes, significantly narrowing inter-group performance gaps in chest X-ray report generation and skin lesion question answering.

## Background & Motivation

Multimodal large language models (MLLMs) have demonstrated strong capabilities in medical imaging tasks, yet exhibit serious fairness concerns:

**Empirical performance disparities**: Systematic performance gaps exist across age, gender, and race groups and cannot be simply attributed to sample imbalance — on MIMIC-CXR, "White" is the largest demographic group, yet multiple MLLMs perform worse on it.

**Sensitive information leakage in medical images**: Models can predict self-reported race from X-ray images with high AUC even when images are corrupted, indicating that demographic signals are systematically encoded in the representations.

**Failure of existing approaches**:
   - Resampling/reweighting: Assumes disparities stem from quantity imbalance, whereas the actual driving factor is cross-attribute dependencies.
   - Adversarial classifiers: Introduce pre-trained discriminators but cause catastrophic forgetting of clinical knowledge.
   - Word-level fairness metrics (e.g., pronoun frequency) are inapplicable, as radiology reports rarely contain demographic marker words.

**Evaluation gap**: Fairness metrics for discriminative tasks (TPR/FPR gap) cannot be directly applied to open-ended text generation.

## Method

### Overall Architecture

Two-stage training pipeline:
- **Stage 1**: The image encoder and language model are frozen; the multimodal projector $\psi$ is fine-tuned to align visual-language representations.
- **Stage 2**: The image encoder and language model backbone are frozen; fairness-aware fine-tuning is performed via LoRA adapters $\theta$ with mutual information regularization.

### Key Designs

1. **Demographic Attribute Classifier (DAC)**:

    - A lightweight variational MLP $\phi$ that maps pooled hidden states $h(x)$ to demographic attributes.
    - Training objective: $\mathcal{L}_{DAC} = -\mathbb{E}[\log \phi(\mathbf{a} | h(x))]$
    - Function: Exposes the locations within hidden states where demographic information leaks.
    - Key constraint: A stop-gradient is applied to $h(x)$; $\mathcal{L}_{DAC}$ updates only $\phi$.

2. **Demographic Information Minimization (DIM)**:

    - Realized via variational approximation of an upper bound on mutual information.
    - $\mathcal{L}_{DIM}$ incorporates positive pairs (attribute and hidden state of the same sample) and negative pairs (cross-matched hidden states and attributes from different samples).
    - $\phi$ is frozen; only LoRA $\theta$ and projector $\psi$ are updated.
    - Intuition: The predictable $h(x) \rightarrow \mathbf{a}$ link is treated as a "leak," and representations are encouraged to discard these signals.

3. **Equity-Scaled Metric (ES-M)**:

    - Addresses the spurious fairness problem where uniformly low performance can yield small fairness gaps.
    - Defined as: $ES\text{-}M_a = \frac{M_{all}}{1 + \Delta M_a}$, jointly accounting for overall performance and inter-group disparity.
    - Compatible with any language evaluation metric (BLEU, RadGraph-F1, GREEN, etc.).
    - Generalizes prior fairness metrics originally designed for discriminative tasks to generative settings.

4. **Joint Multi-Attribute Debiasing**:

    - $\mathcal{L}_{DAC}^{(a)}$ and $\mathcal{L}_{DIM}^{(a)}$ are computed separately for each attribute $a \in \mathcal{A}$ and aggregated for unified optimization.
    - Outperforms single-attribute debiasing, which improves the target attribute while degrading fairness gaps on others.

5. **Hidden Layer Selection**:

    - Ablation studies show that the middle layer (layer 16) achieves the best fairness-performance trade-off.
    - Early, middle, and late layers are associated with visual grounding, reasoning, and task decoding, respectively.
    - In practice, the mean pooling of first, middle, and last layers is used.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{DAC} + \lambda_2 \mathcal{L}_{DIM} + \lambda_3 \mathcal{L}_{LM}$

Alternating optimization protocol:
1. Optimize $\mathcal{L}_{DAC}$ (stop-gradient on $h$), updating $\phi$.
2. Freeze $\phi$; optimize $\mathcal{L}_{DIM} + \mathcal{L}_{LM}$, updating LoRA $\theta$ and projector $\psi$.

Base model: Vicuna-7b-v1.5 + BioMedCLIP image encoder.
Training: 8×RTX A6000, approximately 17 hours on MIMIC-CXR, 1 epoch.
DAC parameter overhead is only ~57K even with 14 attribute classes.

## Key Experimental Results

### Main Results

MIMIC-CXR joint debiasing (All attributes, 12 ES metrics):

| Method | ES-BLEU1(R) | ES-BLEU4(R) | ES-RG-F1(R) | ES-BLEU1(A) | ES-RG-F1(A) | ES-BLEU1(G) | ES-RG-F1(G) |
|------|------------|------------|-------------|------------|-------------|------------|-------------|
| LLaVA-Rad | 5.29 | 2.14 | 4.14 | 8.28 | 1.42 | 28.06 | 9.24 |
| Resampling | 8.71 | 2.32 | 2.95 | 12.54 | 3.01 | 30.38 | 15.97 |
| Reweighting | 1.81 | 1.73 | 3.31 | 7.36 | 2.17 | 11.72 | 9.88 |
| **FairLLaVA** | **13.36** | **8.65** | **6.34** | **21.89** | **4.06** | **24.89** | **19.40** |

FairLLaVA-All achieves the **best performance on 7 out of 12** ES metrics and demonstrates consistent advantages on clinical semantic metrics (RadGraph-F1).

CheXpert-F1 ES metrics (direct comparison with Chen et al.):

| Method | Race↑ | Age↑ | Gender↑ |
|------|-------|------|---------|
| Chen et al. | 24.06 | 23.85 | 24.13 |
| **FairLLaVA** | **69.21** | **68.70** | **69.38** |

FairLLaVA also leads on PadChest and HAM10000, validating cross-modal generalization from grayscale X-rays to RGB skin lesion images.

### Ablation Study

Hidden layer selection ablation (MIMIC-CXR; lower Δ is better):

| Pooling Layer | BLEU4-Δ(R)↓ | RG-F1-Δ(R)↓ | BLEU4-Δ(A)↓ | Overall BLEU4↑ |
|--------|-------------|-------------|-------------|---------------|
| first | 3.40 | 4.42 | 2.53 | 13.62 |
| last | 4.48 | 3.90 | 2.40 | 13.19 |
| mean | 3.07 | 4.52 | 2.16 | **14.84** |
| **mid** | **0.61** | **3.50** | **1.01** | 14.01 |

The middle layer is optimal on 5 out of 6 gap metrics while maintaining reasonable overall performance.

### Key Findings

- **Joint multi-attribute debiasing outperforms single-attribute debiasing**: Single-attribute debiasing improves the target attribute but degrades fairness on others (a "seesaw effect"), whereas joint debiasing yields comprehensive and balanced improvements.
- **FairLLaVA does not trade overall performance for fairness**: On PadChest, it simultaneously achieves the best overall performance and the best ES metrics.
- **The assumptions underlying resampling/reweighting do not hold**: Demographic disparities stem not only from quantity imbalance but also from cross-attribute dependencies and latent demographic signals encoded in images.
- **Middle layers are the primary site of demographic shortcut leakage**: This is consistent with the theoretical understanding that early, middle, and late layers process different types of information.

## Highlights & Insights

1. **The mutual information regularization design is elegant and theoretically grounded**: The alternating optimization — DAC exposing leakage, DIM eliminating it — avoids the instability of direct adversarial training.
2. **The proposed ES-M metric fills an important gap**: It addresses the "low-performance spurious fairness" pitfall in evaluating generative model fairness.
3. **Minimal additional overhead**: Only ~57K MLP parameters plus standard LoRA fine-tuning, requiring no data augmentation or preference data.
4. **Cross-modal generalization**: The same framework is effective on grayscale chest X-rays (MIMIC-CXR, PadChest) and RGB skin lesion images (HAM10000).

## Limitations & Future Work

- The base model is limited to Vicuna-7b; applicability to larger models (e.g., LLaMA-3) remains unverified.
- DAC relies on the availability of demographic labels; label-free scenarios require a self-supervised variant.
- Only three demographic attributes (age, gender, race) are addressed; handling more complex attributes such as socioeconomic status warrants further exploration.
- FairLLaVA does not consistently achieve the best results on the GREEN metric; the variance of LLM-based evaluation metrics requires further analysis.
- Generalization to non-English radiology report datasets has not been evaluated.

## Related Work & Insights

- The MI minimization debiasing strategy is generalizable to any multimodal large model task requiring attribute-invariant representations.
- The ES-M metric can be extended to all fairness evaluation scenarios involving open-ended text generation.
- The finding that "images leak sensitive attributes" (Gichoya et al.) suggests that debiasing at the visual encoder level may also be necessary.

## Rating

- Novelty: ⭐⭐⭐⭐ — The application of MI regularization to MLLM fairness is novel, and the ES-M metric represents a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 datasets × 3 demographic attributes × 6 metrics × multiple baselines, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Fig. 1 and Fig. 2 clearly convey the core ideas; Algorithm 1 is complete and accurate.
- Value: ⭐⭐⭐⭐⭐ — The first systematic treatment of fairness in medical imaging MLLMs, with direct relevance to trustworthy AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)

</div>

<!-- RELATED:END -->
