---
title: >-
  [Paper Note] Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification
description: >-
  [CVPR 2026][Medical Imaging][Large Vision-Language Models] This paper introduces the concept of Attention Imbalance to explain object hallucination in LVLMs, and proposes a lightweight decoding-time intervention method, AIR, which rectifies attention imbalance via cross-modal attention reallocation and variance-constrained projection regularization. AIR reduces hallucination rates by up to 35.1% and improves general capability by up to 15.9% across four LVLMs.
tags:
  - CVPR 2026
  - Medical Imaging
  - Large Vision-Language Models
  - Object Hallucination
  - Attention Imbalance
  - Decoding-Time Intervention
  - Attention Rectification
date: 2026-05-08
content_hash: 90a65feb129ea914
---

# Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification

**Conference**: CVPR 2026
**arXiv**: [2603.24058](https://arxiv.org/abs/2603.24058)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Large Vision-Language Models, Object Hallucination, Attention Imbalance, Decoding-Time Intervention, Attention Rectification

## TL;DR

This paper introduces the concept of Attention Imbalance to explain object hallucination in LVLMs, and proposes a lightweight decoding-time intervention method, AIR, which rectifies attention imbalance via cross-modal attention reallocation and variance-constrained projection regularization. AIR reduces hallucination rates by up to 35.1% and improves general capability by up to 15.9% across four LVLMs.

## Background & Motivation

1. **Background**: Large Vision-Language Models (LVLMs) excel at cross-modal understanding tasks, but object hallucination—generating descriptions of objects absent from the image—severely undermines model reliability in high-stakes scenarios such as autonomous driving and medical imaging.
2. **Limitations of Prior Work**: Existing methods fall into three categories: visual instruction tuning (high training cost), post-processing techniques (additional inference overhead), and contrastive decoding (limited stability and generalizability). More fundamentally, root cause analysis of hallucination remains insufficient.
3. **Key Challenge**: The complex training pipelines and architectures of LVLMs hinder interpretability analysis. Prior studies examining visual information interaction, positional encoding, and anomalous tokens fail to provide a comprehensive understanding.
4. **Goal**: (1) Provide a quantitative framework to explain the attention-mechanism root causes of hallucination; (2) Design a training-free lightweight intervention method based on this framework.
5. **Key Insight**: Through systematic experiments, the authors find that attention allocation imbalance—both across modalities and across tokens—exhibits a strong causal correlation with object hallucination.
6. **Core Idea**: Hallucination stems from attention imbalance; rectifying cross-modal and token-level imbalance in hallucination-sensitive attention heads effectively mitigates the phenomenon.

## Method

### Overall Architecture

AIR is a purely inference-time decoding intervention that requires no additional training. At each decoding step, it performs two operations on hallucination-sensitive attention heads: (1) modality-balanced attention reallocation—when text attention exceeds a threshold, text token weights are suppressed and visual token weights are amplified; (2) variance-constrained projection regularization—attention distributions are smoothed via zero-trace projection, Frobenius energy preservation, and shrinkage regularization.

### Key Designs

1. **Attention Imbalance Definition (MAI + TAI)**:

    - Function: Quantitatively characterize the degree of imbalance in attention allocation.
    - Mechanism: MAI (Modality-wise Attention Imbalance) is defined as the ratio of total attention received by two modalities, $\text{MAI}(M_p, M_q) = A_{M_p}/A_{M_q}$, where values far exceeding 1 indicate dominance by $M_p$. TAI (Token-wise Attention Imbalance) is defined as the ratio of the attention proportion received by a token to its information contribution proportion, where values far exceeding 1 indicate excessive focus.
    - Design Motivation: Provides a quantifiable measurement framework for the attention-level root causes of hallucination. Experiments show that the MAI of hallucination-sensitive heads reaches 5.1 (vs. 1.5 for insensitive heads), and hallucination almost invariably occurs within 15 tokens following the appearance of a high-TAI token.

2. **Modality-Balanced Attention Reallocation**:

    - Function: Rectify attention in hallucination-sensitive heads that is excessively biased toward the text modality.
    - Mechanism: At each decoding step, the cumulative attention $V^{\text{text}}$ received by text tokens is computed for hallucination-sensitive heads. If it exceeds threshold $\tau_{\text{text}}$, text token weights are multiplied by $\lambda \in [0,1]$ (suppression) and visual token weights are multiplied by $\gamma > 1$ (amplification). Defaults: $\lambda=0.1$, $\gamma=3.5$.
    - Design Motivation: Hallucination-sensitive heads inherit the attention patterns of the base language model (cosine similarity 0.81 vs. 0.69 for insensitive heads), excessively attending to text while neglecting visual information. Targeted rectification of these heads preserves the model's normal functionality.

3. **Variance-Constrained Projection Regularization**:

    - Function: Suppress excessive concentration of attention on a small number of tokens.
    - Mechanism: Three steps: (a) adaptively scale $W_{\text{QK}}$ by its spectral energy; (b) zero-trace projection $\hat{A} = A - \frac{\text{tr}(A)}{L}I$ to remove self-alignment bias; (c) after Frobenius energy normalization to preserve magnitude, apply shrinkage regularization $A^* = (1-\beta)\tilde{A} + \beta \cdot \text{mean}(\tilde{A}) \cdot \mathbf{1}$ to produce a more uniform distribution.
    - Design Motivation: TAI analysis reveals that hallucination is always preceded by a token receiving excessively concentrated attention (e.g., the TAI of the `<0x0A>` token reaches 98). Regularization smooths the attention distribution to prevent subsequent hallucination propagation.

### Loss & Training

- AIR is a purely inference-time method and **requires no training**.
- Hallucination-sensitive heads are selected via erasure-based attribution—attention heads are removed one by one to observe changes in hallucination probability, and the top-20 heads with the greatest impact are selected.
- Hyperparameters: $\tau_{\text{text}}=0.3$, $\lambda=0.1$, $\gamma=3.5$, $\xi=0.01$, $\beta=0.3$.

## Key Experimental Results

### Main Results

CHAIR hallucination evaluation (Max New Tokens=256):

| LVLM | Metric | AIR (Ours) | Best Baseline (AD-HH) | Gain |
|------|------|------|----------|------|
| LLaVA-1.5 | $C_S$ ↓ | **28.8** | 35.2 | -18.1% |
| MiniGPT-4 | $C_S$ ↓ | **21.3** | 32.8 | -35.1% |
| InstructBLIP | $C_S$ ↓ | **30.1** | 36.0 | -16.4% |
| Shikra | $C_S$ ↓ | **30.3** | 36.9 | -17.9% |

MM-Vet general capability:

| LVLM | AIR Overall | Greedy Overall | Gain |
|------|------------|----------------|------|
| LLaVA-1.5 | **32.0** | 27.6 | +15.9% |
| MiniGPT-4 | **22.0** | 20.0 | +10.0% |

### Ablation Study

| Configuration | $C_S$ ↓ | $C_I$ ↓ | MM-Vet ↑ | Note |
|------|---------|---------|----------|------|
| Greedy (baseline) | 51.8 | 13.7 | 27.6 | No intervention |
| R-only (reallocation only) | 32.1 | 9.9 | 30.5 | Text suppression + visual amplification effective |
| P-only (projection only) | 38.4 | 11.2 | 29.8 | Attention uniformization effective |
| Full AIR | **28.8** | **8.6** | **32.0** | Complementary, best overall |

### Key Findings

- Attention reallocation contributes more ($C_S$ drops from 51.8 to 32.1), indicating that cross-modal imbalance is the primary cause of hallucination.
- AIR's unique advantage is that it **simultaneously reduces hallucination and improves general capability**—other methods (e.g., AD-HH) reduce hallucination but degrade general capability by 14.8%.
- Hallucination-sensitive heads are predominantly concentrated in the middle layers of the model, consistent with prior research.
- The co-occurrence of high-TAI tokens and hallucination is observed consistently across all four LVLMs, suggesting that attention imbalance is a universal root cause of hallucination.
- Hallucination exhibits a "snowball effect"—a single hallucinated word triggers further hallucinations in subsequent generation.

## Highlights & Insights

- **Clear Causal Chain**: The paper establishes a complete causal analysis chain of hallucination, from TAI/MAI definition → co-occurrence verification → head-level attribution → inheritance hypothesis validation. This constitutes not only a methodological contribution but also a significant advance in LVLM interpretability.
- **Zero Training Overhead**: AIR operates entirely at inference time, introducing no additional parameters or training cost, making it highly practical.
- **Discovery that Hallucination-Sensitive Heads Inherit Base LM Patterns**: This finding suggests that visual alignment training in LVLMs fails to sufficiently alter the text-only bias of certain attention heads, providing direction for future improvements in training strategies.

## Limitations & Future Work

- Identifying hallucination-sensitive heads requires prior erasure-based analysis, adding preparation overhead before deployment.
- Hyperparameters such as $\tau_{\text{text}}$, $\lambda$, and $\gamma$ may require adjustment for different models.
- Only 7B-scale models are evaluated; effectiveness on larger models (70B+) remains unknown.
- Future work may explore incorporating AIR's insights into the training phase to design attention-balanced fine-tuning objectives.

## Related Work & Insights

- **vs VCD (ICLR24)**: VCD mitigates language priors by contrasting output distributions with and without visual input, but exacerbates hallucination on certain LVLMs ($C_S$ increases from 51.8 to 59.4 on LLaVA-1.5). AIR's direct manipulation of attention weights is more precise.
- **vs OPERA (ICML24)**: OPERA alleviates hallucination by penalizing excessive attention to summary tokens, but operates only at the token level. AIR addresses both modality-level and token-level imbalance simultaneously.
- **vs AD-HH**: The previous state-of-the-art baseline, which degrades general capability by 14.8%. AIR achieves stronger hallucination mitigation while improving general capability, demonstrating that attention rectification is a more principled direction for intervention.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The attention imbalance concept and MAI/TAI definitions are entirely novel contributions; the causal chain deriving intervention methods from an interpretability perspective is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four LVLMs, three benchmarks, seven baselines, with detailed ablation and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical definitions are rigorous, analysis is progressively structured, and figures are highly informative.
- Value: ⭐⭐⭐⭐⭐ Simultaneously addresses hallucination and general capability degradation; deployable without training, offering exceptional practical value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CountVid: Open-World Object Counting in Videos](../../AAAI2026/medical_imaging/open-world_object_counting_in_videos.md)
- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gender-adversarial_attention_multi.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[NeurIPS 2025\] Sequential Attention-based Sampling for Histopathological Analysis](../../NeurIPS2025/medical_imaging/sequential_attention-based_sampling_for_histopathological_analysis.md)
- [\[ICCV 2025\] InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation](../../ICCV2025/medical_imaging/insideout_integrated_rgb-radiative_gaussian_splatting_for_comprehensive_3d_objec.md)

<!-- RELATED:END -->
