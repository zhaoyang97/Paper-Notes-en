---
title: >-
  [Paper Note] One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models
description: >-
  [ICCV 2025][Multimodal VLM][Universal Adversarial Perturbation] This paper proposes C-PGC, a framework that trains a conditional perturbation generator via malicious contrastive learning to produce a pair of universal im…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Universal Adversarial Perturbation"
  - "VLP Models"
  - "Contrastive Learning"
  - "Cross-modal Attack"
  - "Adversarial Transferability"
date: 2026-05-08
content_hash: bab72e4de71f5608
---

# One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models

**Conference**: ICCV 2025
**arXiv**: [2406.05491](https://arxiv.org/abs/2406.05491)  
**Code**: N/A  
**Area**: Multimodal VLM / Adversarial Attack
**Keywords**: Universal Adversarial Perturbation, VLP Models, Contrastive Learning, Cross-modal Attack, Adversarial Transferability

## TL;DR

This paper proposes C-PGC, a framework that trains a conditional perturbation generator via malicious contrastive learning to produce a pair of universal image-text adversarial perturbations (UAPs), fundamentally disrupting the multimodal alignment of VLP models and achieving strong attack performance across multiple VLP models and downstream tasks in both white-box and black-box settings.

## Background & Motivation

**Background**: Vision-Language Pre-training (VLP) models establish powerful cross-modal alignment through contrastive learning on large-scale image-text data, achieving strong performance on downstream tasks such as image-text retrieval, image captioning, and visual question answering.

**Limitations of Prior Work**: Existing adversarial attack methods against VLP models (e.g., Co-Attack, SGA, TMM) are instance-specific, requiring a separately generated perturbation for each input sample, which incurs substantial computational cost and precludes reuse.

**Key Challenge**: Although universal adversarial perturbations (UAPs) have been studied for unimodal vision models, directly transferring classical UAP methods (e.g., UAP, GAP) to VLP models yields poor results—because they focus solely on the image modality and ignore textual information and cross-modal interactions, failing to effectively undermine the multimodal alignment that underlies VLP model success.

**Goal**:
- How to design a universal image-text perturbation that can attack diverse input samples with only a single perturbation pair?
- How to fundamentally disrupt the multimodal alignment of VLP models rather than attacking a single modality?
- How to improve the black-box transferability of universal adversarial perturbations?

**Key Insight**: The core capability of VLP models derives from cross-modal alignment established by contrastive learning. The authors propose turning this mechanism against itself—using a malicious variant of contrastive learning to train a generator that produces alignment-destroying UAPs.

**Core Idea**: Train a cross-modal conditional generator via malicious contrastive learning to produce universal adversarial perturbations that fundamentally disrupt the multimodal alignment of VLP models.

## Method

### Overall Architecture

The overall pipeline of C-PGC is as follows. A fixed random noise $z_v$ is passed through a cross-modal conditional generator $G_w(\cdot)$ to produce a universal perturbation $\delta_v$ of the same size as the input image. During generation, the generator fuses text embeddings as cross-modal conditioning via a cross-attention mechanism: $\delta_v = G_w(z_v; f_T(\mathbf{t}))$. The resulting perturbation is added to the original image to obtain the adversarial image $v_{adv} = v + \delta_v$. Training is driven by two losses: a unimodal distance loss $\mathcal{L}_{Dis}$ and a multimodal contrastive loss $\mathcal{L}_{CL}$. The text-modality attack is fully symmetric to the image-modality attack.

### Key Designs

1. **Cross-Modal Conditional Perturbation Generator**

    - **Function**: Introduces a cross-attention module into a decoder-based generator to incorporate embeddings from the other modality as auxiliary conditioning.
    - **Mechanism**: Text embeddings $\bm{e}_t$ are injected into the intermediate generator features $\bm{h}_t$ via cross-attention: $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}}) \cdot V$, where $Q = \bm{h}_t W_q$, $K = \bm{e}_t W_k$, $V = \bm{e}_t W_v$.
    - **Design Motivation**: Prior generative universal attacks are limited to a single modality; naively porting them to vision-language settings loses critical cross-modal interaction information. The cross-attention mechanism enables the generator to leverage textual semantics to guide perturbation generation, yielding more targeted multimodal UAPs.

2. **Multimodal Contrastive Loss $\mathcal{L}_{CL}$**

    - **Function**: Applies contrastive learning with maliciously constructed positive and negative pairs to train the generator to produce UAPs that destroy multimodal alignment.
    - **Mechanism**: The adversarial image $v_{adv}$ serves as the anchor; the originally matched text $\mathbf{t}$ is treated as a negative sample (pushed away), while a farthest-selection strategy retrieves the text $\mathbf{t}_{pos}$ whose features are most distant from the original image as the positive sample (pulled closer). The loss is defined as $\mathcal{L}_{CL} = \log \frac{\sum_i \sum_j s(v_i+\delta_v, t_j)}{\sum_i \sum_j s(v_i+\delta_v, t_j) + \sum_i \sum_j s(v_i+\delta_v, t_j')}$, where $s(v,t) = \exp(\text{sim}(f_I(v), f_T(t))/\tau)$.
    - **Design Motivation**: This completely inverts normal contrastive learning—originally matched image-text pairs are pushed apart while mismatched pairs are pulled together, fundamentally destroying alignment. The farthest-selection strategy ensures maximal semantic discrepancy between the positive sample and the original image, further amplifying the destructive effect. Set-level data augmentation (multi-scale resizing and Gaussian noise) is also applied to obtain more robust optimization directions.

3. **Unimodal Distance Loss $\mathcal{L}_{Dis}$**

    - **Function**: Pushes the adversarial image embedding away from the original image embedding in the unimodal feature space.
    - **Mechanism**: Minimizes the negative Euclidean distance between adversarial and original image embeddings: $\mathcal{L}_{Dis} = -\sum_i \sum_j \|f_I(v_i^{adv}) - f_I(v_j)\|_2$.
    - **Design Motivation**: While the multimodal loss handles cross-modal alignment disruption, unimodal information is equally important—pushing adversarial images away from the original visual semantic region provides effective optimization directions and complements the multimodal loss to enhance overall attack performance.

### Loss & Training

The overall training objective is $\min_w \mathbb{E}_{(v,\mathbf{t}) \sim \mathcal{D}_s, \mathbf{t}_{pos} \sim \mathcal{D}_s}(\mathcal{L}_{CL} + \lambda \mathcal{L}_{Dis})$, where $\lambda = 0.1$ balances the two loss terms. Image perturbations are constrained by the $l_{\infty}$ norm: $\|\delta_v\|_{\infty} \leq 12/255$; text perturbations replace at most 1 word ($\epsilon_t = 1$). The generator is trained for 40 epochs using the Adam optimizer with a learning rate of $2^{-4}$ and temperature parameter $\tau = 0.1$.

## Key Experimental Results

### Main Results: Attack Success Rate on Image-Text Retrieval (Flickr30K)

| Surrogate | Target | Method | TR (%) | IR (%) |
|-----------|--------|--------|--------|--------|
| ALBEF | ALBEF (white-box) | C-PGC | 90.13 | 88.82 |
| ALBEF | ALBEF (white-box) | ETU | 78.01 | 84.56 |
| ALBEF | ALBEF (white-box) | GAP | 69.78 | 81.59 |
| ALBEF | TCL (black-box) | C-PGC | 62.11 | 64.48 |
| ALBEF | TCL (black-box) | ETU | 29.92 | 35.91 |
| ALBEF | CLIP_CNN (black-box) | C-PGC | 54.40 | 72.51 |
| ALBEF | CLIP_CNN (black-box) | ETU | 33.55 | 47.69 |
| BLIP | BLIP (white-box) | C-PGC | 71.82 | 82.82 |
| BLIP | BLIP (white-box) | ETU | 59.52 | 77.82 |

In the white-box setting, C-PGC achieves an average ASR of nearly 90%; in the black-box setting, it outperforms ETU by an average of 17.76%.

### Ablation Study

| Configuration | ALBEF W-B TR | ALBEF W-B IR | TCL B-B TR | TCL B-B IR |
|---------------|-------------|-------------|-----------|-----------|
| C-PGC (full) | 90.13 | 88.82 | 62.11 | 64.48 |
| w/o $\mathcal{L}_{CL}$ (C-PGC_CL) | 76.46 | 77.58 | 34.99 | 47.55 |
| w/o $\mathcal{L}_{Dis}$ (C-PGC_Dis) | 79.54 | 82.46 | 56.52 | 62.21 |
| Random positives (C-PGC_Rand) | 61.87 | 65.17 | 43.69 | 52.54 |
| w/o cross-attention (C-PGC_CA) | 85.18 | 83.07 | 45.76 | 53.73 |

### Key Findings

- **$\mathcal{L}_{CL}$ contributes most**: Removing it causes a 27.12% drop in black-box TR ASR (ALBEF→TCL), confirming that contrastive learning is the core mechanism for destroying multimodal alignment.
- **Farthest-selection strategy is critical**: Random positive selection not only degrades performance but can even cause $\mathcal{L}_{CL}$ to harm white-box performance (cf. C-PGC_CL vs. C-PGC_Rand).
- **Cross-attention significantly improves transferability**: Removing CA leads to a more pronounced drop in black-box attack performance, indicating that cross-modal conditioning primarily enhances adversarial transferability.
- C-PGC also performs strongly on additional downstream tasks (visual grounding, image captioning, visual entailment), e.g., degrading BLIP's captioning quality (B@4 from 39.7 to 21.2).

## Highlights & Insights

- **Malicious contrastive learning paradigm**: Repurposing the core training mechanism of VLP models to attack them is an elegant idea. Contrastive learning is the "sword of alignment"; this paper turns it into the "sword of destruction."
- **Truly multimodal universal attack**: Simultaneously generating UAPs for both image and text modalities distinguishes C-PGC from ETU, which is limited to the image modality. The generator architecture and training losses are fully symmetric across both modalities.
- **Generalizable attack philosophy**: The strategy of exploiting a model's own training paradigm to attack it is transferable to adversarial settings targeting other pre-trained models (e.g., MAE, DINO).

## Limitations & Future Work

- Text perturbations are restricted to replacing a single word; while this ensures imperceptibility, it may limit attack efficacy. More flexible text perturbation strategies warrant exploration.
- Training the generator requires access to a surrogate model and a moderate amount of training data (30,000 images), which remains a limitation in fully black-box scenarios.
- The effect of defense methods (e.g., adversarial training, input purification) on C-PGC is not thoroughly investigated, although the appendix contains partial discussion.
- The fixed noise input means the generated UAP is a single static perturbation that cannot adapt to the characteristics of individual inputs.

## Related Work & Insights

- **vs. Co-Attack/SGA/TMM**: These methods are instance-specific, requiring individual perturbation generation per sample; C-PGC requires only a single perturbation pair to attack all samples, offering substantially greater efficiency.
- **vs. ETU**: A concurrent work, but ETU focuses solely on image UAPs and employs a non-generative approach, resulting in limited black-box transferability; C-PGC outperforms it by an average of 17.76% in black-box settings.
- **vs. GAP**: A classical generative UAP method, but one designed for unimodal classification models; C-PGC extends the paradigm to multimodal settings by introducing cross-modal conditioning and malicious contrastive learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of malicious contrastive learning and a cross-modal conditional generator is creative, though it remains within the established adversarial attack framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 VLP models × 4 downstream tasks, with comprehensive ablations and detailed hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Logically coherent, with well-motivated derivations and rigorous mathematical presentation.
- **Value**: ⭐⭐⭐⭐ Exposes the vulnerability of VLP models to universal adversarial attacks, providing important reference for security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ICCV 2025\] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks](safeguarding_vision-language_models_mitigating_vulnerabilities_to_gaussian_noise.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[ICCV 2025\] MMOne: Representing Multiple Modalities in One Scene](mmone_representing_multiple_modalities_in_one_scene.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)

</div>

<!-- RELATED:END -->
