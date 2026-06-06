---
title: >-
  [Paper Note] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification
description: >-
  [NeurIPS 2025][AI Safety][adversarial robustness] This paper systematically evaluates compression-based adversarial purification defenses and demonstrates that the *realism* of reconstructed images is the critical factor…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "adversarial robustness"
  - "image compression"
  - "adversarial purification"
  - "realistic reconstruction"
  - "adaptive attacks"
date: 2026-05-08
content_hash: fa67c53850adfb8b
---

# Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification

**Conference**: NeurIPS 2025
**arXiv**: [2508.05489](https://arxiv.org/abs/2508.05489)  
**Code**: None  
**Area**: AI Safety
**Keywords**: adversarial robustness, image compression, adversarial purification, realistic reconstruction, adaptive attacks

## TL;DR

This paper systematically evaluates compression-based adversarial purification defenses and demonstrates that the *realism* of reconstructed images is the critical factor for robustness—high-realism compression models maintain significant robustness under strong adaptive attacks, and this robustness is not attributable to gradient masking.

## Background & Motivation

**Background**: Adversarial attacks cause classifiers to produce incorrect predictions via imperceptible perturbations. Adversarial purification applies transformations to input images to remove adversarial noise, offering a defense strategy that does not require retraining the classifier.

**Limitations of Prior Work**: Early studies claimed JPEG compression could effectively defend against adversarial attacks, but subsequent work showed that many preprocessing defenses rely on gradient masking and break down under adaptive attacks. Existing evaluations lack comprehensive adaptive attack analysis.

**Key Challenge**: Is the robustness gain of compression-based defenses merely an artifact of gradient masking? If not, what mechanism truly contributes to robustness?

**Goal**: (1) Do compression-based defenses retain robustness under rigorous adaptive attacks? (2) If so, what is the underlying mechanism?

**Key Insight**: Focus on the overlooked dimension of *realism*, comparing the performance of low-realism and high-realism compression models across various attacks.

**Core Idea**: The realism of reconstructed images—rather than simple distortion control—is the key to the effectiveness of compression-based defenses. High-realism reconstruction combats adversarial noise by maintaining distributional consistency and hallucinating semantically plausible details.

## Method

### Overall Architecture

Defense pipeline: input image (possibly containing adversarial perturbations) → compress-decompress (encoder-decoder) → reconstructed image → classifier → class probabilities. The compression step serves as preprocessing, eliminating adversarial perturbations through lossy compression while preserving semantic content.

### Key Designs

1. **Formal Definition of Realism**:

    - Distortion: $\mathcal{D} = \mathbb{E}[\Delta(x, \hat{x})]$, measuring the pointwise distance between the original and reconstructed image
    - Realism: $\mathcal{R} = -d(p_{\hat{X}}, p_X)$, measuring the divergence between the reconstructed image distribution and the natural image distribution
    - Compression model training loss: $\mathcal{L} = \mathcal{L}_{\text{RATE}} + \lambda \mathcal{D} - \beta \mathcal{R}$
    - Key distinction: distortion is a full-reference metric (requires the original image), whereas realism is a no-reference metric (requires only distributional matching)

2. **Threat Model Design**: Three levels of adversary knowledge are defined:

    - **Black-Box (BB)**: unaware of the defense; attacks only the classifier gradients
    - **Gray-Box (GB)**: aware of the defense and can use it in the forward pass, but cannot compute defense gradients
    - **White-Box (WB)**: fully aware of the defense mechanism and can compute complete gradients

3. **Four Adaptive Attack Methods**:

    - **ST BPDA**: straight-through approximation—uses the compression defense in the forward pass and substitutes the identity function in the backward pass. $\nabla_x h \coloneq \nabla_x f(x)|_{x=g(x)}$
    - **U-Net BPDA**: trains a U-Net to approximate the forward behavior of the compression defense and uses U-Net gradients in the backward pass. $\nabla_x h \coloneq \nabla_x (f \circ g')(x)$
    - **ACM (Attack Compression Model)**: directly attacks the compression model with objective $MSE(x, g(x))$, forcing large distortion in reconstruction
    - **ARA (Adaptive Realism Attack)**: for controllable-realism models, uses gradients from versions with different $\beta'$ to attack the target $\beta$ version

4. **Two Mechanisms by Which Realism Enhances Robustness**:

    - Avoids unnatural artifacts, preventing reconstructed images from deviating from the natural image distribution
    - Masks adversarial noise by hallucinating semantically plausible details (e.g., leaf textures)

### Ruling Out Gradient Masking

Gradient masking is verified by increasing the attack budget $\epsilon$: if a model consistently fails at high $\epsilon$, its robustness at low $\epsilon$ is genuine. The paper introduces a "Hyperprior Noise" variant (replacing gradients with random noise) whose performance is similar to the standard version, confirming that Hyperprior's robustness stems from gradient masking, whereas CRDR HR's robustness is authentic.

## Key Experimental Results

### Main Results

Robust accuracy of a ResNet50 classifier on the ImageNet validation set ($\epsilon = 4/255$, strongest adaptive attack):

| Defense Model | Low Realism (LR) | High Realism (HR) |
|---------------|-----------------|------------------|
| Hyperprior / HiFiC | 10.98 | 11.83 |
| MRIC | 26.68 | **39.00** |
| CRDR | 16.30 | **34.50** |
| JPEG | 15.19 | — |
| ELIC | 16.43 | — |

High-realism models substantially outperform their low-realism counterparts across all settings.

### Ablation Study — Ruling Out Gradient Masking (PGD Steps vs. Robust Accuracy, $\epsilon = 4/255$)

| PGD Steps | CRDR LR | CRDR HR |
|-----------|---------|---------|
| 10 | 26.40 | 46.08 |
| 50 | 20.44 | 38.08 |
| 100 | 20.14 | 37.12 |
| 400 | 19.60 | 36.92 |

CRDR HR maintains approximately 37% accuracy under a 400-step attack, with accuracy decreasing monotonically as the number of steps increases (no sign of gradient masking).

### Comprehensive Attack Comparison ($\epsilon = 4/255$, CRDR)

| Attack | CRDR LR | CRDR HR |
|--------|---------|---------|
| BB PGD | 44.92 | 59.80 |
| WB PGD | 16.30 | 35.88 |
| ST BPDA | 39.36 | 56.36 |
| U-Net BPDA | 28.96 | 47.62 |
| ACM | 41.28 | 55.67 |
| ARA | 16.30 | **34.50** |

### Key Findings

- Realism monotonically improves robustness across all attack settings and all values of $\epsilon$
- Distortion exhibits an inherent trade-off (too low retains noise; too high destroys semantics), whereas realism presents no such trade-off
- Hyperprior's robustness originates from gradient masking (noise gradients achieve similar effects)
- WB PGD and U-Net BPDA constitute the most effective attack combination

## Highlights & Insights

- **First systematic investigation of the central role of realism in adversarial robustness**: prior work focused on distortion and compression rate, while realism as a defense factor was entirely overlooked
- **Rigorous evaluation methodology**: four adaptive attacks are designed and gradient masking is carefully ruled out, consistent with best practices in adversarial robustness evaluation
- **Clear intuition**: high-realism reconstruction "pulls" adversarial examples back onto the natural image manifold, conceptually similar to diffusion-based purification but at substantially lower computational cost
- **Challenge to future attack methods**: overcoming high-realism reconstruction is identified as an important open problem in adversarial attack research

## Limitations & Future Work

- Only $l_\infty$-norm untargeted attacks are evaluated; $l_2$ attacks and targeted attacks are not considered
- Standard accuracy of high-realism compression models is reduced (e.g., CRDR HR achieves only 62% standard accuracy, compared to ~80% for vanilla ResNet)
- Evaluation is limited to ImageNet classification; generalization to other vision tasks such as object detection is unexplored
- Combining adversarial training with compression-based defenses is not investigated
- FID is used as a proxy metric for realism; more precise realism measures warrant future study

## Related Work & Insights

- **Shin & Song (2017)**: demonstrated that making JPEG differentiable can fully circumvent its defense; the high-realism models in this paper maintain robustness under analogous settings
- **DiffPure (Nie et al.)**: diffusion-model-based adversarial purification is conceptually similar to high-realism compression but incurs significantly higher computational cost
- **Blau & Michaeli (2019)**: proposed the theoretical framework for the distortion-perception trade-off; this paper validates the importance of realism in the adversarial robustness setting
- Implication: adversarial robustness may require a paradigm shift from "eliminating noise" to "restoring the natural distribution"

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of the relationship between realism and adversarial robustness, with deep insights
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple attack methods, multiple compression models, and careful elimination of gradient masking yield a highly rigorous evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ Argumentation is clear and fluent; the Feynman epigraph is apt, and the experimental design is logically coherent
- Value: ⭐⭐⭐⭐ Significant implications for both the adversarial robustness and image compression communities

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[AAAI 2026\] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models](../../AAAI2026/ai_safety/toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[AAAI 2026\] Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification](../../AAAI2026/ai_safety/breaking_the_adversarial_robustness-performance_trade-off_in_text_classification.md)

</div>

<!-- RELATED:END -->
