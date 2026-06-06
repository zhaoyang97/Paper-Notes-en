---
title: >-
  [Paper Note] Calibrating Uncertainty for Zero-Shot Adversarial CLIP
description: >-
  [ICML 2026][AI Safety][Adversarial Robustness] The UCAT framework is proposed to reparameterize CLIP logits as concentration parameters of a Dirichlet distribution. By aligning the Dirichlet distributions of clean and ad…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Adversarial Robustness"
  - "Uncertainty Calibration"
  - "CLIP"
  - "Dirichlet Distribution"
  - "Zero-Shot Classification"
date: 2026-05-08
content_hash: f82fbc073a996771
---

# Calibrating Uncertainty for Zero-Shot Adversarial CLIP

**Conference**: ICML 2026  
**arXiv**: [2512.12997](https://arxiv.org/abs/2512.12997)  
**Code**: https://github.com/VivienLu/UCAT  
**Area**: AI Safety  
**Keywords**: Adversarial Robustness, Uncertainty Calibration, CLIP, Dirichlet Distribution, Zero-Shot Classification  

## TL;DR

The UCAT framework is proposed to reparameterize CLIP logits as concentration parameters of a Dirichlet distribution. By aligning the Dirichlet distributions of clean and adversarial samples (via reverse KL divergence), it simultaneously calibrates uncertainty and preserves semantic structures during zero-shot adversarial fine-tuning, achieving the optimal balance between robustness and calibration across 16 benchmarks.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) like CLIP achieve powerful zero-shot recognition through contrastive pre-training but are extremely vulnerable to adversarial attacks—small pixel-level perturbations can lead to confident misclassifications. Existing Zero-Shot Adversarial Robustness (ZSAR) methods primarily improve robustness by adversarially fine-tuning the image encoder while attempting to retain zero-shot generalization.

**Limitations of Prior Work**: Mainstream methods adopt a "single-anchor alignment" strategy, pulling adversarial features toward the text embedding of the ground-truth label, but ignore the relative geometric relationships with other category text embeddings. Other methods align softmax distributions, yet softmax normalization discards absolute logit scale information, which is critical for reliability reasoning in open-vocabulary scenarios.

**Key Challenge**: The authors discovered a counter-intuitive phenomenon—adversarial perturbations not only reduce accuracy but also **suppress prediction uncertainty**, causing the model to produce falsely high-confidence predictions when attacked. This violates the fundamental expectation that "uncertainty should increase as the input becomes harder or deviates from the training distribution," exposing a **reliability gap** beyond mere accuracy.

**Goal**: Design an adversarial fine-tuning method that simultaneously optimizes accuracy and uncertainty calibration, enabling the model to maintain robustness and provide well-calibrated confidence estimates under adversarial attacks.

**Key Insight**: The authors reflect that the softmax probability of CLIP zero-shot classification and the expectation of a Dirichlet distribution share a structural correspondence—both are softmax operations on logits. This implies that CLIP logits can be reinterpreted as evidentiary parameters of a Dirichlet distribution.

**Core Idea**: Reparameterize CLIP logits as Dirichlet concentration parameters and use reverse KL divergence to align the Dirichlet distributions of clean and adversarial samples. This maintains inter-class semantic relationships (epistemic uncertainty) while calibrating evidence strength (aleatoric uncertainty).

## Method

### Overall Architecture

UCAT is based on a CLIP adversarial fine-tuning pipeline: the text encoder is frozen, and only the image encoder is trained. Given a clean image $x$ and its corresponding PGD adversarial sample $x^a$, embeddings from the image encoder are used to calculate similarity logits with text prototypes. The core innovation lies in mapping these logits to Dirichlet concentration parameters $\alpha$ and $\alpha_{\text{adv}}$, aligning the two distributions via reverse KL divergence $\text{KL}(\text{Dir}(\alpha_{\text{adv}}) \| \text{Dir}(\alpha))$, and training jointly with a cross-entropy loss.

### Key Designs

1. **Dirichlet Reparameterization**:
    - **Function**: Seamlessly converts CLIP logits into Dirichlet concentration parameters, enabling zero-shot predictions with closed-form uncertainty decomposition.
    - **Mechanism**: Define $\alpha_k(x) = \exp(h(\ell_k^{v \to t}(x)))$, where $h(\ell) = (\tau \ell + 1) / \tau'$. Since cosine similarity $\tau \ell_k \in [-1, 1]$, adding 1 maps it to $[0, 2]$, which is then scaled by a calibration coefficient $\tau'$ and exponentiated to ensure positivity. When $\tau' = \tau$, the Dirichlet expectation is strictly equivalent to the CLIP softmax prediction ($p_k^{\text{Dir}} = p_k^{\text{CLIP}}$), and the argmax remains unchanged for any $\tau' > 0$.
    - **Design Motivation**: This reparameterization ensures $\alpha_k \geq 1$ always holds, avoiding the corner-concentration effect of the Dirichlet distribution when $\alpha_k < 1$ and digamma numerical instability, while providing adjustable sharpness control via $\tau'$.

2. **Uncertainty Calibration Regularization (UCR)**:
    - **Function**: Aligns the Dirichlet distributions of adversarial and clean samples, simultaneously calibrating relative inter-class semantics and total evidence strength.
    - **Mechanism**: Define the regularization loss as the reverse KL divergence $\mathcal{L}_{\text{ucr}} = \text{KL}(\text{Dir}(\alpha_{\text{adv}}) \| \text{Dir}(\alpha))$, with the final objective $\mathcal{L} = \mathcal{L}_{\text{ce}} + \lambda \mathcal{L}_{\text{ucr}}$. Reverse KL is mode-seeking, allowing the model to maintain low evidence on irrelevant categories while precisely tracking the principal mode of the clean distribution.
    - **Design Motivation**: Forward KL covers all modes leading to spread-out evidence, while probability-level KL (softmax alignment) discards absolute logit scale information. Dirichlet-level reverse KL preserves both relative class structure (shape/epistemic uncertainty) and absolute evidence strength (aleatoric uncertainty), proving to be the optimal combination in experiments.

3. **Closed-form Uncertainty Decomposition**:
    - **Function**: Quantifies Aleatoric Uncertainty (AU) and Epistemic Uncertainty (EU) separately in a single forward pass.
    - **Mechanism**: AU is calculated via the expected Shannon entropy of the categorical distribution under the Dirichlet: $\text{AU}(x) = -\sum_k \frac{\alpha_k}{\alpha_0}(\psi(\alpha_k+1) - \psi(\alpha_0+1))$. EU is calculated via the reciprocal of the total evidence: $\text{EU}(x) = C / (\alpha_0 + C)$, where $\alpha_0 = \sum_k \alpha_k$.
    - **Design Motivation**: Traditional methods require multiple forward passes (e.g., MC Dropout) or additional modules to estimate uncertainty. Dirichlet parameterization provides analytical solutions that are computationally efficient and theoretically capable of separating data ambiguity from insufficient evidence.

### Loss & Training

Adversarial samples are generated using $\ell_\infty$ PGD. The default settings are $\tau' = 0.07$ (standard temperature in contrastive learning) and $\lambda = 10^5 / \beta$, where $\beta = 2/e^{\tau'}$. Only the image encoder is fine-tuned, while the text encoder remains completely frozen.

## Key Experimental Results

### Main Results (Zero-Shot Adversarial Robustness on 16 Single-Label Datasets)

| Method | Clean Avg | PGD-100 Avg | CW Avg | AutoAttack Avg | H (Clean-AA) |
|------|-----------|-------------|--------|----------------|---------------|
| CLIP | 64.45 | 3.46 | 4.06 | 0.51 | 1.01 |
| TeCoA | 43.83 | 29.86 | 29.25 | 28.74 | 34.72 |
| FARE | 53.00 | 12.81 | 12.64 | 2.33 | 4.45 |
| PMG-AFT | 53.72 | 31.63 | 22.25 | 17.88 | 26.83 |
| TGA-ZSR | 49.91 | 31.55 | 31.28 | 30.52 | 37.88 |
| Comp-TGA | 52.09 | 31.40 | 31.16 | 26.24 | 34.90 |
| **UCAT** | **54.17** | **32.20** | **31.41** | **30.58** | **39.09** |

UCAT achieves both the highest clean accuracy (54.17%) and the best Clean-AA harmonic mean (39.09), ranking first or second under most attack settings.

### Ablation Study

| Configuration | Clean | PGD-100 | CW | AutoAttack | Description |
|------|-------|---------|-----|------------|------|
| $\mathcal{L}_{\text{ce}}$ (TeCoA Baseline) | 43.83 | 29.86 | 29.25 | 28.74 | Cross-entropy only |
| + KL(p(x)‖p(xᵃ)) | 45.03 | 30.12 | 29.61 | 29.13 | Prob-level forward KL, slight gain |
| + KL(p(xᵃ)‖p(x)) | 45.05 | 29.98 | 29.28 | 28.80 | Prob-level reverse KL, slight gain |
| + KL(Dir(α)‖Dir(αₐdᵥ)) | 36.72 | 25.01 | 24.66 | 24.36 | Dirichlet-level forward KL, performance drop |
| **+ KL(Dir(αₐdᵥ)‖Dir(α))** | **54.17** | **32.20** | **31.41** | **30.58** | **Dirichlet-level reverse KL, significant gain** |

The ablation clearly reveals two key design choices: (1) Dirichlet-level mapping is far superior to probability-level mapping because it preserves absolute evidence strength; (2) Reverse KL is far superior to forward KL as its mode-seeking property is better suited for adversarial scenarios.

### Cross-Backbone Generalization

| Backbone | Method | Clean | AutoAttack | H |
|---------|------|-------|------------|---|
| CLIP-B/16 | Base | 63.72 | 0.01 | 0.02 |
| CLIP-B/16 | +UCAT | 52.91 | 30.54 | 39.05 |
| CLIP-B/32 | Base | 64.42 | 5.58 | 10.28 |
| CLIP-B/32 | +UCAT | 54.17 | 30.58 | 39.09 |
| SLIP-B/16 | Base | 46.03 | 0.02 | 0.04 |
| SLIP-B/16 | +UCAT | 38.37 | 20.40 | 26.68 |

UCAT significantly improves robustness across different contrastively pre-trained VLMs, demonstrating independence from specific CLIP variants.

## Highlights & Insights

1. The finding that **adversarial perturbations suppress uncertainty** is a significant empirical discovery—models become more "confident" when attacked, which is more dangerous than a simple drop in accuracy as users cannot rely on confidence scores to judge validity.
2. The **mathematical equivalence** between CLIP logits and Dirichlet expectations is an elegant theoretical insight, allowing the framework to gain uncertainty estimation capabilities without modifying the CLIP architecture.
3. The **huge performance gap** brought by Dirichlet-level reverse KL compared to probability-level KL (Clean 43→54, AA 29→31) provides strong evidence for the importance of preserving absolute evidence strength.
4. Multi-label MS-COCO experiments show that the method still holds advantages in semantically ambiguous scenarios, validating the design intuition that distribution alignment preserves inter-class relationships.

## Limitations & Future Work

1. Effectiveness is limited on datasets with strong domain shifts (e.g., PCAM, EuroSAT), as CLIP's inherent clean semantic structure is weak in these domains, leaving Dirichlet alignment without a reliable reference distribution.
2. Currently, adversarial fine-tuning is only applied to the image encoder while the text encoder is frozen; the possibility of joint fine-tuning has not been explored.
3. While $\tau' = 0.07$ is stable in most scenarios, the optimal calibration coefficient may vary across different domains.

## Related Work & Insights

- **TeCoA / FARE / TGA-ZSR**: Previous ZSAR methods mainly focused on single-anchor or softmax alignment, overlooking uncertainty calibration.
- **Evidential Deep Learning (Sensoy et al., 2018)**: The source of the Dirichlet reparameterization idea, but original EDL targeted closed-set classification. This work is the first to integrate it with the CLIP contrastive learning framework.
- **TRADES (Zhang et al., 2019)**: A classic robustness-accuracy trade-off framework; UCAT can be viewed as its Dirichlet-level generalization for open-vocabulary VLMs.

## Rating

- Novelty: 8/10 — The theory mapping CLIP logits to Dirichlet evidence is elegant, and the discovery of uncertainty suppression by adversarial attacks is insightful.
- Experimental Thoroughness: 9/10 — Extensive coverage with 16 datasets, multi-label tasks, cross-backbone tests, detailed ablations, and calibration analysis.
- Writing Quality: 8/10 — The theoretical derivation is rigorous and clear, and the charts are highly informative.
- Value: 8/10 — Introduces an uncertainty calibration perspective to VLM adversarial robustness, offering a strong methodological contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference](../../AAAI2026/ai_safety/diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)
- [\[CVPR 2026\] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](../../CVPR2026/ai_safety/mcsd_uncertainty_estimation.md)
- [\[CVPR 2026\] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](../../CVPR2026/ai_safety/tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)
- [\[ICML 2026\] Exposing Vulnerabilities in Explanation for Time Series Classifiers via Dual-Target Adversarial Attack](exposing_vulnerabilities_in_explanation_for_time_series_classifiers_via_dual-tar.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
