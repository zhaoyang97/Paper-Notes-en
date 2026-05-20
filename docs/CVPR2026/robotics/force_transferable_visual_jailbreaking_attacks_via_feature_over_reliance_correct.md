---
title: >-
  [Paper Note] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction
description: >-
  [CVPR 2026][Robotics][visual jailbreaking] This work identifies the root cause of poor transferability in visual jailbreak attacks as their residence in high-sharpness loss regions — arising from shallow-layer over-relia…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "visual jailbreaking"
  - "adversarial attacks"
  - "transferability"
  - "loss landscape"
  - "MLLM safety"
  - "red-teaming"
date: 2026-05-08
content_hash: 7ee119449d9ade60
---

# FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction

**Conference**: CVPR 2026
**arXiv**: [2509.21029](https://arxiv.org/abs/2509.21029)  
**Code**: [tmllab/2026_CVPR_FORCE](https://github.com/tmllab/2026_CVPR_FORCE)  
**Area**: Robotics
**Keywords**: visual jailbreaking, adversarial attacks, transferability, loss landscape, MLLM safety, red-teaming

## TL;DR

This work identifies the root cause of poor transferability in visual jailbreak attacks as their residence in high-sharpness loss regions — arising from shallow-layer over-reliance on model-specific representations and excessive influence of high-frequency information. FORCE is proposed to address this via layer-aware regularization that broadens the shallow-layer feasible region, and spectral rescaling that suppresses high-frequency non-semantic components, guiding attacks into flatter loss landscapes and substantially improving cross-model transferability.

## Background & Motivation

Multimodal large language models (MLLMs) introduce additional security vulnerabilities when integrating visual and other new modalities. The current landscape of visual jailbreak attacks is as follows:

1. **Text jailbreaks are increasingly constrained**: As the intensity of text alignment (RLHF, DPO) grows, purely text-based attacks against MLLMs continue to decline in effectiveness.
2. **Visual jailbreaks are effective but localized**: Optimization-based visual attacks (PGD and its variants) can reliably bypass the safety mechanisms of open-source MLLMs with imperceptible perturbations, achieving near 100% success rates on source models.
3. **Transferability is extremely poor**: Visual attacks generated on a source model almost never transfer to other MLLMs, especially closed-source commercial models — 93% of attacks fail even after exhausting 100 queries.

Core motivation: To conduct meaningful red-teaming evaluations of security vulnerabilities in closed-source MLLMs, the **cross-model transferability** of optimization-based visual jailbreak attacks must be improved. As one of the first works to study this problem, this paper conducts an in-depth analysis of transfer failure from the geometric perspective of loss landscapes.

## Core Problem

- Why do visual jailbreak attacks succeed on source models but fail to transfer?
- How does the shape of the attack's loss landscape affect transferability?
- Which factors in feature representations lead to model-specific dependencies?

## Method

### Diagnostic Analysis: Loss Landscape

Using LLaVA-v1.5-7B as the source model with standard PGD (step size 2/255, budget 32/255, target "Sure, here is"):

**Input loss landscape**: Pixel perturbations are introduced along the gradient direction and a random direction. The attack is found to reside in an extremely sharp local optimum — a perturbation of merely 0.03 pixels raises the loss from ~0 to above 0.28, rendering the attack ineffective.

**Weight loss landscape**: Small parameter changes simulating model transfer are applied. A weight perturbation of just 0.0002 is sufficient to push the attack out of the feasible region.

**Conclusion**: The attack is trapped in a high-sharpness region, exhibiting extreme sensitivity to any minor perturbation — this is the direct manifestation of transfer failure.

### Diagnostic Analysis: Feature Representations Across Layers

Features of jailbreak samples and natural images are extracted layer by layer, and interpolated via convex combination $(1-\mu) \cdot f_{\theta}(\text{jail}) + \mu \cdot f_{\theta}(\text{nat})$:

- **Deep layers (e.g., layer 31)**: The attack remains effective even when 40% of natural features are injected — the feasible region is broad.
- **Shallow layers (e.g., layer 11)**: The attack must retain 90%+ of adversarial features to succeed; injecting only 30% natural features causes the loss to surge sharply above 1.2 — the feasible region is extremely narrow.

**Conclusion**: Shallow layers over-rely on model-specific features → narrow and fragile feasible region → attack trapped in high-sharpness zone.

### Diagnostic Analysis: Frequency Dependency in the Spectral Domain

A Fourier transform is applied to the perturbation $\delta$, divided into $M=10$ equal-width frequency bands; each band is masked in turn and attack degradation is evaluated:

- **Early iterations (150–250)**: Low-frequency components (semantically rich) are critical for attack success; removing them causes failure — consistent with the prior that natural image semantics reside in low frequencies.
- **Later iterations (350–750)**: High-frequency influence progressively surpasses low-frequency; by iteration 750, removing the third-highest frequency band alone suffices to defeat the attack.

**Conclusion**: The optimization process drives the attack toward increasing reliance on high-frequency (semantically weak) shortcuts → attack becomes model-specific → fails to generalize.

### FORCE Method

Based on the above analysis, two complementary components are proposed:

**Component 1: Layer-Aware Regularization**

$N=10$ reference points are sampled within a neighborhood $\eta=4/255$ of the jailbreak sample. For each layer $l$, the $L_2$ distance $d_l$ between the features of the jailbreak sample and those of the reference points is maximized, while constraining reference points to also reside in the feasible region (i.e., their loss must also be low):

$$d_l = \|f_{\theta,l}(\mathbf{x}_{\text{img}}+\delta, \mathbf{x}_{\text{txt}}) - f_{\theta,l}(\mathbf{x}_{\text{img}}+\delta+\eta, \mathbf{x}_{\text{txt}})\|_2^2$$

$$\ell_{\text{ref}} = \ell(p_\theta(\mathbf{x}_{\text{img}}+\delta+\eta, \mathbf{x}_{\text{txt}}), \mathbf{y})$$

Key design — **gradually decreasing regularization strength**, stronger in shallow layers and weaker in deep layers, consistent with diagnostic findings:

$$\lambda_l = \lambda \cdot \max(1 - (2l/L)^2, 0)$$

Regularization loss: $\ell_{\text{reg}} = \frac{1}{N}\sum_{n=1}^{N}\sum_{l=1}^{L} \lambda_l \cdot \frac{\ell_{\text{ref}}}{d_l}$

**Component 2: Spectral Rescaling**

An FFT is applied to the perturbation $\delta$, divided into $M=10$ equal-width frequency bands. When the influence of band $m$ exceeds $\beta$ times that of the adjacent lower-frequency band, it is rescaled:

$$w_m = \min\left(\beta, \frac{\ell_{m-1}}{\ell_m} \cdot \beta\right)$$

$$S = \sum_{m=1}^{M} (w_m \cdot \mathbb{1}_{B_m})$$

The frequency rescaling matrix $S$ is multiplied with the FFT amplitude spectrum, and the perturbation is reconstructed via IFFT: $\delta_{\text{rescaled}} = \text{IFFT}((A \odot S) \odot e^{i\Phi})$.

Both components are integrated into the standard PGD pipeline: spectral rescaling is applied first, followed by layer-aware regularization.

## Key Experimental Results

### Transfer Across Adapter-Based MLLMs (Source: LLaVA-v1.5-7B, Multiple Queries)

| Target Model | Method | MaliciousInstruct ASR | AdvBench ASR | HADES ASR |
|---|---|---|---|---|
| LLaVA-v1.6-mistral | PGD | 61.0 | 35.2 | 70.0 |
| | FORCE | **69.0** (+12.3%) | **43.8** (+24.6%) | **72.7** (+3.8%) |
| InstructBLIP-Vicuna | PGD | 84.0 | 25.6 | 48.7 |
| | FORCE | **92.0** (+9.5%) | **27.9** (+9.0%) | **49.2** (+1.1%) |
| Idefics3-8B | PGD | 53.0 | 29.8 | 63.1 |
| | FORCE | **64.0** (+20.8%) | **36.0** (+20.6%) | **66.0** (+4.6%) |

### Transfer Across Early-Fusion MLLMs

| Target Model | Method | MaliciousInstruct ASR | AdvBench ASR | HADES ASR |
|---|---|---|---|---|
| LLaMA-3.2-11B | PGD | 1.0 | 1.2 | 6.3 |
| | FORCE | **2.0** (+100%) | **2.3** (+101%) | **10.3** (+63.6%) |
| Qwen2.5-VL-7B | PGD | 5.0 | 1.5 | 25.3 |
| | FORCE | **11.0** (+120%) | **2.7** (+74.7%) | **28.1** (+11.1%) |

### Transfer Across Commercial MLLMs

| Target Model | Method | MaliciousInstruct ASR | HADES ASR |
|---|---|---|---|
| Claude-Sonnet-4 | PGD | 1.0 | 3.0 |
| | FORCE | **2.0** (+100%) | **5.0** (+66.7%) |
| GPT-5 | PGD | 1.0 | 1.0 |
| | FORCE | **2.0** (+100%) | **3.0** (+200%) |

### Ablation Study (Idefics3, MaliciousInstruct)

| Layer Reg | Spectral Rescaling | ASR | Query ↓ |
|---|---|---|---|
| ✗ | ✗ | 53.0 | 50.7 |
| ✓ | ✗ | 55.0 (+3.8%) | 48.5 (+4.7%) |
| ✗ | ✓ | 59.0 (+11.3%) | 44.0 (+15.2%) |
| ✓ | ✓ | **64.0** (+20.6%) | **39.6** (+28.1%) |

The two components exhibit synergistic effects, with their combined performance exceeding the sum of individual contributions.

### Computational Overhead

| Method | Time (s/iter) | Memory (GB) |
|---|---|---|
| PGD | 2.17 | 32.64 |
| FORCE | 2.73 (+26%) | 36.48 (+12%) |

The additional overhead is minimal, as the regularization terms leverage intermediate variables from the standard forward pass, and multi-sample computation can be parallelized.

## Highlights & Insights

- **In-depth and compelling analysis**: The root cause of poor transferability is progressively revealed across three dimensions — loss landscape geometry, layer-wise feature space, and spectral domain — forming a complete analytical chain.
- **Method design closely tied to analysis**: Each component directly corresponds to a specific diagnostic finding (narrow feasible region in shallow layers → layer-aware regularization; excessive high-frequency dependence → spectral rescaling), with no unmotivated components.
- **Elegant decaying regularization design**: Stronger regularization in shallow layers and weaker in deep layers, consistent with the diagnosis that the problem is concentrated in shallow layers, implemented via a concise quadratic decay formula.
- **Broad evaluation coverage**: Three architecture types (adapter-based, early-fusion, commercial) × three datasets × multiple settings (multi-query / zero-shot / blank initialization).

## Limitations & Future Work

- Absolute ASR on early-fusion MLLMs and commercial models remains low (e.g., only 2–3% on GPT-5); while relative gains are substantial, the gap to practical red-teaming utility persists.
- Validation is limited to LLaVA-v1.5-7B and InstructBLIP as source models, offering limited source model diversity.
- Sensitivity analysis of hyperparameters — specifically the number of frequency bands $M=10$ and the rescaling factor $\beta=0.95$ in spectral rescaling — is insufficient.
- The method assumes white-box gradient access to the source MLLM and is not applicable to purely black-box settings.
- Robustness against image preprocessing defenses (e.g., JPEG compression, image resizing) is not explored — only random noise is tested.
- Early-fusion MLLMs use tokenized image representations; perturbations in pixel space cover only a subset of vulnerabilities in token space, representing a fundamental methodological limitation.

## Related Work & Insights

| Dimension | Standard PGD | Ensemble Opt | MI-FGSM/DI-FGSM | FORCE |
|---|---|---|---|---|
| Strategy | Direct gradient optimization | Multi-model joint optimization | Momentum / input diversity | Feature reliance correction |
| Requires multiple models | ✗ | ✓ | ✗ | ✗ |
| Theoretical basis | None | None | Classification transferability | Loss landscape analysis |
| Idefics3 ASR | 53.0 | 60.0 | 62–66 | **64.0** |
| Qwen2.5-VL ASR | 5.0 | 4.0 | 3–9 | **11.0** |

FORCE requires no additional models and achieves transferability comparable to or better than ensemble-based methods by correcting feature dependencies on a single source model alone. Compared to transferability-enhancement methods developed for the classification domain, FORCE is specifically designed for the characteristics of jailbreak attacks and yields superior results.

The relationship between loss landscape flatness and transferability has been studied in the classification adversarial attack literature (e.g., SAM); this paper is among the first to bring this perspective to visual jailbreaking, revealing a consistent underlying nature of the problem. The finding that shallow layers exhibit over-reliance on model-specific features resonates with the feature visualization and probing literature — inter-model differences in shallow-layer representations are greater than in deep layers. The spectral rescaling approach (suppressing high-frequency shortcuts during optimization) is generalizable to other adversarial robustness research. Although the method is framed as an attack, it also offers guidance for defense — strengthening cross-model consistency of shallow-layer features may improve the robustness of safety alignment. This work belongs to the AI safety red-teaming direction, exposing security vulnerabilities in the visual modality of MLLMs and offering direct value to the community's security evaluation practices.

## Rating

- **Novelty**: 8/10 — Among the first to analyze visual jailbreak transferability through the lens of loss landscapes and feature reliance; both the analytical depth and method design are innovative.
- **Experimental Thoroughness**: 8/10 — Covers multiple architectures, datasets, and settings with clear ablations; source model diversity and hyperparameter sensitivity analysis could be further strengthened.
- **Writing Quality**: 9/10 — The logical chain from analysis to motivation to method is exceptionally clear, with intuitive figures and strong readability.
- **Value**: 7/10 — Identifies an important problem and provides effective mitigation, but the low absolute attack success rates limit practical red-teaming utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation](forcevla2_unleashing_hybrid_force-position_control_with_force_awareness_for_cont.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](../../NeurIPS2025/robotics/dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [\[CVPR 2026\] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning](deepsketcher_internalizing_visual_manipulation_for_multimodal_reasoning.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)

</div>

<!-- RELATED:END -->
