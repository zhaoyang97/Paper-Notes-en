---
title: >-
  [Paper Note] Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance
description: >-
  [ICLR 2026][Interpretability][visual invariance] This paper proposes the Stretch-and-Squeeze (SnS) algorithm, a gradient-free, model-agnostic bi-objective optimization framework that systematically probes the invariance…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "visual invariance"
  - "gradient-free optimization"
  - "adversarial examples"
  - "feature visualization"
  - "CNN interpretability"
  - "robust models"
date: 2026-05-08
content_hash: 0c47646e2ca1207e
---

# Stretching Beyond the Obvious: A Gradient-Free Framework to Unveil the Hidden Landscape of Visual Invariance

**Conference**: ICLR 2026
**arXiv**: [2506.17040](https://arxiv.org/abs/2506.17040)  
**Code**: [GitHub](https://github.com/zoccolan-lab/SnS)  
**Area**: Interpretability
**Keywords**: visual invariance, gradient-free optimization, adversarial examples, feature visualization, CNN interpretability, robust models

## TL;DR

This paper proposes the Stretch-and-Squeeze (SnS) algorithm, a gradient-free, model-agnostic bi-objective optimization framework that systematically probes the invariance manifold of visual systems by "stretching" representations at different processing levels while "squeezing" the activation of target units. SnS reveals hierarchical differences in invariance interpretability between standard and robust CNNs.

## Background & Motivation

Understanding which feature combinations are encoded by visual processing units—whether biological neurons or CNN units—is a central problem in visual science and deep learning. Limitations of existing methods:

**Most Exciting Images (MEI)** methods identify only a small number of strongly activating stimuli and cannot reveal the transformations under which a unit remains invariant—yet invariance is critical for generalization.

**Metamers** (representation matching) strictly match representations at a given layer and explore only a narrow neighborhood around the target image.

**Predefined transformation tests** (rotation, translation, scaling, etc.) cannot discover the invariance axes actually learned by a unit.

4. Gradient-based methods cannot be applied to black-box systems such as biological neurons.

The core innovation of SnS lies in systematically sampling the boundary of the invariance manifold by **maximizing** stimulus distance at one representational layer while **maintaining** the activation of the target unit.

## Method

### Overall Architecture

SnS consists of three components:
1. **Generative model $\psi$**: A pretrained DeepGenerator that maps 4096-dimensional vectors to RGB images.
2. **Target network $\phi$**: The visual system under analysis.
3. **Gradient-free optimizer**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

### Key Designs

**Bi-objective optimization**: Given a reference stimulus $\mathbf{x}_{\text{ref}}$, adversarial objectives are defined at layer $\kappa$ (stretch layer) and layer $\ell$ (squeeze layer):

$$\mathcal{L}_{\text{stretch}}^{\kappa} = -\|\mathbf{a}^{\kappa} - \mathbf{a}_{\text{ref}}^{\kappa}\|_2, \quad \mathcal{L}_{\text{squeeze}}^{\kappa} = +\|\mathbf{a}^{\kappa} - \mathbf{a}_{\text{ref}}^{\kappa}\|_2$$

**Invariance search** is solved via Pareto optimization (Equation 5):

$$\Xi_{\text{inv}} = \arg\min_{\mathbf{x}} \left[\mathcal{L}_{\text{stretch}}^{\kappa}(\Gamma(\mathbf{x}, \phi^{\kappa}), \Gamma(\mathbf{x}^{\star}, \phi^{\kappa})), \; |a_u^{\ell} - a_{\text{ref}}^{\ell}|\right]$$

That is, representational distance from the MEI is maximized at layer $\kappa$, while the target unit's activation is preserved at layer $\ell$.

**Adversarial attack search** reverses the objectives (Equation 6): representational distance at layer $\kappa$ is minimized while activation change at layer $\ell$ is maximized.

**Hierarchical invariance probing**: Varying the choice of stretch layer $\kappa$ reveals invariances at different levels of abstraction:
- $\kappa=0$ (pixel space / low_level): primarily produces brightness and contrast variations.
- $\kappa=$ intermediate convolutional layers (mid_level): primarily produces texture and color variations.
- $\kappa=$ deep convolutional layers (high_level): primarily produces viewpoint and pose variations.

### Loss & Training

CMA-ES is used to solve the bi-objective problem in the Pareto-front sense, ranking solution sets by Pareto dominance. Optimization is performed in the 4096-dimensional latent space of the generative model, regularized by a natural image prior to avoid noise-like images.

## Key Experimental Results

### Main Results

**SnS validation** (77 $L_2$-robust ResNet50 readout units):

| Metric | Adversarial Images | Invariant Images |
|--------|--------------------|-----------------|
| Activation drop (relative to MEI) | 111% ± 7% | 34% ± 11% |
| $L_2$ pixel distance | 72 ± 12 | 271 ± 32 |
| Comparison to affine transforms | — | Significantly surpasses rotation/translation/scaling |

Invariant images found by SnS are more "extreme" than affine transformations (larger pixel distance) while having a smaller effect on target unit activation.

**Distinguishability of hierarchical invariances**:

A PCA + SVM classifier is used to discriminate invariant images produced at different stretch layers:
- Standard ResNet50: near-perfect classification is achieved with only a few dozen principal components.
- Robust ResNet50: 80%+ accuracy is achieved.

### Ablation Study

**Human and observer network interpretability evaluation** (12-AFC classification task):

| Condition | Robust ResNet50 Invariant Images | Standard ResNet50 Invariant Images |
|-----------|----------------------------------|-------------------------------------|
| Pixel-space stretching | Human-recognizable (highest) | Human-unrecognizable (lowest) |
| Mid-level stretching | Human-recognizable (moderate) | Human-recognizable (moderate) |
| Deep-level stretching | Human-recognizable (lowest) | Human-recognizable (highest) |

**Key finding**: The interpretability trends of robust and standard networks are **completely opposite**.

- Robust networks: deeper stretch layers yield lower interpretability.
- Standard networks: deeper stretch layers yield higher interpretability.

### Key Findings

1. **$L_2$ adversarial training fails to increase the interpretability of high-level invariances**: Although MEIs and pixel-level invariant images show high human recognition rates, the interpretability gap for deep-layer invariant images is gradually narrowing.
2. **$L_{\infty}$ robustification behaves differently**: its invariant images maintain stable or even increasing interpretability across all observer networks as depth increases.
3. **ViT invariances exhibit distinct patterns**: mid- and deep-layer invariant images are highly similar and interpretable, consistent with the view that ViTs learn more globally distributed features.
4. **Robustness to representational subsampling**: even when only a sparse subset of intermediate-layer neurons is used (analogous to sparse recording in neuroscience experiments), SnS still produces effective invariant images.
5. **Intrinsic dimensionality of the invariance manifold**: lowest at low levels, highest at mid levels, and intermediate at deep levels—consistent with known trends in CNN representational dimensionality.

## Highlights & Insights

1. **Unified framework**: Invariance search and adversarial attacks are realized within the same bi-objective optimization by swapping the stretch/squeeze directions—a conceptually elegant design.
2. **Gradient-free = truly model-agnostic**: SnS can be directly applied to biological neurons, which is infeasible for other methods.
3. **Beyond metamers**: SnS pushes toward the boundary of the invariance manifold, whereas metamers explore only the vicinity of the target image; the two approaches are complementary.
4. **Hierarchical invariances reveal the nature of visual systems**: The progressive invariance from brightness → texture → pose is fundamentally two sides of the same coin—feature selectivity and feature invariance.
5. **The divergence between $L_2$ and $L_{\infty}$ robustification** provides a new perspective for understanding the nature of adversarial training.

## Limitations & Future Work

1. The method depends on the expressive capacity of the pretrained generative model—the generative prior constrains the explorable image space.
2. The evolutionary algorithm incurs high computational cost, requiring many iterations per unit.
3. Validation is limited to ResNet50/ResNet18/VGG16/ViT; larger-scale models (e.g., DINO, ViT-22B) remain unexplored.
4. Closed-loop validation on real biological neurons has not yet been conducted.
5. The invariant images generated are too rich and complex to be concisely described in text.

## Related Work & Insights

- Shares the evolutionary optimization philosophy with XDREAM (Ponce et al., 2019), but extends it to invariance probing.
- Complements the metamers work of Feather et al. (2023): metamers explore nearby regions, while SnS explores distant regions of the invariance manifold.
- Robustified CNNs align well with human vision at the MEI level but diverge at the level of deep-layer invariances—providing a new optimization target for future alignment methods.
- Insight: invariant images rated as "good" or "bad" by SnS could be used to improve training data, guiding networks toward more human-like invariances.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic gradient-free invariance probing framework with an elegant bi-objective design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers standard/robust CNNs, ViTs, and human experiments with comprehensive analysis.
- **Practical Value**: ⭐⭐⭐⭐ — Directly applicable to computational neuroscience and highly instructive for AI interpretability research.
- **Writing Quality**: ⭐⭐⭐⭐ — Method descriptions are clear and figures are of high quality.
- **Overall Rating**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)
- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)
- [\[ICLR 2026\] How Do Transformers Learn to Associate Tokens: Gradient Leading Terms Bring Mechanistic Understanding](how_do_transformers_learn_to_associate_tokens_gradient_leading_terms_bring_mecha.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)

</div>

<!-- RELATED:END -->
