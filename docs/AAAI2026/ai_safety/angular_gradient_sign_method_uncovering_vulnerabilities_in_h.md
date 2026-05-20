---
title: >-
  [Paper Note] Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks
description: >-
  [AAAI 2026][AI Safety][Adversarial Attack] This paper proposes the Angular Gradient Sign Method (AGSM), which decomposes gradients in hyperbolic space into radial (hierarchical depth) and angular (semantic) components…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Adversarial Attack"
  - "Hyperbolic Networks"
  - "Angular Gradient"
  - "Poincaré Ball"
  - "Cross-modal Retrieval"
date: 2026-05-08
content_hash: 01e77c1350232580
---

# Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks

**Conference**: AAAI 2026
**arXiv**: [2511.12985v2](https://arxiv.org/abs/2511.12985v2)  
**Code**: None  
**Area**: AI Security / Adversarial Attacks / Hyperbolic Space
**Keywords**: Adversarial Attack, Hyperbolic Networks, Angular Gradient, Poincaré Ball, Cross-modal Retrieval

## TL;DR
This paper proposes the Angular Gradient Sign Method (AGSM), which decomposes gradients in hyperbolic space into radial (hierarchical depth) and angular (semantic) components, applying perturbations exclusively along the angular direction to generate adversarial examples. AGSM achieves 5–13% greater accuracy degradation than standard FGSM/PGD on image classification and cross-modal retrieval tasks.

## Background & Motivation
Traditional adversarial attacks (FGSM, PGD, etc.) are designed under Euclidean geometric assumptions; however, hyperbolic networks (e.g., Poincaré ResNet, HyCoCLIP) have achieved notable success in hierarchical data representation. Hyperbolic spaces offer exponential representational capacity and hierarchy-preserving structure, making them well-suited for tree-structured data and taxonomies. Directly applying Euclidean adversarial attacks to hyperbolic networks, however, ignores the geometric properties of curved spaces, resulting in inefficient perturbations that are inconsistent with the underlying manifold structure.

A key observation is that in hyperbolic space, radial displacement alters hierarchical depth (from general to specific), while angular displacement changes fine-grained semantics within the same level. Empirical findings show that radial perturbations have negligible effect on classification accuracy, whereas angular perturbations constitute the primary source of attack effectiveness.

## Core Problem
**How can adversarial attack methods be designed to exploit the geometric properties of hyperbolic space, attacking hyperbolic networks more effectively than geometry-agnostic conventional methods?** Specifically, how can the semantically sensitive angular directions in hyperbolic representations be isolated and targeted with perturbations?

## Method

### Overall Architecture
AGSM proceeds in three steps:
1. A standard FGSM is first applied to generate a temporary perturbed sample, yielding a displacement vector in representation space.
2. This displacement vector is decomposed in the tangent space into radial and angular components.
3. Only the angular component is retained; it is back-propagated to input space, and its sign direction is used to apply the final perturbation.

**Input**: Original sample $\mathbf{x}$, label $y$, perturbation budget $\varepsilon$
**Output**: Adversarial example $\mathbf{x}_{adv}$

### Key Designs

1. **Radial–Angular Decomposition**: Given the original representation $\mathbf{h} = f(\mathbf{x})$ and the FGSM-perturbed representation $\tilde{\mathbf{h}}_{adv} = f(\tilde{\mathbf{x}}_{adv})$, the displacement is computed as $\Delta\mathbf{h} = \tilde{\mathbf{h}}_{adv} - \mathbf{h}$. The radial component is $\mathbf{v}_{rad} = \langle\Delta\mathbf{h}, \mathbf{u}_h\rangle \mathbf{u}_h$ (where $\mathbf{u}_h = \mathbf{h}/\|\mathbf{h}\|_2$ is the radial unit vector), and the angular component is $\mathbf{v}_{ang} = \Delta\mathbf{h} - \mathbf{v}_{rad}$. This decomposition applies to both the Poincaré ball and the Lorentz model.

2. **Back-propagation of Angular Direction**: The gradient of the inner product between the representation and the angular component with respect to the input, $\nabla_\mathbf{x}\langle\mathbf{h}, \mathbf{v}_{ang}\rangle = (\partial\mathbf{h}/\partial\mathbf{x})^\top \mathbf{v}_{ang}$, points in the direction that maximizes angular displacement. The final perturbation is $\mathbf{x}_{adv} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x}\langle\mathbf{h}, \mathbf{v}_{ang}\rangle)$.

3. **PAGD Extension (Projected Angular Gradient Descent)**: AGSM is extended to an iterative attack in which the angular direction is recomputed at each step, followed by projection back onto the $\ell_\infty$ constraint ball. This uses $T=20$ iterations with step size $\alpha = \varepsilon/4$ (classification) or $\varepsilon/10$ (retrieval).

### Loss & Training
AGSM is an inference-time attack method and does not involve training. Adversarial training experiments show that augmenting training with AGSM-perturbed samples yields only limited robustness gains at the cost of clean accuracy.

## Key Experimental Results

| Model/Dataset | Metric | Clean | FGSM | AGSM | PGD | PAGD |
|--------------|--------|-------|------|------|-----|------|
| PRN-32 / CIFAR-10 | Top-1 Acc | 86.21 | 54.19 | 41.56 | 8.05 | 7.77 |
| PRN-32 / CIFAR-100 | Top-1 Acc | 53.44 | 19.67 | 13.93 | 9.24 | 7.86 |
| PRN-32 / TinyImageNet | Top-1 Acc | 30.46 | 8.02 | 5.57 | 5.69 | 5.00 |
| HyCoCLIP ViT-B/16 / COCO T2I | R@5 | 69.30 | 15.90 | 12.60 | 4.50 | 4.00 |
| HyCoCLIP ViT-B/16 / Flickr30K I2T | R@5 | 92.60 | 29.50 | 26.60 | 11.70 | 10.70 |

Note: All results evaluated at $\varepsilon = 8.0/255$ unless otherwise specified.

### Ablation Study
- **Angular vs. Radial**: Radial perturbations have virtually no effect on accuracy (53.44% → 53.44%), while angular perturbations reduce accuracy to 25.56%; FGSM reduces it to 19.67% and AGSM further to 13.93%, confirming that the angular direction is the primary source of adversarial effectiveness.
- **$\ell_2$ Norm Constraint**: AGSM outperforms FGSM under $\ell_2$ constraints as well, indicating that the advantage is independent of norm choice.
- **Adversarial Training**: Training with AGSM-augmented samples improves robustness against AGSM (CIFAR-10: 8.30 → 51.07) but reduces clean accuracy (84.76 → 82.31), suggesting that naive adversarial training is not an ideal defense.
- **Hyperbolic Distance Analysis**: AGSM induces larger geodesic distances in Lorentz space (COCO: 0.3883 → 0.4457), confirming that angular perturbations push representations farther along geodesics.
- **Confidence Degradation**: AGSM causes greater MSP reduction than FGSM (CIFAR-10, $\varepsilon$=8.0: 0.4364 → 0.5597).

## Highlights & Insights
- **Elegant Geometric Insight**: The radial–angular decomposition reveals that semantic information in hyperbolic space is predominantly encoded in the angular direction, an important structural property with broader implications for understanding hyperbolic representation spaces.
- **Simple yet Effective Method**: Adding a single orthogonal projection step on top of FGSM yields significant improvements in attack effectiveness at minimal implementation cost.
- **Cross-task Generality**: The method applies uniformly to both Poincaré ball models (classification) and Lorentz models (cross-modal retrieval) under a unified geometric framework.
- **Null Effect of Radial Perturbations**: This negative finding is itself valuable, demonstrating that the hierarchical structure of hyperbolic networks is robust to radial input perturbations, with vulnerability concentrated in the angular dimension.

## Limitations & Future Work
- **Limited Defense Effectiveness**: Adversarial training with AGSM yields only marginal robustness improvements while degrading clean accuracy; geometrically-aware defense strategies require dedicated design.
- **White-box Attack Assumption**: Full model gradient access is required; applicability in black-box settings remains unverified.
- **Restricted Model Coverage**: Only two types of hyperbolic models (Poincaré ResNet and HyCoCLIP) are evaluated; generalization to other hyperbolic architectures (e.g., HNN, HyboNet, L-CLIP) is unexplored.
- **Low-resolution Datasets**: Classification experiments use 32×32 images (CIFAR, Tiny ImageNet); high-resolution settings are not evaluated.
- **Potential for Geometry-aware Defense Design**: Since the vulnerability is localized in the angular direction, angular regularization or targeted adversarial training may serve as promising directions for robustness enhancement.

## Related Work & Insights

| Method | Mechanism | Key Difference from AGSM |
|--------|-----------|--------------------------|
| **FGSM** | Perturbs along the sign of the loss gradient | Geometry-agnostic; simultaneously introduces radial and angular displacement, with the latter as a byproduct |
| **PGD** | Multi-step iterative FGSM with projection | Equally geometry-agnostic; PAGD improves per-step direction selection via angular decomposition |
| **van Spengler et al. (2025)** | Applies FGM/PGD directly to synthetic hyperbolic embeddings | Focuses solely on input space and synthetic data; does not analyze the radial–angular structure in output space |

The key distinction of AGSM is that it is the first method to exploit the hyperbolic geometric structure at the output representation level to guide adversarial attack design.

### Broader Implications
- The radial–angular decomposition framework generalizes naturally to adversarial robustness research in other non-Euclidean spaces (e.g., spherical representations, SPD matrix manifolds).
- The finding that "angular directions encode semantics" has implications for hyperbolic representation learning itself, suggesting that angular-direction robustness could be explicitly encouraged during training.
- This work extends the adversarial training literature from Euclidean to Riemannian manifolds, potentially motivating new geometry-aware adversarial training frameworks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The radial–angular decomposition is novel in the adversarial attack context, though the method is a direct extension of FGSM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both classification and retrieval tasks across multiple models and datasets, with well-executed ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and figures are informative, though a typo ("Hyerpbolic") appears in the related work section.
- **Value**: ⭐⭐⭐ Valuable to the hyperbolic learning community, but with limited scope (restricted to adversarial attacks on hyperbolic networks).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](../../CVPR2026/ai_safety/computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](../../ICLR2026/ai_safety/atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
