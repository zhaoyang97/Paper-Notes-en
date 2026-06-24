---
title: >-
  [Paper Note] Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks
description: >-
  [AAAI 2026][AI Safety][Adversarial Attacks] Proposes the Angular Gradient Sign Method (AGSM), which decomposes gradients in hyperbolic space into radial (hierarchical depth) and angular (semantic) components. By applying perturbations strictly along the angular direction to generate adversarial examples, it reduces accuracy by an additional 5-13% compared to standard FGSM/PGD on image classification and cross-modal retrieval tasks.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Adversarial Attacks"
  - "Hyperbolic Networks"
  - "Angular Gradient"
  - "Poincaré Ball"
  - "Cross-Modal Retrieval"
date: 2026-05-08
content_hash: 8d426b346a66f38b
---

# Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks

**Conference**: AAAI 2026  
**arXiv**: [2511.12985v2](https://arxiv.org/abs/2511.12985v2)  
**Code**: None  
**Area**: AI Safety / Adversarial Attacks / Hyperbolic Space  
**Keywords**: Adversarial Attacks, Hyperbolic Networks, Angular Gradient, Poincaré Ball, Cross-Modal Retrieval  

## TL;DR
Proposes the Angular Gradient Sign Method (AGSM), which decomposes gradients in hyperbolic space into radial (hierarchical depth) and angular (semantic) components. By applying perturbations strictly along the angular direction to generate adversarial examples, it reduces accuracy by an additional 5-13% compared to standard FGSM/PGD on image classification and cross-modal retrieval tasks.

## Background & Motivation
Traditional adversarial attacks (such as FGSM, PGD, etc.) are designed based on Euclidean geometry assumptions. However, hyperbolic networks (e.g., Poincaré ResNet, HyCoCLIP) have recently achieved significant success in hierarchical data representation. Hyperbolic space possesses exponential representation capacity and hierarchy-preserving structure, making it suitable for tree-structured or taxonomical data. Nevertheless, directly applying adversarial attacks from Euclidean space to hyperbolic networks ignores the geometric properties of manifold space, leading to inefficient perturbations that are inconsistent with the manifold structure.

Key observation: In hyperbolic space, radial displacement alters the hierarchical depth (from general to specific), whereas angular displacement changes the fine-grained semantics within the same level. Experiments reveal that radial perturbations have almost no impact on classification accuracy, whereas angular perturbations are the primary driver of attack effectiveness.

## Core Problem
**How can we design adversarial attack methods that leverage the geometric properties of hyperbolic space to attack hyperbolic networks more effectively than geometry-agnostic traditional attacks?** Specifically, how can we isolate the semantically sensitive angular direction in hyperbolic representations to apply targeted perturbations?

## Method

### Overall Architecture
The process of AGSM consists of three steps:
1. Use standard FGSM to generate a temporary perturbed sample and obtain the displacement vector in the representation space.
2. Decompose this displacement vector in the tangent space into radial and angular components.
3. Keep only the angular component, backpropagate it to the input space, and apply the final perturbation along its sign direction.

Input: Original sample $\mathbf{x}$, label $y$, perturbation budget $\varepsilon$  
Output: Adversarial example $\mathbf{x}_{adv}$

### Key Designs

1. **Radial-Angular Decomposition**: Given the original representation $\mathbf{h} = f(\mathbf{x})$ and the representation after FGSM perturbation $\tilde{\mathbf{h}}_{adv} = f(\tilde{\mathbf{x}}_{adv})$, calculate the displacement $\Delta\mathbf{h} = \tilde{\mathbf{h}}_{adv} - \mathbf{h}$. The radial component is $\mathbf{v}_{rad} = \langle\Delta\mathbf{h}, \mathbf{u}_h\rangle \mathbf{u}_h$ (where $\mathbf{u}_h = \mathbf{h}/\|\mathbf{h}\|_2$ is the radial unit vector), and the angular component is $\mathbf{v}_{ang} = \Delta\mathbf{h} - \mathbf{v}_{rad}$. This decomposition applies to both the Poincaré ball and the Lorentz model.

2. **Backpropagation of Angular Direction**: Compute the gradient of the inner product between the representation and the angular component $\nabla_\mathbf{x}\langle\mathbf{h}, \mathbf{v}_{ang}\rangle = (\partial\mathbf{h}/\partial\mathbf{x})^\top \mathbf{v}_{ang}$, which points in the direction that maximizes angular displacement. The final perturbation is $\mathbf{x}_{adv} = \mathbf{x} + \varepsilon \cdot \text{sign}(\nabla_\mathbf{x}\langle\mathbf{h}, \mathbf{v}_{ang}\rangle)$.

3. **PAGD Extension (Projected Angular Gradient Descent)**: Extend AGSM to an iterative attack, where the angular direction is recomputed at each step, and the sample is updated along this direction before being projected back into the $\ell_\infty$ constraint ball. Iterations $T=20$, step size $\alpha = \varepsilon/4$ (classification) or $\varepsilon/10$ (retrieval).

### Loss & Training
AGSM is an inference-time attack method and does not involve training. Adversarial training experiments show that training with AGSM-perturbed samples yields only limited robustness improvements and incurs a loss in clean accuracy.

## Key Experimental Results

| Model / Dataset | Metric | Clean | FGSM | AGSM | PGD | PAGD |
|------------|------|-------|------|------|-----|------|
| PRN-32 / CIFAR-10 | Top-1 Acc | 86.21 | 54.19 | 41.56 | 8.05 | 7.77 |
| PRN-32 / CIFAR-100 | Top-1 Acc | 53.44 | 19.67 | 13.93 | 9.24 | 7.86 |
| PRN-32 / TinyImageNet | Top-1 Acc | 30.46 | 8.02 | 5.57 | 5.69 | 5.00 |
| HyCoCLIP ViT-B/16 / COCO T2I | R@5 | 69.30 | 15.90 | 12.60 | 4.50 | 4.00 |
| HyCoCLIP ViT-B/16 / Flickr30K I2T | R@5 | 92.60 | 29.50 | 26.60 | 11.70 | 10.70 |

Note: All tested under $\varepsilon = 8.0/255$ (unless specified otherwise).

### Ablation Study
- **Angular vs. Radial**: Radial perturbation barely affects accuracy ($53.44\% \rightarrow 53.44\%$), while angular perturbation reduces it to $25.56\%$, FGSM reduces it to $19.67\%$, and AGSM further drops it to $13.93\%$. This demonstrates that the angular direction is the primary source of adversarial effectiveness.
- **$\ell_2$ Norm Constraint**: AGSM also outperforms FGSM under $\ell_2$ constraints, indicating that the attack's effectiveness is independent of the norm choice.
- **Adversarial Training**: Training augmented with AGSM can improve robustness against AGSM attacks (CIFAR-10: $8.30 \rightarrow 51.07$), but clean accuracy decreases ($84.76 \rightarrow 82.31$), indicating that direct adversarial training is not an ideal defense.
- **Hyperbolic Distance Analysis**: AGSM yields larger geodesic distances in the Lorentz space (COCO: $0.3883 \rightarrow 0.4457$), confirming that angular perturbations push representations further along geodesics.
- **Confidence Drop**: AGSM leads to a more severe drop in Mean Softmax Probability (MSP) than FGSM (CIFAR-10 $\varepsilon=8.0$: $0.4364 \rightarrow 0.5597$).

## Highlights & Insights
- **Simple yet Elegant Geometric Insight**: The radial-angular decomposition reveals the crucial property that semantic information is primarily encoded in the angular direction within hyperbolic spaces. This is highly enlightening for understanding the structure of hyperbolic representation spaces.
- **Simple yet Effective Method**: Simply adding an orthogonal projection step on top of FGSM significantly boosts the attack's effectiveness, incurring negligible implementation overhead.
- **Generalizability Across Tasks**: The method is applicable to both the Poincaré ball model (classification) and the Lorentz model (cross-modal retrieval), offering a unified geometric framework.
- **Discovery of Ineffective Radial Perturbations**: This negative result is highly valuable in itself, indicating that the hierarchical structure of hyperbolic networks is robust to input perturbations, and vulnerabilities are concentrated in the angular dimension.

## Limitations & Future Work
- **Limited Defense Effectiveness**: Using AGSM for adversarial training yields only limited improvements and degrades clean accuracy, highlighting the need for dedicated, geometry-aware defense strategies.
- **White-Box Attack Assumption**: Requires full access to model gradients; its applicability in black-box scenarios remains unverified.
- **Tested on Limited Hyperbolic Models**: Tested only on Poincaré ResNet and HyCoCLIP; generalization to other hyperbolic architectures (e.g., HNN, HyboNet, L-CLIP) has not yet been explored.
- **Low-Resolution Datasets**: Classification experiments use $32 \times 32$ images (CIFAR, Tiny ImageNet); high-resolution scenarios have not been evaluated.
- **Scalability to Hyperbolic Defense Design**: Knowing that the vulnerability lies in the angular direction, researchers could design regularization or adversarial training mechanisms focused on the angular direction to enhance robustness.

## Related Work

| Method | Core Idea | Key Difference from AGSM |
|------|---------|-----------------|
| **FGSM** | Perturb along the sign direction of loss gradient | Geometry-agnostic; introduces both radial and angular displacements, where the latter is a "side effect" |
| **PGD** | Multi-step iterative FGSM + projection | Similarly geometry-agnostic; PAGD improves direction selection at each step through angular decomposition |
| **van Spengler et al. (2025)** | Directly apply FGM/PGD on synthetic hyperbolic embeddings | Only focuses on input space and synthetic data, without analyzing the radial-angular structure of the output space |

The core difference of AGSM is that it is the first method to leverage the hyperbolic geometric structure at the output level of the representation space to design attacks.

## Related Work & Insights
- The idea of radial-angular decomposition can be extended to adversarial robustness research in other non-Euclidean spaces (e.g., spherical representations, SPD matrix manifolds, etc.).
- The discovery that "the angular direction encodes semantics" offers insights for hyperbolic representation learning itself: one could consider enhancing the robustness of the angular direction during training.
- Intersection with adversarial training literature: Expanding the domain from Euclidean to Riemannian manifolds may spawn new geometry-aware adversarial training frameworks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of radial-angular decomposition is novel in adversarial attacks, although the method itself is a direct improvement over FGSM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ It covers both classification and retrieval tasks across multiple models and datasets, with comprehensive ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations and illustrative figures, although there is a typo ("Hyerpbolic") in the related work section.
- **Value**: ⭐⭐⭐ Valuable for the hyperbolic learning community, but the scope of application is relatively narrow (limited to adversarial attacks on hyperbolic networks).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[ICLR 2026\] TriQDef: Disrupting Semantic and Gradient Alignment to Block Adversarial Patch Transfer in Quantized Networks](../../ICLR2026/ai_safety/triqdef_disrupting_semantic_and_gradient_alignment_to_prevent_adversarial_patch_.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] POUR: A Provably Optimal Method for Unlearning Representations via Neural Collapse](../../CVPR2026/ai_safety/pour_a_provably_optimal_method_for_unlearning_representations_via_neural_collaps.md)
- [\[CVPR 2026\] Roots Beneath the Cut: Uncovering the Risk of Concept Revival in Pruning-Based Unlearning for Diffusion Models](../../CVPR2026/ai_safety/roots_beneath_the_cut_uncovering_the_risk_of_concept_revival_in_pruning-based_un.md)

</div>

<!-- RELATED:END -->
