---
title: >-
  [Paper Note] Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction
description: >-
  [NeurIPS 2025][Image Generation][adversarial examples] This paper proposes InSUR, a multi-dimensional instruction uncertainty reduction framework that stabilizes adversarial optimization via a ResAdv-DDIM sampler…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "adversarial examples"
  - "semantic constraint"
  - "diffusion models"
  - "3D adversarial"
  - "transfer attack"
date: 2026-05-08
content_hash: c95b4869af5d228b
---

# Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction

**Conference**: NeurIPS 2025
**arXiv**: [2510.22981](https://arxiv.org/abs/2510.22981)
**Code**: Not released
**Area**: Image Generation
**Keywords**: adversarial examples, semantic constraint, diffusion models, 3D adversarial, transfer attack

## TL;DR

This paper proposes InSUR, a multi-dimensional instruction uncertainty reduction framework that stabilizes adversarial optimization via a ResAdv-DDIM sampler, constrains attack scenarios through context-aware encoding, and evaluates semantic fidelity using WordNet-based semantic abstraction. InSUR is the first method to generate 2D/3D semantic-constrained adversarial examples (SemanticAE) from natural language instructions.

## Background & Motivation

Traditional adversarial example research focuses on finding small perturbations around existing data. Generating adversarial examples directly from natural language instructions (SemanticAE) is an emerging yet underexplored direction. Given a semantic description, the goal is to generate data that is semantically correct but cannot be correctly recognized by deep learning models. Existing methods (AdvDiff, SD-NAE, VENOM, etc.) suffer from three limitations:
1. **Referential diversity** causes inconsistent language guidance across multi-step diffusion models, destabilizing adversarial optimization.
2. **Descriptive incompleteness** leads to poor adaptability to attack scenarios.
3. **Ambiguous semantic boundaries** make it difficult to evaluate SemanticAE generators.

## Core Problem

How to generate transferable, adaptive, and effective semantic-constrained adversarial examples from uncertain human natural language instructions?

## Method

### Problem Formulation

$$\text{find } x_{\text{adv}} \in \mathcal{S}(\text{Text}) \quad \text{s.t.} \quad \mathcal{M}(x_{\text{adv}}) \in A_{\text{Text}}$$

where $\mathcal{S}(\text{Text})$ is the set of data satisfying the semantic constraint, $\mathcal{M}$ is the target (black-box) model, and $A_{\text{Text}}$ is the set of erroneous outputs semantically inconsistent with the instruction.

### Module 1: ResAdv-DDIM Sampler (Addressing Referential Diversity)

Core Idea: At each denoising step, a coarse estimate $g_\theta(x_t)$ of $x_0$ is obtained via DDIM residual shortcut prediction, rather than approximating $\nabla_{x_t}\mathcal{L}_{\text{ATK}}$ directly by $\nabla_{x_0}\mathcal{L}_{\text{ATK}}$.

$$g_\theta(x_t) = \underbrace{f_{\theta,\Delta T_1} \circ f_{\theta,\Delta T_2} \circ \cdots \circ f_{\theta,\Delta T_k}}_{k \text{ steps, } k \ll T/\Delta T}(x_t)$$

$$x_{t-\Delta T} = f_{\theta,\Delta T}\left(\arg\max_{x_t} \mathcal{L}_{\text{ATK}}(\mathcal{M}(g_\theta(x_t)))\right)$$

Semantic constraints are enforced via an upper bound on trajectory deviation:

$$\|\text{Denoise}_{\text{DDIM}}(x_{t_s-\Delta T}) - \text{Denoise}_{\text{Adv}}(x_{t_s-\Delta T})\|_2 < \epsilon$$

Adaptive attack optimization employs an early stopping mechanism: optimization terminates when the estimated attack failure probability falls below threshold $\xi_1 = 0.1$ or $\xi_2 = 0.01$.

### Module 2: Context-Encoded Attack Scenario Constraint

**2D Generation**: Conditional and unconditional guidance is redistributed via guidance masking:

$$\epsilon_\theta(x_t, t) = (1-M) \cdot \epsilon_{\theta,\text{Unconditional}}(x_t, t) + M \cdot \epsilon_{\theta,\text{Conditional}}(x_t, t, \text{Text})$$

**3D Generation** (first implementation): ResAdv-DDIM is integrated with a Gaussian Splatting renderer:

$$g_\theta(z_t, \mathbf{pos}, \text{Camera}) = \text{Renderer}_{\text{GS}}(\mathcal{D}_{\text{GS}}(f_{\theta,\Delta T_1} \circ \cdots \circ f_{\theta,\Delta T_k}(z_t, \mathbf{pos}), \mathbf{pos}), \text{Camera})$$

Gradient accumulation over unknown camera poses is performed via the Expectation over Transformation (EoT) method.

### Module 3: Semantic Abstraction Evaluation Enhancement

A hierarchical label taxonomy is constructed based on WordNet, defining escape attack tasks at the abstraction level:

$$\text{Text} = \text{"Realistic image of [AbstractedLabel], specifically, [label]"}$$

$$A_{\text{Text}} = \{\text{label}_{\text{Adv}} \mid \text{AbstractedLabel} \notin \mathbf{Ancestors}(\text{label}_{\text{Adv}})\}$$

The paper proposes a relative attack success rate $ASR_{\text{Relative}}$ and a paired semantic divergence metric $\text{SemanticDiff}_\mathcal{S}$, verifying semantic consistency by simultaneously generating a non-adversarial exemplar $x_{\text{exemplar}}$.

## Key Experimental Results

### 2D SemanticAE ($\epsilon = 2.5$, average ASR across target models)

| Surrogate | Method | Acc.↓ | ASR↑ | CLIP-Q↑ | LPIPS↓ |
|---------|------|-------|------|---------|--------|
| ResNet50 | MI-FGSM | 33.4% | 41.5% | 0.548 | 0.201 |
| ResNet50 | SD-NAE | 37.1% | 47.4% | 0.841 | 0.457 |
| ResNet50 | VENOM | 34.5% | 34.4% | 0.795 | 0.023 |
| ResNet50 | **InSUR** | **15.1%** | **62.0%** | 0.815 | 0.031 |
| ViT-B | VENOM | 30.5% | 40.6% | 0.796 | 0.021 |
| ViT-B | **InSUR** | **10.9%** | **69.7%** | 0.815 | 0.038 |

- Across all surrogate-and-task configurations, InSUR achieves at least a **1.19×** improvement in average ASR and at least **1.08×** in minimum ASR.
- On the ViT-B surrogate, ASR reaches 69.7%, substantially outperforming VENOM at 40.6%.

### Abstract Label Escape Task

| Surrogate | Method | Acc.↓ | ASR↑ | CLIP-Q↑ |
|---------|------|-------|------|---------|
| ResNet50 | VENOM | 51.0% | 34.9% | 0.779 |
| ResNet50 | **InSUR** | **35.2%** | **47.9%** | **0.808** |
| ViT-B | VENOM | 46.3% | 40.3% | 0.780 |
| ViT-B | **InSUR** | **28.7%** | **55.4%** | **0.814** |

## Highlights & Insights

- ⭐ First method to achieve reference-free 3D semantic adversarial example generation from natural language instructions.
- ⭐ ResAdv-DDIM resolves adversarial direction inconsistency in multi-step diffusion models via residual shortcut prediction.
- ⭐ The WordNet-based evaluation framework provides a principled definition of semantic boundaries for SemanticAE.
- The paper systematically decomposes instruction uncertainty into three dimensions and addresses each independently.
- InSUR achieves an excellent Pareto frontier balancing semantic preservation (low LPIPS) and attack effectiveness (high ASR).

## Limitations & Future Work

- The $\epsilon$ parameter requires manual tuning and may vary across scenarios.
- 3D generation relies on the Trellis framework; generalization to other 3D generative models remains unverified.
- Evaluation metrics avoid FID/IS (due to concerns about adversarial interference), but no alternative generation quality metrics are provided.
- The selection strategy for the number of residual steps $k \in \{1,2,3,4\}$ in ResAdv-DDIM is not thoroughly discussed.
- Generation time (7.26s) is slower than VENOM (3.09s), though still faster than SD-NAE (24.43s).

## Related Work & Insights

| Method | Generation Form | Transfer Attack | 3D Support | Semantic Constraint |
|------|---------|---------|---------|------------|
| AdvDiff | Perturbation-based | Weak | ✗ | Implicit |
| SD-NAE | Generative | Moderate | ✗ | End-to-end optimization |
| VENOM | Generative | Moderate | ✗ | Sampling process modification |
| **InSUR** | **Generative** | **Strong** | **✓** | **Multi-dimensional uncertainty reduction** |

The residual prediction concept in ResAdv-DDIM can be adapted to other diffusion model control tasks. The semantic abstraction evaluation methodology can be generalized as a universal semantic consistency assessment framework. The guidance masking partition strategy is applicable to controllable image editing (e.g., foreground/background guidance separation). 3D adversarial example generation has direct application value for autonomous driving safety evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (SemanticAE conceptualization + first 3D implementation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-surrogate, multi-task, comprehensive 2D/3D experiments)
- Writing Quality: ⭐⭐⭐ (dense content, complex notation system, moderate readability)
- Value: ⭐⭐⭐⭐ (provides new tools for AI safety evaluation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)

</div>

<!-- RELATED:END -->
