---
title: >-
  [Paper Note] Manipulating Feature Visualizations with Gradient Slingshots
description: >-
  [NeurIPS 2025][Robotics][Feature Visualization] This paper proposes Gradient Slingshots (GS), a method that "carves" a quadratic activation landscape in the out-of-distribution (OOD) input region of a model, directing the gradient-based optimization of Feature Visualization (FV) toward an arbitrary target image. The approach causes FV to converge to a predefined spurious image while leaving the model's architecture, classification accuracy, and internal feature representations largely intact, thereby exposing a serious vulnerability of FV as a model auditing tool.
tags:
  - NeurIPS 2025
  - Robotics
  - Feature Visualization
  - Gradient Slingshot
  - Activation Maximization
  - XAI Security
  - Adversarial Fine-tuning
date: 2026-05-08
content_hash: 679f3dddbd3df0d7
---

# Manipulating Feature Visualizations with Gradient Slingshots

**Conference**: NeurIPS 2025
**arXiv**: [2401.06122](https://arxiv.org/abs/2401.06122)
**Code**: [GitHub](https://github.com/dilyabareeva/grad-slingshot)
**Area**: Explainable AI / Adversarial Attacks
**Keywords**: Feature Visualization, Gradient Slingshot, Activation Maximization, XAI Security, Adversarial Fine-tuning

## TL;DR

This paper proposes Gradient Slingshots (GS), a method that "carves" a quadratic activation landscape in the out-of-distribution (OOD) input region of a model, directing the gradient-based optimization of Feature Visualization (FV) toward an arbitrary target image. The approach causes FV to converge to a predefined spurious image while leaving the model's architecture, classification accuracy, and internal feature representations largely intact, thereby exposing a serious vulnerability of FV as a model auditing tool.

## Background & Motivation

**Background**: Feature Visualization (FV) is one of the most widely used interpretability techniques in XAI. By synthesizing inputs that maximally activate specific features (neurons or directions), FV reveals the concepts learned by DNNs and is broadly applied to understand internal representations, detect backdoor attacks, and identify biases.

**Limitations of Prior Work**: Despite widespread trust in FV, its reliability has received little scrutiny. Prior work (Geirhos et al.) demonstrated that FV can be manipulated by modifying the model architecture (embedding fooling circuits), but such architectural changes are easily detected upon inspection. Whether FV can be covertly manipulated without altering the architecture remains a critical and unanswered security question.

**Key Challenge**: FV optimization operates in the OOD region (e.g., initialized in the Fourier domain followed by gradient ascent), while the model's classification behavior depends solely on in-distribution (ID) samples. This OOD–ID decoupling means an adversary can freely reshape the gradient field in the OOD region without affecting normal model functionality.

**Goal**: (1) Demonstrate that FV can be manipulated without modifying the model architecture or significantly degrading performance; (2) Provide theoretical guarantees (convergence proofs) and systematic empirical validation; (3) Propose simple defensive measures to detect such attacks.

**Key Insight**: Exploiting the OOD nature of FV optimization, the method "carves" a tunnel in the model's activation landscape from the initialization region to the target image, so that the gradient field naturally steers toward the target—analogous to the accelerating motion of a slingshot.

**Core Idea**: Construct a quadratic activation landscape (the gradient slingshot) along the FV optimization trajectory, causing optimization to "launch" from the initialization toward an arbitrary target image, while a preservation loss maintains the model's original behavior on natural images.

## Method

### Overall Architecture

The adversary selects a target feature $f$ and a target image $\bm{x^t}$, then fine-tunes model parameters to construct three regions in the parameterization domain $\mathcal{Q}$ (e.g., the Fourier domain): a **slingshot region** (around the FV initialization), a **landing region** (around the target image), and a **tunnel** connecting them. Within these three regions, the feature's activation function is replaced by a quadratic function whose extremum coincides with the target point, causing FV's gradient ascent to naturally converge to the target. A preservation loss ensures the model's behavior on natural images remains unchanged.

### Key Designs

1. **Theoretical Framework: Quadratic Attraction Field**

   - **Function**: Guarantees convergence of FV optimization to the target image from any initialization.
   - **Mechanism**: An auxiliary function $\phi$ is constructed such that $\nabla(\phi \circ \eta)(\bm{q}) = \gamma(\bm{q^t} - \bm{q})$, i.e., the gradient always points toward the target. Integration yields the quadratic form $(\phi \circ \eta)(\bm{q}) = -\frac{\gamma}{2}\|\bm{q^t} - \bm{q}\|_2^2 + C$. Under step size $\epsilon < 1/\gamma$, the iteration $\bm{q}^{(i+1)} = (1-\epsilon\gamma)\bm{q}^{(i)} + \epsilon\gamma\bm{q^t}$ guarantees geometric convergence $d^{(i+1)} = (1-\epsilon\gamma)d^{(i)}$.
   - **Design Motivation**: Provides rigorous theoretical convergence guarantees without heuristics, enabling the adversary to precisely control FV outcomes.

2. **Manipulation Loss and Preservation Loss**

   - **Function**: Respectively achieve the two conflicting objectives of "FV output becomes the target image" and "normal model functionality is preserved."
   - **Mechanism**: The manipulation loss $\mathcal{L_M}$ takes two forms—a gradient form that directly constrains the gradient field direction, and an activation form $\mathcal{L_M^{act}}$ that constrains activations to approximate the quadratic function. The preservation loss $\mathcal{L_P}$ uses MSE to constrain the fine-tuned feature extractor to match the original activations on the training set, assigning higher weight $w$ to the target feature $f$. Total loss: $\mathcal{L} = \alpha\mathcal{L_P} + (1-\alpha)\mathcal{L_M}$.
   - **Design Motivation**: Only the OOD region's behavior needs to be modified while the ID region remains intact. The gradient form directly controls the optimization trajectory but requires second-order optimization; the activation form is a more practical alternative.

3. **Three-Region Decomposition: Slingshot Region–Tunnel–Landing Region**

   - **Function**: Precisely delineates the subset of parameter space to be manipulated, minimizing impact on the model.
   - **Mechanism**: The **slingshot region** $\mathbb{B}$ is a neighborhood of the FV initialization distribution—gradients are largest here, producing the "launch" effect. The **landing region** $\mathbb{L}$ is a neighborhood of the target image—gradients vanish here, ensuring stable convergence. The **tunnel** $\mathbb{T}_{B,L}$ is the convex hull connecting the two—ensuring the optimization trajectory remains within the manipulated region. Only points sampled inside the tunnel are used for fine-tuning.
   - **Design Motivation**: This precise spatial decomposition confines manipulation to the OOD region visited by FV optimization, leaving natural image processing entirely unaffected.

### Loss & Training

Total loss: $\mathcal{L}(\theta) = \alpha\mathcal{L_P}(\theta) + (1-\alpha)\mathcal{L_M}(\theta)$, where $\alpha$ controls the preservation–manipulation trade-off. The preservation loss remains effective using only 0.1%–10% of the training set.

## Key Experimental Results

### Main Results

| Architecture | Dataset | FV Method | Target Lbl.↑ | GT Lbl.↓ | CLIP↑ | LPIPS↓ | Accuracy Change | AUROC |
|---|---|---|---|---|---|---|---|---|
| 6-layer CNN | MNIST | Pixel-AM | Near-perfect | Near 0 | ~1.0 | ~0 | Negligible | 1.00 |
| VGG-9 | CIFAR-10 | Fourier FV | Near-perfect | Near 0 | ~1.0 | ~0 | Negligible | 1.00 |
| ResNet-50 | ImageNet | Fourier FV | 0.32 | 0.23 | 0.72 | 0.59 | −5.19% | 1.00 |
| ViT-L/32 | ImageNet | Fourier FV | Effective | Low | Moderate–High | Moderate | Small | 1.00 |

### Ablation Study

| $\alpha$ | AUROC | Accuracy (%) | LPIPS↓ | Notes |
|---|---|---|---|---|
| 0.90 (high preservation) | 1.00 | 76.07 | 0.69 | Manipulation nearly ineffective |
| 0.64 (balanced) | 1.00 | 75.13 | 0.59 | Optimal balance |
| 0.50 | 1.00 | 71.52 | 0.63 | Stronger manipulation but accuracy drops |
| 0.10 (high manipulation) | 0.90 | 30.19 | 0.71 | Excessive manipulation causes failure |

### Key Findings

- **Non-monotonic effect of $\alpha$**: Both excessively high and excessively low $\alpha$ degrade manipulation effectiveness—too high preserves too much of the original gradient; too low may introduce additional local optima. The optimal range lies between 0.5 and 0.65.
- **Model capacity determines manipulation precision**: As VGG width increases from 8× to 64×, LPIPS decreases from 0.17 to 0.02; models with more parameters more readily "memorize" the target image in the OOD region.
- **Target image selection affects results**: In-distribution natural images yield the best results as targets, especially those semantically related to the target neuron.
- **Weapon detection case study**: After label poisoning a CLIP ViT-L/14 weapon detector, FV displays a penguin rather than an assault rifle, yet the feature's classification AUROC remains 1.00, perfectly concealing the manipulation.

## Highlights & Insights

- **Deep insight into OOD/ID decoupling**: FV operates in OOD space while the model functions in ID space, with minimal overlap—this represents a fundamental security vulnerability, not a defect of any specific FV method, but a systemic weakness of the entire class of optimization-based interpretability tools. This insight generalizes to all optimization-based explainability methods.
- **Simplicity of the defense**: Directly checking whether natural activation-maximizing examples (top-activating training samples) are consistent with FV outputs suffices to detect attacks—a Jaccard similarity of 0.84 confirms that the preservation loss indeed maintains in-distribution behavior.
- **PINN-inspired optimization technique**: Embedding the gradient field directly into the loss function (analogous to physics-informed neural networks) to control the FV optimization trajectory is a transferable technique applicable to other settings requiring precise control over optimization dynamics.

## Limitations & Future Work

- **Hyperparameter tuning cost**: Finding the optimal $\alpha$ requires grid search, with computational overhead comparable to training multiple models.
- **Incomplete FV method coverage**: Although validated across multiple FV variants, not all regularization strategies are exhaustively evaluated.
- **Insufficient evaluation of internal representation impact**: Model preservation is assessed solely via classification accuracy and AUROC; finer-grained representation analyses (e.g., CKA, probing) may reveal hidden effects.
- **Defense relies on labeled data**: Natural AM-based detection requires a labeled test set and may degrade in unlabeled settings.
- **Societal impact**: Although the paper also proposes defenses, public disclosure of the attack method may be exploited to circumvent AI auditing.

## Related Work & Insights

- **vs. Geirhos et al. (fooling circuits)**: Their approach embeds additional convolutional layers into the model to encode target images, making it detectable via architectural inspection. GS modifies only weights without altering the architecture, offering far greater covertness.
- **vs. Nanfack et al. (fine-tuning attacks)**: Their fine-tuning approach focuses on preserving task performance without explicitly preserving internal representations, potentially altering the model's actual mechanisms rather than only its explanations. GS simultaneously maintains internal representations and achieves explanation manipulation.
- **vs. Anders et al. (fairwashing)**: Fairwashing manipulates attribution-based explanations (e.g., saliency maps), whereas GS manipulates feature-level explanations (FV), targeting different layers of the XAI stack.
- This work carries important cautionary implications for XAI security research: any optimization-based interpretability tool may be susceptible to analogous OOD manipulation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Elegant theoretical framework (quadratic attraction field + three-region decomposition); deep insight into OOD/ID decoupling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers architectures from 6-layer CNNs to ViT-L and FV methods from Pixel-AM to Fourier FV, with a weapon detection case study and defense validation; highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations, carefully designed experiments, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Important cautionary contribution to research on the trustworthiness of XAI.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [\[NeurIPS 2025\] Breaking the Gradient Barrier: Unveiling Large Language Models for Strategic Classification](breaking_the_gradient_barrier_unveiling_large_language_models_for_strategic_clas.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over-reliance_correct.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)

<!-- RELATED:END -->
