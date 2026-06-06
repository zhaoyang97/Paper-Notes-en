---
title: >-
  [Paper Note] Dual-branch Robust Unlearnable Examples
description: >-
  [ICML 2026][AI Safety][Unlearnable Examples] This paper proposes DUNE: extending unlearnable example (UE) perturbations from a single spatial domain to joint "spatial + color" dual-domain optimization…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Unlearnable Examples"
  - "Data Poisoning"
  - "Spatial-Color Dual Domain"
  - "Ensemble Perturbation"
  - "Robust Defense"
date: 2026-05-08
content_hash: 9102c700c9c315f9
---

# Dual-branch Robust Unlearnable Examples

**Conference**: ICML 2026  
**arXiv**: [2605.01718](https://arxiv.org/abs/2605.01718)  
**Code**: https://github.com/wxldragon/DUNE (available)  
**Area**: AI Security / Data Protection / Unlearnable Examples  
**Keywords**: Unlearnable Examples, Data Poisoning, Spatial-Color Dual Domain, Ensemble Perturbation, Robust Defense

## TL;DR
This paper proposes DUNE: extending unlearnable example (UE) perturbations from a single spatial domain to joint "spatial + color" dual-domain optimization, aligning perturbation features to shift-induced labels and enhancing with pre-trained model ensembles. On CIFAR-10 / ImageNet, DUNE remains robust against 7 mainstream defenses (including ECLIPSE, ISS-J, COIN), reducing average test accuracy by 14.95%–50.82% compared to 12 SOTA UE methods.

## Background & Motivation

**Background**: The use of web-crawled training data has made "unauthorized DNN training" a security concern. Unlearnable Examples (UEs) protect data owners by adding imperceptible perturbations that cause DNNs to learn spurious shortcut features (perturbation ↔ label mapping). Mainstream methods (EM, REM, LSP, SEP, CUDA, OPS, etc.) optimize perturbations within the spatial $\ell_p$-norm ball.

**Limitations of Prior Work**: (1) **Heuristic shortcut**: Methods like CUDA/LSP use heuristic convolution/linear blocks as perturbations without principled optimization, making them vulnerable to adaptive defenses like COIN; (2) **Domain-constrained**: Single spatial domain perturbations have limited frequency structure, so noise suppression defenses like ISS-J (frequency compression) and ECLIPSE (diffusion purification) can easily remove them; (3) Fig. 2 radar chart shows all existing UEs degrade to near-baseline accuracy under certain defenses, indicating a narrow robustness margin.

**Key Challenge**: Robust UEs require "perturbation diversity," but all perturbations within a single $\ell_p$ domain share the same frequency structure and distribution family, allowing defenses to remove them in bulk. Extending to multiple domains raises the optimization challenge of "how to make multi-domain perturbations collaboratively establish shortcut mappings."

**Goal**: (1) Design a UE framework that optimizes perturbations in multiple domains simultaneously; (2) Ensure multi-domain perturbations are orthogonal/complementary to avoid overlap that harms stealthiness; (3) Use ensemble learning to enhance perturbation transferability across architectures.

**Key Insight**: Images can be decomposed into DC components (block mean luminance) and AC components (high-frequency spatial details). Spatial perturbations mainly affect AC, while color perturbations (luminance shift) mainly affect DC—naturally orthogonal. The perturbation direction is changed from "aligning with ground-truth label" to "aligning with shift-induced label" $y^*=(y+\Delta y)\mod k$, decoupling the learned shortcut from the true label.

**Core Idea**: Decompose UE optimization into two independent subproblems—spatial branch uses PGD to optimize $\ell_\infty$ perturbation $\delta_s$, color branch uses PSO to optimize RGB channel luminance shifts $\delta_c$, jointly pushing features toward the shift-induced class, and use a pre-trained model gallery for ensemble robustness enhancement.

## Method

### Overall Architecture
The overall objective of DUNE: $\min_{\delta_u}\mathbb{E}_{(x,y)}[\mathcal{L}_{CE}(f_\theta(\psi(x;\delta_u)), y^*)]$, $\delta_u\in\Phi_s\times\Phi_c$, $y^*=(y+\Delta y)\mod k$. The authors show this can be decoupled into two independent sub-optimizations:

1. **Spatial Branch**: For each class, use PGD to optimize $\ell_\infty$ perturbation $\delta_{s_i}$, pushing features toward $y_p^*$;
2. **Color Branch**: For each class, use PSO to independently search for RGB channel luminance shifts $(\Delta x_r,\Delta x_g,\Delta x_b)$;
3. **Ensemble Enhancement**: Both branches aggregate gradients/losses over a pre-trained model gallery $\{f_{\theta_j}\}_{j=1}^M$;
4. Final UE: $x_u=\text{clamp}(x+\delta_s+\delta_c, 0, 1)$.

### Key Designs

1. **Shift-induced label feature misalignment**:

    - **Function**: Ensures the model learns not the original $y$ but a fixed offset $y^*=(y+\Delta y)\mod k$ on UEs, severing the association between features and true labels.
    - **Mechanism**: The perturbation optimization target is $\mathcal{L}_{CE}(f_\theta(\psi(x;\delta_d)), y^*)$, i.e., making perturbed sample features approach those of the shift-induced class. Each class shares the same offset $\Delta y$, forming a "unified rotation" shortcut mapping across the dataset (Fig. 4). When the model encounters clean samples at test time, the shortcut fails, leading to generalization collapse.
    - **Design Motivation**: Compared to traditional UE's "min loss" (misclassifying UEs to their original class $y$), the shift-induced target is more stable—it establishes a **deterministic** perturbation→label mapping, explicitly decoupled from the original label, making adaptive defense harder to reverse-engineer.

2. **Spatial-Color Dual-domain Decoupled Optimization**:

    - **Function**: Constructs orthogonal perturbations in spatial and color domains, increasing noise diversity.
    - **Mechanism**: Decompose joint optimization as $\delta_u\triangleq\delta_s\oplus\delta_c$, solving two subproblems independently:
        - **Spatial Branch** (PGD, $T$ steps): $g_t=\nabla_{x_i^t}\mathcal{L}_{CE}(f_\theta(x_i^t), y_p^*)$, $x_i^{t+1}=\text{clip}_{\epsilon}(x_i^t-\beta\cdot\text{sign}(g_t))$;
        - **Color Branch** (PSO, gradient-free): Decompose $x_i$ into R/G/B channels, independently add luminance shifts $\Delta x_r,\Delta x_g,\Delta x_b$ to each channel. PSO searches for the offset combination minimizing ensemble loss + naturalness constraint $\lambda\mathcal{L}_{nc}$, with each class sharing a set of $\delta_c$.
    - **Design Motivation**: DC (luminance) and AC (spatial details) are orthogonal—ECLIPSE's Gaussian noise purification only removes AC; ISS-J's high-frequency compression only affects AC, leaving DC shifts intact. The two branches serve as redundant backups, with non-overlapping attack-defense geometry.

3. **Pre-trained Model Ensemble (unlearnability-enhancing ensemble)**:

    - **Function**: Makes perturbations transferable across architectures, maintaining robustness against unknown defense models.
    - **Mechanism**: Maintain a model gallery $\{f_{\theta_j}\}_{j=1}^M$ (different initializations/architectures). The spatial branch aggregates gradients $g_t=\frac{1}{M}\sum_j \nabla\mathcal{L}_{CE}(f_{\theta_j}(x), y_p^*)$; the color branch aggregates loss $\mathcal{L}_{color}=\frac{1}{M}\sum_j\mathcal{L}_{CE}(f_{\theta_j}(x+\delta_c), y_p^*)+\lambda\mathcal{L}_{nc}$.
    - **Design Motivation**: Perturbations generated on a single surrogate model easily overfit that architecture (e.g., ResNet18), failing on others like VGG19. Ensemble, akin to transferability boosting in adversarial attacks, broadens the perturbation frequency spectrum.

### Loss & Training

- Spatial branch: $\mathcal{L}_{CE}(f_\theta(x+\delta_s), y^*)$, $\ell_\infty\le\epsilon$ (CIFAR-10 $\epsilon=8/255$), $T=20$ PGD steps.
- Color branch: $\mathcal{L}_{color}=\frac{1}{M}\sum_j\mathcal{L}_{CE}+\lambda\mathcal{L}_{nc}$, PSO particle search, aggregated over $N$ samples per class.
- Ensemble models $M$ typically 3–5 surrogates; shift offset $\Delta y$ is fixed within the number of classes $k$ (CIFAR-10 usually $\Delta y=1$).
- Training data: CIFAR-10, ImageNet subsets; evaluation architectures: ResNet18 (intra), VGG19 (cross).

## Key Experimental Results

### Main Results

On CIFAR-10 with ResNet18 training, test accuracy under different defenses (lower is better, i.e., more robust UEs), compared to 12 UE methods + 7 defenses:

| Defense \ UE | EM | REM | CUDA | SEM | **DUNE** |
|----------|-----|-----|------|-----|----------|
| w/o defense | 18.26 | 25.81 | 25.48 | 15.94 | **13.26** |
| AT | 69.72 | 59.12 | 49.32 | 32.43 | **24.96** |
| AA | 82.08 | 45.83 | 40.78 | 39.29 | **19.55** |
| OP | 64.37 | 29.45 | 28.66 | 15.99 | **12.81** |
| ISS-G | 89.09 | 38.87 | 22.89 | 31.94 | **10.18** |
| ISS-J | 78.91 | 81.33 | 43.31 | 81.58 | **28.88** |
| ECLIPSE | 82.07 | 87.16 | 34.18 | 85.82 | **57.49** |
| COIN | 19.49 | 33.67 | 72.02 | 24.22 | **19.21** |
| **AVG** | 63.00 | 51.47 | 39.58 | 40.90 | **23.29** |

VGG19 cross-architecture evaluation (surrogate=ResNet18) also shows DUNE consistently leading, with the AVG column indicating DUNE is the only method stably below 30% among 12 methods.

### Ablation Study

| Configuration | CIFAR-10 ResNet18 w/o defense | With AT defense |
|------|--------|--------|
| Spatial branch only (PGD + shift label) | ≈18 | ≈45 |
| Color branch only (PSO + shift label) | ≈25 | ≈40 |
| Dual branch (no ensemble) | ≈15 | ≈35 |
| **DUNE Full (dual branch + ensemble)** | **13.26** | **24.96** |

(Exact ablation numbers are in Table 3 of the paper; trends are summarized here.)

### Key Findings
- **Dual-domain > Single-domain**: Neither branch alone is robust against both ECLIPSE/ISS-J; only the dual-branch combination withstands both frequency compression and diffusion purification.
- **Smoother loss landscape** (Fig. 3): Models trained with DUNE have a much smoother loss landscape than single-domain UEs like LSP/EM/REM, indicating greater robustness to small perturbations, consistent with the sharpness↔robustness theory of Pham et al. 2024.
- **Significant ensemble margin**: Removing the model gallery causes the largest degradation in cross-architecture (VGG19), proving ensemble is key for transferability.
- **Still robust to adaptive defense**: The authors design two adaptive defenses (assuming defenders know the spatial-color domain), and DUNE maintains low accuracy across four architectures.

## Highlights & Insights
- **Orthogonal domain decomposition**: The physical decoupling of DC vs. AC provides geometric intuition for "why the two branches do not interfere," which is more profound than simply "adding a new loss term."
- **Shift-induced label**: Shifting from "minimizing true-label loss" to "aligning with shift-induced label" is a subtle but impactful paradigm shift—providing UEs with a deterministic rather than random shortcut, making reverse engineering harder.
- **PSO for color branch**: Color perturbations are low-dimensional (3 scalars per class × channel), and the gradient is not directly available (due to hue/luminance operations), making PSO's derivative-free nature a fitting engineering choice.
- **Ensemble enhancement as the "adversarial transferability" equivalent in UE**: Adapting the mature ensemble trick from adversarial attack research to UEs, this idea can be directly transferred to other data poisoning tasks.

## Limitations & Future Work
- Evaluation architectures are still relatively small (ResNet18, VGG19); UE robustness on ViT / large models remains unverified.
- The color branch shares one offset per class, meaning all samples in a class have identical color shifts, which may fail under certain hue augmentation schemes; individualized color perturbations are a natural extension.
- For diffusion-based defenses like ECLIPSE, test accuracy remains at 57.49%, indicating high-quality purifiers are still partially effective against DUNE; further work is needed to counter diffusion models.
- The shift offset $\Delta y$ must be manually selected in multi-class scenarios; the authors use $\Delta y=1$ but do not search for the optimal value. For large class counts (e.g., ImageNet 1000), more refined shift design may be needed.
- Computational cost: dual-branch + PSO + ensemble makes UE generation 5–10× slower than single PGD, making deployment on large datasets (e.g., ImageNet) costly.
- Only tested on image classification; UE design for object detection, segmentation, etc., is not addressed.

## Related Work & Insights
- **vs. EM (Huang et al. 2021)**: The classic min-min optimization UE pioneer, only single spatial domain; DUNE is its robust multi-domain, multi-model successor.
- **vs. REM (Fu et al. 2022)**: REM uses tri-level optimization to resist AT, but still single $\ell_\infty$ domain, failing under ISS-J/ECLIPSE; DUNE addresses frequency diversity with dual domains.
- **vs. CUDA (Sadasivan et al. 2023)**: Heuristic convolutional perturbations, directly broken by COIN (matrix transformation); DUNE uses principled optimization to avoid heuristic inversion.
- **vs. ECLIPSE/ISS-J defenses**: DUNE is the first to extend UEs to both spatial and color domains to evade both defense types.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual-domain orthogonal decomposition + shift-induced label is a novel and self-consistent design in the UE field
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 UEs × 7 defenses × 2 datasets × 2 architectures + 2 adaptive defense matrix experiments are very comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear logic chain from motivation to design to experiments, with well-explained DC/AC physical intuition
- Value: ⭐⭐⭐⭐ Provides data owners with a significantly more robust UE tool, with controllable impact on stealthiness

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information](../../ICLR2026/ai_safety/why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[AAAI 2026\] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference](../../AAAI2026/ai_safety/diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
