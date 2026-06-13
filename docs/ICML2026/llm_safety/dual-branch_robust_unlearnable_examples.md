---
title: >-
  [Paper Note] Dual-branch Robust Unlearnable Examples
description: >-
  [ICML 2026][LLM Safety][Unlearnable Examples] This paper proposes DUNE: expanding the perturbation of Unlearnable Examples (UE) from a single spatial domain to a "spatial + color" dual-domain optimization. By aligning pe…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Unlearnable Examples"
  - "Data Poisoning"
  - "Spatial-Color Dual-Domain"
  - "Ensemble Perturbations"
  - "Robustness Defense"
date: 2026-05-08
content_hash: 6de16c2661348be7
---

# Dual-branch Robust Unlearnable Examples

**Conference**: ICML 2026  
**arXiv**: [2605.01718](https://arxiv.org/abs/2605.01718)  
**Code**: https://github.com/wxldragon/DUNE (Available)  
**Area**: AI Security / Data Protection / Unlearnable Examples  
**Keywords**: Unlearnable Examples, Data Poisoning, Spatial-Color Dual-Domain, Ensemble Perturbations, Robustness Defense

## TL;DR
This paper proposes DUNE: expanding the perturbation of Unlearnable Examples (UE) from a single spatial domain to a "spatial + color" dual-domain optimization. By aligning perturbation features to shift-induced labels and enhancing them with pre-trained model ensembles, DUNE maintains robustness against seven mainstream defenses (including ECLIPSE, ISS-J, and COIN) on CIFAR-10 / ImageNet. The average test accuracy is further reduced by 14.95%–50.82% compared to 12 SOTA UE schemes.

## Background & Motivation

**Background**: Training data crawled from the web makes "unauthorized DNN training" a significant risk. Unlearnable Examples (UEs) protect data owners by adding imperceptible perturbations that lead DNNs to learn incorrect shortcut features (perturbation $\leftrightarrow$ label mapping). Mainstream methods (EM, REM, LSP, SEP, CUDA, OPS, etc.) focus on perturbation optimization within an $\ell_p$-norm ball in the spatial domain.

**Limitations of Prior Work**: (1) **Heuristic shortcut**: Methods like CUDA / LSP use empirical convolutional or linear blocks as perturbations without principled optimization, making them easily broken by adaptive defenses like COIN; (2) **Domain-constrained**: Single spatial domain perturbations have a simple frequency structure, which noise-suppression defenses like ISS-J (frequency compression) and ECLIPSE (diffusion purification) can easily remove; (3) The radar chart in Fig. 2 shows that all existing UEs degrade to near-baseline accuracy under certain defenses, indicating narrow robustness boundaries.

**Key Challenge**: UEs require "perturbation diversity" for robustness, but all perturbations in a single $\ell_p$ domain share the same frequency structure and distribution family, allowing defenses to remove them in batches by identifying that family. Expanding to multiple domains introduces the optimization challenge of "how to make multi-domain perturbations synergistically establish shortcut mappings."

**Goal**: (1) Design a UE framework capable of simultaneous multi-domain perturbation optimization; (2) ensure multi-domain perturbations are orthogonal/complementary to avoid destroying stealthiness; (3) use ensemble learning to strengthen cross-architecture transferability.

**Key Insight**: Images can be decomposed into DC components (block mean brightness) + AC components (high-frequency spatial details). Spatial perturbations primarily affect AC, while color perturbations (brightness shifts) primarily affect DC—making them naturally orthogonal. Simultaneously, changing the perturbation direction from "aligning with ground-truth labels" to "aligning with shift-induced labels" $y^*=(y+\Delta y)\mod k$ decouples the learned shortcuts from the true labels.

**Core Idea**: UE optimization is decomposed into two independent sub-problems: the spatial branch uses PGD to optimize $\ell_\infty$ perturbation $\delta_s$, while the color branch uses PSO to optimize RGB three-channel brightness shifts $\delta_c$. Together, they push features toward shift-induced classes, with robustness further enhanced using a gallery of pre-trained models for ensemble learning.

## Method

### Overall Architecture
The objective of DUNE is: $\min_{\delta_u}\mathbb{E}_{(x,y)}[\mathcal{L}_{CE}(f_\theta(\psi(x;\delta_u)), y^*)]$, where $\delta_u\in\Phi_s\times\Phi_c$ and $y^*=(y+\Delta y)\mod k$. The authors prove this can be decoupled into two independent sub-optimizations:

1. **Spatial Branch**: For each class, PGD optimizes the $\ell_\infty$ perturbation $\delta_{s_i}$ to move features toward $y_p^*$;
2. **Color Branch**: For each class, PSO independently searches for brightness offsets $(\Delta x_r, \Delta x_g, \Delta x_b)$ across the RGB channels;
3. **Ensemble Enhancement**: Both branches aggregate gradients or losses across a pre-trained model gallery $\{f_{\theta_j}\}_{j=1}^M$;
4. **Final UE**: $x_u=\text{clamp}(x+\delta_s+\delta_c, 0, 1)$.

### Key Designs

1. **Shift-induced label feature misalignment**:

    - **Function**: Forces the model to learn a fixed offset $y^*=(y+\Delta y)\mod k$ instead of the original $y$ on UEs, thereby severing the association between features and true labels.
    - **Mechanism**: The perturbation optimization target is $\mathcal{L}_{CE}(f_\theta(\psi(x;\delta_d)), y^*)$, which brings the features of perturbed samples close to those of the shift-induced class. Every class shares the same shift $\Delta y$, forming a "unidirectionally rotated" shortcut mapping across the dataset (Fig. 4). At test time, clean samples lack this shortcut, causing generalization to collapse.
    - **Design Motivation**: Compared to traditional UE "min loss" (which makes the model misclassify UEs to the original class $y$), the shift-induced target is more stable—it establishes a **deterministic** perturbation $\rightarrow$ label mapping that is explicitly decoupled from the original label and harder to reverse-engineer for adaptive defenses.

2. **Spatial-Color Dual-domain Decoupled Optimization**:

    - **Function**: Constructs orthogonal perturbations in spatial and color domains respectively to expand noise diversity.
    - **Mechanism**: The joint optimization is decomposed as $\delta_u\triangleq\delta_s\oplus\delta_c$, solved as independent sub-problems:
        - **Spatial Branch** (PGD, $T$ steps): $g_t=\nabla_{x_i^t}\mathcal{L}_{CE}(f_\theta(x_i^t), y_p^*)$, $x_i^{t+1}=\text{clip}_{\epsilon}(x_i^t-\beta\cdot\text{sign}(g_t))$;
        - **Color Branch** (PSO, gradient-free): $x_i$ is split into R/G/B channels, and brightness offsets $\Delta x_r, \Delta x_g, \Delta x_b$ are added to each channel. PSO searches for the offset combination that minimizes the ensemble loss + naturalness constraint $\lambda\mathcal{L}_{nc}$, with an entire class sharing a single $\delta_c$.
    - **Design Motivation**: DC (brightness) and AC (spatial details) are orthogonal. Thus, Gaussian noise purification (like ECLIPSE) only removes AC components; high-frequency compression (like ISS-J) only damages AC, while DC shifts remain intact. The two branches serve as redundant backups, with no geometric overlap in terms of defense.

3. **Unlearnability-enhancing ensemble**:

    - **Function**: Makes perturbations transferable across architectures and robust to unknown defense models.
    - **Mechanism**: A model gallery $\{f_{\theta_j}\}_{j=1}^M$ (different initializations and architectures) is maintained. The spatial branch aggregates gradients $g_t=\frac{1}{M}\sum_j \nabla\mathcal{L}_{CE}(f_{\theta_j}(x), y_p^*)$, and the color branch aggregates losses $\mathcal{L}_{color}=\frac{1}{M}\sum_j\mathcal{L}_{CE}(f_{\theta_j}(x+\delta_c), y_p^*)+\lambda\mathcal{L}_{nc}$.
    - **Design Motivation**: Perturbations generated by a single surrogate model (e.g., ResNet18) tend to overfit its architecture and fail on VGG19. This ensemble approach is similar to transferability boosting in adversarial attacks, broadening the frequency spectrum of the perturbation.

### Loss & Training
- Spatial branch: $\mathcal{L}_{CE}(f_\theta(x+\delta_s), y^*)$, $\ell_\infty\le\epsilon$ (CIFAR-10 $\epsilon=8/255$), $T=20$ PGD steps.
- Color branch: $\mathcal{L}_{color}=\frac{1}{M}\sum_j\mathcal{L}_{CE}+\lambda\mathcal{L}_{nc}$, PSO particle search, aggregated over $N$ samples per class.
- The ensemble $M$ usually consists of 3–5 surrogate models; the shift offset $\Delta y$ is fixed within the number of classes $k$ (CIFAR-10 usually $\Delta y=1$).
- Training data: CIFAR-10, ImageNet subsets; Evaluation architectures: ResNet18 (intra), VGG19 (cross).

## Key Experimental Results

### Main Results

Test accuracy of ResNet18 trained on CIFAR-10 under different defenses (lower is better, indicating more robust UE), comparing 12 UE schemes across 7 defenses:

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

DUNE also leads in VGG19 cross-architecture evaluation (surrogate=ResNet18). In the AVG column, DUNE is the only one among 12 methods to remain consistently below 30%.

### Ablation Study

| Configuration | CIFAR-10 ResNet18 w/o defense | After AT Defense |
|------|--------|--------|
| Spatial branch only (PGD + shift label) | ≈18 | ≈45 |
| Color branch only (PSO + shift label) | ≈25 | ≈40 |
| Dual-branch (no ensemble) | ≈15 | ≈35 |
| **DUNE Full (Dual-branch + ensemble)** | **13.26** | **24.96** |

(Specific ablation numbers are provided in Table 3 of the paper; the trends are approximated here.)

### Key Findings
- **Dual-domain > Single-domain**: Neither branch alone provides sufficient robustness against both ECLIPSE and ISS-J; the dual-branch combination is required to counter both frequency compression and diffusion purification.
- **Smoother loss landscape** (Fig. 3): The loss landscape of models trained with DUNE is significantly smoother than those trained with single-domain UEs like LSP/EM/REM, indicating that the perturbation distribution is more robust to small perturbations, consistent with the sharpness $\leftrightarrow$ robustness theory of Pham et al. (2024).
- **Significant ensemble margin**: Removing the model gallery causes the most severe degradation in cross-arch (VGG19) performance, proving that ensemble learning is key to transferability.
- **Robust against adaptive defenses**: The authors designed two adaptive defenses assuming the defender knows the spatial-color domain information; DUNE still maintains low accuracy across four architectures.

## Highlights & Insights
- **Orthogonal Domain Decomposition**: The physical decoupling of DC vs. AC provides geometric intuition as to why the two branches do not conflict, which is more profound than simply adding an engineering loss term.
- **Shift-induced label**: Moving from "minimizing true-label loss" to "aligning with shift-induced labels" is a seemingly small but effective paradigm shift—it provides UE with a deterministic rather than random shortcut, making it harder to reverse-engineer.
- **PSO for the Color Branch**: Color perturbations are low-dimensional (3 scalars per class), and the gradient direction is not directly differentiable (due to operations on hue/luminance). The derivative-free nature of PSO is a perfect fit for this engineering choice.
- **Ensemble enhancement as the "Adversarial Transferability" equivalent in UE**: Introducing ensemble tricks from the adversarial attack community into the UE field allows for logic that can be directly transferred to other data poisoning tasks.

## Limitations & Future Work
- The evaluation architectures are relatively small (ResNet18, VGG19); UE robustness on ViT or larger models has not been verified.
- The color branch uses a single shift per class, meaning color drifts for samples in the same class are identical, which might fail under certain hue-enhancing augmentations. Individualized color perturbations are a natural extension.
- Accuracy remains at 57.49% under the diffusion-based defense ECLIPSE, indicating that high-quality purifiers are still partially effective against DUNE; further work against diffusion models is needed.
- The shift offset $\Delta y$ must be selected manually in multi-class scenarios; the authors use $\Delta y=1$ but did not perform an optimal value search; for large numbers of classes (e.g., ImageNet 1000), the shift might require more detailed design.
- Computational overhead: The dual-branch + PSO + ensemble approach makes UE generation 5–10× slower than single PGD, resulting in significant deployment costs for large datasets.
- Testing was limited to image classification; designs for tasks like object detection or segmentation have not been explored.

## Related Work & Insights
- **vs. EM (Huang et al. 2021)**: The classic min-min optimization pioneer of UE, restricted to a single spatial domain; DUNE is its multi-domain, multi-model robust successor.
- **vs. REM (Fu et al. 2022)**: REM uses tri-level optimization to combat AT but remains in a single $\ell_\infty$ domain, failing under ISS-J/ECLIPSE; DUNE solves frequency diversity via dual-domains.
- **vs. CUDA (Sadasivan et al. 2023)**: Heuristic convolutional perturbations are directly solved by COIN (matrix transformation); DUNE uses principled optimization to avoid heuristic inversion.
- **vs. ECLIPSE/ISS-J defenses**: DUNE is the first work to extend UE to both spatial and color domains simultaneously to bypass both types of defenses.

## Rating
- Novelty: ⭐⭐⭐⭐ Orthogonal dual-domain decomposition + shift-induced labels is a novel and self-consistent design in the UE field.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The matrix experiments involving 12 UEs × 7 defenses × 2 datasets × 2 architectures + 2 adaptive defenses are very solid.
- Writing Quality: ⭐⭐⭐⭐ The logic chain from motivation to design to experiments is clear, and the DC/AC physical intuition is well-explained.
- Value: ⭐⭐⭐⭐ Provides data owners with a significantly more robust UE tool with controllable impacts on stealthiness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](../../ICLR2026/llm_safety/when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)
- [\[ICCV 2025\] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation](../../ICCV2025/llm_safety/temporal_unlearnable_examples_preventing_personal_video_data_from_unauthorized_e.md)
- [\[ICML 2026\] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)
- [\[ICLR 2026\] Perturbation-Induced Linearization: Constructing Unlearnable Data with Solely Linear Classifiers](../../ICLR2026/llm_safety/perturbation-induced_linearization_constructing_unlearnable_data_with_solely_lin.md)
- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](../../ACL2026/llm_safety/from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)

</div>

<!-- RELATED:END -->
