---
title: >-
  [Paper Note] Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification
description: >-
  [ICML 2026][Calibration] FGR employs DCT low-pass filtering to eliminate high-frequency spurious shortcuts in training images to improve OOD calibration accuracy. It utilizes a geometric projection to resolve gradient co…
tags:
  - "ICML 2026"
  - "Calibration"
  - "distribution shift"
  - "DCT low-pass filtering"
  - "gradient projection"
  - "domain-invariant features"
date: 2026-05-08
content_hash: 84da15fd281fde28
---

# Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification

**Conference**: ICML 2026  
**arXiv**: [2508.19830](https://arxiv.org/abs/2508.19830)  
**Code**: https://github.com/YilinZhang107/FGR-Calib (Available)  
**Area**: Interpretability / Confidence Calibration / Distribution Shift Robustness  
**Keywords**: Calibration, distribution shift, DCT low-pass filtering, gradient projection, domain-invariant features

## TL;DR
FGR employs DCT low-pass filtering to eliminate high-frequency spurious shortcuts in training images to improve OOD calibration accuracy. It utilizes a geometric projection to resolve gradient conflicts between "improving calibration" and "preserving ID performance" as a hard constraint, effectively suppressing OOD ECE while maintaining ID performance without needing to tune loss weights.

## Background & Motivation

**Background**: When deploying deep models, it is crucial not only to be accurate but also to provide reliable confidence scores—in high-risk scenarios like healthcare or autonomous driving, an incorrect prediction with 0.9 confidence is far more dangerous than one with 0.5 confidence. Calibration methods follow two main paths: post-hoc methods (Temperature Scaling, isotonic regression, etc.) that fit a confidence transformation on a fixed model, and train-time methods (Focal Loss, MMCE, Soft-ECE, Dual Focal Loss, Label Smoothing, Mixup, etc.) that add regularization to the loss to suppress overconfidence.

**Limitations of Prior Work**: Existing methods work well on In-Distribution (ID) data but fail under distribution shift (changes in weather, lighting, sensors, hospitals, or geography). A typical ResNet accuracy drops from 76% to 18% on ImageNet-C while maintaining alarmingly high confidence. Previous "calibration under distribution shift" methods often rely on target domain information: requiring multi-domain training data, synthetic validation sets to simulate the target domain, or additional assumptions like Bayesian/feature density, which are often unavailable during deployment.

**Key Challenge**: Maintaining calibration on unknown OOD data requires the model to rely only on features that are stable across distributions. However, removing unstable signals (such as high-frequency textures) inevitably harms the fine-grained decision boundaries for ID data, leading to under-confidence. This creates an irreconcilable conflict between "OOD calibration vs. ID calibration," which conventional multi-task weighting cannot handle as a hard constraint like "ID performance must not degrade."

**Goal**: (1) Improve OOD calibration without access to any target domain information; (2) Preserve ID calibration performance without introducing additional loss balancing coefficients.

**Key Insight**: Analyzing distribution shift from the frequency domain—evidence from (Yin et al. 2019 / Fridovich-Keil et al. 2022 / Li et al. 2023) suggests that models often use high-frequency statistics as classification shortcuts, and distribution shifts primarily perturb these high-frequency components. Actively masking high-frequency signals during training forces the model to capture domain-invariant features like "shape" or "semantics." The side effects of masking (ID under-confidence) are then addressed via a hard constraint mechanism at the optimizer level.

**Core Idea**: "Frequency domain filtering to build domain-invariant features + Gradient projection treating ID calibration as a hard constraint"—the former provides robustness from the data perspective, while the latter acts as a safety net during optimization. These two components are decoupled yet work in tandem.

## Method

FGR is a train-time framework consisting of "low-pass filtering to generate a mixed training set" and "gradient projection." This training process is appended to regular classification training (inserted from the 200th epoch in experiments).

### Overall Architecture
At the start of each epoch, a proportion $\rho$ of training samples is randomly selected for DCT low-pass filtering to form $\mathcal{D}_{\text{filt}}$, while the remaining $(1-\rho)$ is kept as $\mathcal{D}_{\text{orig}}$. The union is the mixed training set $\mathcal{D}_{\text{mix}}=\mathcal{D}_{\text{filt}}\cup\mathcal{D}_{\text{orig}}$. For each training step, two gradients are calculated: the main gradient $\mathbf{g}_{\text{main}}=\nabla_\theta\mathcal{L}_{\text{main}}(\theta;\mathcal{D}_{\text{mix}})$ computes Dual Focal Loss on the mixed set, and the calibration gradient $\mathbf{g}_{\text{calib}}=\nabla_\theta\mathcal{L}_{\text{calib}}(\theta;\mathcal{D}_{\text{orig}})$ computes Soft-ECE only on original data. When they conflict, the main gradient is projected onto the half-space orthogonal to $\mathbf{g}_{\text{calib}}$ before updating.

### Key Designs

1.  **DCT Block-wise Low-pass Filtering (Robust Feature Builder)**:
    - **Function**: Removes domain-specific high-frequency details from training samples without knowing the target domain, forcing the model to use shape and global structure for classification.
    - **Mechanism**: Converts images to YCbCr, divides each channel into $8\times 8$ non-overlapping blocks $\bm{x}_b$, performs 2D-DCT to get $\mathbf{F}_b$, and quantizes using a JPEG quantization table $\mathbf{Q}_\lambda$ with intensity parameter $\lambda$: $\mathbf{F}_b^{(q)}=\text{round}(\mathbf{F}_b/\mathbf{Q}_\lambda)$. It then applies inverse quantization and inverse transform $\hat{\bm{x}}_b=\text{DCT}^{-1}(\mathbf{F}_b^{(q)}\cdot\mathbf{Q}_\lambda)$ to reconstruct the RGB image. Smaller $\lambda\in[1,100]$ results in more aggressive filtering. Filtering only a subset of samples preserves original signals for learning fine boundaries.
    - **Design Motivation**: (a) DCT energy concentration allows low-frequency coefficients to carry main semantics while discarding high-frequency spurious textures, avoiding the global ringing effects of Fourier transforms; (b) Block-wise processing is robust to common texture distortions; (c) "Filtering only half" is a deliberate compromise—filtering all samples would cause total ID under-confidence; mixed inputs create domain-invariant pressure without destroying discriminative boundaries.

2.  **FGR Rectification (Gradient Projection, Core Innovation)**:
    - **Function**: Reconceptualizes "improving OOD calibration" and "preserving ID calibration" from weighted losses into a "main objective + hard constraint" structure, ensuring ID calibration loss does not increase in a first-order sense without manual tuning.
    - **Mechanism**: Defines the feasible half-space $\mathcal{C}_\text{ID}=\{\mathbf{g}\mid \mathbf{g}^\top\mathbf{g}_{\text{calib}}\ge 0\}$, representing all update directions that do not degrade ID calibration. If $\mathbf{g}_{\text{main}}\cdot\mathbf{g}_{\text{calib}}\ge 0$, $\mathbf{g}_{\text{main}}$ is used directly; otherwise, it performs Euclidean projection: $\mathbf{g}_\text{final}=\mathbf{g}_{\text{main}}-\frac{\mathbf{g}_{\text{main}}\cdot\mathbf{g}_{\text{calib}}}{\|\mathbf{g}_{\text{calib}}\|^2+\epsilon}\mathbf{g}_{\text{calib}}$. Proposition 4.1 proves this is the Euclidean projection of $\mathbf{g}_{\text{main}}$ onto $\mathcal{C}_\text{ID}$, ensuring $\mathcal{L}_{\text{calib}}(\theta-\eta\mathbf{g}_\text{final})\le\mathcal{L}_{\text{calib}}(\theta)+\mathcal{O}(\eta^2)$ for small steps $\eta$.
    - **Design Motivation**: Unlike "symmetric multi-task" methods like PCGrad or CAGrad, FGR's signature is asymmetry—it only modifies $\mathbf{g}_{\text{main}}$ and leaves $\mathbf{g}_{\text{calib}}$ untouched, treating ID calibration as a "red line" rather than a negotiable goal. This avoids manual OOD-vs.-ID trade-off coefficients.

3.  **Loss & Training**:
    - **Function**: The main loss $\mathcal{L}_{\text{main}}=-\sum_k y_k(1-\hat{p}_k+\hat{p}_j)^\gamma\log\hat{p}_k$ ($j$ for the highest wrong class) penalizes both overconfidence and under-confidence, which is more suitable for calibration than CE or standard Focal Loss. The constraint loss Soft-ECE uses soft temperature-based binning to create a differentiable approximation: $\mathcal{L}_{\text{calib}}=(\sum_m\frac{|S_m|}{N}|\text{acc}(S_m)-\text{conf}(S_m)|^2)^{1/2}$.
    - **Mechanism**: DFL learns a "robust predictive distribution" on the mixed set, while Soft-ECE provides a "geometric direction for ID calibration" on original data. They collaborate via the projection mechanism without weighting hyperparameters.
    - **Design Motivation**: This combination is an instance of "projection mechanism + any calibration-oriented loss." DFL is chosen for its inherent calibration potential, which yields super-linear gains when combined with the projection mechanism.

### Loss & Training
ResNet-50/110, DenseNet-121, and Wide-ResNet-26 are trained for 350 epochs. Standard training for the first 200 epochs stabilizes the decision boundary, followed by the insertion of DCT filtering and gradient projection. The WILDS dataset follows official protocols for fine-tuning ImageNet pre-trained models. Total training time increases by only 18% compared to standard training. A two-stage fine-tuning interface is also provided for incremental calibration of existing models.

## Key Experimental Results

### Main Results
Key calibration metrics under synthetic shift (CIFAR / Tiny-ImageNet -C, DenseNet-121, average of 15 corruptions × 5 severities) and real shift (WILDS):

| Dataset | Method | Acc.↑ | ECE↓ | w/ TS ECE↓ | CECE↓ | ACE↓ |
|---------|--------|-------|------|-----------|-------|------|
| CIFAR-10-C | DFL | 70.18 | 16.19 | 15.12 | 4.28 | 4.23 |
| CIFAR-10-C | MaxEnt | 71.98 | 11.62 | 13.63 | 3.62 | 3.62 |
| CIFAR-10-C | **FGR** | **75.12** | **9.02** | 9.90 | **3.12** | **3.09** |
| CIFAR-100-C | DFL | 50.17 | 9.99 | 8.82 | 0.51 | 0.49 |
| CIFAR-100-C | **FGR** | 52.66 | **8.53** | **7.57** | **0.47** | **0.46** |
| Camelyon17 (Med) | DFL | 88.03 | 2.74 | 2.12 | 9.957 | 9.956 |
| Camelyon17 | **FGR** | **89.19** | **2.36** | **1.82** | **5.714** | **5.691** |
| iWildCam (Wildlife) | **FGR** | 76.11 | 3.34 | **2.97** | 0.155 | 0.152 |
| FMoW (Remote) | **FGR** | 51.95 | **25.06** | 3.84 | **0.92** | **0.74** |

Semantic shift on Office-Home (leave-one-domain-out average):

| Method | OOD Acc.↑ | OOD ECE↓ | OOD TS-ECE↓ | OOD CECE↓ | OOD ACE↓ |
|--------|-----------|----------|-------------|-----------|----------|
| CE | 34.20 | 36.45 | 15.11 | 1.429 | 1.238 |
| DFL | 34.17 | 22.91 | 14.51 | 1.061 | 0.975 |
| BSCE-GRA | 32.55 | 21.09 | 15.29 | 1.052 | 0.991 |
| **FGR** | 34.03 | **20.41** | **13.93** | **1.018** | **0.971** |

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| Full FGR | Achieves global optimum for OOD ECE / CECE / ACE. |
| DCT Filtering Only | Improved OOD but caused ID under-confidence and ECE rebound. |
| Gradient Projection Only | Lacks OOD robustness source; performance close to DFL baseline. |
| FGR vs PCGrad (Symmetric) | FGR is superior—hard constraint vs. soft compromise. |
| FGR vs CAGrad (Symmetric) | Ditto, confirming "asymmetric projection" is key. |
| Filtering intensity $\lambda$ scan | Lower $\lambda$ improves OOD robustness but increases ID under-confidence, confirming trade-off. |

### Key Findings
- **Filtering + Projection must be paired**: Using filtering alone on Camelyon17 reduced ECE significantly but damaged ID calibration. Pairing them achieved ECE 2.36 / CECE 5.71 (a 43% reduction compared to DFL).
- **Symmetric multi-task methods fail**: PCGrad/CAGrad treat two objectives as equal compromises, allowing ID performance to degrade. FGR’s asymmetric projection locks ID performance, forcing OOD progress in the remaining feasible directions.
- **Compatible with post-hoc calibration**: FGR scores across all datasets further improved with Temperature Scaling (TS), showing it learns "feature-side" robustness rather than competing with TS.

## Highlights & Insights
- **"ID calibration as a hard constraint + geometric projection"** is the most transferable design. Any scenario involving "Main goal vs. Red-line goal" (fairness, safety, or sparsity constraints) can apply this asymmetric projection template.
- **Attributing OOD robustness to the frequency domain** provides a tangible engineering interface for "domain-invariant features." Unlike abstract invariant learning, FGR applies priors via DCT block-wise low-pass filtering, which also preserves local spatial structures better than global Fourier filtering.
- **Mixed data strategy**: Filtering only a subset of samples allows the model to see both "clean fine boundaries" and "robust coarse features," which is more effective than simple data augmentation for the "OOD gain without ID loss" objective.

## Limitations & Future Work
- **Task Scope Limitations**: Experiments focus on image classification with CNN/DenseNet backbones. Interaction between Transformer/ViT patches and DCT block sizes may not be trivial.
- **Strong Assumption: High Frequency = Spurious Shortcut**: While supported by literature, in medical imaging or fine-grained recognition, high-frequency signals might be task-relevant. Although FGR improved ID accuracy on Camelyon17, caution is needed for high-frequency discrimination tasks.
- **First-order Projection**: Proposition 4.1 only guarantees non-increasing ID calibration in an $\mathcal{O}(\eta^2)$ sense. It doesn't strictly prevent slow drift under large learning rates or long dynamics; the "insert after 200 epochs" choice mitigates this but lacks long-term stability theory.
- **Future Directions**: Replacing DCT with learnable frequency masks; extending hard constraints to multiple objectives (e.g., ID calibration + ID accuracy); combining with TTA for joint train-test calibration.

## Related Work & Insights
- **vs. Adaptive Temperature Scaling (Yu et al. 2022 / Wang et al. 2024)**: These require target domain access to train temperature regressors; FGR is target-agnostic and easier to deploy.
- **vs. Focal / MaxEnt / Dual Focal Loss**: These only regularize the loss without an explicit OOD source. FGR’s filtering simulates distribution shift during training.
- **vs. PCGrad (Yu et al. 2020) / CAGrad (Liu et al. 2021)**: These are symmetric methods; FGR promotes ID calibration to a hard constraint via asymmetric projection, removing hyperparameter tuning.
- **vs. AugMix (Hendrycks et al. 2020)**: Strong on synthetic shift but weak on WILDS real-world shift; FGR is robust on both, suggesting frequency priors are more general than pixel-level mixing.

## Rating
- Novelty: ⭐⭐⭐⭐ The "frequency filtering + hard constraint projection" combo is novel. While parts have precedents, upgrading ID to a hard constraint via asymmetric projection is a conceptual leap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers synthetic (CIFAR-C), real (Camelyon17 / iWildCam), and semantic (Office-Home) shifts, with comparisons to PCGrad / CAGrad / post-hoc TS.
- Writing Quality: ⭐⭐⭐⭐ Formulaic and geometric intuitions are clear. Proposition 4.1 clarifies the optimization semantics.
- Value: ⭐⭐⭐⭐ Addresses the deployment pain point of target-domain dependency in OOD calibration. Significant OOD improvement for only 18% extra training time is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[ICML 2026\] Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach](markov_chain_monte_carlo_without_evaluating_the_target_an_auxiliary_variable_app.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/others/measuring_uncertainty_calibration.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
