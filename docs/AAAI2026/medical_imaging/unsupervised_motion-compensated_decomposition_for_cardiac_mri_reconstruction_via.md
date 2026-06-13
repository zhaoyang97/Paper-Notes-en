---
title: >-
  [Paper Note] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation
description: >-
  [AAAI 2026][Medical Imaging][Cardiac MRI reconstruction] This paper proposes MoCo-INR, which for the first time integrates implicit neural representation (INR) into a motion compensation (MoCo) framework. Through an unsu…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Cardiac MRI reconstruction"
  - "motion compensation"
  - "implicit neural representation"
  - "unsupervised learning"
  - "undersampled reconstruction"
date: 2026-05-08
content_hash: 7972f182c544a2f3
---

# Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation

**Conference**: AAAI 2026
**arXiv**: [2511.11436](https://arxiv.org/abs/2511.11436)  
**Code**: [MoCo-INR](https://github.com/MeijiTian/MoCo-INR)  
**Area**: Medical Imaging
**Keywords**: Cardiac MRI reconstruction, motion compensation, implicit neural representation, unsupervised learning, undersampled reconstruction

## TL;DR

This paper proposes MoCo-INR, which for the first time integrates implicit neural representation (INR) into a motion compensation (MoCo) framework. Through an unsupervised approach, it achieves high-quality dynamic reconstruction of cardiac MRI, significantly outperforming existing unsupervised methods at ultra-high acceleration factors (20× Cartesian / 69× Non-Cartesian).

## Background & Motivation

1. **Background**: Cardiac magnetic resonance (CMR) imaging is an essential tool for assessing cardiac morphology and function; however, long acquisition times make accurate imaging of dynamic cardiac motion extremely challenging. Reconstructing artifact-free dynamic MR images from undersampled k-t space data is a highly ill-posed inverse problem.
2. **Limitations of Prior Work**: Existing methods fall into two categories: (a) supervised motion-compensated methods (e.g., Pan et al. 2024) achieve strong performance but rely on fully sampled cine CMR data (requiring breath-hold acquisition), limiting clinical practicality and generalizability; (b) INR-based methods (e.g., ST-INR) enable unsupervised reconstruction but suffer from slow convergence and difficulty representing high-frequency details, while discrete feature representations from hash encoding compromise the continuity of INR.
3. **Key Challenge**: The continuous priors of unsupervised methods are insufficient for extremely undersampled inverse problems, whereas explicit motion modeling in MoCo effectively exploits spatiotemporal redundancy but existing implementations depend on supervised training and discrete interpolation.
4. **Goal**: To achieve accurate cardiac motion decomposition and high-fidelity CMR reconstruction without any training data.
5. **Key Insight**: Combining the continuous representation capacity of INR with the explicit motion modeling of motion compensation—using two INR networks to continuously parameterize the displacement vector field (DVF) and a shared canonical image, respectively.
6. **Core Idea**: Replacing discrete matrices with continuous INR functions to parameterize the deformation field and canonical image within motion compensation, combined with coarse-to-fine hash encoding and a CNN decoder to enable stable optimization and recovery of high-frequency details.

## Method

### Overall Architecture

MoCo-INR decomposes a dynamic CMR sequence into two continuous functions: (1) a deformation network $\mathcal{F}_\Phi$ that takes spatiotemporal coordinates $(p,t)$ as input and predicts the DVF $u_t(p)$; and (2) a canonical network $\mathcal{G}_\Psi$ that takes warped spatial coordinates $\tilde{p}$ as input and predicts complex-valued intensities of the canonical image. A differentiable forward model maps the predicted CMR images to k-space for comparison with acquired data, enabling joint optimization.

### Key Designs

1. **Continuous MoCo Representation (Continuous DVF + Canonical INR)**:
    - Function: Replaces conventional discrete matrix-based motion compensation representations with continuous functions.
    - Mechanism: The DVF is defined as $f: (p,t) \in \mathbb{R}^3 \mapsto u_t(p) = (\Delta x, \Delta y) \in \mathbb{R}^2$, and the canonical image is defined as $g: \tilde{p} \in \mathbb{R}^2 \mapsto x_{cano}(\tilde{p}) = a(\tilde{p}) + jb(\tilde{p}) \in \mathbb{C}$. The reconstruction proceeds as: $\tilde{p} = p + u_t(p)$, $\hat{x}_t(\tilde{p}) = \mathcal{G}_\Psi(\tilde{p})$.
    - Design Motivation: INR exhibits a spectral bias toward low-frequency continuous signals, making it naturally suited for representing smooth motion fields; continuous representation avoids the high-frequency detail loss caused by discrete interpolation.

2. **Coarse-to-Fine Hash Encoding**:
    - Function: Stabilizes DVF estimation and prevents overfitting to high-frequency artifacts.
    - Mechanism: Hash encoding maps coordinates to multi-resolution features $\gamma(p) = \gamma_1(p) \oplus \cdots \oplus \gamma_L(p)$. During training, low-frequency features ($\gamma_1$, global motion) are learned first and frozen before progressively optimizing higher-frequency features ($\gamma_2, \gamma_3, \ldots$, fine motion details).
    - Design Motivation: Global structure is more critical for motion correction; learning low frequencies before high frequencies prevents erroneous high-frequency motion estimates from interfering with global motion capture.

3. **CNN-based Decoder**:
    - Function: Replaces the conventional MLP decoder to enhance spatial continuity and resistance to overfitting.
    - Mechanism: A three-layer CNN (64 filters of size 3×3, with nonlinear activations in the first two layers) replaces the MLP, leveraging CNN's inductive bias toward local structures to better approximate the continuous functions $f$ and $g$.
    - Design Motivation: The pixel-wise mapping of MLP fails to capture spatial continuity in images; the strong fitting capacity introduced by hash encoding may cause overfitting to undersampled data and produce high-frequency artifacts, which are naturally suppressed by CNN's local receptive field.

### Loss & Training

The total loss function is:
$$\mathcal{L} = \underbrace{\|\hat{y}_t - y_t\|_1}_{\mathcal{L}_{DC}} + \mathcal{L}_{DVF}$$

where the data consistency loss $\mathcal{L}_{DC}$ minimizes the L1 distance between predicted and acquired k-space data; the DVF regularization is:
$$\mathcal{L}_{DVF} = \|u_t\|_1 + \|\nabla u_t\|_1 + \|\nabla^2 u_t\|_1$$

comprising three terms: DVF magnitude sparsity, first-order gradient smoothness, and second-order gradient smoothness, ensuring physically plausible deformation estimation.

## Key Experimental Results

### Main Results

| Sampling Mode | Acceleration | Metric | MoCo-INR | ST-INR(L&S) | TDDIP | Gain |
|---|---|---|---|---|---|---|
| VISTA (Cartesian) | 12× | PSNR/SSIM | 42.25/0.971 | 41.35/0.972 | 38.05/0.943 | +0.90 dB |
| VISTA | 20× | PSNR/SSIM | **39.53/0.957** | 36.26/0.937 | 36.58/0.929 | **+3.27 dB** |
| GA Radial | 26× | PSNR/SSIM | **40.33/0.960** | 38.85/0.956 | 34.10/0.895 | +1.48 dB |
| GA Radial | 69× | PSNR/SSIM | **37.75/0.940** | 33.92/0.910 | 33.62/0.883 | **+3.83 dB** |

The advantage is most pronounced at ultra-high acceleration factors (20×, 69×); all comparisons reach statistical significance (p<0.001).

### Ablation Study

| Configuration | PSNR | SSIM | Note |
|---|---|---|---|
| w/o $\mathcal{L}_{DVF}$ | 34.42 | 0.895 | DVF estimation fails; large performance drop |
| w/o Coarse2fine | 35.51 | 0.926 | DVF broadly reasonable but anomalies in static regions |
| Full (MoCo-INR) | **37.75** | **0.940** | Both components synergistically optimal |
| MLP decoder vs CNN decoder | MLP lower | More artifacts | CNN is more stable with fewer high-frequency artifacts |

### Key Findings

- Significant runtime advantage: In the GA Radial prospective study, MoCo-INR requires only 3.4 minutes vs. 6.7 minutes for ST-INR(L&S) vs. 19.3 minutes for TDDIP.
- The learned DVFs exhibit motion patterns consistent with cardiac biomechanics (myocardial relaxation/contraction) in both diastole and systole.
- Superior performance is also demonstrated on prospective free-breathing CMR data, validating clinical applicability.

## Highlights & Insights

- **A pioneering combination of INR and MoCo**: The integration of continuous representation with explicit motion modeling is natural and effective—INR's continuity is inherently well-suited to representing smooth motion fields, while motion compensation decomposes temporal variation into a global canonical image plus inter-frame deformation, substantially reducing problem difficulty.
- **Intuition behind the coarse-to-fine strategy**: Learning global motion before local details is consistent with how humans understand motion and aligns with the frequency bias of INR.
- **Insight on replacing MLP with CNN**: In highly undersampled inverse problems, excessively strong fitting capacity is detrimental; CNN's local inductive bias serves as an implicit regularizer.
- **Truly unsupervised**: No training data is required; the method directly optimizes from undersampled data of a single slice, eliminating dependence on fully sampled data and breath-hold acquisition.

## Limitations & Future Work

- Currently supports only 2D+t reconstruction; high-resolution 3D+t reconstruction is a future direction.
- The motion compensation assumption that all frames share the same canonical image is unsuitable for scenarios with contrast variation, such as dynamic contrast-enhanced (DCE) MRI.
- Robustness to extreme motion (e.g., arrhythmia) remains to be validated.
- Hyperparameters of the coarse-to-fine schedule (number of training steps per stage) require manual tuning.

## Related Work & Insights

- **vs. ST-INR (L&S) (Feng et al. 2025)**: ST-INR employs hash encoding with low-rank sparse constraints for unsupervised INR reconstruction but lacks explicit motion modeling, resulting in notably inferior performance compared to MoCo-INR at ultra-high acceleration factors.
- **vs. Supervised MoCo methods (Pan et al. 2024)**: Supervised methods generalize poorly to changes in sampling patterns and require fully sampled data for training; MoCo-INR is entirely unsupervised and offers greater adaptability.
- **vs. TDDIP (Yoo et al. 2021)**: DIP-based methods fail to adequately capture temporal dynamics, producing nearly identical cardiac anatomy across different phases.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of INR and MoCo is a natural yet effective innovation; the coarse-to-fine hash encoding and CNN decoder reflect insightful design choices.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Retrospective and prospective experiments, multiple sampling patterns, multiple acceleration factors, comprehensive ablation analysis, DVF visualization, and runtime comparisons.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, figures and tables are abundant, and formulas are accurate.
- Value: ⭐⭐⭐⭐⭐ Unsupervised, fast convergence, and ultra-high acceleration factors make this work highly valuable for clinical real-time cardiac MRI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[AAAI 2026\] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)

</div>

<!-- RELATED:END -->
