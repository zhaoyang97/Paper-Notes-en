---
title: >-
  [Paper Note] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging
description: >-
  [NeurIPS 2025][Medical Imaging][Continual learning] This paper proposes an exemplar-free continual learning framework that combines class-conditional DDPM diffusion replay with Elastic Weight Consolidation (EWC)…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Continual learning"
  - "diffusion replay"
  - "EWC"
  - "exemplar-free"
  - "privacy preservation"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: c043352593ea57f8
---

# EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging

**Conference**: NeurIPS 2025
**arXiv**: [2509.23906](https://arxiv.org/abs/2509.23906)
**Code**: To be confirmed
**Area**: Medical Imaging / Continual Learning
**Keywords**: Continual learning, diffusion replay, EWC, exemplar-free, medical imaging, privacy preservation, catastrophic forgetting

## TL;DR

This paper proposes an exemplar-free continual learning framework that combines class-conditional DDPM diffusion replay with Elastic Weight Consolidation (EWC), achieving an AUROC of 0.851 on MedMNIST v2 (8 tasks across 2D/3D) and CheXpert, reducing forgetting by over 30% compared to DER++, approaching the joint training upper bound (0.869), while requiring no storage of original patient data.

## Background & Motivation

**Continual adaptation requirements for medical AI**: Deployed foundation models must continuously adapt to new diseases, imaging protocols, and workflows, making full retraining costly and impractical.

**Data non-storability under privacy constraints**: Strict patient data privacy regulations (HIPAA/GDPR) in the medical domain severely limit the storage and replay of raw samples, rendering traditional exemplar-based replay methods (e.g., DER++, SPM) infeasible in real clinical settings.

**Catastrophic forgetting**: Sequential learning of new tasks severely erodes memory of previously learned tasks, especially in medical imaging where distribution gaps across modalities and disease types are substantial.

**Limitations of existing approaches**: Regularization methods (EWC, EFT) degrade under distribution shift; generative replay methods (VAE, GAN) fail to capture fine-grained medical texture details; dynamic expansion methods (PMoE, CoPE) incur high computational overhead.

**Generative quality advantage of diffusion models**: DDPMs have surpassed GANs in image synthesis quality and can more faithfully reconstruct fine-grained medical image structures, yet they have not been systematically applied to continual learning scenarios.

**Absence of theoretical analysis**: Existing continual learning methods lack a theoretical framework that decomposes forgetting into measurable factors, making it difficult to diagnose the root causes of forgetting.

## Method

### Overall Architecture

The framework integrates three core components, inspired by the Complementary Learning Systems (CLS) dual-memory theory: the DDPM serves as fast recall (analogous to the hippocampus), while EWC performs gradual consolidation (analogous to the neocortex).

### Component 1: Class-Conditional DDPM Replay

A class-conditional DDPM $p_k(x|y)$ is trained for each task $\mathcal{T}_k$, generating synthetic samples from previous tasks via the reverse diffusion process:

- Forward process: $q(x_{1:T}|x_0) = \prod_{t=1}^T \mathcal{N}(x_t; \sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)\mathbf{I})$
- Training objective: minimize noise prediction error $\|\epsilon - \epsilon_\phi(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, y)\|^2$
- $T=1000$ steps, cosine $\beta_t$ schedule, 200 epochs per task
- 256 class-balanced synthetic samples per task are stored in the replay buffer, with a total budget of 100MB

### Component 2: Lightweight ViT Classifier

A lightweight Vision Transformer (patch size 16, 6 layers, 8 heads, hidden dimension 512) is adopted with ImageNet pre-trained initialization. Images are split into patches, projected linearly with positional encodings, processed through Transformer layers, and classified via an MLP head applied to the [CLS] token.

### Component 3: EWC Regularization

Important parameter drift is penalized via the Fisher information matrix $F_i$: $\Omega_{\text{EWC}} = \sum_i F_i(\theta_i - \theta_i^*)^2$. Fisher information is estimated using 500 samples per class, with $\lambda \in \{10, 50, 100\}$ tuned on the validation set.

### Joint Optimization Objective

$$\mathcal{L}_{\text{total}}^{(k)} = \mathbb{E}_{(x,y) \sim \mathcal{D}_k \cup \hat{\mathcal{D}}_{<k}} [\mathcal{L}_{\text{CE}}(f_\theta(x), y)] + \lambda \sum_i F_i(\theta_i - \theta_{i,<k}^*)^2$$

Each batch mixes real and replayed data at a 1:1 ratio. Training uses AdamW (lr $3 \times 10^{-4}$, weight decay 0.01), batch size 64, with results averaged over 5 seeds.

### Theoretical Forgetting Bound

Forgetting is decomposed into two measurable sources, yielding a unified upper bound:

$$\bar{F} \leq \alpha \cdot D_{\text{KL}}(p_j \| \hat{p}_j) + \beta \sum_i F_i(\theta_i - \theta_i^*)^2$$

- **First term (distribution shift)**: KL divergence between replayed and real distributions, derived via Pinsker's inequality.
- **Second term (parameter shift)**: Fisher-weighted parameter deviation, derived via second-order Taylor expansion.
- This bound directly maps to the method design: diffusion replay reduces the KL term, while EWC constrains the Fisher-weighted drift term.

## Key Experimental Results

### Table 1: Continual Learning Main Results (mean over 5 runs)

| Method | MedMNIST-2D Acc↑ | Forgetting↓ | AUC↑ | CheXpert Acc↑ | Forgetting↓ | AUC↑ |
|--------|-------------------|-------------|------|---------------|-------------|------|
| Finetune | 67.4 | 27.5 | 0.820 | 64.8 | 26.9 | 0.802 |
| EWC | 72.9 | 19.7 | 0.842 | 70.5 | 19.4 | 0.824 |
| DER++ | 75.6 | 14.2 | 0.853 | 73.2 | 13.8 | 0.838 |
| VAE+Replay | 74.2 | 15.6 | 0.851 | 71.7 | 15.1 | 0.833 |
| **Ours (DDPM+EWC)** | **78.1** | **10.5** | **0.866** | **76.4** | **10.9** | **0.851** |
| Joint (upper bound) | 81.4 | 0.0 | 0.879 | 79.1 | 0.0 | 0.869 |

### Table 2: Ablation Study

| Variant | MedMNIST-2D Acc↑ | Forgetting↓ | CheXpert Acc↑ | Forgetting↓ |
|---------|-------------------|-------------|---------------|-------------|
| Full model (DDPM+EWC) | 72.8 | 11.3 | 68.5 | 13.7 |
| w/o DDPM (EWC only) | 67.0 | 17.1 | 62.3 | 21.1 |
| w/o EWC (DDPM only) | 69.2 | 14.5 | 64.8 | 18.9 |

### Key Findings

- **30%+ forgetting reduction on CheXpert**: Compared to DER++ (forgetting 13.8), the proposed method achieves forgetting of only 10.9, further reducing forgetting beyond DER++ without storing any patient data.
- **Near joint training upper bound**: CheXpert AUC reaches 0.851 vs. 0.869 for joint training, a gap of only 0.018.
- **Complementarity of two components**: Removing DDPM causes CheXpert forgetting to surge from 13.7 to 21.1 (+54%); removing EWC raises forgetting to 18.9 (+38%), validating the complementarity of replay and regularization.
- **Superior retention on early tasks**: In task-level analysis on CheXpert, T1 accuracy reaches 65.7%, far surpassing Finetune (43.8%) and DER++ (61.5%).

## Highlights & Insights

- **Closed loop between theory, method, and experiment**: The forgetting upper bound directly links design decisions to observable quantities (KL divergence, Fisher drift), serving both as a design guide and a diagnostic tool; regression analysis confirms both terms are positively correlated with forgetting.
- **Practical privacy preservation**: Original patient data is entirely avoided within a 100MB memory budget, satisfying HIPAA/GDPR compliance requirements critical for clinical deployment.
- **Unified 2D/3D handling**: A single diffusion model simultaneously handles 2D (6 MedMNIST tasks) and 3D (OrganMNIST3D, NoduleMNIST3D) medical images, demonstrating modality-agnostic generality.
- **Inspiration from dual-memory theory**: The framework is grounded in the cognitive science Complementary Learning Systems theory; the design intuition of DDPM fast recall + EWC gradual consolidation is conceptually compelling.

## Limitations & Future Work

- **Diffusion training overhead**: Training DDPM for 200 epochs per task incurs substantial computational cost when the number of tasks or image resolution is large; acceleration strategies such as generator distillation are not discussed.
- **Fixed task order**: Main experiments use a fixed task sequence; although appendix analyses address order robustness, validation in online/streaming scenarios is absent.
- **Calibration and fairness underexplored**: While the paper mentions calibration and fairness as future directions, current experiments do not evaluate performance disparities across subgroups.
- **Limited replay sample count**: Only 256 synthetic samples per task are generated; for tasks with large label spaces (e.g., CheXpert with 14 labels), the per-class sample count is small and may hinder long-sequence learning.
- **Lack of comparison with recent diffusion-based continual learning methods**: No comparison is made with other works from 2024–2025 that apply diffusion models to continual learning.

## Related Work & Insights

| Dimension | Ours (DDPM+EWC) | DER++ (Buzzega et al., 2020) | VAE+Replay (Shin et al., 2017) |
|-----------|-----------------|------------------------------|-------------------------------|
| Stores raw data | No (privacy-safe) | Yes (requires exemplars) | No |
| Forgetting (CheXpert) | **10.9** | 13.8 | 15.1 |
| Generation quality | High (diffusion) | N/A | Low (VAE blurriness) |
| AUC (CheXpert) | **0.851** | 0.838 | 0.833 |

| Dimension | Ours | PMoE (Jung & Kim, 2024) | EWC (Kirkpatrick et al., 2017) |
|-----------|------|-------------------------|-------------------------------|
| Strategy | Generative replay + regularization | Dynamic expansion | Regularization only |
| Computational cost | Moderate (DDPM training) | High (expert network expansion) | Low |
| Forgetting (2D) | **10.5** | 15.3 | 19.7 |
| Theoretical support | Forgetting bound decomposition | None | Fisher information theory |

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of diffusion replay and EWC is not entirely novel in concept, but its systematic integration in the medical imaging setting is innovative; the theoretical forgetting bound is a highlight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Nine tasks across three datasets, full ablations, 5-seed averaging, and task-level analysis provide comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, method components are well-decoupled, and experimental presentation is rigorous.
- **Value**: ⭐⭐⭐⭐ Privacy-preserving continual learning addresses a clear and practically important clinical need in medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[NeurIPS 2025\] DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](dison_decentralized_isolation_networks_for_out-of-distribution_detection_in_medi.md)
- [\[AAAI 2026\] ConSurv: Multimodal Continual Learning for Survival Analysis](../../AAAI2026/medical_imaging/consurv_multimodal_continual_learning_for_survival_analysis.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](../../AAAI2026/medical_imaging/learning_with_preserving_for_continual_multitask_learning.md)

</div>

<!-- RELATED:END -->
