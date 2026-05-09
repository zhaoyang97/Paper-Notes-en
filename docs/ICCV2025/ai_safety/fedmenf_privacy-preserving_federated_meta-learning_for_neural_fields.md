---
title: >-
  [Paper Note] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields
description: >-
  [ICCV 2025][AI Safety][Federated Learning] This paper is the first to study federated meta-learning for Neural Fields (NFs) under private data settings. It reveals the severe privacy leakage mechanisms of existing federated meta-learning methods on neural field tasks, and proposes FedMeNF, which regularizes private information in local meta-gradients via a privacy-preserving loss function, effectively protecting client data privacy while retaining fast adaptation capability.
tags:
  - ICCV 2025
  - AI Safety
  - Federated Learning
  - Meta-Learning
  - Neural Fields
  - Privacy Preservation
  - Implicit Neural Representations
date: 2026-05-08
content_hash: 4defb57754474185
---

# FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields

**Conference**: ICCV 2025
**arXiv**: [2508.06301](https://arxiv.org/abs/2508.06301)
**Code**: [GitHub](https://github.com/junhyeog/FedMeNF)
**Area**: AI Security
**Keywords**: Federated Learning, Meta-Learning, Neural Fields, Privacy Preservation, Implicit Neural Representations

## TL;DR
This paper is the first to study federated meta-learning for Neural Fields (NFs) under private data settings. It reveals the severe privacy leakage mechanisms of existing federated meta-learning methods on neural field tasks, and proposes FedMeNF, which regularizes private information in local meta-gradients via a privacy-preserving loss function, effectively protecting client data privacy while retaining fast adaptation capability.

## Background & Motivation
**Neural Fields (NFs)** approximate continuous signals (e.g., images, video, 3D scenes) using deep networks, offering advantages such as memory efficiency and multi-modal support. However, optimizing NFs typically requires large amounts of data and computation, posing a challenge for resource-constrained edge devices.

**Practical scenario**: A user captures a few photos of an object with a smartphone and wishes to obtain a high-quality 3D model quickly. Training an NF from scratch is infeasible due to data scarcity and computational cost.

**Natural solution**: **Federated Meta-Learning (FML)** — multiple users collaboratively train a global meta-learner capable of rapidly adapting to new tasks.

**Key Challenge**: Traditional federated learning's privacy guarantee relies on the assumption that "sharing model parameters rather than raw data is safe." In the neural field setting, this assumption **completely breaks down** for two reasons:

Each client typically has only a **single task instance** (e.g., one's own car or face), causing local meta-optimization to degenerate into standard supervised training, where the local meta-learner becomes an NF optimized on private data.

**NFs are inherently compact representations of data**, directly encoding raw data information. A malicious server can reconstruct private data from the shared NF parameters.

**Key Insight**: The paper defines a quantifiable privacy metric $\text{PSNR}_p$, theoretically analyzes the mathematical mechanism of privacy leakage (originating from the $g_K$ term in the meta-gradient), and designs a regularization method to eliminate this term.

## Method

### Overall Architecture
FedMeNF follows the standard federated meta-learning pipeline: the server distributes the global meta-learner $\theta$ → clients perform local meta-optimization (inner loop trains the NF + outer loop updates the meta-learner) → clients upload local meta-learners → the server aggregates. The key innovation lies in introducing a privacy-preserving loss function into the local meta-optimization.

### Key Designs
1. **Privacy Metric $\text{PSNR}_p$**:

    - **Function**: Quantifies the degree of privacy leakage in federated meta-learning.
    - **Mechanism**: $\text{PSNR}_p = \text{PSNR}(Q^m, f_w(\text{Coord}(Q^m)))$, measuring how well the server can reconstruct the client's query set $Q^m$ using the shared local meta-learner $w$. A higher $\text{PSNR}_p$ indicates more severe privacy leakage.
    - **Design Motivation**: PSNR is applicable to diverse signal modalities (visual, audio, sensor) and serves as the standard metric for both FL reconstruction attacks and NF reconstruction quality. $\text{SSIM}_p$ and $\text{LPIPS}_p$ are additionally defined as auxiliary indicators.

2. **Theoretical Analysis of Privacy Leakage**:

    - **Function**: Explains why $\text{PSNR}_p$ persistently increases under existing FML methods.
    - **Mechanism**: Through first-order approximation analysis of the meta-gradient (Propositions 1 and 2), the paper shows that:
        - The meta-gradient satisfies $g_M \approx g_K - \lambda_i \mathcal{I}_K$, where $g_K$ is the gradient on the query set and $\mathcal{I}_K$ is an inner-product term.
        - At each outer-loop iteration, the loss change $\Delta L_{i+1} \approx -\lambda_o \cdot g_K^2 \leq 0$, meaning the meta-learner $w$ monotonically reduces its loss on the query set.
        - Since $\text{PSNR}_p$ is inversely related to the loss, $\text{PSNR}_p$ increases monotonically — leading to continuous privacy leakage.
    - **Design Motivation**: This precisely locates the mathematical source of privacy leakage (the $g_K$ term), providing a theoretical basis for regularization.

3. **Privacy-Preserving Loss Function $L_{pp}$**:

    - **Function**: Regularizes the $g_K$ term in the meta-gradient to prevent $\text{PSNR}_p$ from increasing.
    - **Mechanism**:
    $L_{pp}(\gamma, w_i, \varphi_K, B_K) = L(\varphi_K, B_K) - \gamma L(w_i, B_K)$
    The corresponding privacy-preserving meta-gradient becomes $g_{pp} \approx (1-\gamma) \cdot g_K - \lambda_i \mathcal{I}_K$, and the loss change reduces to $\Delta L_{i+1} \approx -\lambda_o(1-\gamma)(g_K)^2$.
    - When $\gamma = 1$, the $g_K$ term is fully eliminated; the meta-learner no longer memorizes private data and instead learns only a fast optimization strategy (gradient alignment).
    - **Design Motivation**: $g_K$ is the direct cause of the meta-learner converging toward the optimal NF (i.e., memorizing data); subtracting $\gamma L(w_i, B_K)$ precisely controls the weight of this term.

4. **Adaptive Privacy Budget $\zeta$**:

    - **Function**: Allows clients to dynamically adjust $\gamma$ according to their privacy requirements.
    - **Mechanism**: Analogous to $\epsilon$ in differential privacy, the privacy budget $\zeta$ bounds the total loss change:
    $|\Delta L_{i+1}| \cdot R \cdot E \cdot M/N \leq \zeta$
    $\gamma = \min(\max(1 - N\zeta / (REM\lambda_o(g_K)^2), 0), 1)$
    - **Design Motivation**: A fixed $\gamma$ may be overly strict or permissive; the adaptive approach dynamically balances privacy and performance based on the current gradient magnitude.

### Loss & Training
- Inner loop (training the NF): $\varphi_{k+1} \leftarrow \varphi_k - \lambda_i \nabla_{\varphi_k} L(\varphi_k, B_k)$ (standard SGD)
- Outer loop (updating the meta-learner): $w_{i+1} \leftarrow w_i - \lambda_o \nabla_{w_i} L_{pp}$ (with privacy regularization)
- Server aggregation: $\theta_{r+1} \leftarrow \sum_m \alpha^m w_*^m$ (FedAvg or FedProx)
- First-order approximation (FOMAML-style) is adopted to reduce computational overhead.

## Key Experimental Results

### Main Results (Multi-modal, Multi-dataset — FedAvg Aggregation)

| Method | PetFace (Image) PSNR↑ | PetFace PSNRp↓ | Δ(↑) | Cars (3D) PSNR↑ | Cars PSNRp↓ | Δ(↑) |
|--------|-----------------------|----------------|------|-----------------|-------------|------|
| Local | 22.29 | - | - | 17.13 | - | - |
| MAML | 27.39 | 16.57 | 10.82 | 23.08 | 19.73 | 3.35 |
| FOMAML | 23.15 | 18.52 | 4.63 | 23.66 | 19.73 | 3.93 |
| Reptile | 22.52 | 17.39 | 5.13 | 21.98 | 19.96 | 2.02 |
| meta-NSGD | 5.15 | 12.49 | -7.34 | 10.62 | 6.85 | 3.77 |
| **FedMeNF** | **27.00** | **14.77** | **12.23** | **24.05** | **12.15** | **11.90** |

$\Delta = \text{PSNR} - \text{PSNR}_p$; higher values indicate better reconstruction quality with lower privacy leakage (ideal behavior).

### Ablation Study

| Configuration | PSNR | PSNRp | Δ | Notes |
|---------------|------|-------|---|-------|
| γ = 0 (no privacy protection) | 27.39 | 16.57 | 10.82 | Equivalent to MAML |
| γ = 0.25 | 27.28 | 15.89 | 11.39 | Light protection |
| γ = 0.50 | 27.15 | 15.21 | 11.94 | Moderate protection |
| γ = 0.75 | 27.00 | 14.77 | 12.23 | Recommended setting |
| γ = 1.0 | 26.42 | 13.85 | 12.57 | Strongest protection, slight performance drop |
| Adaptive γ (ζ-controlled) | 27.05 | 14.50 | 12.55 | Dynamic balance |

### Key Findings
- **MAML exhibits the most severe privacy leakage**: Although reconstruction quality is high, $\text{PSNR}_p$ is also high, allowing the server to reconstruct private data from shared parameters.
- **meta-NSGD falls into the opposite extreme**: Privacy leakage is low, but reconstruction quality falls even below the Local (non-federated) baseline, negating the benefit of meta-learning.
- **FedMeNF achieves the best $\Delta$ across all modalities**: consistently effective on images, video, and 3D (NeRF).
- FedMeNF demonstrates superior robustness over baselines under few-shot (2-shot) and non-IID data distributions.
- $\gamma \approx 0.75$ provides the best privacy–performance trade-off.

## Highlights & Insights
- **First disclosure of privacy vulnerabilities in NF + FML**: NFs, as compact data representations, are inherently at odds with the FL assumption that "parameter sharing is safe" — a significant and practically important insight.
- **Concise theoretical analysis**: The paper precisely identifies $g_K$ as the source of privacy leakage through meta-gradient decomposition, with a clear and succinct derivation.
- **Minimalist solution**: $L_{pp}$ merely subtracts a regularization term from the standard meta-loss — extremely simple to implement yet theoretically grounded.
- **Multi-modal validation**: Experiments span four datasets covering images (PetFace), video (GolfDB), and 3D (Cars, FaceScape), demonstrating the method's generality.
- **$\Delta = \text{PSNR} - \text{PSNR}_p$ as an evaluation metric**: elegantly captures both task performance and privacy protection in a single measure.

## Limitations & Future Work
- $\text{PSNR}_p$ is based on pixel-level MSE and may fail to capture semantic-level privacy leakage.
- The theoretical analysis relies on first-order approximations; the influence of higher-order terms is not quantified.
- The optimal choice of $\gamma$ may vary across data modalities and tasks, lacking theoretical guidance for automatic selection.
- Stronger adversaries (e.g., those exploiting parameter differences across multiple rounds to infer private information) are not considered.
- Experiments involve only one or a few task instances per client; larger-scale settings remain unvalidated.
- Comparison with differential privacy (DP) methods is insufficient.

## Related Work & Insights
- This work extends privacy protection from conventional FL (large datasets + classification) to NFs (small data + signal reconstruction), opening a new research direction.
- The design philosophy of $L_{pp}$ — subtracting the leakage-inducing term from the gradient — may generalize to privacy preservation in other settings.
- The improved non-IID robustness suggests that reducing data-specific information in gradients can actually benefit federated aggregation.
- The work provides a privacy-safe training paradigm for NF applications on edge devices (e.g., personalized 3D scanning, facial animation).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First study of privacy in NF + FL; insight is deep and the solution is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-modal validation with thorough ablations, though adversarial attack testing is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear, though the heavy notation and algorithm descriptions could be further streamlined.
- **Value**: ⭐⭐⭐⭐⭐ — Identifies an important privacy vulnerability and provides an effective remedy; significant impact for the NF + FL community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](../../CVPR2026/ai_safety/fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_double-blind_federated_adaptation_of_foundation_models.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)

<!-- RELATED:END -->
