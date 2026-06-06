---
title: >-
  [Paper Note] Learning with Preserving for Continual Multitask Learning
description: >-
  [AAAI 2026][Medical Imaging][Continual Multitask Learning] This paper proposes the Learning with Preserving (LwP) framework, which maintains the geometric structure of the shared representation space via a Dynamically We…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Continual Multitask Learning"
  - "Representation Space Preservation"
  - "Distance-Preserving Loss"
  - "Catastrophic Forgetting"
  - "Replay-Free"
date: 2026-05-08
content_hash: 414e84d674bf7976
---

# Learning with Preserving for Continual Multitask Learning

**Conference**: AAAI 2026
**arXiv**: [2511.11676](https://arxiv.org/abs/2511.11676)  
**Code**: [LwP](https://github.com/AICPS-Lab/lwp)  
**Area**: Continual Learning / Multitask Learning
**Keywords**: Continual Multitask Learning, Representation Space Preservation, Distance-Preserving Loss, Catastrophic Forgetting, Replay-Free

## TL;DR

This paper proposes the Learning with Preserving (LwP) framework, which maintains the geometric structure of the shared representation space via a Dynamically Weighted Distance Preserving (DWDP) loss. Without requiring a replay buffer, LwP addresses catastrophic forgetting in Continual Multitask Learning (CMTL) and significantly outperforms existing continual learning methods on benchmarks including BDD100k, CelebA, and PhysiQ. It is the only method to surpass the single-task learning baseline.

## Background & Motivation

**Background**: In safety-critical applications such as autonomous driving and medical imaging, AI systems must continuously learn new tasks over a shared data stream (e.g., first learning traffic sign detection, then pedestrian classification). This constitutes the CMTL paradigm—sequentially learning multiple tasks over the same input domain.

**Limitations of Prior Work**: Conventional continual learning (CL) methods (EWC, SI, ER, etc.) are primarily designed for task-incremental learning and rely on parameter isolation or replay buffers to prevent forgetting. Under CMTL settings, these methods learn fragmented, task-specific features, leading to: (1) interference between task features; (2) failure to establish a unified representation beneficial to multitask learning; and (3) performance that falls below independently trained single-task baselines.

**Key Challenge**: The isolation strategies of CL fundamentally conflict with the unified representation requirements of MTL—the very mechanisms used to protect prior task knowledge (e.g., parameter freezing, replay) obstruct the formation of cross-task shared representations.

**Key Insight**: The paper shifts from protecting task outputs to preserving the geometric structure of the representation space. The key insight is that if pairwise distances between data points in the latent space are preserved, any learning problem defined on that space retains its optimal solution (proved via RKHS equivalence).

**Core Idea**: Regularize the shared representation space with a dynamically weighted pairwise distance preservation loss, so that learning new tasks does not disrupt the geometric relationships encoded by prior tasks.

## Method

### Overall Architecture

LwP adopts a shared feature extractor with task-specific heads. When learning a new task $\mathcal{T}_t$: (1) the previous model is copied and frozen as a teacher; (2) a new task head $g_{\theta_t}$ is added; (3) the current model is trained with a three-part composite loss $\mathcal{L}_{lwp} = \lambda_c \mathcal{L}_{cur} + \lambda_o \mathcal{L}_{old} + \lambda_d \mathcal{L}_{DWDP}$, comprising the current task supervised loss, the old task distillation loss, and the core DWDP geometric preservation loss.

### Key Designs

1. **Representation Space Geometric Preservation (Preservation Loss)**:

    - **Function**: Ensures that pairwise Euclidean distances between data points in the current model's latent space are consistent with those of the frozen teacher model.
    - **Mechanism**: The base preservation loss is $\mathcal{L}_{pres} = \frac{1}{N^2}\sum_{i,j}(d(\mathbf{z}_i, \mathbf{z}_j) - d(\mathbf{z}'_i, \mathbf{z}'_j))^2$, where $d$ denotes squared Euclidean distance. The paper proves that preserving pairwise distances is equivalent to preserving the Gaussian kernel Gram matrix, i.e., maintaining an isometric mapping $\phi(\mathbf{z}'_i) = T(\phi(\mathbf{z}_i))$ in the RKHS. Consequently, any learning problem defined on the old representation has an equivalently optimal solution transferable to the new representation.
    - **Design Motivation**: This is more fundamental than preserving parameters (e.g., EWC) or task outputs (e.g., LwF)—it preserves the functional equivalence of representations, allowing parameters to adapt freely without loss of information.

2. **Dynamically Weighted Distance Preserving (DWDP) Loss**:

    - **Function**: Introduces a dynamic mask for pairwise distance preservation, retaining distances only for same-class sample pairs to avoid conflicts with the classification objective.
    - **Mechanism**: A mask $m_{ij} = \mathbb{1}[y^{[t]}_i = y^{[t]}_j]$ is introduced, yielding $\mathcal{L}_{DWDP} = \frac{1}{N^2}\sum_{i,j} m_{ij}(\Delta d_{ij})^2$. Distances are preserved only for pairs with the same current task label, while inter-class pairwise distances remain free to adjust.
    - **Design Motivation**: Preserving all pairwise distances without masking restricts the model's ability to learn new discriminative features. The dynamic mask preserves intra-class structure while permitting inter-class separation; ablation experiments confirm the importance of masking.

3. **Teacher–Student Distillation**:

    - **Function**: The previous model is frozen as a teacher, and pseudo-labels are used to supervise old task heads.
    - **Mechanism**: The teacher model generates pseudo-labels $\tilde{y}_o$ for old tasks $o < t$, and the current model is trained to match these outputs.
    - **Design Motivation**: The distillation loss preserves explicit task knowledge, while DWDP preserves implicit structural knowledge; the two are complementary.

### Loss & Training

The total loss is $\mathcal{L}_{lwp} = \lambda_c \mathcal{L}_{cur} + \lambda_o \mathcal{L}_{old} + \lambda_d \mathcal{L}_{DWDP}$, where $\mathcal{L}_{cur}$ is cross-entropy for the current task, $\mathcal{L}_{old}$ is MSE distillation for old tasks, and $\mathcal{L}_{DWDP}$ is the core geometric preservation term. No replay buffer is required, making the method suitable for privacy-sensitive scenarios.

## Key Experimental Results

### Main Results (No Distribution Shift)

| Method | BDD100k (3 tasks) | CelebA (10 tasks) | PhysiQ (3 tasks) | FairFace (3 tasks) |
|--------|------------------|------------------|-----------------|-------------------|
| STL (Single-task) | 75.12 | 72.23 | 87.17 | 64.44 |
| LwF | 76.65 | 64.63 | 69.95 | 61.03 |
| oEWC | 74.87 | 69.67 | 82.64 | 63.60 |
| DER++ | 76.68 | 67.69 | 82.84 | 63.81 |
| OBC | 76.99 | 70.83 | 84.00 | 63.87 |
| **LwP (Ours)** | **78.30** | **73.48** | **88.24** | **66.48** |

### Under Distribution Shift (BDD100k)

| Method | Weather Shift | Scene Shift | Time-of-Day | Combined |
|--------|-------------|------------|-------------|----------|
| STL | 76.76 | 76.79 | 76.42 | 76.75 |
| LwF | 76.79 | 77.50 | 76.03 | 76.94 |
| SI | 75.85 | 77.89 | 74.82 | 74.57 |
| **LwP** | **77.94** | **78.20** | **77.64** | **77.77** |

### Key Findings

- LwP is the only method to **surpass the single-task baseline (STL)** across all datasets and settings, demonstrating that positive knowledge transfer in CMTL is achievable.
- All conventional CL methods (EWC, ER, SI) fall below the STL baseline under CMTL, validating that their isolation strategies are ill-suited to CMTL.
- LwP exhibits particularly strong robustness under distribution shift, as geometric preservation renders representations more stable to changes in input distribution.
- Ablation studies show that squared Euclidean distance outperforms RBF kernel distance (ablation Section 4.6), and the dynamic mask contributes approximately 1–2% performance improvement.

## Highlights & Insights

- **Deep Insight into Geometric Structure Preservation**: Via RKHS equivalence, preserving pairwise distances fully preserves the functional capacity of the representation space—a more fundamental guarantee than preserving parameters or outputs. This insight extends beyond CMTL and may inspire broader research in transfer learning and federated learning.
- **Significance of Surpassing the STL Baseline**: This demonstrates that inter-task knowledge in CMTL can transfer positively, rather than merely mitigating forgetting.
- **Replay-Free Design**: No historical data storage is required, making the method suitable for privacy-sensitive domains such as medical imaging, while still outperforming replay-based methods (e.g., ER, DER++).

## Limitations & Future Work

- **Computational Complexity of DWDP**: Pairwise distance computation scales as $O(N^2)$, which may substantially increase training overhead for large batch sizes.
- **Scalability to More Tasks**: Experiments cover at most 10 tasks (CelebA); performance on very large task sequences (e.g., 50+) remains unvalidated.
- **Mask Limited to Current Task Labels**: The DWDP mask is based on current task labels $y^{[t]}$, which may not optimally preserve inter-class relationships from prior tasks.
- **Gap Between Theory and Practice**: The RKHS equivalence theory assumes exact distance preservation, whereas practical optimization is only approximate.

## Related Work & Insights

- **vs. EWC/SI**: These methods prevent forgetting via parameter regularization but isolate task-specific knowledge. LwP instead protects the shared representation structure, allowing parameters to change freely.
- **vs. PODNet**: PODNet uniformly preserves pairwise distances of spatial features; LwP uses a dynamic mask to distinguish intra-class from inter-class pairs, avoiding optimization conflicts.
- **vs. RKD**: RKD preserves all pairwise distances for knowledge distillation without class distinction. The dynamic mask in LwP constitutes the key difference.
- The formal definition of CMTL and its experimental evidence indicate that this is a direction worthy of independent investigation, rather than a trivial subproblem of Task-IL.

## Rating

- Novelty: ⭐⭐⭐⭐ The DWDP loss and the theoretical treatment of geometric preservation are deep; the formalization of CMTL is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets + distribution shift + full ablation + comparison against 12 methods.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, problem motivation is clear, and the RKHS proof is elegant.
- Value: ⭐⭐⭐⭐ Replay-free continual learning that surpasses STL has direct practical significance for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](../../CVPR2026/medical_imaging/residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](../../NeurIPS2025/medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[AAAI 2026\] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach](radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)

</div>

<!-- RELATED:END -->
