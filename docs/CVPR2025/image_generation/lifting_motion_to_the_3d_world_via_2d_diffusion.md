---
title: >-
  [Paper Note] Lifting Motion to the 3D World via 2D Diffusion
description: >-
  [CVPR 2025][Image Generation][3D Motion Estimation] MVLift proposes a multi-stage framework trained solely on single-view 2D pose sequences. It establishes multi-view consistency through a progressive strategy (line-conditioned diffusion model $\rightarrow$ multi-view optimization $\rightarrow$ synthetic data generation $\rightarrow$ multi-view diffusion model) to achieve global 3D motion (including joint rotation and root trajectory) estimation without 3D supervision. It out…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "3D Motion Estimation"
  - "2D Diffusion Model"
  - "Multi-View Consistency"
  - "Global Trajectory"
  - "No 3D Supervision"
date: 2026-05-08
content_hash: 624d7829328f20c5
---

# Lifting Motion to the 3D World via 2D Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2411.18808](https://arxiv.org/abs/2411.18808)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: 3D Motion Estimation, 2D Diffusion Model, Multi-View Consistency, Global Trajectory, No 3D Supervision

## TL;DR

MVLift proposes a multi-stage framework trained solely on single-view 2D pose sequences. It establishes multi-view consistency through a progressive strategy (line-conditioned diffusion model $\rightarrow$ multi-view optimization $\rightarrow$ synthetic data generation $\rightarrow$ multi-view diffusion model) to achieve global 3D motion (including joint rotation and root trajectory) estimation without 3D supervision. It outperforms the 3D-supervised WHAM (164.3mm) with a root trajectory error of 67.6mm on AIST++.

## Background & Motivation

**Background**: Estimating 3D motion from 2D observations is a long-standing challenge. SOTA methods like MotionBERT and WHAM require training on motion capture datasets containing 3D ground truth (GT), which limits their generalization to the motion types covered by the training data. Some approaches attempt training solely with 2D poses (e.g., ElePose, MAS), but ElePose only processes single frames and is unstable, while MAS cannot predict the global root trajectory.

**Limitations of Prior Work**: (1) Methods relying on 3D GT training cannot generalize to out-of-distribution motions (e.g., complex dances, gymnastics, animal movements) where mocap data is difficult to acquire; (2) Methods independent of 3D data cannot predict the global root trajectory in the world coordinate system, only estimating local poses; (3) Single-view 2D-to-3D lifting suffers from severe depth ambiguity, making direct optimization prone to sudden changes.

**Key Challenge**: When only single-view 2D data is available, there is a lack of direct 3D supervision or multi-view consistency constraints to resolve depth ambiguity, while global 3D motion estimation (including root trajectory in the world coordinate system) requires accurate depth and scale information.

**Goal**: Estimate global 3D motion including joint rotations and root trajectory in the world coordinate system, using only domain-specific single-view 2D pose sequences (e.g., human, animal, or interaction).

**Key Insight**: The authors observe that while single-view 2D sequences provide limited 3D information, diffusion models trained on large and diverse 2D motions can learn rich priors about pose appearances from different perspectives. The learned 2D diffusion priors coupled with geometric constraints can be leveraged to progressively establish multi-view consistency, thereby recovering 3D motion.

**Core Idea**: By combining priors learned from a 2D motion diffusion model with epipolar geometry constraints, multi-view consistent 2D pose sequences are progressively generated, allowing global 3D motion to be recovered without 3D annotations.

## Method

### Overall Architecture

A four-stage progressive framework. **Stage 1**: Train a line-conditioned 2D motion diffusion model—inputting simulated epipolar line constraints to generate 2D pose sequences aligned with these epipolar lines, establishing basic pairwise geometric consistency. **Stage 2**: Multi-view 2D motion optimization—jointly optimizing 2D sequences from 5 unobserved views, utilizing Score Distillation Sampling (SDS) to guarantee motion realism and a multi-view consistency loss to preserve geometric relationships. **Stage 3**: Synthetic data generation—recovering 3D motion from the multi-view 2D sequences of Stage 2 and reprojecting them to multiple views to obtain a strictly consistent multi-view 2D dataset. **Stage 4**: Train a multi-view 2D motion diffusion model—training on the synthetic data to efficiently generate multi-view consistent 2D sequences in a single forward pass for final 3D motion reconstruction.

### Key Designs

1. **Line-Conditioned 2D Motion Diffusion**:

    - **Function**: Generate 2D pose sequences that satisfy epipolar constraints, establishing basic cross-view geometric consistency.
    - **Mechanism**: Define a line condition $\mathbf{L} \in \mathbb{R}^{T \times J \times 3}$, where the line for each joint is parameterized by coefficients $(a_t^j, b_t^j, c_t^j)$. During training, epipolar line constraints are simulated by randomly sampling virtual epipoles (without requiring real multi-view data), and a Line Matching Loss $\mathcal{L}_{\text{line}} = \sum_{t,j} |a_t^j \hat{x}_t^j + b_t^j \hat{y}_t^j + c_t^j|$ is added to ensure that predicted joints fall on their corresponding epipolar lines. A Transformer architecture is used, where the line condition is concatenated with noisy pose features for processing.
    - **Design Motivation**: Epipolar geometry provides the most fundamental cross-view constraints. By randomly simulating epipolar lines during training, the model learns to generate realistic 2D motion under geometric constraints.

2. **Multi-View Optimization**:

    - **Function**: Upgrade from pairwise consistency to global multi-view consistency.
    - **Mechanism**: Set up 6 camera views (arranged horizontally in a ring at 60° intervals) and jointly optimize the 2D sequences of 5 unobserved views. Two optimization objectives are employed: (1) SDS (Score Distillation Sampling) to ensure each view's 2D sequence conforms to the learned line-conditioned diffusion prior distribution; (2) A multi-view consistency loss traversing all $\binom{6}{2}=15$ view pairs, constraining the epipolar distances between each pair to zero. The SDS gradient is calculated as $\nabla \mathcal{L}_{\text{SDS}} = \mathbb{E}[\omega(n)(\epsilon_\theta - \epsilon)]$.
    - **Design Motivation**: The epipolar lines in Stage 1 only constrain pairwise consistency and do not guarantee global consistency. Joint optimization, which enforces geometric constraints on all view pairs along with SDS to maintain motion realism, significantly improves multi-view consistency.

3. **Synthetic Data Generation & Multi-View Diffusion**:

    - **Function**: Convert the optimization-based method into an efficient feedforward generative model.
    - **Mechanism**: Stage 3 reconstructs 3D joint positions from the optimized multi-view 2D sequences (by minimizing reprojection error) and fits SMPL parameters (using VPoser), then reprojects them to 4 views spaced at 90° intervals to obtain strictly consistent multi-view data. Stage 4 trains a multi-view diffusion model on this data: taking a single-view 2D sequence as input, it simultaneously generates 2D sequences for 3 other views. The network incorporates a cross-view attention layer after the self-attention in each Transformer block to exchange information between views.
    - **Design Motivation**: Optimization is slow and cannot guarantee perfect consistency. Through a self-training pipeline of "optimization $\rightarrow$ synthetic data $\rightarrow$ training a generative model", the slow optimization process is distilled into highly efficient feedforward inference.

### Loss & Training

Stage 1: $\mathcal{L} = \mathbb{E}[\|\hat{X}_\theta - X_0\|_1] + \mathcal{L}_{\text{line}}$ (denoising reconstruction + line matching). Stage 2: SDS loss + multi-view consistency loss. Stage 4: Denoising reconstruction loss of the multi-view diffusion model. 3D recovery is performed by minimizing reprojection error + VPoser SMPL fitting.

## Key Experimental Results

### Main Results

**AIST++ Dataset (with 3D GT):**

| Method | Requires 3D Supervision | $T_{\text{root}}$↓ | MPJPE↓ | PA-MPJPE↓ |
|------|:---:|---:|---:|---:|
| MotionBERT | ✓ | 101.6 | 134.0 | 108.6 |
| WHAM| ✓ | 164.3 | 104.8 | 75.1 |
| ElePose | ✗ | N/A | 269.4 | 215.1 |
| MAS | ✗ | N/A | 191.1 | 155.6 |
| SMPLify | ✗ | 77.4 | 171.6 | 146.7 |
| **MVLift** | **✗** | **67.6** | **110.7** | **79.2** |

**OMOMO Human-Object Interaction Dataset:**

| Method | $T_{\text{root}}$↓ | MPJPE↓ | $T_{\text{root}}^O$↓ | O-MPJPE↓ |
|------|---:|---:|---:|---:|
| SMPLify | 97.9 | 142.0 | 751.8 | 106.7 |
| **MVLift** | **54.9** | **67.0** | **172.9** | **76.9** |

### Ablation Study

| Configuration | $T_{\text{root}}$↓ | MPJPE↓ | PA-MPJPE↓ | $J_{2D}$↓ |
|------|---:|---:|---:|---:|
| MVLift-Stage 1 | 73.1 | 135.2 | 104.4 | 31.0 |
| MVLift-Stage 2 | 65.3 | 127.4 | 96.2 | 19.7 |
| SDS for 3D | 72.9 | 137.3 | 103.5 | 25.2 |
| SDS for 3D w/o $l_{\text{epi}}$ | 752.3 | 230.4 | 186.2 | 54.9 |
| **MVLift (final)** | **67.6** | **110.7** | **79.2** | **14.0** |

### Key Findings

- Each stage brings significant improvement: Stage 1 $\rightarrow$ Stage 2 MPJPE drops from 135.2 to 127.4, and the final model drops to 110.7.
- Direct SDS optimization in 3D without line conditions (no epipolar constraints) completely collapses (MPJPE 230.4), proving that epipolar constraints are key to success.
- MVLift even outperforms the 3D-supervised WHAM (164.3mm) and MotionBERT (101.6mm) in root trajectory accuracy (67.6mm).
- In human perception experiments, 39% of the participants rated the results generated by MVLift as better than the GT, and 11% found them indistinguishable.
- The method successfully generalizes to animal poses (CatPlay) and human-object interactions (OMOMO), demonstrating broad domain generalizability.

## Highlights & Insights

- **The four-stage design of "progressively establishing multi-view consistency" is highly ingenious**—moving from pairwise epipolar constraints (Stage 1) $\rightarrow$ global multi-view optimization (Stage 2) $\rightarrow$ synthesizing strictly consistent data (Stage 3) $\rightarrow$ efficient feedforward generation (Stage 4). Each stage addresses the limitations of the previous, ultimately distilling slow optimization into fast inference. This progressive self-training strategy can be widely applied to tasks lacking direct supervision signals.
- **Trained solely on 2D data but outperforming 3D-supervised methods**—surpassing WHAM and MotionBERT in root trajectory metrics, which suggests that multi-view consistency constraints can resolve depth ambiguity more effectively than direct 3D regression.
- **Strong domain generalizability**—the same framework is applicable to humans, animals, and human-object interactions, requiring only domain-specific 2D pose sequences and no skeleton or template priors (SMAL for animals / SMPL for humans are only utilized in post-processing).

## Limitations & Future Work

- The four-stage training pipeline is complex, and the optimization in Stage 2 is particularly time-consuming.
- The optimization results in Stage 2 cannot be guaranteed to perfectly align with the input 2D sequence, only ensuring motion realism.
- The quality of the 3D motion in the synthetic data directly limits the performance upper bound of the Stage 4 model.
- Joint visibility under severe occlusion remains unaddressed.
- The camera arrangement strategy (equispaced circular ring) is relatively fixed, whereas actual scenarios may feature more complex camera distributions.
- Exploring end-to-end training to replace the multi-stage pipeline could reduce error accumulation.

## Related Work & Insights

- **vs WHAM**: WHAM trains its global trajectory prediction module on 3D datasets like AMASS, showing poor generalization to out-of-distribution motions. MVLift, without 3D data, outperforms it by 2.4x in root trajectory accuracy, although MPJPE is slightly inferior (110.7 vs 104.8).
- **vs MAS**: MAS optimizes 3D motion using an unconditional 2D motion diffusion model, but cannot generate consistent multi-view sequences and does not support global trajectories. MVLift resolves both issues via line-conditioning and multi-view consistency constraints.
- **vs ElePose**: ElePose simultaneously predicts 3D poses and cameras using projection loss + a 2D pose prior, but training is unstable and often produces unrealistic poses. MVLift constrains generation quality much better through diffusion model priors.
- **Insights**: The concept of progressive establishment of multi-view consistency can be transferred to other tasks like 3D object reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to achieve full global 3D motion estimation without 3D supervision, with an ingeniously designed multi-stage progressive strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Tested on 5 datasets across 3 domains (human/animal/interaction), including human perception experiments and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The four-stage framework is clear, though it contains many details that require careful reading.
- Value: ⭐⭐⭐⭐⭐ Tackles the long-standing "2D-to-3D motion lifting without 3D supervision" problem, offering broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Move-in-2D: 2D-Conditioned Human Motion Generation](move-in-2d_2d-conditioned_human_motion_generation.md)
- [\[CVPR 2025\] Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction](nonisotropic_gaussian_diffusion_for_realistic_3d_human_motion_prediction.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[CVPR 2025\] MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](mirrorverse_pushing_diffusion_models_to_realistically_reflect_the_world.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)

</div>

<!-- RELATED:END -->
