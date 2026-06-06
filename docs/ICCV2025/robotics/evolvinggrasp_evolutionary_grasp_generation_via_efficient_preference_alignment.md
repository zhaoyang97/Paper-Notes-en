---
title: >-
  [Paper Note] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment
description: >-
  [ICCV 2025][Robotics][Dexterous Grasping] This paper proposes EvolvingGrasp, which achieves efficient evolutionary generation and human preference alignment for dexterous grasp pose synthesis via Handpose-wise Preference…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Dexterous Grasping"
  - "Preference Alignment"
  - "Consistency Models"
  - "Diffusion Models"
  - "Physical Constraints"
date: 2026-05-08
content_hash: c98f944df723274b
---

# EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment

**Conference**: ICCV 2025
**arXiv**: [2503.14329](https://arxiv.org/abs/2503.14329)  
**Code**: [https://evolvinggrasp.github.io/](https://evolvinggrasp.github.io/)  
**Area**: Robotics
**Keywords**: Dexterous Grasping, Preference Alignment, Consistency Models, Diffusion Models, Physical Constraints

## TL;DR

This paper proposes EvolvingGrasp, which achieves efficient evolutionary generation and human preference alignment for dexterous grasp pose synthesis via Handpose-wise Preference Optimization (HPO) and a Physics-Aware Consistency Model (PCM), attaining state-of-the-art performance on four benchmark datasets with a 30× inference speedup.

## Background & Motivation

The generalization capability of dexterous robotic hands in complex environments is constrained by insufficient diversity in training data. The infinite variety of real-world scenarios makes it impractical to predefine all grasping strategies. Existing approaches can be categorized as:
- **Optimization-based methods**: optimize hand poses via force-closure criteria, but incur high computational cost.
- **Learning-based methods**: directly regress mappings from features to grasp poses, but suffer from mode collapse.
- **Generative methods**: e.g., DexGrasp Anything employs diffusion models, but requires hundreds of sampling steps and cannot align with human preferences.

The core challenges are: (1) existing methods cannot continuously adapt after deployment and fail to handle distributional shifts; (2) iterative sampling in diffusion models and physical constraint computation result in low efficiency; (3) no mechanism exists for aligning with human grasping preferences. Inspired by evolutionary principles—systems iteratively improve through continuous feedback from successes and failures—this paper proposes an evolutionary grasp generation framework.

## Method

### Overall Architecture

EvolvingGrasp consists of two core modules:
1. **Handpose-wise Preference Optimization (HPO)**: formalizes preference alignment as posterior probability optimization, enabling the model to iteratively learn from successful and failed grasp samples.
2. **Physics-Aware Consistency Model (PCM)**: distills a diffusion teacher model into a lightweight consistency model and integrates physical constraints to ensure physically feasible generation.

Given an object point cloud $O \in \mathbb{R}^{N \times 3}$, the goal is to generate dexterous grasp poses with high success rates and low penetration by sampling from the posterior distribution $P(x|O)$, where the pose parameters include joint angles $\theta_h \in \mathbb{R}^{24}$, global translation $T_{global} \in \mathbb{R}^3$, and global rotation $R_{global} \in SO(3)$.

### Key Designs

**HPO (Handpose-wise Preference Optimization)**:
- Introduces DPO into the dexterous grasping domain for the first time, extending it into a more flexible formulation.
- Standard DPO requires paired preference data (one positive and one negative sample); HPO relaxes this constraint to allow unequal numbers of positive and negative samples.
- Models preference probability via the Bradley-Terry model; the optimization objective increases the probability of successful grasps while decreasing that of failed ones.
- Preference labels can be obtained via simulation evaluation (six-direction stability tests) or human-in-the-loop selection.
- Employs LoRA for lightweight fine-tuning to achieve efficient preference alignment.

**Physics-Aware Consistency Model (PCM)**:
- **Physics-Aware Distillation**: trains a diffusion teacher model first, then distills it into a consistency student model, incorporating three categories of physical constraints into the distillation loss:
    - Surface Pulling Force: maintains stable contact between fingers and the object.
    - External Penetration Repulsion: prevents fingers from penetrating the object.
    - Self-Penetration Repulsion: avoids collisions between fingers.
- **Physics-Aware Sampling**: during inference, corrects the sampling mean via gradients of physical constraints to guide trajectories toward physically feasible poses.

### Loss & Training

The overall training proceeds in three stages:
1. **Diffusion Pre-training**: trains the teacher model with a standard noise prediction loss.
2. **Physics-Aware Distillation**: $\mathcal{L}_{PAD} = \mathcal{L}_{CD} + \sum \alpha_i L_{PA_i}$, combining consistency distillation loss with physical constraints.
3. **Preference Fine-tuning**: applies HPO loss via LoRA lightweight fine-tuning of the entire model.
    - Successful and failed samples are collected from simulation; successes serve as positive examples and failures as negative examples.
    - Online iteration: grasp performance improves continuously as more samples are generated.

## Key Experimental Results

### Main Results

Evaluation on four datasets—DexGraspNet, MultiDex, RealDex, and DexGRAB:

| Method | DexGraspNet Suc.6↑ | MultiDex Suc.6↑ | RealDex Suc.6↑ | DexGRAB Suc.6↑ | Time↓ |
|------|-------|-------|-------|-------|-------|
| UniDexGrasp | 33.9 | 21.6 | 27.1 | 20.8 | 0.46s |
| DexGrasp Any. | 53.6 | 72.2 | 34.6 | 56.5 | 32.91s |
| Ours w/o HPO (4-step) | 63.8 | 75.3 | 51.6 | 55.6 | 1.41s |
| **Ours (4-step)** | **65.2** | **76.8** | **50.6** | **57.7** | **1.41s** |
| Ours (8-step) | 65.4 | 80.3 | 64.4 | 60.8 | 2.71s |
| Real-time (2-step) | 55.2 | 63.7 | 46.5 | 48.9 | **0.06s** |

Compared to the prior SOTA method DexGrasp Anything, EvolvingGrasp achieves a **30× speedup** (32s → 1.41s) alongside substantially improved success rates.

### Ablation Study

Module contributions validated on the MultiDex dataset (4-step):

| Config | CM | PGD | PGS | HPO | Suc.6↑ | Pen.↓ |
|------|:---:|:---:|:---:|:---:|--------|-------|
| a | ✓ | | | | 60.0 | 14.0 |
| b | ✓ | ✓ | | | 64.3 | 12.5 |
| e | ✓ | ✓ | ✓ | | 75.3 | 13.1 |
| **f** | ✓ | ✓ | ✓ | ✓ | **76.8** | **13.0** |

- Physics-aware distillation (PGD) improves Suc.6 from 60.0 to 64.3 (+4.3).
- Physics-aware sampling (PGS) yields the largest gain, reaching 75.3 (+11.0).
- HPO preference alignment further improves performance to 76.8.

### Key Findings

1. As fine-tuning epochs increase, the Suc.6 metric improves continuously and penetration depth shows an overall downward trend.
2. Starting from a suboptimal model trained on degraded data, evolutionary fine-tuning via HPO ultimately surpasses the accuracy of the original model.
3. The real-time mode without physical guidance (2-step) requires only 0.06s, making it suitable for real-time applications.
4. Successful deployment on a real ShadowHand robot validates the evolutionary grasping capability.

## Highlights & Insights

1. **First application of DPO to dexterous grasping**, extended into HPO that removes the strict pairing requirement, making it better suited to robotic scenarios.
2. The combination of **consistency models and physical constraints** is elegant: few-step generation efficiency is ensured while physical plausibility is guaranteed through dual constraints at both distillation and sampling stages.
3. **Evolutionary self-improvement**: the model can continuously improve after deployment using its own generated success/failure samples, without requiring additional annotations.
4. The 30× speedup represents a significant practical engineering breakthrough—reducing inference time from 32s to 1.41s and enabling real-time grasping.

## Limitations & Future Work

1. Preference fine-tuning may reduce generation diversity, as alignment-oriented strategies can limit the exploration space.
2. Preference data currently originate from simulation (six-direction stability tests); preference definitions may require adjustment when transferring to complex real-world scenarios.
3. Physical constraints (e.g., penetration forces) rely on known object geometry; generalization to unknown objects remains to be validated.
4. The sensitivity of LoRA fine-tuning hyperparameters (rank, learning rate) across different scenarios is not thoroughly discussed.

## Related Work & Insights

- **DexGrasp Anything**: a physically constrained diffusion model that is slow but produces high-quality grasps; serves as an important baseline for this work.
- **Diffusion-DPO**: extends DPO to multi-step MDPs for diffusion model preference alignment; the direct inspiration for HPO.
- **Consistency Models (CM/sCMs)**: the consistency model framework enables few-step sampling; this work augments it with physical constraints.
- Insight: the combination of preference learning and physical constraints is generalizable to other embodied manipulation tasks (e.g., assembly, tool use).

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4 |
| Value | 4.5 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [\[ICCV 2025\] Embodied Representation Alignment with Mirror Neurons](embodied_representation_alignment_with_mirror_neurons.md)
- [\[NeurIPS 2025\] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges](../../NeurIPS2025/robotics/grasp2grasp_vision-based_dexterous_grasp_translation_via_schrödinger_bridges.md)
- [\[ICCV 2025\] PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency](pacgdc_label-efficient_generalizable_depth_completion_with_projection_ambiguity_.md)
- [\[ICCV 2025\] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)

</div>

<!-- RELATED:END -->
