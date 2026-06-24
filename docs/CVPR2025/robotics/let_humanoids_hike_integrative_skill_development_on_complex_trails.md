---
title: >-
  [Paper Note] Let Humanoids Hike! Integrative Skill Development on Complex Trails
description: >-
  [CVPR 2025][Robotics][Humanoid Robots] The LEGO-H framework is proposed, which unifies navigation perception and low-level locomotion control via TC-ViT (Temporal-Conditioned ViT). Combined with Hierarchical Latent Matching (HLM) for efficient distillation from an oracle policy, it enables the Unitree H1 humanoid robot to achieve a 68.4% success rate on complex outdoor hiking trails.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Humanoid Robots"
  - "Outdoor Navigation"
  - "Locomotion Skills"
  - "Hierarchical Latent Matching"
  - "Temporal ViT"
date: 2026-05-08
content_hash: 1a456cf4428683a5
---

# Let Humanoids Hike! Integrative Skill Development on Complex Trails

**Conference**: CVPR 2025  
**arXiv**: [2505.06218](https://arxiv.org/abs/2505.06218)  
**Code**: [https://LEGO-H-HumanoidRobotHiking.github.io](https://LEGO-H-HumanoidRobotHiking.github.io)  
**Area**: Robotics / Humanoid Robots  
**Keywords**: Humanoid Robots, Outdoor Navigation, Locomotion Skills, Hierarchical Latent Matching, Temporal ViT

## TL;DR

The LEGO-H framework is proposed, which unifies navigation perception and low-level locomotion control via TC-ViT (Temporal-Conditioned ViT). Combined with Hierarchical Latent Matching (HLM) for efficient distillation from an oracle policy, it enables the Unitree H1 humanoid robot to achieve a 68.4% success rate on complex outdoor hiking trails.

## Background & Motivation

### Background

**Background**: Quadrupedal robots have made significant progress in outdoor terrain navigation (e.g., ANYmal, Go2). However, outdoor locomotion remains a formidable challenge for humanoid robots due to their higher center of mass, higher degrees of freedom, and more complex balance requirements.

**Limitations of Prior Work**: Existing humanoid navigation methods divide the perception-planning-control pipeline into independent modules, which leads to information loss at interfaces. Direct end-to-end training struggles to converge due to the massive action space. Directly transferring quadrupedal methods (such as RMA) to humanoid robots also yields poor results (42.97% success rate).

**Key Challenge**: High-level navigation (where to go) and low-level locomotion (how to walk) require unified decision-making—on rugged terrain, "avoiding obstacles" and "adjusting gait" are different levels of the same action.

**Key Insight**: A two-stage training scheme is adopted: first training an oracle policy with privileged information to acquire diverse locomotion skills, and then distilling it into an unprivileged unified policy using HLM (Hierarchical VAE Latent Matching).

**Core Idea**: TC-ViT unified navigation + control combined with HLM distillation from the oracle yields end-to-end humanoid outdoor locomotion.

## Method

### Key Designs

1. **TC-ViT (Temporal-Conditioned Vision Transformer)**:

    - **Function**: Extracts navigation-related features from depth maps and fuses target information.
    - **Mechanism**: Patchifies the depth map and feeds it into the ViT along with a target direction token, enabling early fusion of target information within the attention layers to output navigation perception embeddings. The key is to inject target information before the positional embedding.
    - **Design Motivation**: The ConvGRU baseline only achieves a 42.97% success rate, whereas the multi-scale global attention of TC-ViT is better suited for perception in complex terrains.

2. **Hierarchical Latent Matching (HLM)**:

    - **Function**: Preserves action structures when distilling from the oracle policy to the unprivileged policy.
    - **Mechanism**: Trains a masked VAE to encode the oracle's action sequences, aligning the action distributions of the student and teacher in the latent space using cosine similarity and triplet loss. This outperforms direct $L_2$ action matching, as $L_2$ ignores the coordination relationships among joints.
    - **Design Motivation**: Ablation studies show that HLM reduces the collision rate from 10.40% to 7.84% and improves stability (time before falling) from 7.00s to 7.46s.

### Loss & Training

The Oracle is trained using PPO with RL rewards (direction tracking, torso height, fall penalty), while distillation uses $\mathcal{L}_{im} + \mathcal{L}_{hie}$, where $\mathcal{L}_{hie}$ includes VAE latent space cosine similarity and triplet loss. 512 robots are trained in parallel within Isaac Gym across hiking trail scenes of 5 difficulty levels.

## Key Experimental Results

### Main Results

| Method | Success Rate | Collision Rate | Time Before Falling |
|------|--------|--------|-----------|
| Vanilla ConvGRU | 42.97% | - | 5.36s |
| w/o HLM (TC-ViT only) | 64.73% | 10.40% | 7.00s |
| **LEGO-H (Ours)** | **68.40%** | **7.84%** | **7.46s** |
| Oracle (Upper Bound) | 71.20% | - | - |

### Key Findings
- **TC-ViT vs ConvGRU**: The success rate increases from 42.97% to 64.73%, proving that the global attention of ViT is critical for complex terrains.
- **HLM Improves Safety**: The collision rate decreases from 10.40% to 7.84%, showing the effectiveness of action structure regularization.
- **Close to Oracle Upper Bound**: 68.4% vs 71.2%, with a distillation performance gap of only 3 percentage points.

## Highlights & Insights
- **First Outdoor Mountain Trail Navigation for Humanoid Robots**—Extends the capability boundary of humanoid robots from flat ground/stairs to rugged mountain trails.
- **End-to-End Unified Design**—Avoids separating navigation and locomotion control, optimizing them jointly within the same network.

## Limitations & Future Work
- Fixed upper body (only controlling lower limbs), which limits balance recovery capability.
- Verified only in simulation, without deployment on real robots.
- Utilizes only depth sensors, lacking RGB/semantic information.
- Comparison baselines are adapted from quadrupedal methods, rather than being dedicated humanoid approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of TC-ViT + HLM is effective in humanoid navigation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 difficulty levels, 512 parallel robots, with comparisons across multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear.
- Value: ⭐⭐⭐⭐ Pushes the capability boundary of humanoid robots in the wild.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Periodic Skill Discovery](../../NeurIPS2025/robotics/periodic_skill_discovery.md)
- [\[ICLR 2026\] Visual Planning: Let's Think Only with Images](../../ICLR2026/robotics/visual_planning_lets_think_only_with_images.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[ICLR 2026\] Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning](../../ICLR2026/robotics/memory_benchmark_robots_a_benchmark_for_solving_complex_tasks_with_reinforcement.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](../../NeurIPS2025/robotics/policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)

</div>

<!-- RELATED:END -->
