---
title: >-
  [Paper Note] AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation
description: >-
  [ICCV 2025][Robotics][bimanual manipulation] This paper proposes AnyBimanual, a plug-and-play framework that transfers pretrained unimanual manipulation policies to general bimanual manipulation scenarios via a Skill Manager and a Visual Aligner, achieving significant multi-task generalization with only a small number of bimanual demonstrations.
tags:
  - ICCV 2025
  - Robotics
  - bimanual manipulation
  - policy transfer
  - skill primitives
  - visual alignment
  - behavior cloning
date: 2026-05-08
content_hash: 9c8bf63b4f13fbe8
---

# AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation

**Conference**: ICCV 2025
**arXiv**: [2412.06779](https://arxiv.org/abs/2412.06779)
**Code**: [https://anybimanual.github.io/](https://anybimanual.github.io/)
**Area**: Robotic Manipulation
**Keywords**: bimanual manipulation, policy transfer, skill primitives, visual alignment, behavior cloning

## TL;DR

This paper proposes AnyBimanual, a plug-and-play framework that transfers pretrained unimanual manipulation policies to general bimanual manipulation scenarios via a Skill Manager and a Visual Aligner, achieving significant multi-task generalization with only a small number of bimanual demonstrations.

## Background & Motivation

Bimanual manipulation systems play an important role in domestic service, robotic surgery, and industrial assembly. Compared to single-arm systems, bimanual systems offer a larger workspace and can accomplish more complex tasks (e.g., one arm stabilizes the target while the other operates), but face critical bottlenecks:

**Expensive Data**: The action space of bimanual manipulation is extremely high-dimensional; collecting teleoperation demonstrations requires dedicated systems, additional sensors, and precise calibration, incurring substantial labor costs.

**Generalization Difficulty**: Constrained by data volume, directly learned bimanual policies struggle to generalize across diverse tasks.

**Limitations of Existing Approaches**:
   - LLM/VLM-based high-level planning methods: constrained by predefined low-level executors and unable to handle contact-rich tasks.
   - Fixed role assignment (stabilizer/actor): inflexible collaboration modes.
   - Parameterized atomic actions: difficult to specify manually, limiting deployment scenarios.

**Key Insight**: Unimanual policies (e.g., PerAct, RVT) have demonstrated impressive cross-task generalization through large-scale parameters and training data. Bimanual tasks can often be decomposed into combinations of unimanual subtasks; therefore, the general manipulation knowledge embedded in unimanual policies can be extracted and transferred.

## Method

### Overall Architecture

AnyBimanual consists of two core modules: a Skill Manager handling the language branch and a Visual Aligner handling the visual branch. Two pretrained unimanual policy models predict the actions of the left and right arms respectively.

### Key Designs

1. **Skill Manager**

    - Maintains a discrete set of skill primitives $\mathcal{Z} = \{z_1, z_2, ..., z_K\}$, where each skill $z_k \in \mathbb{R}^D$ is an implicit embedding.
    - Expresses the language embeddings for each arm as a linear combination of skill primitives plus a task-oriented compensation term:
    $\hat{l}_t^{left} = \sum_{k=1}^K \hat{w}_{k,t}^{left} z_k + \epsilon_t^{left}, \quad \hat{l}_t^{right} = \sum_{k=1}^K \hat{w}_{k,t}^{right} z_k + \epsilon_t^{right}$
    - Employs a multimodal Transformer to dynamically predict the combination weights at each step: $(\\hat{w}_t^{left}, \epsilon_t^{left}, \hat{w}_t^{right}, \epsilon_t^{right}) = f_\theta(v_t, l, p_t)$
    - Skill primitives can be initialized with language template tokens from pretrained unimanual policies to mitigate the domain gap.
    - **Design Motivation**: For example, a bimanual handover task can be decomposed into a "place" skill for the left arm and a "pick" skill for the right arm.

2. **Generalizable Skill Representation Learning**

    - Sparse regularization encourages reconstruction of language embeddings using the minimum number of skill primitives:
    $\mathcal{L}_{skill} = \|\hat{w}^{left}\|_1 + \|\hat{w}^{right}\|_1 + \lambda_\epsilon(\|\epsilon^{left}\|_{2,1} + \|\epsilon^{right}\|_{2,1})$
    - The $\ell_1$ norm enforces sparse selection, enabling each skill representation to capture an independent primitive motion.
    - The $\ell_{2,1}$ norm on the compensation term ensures task-specific knowledge is introduced only when necessary.

3. **Visual Aligner**

    - Generates two spatial soft masks to decompose the voxel space, aligning the visual input of each arm to its pretraining distribution:
    $v_t^{left} = (\hat{v}_t^{left} \odot v_t) \oplus v_t, \quad v_t^{right} = (\hat{v}_t^{right} \odot v_t) \oplus v_t$
    - Mutual exclusivity is enforced by maximizing the Jensen-Shannon divergence:
    $\mathcal{L}_{voxel} = -D_{KL}(\hat{v}_t^{left}\|\hat{v}_t^{right})/2 - D_{KL}(\hat{v}_t^{right}\|\hat{v}_t^{left})/2$
    - **Intuition**: During asynchronous bimanual collaboration, the left and right arms attend to different regions of the workspace; mutual exclusive decomposition naturally restores the bimanual scene to a unimanual configuration.

### Overall Training Objective

$$\mathcal{L}_{total} = \mathcal{L}_{BC} + \lambda_{skill}\mathcal{L}_{skill} + \lambda_{voxel}\mathcal{L}_{voxel}$$

where $\mathcal{L}_{BC}$ is the cross-entropy loss for behavior cloning. The framework is model-agnostic and supports different architectures including Transformer-based and diffusion-based policies.

## Key Experimental Results

### Main Results (RLBench2, 100 demonstrations)

| Method | straighten rope | sweep dustpan | press buttons | put in fridge | Average |
|--------|----------------|---------------|---------------|---------------|---------|
| PerAct2 | 8 | 52 | 41 | 16 | 14.67 |
| PerAct2+Pretrain | 17 | 55 | 40 | 22 | - |
| PerAct-LF | 11 | 47 | 9 | 7 | - |
| **PerAct+AnyBimanual** | **24** | **57** | **14** | **26** | **32.00** |

Overall average success rate: AnyBimanual (32.00%) vs. PerAct2 (14.67%), a gain of **17.33%**.

### Ablation Study

| Row | Skill Manager | Visual Aligner | Long | Generalized | Sync | Average |
|-----|---------------|----------------|------|-------------|------|---------|
| 1 | - | - | 16.29 | 23.50 | 3.50 | 14.67 |
| 2 | ✗ | ✗ | 19.57 | 25.50 | 9.50 | 16.75 |
| 3 | ✗ | ✓ | 21.57 | 44.00 | 15.50 | 19.75 |
| 4 | ✓ | ✗ | 23.71 | 42.00 | 17.00 | 25.67 |
| 5 | ✓ | ✓ | 27.29 | 44.00 | 25.00 | **32.00** |

### Key Findings

- AnyBimanual improves over the PerAct2 state of the art by 17.33%, with particularly pronounced advantages on long-horizon, multi-variant, and synchronous coordination tasks.
- As a plug-and-play method, it also improves the performance of PerAct-LF (+72.76%) and RVT-LF (+39.41%).
- The Skill Manager contributes most to long-horizon tasks (+8.92%), while the Visual Aligner contributes significantly to generalization and synchronization tasks.
- The two modules exhibit synergistic effects exceeding the sum of their individual contributions: from 14.67% to 32.00%.
- The average success rate across 9 real-world bimanual tasks is 84.62%, demonstrating strong practical utility.
- Visualizations show that the Skill Manager dynamically schedules reasonable skill combinations, and the Visual Aligner effectively decomposes the voxel space.

## Highlights & Insights

- **Elegant Framework Design**: The core idea is concise—bimanual manipulation is treated as the coordinated composition of two unimanual skills, realized through sparse decomposition and mutual exclusive masking.
- **Model Agnosticism**: Compatible with different unimanual policies such as PerAct and RVT, offering strong practical versatility.
- **Data Efficiency**: Transfer is achievable with only 20–100 bimanual demonstrations, substantially reducing data requirements.
- **Good Interpretability**: Skill scheduling weights and voxel decomposition masks are both visualizable, facilitating understanding of model behavior.

## Limitations & Future Work

- Performance is poor on simple tasks requiring precise rotation (e.g., Rotate Toothbrush, 20% success rate).
- Introducing additional complexity on short-horizon simple tasks (e.g., Lift ball) may slightly degrade performance.
- The number of skill primitives $K$ must be set manually.
- Evaluation is currently limited to RLBench2 and constrained real-world scenarios; validation on larger-scale benchmarks is lacking.
- Performance depends on the quality of keyframe extraction.

## Related Work & Insights

- This work contrasts with the fixed architectural design of PerAct2, demonstrating the paradigm advantage of "transfer rather than retrain."
- The sparse skill decomposition idea is generalizable to other multi-agent collaboration scenarios.
- The mutual exclusive visual masking design is inspired by observations of asynchronous bimanual collaboration and can be extended to multi-arm systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework design for transferring unimanual policies to bimanual settings is novel, with clear skill scheduling and visual alignment mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 12 simulation and 9 real-world tasks, with multiple baselines, complete ablations, and rich visualizations.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and the method is described systematically.
- Value: ⭐⭐⭐⭐⭐ A highly practical plug-and-play solution with significant impact on robotic manipulation research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Action–Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](../../CVPR2026/robotics/actiongeometry_prediction_with_3d_geometric_prior.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](../../ICLR2026/robotics/twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](../../ICLR2026/robotics/vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](../../CVPR2026/robotics/bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)

<!-- RELATED:END -->
