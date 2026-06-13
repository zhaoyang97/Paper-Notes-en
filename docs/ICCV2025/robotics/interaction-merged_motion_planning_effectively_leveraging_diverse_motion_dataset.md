---
title: >-
  [Paper Note] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning
description: >-
  [ICCV 2025][Robotics][Motion Planning] This paper proposes IMMP (Interaction-Merged Motion Planning), a two-stage strategy — Interaction-Conserving Pre-Merging (constructing a multi-metric checkpoint pool) and Interactio…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Motion Planning"
  - "Model Merging"
  - "Domain Adaptation"
  - "Autonomous Driving"
  - "Task Vectors"
date: 2026-05-08
content_hash: e5c82f303805f2d1
---

# Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning

**Conference**: ICCV 2025
**arXiv**: [2507.04790](https://arxiv.org/abs/2507.04790)  
**Code**: [GitHub](https://github.com/wooseong97/IMMP)  
**Area**: Robotics
**Keywords**: Motion Planning, Model Merging, Domain Adaptation, Autonomous Driving, Task Vectors

## TL;DR

This paper proposes IMMP (Interaction-Merged Motion Planning), a two-stage strategy — Interaction-Conserving Pre-Merging (constructing a multi-metric checkpoint pool) and Interaction Transfer with Merging (task-vector-based weighted merging grouped by interaction modules) — to transfer agent behavior and interaction knowledge from diverse trajectory datasets to a target domain, effectively improving cross-domain adaptability of motion planning.

## Background & Motivation

Motion planning is a core component of autonomous robots. Although numerous trajectory datasets exist (ETH-UCY, CrowdNav, THOR, SIT, etc.), effectively leveraging them poses significant challenges:

**Large dataset discrepancies**:
- **Indoor vs. Outdoor**: Indoor environments exhibit constrained motion and lower speeds; outdoor environments are dynamic with higher speeds.
- **HHI vs. HRI**: Human-Human Interaction and Human-Robot Interaction have fundamentally different dynamics.
- **Unbalanced data scales**: Dataset sizes vary considerably.

**Limitations of prior work**:

**Domain Generalization / Domain Adaptation** (jointly training on multiple datasets):
- **Domain imbalance**: Certain datasets dominate optimization, suppressing learning from others.
- **Catastrophic forgetting**: Newly introduced domains overwrite previously learned information.

**Ensemble learning** (merging predictions from multiple models):
- Requires multiple full models at inference, multiplying computational cost (7× overhead).
- Ensemble-AVG often underperforms Ensemble-WTA, indicating that only a subset of source domains are beneficial.

**Conventional model merging** (e.g., Task Arithmetic, Ties Merging):
- Does not account for the hierarchical feature structure inherent to motion planning models.
- May disrupt the feature hierarchy of trajectory encoding and interaction modeling.
- Performs poorly when applied directly to motion planning.

**Core insight**: Motion planning models possess an inherent hierarchical structure — ego encoder, surrounding agent encoder, interaction encoder, and decoder each encode different levels of information. Merging should respect this hierarchy by setting independent merging weights per module.

## Method

### Overall Architecture

IMMP consists of two stages:

1. **Interaction-Conserving Pre-Merging**: Planning models are trained on each source domain; multi-metric optimal checkpoints and intermediate checkpoints collected during training form a rich parameter checkpoint pool $\mathcal{P}$.
2. **Interaction Transfer with Merging**: Model parameters are grouped by interaction-related modules; task vectors are extracted per group and merging weights are learned jointly to optimize target-domain performance.

### Key Designs

1. **Multi-Metric Checkpoint Collection Strategy**:

    - *Function*: Builds a rich and diverse parameter checkpoint pool for the planning model.
    - *Mechanism*: During source-domain training, model parameters $\Theta^{best,m}$ are recorded whenever each evaluation metric (ADE, FDE, CR, MR) reaches its optimum; intermediate checkpoints are also saved every $C$ iterations. This yields multiple checkpoints per source domain, each encoding different facets of domain knowledge.
    - *Design Motivation*: Unlike classification tasks with a single accuracy metric, motion planning requires balancing multiple trade-off metrics — effectiveness, safety, and goal-reaching rate. Parameters optimal for different metrics reflect different behavioral characteristics. Intermediate checkpoints generalize better as they are not overfitted to the source domain.

2. **Interaction-Level Module-Grouped Merging**:

    - *Function*: Partitions model parameters by functional module, with each group independently learning its merging weights.
    - *Mechanism*: Model parameters $\Theta$ are divided into four groups: $\{\theta_{ego}, \theta_{surr}, \theta_{inter}, \theta_{else}\}$, corresponding to the ego encoder, surrounding agent encoder, interaction encoder, and remaining parameters. Task-vector merging is applied independently to each group: $\theta^* = \theta_0 + \sum_{i=1}^{|\mathcal{P}|} w_{i,\theta} \cdot \tau_i$, where $\tau_i = \theta_i - \theta_0$ denotes the task vector.
    - *Design Motivation*: Human motion patterns, robot trajectories, and human-robot interaction dynamics exhibit different distributional shifts across datasets. Module-level grouping enables the merging process to selectively extract information at each level from the most relevant source checkpoints, rather than applying a single shared set of weights.

3. **Task-Vector-Based Parameter Merging**:

    - *Function*: Combines knowledge from pre-trained checkpoints by learning only linear combination weights.
    - *Mechanism*: Given checkpoint pool $\mathcal{P} = \{\Theta_1, ..., \Theta_{|\mathcal{P}|}\}$, the merged parameters are $\Theta = \Theta_0 + \lambda\sum_{i=1}^{|\mathcal{P}|} w_i \cdot \tau_i$. The optimization objective is $\Theta^* = \arg\min_\Theta \sum_i \sum_j \mathcal{L}(\Theta, X_j^{t,i}, Y_j^{t,i})$, updating only the weights $\{w_{i,\theta}\}$.
    - *Design Motivation*: Adaptation is performed directly in parameter space rather than over data, which (1) eliminates the need to access source-domain data, (2) effectively mitigates domain imbalance and catastrophic forgetting, and (3) incurs the same computational cost as a single model (1× cost).

### Loss & Training

- The pre-merging stage uses each planning model's original multi-objective loss $\mathcal{L}_{total}$ (comprising trajectory deviation loss and collision penalty).
- The merging stage uses the target-domain $\mathcal{L}_{total}$ to learn the merging weights.
- The base parameters $\Theta_0$ may be a model trained from scratch or any fine-tuned model.
- IMMP + Finetune: The merged parameters serve as initialization for subsequent fine-tuning on the target domain.

## Key Experimental Results

### Main Results

**SIT Target Domain (GameTheoretic Planning Model):**

| Method | ADE↓ | Col.Rate↓ | FDE↓ | Miss Rate↓ | Cost |
|--------|------|-----------|------|------------|------|
| Domain Generalization | 0.8338 | 9.87E-04 | 1.8594 | 0.9355 | ×1 |
| Domain Adaptation | 0.4388 | 1.26E-03 | 1.0611 | 0.7201 | ×1 |
| Target Only | 0.4343 | 3.41E-04 | 0.9014 | 0.6272 | ×1 |
| Ensemble-WTA | 0.3695 | 5.75E-05 | 0.8283 | 0.6185 | ×7 |
| Task Arithmetic | 0.4132 | 1.37E-04 | 0.8936 | 0.7364 | ×1 |
| Ties Merging | 1.1876 | 5.53E-04 | 2.2440 | 0.9872 | ×1 |
| **IMMP** | **0.3380** | **5.12E-05** | **0.7626** | 0.6446 | ×1 |
| **IMMP + Finetune** | **0.3157** | **4.28E-05** | **0.7300** | **0.5934** | ×1 |

**THOR Target Domain:**

| Method | ADE↓ | FDE↓ | Miss Rate↓ |
|--------|------|------|------------|
| Target Only | 0.1003 | 0.2153 | 0.0929 |
| Domain Adaptation | 0.1133 | 0.2516 | 0.1268 |
| **IMMP + Finetune** | **0.0975** | **0.2108** | **0.0912** |

### Ablation Study

**Merging Granularity Comparison (SIT, GameTheoretic):**

| Granularity | ADE↓ | Col.Rate↓ | FDE↓ | Miss Rate↓ | Description |
|-------------|------|-----------|------|------------|-------------|
| Model-level | 0.3687 | 7.15E-05 | 0.8365 | 0.8002 | Uniform weight over full model |
| Parameter-level | 0.3798 | 9.16E-05 | 0.7754 | 0.7433 | Per-parameter weights |
| **Interaction-level** | **0.3380** | **5.12E-05** | **0.7626** | **0.6446** | Module-grouped (best) |

**Checkpoint Type Ablation:**

| All Metric | Epoch Ckpt | ADE↓ | FDE↓ | Miss Rate↓ |
|-----------|------------|------|------|------------|
| ✓ | | 0.3646 | 0.8063 | 0.6516 |
| ✓ | ✓ | 0.3543 | 0.7730 | 0.6196 |

### Key Findings

1. **Domain Generalization performs poorly**: Jointly training on multiple datasets performs even worse than Target Only, underscoring the severity of the domain imbalance problem.
2. **Conventional merging methods fail**: Averaging, Task Arithmetic, and Ties Merging all perform poorly in motion planning; Ties Merging degrades nearly to random performance.
3. **IMMP surpasses ensembles at single-model cost**: IMMP (×1 cost) outperforms Ensemble-WTA (×7 cost) on most metrics.
4. **Interaction-level merging significantly outperforms model-level and parameter-level**: Miss Rate drops from 0.80/0.74 to 0.64, validating the necessity of hierarchical merging.
5. **Learned merging weights align with source-domain relevance**: Qualitative analysis shows that source domains with poor target-domain performance are automatically assigned low weights.
6. **Consistent effectiveness across three planning models**: Significant improvements are observed on GameTheoretic, DIPP, and DTPP.

## Highlights & Insights

- **First systematic introduction of model merging into motion planning**: The paper identifies why directly applying existing merging methods fails and proposes a targeted solution.
- **Interaction-level merging is broadly generalizable**: The principle of grouping parameters by functional module can extend to any model with a hierarchical feature structure.
- **Source-data-free adaptation**: After merging, source-domain data are no longer required, substantially reducing the privacy and storage costs of domain adaptation.
- **Validates the value of multi-metric checkpoint collection**: Checkpoints optimal for different metrics encode distinct behavioral characteristics, enriching the basis for merging.

## Limitations & Future Work

1. A full model must be trained separately for each source domain; the computational cost of the pre-merging stage scales linearly with the number of source domains.
2. Optimizing the merging weights still requires annotated target-domain data, precluding truly zero-shot adaptation.
3. The module grouping (ego/surr/inter/else) relies on prior knowledge of the planning model's internal structure and may not be applicable to end-to-end black-box models.
4. The intermediate checkpoint sampling interval $C$ is a hyperparameter whose optimal value may vary across models and datasets.
5. Evaluation is limited to 2D pedestrian/robot trajectory planning; effectiveness on complete autonomous driving pipelines involving perception remains unknown.

## Related Work & Insights

- Borrows the task-vector concept from Task Arithmetic, with key adaptations for the multi-metric nature and hierarchical structure of motion planning.
- Complements dataset-level domain generalization approaches such as UniTraj: IMMP operates in parameter space, avoiding the problems introduced by data mixing.
- The interaction-level merging idea also provides insights for modular design in multi-task learning (e.g., shared vs. task-specific layer decisions in MTL).
- Can be combined with parameter-efficient methods such as LoRA to further reduce the training cost per source domain.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying model merging to motion planning is a novel attempt; interaction-level merging is the key innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three planning models, multiple baselines, two target domains, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, though mathematical notation is somewhat dense.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient solution for cross-domain adaptation in motion planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](../../ICLR2026/robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](../../NeurIPS2025/robotics/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play](../../NeurIPS2025/robotics/volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)
- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](../../CVPR2026/robotics/dawn_pixel_motion_diffusion_robot_control.md)

</div>

<!-- RELATED:END -->
