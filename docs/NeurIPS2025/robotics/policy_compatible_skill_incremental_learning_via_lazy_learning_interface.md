---
title: >-
  [Paper Note] Policy Compatible Skill Incremental Learning via Lazy Learning Interface
description: >-
  [NeurIPS 2025][Robotics][Skill Incremental Learning] This paper proposes SIL-C, a framework that achieves skill-policy compatibility in skill incremental learning via a bilateral lazy learning interface…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Skill Incremental Learning"
  - "Hierarchical Policy"
  - "Continual Learning"
  - "Lazy Learning"
  - "Skill-Policy Compatibility"
  - "Embodied Intelligence"
date: 2026-05-08
content_hash: 57f2014b798a8690
---

# Policy Compatible Skill Incremental Learning via Lazy Learning Interface

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.20612](https://arxiv.org/abs/2509.20612)  
**Authors**: Daehee Lee (SKKU), Dongsu Lee (UT Austin), TaeYoon Kwack (SKKU), Wonje Choi (SKKU), Honguk Woo (SKKU)
**Code**: [GitHub](https://github.com/L2dulgi/SIL-C)  
**Area**: Robotics
**Keywords**: Skill Incremental Learning, Hierarchical Policy, Continual Learning, Lazy Learning, Skill-Policy Compatibility, Embodied Intelligence

## TL;DR

This paper proposes SIL-C, a framework that achieves skill-policy compatibility in skill incremental learning via a bilateral lazy learning interface, enabling incrementally updated skills to directly improve downstream policy performance without retraining or structural modification.

## Background & Motivation

### State of the Field
Lifelong embodied agents must continuously acquire new skills from data streams, integrate them into a skill library, and leverage existing skills to solve downstream tasks. Skill Incremental Learning (SIL) supports agents in progressively expanding and refining their skill sets. In hierarchical policies, a high-level policy selects subtasks while a low-level skill decoder executes concrete actions; the two are coupled through a shared latent space.

### Limitations of Prior Work
- **Skill updates invalidate policies**: As the skill library evolves over time, downstream policies that rely on these skills may fail due to changes in the skill interface, limiting reusability and generalization.
- **Type I methods (BUDS/PTGM + continual learning)**: Simple skill-appending schemes perform poorly in terms of skill forgetting and compatibility. Fine-tuning (FT) causes severe forgetting, while experience replay (ER) and adapter appending (AA) provide only partial remedies.
- **Type II methods (semantic label-driven)**: These depend on predefined semantic skill labels, limiting scalability. BWT is typically near zero—maintaining only initial performance without benefiting from new skills.
- **Synchronous update assumption**: Existing SIL methods generally assume that the skill library and downstream policies are updated synchronously, precluding decoupled independent evolution.

### Root Cause
The paper identifies and addresses two compatibility problems in SIL: (1) **Forward Skill Compatibility (FwSC)**—newly added skills can be effectively utilized by future policies; (2) **Backward Skill Compatibility (BwSC)**—existing policies can leverage newly added or updated skills and achieve improved performance without retraining.

## Method

### Overall Architecture
SIL-C consists of three core components: a high-level policy $\pi_h^\tau$ that selects subtasks $z_h$, a low-level skill decoder $\pi_l^p$ that executes actions, and a lazy learning interface $\mathcal{I}$ connecting the two. The execution flow is:

$$z_h \sim \pi_h^\tau(\cdot|s), \quad z_l = \mathcal{I}(s, z_h), \quad a \sim \pi_l^p(\cdot|s, z_l)$$

### Bilateral Lazy Learning Interface
The interface $\mathcal{I}$ operates across two spaces—the subtask space $\mathcal{X}_h$ and the skill space $\mathcal{X}_l$—and performs trajectory distribution similarity matching via an instance-based classifier.

**Instance Classifier**: Each label $c$ is modeled by a multi-modal Gaussian prototype $\chi_c = \{(\mu_{c,k}, \Sigma_{c,k})\}_{k=1}^{K_c}$. Given a query $x$, the Mahalanobis distance is computed as:

$$d_c(x) = \min_{k \leq K_c} \sqrt{(x - \mu_{c,k})^\top \Sigma_{c,k}^{-1} (x - \mu_{c,k})}$$

Two operations are supported: (i) classification—nearest prototype selection; (ii) verification—OOD detection based on chi-squared quantiles (99th percentile threshold).

**Skill Space Update**: At each SIL phase $p$, unsupervised skill clustering → K-means sub-clustering → Gaussian prototype computation is performed on the new dataset $\mathcal{D}_p$, and results are stored in skill memories $\mathcal{X}_l^{s,p}$ and $\mathcal{X}_l^{g,p}$.

**Subtask Space Update**: A similar procedure is applied to expert demonstrations $\mathcal{D}_\tau$ for each downstream task $\tau$, generating subtask prototypes stored in $\mathcal{X}_h^{s,\tau}$.

### Skill Verification and Hooking at Inference
The task-side module $\Psi_h^s$ predicts a subgoal $g$ conditioned on the current state $s$; the skill-side module $\Psi_l^g$ verifies whether the current skill $z_h$ can achieve that subgoal. If verification passes, the skill is executed directly as $z_l = z_h$; otherwise, skill hooking is triggered to select the most suitable alternative skill from the candidate set based on trajectory similarity: $z_l = \Psi_l^s(s; \mathcal{Z}')$.

### Policy Learning Integration
An energy-based prior guides high-level policy learning: given state $s$, the skill decoder evaluates all candidate skill pairs and selects the subtask label whose decoded action is closest to the expert action, optimized via behavioral cloning.

## Key Experimental Results

### Experiment 1: Skill-Policy Compatibility in the Kitchen Environment

Evaluation is conducted in the Franka Kitchen environment under both Emergent SIL and Explicit SIL scenarios, spanning 4 SIL phases and 24 downstream tasks.

| Method | Type | Scenario | BwSC BWT | BwSC AUC | Overall AUC | FwSC AUC | Final FWT |
|--------|------|----------|----------|----------|-------------|----------|-----------|
| PTGM+FT | I | Emergent | -36.9% | 17.3% | 24.1% | 36.2% | 24.7% |
| PTGM+ER | I | Emergent | -19.7% | 32.1% | 42.5% | 54.0% | 57.7% |
| PTGM+AA | I | Emergent | +2.6% | 46.9% | 58.2% | 66.1% | 83.6% |
| iSCIL* | II | Emergent | +0.5% | 55.5% | 55.7% | 55.8% | 56.9% |
| iManip* | II | Emergent | +18.5% | 67.8% | 70.3% | 68.7% | 77.4% |
| **SIL-C** | **III** | **Emergent** | **+18.6%** | **66.8%** | **71.8%** | **71.9%** | **87.2%** |
| PTGM+AA | I | Explicit | +0.0% | 14.6% | 36.7% | 53.3% | 79.8% |
| **SIL-C** | **III** | **Explicit** | **+42.5%** | **46.5%** | **54.1%** | **51.9%** | **80.6%** |

Key finding: SIL-C achieves a BwSC BWT of +42.5% under the Explicit SIL scenario, substantially outperforming all baselines (best prior: +0.0%), demonstrating that the interface can effectively leverage newly added skills to improve the performance of existing policies.

### Experiment 2: Few-Shot Imitation Learning

Sample efficiency of SIL-C is evaluated under limited expert demonstrations in the Kitchen Emergent SIL scenario.

| Method | Shots | Ratio | Initial FWT | BwSC AUC | Overall AUC | Final FWT |
|--------|-------|-------|-------------|----------|-------------|-----------|
| PTGM+AA | 5 | 100% | 40.5% | 40.8% | 49.1% | 67.6% |
| **SIL-C** | **5** | **100%** | **47.4%** | **55.9%** | **62.4%** | **75.8%** |
| PTGM+AA | 1 | 100% | 26.5% | 27.7% | 31.7% | 37.8% |
| **SIL-C** | **1** | **100%** | **43.1%** | **52.0%** | **56.5%** | **65.7%** |
| PTGM+AA | 1 | 20% | 25.7% | 23.5% | 27.0% | 30.5% |
| **SIL-C** | **1** | **20%** | **37.3%** | **46.4%** | **49.7%** | **58.5%** |

Key finding: Under the extreme few-shot setting (1-shot, 20% transitions), SIL-C achieves a BwSC AUC of 46.4%, nearly double that of the baseline (23.5%), demonstrating strong low-data generalization.

### Robustness Experiment

As input noise increases from ×1 to ×5, SIL-C maintains a BWT of +4.3% at ×5 noise, while PTGM+AA drops to −1.2%. At ×3 noise, the Final FWT gap grows from an initial −0.2% to +12.4%, indicating that SIL-C's advantage becomes more pronounced as the skill library grows larger.

## Highlights & Insights

- **First systematic definition of SIL compatibility**: The paper explicitly introduces Forward (FwSC) and Backward (BwSC) skill-policy compatibility, providing a clear problem formulation for hierarchical policies in continual learning.
- **Elegant lazy learning interface design**: By reformulating the alignment problem as an instance-based classification problem and deferring decisions to inference time, skills and policies can evolve independently without synchronous updates.
- **Comprehensive and rigorous experiments**: Experiments span two environments (Kitchen/Meta-World), two SIL scenarios (Emergent/Explicit), and multiple baseline configurations; BwSC BWT of +42.5% under Explicit SIL far exceeds all baselines.
- **Modular design**: SIL-C can be applied as a plug-in to different baselines (BUDS/PTGM) and different SIL algorithms (FT/ER/AA), offering broad generality.

## Limitations & Future Work

- **Simulation-only evaluation**: Validation is limited to Kitchen and Meta-World; real-robot scenarios and richer state representations remain unexplored.
- **Dependence on unsupervised clustering quality**: The resolution of skill and subtask spaces is determined by the clustering algorithm; noisy or highly diverse skill distributions may degrade prototype quality.
- **No support for skill removal**: The current framework follows an append-only design and does not support skill deletion or merging.
- **Expressiveness of Gaussian prototypes**: The diagonal covariance assumption may fail to capture complex skill distributions.
- **No online interactive learning**: The framework requires offline expert demonstrations and does not explore online autonomous exploration.

## Related Work & Insights

- **BUDS/PTGM (Type I)**: Provide skill clustering and decoder architectures without compatibility guarantees. SIL-C adds an interface layer on top, improving BwSC BWT from +2.6% to +18.6%.
- **iSCIL (Type II)**: Prototype-based skill retrieval that depends on predefined semantic labels; BWT is only +0.5% (nearly unchanged), offering no benefit from new skills.
- **iManip (Type II)**: Instruction-driven temporal replay and model expansion; BWT reaches +18.5% but similarly relies on semantic labels. SIL-C achieves comparable performance without requiring labels.
- **Continual pretraining methods**: Approaches such as AdapterAppend maintain forward compatibility but cannot improve backward compatibility. SIL-C addresses backward compatibility through dynamic mapping at inference time.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic definition and resolution of skill-policy compatibility in SIL; the lazy learning interface design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two environments, multiple scenarios, complete ablation, robustness, few-shot, and resolution analyses.
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear, figures are intuitive, and experimental organization is systematic.
- Value: ⭐⭐⭐⭐ — Addresses a practical pain point in hierarchical continual learning; modular design has engineering value, though simulation-only evaluation limits direct applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning Parameterized Skills from Demonstrations](learning_parameterized_skills_from_demonstrations.md)

</div>

<!-- RELATED:END -->
