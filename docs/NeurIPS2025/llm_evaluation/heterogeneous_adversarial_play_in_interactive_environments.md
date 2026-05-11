---
title: >-
  [Paper Note] Heterogeneous Adversarial Play in Interactive Environments
description: >-
  [NeurIPS 2025][LLM Evaluation][Adversarial Curriculum Learning] This paper proposes **HAP (Heterogeneous Adversarial Play)**, which formalizes teacher-student interaction as a minimax game: a teacher network automaticall…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Adversarial Curriculum Learning"
  - "Heterogeneous Self-Play"
  - "Teacher-Student"
  - "Multi-Task RL"
  - "Adaptive Curriculum"
date: 2026-05-08
content_hash: 345fb141820b6174
---

# Heterogeneous Adversarial Play in Interactive Environments

**Conference**: NeurIPS 2025
**arXiv**: [2510.18407](https://arxiv.org/abs/2510.18407)
**Code**: [Project Page](https://sites.google.com/view/hap-learning)
**Area**: Curriculum Learning / Reinforcement Learning
**Keywords**: Adversarial Curriculum Learning, Heterogeneous Self-Play, Teacher-Student, Multi-Task RL, Adaptive Curriculum

## TL;DR

This paper proposes **HAP (Heterogeneous Adversarial Play)**, which formalizes teacher-student interaction as a minimax game: a teacher network automatically generates challenge tasks targeting student weaknesses, while the student policy continuously adapts and evolves, forming an adaptive curriculum without manual design. HAP outperforms state-of-the-art baselines in multi-task RL environments, and the generated curriculum proves effective for human learners as well.

## Background & Motivation

**The Dilemma of Automatic Curriculum Learning (ACL)**:

1. Traditional CL relies on manually predefined task difficulty hierarchies and cannot adapt to changes in learner capability.
2. Existing ACL methods operate unidirectionally (only selecting tasks or evaluating difficulty), lacking bidirectional feedback between teacher and student.
3. Symmetric self-play requires agents to share the same role, making it unsuitable for inherently asymmetric teaching scenarios.

**Insights from Cognitive Science**: Effective teaching requires:
- Personalized and adaptive task selection ("hypothesis space navigation")
- Dynamic updates to teaching strategy based on the learner's current state of understanding
- A closed-loop bidirectional feedback mechanism

## Method

### Minimax Optimization Framework

The student maximizes cumulative reward:

$$\max_\theta J_{\text{student}}(\theta) = \mathbb{E}_{T \sim p_\phi(T)} \left[\mathbb{E}_{\tau \sim \pi(\cdot|T;\theta)} [R(\tau;T)]\right]$$

The teacher minimizes the student's success rate (adversarial objective):

$$\max_\phi J_{\text{teacher}}(\phi) = \mathbb{E}_{T \sim p_\phi(T)} \left[\mathbb{E}_{\tau \sim \pi(\cdot|T;\theta)} [-R(\tau;T)]\right]$$

Together, these constitute a minimax game:

$$\min_\phi \max_\theta J(\theta, \phi)$$

### Teacher Policy Gradient

The teacher acts as a task distribution generator, with its policy gradient defined as:

$$\nabla_\phi J_{\text{teacher}}(\phi) = -\mathbb{E}_{T \sim p_\phi(T)} \left[\nabla_\phi \log p_\phi(T) \cdot \mathbb{E}_\tau[R(\tau;T)]\right]$$

Intuitively, this increases the sampling probability of tasks where the student fails and decreases it for already-mastered tasks.

### Engineering Stabilization Techniques

**Cold Start**: The student first explores each task independently for a period to establish baseline competence before adversarial training begins.

**Task Overload**: Entropy regularization is applied to prevent the teacher from assigning too many tasks simultaneously:

$$J_{\text{teacher}} = \mathbb{E}_T[-R(\tau;T)] + \lambda \cdot \mathcal{H}(p_\phi(T))$$

**Catastrophic Forgetting**: A lower bound is imposed on task selection probabilities to ensure mastered tasks retain a minimum exposure frequency.

### Alternating Update Algorithm

Each iteration consists of three steps:
1. The teacher samples a task $T \sim p_\phi(T)$
2. The student executes policy $\pi(\cdot|T;\theta)$ and updates $\theta \leftarrow \theta + \alpha \nabla_\theta \mathbb{E}_\tau[R(\tau;T)]$
3. The teacher updates $\phi \leftarrow \phi - \beta \nabla_\phi \mathbb{E}_T[R(\tau;T)]$

## Key Experimental Results

### Multi-Environment Multi-Task Evaluation

| Algorithm | Minigrid General | CRAFT General | Crafter General |
|------|-----------------|---------------|-----------------|
| DQN | 0.407 | 0.278 | 0.297 |
| PPO | 0.397 | 0.415 | 0.387 |
| SAC | 0.457 | 0.413 | 0.533 |
| DreamerV3 | 0.493 | 0.516 | 0.697 |
| TSCL | 0.443 | 0.307 | 0.423 |
| EXP3 | 0.463 | 0.513 | 0.490 |
| **HAP** | **0.527** | **0.562** | **0.723** |
| Human | 0.747 | 0.802 | 0.850 |

### Performance on Hard Tasks (Maximum Gap)

| Difficulty | Minigrid Hard | CRAFT Hard | Crafter Hard |
|---------|--------------|------------|-------------|
| DreamerV3 | 0.18 | 0.27 | 0.52 |
| **HAP** | **0.20** | **0.31** | **0.58** |
| Human | 0.46 | 0.66 | 0.74 |

HAP outperforms DreamerV3 by 15% on CRAFT Hard tasks.

### Adversarial Dynamics Analysis (Navigation Experiment)

- HAP reaches optimal performance at approximately 35k steps, exhibiting the fastest convergence.
- Positive feedback: increased sampling probability for failed tasks → accelerated targeted skill acquisition.
- Negative feedback: decreased sampling probability for mastered tasks → avoidance of redundant practice.
- Success rates improve uniformly across all four difficulty levels, eliminating the "easy-hard gap" commonly observed in baselines.

### Extension to Supervised Learning

On CIFAR-100 (imbalanced) and RTE (noisy labels), HAP achieves competitive performance compared to state-of-the-art curriculum methods such as ScreenerNet and MW-Net.

### Human Study

Thirty participants were evaluated in Minigrid under three conditions: no tutorial, expert-designed tutorial, and HAP-generated tutorial. HAP achieves final performance comparable to the expert tutorial while providing more personalized and adaptive adjustments.

## Highlights & Insights

- **Heterogeneity as the Key Innovation**: Breaks the constraint of symmetric self-play; the teacher and student can have entirely different architectures, objectives, and capabilities.
- **Bidirectional Feedback Loop**: The teacher adjusts the curriculum in real time based on student progress, which is more efficient than unidirectional task selection.
- **Cross-Modal Validation**: The adversarial curriculum principle is validated across RL, supervised learning, and human learning, demonstrating its generality.
- **Automatic Discovery of Teaching Strategies**: HAP autonomously discovers strategies consistent with human pedagogy, including scaffolding and adaptive difficulty regulation.

## Limitations & Future Work

1. The performance advantage narrows in open-world environments (Crafter), suggesting the need for additional intrinsic exploration mechanisms.
2. All algorithms remain substantially below human performance on the hardest tasks (HAP's best reaches only 65%–78% of human-level).
3. The task space must be predefined as a discrete task set; the framework has not been extended to continuous task parameterization.
4. Entropy regularization and probability lower bounds are heuristic designs lacking theoretical convergence guarantees.
5. The teacher network outputs only a task distribution rather than generating tasks themselves, and thus does not constitute true procedural task generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The formalization of heterogeneous adversarial play for curriculum learning is clear and thought-provoking.
- **Technical Depth**: ⭐⭐⭐ — The core approach relies on standard REINFORCE combined with minimax optimization; theoretical analysis is relatively thin.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive and rigorous, spanning three RL environments, supervised learning, and a human study.
- **Practicality**: ⭐⭐⭐⭐ — The general framework is applicable to a wide range of multi-task learning scenarios.
- **Overall**: ⭐⭐⭐⭐

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems](small_language_models_as_compiler_experts_auto-parallelization_for_heterogeneous.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)

</div>

<!-- RELATED:END -->
