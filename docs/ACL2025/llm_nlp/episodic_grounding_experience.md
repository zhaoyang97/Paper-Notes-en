---
title: >-
  [Paper Note] Growing Through Experience: Scaling Episodic Grounding in Language Models
description: >-
  [ACL 2025][LLM (Other)][Episodic Grounding] This paper proposes a weak-to-strong episodic grounding framework that collects structured experience data via MCTS, transfers the episodic grounding capabilities of smaller models to larger models through behavioral ratio distillation, and leverages DPO optimization to learn from both successful and failed experiences. This approach outperforms SOTA models, including GPT-4o, by 3.45% on physical planning tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Episodic Grounding"
  - "Weak-to-Strong Distillation"
  - "MCTS"
  - "Physical Planning"
  - "preference optimization"
date: 2026-05-08
content_hash: ec2b3d00a5673978
---

# Growing Through Experience: Scaling Episodic Grounding in Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.01312](https://arxiv.org/abs/2506.01312)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Episodic Grounding, Weak-to-Strong Distillation, MCTS, Physical Planning, preference optimization

## TL;DR

This paper proposes a weak-to-strong episodic grounding framework that collects structured experience data via MCTS, transfers the episodic grounding capabilities of smaller models to larger models through behavioral ratio distillation, and leverages DPO optimization to learn from both successful and failed experiences. This approach outperforms SOTA models, including GPT-4o, by 3.45% on physical planning tasks.

## Background & Motivation

Language models perform exceptionally well on various generative tasks, yet they still struggle with physical planning tasks. The core reason is the lack of **episodic grounding**, which is the ability to learn from past experiences and apply them to new situations. This challenge is closely linked to the role of episodic memory in cognition within brain science.

Current approaches face a fundamental **scale paradox**:

- **Small Models (1.3B-7B)**: Can be easily fine-tuned on episodic data, but their hierarchical representation and long-term context recall capabilities are insufficient, achieving only 54.76% accuracy on planning tasks—far below the 74.34% of large models (405B).
- **Large Models (70B-405B)**: Possess deep architectures and rich pre-training knowledge, yet lack efficient pathways for experience integration—their parameter scale makes direct fine-tuning cost-prohibitive.

This architectural asymmetry implies that the largest models, which are the most capable of leveraging episodic experiences, are precisely the most difficult to train to integrate these experiences. This work aims to break this paradox.

## Method

### Overall Architecture

The framework consists of two stages:

1. **Experience Collection**: Leverages MCTS to gather structured episodic experiences (including successful and failed exploration trajectories) from a physical simulator (VirtualHome).
2. **Weak-to-Strong Distillation**: Transfers the learned episodic behaviors from the small model to the large model, while utilizing DPO to learn preferences from both positive and negative experiences.

### Key Designs

#### 1. MCTS Experience Collection

Employs Monte Carlo Tree Search to collect episodic data from physical simulators. MCTS comprises four standard steps:

- **Selection**: Selects promising nodes using the UCT formula: $UCT = Q(s,a) + C \cdot \sqrt{\frac{\log(N(s))}{N(s,a)}}$
- **Expansion**: Expands leaf nodes, adding child nodes for unexplored actions.
- **Rollout**: Executes action sequences in the simulator, with a reward function of +2 for satisfying target predicates and -0.1 per step for irrelevant actions.
- **Backpropagation**: Backpropagates rewards to update Q-values and visit counts.

Successful explorations (satisfying all target predicates) are labeled as positive samples $y^+$, while failed explorations are labeled as negative samples $y^-$. Additionally, redundant, verbose plans are introduced as extra negative samples, training the model to avoid generating inefficient action sequences.

#### 2. Small Model Training

The physical planning task is formulated as a sequence prediction problem, where the small model (< 8B parameters) serves as a policy function $\pi$ mapping the input $\mathbf{x}$ to an action sequence $\mathbf{y}$. The training objective is:

$$\mathcal{L}_V = \sum_{v \in V} \alpha_v \sum_{m=1}^{M} \log \pi(y_m | \mathbf{y}_{<m}, \mathbf{x})$$

#### 3. Behavioral Ratio Distillation (Episodic Distillation)

The core innovation lies in utilizing the behavioral changes of the small model before and after training to guide the large model. Let $\pi^{\mathcal{E}}$ be the post-training small-model policy and $\pi^{\mathcal{N}}$ be the initial small-model policy. Their behavioral ratio $\frac{\pi^{\mathcal{E}}(y_m|\mathbf{y}_{<m},\mathbf{x})}{\pi^{\mathcal{N}}(y_m|\mathbf{y}_{<m},\mathbf{x})}$ captures the effect of episodic grounding.

The adjusted policy distribution of the large model is defined as:

$$\bar{\pi}(y_m|\mathbf{y}_{<m},\mathbf{x}) = \frac{1}{\bar{Z}} \pi^{\mathcal{L}}(y_m|\mathbf{y}_{<m},\mathbf{x}) \times \frac{\pi^{\mathcal{E}}(y_m|\mathbf{y}_{<m},\mathbf{x})}{\pi^{\mathcal{N}}(y_m|\mathbf{y}_{<m},\mathbf{x})}$$

Alignment is then achieved by minimizing the reverse KL-divergence:

$$\mathcal{L}_{\text{RKL}} = \mathbb{E}_{\mathbf{x},\mathbf{y} \sim \bar{\pi}} \left[\sum_{m=1}^{M} \log \frac{\bar{\pi}(y_m|\mathbf{y}_{<m},\mathbf{x})}{\pi^{\mathcal{L}}(y_m|\mathbf{y}_{<m},\mathbf{x})}\right]$$

The reverse KL is chosen over the forward KL because of its mode-seeking property, steering the large model to generate more precise, confident, and goal-aligned action sequences.

#### 4. DPO Preference Optimization

To learn simultaneously from both successful and failed experiences, a modified DPO loss is introduced:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(\mathbf{x},y^+,y^-) \sim \mathcal{D}} \left[\log \sigma\left(\beta \cdot (\log \pi(y^+|\mathbf{x}) - \log \pi(y^-|\mathbf{x}))\right)\right] + \lambda \cdot \mathbb{E}_{\mathbf{x},y \sim \pi}\left[\log \frac{\pi(y|\mathbf{x})}{\pi_0(y|\mathbf{x})}\right]$$

where $\beta$ controls the sharpness of the preference learning, and $\lambda$ weights the reverse KL regularization term to maintain proximity to the initial policy.

### Loss & Training

The entire training is split into a three-stage pipeline:

1. **Stage 1**: Train the small model on instruction data collected via MCTS (SFT) to learn episodic grounding.
2. **Stage 2**: Distill the episodic behavior into the large model using behavioral ratios (via reverse KL optimization).
3. **Stage 3**: Perform DPO preference optimization on the large model using positive and negative samples from MCTS.

## Key Experimental Results

### Main Results

**Evaluation Tasks**: Physical Planning (VirtualHome) and Question Answering (VirtualHome QA)

**Plan Generation Results (Accuracy %)**:

| Model | VS | VU | CS | CU | Path | Avg. |
|------|-----|-----|-----|-----|------|------|
| GPT-4o (base) | 52.67 | 49.35 | 47.54 | 46.22 | 81.23 | 55.40 |
| GPT-Neo 1.3B-ewc | 49.70 | 49.27 | 46.88 | 42.34 | 85.91 | 54.82 |
| GPT-J 6B-ewc | 51.23 | 49.58 | 48.94 | 45.60 | 98.67 | 58.80 |

**Question Answering Results (Accuracy %)**:

| Model | HW | Neg. | Recog. | Inf. | Count. | Loc. | Avg. |
|------|------|------|--------|------|--------|------|------|
| GPT-4o (base) | 85.37 | 84.31 | 95.60 | 84.85 | 78.43 | 74.21 | 83.80 |
| GPT-J 6B-ewc | 85.44 | 39.51 | 88.52 | 74.43 | 67.01 | 34.50 | 64.90 |

**Overall Average**: Ours outperforms GPT-4o by 3.45% on average across all tasks.

**Key Comparisons**:
- GPT-Neo 1.3B base → ewc: 34.81% → 54.82% (+20%)
- GPT-J 6B base → ewc: 45.51% → 61.29% (+15.78%)
- Small models exhibit significant improvement after episodic grounding training.

### Key Findings

1. **Weak-to-strong Effectiveness**: Through behavioral ratio distillation, 70B and 405B models maintain stable accuracy in complex, multi-step planning, whereas baselines suffer severe performance degradation beyond 4 steps.

2. **Resolution of the Scale Paradox**: Directly training small models is effective but has a low ceiling. Through behavioral ratio distillation, large models inherit task-specific grounding capabilities while preserving their general-purpose abilities.

3. **Layer-wise Probing Analysis**: Deeper layers of the model achieve 90% accuracy on episodic reasoning tasks, providing empirical evidence akin to hierarchical processing in the human neocortex. Shallow layers encode basic perceptual information, while deeper layers develop complex episodic reasoning capabilities.

4. **Value of Failed Experiences**: DPO preference optimization utilizes failed explorations as negative examples, significantly improving generalization and preventing over-fitting that arises from learning purely from positive instances.

## Highlights & Insights

1. **Precise Characterization of the Scale Paradox**: The paper clearly diagnoses the paradox in episodic grounding where "the models that need this capability the most are the hardest to acquire it", and proposes behavioral ratio distillation as an elegant solution.
2. **Cognitive Science Analogy**: Inspired by episodic memory and the hierarchical processing of the neocortex, the layer-wise probing analysis reveals an interesting correspondence between LLM internal representations and cognitive science.
3. **Joint Utilization of Positive and Negative Experiences**: The framework learns not only from successful trajectories but also extracts valuable signals from failures, incorporating redundant plans as extra negative inputs to cohesively enhance model efficiency.
4. **No Direct Fine-tuning of Large Models Required**: By adjusting the inference distribution via behavioral ratios, it bypasses the prohibitive fine-tuning costs of large models.

## Limitations & Future Work

1. **Environment Dependency**: Experience collection relies heavily on the VirtualHome simulator, and its transferability to real physical environments remains unverified.
2. **Distillation Assumptions**: Behavioral ratio distillation assumes alignment in the token space between small and large models; hence, transfer across different model families may be restricted.
3. **Computational Overhead**: MCTS exploration inside the simulator incurs high computational costs and requires recollection of experience for each new environment.
4. **QA Task Degradation**: Under certain QA sub-tasks (e.g., Negation, Location), the ewc-trained models show regression compared to the base models, suggesting that training for episodic grounding may occasionally compromise certain pre-existing general capabilities.

## Related Work & Insights

- **Embodied AI**: Simulators such as VirtualHome (Puig et al., 2018) and ProcTHOR (Deitke et al., 2022) provide the foundation for episodic data collection.
- **LM Grounding**: Approaches like SayCan (Ichter et al., 2022) and DEPS (Wang et al., 2023) explore LLM grounding in physical environments, though mostly confined to specific tasks.
- **Weak-to-strong learning**: The behavioral ratio distillation proposed in this paper can be viewed as an extension of speculative decoding ideas, employed here for knowledge transfer instead of inference acceleration.
- **Insights**: The weak-to-strong distillation approach in this framework can be generalized to other scenarios requiring large-scale fine-tuning, such as guiding large models using reward signals from smaller models in RLHF.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4.5 |
| Technical Depth | 4.5 |
| Experimental Thoroughness | 4 |
| Value | 3.5 |
| Writing Quality | 4 |
| Overall Rating | 4.1 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs](autogui_scaling_gui_grounding_with_automatic.md)
- [\[ACL 2025\] TestNUC: Enhancing Test-Time Computing Approaches and Scaling through Neighboring Unlabeled Data Consistency](testnuc_enhancing_test-time_computing_approaches_and_scaling_through_neighboring.md)
- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] Cooperating and Competing Through Natural Language](cooperating_and_competing_through_natural_language.md)
- [\[ACL 2025\] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)

</div>

<!-- RELATED:END -->
