---
title: >-
  [Paper Note] Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer
description: >-
  [ICML2025][Reinforcement Learning][Multi-Task Reinforcement Learning] This paper proposes the M3DT framework, which integrates Mixture-of-Experts (MoE) into the Decision Transformer to achieve parameter decoupling. By grouping tasks, each expert only learns task-specific knowledge for a small subset of tasks. Coupled with a three-stage training mechanism (backbone $\rightarrow$ experts $\rightarrow$ router) to prevent gradient conflict, increasing the number of experts simult…
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Multi-Task Reinforcement Learning"
  - "Decision Transformer"
  - "Mixture-of-Experts"
  - "Task Scalability"
  - "Parameter Scalability"
date: 2026-05-08
content_hash: da7f191f00c3ad5f
---

# Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer

**Conference**: ICML2025  
**arXiv**: [2505.24378](https://arxiv.org/abs/2505.24378)  
**Code**: TBD  
**Area**: Reinforcement Learning  
**Keywords**: Multi-Task Reinforcement Learning, Decision Transformer, Mixture-of-Experts, Task Scalability, Parameter Scalability

## TL;DR

This paper proposes the M3DT framework, which integrates Mixture-of-Experts (MoE) into the Decision Transformer to achieve parameter decoupling. By grouping tasks, each expert only learns task-specific knowledge for a small subset of tasks. Coupled with a three-stage training mechanism (backbone $\rightarrow$ experts $\rightarrow$ router) to prevent gradient conflict, increasing the number of experts simultaneously scales up parameters and reduces task load, successfully scaling offline multi-task RL to 160 simulated control tasks.

## Background & Motivation

**Background**: Decision Transformer (DT) models offline RL as a sequence prediction problem, and high-capacity Transformer architectures have also been applied to Multi-Task RL (MTRL). Inspired by the generalization capabilities of LLMs across massive tasks, researchers aim to train RL agents capable of mastering a vast and diverse set of tasks.

**Limitations of Prior Work**:
- **Insufficient task scalability**: Most existing works handle only dozens of tasks (e.g., Atari or Meta-World), with performance degrading significantly when scaled up. While Gato covers over 600 tasks, its performance on control tasks remains poor.
- **Inefficient parameter scaling**: Simply increasing the size of the DT model quickly hits a performance ceiling. Scaling beyond 20M parameters yields almost no improvement, which contrasts sharply with the "bigger is better" scaling laws observed in NLP.

**Key Challenge**: Gradient conflicts arising from shared parameters intensify as the number of tasks increases, leading to performance degradation. Simply scaling up the shared parameters fails to alleviate this, as the fundamental issue is not insufficient parameter capacity, but optimization conflicts caused by all tasks sharing the same parameters.

**Goal**: How to maintain learning quality for each individual task while achieving efficient parameter scaling when the number of tasks is extremely large (e.g., 160 tasks)?

**Key Insight**: Through systematic empirical analysis, the authors uncover two key insights:
- Performance degradation is most severe when the number of tasks is small ($<40$), and flattens out as the task count increases $\rightarrow$ **Conversely, reducing the number of tasks that each parameter subset needs to learn to a sufficiently small scale can significantly boost performance.**
- Parameter scaling is most effective when it "simultaneously reduces the task load" $\rightarrow$ **Scaling up parameters must be accompanied by reducing the number of tasks allocated to each parameter subset.**

**Core Idea**: Utilize MoE to achieve "parameter decoupling + task grouping", scaling up parameters while reducing the task load by increasing the number of experts, complemented by a three-stage training mechanism to avoid gradient interference.

## Method

### Overall Architecture

The overall pipeline of M3DT is as follows:

- **Input**: Identical to Prompt-DT, the trajectory sequence $(r^*_{1}, s^*_1, a^*_1, \ldots, \hat{r}_{t-K+1}, s_{t-K+1}, a_{t-K+1}, \ldots, \hat{r}_t, s_t, a_t)$, prefixed with a trajectory prompt.
- **Backbone**: A Prompt-DT with 5.29M parameters (6 layers, 8 heads, 256 dimensions) to learn shared knowledge across tasks.
- **MoE Layer**: Parallel expert FFNs and a router MLP are appended alongside the standard FFN in each Transformer block. The feed-forward output is $f(x) = x + f_{\text{FFN}}(x) + f_{\text{MoE}}(x)$.
- **Router**: A 5-layer MLP that applies softmax to hidden states to obtain routing weights for each expert.
- **Output**: Predicts action $a_t$, optimized via MSE loss.

### Key Designs

1. **MoE-Enhanced DT Architecture**

    - Function: Adds multiple expert FFNs in parallel to the standard FFN in each Transformer block of the DT, while retaining the original FFN to maintain backbone knowledge.
    - Mechanism: The MoE output is formulated as $f_{\text{MoE}}(x) = \sum_{i=1}^N \text{Softmax}(f_{\theta_r}(x))_i \cdot f_{\theta_i}(x)$, where each expert shares the same structure as the original FFN. The original FFN is kept intact, and the MoE output is added as a residual connection, ensuring that the shared backbone knowledge is not corrupted.
    - Design Motivation: The authors find in Appendix B.2 that gradient conflicts in the MLP layers of the Transformer block are far more severe than in the Attention layers, and merely expanding the MLP yields limited benefits. Consequently, they choose to incorporate MoE into the MLP/FFN side for parameter decoupling, ensuring each expert only handles a few tasks, thereby significantly mitigating gradient conflicts within the FFN.

2. **Task Grouping Strategy**

    - Function: Explicitly partitions all tasks into several groups, with each group allocated to a specific expert for independent training.
    - Mechanism: Two grouping strategies are provided: (1) **Random Grouping**: Randomly splits tasks into equal subsets, which is simple and starting point effective; (2) **Gradient-based Grouping**: Calculates the agreement vector $A(\mathcal{T}_i) = g_i \odot \frac{1}{N}\sum_{i=1}^N g_i$ for each task (element-wise product of individual task gradient and mean gradient) to reflect gradient consistency, followed by K-means clustering. In experiments, gradient-based grouping outperforms random grouping by approximately 2% on 160 tasks (77.89 vs 80.14).
    - Design Motivation: The top-k dynamic routing used in standard MoEs is unstable in RL scenarios—experiments show that top-4 routing cannot scale with the number of experts, and performance even decreases as experts increase. Explicit grouping circumvents routing instability and load imbalance issues, while leveraging task structural info to minimize within-group gradient conflicts.

3. **Three-Stage Training Mechanism**

    - Function: Divides the training process into three sequential stages to independently optimize the backbone, experts, and router.
    - Mechanism:
        - **Stage 1 — Backbone Training (400k steps)**: Trains the Prompt-DT backbone across all tasks to learn shared cross-task representations. Crucially, early stopping is applied when gradient conflict peaks (around 400k steps) to prevent parameters from overfitting to tasks that dominate the gradient.
        - **Stage 2 — Expert Training (200k steps)**: Freezes the backbone and trains each expert independently on its designated task subset. Experts can be trained in parallel (~1.8 hours per expert).
        - **Stage 3 — Router Training (400k steps)**: Freezes both the backbone and experts, training only the router across all tasks to learn dynamic weight assignment. Total training takes about 24.2 hours on an RTX 4090.
    - Design Motivation: End-to-end training of MoE yields poor performance in RL—experiments reveal that gradient conflicts in end-to-end trained MoE models are even more severe than those in standard Prompt-DT MLPs. Multi-stage training allows each module to learn in a conflict-free environment, and the backbone early-stopping strategy prevents performance degradation after gradient conflict peaks.

### Loss & Training

- Action prediction MSE loss: $\mathcal{L}_{DT} = \mathbb{E}[\frac{1}{K}\sum_{m=t-K+1}^t (a_{i,m} - \pi(\tau_i^*, \tau_{i,m}))^2]$
- All three stages utilize the Adam optimizer with a learning rate of 1e-4 and a batch size of 16.
- No auxiliary MoE load-balancing loss is needed, as the explicit grouping strategy inherently ensures balanced load.

## Key Experimental Results

### Main Results: Performance Comparison across Task Scales

Normalized scores evaluated over 160 tasks in total: Meta-World (50 tasks) + DMControl (30 tasks) + Mujoco Locomotion (80 tasks).

| Task Scale | Method | Normalized Score | Parameter Count |
|---------|------|-----------|--------|
| 10 tasks | PromptDT-Small | ~88% | 1.47M |
| 10 tasks | HarmoDT-Large | ~89% | 173.30M |
| 10 tasks | M3DT-Gradient | **89.23%** | 47.87M |
| 80 tasks | PromptDT-Large | ~73% | 173.30M |
| 80 tasks | M3DT-Gradient | **79.58%** (+6.6%) | 98.37M |
| 160 tasks | PromptDT-Large | ~68% | 173.30M |
| 160 tasks | HarmoDT-Large | ~72% | 173.30M |
| 160 tasks | M3DT-Gradient | **80.14%** (+7.5%) | 174.12M |

The normalized score of M3DT on 160 tasks is even higher than those of other baselines evaluated on only 80 tasks.

### Ablation Study

| Configuration | Normalized Score (160 tasks) | Description |
|------|----------------------|------|
| M3DT-Random | 77.89 | Random grouping |
| M3DT-Gradient | **80.14** | Gradient grouping, optimal |
| M3DT w/o 3-stage training | ~68 | End-to-end training, degrades to PromptDT level |
| M3DT w/o grouping | 67.34 | All experts trained jointly across all tasks, worse than the baseline |
| M3DT-R w/o expert freezing | Suboptimal | Fine-tuning experts in Stage 3, overwriting learned knowledge |
| M3DT-G w/o expert freezing | Suboptimal | Same as above |

### Key Findings

- **MoE structure alone is insufficient**: Simply adding MoE with end-to-end training (w/o 3-stage) yields performance comparable to Prompt-DT, with even worse gradient conflicts. **Task grouping + three-stage training is the actual core.**
- **Number of experts correlates positively with performance but has an upper limit**: Performance steadily improves as experts scale from 8 to 40 (yielding an 11.7% increase on 160 tasks), but shows diminishing returns beyond 40. The upper bound is defined by three factors: (1) limited shared knowledge in the backbone; (2) diminishing returns once task subsets are already sufficiently small; (3) routing becomes increasingly difficult as the number of experts grows.
- **Early stopping of the backbone is critical**: The optimal stopping point occurs precisely when gradient conflict reaches its peak (400k steps). Too little training (200k steps) results in insufficient backbone knowledge; too much training ($>400$k steps) causes parameters to overfit to dominant tasks, hindering subsequent expert learning.
- **Top-K routing fails in RL**: Top-4 routing fails to scale with the number of experts, confirming the instability of sparse gating in RL. M3DT requires a weighted combination of all experts instead.
- **Random grouping is already considerably effective**: M3DT-Random already outperforms all baselines, indicating that "reducing the task load" per expert is indeed the primary source of gain; gradient-based grouping adds an additional ~2% improvement.

## Highlights & Insights

- **Counter-intuitive thinking of "reducing task load"**: Recognizing that degradation primarily occurs when scaling through small task numbers, the authors deduce that the core remedy is not simply scaling up model size, but reducing the task load exposed to each parameter subset. This insight is profound and counter-intuitive.
- **Crucial adaptation of MoE for RL**: While MoE in NLP typically relies on dynamic top-k routing, RL data distributions differ drastically, leading to routing instability. M3DT replaces top-k sparse routing with explicit task grouping + all-expert weighted combination, establishing a crucial domain-adaptation paradigm.
- **Gradient conflict peak as an early-stopping signal**: Aligning the gradient conflict dynamics (which rise and then plateau) with the training phases, the framework switches to group training precisely at the conflict peak, balancing shared knowledge acquisition with conflict evasion.

## Limitations & Future Work

- **Limited to simulated environments**: All experiments compile on Meta-World/DMC/Mujoco simulations with low-dimensional continuous vector states (at most 39 dimensions). Verification on high-dimensional visual inputs or real-world robots is missing.
- **Inference cost scales linearly with the number of experts**: All experts participate in feed-forward passes, unlike the sparse MoE in NLP where only a fraction is activated. The authors attempted top-k routing, but it yielded poor results; designing a sparse gating mechanism compatible with three-stage training remains an important engineering direction.
- **Hyperparameters in three-stage training**: Training steps for each stage (400k/200k/400k) require extra tuning, and the backbone stop point relies on monitoring gradient conflicts. Adaptive stage-switching mechanisms warrant future exploration.
- **Task grouping relies on priors**: Gradient-based grouping requires pre-training the backbone and computing agreement vectors, incurring extra computational overhead. How to online update groups for entirely new task sets or continually arriving tasks is an open question.
- **Unverified generalization and continual learning**: All experiments train and evaluate on the same task sets. Generalization capability to held-out tasks and learning stability under continual tasks arrive remain unexplored.

## Related Work & Insights

- **vs Multi-Game DT (Lee et al., 2022)**: Naively scales up model parameters to handle multi-task Atari; the authors demonstrate that this naive scaling approach saturates at around 20M parameters. M3DT achieves significantly better performance with a similar parameter budget via MoE.
- **vs Gato (Reed et al., 2022)**: A generalist multimodal agent covering 600+ tasks but exhibiting sub-optimal performance on control tasks. M3DT focuses specifically on control task scenarios, achieving superior performance through task grouping and expert specialization. Their ideas are complementary—MoE could be integrated into Gato-like frameworks.
- **vs HarmoDT (Hu et al., 2024)**: HarmoDT mitigates gradient conflict using task-specific masks to shield conflicting parameters, operating under a "shared parameters + masks" paradigm. M3DT's "shared backbone + independent experts" represents a more thorough parameter decoupling scheme, offering a clear advantage on 160 tasks (+7.5%).
- **vs Soft MoE in RL (Obando-Ceron et al., 2024)**: This work finds that top-k MoE is difficult to scale in deep RL, which aligns with the top-4 experiment conclusions of this paper. The three-stage training + all-expert weighted routing of M3DT is an effective solution to bypass this hurdle.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative application of MoE in RL. Its core contributions lie in the key adaptations for RL scenarios (explicit grouping + 3-stage training) and empirical insights (task count vs. performance curve).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic experiments covering task scales from 10 to 160. The ablation analysis is exceptionally comprehensive (grouping strategy, expert count, backbone early-stopping, router design, and expert architecture), with domain-wise analysis eliminating bias.
- Writing Quality: ⭐⭐⭐⭐ Clear and complete logical chain from empirical observations to methodology design. Figures 2 and 3 provide compelling visual evidence of the identified problems.
- Value: ⭐⭐⭐⭐ Strongly drives research on the scalability of multi-task RL. The "parameter decoupling + task grouping + multi-stage training" paradigm is highly generalizable to other multi-task settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mixture-of-World Models: Scaling Multi-Task Reinforcement Learning with Modular Latent Dynamics](../../ICLR2026/reinforcement_learning/mixture-of-world_models_scaling_multi-task_reinforcement_learning_with_modular_l.md)
- [\[ICML 2025\] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments](fast_and_robust_task_sampling_with_posterior_and_diversity_synergies_for_adaptiv.md)
- [\[ICLR 2026\] STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure Transformer for Offline Multi-Task Multi-Agent Reinforcement Learning](../../ICLR2026/reinforcement_learning/stairs-former_spatio-temporal_attention_with_interleaved_recursive_structure_tra.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](../../NeurIPS2025/reinforcement_learning/modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)

</div>

<!-- RELATED:END -->
