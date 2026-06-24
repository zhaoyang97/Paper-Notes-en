---
title: >-
  [Paper Note] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning
description: >-
  [CVPR 2025][Causal Inference][Causal Prompt] Proposed the JSCPT (Joint Scheduling of Causal Prompts and Tasks) framework, which first designs Multi-Task Vision-Language Prompts (MTVLP) and eliminates spurious correlation features in prompts through causal intervention, and then adjusts learning order and weights using an adaptive task scheduler based on the dynamic changes in task relationships during training, achieving significant improvements across multiple multi-task vis…
tags:
  - "CVPR 2025"
  - "Causal Inference"
  - "Causal Prompt"
  - "Task Scheduling"
  - "Multi-Task Learning"
  - "VLM Prompt Learning"
  - "Spurious Correlation Elimination"
date: 2026-05-08
content_hash: e6eee32c1ad23461
---

# Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning

**Conference**: CVPR 2025  
**Code**: Unreleased  
**Area**: Causal Inference / Multi-Task Learning / Vision-Language Models  
**Keywords**: Causal Prompt, Task Scheduling, Multi-Task Learning, VLM Prompt Learning, Spurious Correlation Elimination

## TL;DR
Proposed the JSCPT (Joint Scheduling of Causal Prompts and Tasks) framework, which first designs Multi-Task Vision-Language Prompts (MTVLP) and eliminates spurious correlation features in prompts through causal intervention, and then adjusts learning order and weights using an adaptive task scheduler based on the dynamic changes in task relationships during training, achieving significant improvements across multiple multi-task visual recognition benchmarks.

## Background & Motivation

**Background**: Prompt learning is the mainstream paradigm for efficiently adapting pre-trained vision-language models (VLMs, e.g., CLIP) to downstream tasks. By learning a small number of prompt vectors, new tasks can be adapted without fine-tuning the entire model. Multi-task prompt learning further extends this to handle multiple tasks simultaneously, bridging several downstream visual recognition tasks with shared prompts.

**Limitations of Prior Work**: Existing multi-task prompt learning methods suffer from two overlooked key issues: (1) **Spurious Correlation**—there are task-irrelevant but statistically associated features in training data (e.g., co-occurrence of "water" and "boat"). Prompts might learn these spurious associations rather than causal features, resulting in poor generalization on out-of-distribution data. (2) **Dynamic Task Relationships**—the gradient relationships among multiple tasks continuously change during training (e.g., tasks A and B facilitate each other in the early stages, but might interfere later). Fixed joint training strategies (equal weights, fixed order) cannot adapt to this dynamism.

**Key Challenge**: Prompts need to serve multiple tasks simultaneously, but different tasks may introduce different spurious correlations (varying bias directions), and the synergy/conflict relationships among tasks evolve across training stages. This limits the performance of simple shared prompts in multi-task scenarios.

**Goal**: (1) Introduce causal inference into prompt learning to learn prompt features from a causal rather than correlation-based perspective. (2) Model dynamic task relationships to adaptively adjust scheduling strategies in multi-task learning.

**Key Insight**: Establish a causal graph to describe the causal and spurious pathways between prompt features and task labels, and then perform causal intervention via do-calculus to block spurious pathways.

**Core Idea**: Purify prompt features via causal intervention (de-biasing spurious correlations) and optimize the learning process via dynamic task scheduling (de-conflicting), with the two jointly optimized into a unified framework.

## Method

### Overall Architecture
JSCPT consists of three core components: (1) **Multi-Task Vision-Language Prompt (MTVLP)**: Designing learnable shared prompts for CLIP's text and vision encoders; (2) **Causal Prompt Learning Module**: Eliminating spurious correlation features in prompts based on a causal graph and counterfactual reasoning; (3) **Adaptive Task Scheduler**: Monitoring gradient relationships among tasks during training to dynamically adjust task weights and learning order.

### Key Designs

1. **Multi-Task Vision-Language Prompt (MTVLP)**:

    - **Function**: Learn shared prompt vectors for multiple downstream tasks
    - **Mechanism**: Add learnable prompt tokens $\mathbf{P}_t \in \mathbb{R}^{N \times D}$ and $\mathbf{P}_v \in \mathbb{R}^{M \times D}$ in the input layers of CLIP's text encoder and vision encoder, respectively. All tasks share the same set of prompts, and task-specific lightweight adapter layers (linear projection) are employed to provide each task with targeted prompt representations $\mathbf{P}_t^{(k)} = W_k \mathbf{P}_t$.
    - **Design Motivation**: Shared prompts reduce parameter overhead and encourage cross-task knowledge transfer, while task-specific projections maintain the distinctiveness of each task.

2. **Causal Prompt Learning**:

    - **Function**: Eliminate spurious correlation signals within prompt features
    - **Mechanism**: Construct a causal graph $\mathcal{G}$, separating prompt features $Z$ into causal features $Z_c$ (causally related to labels $Y$) and spurious features $Z_s$ (statistically correlated with $Y$ only). Causal intervention $P(Y | do(Z_c))$ is used to block the influence of confounding factors $C$ on the predictions. A feature decoupling network separates the prompt features into two parts, applying the backdoor adjustment formula $P(Y | do(Z_c)) = \sum_c P(Y | Z_c, c) P(c)$ to the causal feature component to integrate out the effects of the confounder $c$. In practice, this summation is approximated by sampling the distribution of confounders across the training set.
    - **Design Motivation**: Traditional prompt learning relies on standard cross-entropy training, which indiscriminately exploits all correlated features (including spurious ones). Causal intervention ensures that prompts only encode features causally related to the tasks.

3. **Adaptive Task Scheduler**:

    - **Function**: Adjust learning strategies based on the dynamic relationships between tasks during training
    - **Mechanism**: In each training epoch, calculate gradients of each task with respect to the shared prompt parameters $\nabla_{\mathbf{P}} \mathcal{L}_k$, and measure the degree of cooperation/conflict between tasks using the gradient cosine similarity $\cos(\nabla_k, \nabla_j)$. When gradient directions of two tasks are similar (cooperative), their weights are increased to encourage joint learning; when gradient directions are opposite (conflicting), the weights of conflicting tasks are reduced or set aside temporarily. The scheduling policy also considers the current loss value of each task, prioritizing poorly learned tasks.
    - **Design Motivation**: "Negative transfer" is a core challenge in multi-task learning. Adaptive scheduling continuously monitors task relationships to achieve a flexible "unite when aligned, separate when conflicting" strategy.

### Loss & Training
The overall loss is the weighted sum of causal cross-entropy loss and task scheduling weights: $\mathcal{L} = \sum_{k=1}^{K} w_k^{(t)} \cdot \mathcal{L}_{causal}^{(k)}$, where $w_k^{(t)}$ is the dynamic weight of task $k$ at epoch $t$. The training process employs alternating optimization: first fix scheduling weights to update prompt parameters, then fix prompts to update the scheduler.

## Key Experimental Results

### Main Results

| Method | Office-Home (Avg Acc) | DomainNet (Avg Acc) | VTAB (Avg Acc) | Average Gain |
|------|----------------------|--------------------|----|----------|
| CoOp (Single-Task Prompt) | 72.8 | 54.3 | 68.1 | baseline |
| MaPLe (Multi-Task Prompt) | 75.1 | 56.8 | 70.5 | +2.4 |
| TaskPrompter | 76.3 | 58.2 | 71.8 | +3.7 |
| **JSCPT (Ours)** | **79.6** | **61.7** | **74.2** | **+6.8** |

### Ablation Study

| Configuration | Office-Home | DomainNet | Description |
|------|-------------|-----------|------|
| Full JSCPT | 79.6 | 61.7 | Full Model |
| w/o Causal Intervention | 76.8 | 58.9 | Degenerates to standard prompt + scheduling |
| w/o Task Scheduling | 77.4 | 59.5 | Causal prompt + equal-weight training |
| w/o MTVLP (Independent Prompts) | 75.2 | 56.4 | Non-shared prompts |
| Fixed Task Weights | 77.1 | 59.1 | Without dynamic weight adjustment |

### Key Findings
- Causal intervention contributes approximately +2.8% absolute accuracy gain (Office-Home), validating the importance of eliminating spurious correlations.
- Dynamic task scheduling contributes approximately +2.2%, with larger improvements on DomainNet where task conflicts are more pronounced.
- The joint effect of both is superior to the sum of using them individually (+6.8 > 2.8+2.2), demonstrating positive synergy.
- Causal prompting has a more pronounced advantage on out-of-distribution test sets (+3.5% vs. +2.1% in-distribution), consistent with the expectation that causal inference boosts generalization.
- Shared prompts (MTVLP) perform 4.4% better than independent prompts, confirming the value of cross-task knowledge transfer.

## Highlights & Insights
- **First integration of causal inference and prompt learning**: Introducing do-calculus into VLM prompt learning represents a theoretically profound innovation, opening up new directions for robustness research in prompt learning.
- **Two-pronged strategy of "de-biasing spurious correlations + de-conflicting tasks"**: Eliminating spurious correlations at the prompt level and mitigating negative transfer at the task level complement each other, addressing the two core bottlenecks of multi-task prompt learning.
- **Gradient signals as real-time probes for task relationships**: Measuring task synergy/conflict through gradient cosine similarity is a lightweight and effective approach.

## Limitations & Future Work
- The decomposition of causal and spurious variables in causal graphs relies on prior assumptions, and different tasks may require different causal graph structures.
- When the number of tasks is large, the calculation of the gradient similarity matrix and the search space for the scheduling strategy grow rapidly.
- It is currently validated only on visual classification tasks; its effectiveness on generative tasks (e.g., image captioning, VQA) remains unknown.
- Sampling approximations of the confounder distribution in causal interventions can introduce estimation bias.

## Related Work & Insights
- **vs. CoOp/CoCoOp (Prompt Learning)**: CoOp learns single-task prompts, and CoCoOp adds conditional generation prompts. Neither considers spurious correlation, which JSCPT fundamentally addresses from a causal perspective.
- **vs. PCGrad/CAGrad (Multi-Task Gradient Methods)**: PCGrad eliminates conflicts via gradient projection, while CAGrad searches for a mean gradient direction. JSCPT's task scheduling is more flexible, adjusting both gradient directions and task weights.
- **vs. CausalVLM (Causal VLM)**: CausalVLM introduces causal objectives during the pre-training stage, while this work introduces them during the prompt fine-tuning stage, making it more lightweight and easier to deploy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The three-way integration of causal inference, prompt learning, and dynamic task scheduling is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple benchmarks with thorough ablation studies and out-of-distribution generalization analysis.
- Writing Quality: ⭐⭐⭐⭐ The construction of the causal graph and theoretical derivations are clear.
- Value: ⭐⭐⭐⭐ Provides a theory-driven new paradigm for multi-task VLM fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](../../ICML2026/causal_inference/tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICCV 2025\] Social Debiasing for Fair Multi-modal LLMs](../../ICCV2025/causal_inference/social_debiasing_for_fair_multi-modal_llms.md)
- [\[ICLR 2026\] Causal Score Conditioning for Multi-Resolution Latent Systems](../../ICLR2026/causal_inference/causal_score_conditioning_for_multi-resolution_latent_systems.md)
- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](../../ICLR2026/causal_inference/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)

</div>

<!-- RELATED:END -->
