---
title: >-
  [Paper Note] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning
description: >-
  [ACL 2025][Model Compression][LoRA] MoRE (Mixture of Low-Rank Experts) is proposed to treat different ranks in LoRA as distinct experts. Through an adaptive rank selector, the most suitable rank is dynamically chosen for each task. Combined with task embeddings optimized by contrastive learning and a balanced data sampling strategy, efficient multi-task fine-tuning is achieved using a single LoRA module.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LoRA"
  - "Multi-Task Learning"
  - "Mixture of Experts"
  - "Adaptive Rank Selection"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 820de73a57692674
---

# MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning

**Conference**: ACL 2025  
**arXiv**: [2505.22694](https://arxiv.org/abs/2505.22694)  
**Code**: [GitHub](https://github.com/NLPfreshman0/MoRE)  
**Area**: Others  
**Keywords**: LoRA, Multi-Task Learning, Mixture of Experts, Adaptive Rank Selection, Parameter-Efficient Fine-Tuning

## TL;DR

MoRE (Mixture of Low-Rank Experts) is proposed to treat different ranks in LoRA as distinct experts. Through an adaptive rank selector, the most suitable rank is dynamically chosen for each task. Combined with task embeddings optimized by contrastive learning and a balanced data sampling strategy, efficient multi-task fine-tuning is achieved using a single LoRA module.

## Background & Motivation

While LoRA, as the primary parameter-efficient fine-tuning method, performs excellently in single-task scenarios, it faces a core contradiction in multi-task scenarios:

**Different tasks require different ranks**: The authors discover through experiments (Table 1) that T5-base achieves the optimal rank $r=1$ on MRPC, $r=8$ on CoLA, and $r=4$ on RTE. A fixed rank cannot adapt to all tasks.

**High cost of searching for the optimal rank**: Independently searching for and training multiple LoRAs for each task is computationally and storage-wise uneconomical.

**Limitations of Prior Work**:
   - Dynamic rank methods such as DyLoRA, AdaLoRA, and SoRA are only designed for single tasks, without considering the relationships and differences between tasks.
   - Parallel LoRA schemes (e.g., MultiLoRA, MixLoRA) increase the parameter size, violating the original purpose of LoRA to reduce parameters.
   - Prompt tuning methods require two-stage training, which is inefficient.

Core Problem: **How to achieve efficient LLM fine-tuning in multi-task scenarios?** The key lies in serving multiple tasks simultaneously with a single LoRA module and automatically allocating the appropriate rank.

## Method

### Overall Architecture

MoRE introduces modified LoRA modules into the attention and FFN layers of the Transformer. The core pipeline is as follows:
1. Assign a learnable task embedding to each task.
2. Select the appropriate rank (i.e., "rank expert") based on the task embedding through an adaptive rank selector.
3. Perform the forward computation by truncating the LoRA matrix using the selected rank.
4. Optimize the quality of task embeddings via contrastive learning and stabilize training through balanced sampling.

### Key Designs

1. **Low-Rank Experts**:

    - Function: Treat each rank $r_i \in [1, r]$ in LoRA as an independent "expert".
    - Mechanism: Experts of different ranks share information through the overlapping parts of the LoRA matrix—experts with ranks $r_i$ and $r_j$ share parameters of the first $\min(r_i, r_j)$ rows/columns.
    - Design Motivation: Similar tasks should select close ranks (sharing more parameters), while different tasks should select distinct ranks (maintaining personalization).

2. **Adaptive Rank Selector**:

    - Function: Output the probability distribution of ranks $\mathbf{p}_t = \text{softmax}(\mathbf{W}_g \mathbf{e}_t + \mathbf{b}_g)$ based on the task embedding $\mathbf{e}_t$.
    - Select the rank corresponding to the maximum probability: $r_t = \arg\max \mathbf{p}_t$.
    - Use Straight-Through Estimator (STE) to solve the non-differentiable problem of the argmax function.
    - Linear scaling: $\frac{r_t}{|T|}$ balances the update frequency of different rank experts (the low-rank parts are shared by more tasks and updated more frequently, thus requiring downscaled learning rates).

3. **Task Embedding and Contrastive Learning Optimization**:

    - Function: Assign a learnable embedding vector $\mathbf{e}_t$ for each task.
    - Mechanism: Use contrastive learning to ensure that representations of samples from the same task are close to their corresponding task embedding, and far from other task embeddings.
    - Loss function: $$\mathcal{L}_{con} = \frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{e}_t)/\tau)}{\sum_{k=1}^{T}\exp(\text{sim}(\mathbf{h}_i, \mathbf{e}_k)/\tau)}$$

4. **Balanced Data Sampling**:

    - Function: Assign sampling weights $\phi_t = \exp(|\mathcal{D}_t| / \sum_i |\mathcal{D}_i|)$ to each dataset.
    - Design Motivation: In GLUE, MNLI has 392K samples while RTE only has 2.5K. Unbalanced sampling can lead to underfitting on smaller datasets.

### Loss & Training

Total loss function: $\mathcal{L} = \mathcal{L}_{gen} + \lambda \mathcal{L}_{con}$
- $\mathcal{L}_{gen}$: Cross-entropy generation loss
- $\mathcal{L}_{con}$: Contrastive learning loss
- $\lambda = 0.1$, temperature $\tau = 0.05$
- Optimizer: AdamW, learning rate $3 \times 10^{-4}$, linear decay + 500 warmup steps

## Key Experimental Results

### Main Results — GLUE Benchmark (Table)

| Method | Params/Task | MNLI | SST-2 | MRPC | RTE | CoLA | AVG |
|------|-----------|------|-------|------|-----|------|-----|
| LoRA (r=8) | 0.39M | 85.8 | 93.2 | 89.9 | 76.3 | 62.8 | 85.1 |
| MultiLoRA | 1.56M | 85.9 | 94.5 | 88.2 | 80.6 | 66.9 | 86.0 |
| MOELoRA | 0.78M | 86.3 | 94.2 | 89.7 | 81.3 | 68.4 | 86.7 |
| **MoRE** | **0.78M** | **86.2** | **93.7** | **91.2** | **83.5** | **69.9** | **87.3** |

On LLaMA2-7B:

| Method | Params/Task | AVG |
|------|-----------|-----|
| LLaMA2-LoRA | 2.5M | 87.8 |
| LLaMA2-MOELoRA | 5M | 87.3 |
| **LLaMA2-MoRE** | **5M** | **88.8** |

### Commonsense Reasoning Experiment (Table)

| Method | BoolQ | PIQA | OBQA | ARC-E | ARC-C | AVG |
|------|-------|------|------|-------|-------|-----|
| LoRA | 80.9 | 77.7 | 79.0 | 83.7 | 76.9 | 79.6 |
| MixLoRA | 84.3 | 79.5 | 82.6 | 86.8 | 76.3 | 81.9 |
| **MoRE** | **87.2** | **82.3** | **83.0** | **86.7** | **74.2** | **82.7** |

### Ablation Study

| Condition | GLUE AVG |
|------|----------|
| MoRE (Full) | 87.3 |
| w/o Linear Scaling | 87.0 (-0.3) |
| w/o Task Embedding | 86.1 (-1.2) |
| w/o Contrastive Learning Optimization | 86.3 (-1.0) |
| w/o STE | 86.4 (-0.9) |
| w/ Subset Experts | 86.2 (-1.1) |
| w/ Random Sampling | 86.2 (-1.1) |

### Key Findings

1. MoRE achieves a higher average score (87.3 vs 86.7) with the same number of parameters as MOELoRA (0.78M/task), proving that "rank-as-expert" is more efficient than "multi-LoRA parallelization".
2. Task embeddings and contrastive learning are the most critical components—removing them leads to performance drops of 1.2 and 1.0, respectively.
3. Visualization shows that similar tasks (e.g., MRPC and STS-B) are assigned close ranks, while different tasks (e.g., CoLA) are assigned distinct ranks.
4. In few-shot transfer experiments, MoRE reaches 83.8% on 4-shot SciTail, outperforming all baselines.

## Highlights & Insights

1. **Rank-as-Expert**: This concept is simple and elegant—instead of requiring multiple independent LoRA modules, expert differentiation is achieved simply by truncating different rows/columns of the same LoRA. This guarantees natural parameter sharing.
2. **Zero Inference Overhead**: Unlike traditional MoEs, MoRE only needs to use the sub-matrix of the selected rank during inference, incurring no extra routing overhead or additional parameters of parallel LoRAs.
3. **Clever Use of Contrastive Learning**: Utilizing contrastive learning to make task embeddings automatically learn to distinguish task characteristics in the absence of supervised signals is a highly practical technique.

## Limitations & Future Work

1. **Constraint on Maximum Rank**: The maximum rank $r$ is pre-specified. If a certain task requires a rank larger than $r$, the framework cannot accommodate it.
2. **Dependency on Task Definitions**: The model must explicitly know which task the input belongs to in order to look up the task embedding, making it unsuitable for scenarios with blurry task boundaries.
3. **Limited Evaluation Scale**: Experiments are mainly conducted on T5-base and LLaMA2-7B, without validation on larger scale models (13B+).
4. **Actual Distribution of Dynamic Ranks**: The paper does not fully analyze the differences in rank selection across different layers—whether the rank allocation patterns vary across different layers remains uninvestigated.

## Related Work & Insights

- DyLoRA dynamically trains all ranks but is limited to single tasks. MoRE extends this idea to multi-task scenarios and introduces a task-aware selection mechanism.
- Relation to MoE: MoRE draws inspiration from the MoE philosophy, but the experts are not independent modules; instead, they are different truncations of the same matrix, which is more parameter-efficient.
- The linear scaling strategy $r_t/|T|$ shares a similar underlying principle with the $\alpha/r$ scaling in the original LoRA paper.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — "Rank-as-Expert" is a natural yet previously unproposed perspective. The idea of unifying the internal structure of LoRA with MoE is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments on GLUE, commonsense reasoning, few-shot transfer, and ablation studies, validated on two backbone models.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation visually demonstrates that different tasks require different ranks using Table 1, and the method description is very clear.
- **Value**: ⭐⭐⭐⭐ — Highly practical with zero extra inference overhead. The code has been open-sourced, making it directly applicable to multi-task fine-tuning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](../../NeurIPS2025/model_compression/multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)
- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](../../CVPR2025/model_compression/tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)

</div>

<!-- RELATED:END -->
