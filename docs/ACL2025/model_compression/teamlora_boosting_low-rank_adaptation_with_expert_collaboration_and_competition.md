---
title: >-
  [Paper Note] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition
description: >-
  [ACL 2025][Model Compression][Parameter-Efficient Fine-Tuning] TeamLoRA is proposed to optimize the Multi-LoRA architecture through an asymmetric collaboration module (a "plug-and-play" structure with shared A matrices and multiple expert B matrices) and a competition module based on Shapley values. This achieves a better performance-efficiency trade-off in multi-task learning—reducing training time by 30% and increasing inference speed by 40% compared to MoELoRA…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Parameter-Efficient Fine-Tuning"
  - "LoRA"
  - "Mixture-of-Experts"
  - "Multi-Task Learning"
  - "Game Theory"
date: 2026-05-08
content_hash: 0805b6d5e35f526b
---

# TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition

**Conference**: ACL 2025  
**arXiv**: [2408.09856](https://arxiv.org/abs/2408.09856)  
**Code**: [https://github.com/Lin-Tianwei/TeamLoRA](https://github.com/Lin-Tianwei/TeamLoRA)  
**Area**: Model Compression/Parameter-Efficient Fine-Tuning  
**Keywords**: Parameter-Efficient Fine-Tuning, LoRA, Mixture-of-Experts, Multi-Task Learning, Game Theory

## TL;DR

TeamLoRA is proposed to optimize the Multi-LoRA architecture through an asymmetric collaboration module (a "plug-and-play" structure with shared A matrices and multiple expert B matrices) and a competition module based on Shapley values. This achieves a better performance-efficiency trade-off in multi-task learning—reducing training time by 30% and increasing inference speed by 40% compared to MoELoRA, while achieving superior performance.

## Background & Motivation

**Background**: LoRA is currently the most popular parameter-efficient fine-tuning (PEFT) method, adding a trainable bypass over pre-trained weights through low-rank decomposition $\Delta W = AB$. However, LoRA performs poorly in multi-dimensional task scenarios, primarily due to catastrophic forgetting and interference between different tasks. Multi-LoRA (MoELoRA) introduces multiple LoRA experts and a routing mechanism to tackle multi-task scenarios, but introduces new issues.

**Limitations of Prior Work**: MoELoRA suffers from two major problems: (1) **Low efficiency**—$k$ experts introduce approximately $2k$ times additional matrix operations. When $k=4$, training is 62% slower than LoRA, and 138% slower when $k=8$; (2) **Poor expert combination performance**—the routing mechanism suffers from load imbalance and overconfidence. Experiments find that keeping only the Top-1 best expert can achieve 98.5% of the performance of all experts, indicating that a large number of experts learn redundant knowledge.

**Key Challenge**: The symmetric structure of MoELoRA (where each expert has independent A and B matrices) not only wastes computational resources (redundant matrix operations) but also leads to knowledge redundancy (independent experts learning similar features), violating the original intent of PEFT being "efficient."

**Goal**: Design a Multi-LoRA architecture that simultaneously optimizes efficiency and performance, addressing both redundant computation and expert coordination.

**Key Insight**: It is observed that the A and B matrices in LoRA have a hierarchical relationship in terms of function—the A matrix is responsible for general feature projection, while the B matrix is responsible for task-specific knowledge. Therefore, multiple experts can "share" the A matrix (collaboration) while using Shapley values from game theory to refine expert weight distribution (competition).

**Core Idea**: Organize multiple LoRA experts into a "Team"—achieving efficient collaboration through a shared A matrix and split expert B matrices, and effective competition through an interaction matrix based on Shapley values.

## Method

### Overall Architecture

TeamLoRA adds an asymmetric LoRA bypass to each linear layer. The input $x$ first passes through a shared general matrix $A \in \mathbb{R}^{d_{in} \times r_A}$ ($r_A = k \cdot r_B$). The intermediate representation is evenly split into $k$ segments, each of which passes through its corresponding expert matrix $B_i \in \mathbb{R}^{r_B \times d_{out}}$. The outputs of each expert are weighted and summed using weights calculated by the competition module, and then added to the output of the pre-trained weights as a bypass.

### Key Designs

1. **Efficient Collaboration**:

    - **Function**: Share and organize knowledge while reducing computational overhead.
    - **Mechanism**: Define a shared general module $A \in \mathbb{R}^{d_{in} \times r_A}$ and $k$ expert modules $B_i \in \mathbb{R}^{r_B \times d_{out}}$ (where $r_A = k \cdot r_B$). The input passes through $A$ to obtain $z = xA$, and then $z$ is evenly split along the last dimension into $k$ segments $z_i = \text{split}(z)_i$. Each segment passes through its corresponding $B_i$ to yield the expert output $h_i = z_i \cdot B_i$. This "split" operation allows each expert to process only $r_B$-dimensional intermediate representations instead of the full $r_A$ dimensions.
    - **Design Motivation**: The A matrix captures cross-task general homogeneous features (domain-agnostic), while the B matrix acts as a "plug-in" to capture task-specific knowledge (domain-specific). Compared to the $k$ independent $(A_i, B_i)$ pairs in MoELoRA, TeamLoRA only requires one A matrix multiplication and $k$ small B matrix multiplications. When $k=2/4/8$, the training times are 87%/70%/63% of MoELoRA, respectively.

2. **Effective Competition via Shapley Values**:

    - **Function**: Replace traditional Softmax routing, refining expert weight allocation from a game-theoretic perspective.
    - **Mechanism**: Introduce the concept of fuzzy Shapley values, allowing experts to participate with continuous degrees from 0 to 1, instead of traditional binary selection. An MLP is used to approximate the Shapley value calculator $\phi_i(x; \theta_S) \leftarrow \text{Softmax}(S(x; \theta_S))_i$. A learnable interaction matrix $M$ (initialized to a uniform distribution with diagonal elements as 1) is then introduced to capture the competitive relationships among experts: $\omega_i = \sum_{j=1}^{k} M_{ij} \phi_j(x; \theta_S)$. The final output is $h = xW_0 + \mathcal{M}_{col}(x; A, \{B_i\}) \odot \mathcal{M}_{cop}(x; \theta_S, M)$.
    - **Design Motivation**: Traditional Softmax routing can lead to weight collapse (few experts dominate) and load imbalance. Shapley values consider the "cooperative effect" among experts—evaluating the "average marginal contribution" under all possible combinations of other experts, thereby achieving fairer and more effective knowledge transfer.

3. **Comprehensive Multi-task Evaluation (CME)**:

    - **Function**: Comprehensively evaluate the multi-task learning capability of PEFT methods.
    - **Mechanism**: Integrate 22 datasets with a total of 2.5 million training samples, covering 11 evaluation tasks (text summarization, sentiment analysis, natural language inference, paraphrase detection, textual entailment, commonsense reasoning, scientific reasoning, open-domain QA, reading comprehension, knowledge reasoning, etc.).
    - **Design Motivation**: Existing evaluations are typically tested on only a few tasks, failing to comprehensively reflect multi-task learning capabilities.

### Loss & Training

Cross-entropy loss is used for training, and only the auxiliary module parameters (A matrix, B matrix, and competition module parameters) are updated. The base model is Chinese LLaMA-2-7B (which extends LLaMA-2 with Chinese vocabulary and a general corpus). All LoRA methods insert parameters only into the FFN modules. Experiments are conducted on 8×A800 GPUs.

## Key Experimental Results

### Main Results (CME Benchmark)

| Method | MoE? | Rank | Training Time | Param % | Avg Score |
|--------|------|------|------|------|------|
| LoRA | ✗ | 32 | 25h | 0.67% | 57.44 |
| LoRA | ✗ | 128 | 26h | 2.68% | 58.81 |
| MoELoRA | ✓ | 32 | 42h | 2.71% | 59.69 |
| HydraLoRA | ✓ | 32 | 34h | 1.84% | 59.06 |
| TeamLoRA | ✓ | 32 | **29h** | 2.71% | **60.29** |
| TeamLoRA | ✓ | 16 | 28h | 1.35% | 59.95 |

### Ablation Study

| Collaboration Module | Competition Module | Avg Score (r=32) |
|------|---------|------|
| ✗ | ✗ | 59.69 (MoELoRA Baseline) |
| ✓ | ✗ | 59.77 (+0.08) |
| ✗ | ✓ | 60.24 (+0.55) |
| ✓ | ✓ | **60.29** (+0.60) |

### Key Findings

- TeamLoRA (Rank=32) outperforms MoELoRA (42h) with a training time of 29h, while achieving a higher average score (60.29 vs 59.69)—being both faster and better.
- TeamLoRA (Rank=16) requires only half the parameters of MoELoRA (1.35% vs 2.71%) but achieves almost identical performance (59.95 vs 59.69), demonstrating extreme parameter efficiency.
- The contribution of the competition module (+0.55) is significantly larger than that of the collaboration module (+0.08), showing that addressing routing issues is more critical than optimizing computation structures.
- Validation on Llama-3-8B consistently shows superior performance over MoELoRA (55.42 vs 54.56).
- It is equally effective on multimodal LLaVA-1.5-7B (60.44 vs 59.80), demonstrating good generalization.
- Performance improves continuously as the number of experts increases from 1 to 4, and drops slightly at 8—making 4 experts the optimal configuration.
- The expert redundancy issue in MoELoRA is severe: among 4 independent experts, the Top-1 expert alone achieves 98.5% of the total performance.
- The load balancing of TeamLoRA is significantly superior to MoELoRA, demonstrating more uniform expert utilization across 57 tasks in MMLU.

## Highlights & Insights

- The design of **asymmetric A/B division** is simple yet effective: the shared A learns general knowledge, and split B matrices learn specific knowledge. This "backbone + plug-in" concept can be extended to other scenarios requiring multi-expert collaboration. The split operation itself introduces no extra parameters, reducing computation purely through structural reorganization.
- Substituting Softmax routing with **Shapley values** offers an interesting perspective—modeling expert selection as "value allocation in cooperative games," considering the interaction effects among experts. Although the actual implementation relies on MLP approximation, the theoretical framework provides a better inductive bias.
- The source of efficiency gains is clear: MoELoRA requires $2k$ matrix multiplications, whereas TeamLoRA only requires $1 + k$ (1 shared A + $k$ small B's). This reduces matrix operations by approximately 37% when $k=8$.

## Limitations & Future Work

- The accuracy of approximating Shapley values using an MLP in the competition module is limited, and it introduces additional parameters (albeit very few).
- Currently, LoRA parameters are only added to the FFN layers, without exploring adaptation for attention layers.
- When Rank=256, the performance of both TeamLoRA and MoELoRA drops significantly, suggesting that additional regularization might be needed in large rank scenarios.
- Although the CME benchmark covers 11 types of tasks, it is mainly NLP-focused and lacks evaluation on reasoning-intensive tasks (such as mathematics and programming).
- The interpretability analysis of the interaction matrix $M$ could be deeper.

## Related Work & Insights

- **vs MoELoRA**: TeamLoRA completely dominates in efficiency (30% faster training, 40% faster inference) and consistently performs better, primarily by avoiding redundant A-matrix calculations and improving the routing mechanism.
- **vs HydraLoRA**: HydraLoRA also attempts to share the A matrix but does not perform splitting, leading to longer training times (34h vs 29h), more parameters (1.84%) than TeamLoRA-16 (1.35%), and lower performance.
- **vs AdaLoRA**: Optimizes LoRA by adaptively adjusting ranks but fails to address multi-task coordination, achieving performance comparable to LoRA-128.
- **vs MoSLoRA**: Provides improvements similar to MoELoRA from a matrix decomposition perspective, but offers limited improvements in training efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The asymmetric design of the collaboration module is intuitive and clear, and the Shapley value perspective for the competition module is novel, though the overall work is an incremental improvement over MoELoRA.
- Experimental Thoroughness: ⭐⭐⭐⭐ The CME benchmark is comprehensive, with multi-model and multi-modal validation and detailed efficiency analysis. However, individual evaluations on more downstream tasks are lacking.
- Writing Quality: ⭐⭐⭐⭐ The presentation is clear with rich figures and tables, but the formalization in the game theory section is slightly verbose.
- Value: ⭐⭐⭐⭐ Highly valuable for practical applications of PEFT in multi-task learning scenarios; the solution can be directly integrated into existing training frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](../../NeurIPS2025/model_compression/gora_gradient-driven_adaptive_low_rank_adaptation.md)

</div>

<!-- RELATED:END -->
