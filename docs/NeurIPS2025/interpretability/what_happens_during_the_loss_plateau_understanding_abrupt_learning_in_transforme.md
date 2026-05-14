---
title: >-
  [Paper Note] What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers
description: >-
  [NeurIPS 2025][Interpretability][abrupt learning] This paper systematically investigates the phenomenon of "abrupt learning" in Transformer training…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "abrupt learning"
  - "loss plateau"
  - "Transformer training dynamics"
  - "representation collapse"
  - "repetition bias"
  - "attention map"
date: 2026-05-08
content_hash: b76be2f887637f44
---

# What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2506.13688](https://arxiv.org/abs/2506.13688)
**Code**: [github.com/pulkitgopalani/tf-loss-plateau](https://github.com/pulkitgopalani/tf-loss-plateau)
**Area**: Interpretability
**Keywords**: abrupt learning, loss plateau, Transformer training dynamics, representation collapse, repetition bias, attention map

## TL;DR

This paper systematically investigates the phenomenon of "abrupt learning" in Transformer training, revealing that during the loss plateau the model has already learned partial solutions while simultaneously exhibiting output repetition bias and representation collapse. It further demonstrates that the slow learning of attention maps constitutes the key bottleneck, with findings validated in the early pretraining stages of LLMs such as Pythia and OLMo.

## Background & Motivation

**Abrupt learning** is a common phenomenon in Transformer training: model performance stagnates during a prolonged loss plateau before suddenly improving dramatically. This phenomenon is regarded as a manifestation of "emergence."

Although prior work has studied this phenomenon in specific settings (e.g., in-context learning, sparse parity, syntactic learning), it lacks:

**A unified understanding**: Common patterns and mechanisms across tasks remain unclear.

**Generality**: Most prior work relies on task-specific assumptions, limiting the generalizability of conclusions.

**Connection to LLMs**: Whether findings from small-scale experiments transfer to large-model pretraining has not been verified.

Three core questions addressed in this paper:
- What common patterns exist in the input–output behavior and internal representations of models during the plateau?
- What hidden progress accumulates prior to the abrupt improvement?
- Are these findings universal?

## Method

### Overall Architecture

The primary research platform consists of shallow Transformers (1–2 layers) trained on algorithmic tasks, supplemented by validation on LLM pretraining. The core algorithmic task is the **Moving Window Sum (MWS)**:

Given input sequence $x_1, x_2, \ldots, x_n$, the output $y_i$ is:

$$y_i = \begin{cases} x_1 & i=1 \\ (x_{i-1} + x_i) \bmod p & i \geq 2 \end{cases}$$

Experimental setup: $n=16, p=17$, using a 1-layer 1-head linear-attention Transformer.

### Key Finding 1: Partial Solutions During the Plateau

During the loss plateau, the model has already learned to correctly predict a **subset of simpler tokens**. For instance, in the MWS task, $y_1 = x_1$ requires only copying; the model learns to predict the first output token correctly at an early stage while subsequent tokens remain incorrect. This pattern appears consistently across multiple algorithmic tasks.

### Key Finding 2: Repetition Bias

During the plateau, the model tends to generate repeated tokens, e.g., $x, x, x, \ldots$. The **repetition frequency** is defined as:

$$\rho = \frac{1}{n-1} \sum_{i=1}^{n-1} \mathbf{1}[y_i = y_{i+1}]$$

In early training (approximately the first 50 steps), $\rho$ rapidly rises from near 0 to approximately 0.8, indicating that this is an implicit bias introduced by gradient-based training.

For tasks without consecutive repetitions (e.g., prefix sums), the repetition bias manifests differently — only a small number of distinct tokens appear in the output. This is quantified using **sequence entropy**:

$$\text{SeqEnt}(y_1, \ldots, y_n) = \sum_{i=1}^{|V|} p_i \log(1/p_i)$$

### Key Finding 3: Representation Collapse

Hidden representations at different output positions become nearly parallel. This is measured via pairwise cosine similarity:

$$\text{COS}_{i,j} = \frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\| \|\mathbf{h}_j\|}$$

In early training, the average cosine similarity across output positions (excluding the already-learned first position) rapidly rises to approximately 0.95, then drops significantly as performance improves.

An important distinction: this differs from the initialization-time rank collapse observed in deep softmax Transformers — the representation collapse studied here occurs several training steps after initialization, not at initialization, and uses shallow linear attention.

### Key Finding 4: Attention Map Learning as the Key Bottleneck

**Attention Progress Metric (APM)**:

$$\text{APM} = \frac{\sum_{(i,j) \in \Omega} |A_{ij}|}{\sum_{(i,j)} |A_{ij}|}$$

where $\Omega$ denotes the set of positions in the optimal attention map. APM **monotonically increases** during the plateau, from near 0 to approximately 0.8, and its growth is smoother than changes in loss or accuracy — indicating that the attention map makes steady progress even when the loss appears stagnant.

### Intervention: Biasing the Attention Map

By applying a multiplicative mask $M$ to the attention map during training — scaling entries at optimal positions $(i,j) \in \Omega$ by a factor $c$:

- **$c > 1$ (biased toward optimal)**: Representation collapse is reduced (peak cosine drops from 0.95 to approximately 0.6), repetition frequency is lower, and convergence is faster.
- **$0 < c < 1$ (biased away from optimal)**: The plateau is prolonged, and both representation collapse and repetition bias worsen.

### Optimal Initialization Experiments

- Fixing attention layer weights to their optimal values → almost no plateau, rapid convergence, no representation collapse.
- Fixing the MLP or Embedding to optimal values → training dynamics are nearly unchanged; plateau and collapse still occur.

**Conclusion**: The attention map is the primary bottleneck responsible for representation collapse and the loss plateau.

### Loss & Training

- Adam optimizer, learning rate $10^{-4}$, no weight decay.
- Online training (256 new samples drawn per step), no fixed training set.
- Consistency of findings verified across different optimizers (SGD, Muon) and hyperparameter settings.

## Key Experimental Results

### Main Results: Multi-Task Validation

| Task | Partial Solution | Repetition Bias | Representation Collapse |
|------|-----------------|-----------------|------------------------|
| MWS (Moving Window Sum) | $y_1 = x_1$ copy | $\rho \approx 0.8$ | COS $\approx 0.95$ |
| Prefix Sum | First token | Low sequence entropy | COS $\approx 0.80$ |
| Multi-digit Addition | Final carry | Present | Present |
| Permutation | Partial positions | Present | Present |
| Histogram | Partial bins | Present | Present |

All three phenomena are consistently observed across all tasks, confirming the universality of the findings.

### Ablation Study: Attention Intervention

| Intervention Condition | Peak COS | Convergence Time | Peak Repetition Frequency |
|-----------------------|----------|-----------------|--------------------------|
| $c = 1$ (baseline) | $\approx 0.95$ | Normal | $\approx 0.8$ |
| $c = 10$ (strongly biased toward optimal) | $\approx 0.6$ | Significantly faster | Low |
| $c = 0.2$ (biased away from optimal) | $> 0.95$ | Significantly delayed | High and persistent |
| KQ fixed at optimal | $\approx 0.45$ | Very fast | Low |
| KQOV fixed at optimal | $\approx 0.15$ | Fastest | Very low |
| MLP fixed at optimal | $\approx 0.95$ | Essentially unchanged | Essentially unchanged |

### LLM Validation

| Model | Early Peak COS | Repetition Bias |
|-------|---------------|-----------------|
| Pythia-14M | $> 0.9$ | Present |
| Pythia-1B | $> 0.9$ | Present |
| Pythia-1.4B | $> 0.9$ | Present |
| Pythia-2.8B | $> 0.9$ | Present |
| OLMo-2 7B (step 150) | $\approx 0.93$ | Present |
| OLMo-2 7B (step 600) | $\approx 0.43$ | Dissipated |

### Key Findings

1. **Repetitive sequences are easier to learn**: The REPEAT1 task ($y_i = x_1$) exhibits almost no loss plateau; a single gradient step drives COS to approximately 0.5.
2. **The $\alpha_1$ metric** ($\frac{1}{n}\sum \mathbf{1}[y_i = y_1]$) rapidly approaches 1.0 in early training — the model tends to copy the first token to all positions.
3. **The attention layer, not the MLP, causes collapse**: Comparison of residual streams before and after the attention layer confirms that collapse occurs **after** the attention layer.
4. **APM shows progress before the loss drops**: Hidden progress accumulates during the plateau before any visible improvement in loss.

## Highlights & Insights

1. **Unified perspective**: This work is the first to systematically understand Transformer training dynamics by treating partial solutions, repetition bias, representation collapse, and attention learning as an integrated whole.
2. **Causal validation**: Through attention map intervention experiments, the paper establishes not merely correlation but a causal chain: attention learning → collapse/repetition → loss plateau.
3. **"Search" hypothesis**: The plateau is likened to a search process by the attention layer over the token space, analogous to the first layer of a two-layer network aligning with the support of the target function.
4. **Bridging small models and LLMs**: Validation of the same phenomena in the early pretraining of Pythia and OLMo substantially strengthens the generalizability of the conclusions.
5. **Methodological clarity**: Multiple quantitative metrics — APM, repetition frequency, sequence entropy — render abstract training dynamics measurable and comparable.

## Limitations & Future Work

1. **Insufficient theoretical explanation**: Why does gradient training introduce representation collapse bias? Only a "search" hypothesis is proposed, without rigorous theoretical verification.
2. **Dominated by small models**: Primary conclusions are based on 1–2 layer Transformers; how layers interact in deeper multi-layer models remains unclear.
3. **Specificity of algorithmic tasks**: The tasks used have well-defined optimal solutions; how findings generalize to settings without clear optima, such as natural language, requires further investigation.
4. **Linear attention limitations**: Although some conclusions are partially verified on softmax attention, properties specific to linear attention may limit the generalizability of certain findings.
5. **Limited LLM validation**: Only early checkpoints are examined for representation collapse; partial solutions and attention progress in LLMs are not analyzed in depth.

## Related Work & Insights

- **Distinction from grokking**: Grokking refers to post-memorization generalization on a fixed dataset, whereas this paper studies the breakthrough from a plateau in online training — the mechanisms differ.
- **Distinction from rank collapse**: Rank collapse occurs at initialization in deep softmax Transformers, while the representation collapse studied here is training-induced and occurs in shallow linear-attention models.
- **Implications for LLM training**: Understanding early training dynamics can inform better learning rate schedules, attention initialization strategies, and curriculum learning designs.
- **Attention transfer**: The findings are consistent with the effectiveness of pretrained attention transfer in vision Transformers, suggesting that attention patterns constitute transferable knowledge across tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work to unify the multi-dimensional characteristics of the loss plateau and establish the central role of attention learning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-task validation, intervention experiments, diverse metrics, and LLM verification; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logic, polished figures, and cleverly designed quantitative metrics.
- Value: ⭐⭐⭐⭐⭐ — Provides a foundational contribution to understanding Transformer training dynamics with practical implications for LLM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](../../ICLR2026/interpretability/towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)

</div>

<!-- RELATED:END -->
