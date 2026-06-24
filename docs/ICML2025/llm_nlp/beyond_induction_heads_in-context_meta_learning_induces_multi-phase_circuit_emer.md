---
title: >-
  [Paper Note] Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence
description: >-
  [ICML 2025][LLM (Other)][In-Context Learning] This paper designs an In-Context Meta-Learning (ICML) experimental setup to reveal that the internal circuits of Transformers undergo three distinct phases of emergence (Bigram $\rightarrow$ Label Attention $\rightarrow$ Chunk Example) during the training process of acquiring in-context meta-learning capabilities, rather than the single-stage sudden jump observed in prior induction head studies. This provides a new perspective on…
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "In-Context Learning"
  - "Induction Head"
  - "Multi-Phase Circuit"
  - "Meta-Learning"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: ccc6d31e75c11afe
---

# Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence

**Conference**: ICML 2025  
**arXiv**: [2505.16694](https://arxiv.org/abs/2505.16694)  
**Code**: [gouki510/In-Context-Meta-Learning](https://github.com/gouki510/In-Context-Meta-Learning)  
**Area**: LLM/NLP · Mechanistic Interpretability · In-Context Learning  
**Keywords**: In-Context Learning, Induction Head, Multi-Phase Circuit, Meta-Learning, Mechanistic Interpretability  

## TL;DR

This paper designs an In-Context Meta-Learning (ICML) experimental setup to reveal that the internal circuits of Transformers undergo three distinct phases of emergence (Bigram $\rightarrow$ Label Attention $\rightarrow$ Chunk Example) during the training process of acquiring in-context meta-learning capabilities, rather than the single-stage sudden jump observed in prior induction head studies. This provides a new perspective on understanding the deep mechanisms of ICL.

## Background & Motivation

### Background

Transformer language models exhibit remarkable In-Context Learning (ICL) capabilities, adaptively completing new tasks with only a few in-context examples without updating model parameters. Previous mechanistic interpretability studies of ICL primarily focused on **induction heads**, which recognize duplicate patterns $[A][B] \ldots [A]$ in the context and predict $[B]$ via a match-and-copy mechanism. Olsson et al. (2022) found that induction heads emerge through a sudden accuracy jump during training.

### Key Challenge

Induction heads can only explain the "copying answers from context" aspect of ICL. However, actual ICL in LLMs goes far beyond this—the model needs to **infer the task itself** from the in-context examples (e.g., given "country $\rightarrow$ capital" example pairs, inferring that this is a country-to-capital mapping task) and then apply the inferred task to a new query. How this meta-learning capability is acquired during training, and what internal circuits implement it, remains virtually unexplored in prior research.

### Design Motivation

The authors aim to address a fundamental question: **How do Transformers progressively acquire the meta-learning capability of ICL during training?** To this end, it is necessary to go beyond simple copy tasks, design an experimental setup that truly requires task inference, and track the dynamic changes of the model's internal circuits during training.

## Method

### Overall Architecture: In-Context Meta-Learning (ICML) Setup

Building upon the copy task of Reddy (2023), the authors design an experimental framework named **In-Context Meta-Learning (ICML)**. The core idea is to expose the model to **multiple tasks** (different item-label mappings), forcing it to infer the current task from the in-context examples rather than simply copying.

The input sequence format is:

$$\underbrace{x_1, \ell_1^{\tau}, x_2, \ell_2^{\tau}, \ldots, x_N, \ell_N^{\tau}}_{\text{examples}}, \underbrace{x_q}_{\text{query}}, \underbrace{?}_{\text{prediction}}$$

where $\tau$ denotes the current task, and each task defines a different $(x, \ell)$ mapping. The query $x_q$ might not appear in the in-context examples, meaning the model must infer the task $\tau$ to make correct predictions.

### Key Designs

- Each item $x$ and label $\ell$ is represented as a $(P+D)$-dimensional vector, where $P=65$ dimensions correspond to one-hot position encodings and $D=63$ dimensions represent content features.
- Each class $k$ is associated with a $D$-dimensional mean vector $\mu_k$, with elements independently sampled from $\mathcal{N}(0, 1/D)$.
- The actual item vector incorporates noise: $x_i = \frac{\mu_k + \epsilon \eta}{\sqrt{1 + \epsilon^2}}$, where $\epsilon$ controls the intra-class variation.
- Default parameters: $T=3$ tasks, $K=64$ classes, $L=32$ labels, $N=4$ in-context examples, $\epsilon=0.1$, and $p_B=0$ (the query does not appear in the context).

### Network Architecture

A two-layer attention-only Transformer (without FFN) followed by a two-layer MLP classifier is used:

- Each layer has $m$ attention heads, utilizing causal masking.
- Attention weight computation: $p_{ij}^{(\mu,h)} = \frac{\exp((K_\mu^{(h)} u_j)^\top (Q_\mu^{(h)} u_i))}{\sum_{k \leq i} \exp((K_\mu^{(h)} u_k)^\top (Q_\mu^{(h)} u_i))}$
- Both the Query/Key dimensions and the MLP hidden layer dimensions are 128.
- Trained using cross-entropy loss, vanilla SGD, a learning rate of 0.01, and a batch size of 128.

### Key Findings: Three-Phase Circuit Emergence

During training, the model progresses through three distinct phases, with different attention circuits emerging in each phase:

**Phase 1 — Non-Context Circuit (NCC):**
- Both layers are **Bigram** attention: the query token mainly attends to itself.
- The model completely ignores the context, relying solely on weights for memorization.
- Accuracy stagnates at $\sim 1/T$ (about 30-40%).

**Phase 2 — Semi-Context Circuit (SCC):**
- First layer: **Label Attention**—the query attends to the label tokens in the context.
- Second layer: Still **Bigram**—attending to the query itself.
- The model begins to utilize the label information in the context (though without considering the item-label correspondences).
- Accuracy improves to approximately 75%.

**Phase 3 — Full-Context Circuit (FCC):**
- First layer: **Chunk Example**—aggregates $(x, \ell)$ pairs into a single token (similar to a previous token head).
- Second layer: **Label Attention**—attends to the aggregated tokens.
- The model utilizes the full context for task inference.
- Accuracy reaches 100%.

### Circuit Quantization Metrics

The authors define three metrics based on attention maps to quantify circuit emergence:

| Metric | Formula | Meaning |
|------|------|------|
| Bigram | $p_{2N+1, 2N+1}^{\mu,h}$ | Attention of the query token on itself |
| Label Attention | $\sum_{k=1}^{N} p_{2N+1, 2k}^{\mu,h}$ | Total attention of the query on all label tokens in the context |
| Chunk Example | $\frac{1}{N} \sum_{k=1}^{N} p_{2k, 2k-1}^{\mu,h}$ | Average attention of label tokens on their corresponding item tokens |

Phase boundaries are determined by $\Delta \text{Accuracy} = \text{Acc}(t + \Delta t) - \text{Acc}(t) > 0.025$ ($\Delta t = 100$).

### Theoretical Analysis of SCC

Under simplified conditions ($K=L$, no duplicate classes, $T=2$), a theoretical explanation is derived for how the SCC improves accuracy:

When the context contains one of the candidate labels for the query, this label is clearly not the correct answer (as context labels belong to other items), thereby reducing a 50-50 choice to a unique certainty. The probability of the candidate label appearing is:

$$p = 1 - \frac{\binom{K-2}{4}}{\binom{K-1}{4}}$$

Thus, the theoretical accuracy is:

$$\text{Theoretical Accuracy} = p \cdot 1 + (1-p) \cdot 0.5$$

### Loss & Training

The standard cross-entropy loss is used for training the classification task:

$$\mathcal{L} = -\sum_{i=1}^{L} y_i \log \hat{y}_i$$

where $y_i$ is the one-hot ground truth label, and $\hat{y}_i$ is the model's predicted probability for label $i$.

## Key Experimental Results

### Main Results: Three-Phase Accuracy Changes

| Phase | Circuit Type | Layer 1 Attention | Layer 2 Attention | Accuracy (T=3) |
|------|----------|-----------|-----------|---------------|
| Phase 1 | NCC | Bigram | Bigram | 30-40% |
| Phase 2 | SCC | Label Attention | Bigram | ~75% |
| Phase 3 | FCC | Chunk Example | Label Attention | 100% |

### Ablation Study: Impact of Data Properties on Circuit Emergence

| Parameter Change | Effect |
|----------|------|
| $T=1$ | Degenerates into an induction head, with a single-stage jump |
| $T \geq 2$ | Stable emergence of three phases         |
| Small $K$ (32) | Skips Phase 1, directly enters Phase 2 |
| Large $K$ (128, 256) | Skips Phase 2, transitioning directly from Phase 1 to Phase 3 |
| Increasing $\epsilon$ | Skips Phase 2; also skips Phase 1 when $\epsilon=1$ |
| Increasing $\alpha$ (class bias) | Skips Phase 1 or Phase 2 |
| Increasing $\beta$ (task bias) | Little change in overall trend, but significant performance differences among tasks |

### Theoretical Verification of SCC

Under the conditions $K=\{8, 16, 32\}$, the theoretical accuracy highly matches the experimental accuracy, verifying the hypothesis that SCC improves performance through a label-exclusion mechanism.

### Random-Label Accuracy (RLA) Robustness

As training accuracy jumps in Phase 2, the Random-Label Accuracy (RLA) also rises in tandem, indicating that the SCC relies solely on the label set information rather than the item-label pairings. This aligns with the observation by Min et al. (2022b) that "ICL remains effective even with random labels."

### Multi-Head Attention Experiments

| Setup | Phenomenon |
|------|------|
| Single-head | Accuracy shows a clear three-phase jump |
| Two-head | Different heads explore different circuits in parallel (Head 1 $\rightarrow$ NCC, Head 2 $\rightarrow$ FCC), leading to a smooth accuracy increase |
| Hidden circuit emergence | Even with a smooth accuracy curve, circuit metrics still undergo an SCC-like abrupt transition around step 30,000 |

### Empirical Validation on GPT2-XL

On the SST2 sentiment classification 2-shot task, GPT2-XL (48 layers) exhibits hierarchical patterns consistent with the toy model:
- **Early layers**: High Chunk Example metrics (aggregating the review-label pairs).
- **Mid-to-late layers**: High Label Attention metrics (utilizing label information for prediction).

## Highlights & Insights

1. **Discovery of Multi-Phase Emergence**: Unlike the single-stage sudden jump of induction heads, the acquisition of meta-learning capability requires the ordered emergence of three circuit phases (NCC $\rightarrow$ SCC $\rightarrow$ FCC), revealing the complexity of ICL skill acquisition.
2. **SCC Explains the Random-Label Mystery**: The Semi-Context Circuit focuses only on the label set while ignoring the item-label relationships, providing a circuit-level explanation for the finding of Min et al. (2022b) that "ICL remains effective under random labels."
3. **Hidden Circuit Emergence**: Multi-head attention smooths the accuracy curve, but circuit metrics reveal that abrupt internal transitions still occur—implying that even if the loss decreases smoothly during LLM training, the internal circuits may undergo drastic structural shifts.
4. **A Bridge from Toy Models to LLMs**: Demonstrating a similar hierarchical circuit pattern in GPT2-XL enhances the practical value of the research conclusions.
5. **Unifying Induction Heads**: When $T=1, p_B=1$, the ICML setup degenerates into the induction head configuration, indicating that the induction head is a special case of a more general meta-learning circuit.

## Limitations & Future Work

1. **Extremely Small Model Scale**: The primary experiments are based on a 2-layer attention-only Transformer, which is much smaller than real-world LLMs. The scalability of the conclusions requires further verification.
2. **Synthetic Data**: The simplified classification task used deviates significantly from the complexity of natural language ICL tasks.
3. **Lack of FFN Layers**: The attention-only architecture overlooks the potential role that FFNs might play in ICL (such as storing memorized knowledge).
4. **Limited GPT2-XL Validation**: Consistency of the attention patterns is validated only on the SST2 task, lacking broader task and model coverage.
5. **Questionable Universality of the Three-Phase Pattern**: Under different data distribution parameters, certain phases may be skipped, suggesting that the three-phase pattern is not a universal law.
6. **Lack of Causal Intervention**: The circuit analysis is primarily based on correlational observations of attention patterns and does not employ causal methods like activation patching to stringently verify circuit functionality.

## Related Work & Insights

- **In-Context Learning Theory**: Von Oswald et al. (2023) demonstrated that Transformers can implement in-context linear regression via meta-gradient descent; Xie et al. (2021) interpreted ICL as implicit Bayesian inference.
- **Induction Head**: Olsson et al. (2022) identified induction heads as a critical circuit for ICL; Singh et al. (2024) investigated the emergence conditions of induction heads under multi-head attention.
- **Circuit Discovery & Interpretability**: Wang et al. (2022) discovered the circuit for indirect object identification in GPT-2; Conmy et al. (2023) explored automated circuit discovery.
- **Random Labels & ICL**: Min et al. (2022b) showed that ICL maintains performance under random labels; Chan et al. (2022) revealed the impact of data distribution properties on ICL emergence.
- **Task Vectors**: Hendel et al. (2023) and Todd et al. (2024) found that LLMs internally represent tasks as vector forms.
- **Grokking & Phase-based Learning**: Nanda et al. (2023) and Furuta et al. (2024) studied the relation between the grokking phenomenon in Transformers and internal circuits.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Rigor | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐ |
| Clarity | ⭐⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

> This paper makes a key contribution to the field of mechanistic interpretability by extending induction head research to meta-learning scenarios, discovering the phenomenon of multi-phase circuit emergence, and providing circuit-level explanations for several known puzzles in ICL (e.g., robustness to random labels). The experimental design is elegant, the analysis is clear, and the theoretical derivations align closely with the experimental results. The primary limitation is the small model scale, but the authors conduct preliminary cross-model validation on GPT2-XL, demonstrating the potential bridge from toy experiments to actual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Prompt-MII: Meta-Learning Instruction Induction for LLMs](../../ICLR2026/llm_nlp/prompt-mii_meta-learning_instruction_induction_for_llms.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](../../NeurIPS2025/llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[ACL 2025\] Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](../../ACL2025/llm_nlp/beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)
- [\[NeurIPS 2025\] System Prompt Optimization with Meta-Learning](../../NeurIPS2025/llm_nlp/system_prompt_optimization_with_meta-learning.md)
- [\[ACL 2025\] Beyond In-Context Learning: Aligning Long-form Generation of LLMs via Task-Inherent Attribute Guidelines](../../ACL2025/llm_nlp/beyond_in-context_learning_aligning_long-form_generation_of_large_language_model.md)

</div>

<!-- RELATED:END -->
