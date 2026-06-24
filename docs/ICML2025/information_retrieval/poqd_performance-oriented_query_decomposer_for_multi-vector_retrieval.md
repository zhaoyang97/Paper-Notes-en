---
title: >-
  [Paper Note] POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval
description: >-
  [ICML 2025][Information Retrieval & RAG][Multi-Vector Retrieval] POQD, a performance-oriented query decomposition framework, is proposed. It utilizes an LLM-based Prompt Optimizer to iteratively optimize query decomposition prompts, and jointly optimizes the prompts and downstream RAG model parameters through an alternating training algorithm, significantly outperforming existing methods on retrieval and end-to-end QA tasks.
tags:
  - "ICML 2025"
  - "Information Retrieval & RAG"
  - "Multi-Vector Retrieval"
  - "Query Decomposition"
  - "RAG"
  - "Prompt Optimization"
  - "LLM-based Optimizer"
date: 2026-05-08
content_hash: 5b66c7d7d6f17928
---

# POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval

**Conference**: ICML 2025  
**arXiv**: [2505.19189](https://arxiv.org/abs/2505.19189)  
**Code**: [PKU-SDS-lab/POQD-ICML25](https://github.com/PKU-SDS-lab/POQD-ICML25)  
**Area**: Information Retrieval  
**Keywords**: Multi-Vector Retrieval, Query Decomposition, RAG, Prompt Optimization, LLM-based Optimizer

## TL;DR

POQD, a performance-oriented query decomposition framework, is proposed. It utilizes an LLM-based Prompt Optimizer to iteratively optimize query decomposition prompts, and jointly optimizes the prompts and downstream RAG model parameters through an alternating training algorithm, significantly outperforming existing methods on retrieval and end-to-end QA tasks.

## Background & Motivation

**Multi-Vector Retrieval (MVR)** calculates similarity based on the MaxSim operator by decomposing queries and documents into fine-grained units (e.g., tokens/phrases), showing excellent performance in information retrieval tasks. A representative method, ColBERT, decomposes queries at the token level. However, the authors find that:

**Fatal Defect of Token-Level Decomposition**: Decomposing queries by tokens in ColBERT leads to semantic ambiguity. For example, the query "Hong Kong" is split into two tokens, "Hong" and "Kong", where "Kong" might match King Kong (the gorilla), leading to the retrieval of completely irrelevant documents. Treating "Hong Kong" as an entire phrase enables correct retrieval.

**Poor Quality of Manual Prompt Decomposition**: Using hand-crafted prompts to let LLMs decompose queries (such as ICL-QD) can generate irrelevant sub-queries (e.g., generating uninformative words like "type"), which interferes with the MaxSim similarity calculation and causes retrieval errors.

**Core Optimization Challenges**: (a) The sub-query search process is non-differentiable, making gradient backpropagation impossible; (b) evaluating candidate sub-queries requires training the full downstream model, which incurs huge computational overheads.

These issues lead to the core research question: **How to automatically generate sub-queries at an arbitrary granularity to maximize the performance of downstream retrieval systems?**

## Method

### Overall Architecture

POQD consists of two core LLMs and an alternating training algorithm:

- **Query Decomposer**: Takes a prompt $p$ and an input query $Q$, and outputs a set of decomposed sub-queries $\{q_i\}_{i=1}^K$
- **Prompt Optimizer**: Iteratively generates better prompt prefixes $p_0$ based on historical solution-score pairs
- **End-to-End Training (Algorithm 2)**: Alternatingly optimizes the prompt $p$ and downstream model parameters $\Theta$

The global loss function is defined as:

$$\mathcal{L}(\Theta; p) = -\log\left(\sum_{D \in D_K} P_\theta(a|Q,D) P_\beta(D|Q)\right)$$

where $D_K$ is the top-K retrieved documents based on MVR similarity, $P_\theta$ is the answer likelihood of the generator model, and $P_\beta$ is the document relevance probability of the retrieval model. The retrieval model $\beta$ remains frozen, and only the generator $\theta$ is trained.

### Key Designs

#### 1. MVR Similarity Calculation (MaxSim)

For query $Q$ decomposed into sub-queries $\{q_i\}_{i=1}^K$ and document $D$ decomposed into segments $\{d_j\}_{j=1}^m$:

$$\text{SIM}_\theta(Q, D) = \frac{1}{K} \sum_{i=1}^K \max_{1 \le j \le m} E_\theta(q_i)^\top E_\theta(d_j)$$

Core Idea: Each sub-query matches the most similar segment in the document, and the maximum similarities of all sub-queries are aggregated as the overall score.

#### 2. Query Decomposition Optimization (Algorithm 1)

Given fixed downstream model parameters $\Theta$, the optimal prompt is iteratively searched via a two-step process:

- **Step 1 - Prompt Generation**: The Prompt Optimizer takes two pieces of meta-prompts and a historical solution-score pair list $LS$ to generate a candidate prompt prefix $p_0$. Then $p_0$ is concatenated with a fixed task description template to form the complete prompt $p$.
- **Step 2 - Evaluation and Recording**: The Query Decomposer is driven by $p$ to decompose all queries in the training set, computing the training loss $\mathcal{L}(\Theta; p)$ and adding $(p, \mathcal{L}(\Theta; p))$ to $LS$.

**Convergence Condition**: When $\mathcal{L}(\Theta; p) - \mathcal{L}(\Theta; p^{\text{old}}) \le \alpha$ or iterations reach $\kappa$ times.

**Hallucination Filtering**: The Query Decomposer may generate tokens not present in the original query. POQD removes these irrelevant tokens through a filtering step.

#### 3. End-to-End Alternating Training (Algorithm 2)

```text
Input: Training query set Q^train, downstream model parameters Θ
Initialize random p^old
while not converged:
    Call Algorithm 1 to get new prompt p^new and optimized sub-queries
    if p^new == p^old: break
    Train Θ for τ steps using optimized sub-queries (minimize L(Θ; p^new))
    p^old ← p^new
Finally: Fix p^new, train Θ until convergence
```

Key Design Point: Instead of training $\Theta$ to convergence in each round, it is trained for only $\tau$ steps (default $\tau=3$), greatly saving computational cost.

### Loss & Training

**Theoretical Guarantee (Theorem 4.4)**: Under the $\mu$-PL* condition and $L$-smoothness assumption, when the prompt is updated from $p^{\text{old}}$ to $p^{\text{new}}$:

$$\mathcal{L}(\Theta^*(p^{\text{old}}); p^{\text{old}}) - \mathcal{L}(\Theta^*(p^{\text{new}}); p^{\text{new}}) \ge \alpha - (1 - \frac{\mu}{2L})^\tau M$$

where $M$ is the upper bound of the loss. Since $(1 - \frac{\mu}{2L}) \in (0, 1)$, appropriately setting $\tau$ (e.g., $\tau = \log_{1-\frac{\mu}{2L}}(\frac{\alpha}{2M})$) guarantees that the convergence loss strictly decreases. The default $\tau=3$ achieves a balance between efficiency and performance.

**Weakly Supervised Optimization**: POQD optimizes the end-to-end performance of downstream RAG rather than intermediate retrieval metrics, thus operating in a weakly supervised manner. This allows it to work effectively in scenarios like multi-hop QA where queries are dynamically generated.

**Default Hyperparameter Configuration**: $\alpha=0.02$, $\tau=3$, $\kappa=5$.

## Key Experimental Results

### Main Results

**Retrieval Accuracy (Hit@K)**:

| Dataset | Metric | POQD | ColBERT-orig | ICL-QD | Dense Retrieval | Max Gain |
|--------|------|------|-------------|--------|----------------|---------|
| WebQA (Image) | Hit@1 | **42.33** | 38.95 | 39.39 | 41.38 | +0.64 |
| MultiModalQA (Image) | Hit@1 | **58.26** | 36.96 | 54.34 | 50.00 | +3.92 |
| ManyModalQA (Image) | Hit@1 | **28.67** | 21.05 | 27.76 | 27.38 | +0.78 |
| WebQA (Text) | Hit@2 | **53.24** | 52.16 | 41.37 | 43.52 | +1.08 |
| MultiModalQA (Text) | Hit@2 | **80.58** | 79.89 | 71.43 | 66.44 | +0.69 |
| ManyModalQA (Text) | Hit@2 | **92.35** | 87.07 | 85.14 | 49.25 | +5.28 |

**End-to-End QA Accuracy (Exact Match)**:

| Dataset | POQD | ColBERT-orig | ICL-QD | w/o RAG | Max Gain |
|--------|------|-------------|--------|---------|---------|
| WebQA (Image) | **82.83** | 81.96 | 82.37 | 80.31 | +0.46 |
| MultiModalQA (Image) | **61.74** | 49.13 | 46.78 | 16.52 | **+12.61** |
| ManyModalQA (Image) | **37.92** | 32.29 | 34.70 | 30.98 | +3.22 |
| WebQA (Text) | **62.22** | 61.14 | 58.63 | 56.47 | +1.08 |
| MultiModalQA (Text) | **68.10** | 61.73 | 61.86 | 40.36 | +6.24 |
| ManyModalQA (Text) | **81.27** | 77.66 | 76.69 | 32.28 | +3.61 |
| StrategyQA (Text) | **75.55** | 65.50 | 60.70 | 58.76 | +10.05 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $\alpha=0.01$ | Slow convergence | The required loss reduction for prompt updates per round is too small, leading to low optimization efficiency |
| $\alpha=0.02$ (Default) | Best balance | Smooth decrease in training loss, no sudden changes |
| $\alpha=0.05$ | Underfitting | Algorithm 1 struggles to find a prompt that meets the condition |
| $\tau=0$ | Worst performance | Straightforward prompt optimization without training the downstream model |
| $\tau=3$ (Default) | Best efficiency/performance | 3 steps of training are sufficient to provide signals for prompt optimization |
| $\tau=5$ | Slightly better performance but significantly increased training time | Diminishing returns, not cost-effective |
| GPT-4 as Decomposer | 59.71 QA acc | POQD consistently outperforms ICL-QD/ICLF-QD under different LLMs |
| DeepSeek-V3 as Decomposer | 60.43 QA acc | Same as above, showing strong consistency |
| vs Query Rewrite | 61.87 vs 52.16 | POQD significantly outperforms the query rewrite strategy |

### Key Findings

1. **Retrieval Improvement $\rightarrow$ QA Improvement**: POQD leads comprehensively in both retrieval accuracy and end-to-end QA accuracy, validating the full-link value of high-quality sub-queries for RAG systems.
2. **High Training Efficiency**: The time overhead of the prompt optimization phase is almost negligible, with the main overhead coming from generator training; during inference, the query decomposition time is far less than the model inference time.
3. **Equally Effective on Multi-Hop QA**: On StrategyQA, POQD achieves a 10.05% gain over ColBERT-orig, indicating that POQD can also function effectively in dynamic query generation scenarios.
4. **Limitations of ColBERT's Token-Level Decomposition**: The performance of ColBERT (when not using the original encoder) is significantly lower than other methods, validating the irrationality of token-level decomposition.

## Highlights & Insights

1. **Introducing Prompt Optimization to Retrieval Systems**: First to use an LLM-based optimizer for MVR query decomposition, cleverly bypassing the non-differentiable challenge.
2. **Alternating Training + Theoretical Guarantees**: The design of Algorithm 2 is both practical (requiring extremely low cost with $\tau=3$) and theoretically sound (Theorem 4.4).
3. **Weakly Supervised Optimization Paradigm**: Instead of directly optimizing retrieval metrics, it optimizes end-to-end QA performance, allowing the query decomposition to naturally adapt to downstream tasks.
4. **Highly Persuasive Motivating Example**: The case "Hong Kong" $\rightarrow$ "Kong" matching King Kong intuitively demonstrates the absurdity of token-level decomposition.
5. **High Generality**: POQD can be seamlessly integrated into any retrieval-based system, not limited to RAG.

## Limitations & Future Work

1. **LLM Calling Overhead**: Each prompt optimization and query decomposition requires calling LLMs, which may incur additional overhead for large-scale deployment.
2. **Frozen Retrieval Model**: Currently, only the generator is trained. Jointly updating the retrieval model might bring greater gains (but with higher computational overhead).
3. **Sub-Queries Sourced Only from Original Query Tokens**: Restricting sub-queries to consist only of tokens from the original query may limit query expansion capability.
4. **Limited Dataset Scale**: The experimental datasets are relatively small; the effectiveness in large-scale industrial scenarios (e.g., search engines) remains to be verified.
5. **Controllability of the Prompt Optimizer**: The optimization process of an LLM-based optimizer is uncontrollable, potentially converging to local optima in some domains.

## Related Work & Insights

- **ColBERT** (Khattab & Zaharia, 2020): The pioneer work of MVR, utilizing token-level decomposition + late interaction, which serves as the starting point for POQD.
- **OPRO** (Yang et al., 2024): The pioneering work of using LLM as an Optimizer, upon which the Prompt Optimizer of POQD is directly based.
- **Search-in-the-Chain** (Xu et al., 2024): SOTA for multi-hop QA, from which POQD adopts the framework on StrategyQA.
- **Insights**: Using an LLM-based optimizer to optimize other non-differentiable system components (such as reranking strategies or retrieval strategy selection in RAG) is a promising direction.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Innovation | ⭐⭐⭐⭐ | The combination of LLM optimizer + MVR query decomposition is novel |
| Theoretical Depth | ⭐⭐⭐⭐ | Backed by solid convergence theories |
| Experiments | ⭐⭐⭐⭐ | Comprehensive coverage across multiple datasets, multi-modal tasks, and extensive ablation studies |
| Practicality | ⭐⭐⭐⭐ | Direct integration into existing RAG systems, open-source code |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivating examples, complete framework description |
| Overall Rating | ⭐⭐⭐⭐ | Solid work, combining theory, experiments, and practicality |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](../../NeurIPS2025/information_retrieval/reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)
- [\[AAAI 2026\] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](../../AAAI2026/information_retrieval/opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](../../ACL2025/information_retrieval/investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)

</div>

<!-- RELATED:END -->
