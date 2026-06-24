---
title: >-
  [Paper Note] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models
description: >-
  [ICML 2026][Graph Learning][Implicit Reasoning] Starting from the Knowledge Graph Completion (KGC) task, this paper proves and measures that the "minimal parameter budget required for implicit reasoning" follows a linear scaling law based on **Graph Search Entropy** as the complexity metric. Each parameter supports approximately $0.008$ bits of reasoning information, challenging the naive intuition that "larger models always yield stronger reasoning."
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Implicit Reasoning"
  - "Minimal Parameter Budget"
  - "Graph Search Entropy"
  - "U-shaped scaling"
  - "Knowledge Graph Completion"
date: 2026-05-08
content_hash: 91c6bc34ec6b3f5c
---

# Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models

**Conference**: ICML 2026  
**arXiv**: [2504.03635](https://arxiv.org/abs/2504.03635)  
**Code**: https://github.com/WANGXinyiLinda/reasoning-scaling-law (Available)  
**Area**: LLM Reasoning / Pre-training Scaling Law  
**Keywords**: Implicit Reasoning, Minimal Parameter Budget, Graph Search Entropy, U-shaped scaling, Knowledge Graph Completion

## TL;DR
Starting from the Knowledge Graph Completion (KGC) task, this paper proves and measures that the "minimal parameter budget required for implicit reasoning" follows a linear scaling law based on **Graph Search Entropy** as the complexity metric. Each parameter supports approximately $0.008$ bits of reasoning information, challenging the naive intuition that "larger models always yield stronger reasoning."

## Background & Motivation

**Background**: Classic LM scaling laws proposed by Kaplan, Hoffmann, et al., are built on the assumption that loss decreases monotonically with parameter count. Allen-Zhu & Li (2025) further established a memory capacity scaling law as "2 bits of knowledge per parameter." Most works assume: larger model $\to$ lower test loss $\to$ stronger capability.

**Limitations of Prior Work**: This paradigm only describes "memory" but **fails to characterize the parameter budget required for "reasoning."** Methods like Chain-of-Thought (CoT) and RL post-training are "secondary processing" atop pre-trained representations. The fundamental question remains: how large must a model be during pre-training for reasoning to emerge? Existing scaling laws cannot answer this, and counter-examples like inverse scaling or broken scaling laws suggest the monotonic assumption is not universal.

**Key Challenge**: **Memory is about "packing information into parameters," where more is better; reasoning is about "internalizing structures into functions," where excessive parameters might lead to overfitting on specific triplets and loss of underlying rules.** Mixing these two capacities is the root cause of imprecise scaling laws.

**Goal**: To decompose the problem into two questions: (1) Is the concept of a "minimal model supporting optimal implicit reasoning" mathematically well-defined and identifiable? (2) If it exists, what data attributes determine this minimal size? Can an extrapolatable scaling law be derived for real-world data?

**Key Insight**: Abstract world knowledge as a **Knowledge Graph (KG)**, represent pre-training corpora as a "stream of triplets on the graph," and define reasoning as "completing unseen edges derivable via rules." By replacing entity names with random IDs to strip lexical signals, the problem is reduced to a pure "graph structure $\to$ parameter budget" mapping.

**Core Idea**: Use **Graph Search Entropy** (the information rate of a maximum entropy random walk on the graph) as the data complexity metric. Prove that the optimal model size $N_\theta^*(G) = \Theta(H(G))$ and verify this on both synthetic and real-world graphs (FB15K-237).

## Method

### Overall Architecture

To determine when implicit reasoning emerges during pre-training, the authors abstract world knowledge into a KG and reasoning into "knowledge graph completion." Different-sized LMs are trained from scratch on controlled synthetic graphs. The "optimal size" yielding the lowest test loss is measured and explained via a complexity metric derived solely from the graph structure. The pipeline involves: generating synthetic graphs $G$ where parameters like rule count $N_h$, relations $N_r$, entities $N_e$, and derivable ratio $\gamma$ are varied; replacing entities/relations with random IDs and tokenizing them by character; pre-training Llama-based models from scratch using a standard next-token prediction objective $L(\theta) = \frac{1}{N} \sum_i -\log P_\theta(e_i^h, r_i, e_i^t)$ with batch size $1024$. Evaluation is a 10-choice multiple-choice task on held-out derivable triplets to record test loss and accuracy. Finally, Graph Search Entropy $H(G)$ and the "optimal parameter budget" $N_\theta^*(G)$ are calculated to prove their linear relationship and extrapolate to FB15K-237. This works because "random IDs + character tokenization" removes all lexical signals that would otherwise contaminate the scaling curves.

### Key Designs

**1. Formalizing "Optimal Model Size": Turning a vague sweet spot into a provable quantity**

Empirically, a U-shaped curve where test loss first decreases and then increases with model size is often observed, but this "sweet spot" remains ill-defined. This paper defines the "optimal test loss achieved after early-stopping within $t$ steps" as $\underline{\ell}_t(\theta, G) := \min_{0 \le s \le t} \ell(\theta_s, G)$. Subsequently, the $\epsilon$-optimal model size is defined as $N_{\theta,t}^*(G) := \min\{N_\theta : \exists \theta, \underline{\ell}_t(\theta, G) \le \underline{\ell}_t^*(G) + \epsilon\}$. **Theorem 3** proves its convergence: given a mild "gap condition" (where models smaller than the optimal size are significantly worse), as $t \to \infty$, $N_{\theta,t}^*(G) \to N_\theta^*(G)$. This transforms an engineering intuition into a well-defined mathematical target, linking U-shaped curves to the theory of benign overfitting/double descent.

**2. Graph Search Entropy $H(G)$: Quantifying reasoning complexity with a structural scalar**

To predict the optimal size, a complexity metric independent of training details is needed. This is constructed using a maximum entropy random walk on the graph: let $\lambda$ be the principal eigenvalue of adjacency matrix $A$ and $\psi$ its eigenvector. The stationary distribution is $\rho_i = \psi_i^2 / \|\psi\|_2^2$, and the transition matrix is $S_{ij} = (A_{ij}/\lambda)(\psi_j/\psi_i)$. By merging entity transitions into entity-relation transitions $S^r_{ij}$, the relation entropy rate $H^r(G)$ is obtained. The final Graph Search Entropy:
$$H(G) = N_e \cdot (\log \lambda + H^r(G))$$
captures uncertainty in both "which entity" and "which relation." This differs from "Knowledge Entropy" used for memory; it measures the complexity of traversing the graph, explaining why reasoning capacity ($0.008$ bit/param) is $\approx 250\times$ smaller than memory capacity ($2$ bit/param).

**3. $N_\theta^*(G) = \Theta(H(G))$: Linking graph complexity and parameter budget linearly**

**Theorem 4** establishes $N_\theta^*(G) = \Theta(H(G))$ under three assumptions: (i) random IDs prevent semantic sharing; (ii) $N$ parameters have $O(N)$ capacity; (iii) Bayes conditional distributions are approximated by sparse coefficients $a_x$ where $\|a_x\|_0 \le \alpha H(Y|X=x) + \beta$. Empirically, the authors fit a regression line $(R^2 = 0.85)$ on synthetic data $(H(G), N_\theta^*)$ and successfully extrapolated it to predicted FB15K-237 results, matching actual observations closely.

### Loss & Training

The objective is standard next-token prediction Cross-Entropy loss. Experiments used a fixed batch size of $1024$, $10$k training steps, and Llama architecture. Model size was varied by adjusting hidden dimensions and layers. Evaluation used a 10-choice format for held-out derivable triplets.

## Key Experimental Results

### Main Results

| Setup | Phenomenon | Key Numbers |
|------|------|----------|
| FB15K-237 + Random IDs | Clear U-shaped test loss; monotonic train loss | $N_e = 14{,}505, N_r = 237, N = 310{,}116$ |
| Synthetic Graph Sweep | Optimal size stable over training steps, increases with $N$ and $N_r$, insensitive to $N_h$ | 6-dimension ablation |
| $(H, N_\theta^*)$ Regression | Strong linear relationship | $R^2 = 0.85$ |
| FB15K-237 Extrapolation | Real graph falls within 95% CI of synthetic fit | High prediction accuracy |
| Reasoning Capacity Scaling | $1$ bit Search Entropy $\approx 124$ parameters | $\approx 0.008$ bit / parameter |

### Ablation Study

| Graph Property Modified | Impact on Optimal Model Size | Impact on Reasoning Performance |
|------|------|------|
| Training steps $t$ ↑ | Decreases then stabilizes | Improves until saturation |
| Triplets $N$ ↑ | Increases (Standard scaling) | Improves |
| Rule count $N_h$ ↑ | **Negligible change** | Affects Accuracy but not Search Complexity |
| Relation count $N_r$ ↑ | Increases | Improves (reduces spurious correlation) |
| Derivable ratio $\gamma$ ↑ | Increases then saturates | Improves then saturates |
| Entity count $N_e$ ↑ | Increases | Decreases when rules/relations are sparse |

### Key Findings

- **The discovery that $N_h$ does not affect optimal size** is counter-intuitive: more rules affect accuracy, but not the search complexity of the graph. $H(G)$ captures the true bottleneck for parameter budget.
- **Reasoning capacity ($0.008$ bit/param)** vs **Memory capacity ($2$ bit/param)** highlights that reasoning is over two orders of magnitude more "parameter-hungry" than memory.
- **U-shaped curves appear only in "over-trained" regimes**: Large models are not incapable of reaching the optimum; they are merely "excessive" and prone to overfitting.

## Highlights & Insights

- **Bifurcating "scaling laws for memory" and "scaling laws for reasoning"**: Categorizing capacity by task type is a major conceptual breakthrough in scaling research.
- **Graph Search Entropy is an extrapolatable metric**: Since real-world text can be parsed into KGs, one can compute $H$ to predict the optimal model size for specific domains like medicine or law.
- **Random ID + Character tokenization** is a vital trick for removing lexical bias, which can be reused in other controlled scaling studies.

## Limitations & Future Work

- **Architecture Dependence**: Theorem 4's upper bound relies on Transformer attention KV-memory. Whether this holds for non-attention architectures like Mamba is unverified.
- **Limited Training Duration**: $10$k steps is an approximation of the "infinite budget" limit; optimal size localization is still affected by discrete model sizes and early-stopping noise.
- **Single Real-world Data Point**: Extrapolation was only verified on FB15K-237.
- **Narrow Definition of Reasoning**: The relationship between Search Entropy and more complex reasoning (CoT, multi-step math) remains unexplored.

## Related Work & Insights

- **vs. Kaplan / Hoffmann**: They describe monotonic loss reduction; this work reveals the U-shaped loss in reasoning tasks and relates optimal size to data complexity rather than just compute.
- **vs. Allen-Zhu & Li (2025)**: Complementary capacity studies ($2$ bit/param for memory vs $0.008$ bit/param for reasoning).
- **vs. Broken scaling laws / Inverse scaling**: This work provides a theoretical explanation (benign overfitting + gap condition) for these phenomena.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Decouples reasoning from memory in scaling laws with an extrapolatable metric.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive synthetic sweeps, though real-world verification is limited to one dataset.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous connection between theorems and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a principled framework for selecting model sizes for reasoning-heavy tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning](../../ACL2026/graph_learning/craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[ACL 2026\] Comparing Human and Large Language Model Interpretation of Implicit Information](../../ACL2026/graph_learning/comparing_human_and_large_language_model_interpretation_of_implicit_information.md)
- [\[ICLR 2026\] Scaling Knowledge Graph Construction through Synthetic Data Generation and Distillation](../../ICLR2026/graph_learning/scaling_knowledge_graph_construction_through_synthetic_data_generation_and_disti.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)

</div>

<!-- RELATED:END -->
