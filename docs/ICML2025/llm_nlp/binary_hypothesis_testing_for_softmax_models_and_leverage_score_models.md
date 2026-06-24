---
title: >-
  [Paper Note] Binary Hypothesis Testing for Softmax Models and Leverage Score Models
description: >-
  [ICML 2025][LLM (Other)][Binary Hypothesis Testing] This work investigates the binary hypothesis testing problem for Softmax and Leverage Score models from a theoretical perspective. It establishes tight bounds on the number of queries required to distinguish between two parameterized models under an energy constraint, which is relevant to understanding the discriminative capabilities of LLMs across different domains.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Binary Hypothesis Testing"
  - "Softmax Models"
  - "Leverage Score"
  - "Sample Complexity"
  - "Theoretical Analysis"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: b41a944cffe2bd09
---

# Binary Hypothesis Testing for Softmax Models and Leverage Score Models

**Conference**: ICML 2025  
**arXiv**: [2405.06003](https://arxiv.org/abs/2405.06003)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Binary Hypothesis Testing, Softmax Models, Leverage Score, Sample Complexity, Theoretical Analysis, Attention Mechanism

## TL;DR

This work investigates the binary hypothesis testing problem for Softmax and Leverage Score models from a theoretical perspective. It establishes tight bounds on the number of queries required to distinguish between two parameterized models under an energy constraint, which is relevant to understanding the discriminative capabilities of LLMs across different domains.

## Background & Motivation

### Background

**Background**: The training cost of LLMs is extremely high, and uncertainty remains regarding their inference capabilities across different domains. A fundamental question is: **can different capability domains of LLMs be distinguished using only a limited number of parameter samples?**

This question is of critical importance in the following scenarios:

### Limitations of Prior Work

**Limitations of Prior Work**: RAG (Retrieval-Augmented Generation): The need to understand how different domains function.

### Key Challenge

**Key Challenge**: LLM Sparsification: Addressing efficiency issues by identifying the capability domains of the model.

### Proposed Solution

**Proposed Solution**: Model Selection: Determining how many queries are required to distinguish between two approximate models.

The core component of Transformers, the self-attention mechanism, employs the softmax distribution. Therefore, understanding the binary hypothesis testing problem for softmax distributions is of fundamental significance. This paper abstracts the attention unit as a softmax model and investigates its distinguishability from the theoretical perspective of hypothesis testing.

## Method

### Overall Architecture

The problem is formulated as the binary hypothesis testing of parameterized distribution families. Given two known models (with parameter matrices $A$ and $B$), and an unknown model $P$ that is one of the two, the goal is to determine which model $P$ represents using as few queries as possible.

### Key Designs

**Softmax Model Definition**: A parameter matrix $A \in \mathbb{R}^{n \times d}$ is given. For an input $x \in \mathbb{R}^d$, the probability of outputting $i \in [n]$ is defined as:

$$p_i = \frac{\exp(Ax)_i}{\langle \exp(Ax), \mathbf{1}_n \rangle}$$

**Energy Constraint**: The input is restricted to $\|x\|_2 \leq E$. This constraint is necessary; otherwise, one could amplify minute parameter differences using extremely large inputs, making distinguishability trivial. This corresponds to techniques like batch normalization in practice.

**Leverage Score Model Definition**: A parameter matrix $A \in \mathbb{R}^{n \times d}$ is given. For an input $s \in (\mathbb{R} \setminus \{0\})^n$, the probability of outputting $i \in [n]$ is defined as:

$$p_i = (A_s(A_s^\top A_s)^{-1} A_s^\top)_{i,i} / d$$

where $A_s = S^{-1}A$ and $S = \text{Diag}(s)$. Similarly, the input must satisfy the constraint $c \leq s_i^2 \leq C$.

**Core Theoretical Results**:

**Theorem 1 (Softmax Model)**:
- **Lower Bound**: If $\|B - A\|_{2 \to \infty} \leq \epsilon$, any successful algorithm requires $\Omega(\epsilon^{-2} E^{-2})$ queries.
- **Upper Bound**: When $B = A + \epsilon M$ (for sufficiently small $\epsilon$), there exists an algorithm that solves the problem using $O(\epsilon^{-2} \nu^{-1})$ queries, where $\nu = \sup_{x:\|x\|_2 \leq E} \text{Var}_{\text{SoftMax}_A(x)}(Mx)$.

**Theorem 2 (Leverage Score Model)**:
- **Lower Bound**: If $\sum_{i \in [n]} \|B_{i,*}^\top B_{i,*} - A_{i,*}^\top A_{i,*}\|_{\text{op}} \leq \epsilon$, then $\Omega(c\delta / (C\epsilon))$ queries are required.
- **Upper Bound**: Similarly, an upper bound of $O(\epsilon^{-2} \nu^{-1})$ exists.

**Key Proof Lemma** (Lemma 3.3): For the softmax distribution, if the $\ell_\infty$ distance between two sets of parameters is $\leq \epsilon$, then:
$$H^2(P, Q) = O(\epsilon^2), \quad \text{TV}(P, Q) = O(\epsilon)$$

Proof strategy: Construct a sequence of step-by-step perturbations of the parameters and accumulate them progressively using the triangle inequality of Hellinger distance.

### Loss & Training

This is a purely theoretical work with no training process. The core proof techniques include:
- Utilizing the classic relationship between sample complexity and Hellinger distance in hypothesis testing: $\Theta(H^{-2}(P_0, P_1))$.
- Analytically examining the sensitivity of softmax/leverage score distributions to parameter perturbations using Taylor expansion.
- Precise matrix perturbation analysis and operator norm bounding.

## Key Experimental Results

### Main Results

This is a purely theoretical paper with no experimental data. The main contribution lies in the tight bounds of the following theoretical results:

| Model Type | Lower Bound | Upper Bound (Local) | Do Bounds Match? |
|---------|------|------------|-------------|
| Softmax | $\Omega(\epsilon^{-2} E^{-2})$ | $O(\epsilon^{-2} \nu^{-1})$ | ✓ Locally tight |
| Leverage Score | $\Omega(c\delta/(C\epsilon))$ | $O(\epsilon^{-2} \nu^{-1})$ | Partially (Linear lower bound vs Quadratic upper bound) |

### Ablation Study

- **Analysis of $\nu$ across different parameter matrix structures**: When $A$ is an all-zero matrix and $M$ is non-zero in only one row, $\nu = O(1/n)$, making the sample complexity proportional to $n$.
- **Non-tight case in Leverage Score**: The lower bound depends linearly on $\epsilon$ ($\epsilon^{-1}$); improving this to a quadratic dependence remains an open problem.

### Key Findings

1. **Energy constraints are necessary**: Without constraints, any minute difference can be detected within 1 query (by sending an extremely large input).
2. **Existence of parameter equivalence classes**: In the Softmax model, $B = A + \mathbf{1}_n^\top w$ is indistinguishable from $A$; in the Leverage Score model, $B = AR$ is indistinguishable.
3. **Sample complexity may depend on $n$**: This occurs when differences are concentrated in low-probability rows.
4. **Local results are tight**: The upper and lower bounds match in the sense of local perturbations ($\Theta(\epsilon^{-2} \nu^{-1})$).

## Highlights & Insights

1. **Abstracting attention as a softmax model** is a significant theoretical modeling effort, providing a new tool to understand Transformers.
2. **The introduction of energy constraints** cleverly avoids trivial cases and has practical counterparts such as batch normalization.
3. **Unified framework**: Both softmax and leverage score models are treated under the same hypothesis testing framework, revealing deep connections between the two distribution families.
4. **Providing a theoretical foundation for LLM capability domain partitioning**: The results indicate that distinguishing different "capability areas" of a model indeed requires a certain amount of sampling.
5. **Innovative proof techniques**: Generalizing classical hypothesis testing theory to structured parameterized distribution families.

## Limitations & Future Work

1. **Loose lower bound for Leverage Score**: The lower bound is $\epsilon^{-1}$ while the upper bound is $\epsilon^{-2}$. Closing this gap remains an open problem.
2. **Purely local results**: The upper and lower bounds only hold when $\epsilon$ is sufficiently small; large perturbations are not covered.
3. **Lack of experimental validation**: The paper does not demonstrate the application of these results on real LLMs/Transformers.
4. **The computation of $\nu$**: Computing $\nu$ is an optimization problem, and no efficient algorithm is provided.
5. **Only binary testing is considered**: The work has not been extended to more practical testing types such as goodness-of-fit testing or two-sample testing.
6. **Strong model assumptions**: Real-world LLMs do not rely solely on a single-layer softmax, and the compound effects of multi-layer attention are not considered.

## Related Work & Insights

- **Connection to classical hypothesis testing**: The Neyman-Pearson theory provides a tight characterization of hypothesis testing for distributions as $\Theta(H^{-2}(P_0, P_1))$; this paper's contribution is generalizing it to the softmax/leverage score parameter families.
- **Relevance to theoretical studies of Transformers**: Work in the Alman & Song series investigates the computational complexity of attention; this paper approaches the problem from an information-theoretic perspective.
- **Relationship with understanding LLM capabilities**: This work is complementary to research on circuit complexity investigating LLM expressivity limits, focusing on distinguishability.
- **Insights**: If applied in practice, these results could help design more efficient RAG systems (by determining a model's reliability in a certain domain with fewer samples) and could inspire theoretical analysis of model merging.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (4/5) — First to study hypothesis testing for softmax models, offering a novel modeling perspective.
- Experimental Thoroughness: ⭐⭐⭐ (2/5) — Purely theoretical work, no experiments.
- Writing Quality: ⭐⭐⭐⭐ (3/5) — Clear results, but the related work section is somewhat wordy.
- Value: ⭐⭐⭐⭐ (3/5) — Solid theoretical contributions, but limited practical value due to the lack of a bridge to practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] B-score: Detecting biases in large language models using response history](b-score_detecting_biases_in_large_language_models_using_response_history.md)
- [\[ICML 2025\] The Lock-in Hypothesis: Stagnation by Algorithm](the_lock-in_hypothesis_stagnation_by_algorithm.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)
- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](../../ICLR2026/llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](../../ACL2025/llm_nlp/bco_binary_classifier_alignment.md)

</div>

<!-- RELATED:END -->
