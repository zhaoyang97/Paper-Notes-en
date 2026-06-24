---
title: >-
  [Paper Note] Tokenized Bandit for LLM Decoding and Alignment
description: >-
  [ICML 2025][LLM Pretraining][Multi-Armed Bandit] Formulates the LLM decoding and alignment problem as a **tokenized bandit** problem, proposes the DDMC (Diminishing Distance with More Commons) assumption, proves that greedy decoding is near-optimal under this assumption, and designs online learning algorithms EOFUL and GreedyETC with sublinear regret.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Multi-Armed Bandit"
  - "LLM Decoding"
  - "Decoding-Time Alignment"
  - "Greedy Decoding"
  - "Sequence Function Optimization"
date: 2026-05-08
content_hash: a79c6ea9b319baa8
---

# Tokenized Bandit for LLM Decoding and Alignment

**Conference**: ICML 2025  
**arXiv**: [2506.07276](https://arxiv.org/abs/2506.07276)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Multi-Armed Bandit, LLM Decoding, Decoding-Time Alignment, Greedy Decoding, Sequence Function Optimization

## TL;DR

Formulates the LLM decoding and alignment problem as a **tokenized bandit** problem, proposes the DDMC (Diminishing Distance with More Commons) assumption, proves that greedy decoding is near-optimal under this assumption, and designs online learning algorithms EOFUL and GreedyETC with sublinear regret.

## Background & Motivation

LLM alignment is one of the core challenges in current LLM research. The dominant method, RLHF, achieves alignment via fine-tuning but suffers from high computational costs, the need for extensive human annotations, and frequent model updates, making it difficult to scale to personalized scenarios.

Recently, **decoding-time alignment** methods have emerged, aiming to adjust LLM outputs online during the inference phase to match user preferences without fine-tuning the model. However, the theoretical foundation of these methods remains weak, lacking rigorous sample efficiency analysis.

A more fundamental question is: even for the simplest decoding algorithms (such as greedy decoding and beam search), there is a lack of deep theoretical understanding—why does greedy decoding perform remarkably well on many tasks?

This work takes a novel perspective, modeling the LLM decoding and alignment problem as **tokenized variants of linear/multi-armed bandits**, where the decision-maker must construct a sequence token-by-token irrevocably and eventually receive utility feedback from the user.

## Method

### Overall Architecture

This work proposes two core problem variants:

1. **Tokenized Linear Bandit (TLB)**: The utility function has a linearly parameterized structure $u_t(x_t, \mathbf{y}) = \langle \theta, e(x_t, \mathbf{y}) \rangle$, where $e(\cdot)$ is an embedding function and $\theta$ is the latent parameter to be learned. In each round, the user submits a different query (context), and the decision-maker needs to learn $\theta$ online.

2. **Tokenized Multi-Armed Bandit (TMAB)**: The utility function can be arbitrary (no linear structure), but the context remains fixed $x_t = x$. The problem degenerates to learning the optimal token sequence under a fixed context.

**Unified Problem Formulation**: In each round $t \in [T]$, the user submits a query $x_t$, and the decision-maker irrevocably selects tokens token-by-token from a vocabulary $\mathcal{V}$ ($|\mathcal{V}|=n$) to construct a sequence $\mathbf{y}_t$. Upon completion of the sequence, a noisy reward $r_t(\mathbf{y}_t) = u_t(\mathbf{y}_t) + \eta_t$ is observed. The goal is to minimize cumulative pseudo-regret:

$$\text{Reg} = \sum_{t=1}^{T} \left[ \max_{\mathbf{z} \in \mathcal{V}^*} u_t(\mathbf{z}) - u_t(\mathbf{y}_t) \right]$$

### Key Designs

#### 1. Fundamental Impossibility Results

The authors first prove two lower bounds:

- **TLB**: Without structural assumptions on the sequence function, the worst-case regret of any algorithm is $\Omega(T(1-1/2^{L-2}))$, growing exponentially with sequence length $L$.
- **TMAB**: Under no structural assumptions, the worst-case regret lower bound is $\Omega(\min(\sqrt{n^L T}, T))$, which is also exponential.

This suggests that the raw problem is computationally intractable, making it necessary to introduce reasonable structural assumptions.

#### 2. DDMC Assumption (Diminishing Distance with More Commons)

This is the most core conceptual innovation of this paper. For any two sequences $\mathbf{y}, \mathbf{z}$ of equal length and any token $\tau$:

$$|u(x_t, \mathbf{y}:\tau) - u(x_t, \mathbf{z}:\tau)| \leq |u(x_t, \mathbf{y}) - u(x_t, \mathbf{z})|$$

Intuitive explanation: If two outputs share more common tokens in their suffix, the difference in user-perceived utility becomes smaller. This formalizes the phenomenon of "common suffixes reducing discrepancy".

**Relationship with Submodularity**: DDMC is similar to but essentially different from classical submodularity—submodularity describes "diminishing returns", whereas DDMC describes "diminishing distance". Submodularity focuses on set containment, while DDMC focuses on common suffixes of equal-length sequences. Neither implies the other; they are complementary.

#### 3. EOFUL Algorithm (Excessive Optimism Under the Face of Uncertainty)

For the TLB problem, combining greedy decoding with the LinUCB framework:

- **Mechanism**: During each round of decoding, maintain a confidence ellipsoid $C_t$ for the parameter $\theta$, optimistically estimate the utility of each candidate token, and select the token jointly determined by the most optimistic parameter in the confidence set and the optimal token.
- **Confidence Set Construction**: Ridge regression is used to estimate $\hat{\theta}_t = \Sigma_t^{-1} \sum_{i=1}^{t-1} r_i \cdot \mathbf{y}_i$, and the confidence set is an ellipsoid $C_t = \{\vartheta: (\vartheta - \hat{\theta}_t)^\top \Sigma_t (\vartheta - \hat{\theta}_t) \leq \beta_t\}$.
- **Decoding Process**: For each position $k$, compute $\tau^* = \arg\max_{\tau \in \mathcal{V}, \theta \in C_t} \langle \theta, e(x_t, \mathbf{y}:\tau) \rangle$, and terminate if EOS is selected.

**Five-Step Proof of Regret Analysis**:

| Step | Content | Impact/Role |
|------|------|------|
| Step 1 | Length Equalization | Pad the algorithm sequence and the optimal sequence to the same length |
| Step 2 | Level-k Regret Definition | Define regret incurred by token selection at the $k$-th level, preparing for recursive analysis |
| Step 3 | Fictitious Extension | Construct $\mathbf{f}_t^{(1:k)} = \mathbf{y}_t^{(1:k-1)}:\mathbf{o}_t^{(k)}$ to propagate level-k regret to level-(k-1) |
| Step 4 | Regret Decomposition | Use DDMC to represent the global regret as level-1 regret plus the estimation error of each prefix |
| Step 5 | Sum-of-Squares Regret | Derive the final bound using standard sum-of-squares analysis from linear bandits |

#### 4. GreedyETC Algorithm

For the TMAB problem, an "explore-then-commit" strategy combined with greedy decoding is employed:

- **Exploration Phase**: For each position $k$, attempt all $n$ tokens $N$ times each, and calculate the average reward.
- **Commitment Phase**: Select the token with the highest average reward, fix it, and proceed to the next level.
- **Key Trick**: Take union bounds only along the path selected by the algorithm, avoiding the exponential explosion that would result from taking union bounds over all $n^L$ sequences.

### Loss & Training

This work is purely theoretical and does not involve gradient training. The core optimization objective is to minimize cumulative pseudo-regret. The two algorithms achieve:

- **EOFUL (TLB)**: $\text{Reg} = O(cL\sqrt{dT\log T})$, which is sublinear in $T$ and linear in $L$.
- **GreedyETC (TMAB)**: $\text{Reg} = O(nLT^{2/3}(\log T)^{1/3})$, which is sublinear in $T$ and linear in $L$.

**LLM Alignment Application**: If the user utility is $u(x_t, \mathbf{y}) = \gamma v(p(x_t, \mathbf{y})) + (1-\gamma) f(x_t, \mathbf{y})$, where $p$ is the log-probability of the frozen LLM and $f$ is a linearly realizable non-aligned utility function, we can construct the augmented parameter $\theta' = [(1-\gamma)\theta : \gamma]$ to reduce the problem to TLB.

## Key Experimental Results

### Main Results

#### DDMC Assumption Validation

On TruthfulQA and HH-RLHF datasets, embedding extraction from Llama3-8B-Instruct is used to validate whether DDMC holds under two distance metrics:

| Dataset | Distance Function | DDMC Trend | Explanation |
|--------|-------------------|------------|-------------|
| TruthfulQA | $d_1$ ($\ell_1$ distance) | ✅ Diminishing | Discrepancy decreases as the number of common suffix tokens increases |
| TruthfulQA | $d_2$ ($\ell_2$ distance) | ✅ Diminishing | Same as above, holds for $\ell_2$ distance too |
| HH-RLHF | $d_1$ ($\ell_1$ distance) | ✅ Strongly Diminishing | The diminishing trend is more pronounced than on TruthfulQA |
| HH-RLHF | $d_2$ ($\ell_2$ distance) | ✅ Strongly Diminishing | Different tasks exhibit distinct diminishing structures |

- The diminishing curvature is more conspicuous when there are fewer common suffix tokens.
- The decline in HH-RLHF is more dramatic than in TruthfulQA, implying that different tasks have different sequential function structures.

#### EOFUL Performance Validation

In synthetic data experiments with $L=30$, top-15 token truncation, and $\gamma=0.8$:

| Algorithm | Regret Trend | Explanation |
|-----------|--------------|-------------|
| EOFUL | Sublinear Growth | Effectively learns the latent parameter $\theta$ |
| Theoretical Upper Bound (scaled 0.1×) | Sublinear | Actual performance is much better than the theoretical upper bound |
| WrongTheta | Linear Growth | Alignment fails using incorrect $\theta'=(-0.5,...,-0.5)$ |
| Misaligned Greedy | Linear Growth | Standard greedy decoding using only LLM probability fails to adapt to preference |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|---------------|-------------|-------------|
| Assumption 3.6 Validation | $\rho_t^l = \|\mathbf{y}^{(1:l)}\|_{\Sigma_t} / \|\mathbf{y}\|_{\Sigma_t} \leq 1.25$ | The ratio of substring Mahalanobis norms is bounded above by 1.25 throughout |
| Max Ratio | $\leq 1.25$ | The maximum ratio across all token positions remains bounded |
| Mean Ratio | Far $< 1.25$ | The mean ratio is much smaller, indicating the assumption is loose in practice |

### Key Findings

1. **DDMC Holds on Real Data**: Embedding distance experiments on TruthfulQA and HH-RLHF validate the rationality of the DDMC assumption.
2. **Actual Regret of EOFUL is Much Better than the Theoretical Bound**: This suggests that the current analysis might not be tight.
3. **Assumption 3.6 is Highly Lenient in Practice**: The upper bound of the ratio is only 1.25, having minimal impact on regret.
4. **Greedy Decoding is Near-Optimal under DDMC** (Theorem 5.1): It requires only $nLT$ queries to find the optimal sequence.

## Highlights & Insights

1. **Conceptual Innovation**: The DDMC assumption is simple and natural, capturing the intuition that a "common suffix minimizes discrepancy". It provides a new tool for the theoretical analysis of sequence functions and is complementary to submodularity.
2. **Theoretical Defense of Greedy Decoding**: For the first time, it formally proves that greedy decoding is near-optimal under DDMC, providing a theoretical foundation for the widespread use of greedy decoding in models like Gemini.
3. **Unified Decoding-Alignment Framework**: Unifies decoding and alignment within a bandit framework, degenerating to pure decoding when $\gamma=0$ and pure learning when $\gamma=1$.
4. **Level-k Regret + Fictitious Extension**: A clever recursive analysis technique that avoids the exponential explosion of the sequence space.
5. **Practical Value**: EOFUL can learn user preferences online during inference without fine-tuning LLMs, making it highly suitable for personalized alignment.

## Limitations & Future Work

1. **EOFUL Relies on Linearly Realizable Assumptions and Assumption 3.6**: The utility functions of real-world LLMs may be non-linear, where the linear approximation capability is limited.
2. **GreedyETC's Regret is of Order $T^{2/3}$**: Inferior to TLB's $\sqrt{T}$; achieving $O(\sqrt{T})$ under TMAB remains an open problem.
3. **Limited Experimental Scale**: The performance of EOFUL is validated only on synthetic data, lacking end-to-end experiments on large-scale real LLM alignment.
4. **Cold Start Problem**: In its initial phase, EOFUL may generate highly misleading outputs, which degrades user experience.
5. **Single-Sequence Feedback**: Currently, it only observes the reward of the complete sequence; utilizing token-level feedback might further improve sample efficiency.
6. **DDMC Validation is in the Average Sense**: It is not proven to hold strictly for every single pair of sequences, leaving a gap between theory and empirical results.
7. **KL Regularization is Unconsidered**: The KL divergence constraint in standard RLHF is not incorporated, which the authors themselves identify as a future direction.

## Related Work & Insights

- **Bandit Tree Search (BTS)**: TMAB is closely related to BTS, and the decision version of BTS can be reduced to TMAB.
- **Sequence Submodular Maximization**: DDMC is complementary to sequence submodularity, as they characterize 'diminishing distance' and 'diminishing returns' respectively.
- **LinUCB / OFUL**: EOFUL directly extends the confidence ellipsoid method of classical linear bandits to the tokenized setting.
- **DPO / RLHF**: The proposed method does not require fine-tuning, serving as an inference-time alternative to DPO/RLHF.
- **Controlled Decoding**: Shared goals with inference-time alignment methods like Mudgal et al. (2023) but provides formal theoretical guarantees.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Brand-new problem formulation + DDMC assumption + proof of greedy decoding optimality
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Complete upper and lower bounds, with elegant five-step proof analysis.
- **Experimental Thoroughness**: ⭐⭐⭐ — Convincing assumption validation, but algorithm performance is only tested on synthetic data.
- **Value**: ⭐⭐⭐⭐ — Theoretical insights provide guidance for decoding strategy selection, though direct engineering applications require more work.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, but heavily loaded with notations, requiring some background in bandit theory.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Evaluating Morphological Alignment of Tokenizers in 70 Languages](evaluating_morphological_alignment_of_tokenizers_in_70_languages.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/llm_pretraining/composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[ACL 2025\] TokAlign: Efficient Vocabulary Adaptation via Token Alignment](../../ACL2025/llm_pretraining/tokalign_vocab_adaptation.md)
- [\[ACL 2025\] Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](../../ACL2025/llm_pretraining/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)

</div>

<!-- RELATED:END -->
