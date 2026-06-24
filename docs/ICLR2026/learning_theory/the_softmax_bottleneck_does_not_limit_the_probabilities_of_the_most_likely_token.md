---
title: >-
  [Paper Note] The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens
description: >-
  [ICLR 2026][Learning Theory][softmax bottleneck] This paper theoretically revisits the "softmax bottleneck": it proves that even randomly initialized output projection matrices can assign arbitrarily accurate probabilities to a significant number of top tokens (approx. 26 for GPT-2 scale, 95 in practice, and over 1000 for Llama2), questioning whether the softmax bottleneck significantly limits LLM capabilities in realistic scenarios.
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "LLM Expressivity Analysis"
  - "softmax bottleneck"
  - "output projection matrix"
  - "random matrices"
  - "sign rank"
  - "embedding dimension"
date: 2026-05-08
content_hash: f409519121af007a
---

# The Softmax Bottleneck Does Not Limit the Probabilities of the Most Likely Tokens

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DgJqQk6y19](https://openreview.net/forum?id=DgJqQk6y19)  
**Code**: https://github.com/ronenbasri/The-Softmax-Bottleneck-Does-Not-Limit-the-Probabilities-of-the-Most-Likely-Tokens  
**Area**: Learning Theory / LLM Expressivity Analysis  
**Keywords**: softmax bottleneck, output projection matrix, random matrices, sign rank, embedding dimension

## TL;DR
This paper theoretically revisits the "softmax bottleneck": it proves that even randomly initialized output projection matrices can assign arbitrarily accurate probabilities to a significant number of top tokens (approx. 26 for GPT-2 scale, 95 in practice, and over 1000 for Llama2), questioning whether the softmax bottleneck significantly limits LLM capabilities in realistic scenarios.

## Background & Motivation

**Background**: Transformers use an $N \times d$ output projection matrix (OPM, also called unembedding/readout matrix) in the final layer to linearly map a $d$-dimensional embedding $x$ to $N$-dimensional logits $y = Ax$ (where $N$ is the vocabulary size, $N \gg d$), followed by a softmax to obtain the next token distribution.

**Limitations of Prior Work**: A series of works (Yang et al. [26], Chang & McCallum [2], Grivas et al. [7], etc.) point out that because $d \ll N$, logits are restricted to a $d$-dimensional linear subspace, and the log-probability space can only fall within a low-dimensional subspace—this is known as the **softmax bottleneck**. They argue that the log-probability distributions of natural language are high-dimensional, and this bottleneck prevents LLMs from precisely fitting the true statistics of the next token, thereby harming performance. Various "remedies" like Mixture of Softmaxes, multi-facet softmax, and Linear-Monotonic-Softmax have been proposed.

**Key Challenge**: Prior works measure whether the model can express **any** complete distribution. However, the authors point out an overlooked fact: during inference using nucleus sampling or beam search, **only a small number of highest-probability tokens truly influence generation**, while the precise probabilities of extremely unlikely tokens are irrelevant. Therefore, the real question is not "can it express any distribution," but "can it accurately specify the probabilities of the **top-$m$** most likely tokens." This is a much weaker and more realistic requirement.

**Goal**: To decompose the problem into two sub-questions: (1) Given any $m$ tokens, can the OPM produce a distribution where they are the most likely and their probabilities are arbitrarily specified (and sum to nearly 1)? How large can $m$ be? (2) For the best (trained) OPM, what is the maximum $m$ achievable, or conversely, what is the minimum required embedding dimension?

**Key Insight**: The authors reframe the problem from a "top-$m$ perspective" and use two sets of tools: **random matrix theory** (to provide performance lower bounds) and **sign rank** (to provide upper bounds for optimal matrices), followed by empirical validation using linear programming.

**Core Idea**: While the softmax bottleneck indeed restricts LLMs to expressing only a measure-zero subset of all possible distributions, this subset imposes almost no constraints on "accurately characterizing the most likely tokens." Thus, the bottleneck is likely not a true performance bottleneck in realistic settings.

## Method

### Overall Architecture

The paper is an analytical theoretical work that does not introduce a new model. The goal is to find upper/lower bounds and provide empirical measurements for the maximum $m$ for which an OPM can precisely express top-$m$ probabilities. The approach is three-fold:

First leg (Section 4, **Performance Lower Bound**): Treating the OPM as a random Gaussian matrix $A \sim \mathcal{N}(0,1)$, asking "what $m$ can be specified even at random initialization." It first solves for "making top-$m$ logits the largest" (4.1), then upgrades this to "assigning top-$m$ tokens arbitrary specified probabilities that sum to nearly 1" via scaling (4.2).

Second leg (Section 5, **Optimal Upper Bound**): Moving beyond random matrices, asking "what is the maximum $m$ achievable by the best possible OPM," i.e., the minimum embedding dimension $d^*$. The authors reduce this combinatorial problem to the **sign rank** of a $\{-1,+1\}$ indicator matrix, using results from Alon et al. to calculate $d^* \approx 2m$.

Third leg (Section 6, **Empirical Validation**): Using linear programming (LP) to solve for feasible embeddings $x$ on trained OPMs from GPT-2, GPT2-XL, TinyLlama, T5-Large, and Llama2, as well as random matrices of identical size. They measure the actual $m$ and find that trained matrices perform almost identically to random matrices.

### Key Designs

**1. Top-$m$ Lower Bound for Random OPM: Approximation via inverse Wishart**

To determine what $m$ can be specified at random initialization, the authors construct an embedding $x_\parallel = A_m^T w$ that lies solely in the row space of $A_m$ (the first $m$ rows of $A$). This ensures the first $m$ logits take target values $y_m$ (uniquely determined by $w=(A_m A_m^T)^{-1}y_m$). They then estimate the probability that the remaining $N-m$ logits remain smaller than the minimum of the top-$m$ logits. The key technique involves the fact that since elements of $A$ are i.i.d. Gaussian, $(A_m A_m^T)^{-1}$ follows an inverse Wishart distribution $W_m^{-1}(I,d)$. Elements of the vector $a_j^T A_m^T (A_m A_m^T)^{-1}$ are approximately i.i.d. with zero mean and variance:

$$v \approx \frac{d(d-1)}{(d-m)(d-m-1)(d-m-3)}$$

Consequently, the bottom logits are approximately $y_j \sim \mathcal{N}(0, \|y_m\|^2 v)$. This yields the lower bound (Proposition 1):

$$P\big(\exists x,\ y_{m+1},\dots,y_N < y_1,\dots,y_m\big) \gtrsim \Phi^{N-m}\!\left(\frac{\min_{i\in[m]} y_i}{\|y_m\|\sqrt{v}}\right)$$

where $\Phi$ is the standard normal CDF. Corollary 1 further proves this probability is maximized when **top-$m$ logits are equal and positive** ($y_m = c\mathbf{1}_m$), simplifying the bound to $\Phi^{N-m}(1/\sqrt{mv})$. Applying this to 22 common models, the authors find lower bounds ranging from $m\approx 26$ for GPT-2 to $m\approx 418$ for GPT3-175B.

**2. Upgrading to Exact Probabilities: Scaling probability mass into top-$m$**

Simply making top-$m$ logits the largest is insufficient because softmax probabilities depend on all logits. The authors require the top-$m$ tokens to have **arbitrarily specified ratios** $p_i/p_m$ and for their sum to approach 1 (with the remainder $\le \delta$). The difficulty is that "specified ratios" and "summing to 1" seem to conflict. The solution is introduced via a scale factor $s$, setting $x_s = s\bar{x}_s$, where $\bar{x}_s$ is constructed via:

$$\bar{y}_s(i) = \frac{1}{s}\log\!\left(\frac{p_i}{p_m}\right) + 1$$

This ensures the top-$m$ probability ratios exactly match the targets for any $s$. As $s\to\infty$, all $\bar{y}_s(i)$ approach 1 (becoming uniform). As long as the remaining logits satisfy $\bar{y}_s(j) < 1-\epsilon$, scaling will exponentially push probability mass into the top-$m$: taking $s \ge -\frac{1}{\epsilon}\log\frac{m\delta}{N}$ guarantees $\sum_{j>m} p_j < \delta$ (Proposition 2). As $\epsilon,\delta \to 0$, the probability of this event converges to the lower bound from Design 1. Conclusion: **If a random OPM can make $m$ tokens the most likely, it can assign them arbitrary (summing to 1) exact probabilities.**

**3. Optimal OPM Upper Bound: Reduction to Sign Rank ($d^* \approx 2m$)**

What is the best a trained matrix could do? The authors ask: what is the minimum embedding dimension $d^*$ required such that **any** subset of $m$ tokens can be selected as the top-$m$ by some embedding? They reduce the rank of the logit matrix to the **sign rank** of a $\{-1,+1\}$ indicator matrix $S\in\{-1,+1\}^{N\times\binom{N}{m}}$, where each column corresponds to one $m$-subset. Proposition 4 proves $|d^* - \text{signrank}(S)| \le 1$. Using lemmas from Alon et al. [1], Proposition 5 shows that for $N \ge 3m+1$, $\text{signrank}(S) = 2m+1$, leading to (Corollary 3):

$$2m \le d^* \le 2m+2$$

This means an OPM with an embedding dimension of only $2m$ can select any $m$-subset. For GPT-2 ($d=768$), the optimal $m=383$, far exceeding the random matrix bound of 26.

### A Complete Example

Using **GPT-2** ($N=50257$, $d=768$) as an example:

*   **Random Matrix Theory Lower Bound**: $m \approx 26$ (restricting embedding to $A_m$ row space).
*   **LP Empirical Measurement**: Allowing embeddings in the full space (using $x_\perp$ to suppress other logits), both random and trained GPT-2 matrices reach $m \approx \mathbf{95}$.
*   **Sign Rank Optimal Upper Bound**: The best possible OPM could reach $m = 383$.

The progression 26 $\to$ 95 $\to$ 383 shows the gap between conservative theory, actual random capability, and theoretical optimum. Crucially, the trained GPT-2 and random matrix curves almost overlap—**training neither significantly improves nor degrades the ability to specify top-$m$ tokens.**

## Key Experimental Results

### Main Results

Experiments were conducted on OPMs from GPT-2, GPT2-XL, TinyLlama-1.1B, T5-Large, and Llama2-7B. Using LP, they solved for feasible $x$ and calculated the success rate for selecting random $m$ tokens.

| Model | $N$ | $d$ | Theoretical Lower Bound $m$ | LP Measured $m$ | Sign Rank Optimal $m$ |
|------|-----|-----|------|------|------|
| GPT-2 | 50257 | 768 | 26 | ~95 | 383 |
| TinyLlama-1.1B | 32000 | 2048 | 71 | ~400 | 1023 |
| Llama2-7B | 32000 | 4096 | 143 | ~1070 | — |
| GPT3-175B | 50257 | 12288 | 418 | — | — |

The LP measurements are several times higher than the theoretical lower bounds because the bounds only account for the row space of $A_m$.

### Key Findings
- **Trained OPMs performed similarly to random matrices**: Training does not significantly change the top-$m$ capacity.
- **Lower bounds are conservative; actual capacity is higher**: Measured $m$ values (e.g., >1000 for Llama2) suggest the number of high-probability tokens that can be precisely characterized is quite large.
- **The $m \approx d/5$ empirical law**: Holds approximately when $d \ll N$, providing a reference for model design.
- The authors suggest that in practice (under nucleus sampling), probabilities of tokens beyond the top-1000 have negligible impact on performance.

## Highlights & Insights
- **Reframing the problem**: Shifting from "any distribution" to "top-$m$" shows that the bottleneck is largely irrelevant for practical inference.
- **Tool synergy**: Combining random matrix theory, sign rank, and LP provides a comprehensive view.
- **Sign rank reduction**: An elegant application of combinatorial geometry to analyze LLM expressivity.

## Limitations & Future Work
- **Absolute vs. relative statistics**: The work focuses on top-$m$ probabilities but does not address the relative structure of the long tail.
- **Training vs. Inference**: Whether the bottleneck impacts the training dynamics (rather than just inference capacity) remains an open question.
- **Sign rank constraints**: The upper bound only guarantees tokens can be "selected" as top-$m$; specifying their exact individual probabilities might require higher dimensions.

## Related Work & Insights
- **vs. Yang et al. [26]**: While they argue OPM cannot cover high-dimensional log-probability spaces, this work argues that the subspace coverage is sufficient for top-$m$ tokens.
- **vs. Grivas et al. [7]**: Empirical improvements from increasing OPM rank might result from implicit regularization rather than overcoming the softmax bottleneck itself.
- **vs. [6, 9, 3]**: Extends prior LP frameworks for single-token selection to arbitrary $m \ge 1$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Softmax is not Enough (for Adaptive Conformal Classification)](softmax_is_not_enough_for_adaptive_conformal_classification.md)
- [\[ICLR 2026\] Softmax Transformers are Turing-Complete](softmax_transformers_are_turing-complete.md)
- [\[ICLR 2026\] Language Identification in the Limit with Computational Trace](language_identification_in_the_limit_with_computational_trace.md)
- [\[ICLR 2026\] Testing Most Influential Sets](testing_most_influential_sets.md)
- [\[ICLR 2026\] To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking](to_augment_or_not_to_augment_diagnosing_distributional_symmetry_breaking.md)

</div>

<!-- RELATED:END -->
