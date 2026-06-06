---
title: >-
  [Paper Note] Provably Learning Attention with Queries
description: >-
  [ICML 2026][Interpretability][model stealing] The authors prove that single-head softmax attention can be exactly recovered with remarkable simplicity under value-query access—requiring only $O(d^2)$ queries…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "model stealing"
  - "value query"
  - "single-head attention"
  - "parameter recovery"
  - "compressed sensing"
date: 2026-05-08
content_hash: aa0e7b03d611d2ca
---

# Provably Learning Attention with Queries

**Conference**: ICML 2026  
**arXiv**: [2601.16873](https://arxiv.org/abs/2601.16873)  
**Code**: None  
**Area**: Learning Theory / Model Extraction / Learnability of Transformers  
**Keywords**: model stealing, value query, single-head attention, parameter recovery, compressed sensing

## TL;DR
The authors prove that single-head softmax attention can be exactly recovered with remarkable simplicity under value-query access—requiring only $O(d^2)$ queries, which is much easier than for ReLU MLPs of similar structure. When the head dimension $r\ll d$, compressed sensing reduces this to $O(rd)$. The results extend to noisy oracles, membership queries, and the unidentifiability of multi-head attention.

## Background & Motivation
**Background**: Transformers are now the backbone of industrial deployment, making model extraction attacks a core security concern. Since Tramèr 2016, there has been extensive empirical and theoretical work on extracting feedforward networks; recently, Carlini 2024 even managed to extract embedding matrices and widths from production LLMs. Yet, the fundamental question—whether softmax attention parameters can be provably recovered under black-box API access—remains unanswered.

**Limitations of Prior Work**: (1) Existing query learning theory focuses almost exclusively on ReLU FFNs, requiring strong assumptions like Gaussian input, parameter linear independence, and general position; (2) For attention, even passive learning (without queries) is already very hard due to the non-convexity of softmax plus bilinear scores, often requiring token-selection, max-margin, or SVM tools and only proven in restricted cases; (3) Softmax couples the bilinear token-pair $x_i^\top W x_N$ with cross-position weighting, making it unclear how to solve for $W$ term by term.

**Key Challenge**: Although attention's nonlinearity seems more complex than MLPs (due to softmax normalization and variable sequence length), the authors find that this "softer" nonlinearity actually gives attackers a powerful tool—by controlling the sequence length $N$, softmax degenerates to an invertible sigmoid, reducing attention to a linear system. This is a convenience not available for ReLU MLPs.

**Goal**: (1) Provide the first polynomial-time algorithm and query complexity for recovering single-head attention parameters; (2) Combine this with a ReLU FFN learner to establish the learnability of a one-layer Transformer; (3) Extend results to more realistic settings: low-rank, noisy oracles, membership queries, and multi-head unidentifiability.

**Key Insight**: Leverage the unique advantage of attention—controllable sequence length. For $N=1$, the softmax weight is always 1, yielding $f(X)=x^\top v$ and allowing independent recovery of $v$. For $N=2$, softmax degenerates to a sigmoid, so the oracle output $y$ can be inverted via $\sigma^{-1}(\cdot)$ to obtain a linear equation for each column of $W$.

**Core Idea**: Use "single-token queries to recover $v$" plus "two-token queries with sigmoid inversion to recover each column of $W$" for a total of $d^2+d$ queries to exactly recover $(W^\star,v^\star)$. The same approach, combined with compressed sensing, Lipschitz clipping, and antisymmetric queries, covers all variants: low-rank, noisy, and with ReLU FFN.

## Method

### Overall Architecture
The attacker faces a value oracle for $f_{W^\star,v^\star}(X)=\text{softmax}(x_1^\top W^\star x_N,\dots,x_N^\top W^\star x_N)^\top(Xv^\star)$, aiming to recover $(W^\star,v^\star)\in\mathbb R^{d\times d}\times\mathbb R^d$. The algorithm has two stages: (1) For length-1 input $X=[e_i^\top]$, directly read out $v^\star_i$ with $d$ queries; (2) For each column $w_j=W^\star e_j$, construct a length-2 input $X=[(u+e_j)^\top; e_j^\top]$ so that softmax degenerates to $\sigma(u^\top w_j)$. From the oracle output $y=v^\star_j+\sigma(u^\top w_j)\cdot u^\top v^\star$, invert to get $u^\top w_j=\sigma^{-1}((y-v^\star_j)/(u^\top v^\star))$. Using $d$ linearly independent $u$ suffices to solve for $w_j$. For a one-layer Transformer, antisymmetric queries $\widetilde{\text{VQ}}(X)=\text{VQ}(X)-\text{VQ}(-X)$ eliminate ReLU, after which the attention learner is applied; the FFN part uses existing $\mathcal A_{\text{FFN}}$ (e.g., Milli 2019 or Daniely-Granot 2023). The same approach, with additional tools and patches, handles low-rank, noisy, and membership query settings.

### Key Designs

1. **Length-2 Probe + Sigmoid Inversion (Thm 4.1)**:

    - **Function**: Precisely reduces the nonlinear attention structure to a linear system, recovering $W^\star$ column by column.
    - **Mechanism**: Fix column $j$, set $X=[(u+e_j)^\top; e_j^\top]$. The two scores are $s_1=(u+e_j)^\top W^\star e_j$ and $s_2=e_j^\top W^\star e_j$, so $s_1-s_2=u^\top w_j$ cancels the $e_j^\top W^\star e_j$ term. The attention weight at position 1 is $\alpha=\sigma(u^\top w_j)$. The oracle returns $y=v^\star_j+\alpha\,(u^\top v^\star)$; as long as $u^\top v^\star\neq 0$, $\alpha=(y-v^\star_j)/(u^\top v^\star)\in(0,1)$ allows inversion to $u^\top w_j=\sigma^{-1}(\alpha)$. Using $d$ linearly independent $u$ (i.i.d. Gaussian suffices with high probability) recovers the entire column.
    - **Design Motivation**: The "global normalization" of softmax is the main challenge, but for $N=2$ it degenerates to a sigmoid—a globally invertible, smooth scalar function. First, use $N=1$ to recover $v^\star$, then $N=2$ to isolate $W^\star$. The complexity is "number of columns × probes per column," yielding $O(d^2)$.

2. **Low-Rank Compressed Sensing Recovery of $W^\star$ (Thm 5.1)**:

    - **Function**: In practice, head dimension $r\ll d$ (e.g., 128 vs 4096), so $W^\star=K^\top Q$ is rank-$r$, reducing $d^2$ to $O(rd)$.
    - **Mechanism**: Modify the probe to i.i.d. Gaussian $a,b\sim\mathcal N(0,I_d)$, $X=[(a+b)^\top; b^\top]$, again yielding $\alpha=\sigma(a^\top W^\star b)$. Invert to obtain a rank-1 measurement $t=\langle ab^\top, W^\star\rangle$. Collect $m=O(rd)$ such ROP (rank-one projection) measurements and solve the convex program $\min\|W\|_\ast\ \text{s.t.}\ \langle a_kb_k^\top,W\rangle=t_k$. By Cai-Zhang 2015's RUB, $m\geq Cr(2d)$ ensures exact recovery with high probability.
    - **Design Motivation**: Instead of querying each entry of the $d\times d$ matrix individually, each query gives a "rank-1 linear snapshot" of $W^\star", and compressed sensing theory is used for recovery. This methodological shift—changing the problem, not the algorithm—is a major highlight.

3. **Stable Recovery under Noisy Oracle (Thm 6.1)**:

    - **Function**: Real APIs add small noise to outputs, and $\sigma^{-1}$ is not Lipschitz near 0/1, so naive algorithms can blow up. The paper provides $\epsilon$-accurate polynomial recovery under norm + margin assumptions.
    - **Mechanism**: Scale probes as $a=1/2$, $b=1/W$, construct $X=[(b u+ae_j)^\top; (ae_j)^\top]$ to keep the logit within $|ab\,W^\star_{ij}|\leq 1/2$, so $\alpha^\star=\sigma(ab\,W^\star_{ij})\in[\sigma(-1/2),1-\sigma(-1/2)]$ stays in the Lipschitz region. During estimation, clip $\hat\alpha$ to $[\tau_{\text{clip}},1-\tau_{\text{clip}}]$ before applying $\sigma^{-1}$; by Lemma A.1, $|\sigma^{-1}(\text{clip}(\hat\alpha))-\sigma^{-1}(\alpha^\star)|\leq 5|\hat\alpha-\alpha^\star|$ ensures linear error propagation. Ultimately, $\tau=\mathcal O(\min\{\mu,\epsilon_v/\sqrt d,\mu\epsilon_W/(W^2 d)\})$ suffices for $\|\hat W-W^\star\|_F\leq\epsilon_W,\|\hat v-v^\star\|_2\leq\epsilon_v$.
    - **Design Motivation**: The main risk with noisy oracles is that $\sigma^{-1}$ can explode when $\hat\alpha$ is near $\{0,1\}$. By designing probe scales $a,b$ to "naturally lock the logit in $[-1/2,1/2]$", $\alpha^\star$ always stays in the Lipschitz region, and clipping abnormal estimates ensures stability—an exemplary integration of smoothness analysis into algorithm design.

### Loss & Training
No training is performed; all proofs revolve around query complexity and probabilistic accuracy. The convex program for the low-rank case, $\min\|W\|_\ast$, is guided by the nuclear norm. Multi-head unidentifiability is shown by constructing two different $\{(W_h,v_h)\}$ that induce the same input-output mapping (Prop 7.1).

## Key Experimental Results
This is a theoretical paper with no empirical experiments. The main "data" are query and accuracy complexities under various settings.

### Main Results

| Setting | Query Complexity | Guarantee | Assumptions |
|---------|-----------------|-----------|-------------|
| Exact single-head attention recovery (Thm 4.1) | $O(d^2)$ | Exact | $v^\star\neq 0$ |
| Low-rank single-head attention (Thm 5.1) | $O(rd)$ | Exact, probability $1-e^{-\Omega(m)}$ | $\text{rank}(W^\star)\leq r$, $v^\star\neq 0$ |
| Noisy oracle (Thm 6.1) | $O(d^2)$ | $\|\hat W-W^\star\|_F\leq\epsilon_W$ | $\|W^\star\|_F\leq W$, $\min v^\star\geq\mu$ |
| One-layer Transformer (with ReLU MLP) | $Q_{\text{FFN}}(d,m)+O(d^2)$ | Exact, depends on $\mathcal A_{\text{FFN}}$ | $A^\star w_o^\star\neq 0$ |
| Multi-head attention | Unidentifiable | No algorithm exists | No additional structure |

### Ablation Study

| Variant | Query Count / Accuracy | Notes |
|---------|-----------------------|-------|
| Value query | $O(d^2)$, exact | Baseline |
| Membership query (App. B) | poly + bisection | Only returns ±1 labels, higher complexity |
| Antisymmetric query removes ReLU | $2\times$ single query | Used for the attention part of one-layer Transformer |

### Key Findings
- Single-head attention and single-hidden-layer ReLU MLP both have "one nonlinearity + one matrix + one vector," but the former is extremely easy to learn via queries, while the latter still requires strong assumptions—because softmax is a globally invertible smooth function, unlike ReLU.
- Careful choice of probe scale (Gaussian for low-rank, $b=1/W$ for noisy) is crucial for algorithm closure and is the most practical methodological insight.
- Multi-head attention is unidentifiable due to arbitrary permutation and linear combination of heads; additional structure (e.g., orthogonality) is required for identifiability.
- The step of directly reading $v^\star$ via length-1 queries may seem trivial, but it is necessary for subsequent sigmoid inversion: without $v^\star$, the denominator for inferring $\alpha$ from $y$ is missing.
- In the membership query setting (only binary labels), the authors use bisection to reduce sigmoid inversion to multiple comparison queries; complexity remains polynomial but with a much larger constant.

## Highlights & Insights
- "Leveraging controllable sequence length to degenerate softmax into sigmoid" is an attack surface unique to attention, placing the security of a one-layer Transformer at a much weaker position than MLPs—a clear warning for LLM API providers.
- Using antisymmetric queries $f(X)-f(-X)$ to eliminate ReLU and reduce to a linear equivalent problem is a trick that can be reused for any network structure with "odd transformation + even nonlinearity."
- Framing model extraction in a PAC-style query complexity framework bridges the security and learning theory communities—most prior attack papers are empirical.
- Reducing queries from $O(d^2)$ to $O(rd)$ in the low-rank setting is well-suited to modern LLMs with head dimension $r\sim 128$ and width $d\sim 4096$—meaning SOTA-scale attention parameters can be extracted in practice.
- Designing probe scales $a=1/2, b=1/W$ to lock $\sigma^{-1}$ in the Lipschitz region exemplifies "embedding smoothness analysis deep into algorithm design," providing a reference for all estimation problems involving unstable inverse functions.

## Limitations & Future Work
- Single-head + linear MLP is the simplest version; real Transformers are multi-layer, multi-head, with LayerNorm and position encoding, so there are many abstraction layers between theory and practice.
- The noisy setting requires margin $\mu>0$, which is unfriendly to the sparse/near-zero weights common in LLMs.
- For multi-head unidentifiability, only a counterexample is given; the identifiability boundaries under structural assumptions (e.g., orthogonal heads, FFN gates) are not deeply explored.
- The assumption $v^\star\neq 0$—in the degenerate case $v^\star=0$, $W^\star$ is completely unidentifiable, but the paper does not provide engineering guidance on how to detect this boundary in practice.

## Related Work & Insights
- **vs Chen et al. 2021 (Gaussian-input ReLU MLP)**: They also recover 2-layer ReLU MLPs in the query model, but require Gaussian input and distribution-dependent arguments; this work proves attention needs no distributional assumptions.
- **vs Daniely-Granot 2023 (general position ReLU)**: The FFN subroutine for one-layer Transformer can directly use their algorithm, combining "existing FFN learner + our attention learner."
- **vs Carlini et al. 2024**: Their work is empirical extraction on industrial LLMs; this paper is the minimal, provable, passive version of the same problem—the two lines are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First provable parameter recovery result for softmax attention, with elegant methods
- Experimental Thoroughness: ⭐⭐ Entirely theoretical, no empirical or toy demos
- Writing Quality: ⭐⭐⭐⭐⭐ Main theorem proofs are readable within a page or two, with clear motivation for each lemma
- Value: ⭐⭐⭐⭐ Opens new tools for both model extraction security research and theoretical analysis of attention

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](singular_vectors_of_attention_heads_align_with_features.md)
- [\[ICML 2026\] The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](../../NeurIPS2025/interpretability/learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)

</div>

<!-- RELATED:END -->
