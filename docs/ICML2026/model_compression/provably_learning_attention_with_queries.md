---
title: >-
  [Paper Note] Provably Learning Attention with Queries
description: >-
  [ICML 2026][Model Compression][model stealing] The authors demonstrate that single-head softmax attention can be precisely recovered with surprising simplicity under value-query access—requiring only $O(d^2)$ queries…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "model stealing"
  - "value query"
  - "single-head attention"
  - "parameter recovery"
  - "compressed sensing"
date: 2026-05-08
content_hash: 6980cd31748a9955
---

# Provably Learning Attention with Queries

**Conference**: ICML 2026  
**arXiv**: [2601.16873](https://arxiv.org/abs/2601.16873)  
**Code**: None  
**Area**: Learning Theory / Model Extraction / Transformer Learnability  
**Keywords**: model stealing, value query, single-head attention, parameter recovery, compressed sensing

## TL;DR
The authors demonstrate that single-head softmax attention can be precisely recovered with surprising simplicity under value-query access—requiring only $O(d^2)$ queries, which is significantly easier than ReLU MLPs of similar structure. When the head dimension $r \ll d$, the complexity can be reduced to $O(rd)$ using compressed sensing. The findings are extended to noisy oracles, membership queries, and the non-identifiability of multi-head attention.

## Background & Motivation
**Background**: Transformers are the backbone of industrial deployment, making model stealing attacks a core concern in security research. Since Tramèr 2016, there has been extensive empirical and theoretical work on extracting feed-forward networks; recently, Carlini 2024 even demonstrated the extraction of embedding matrices and widths from production-level LLMs. However, the fundamental question of whether softmax attention parameters can be provably recovered under black-box API access had not been directly addressed.

**Limitations of Prior Work**: (1) Existing query learning theories focus almost exclusively on ReLU FFNs and require strong assumptions such as Gaussian inputs, linearly independent parameters, or general position. (2) For attention, the passive learning problem (without queries) is inherently difficult because the softmax and bilinear scores are non-convex; researchers often rely on specialized tools like token-selection, max-margin, or SVMs in restricted settings. (3) Softmax couples the bilinear term $x_i^\top W x_N$ of token pairs with cross-position weighting, making it unclear how to solve for $W$ entry-by-entry using naive methods.

**Key Challenge**: While attention non-linearity appears more complex than MLPs (due to softmax normalization and variable sequence length), the authors discovered that this "relatively soft" non-linearity actually aids the attacker. By controlling the sequence length $N$, the softmax can be reduced to an invertible sigmoid, allowing attention to be reformulated as a system of linear equations—a convenience entirely absent in ReLU MLPs.

**Goal**: (1) Provide the first polynomial-time algorithm and query complexity for recovering single-head attention parameters. (2) Connect it to a ReLU FFN learner to achieve learnability for a single Transformer layer. (3) Provide results for more realistic scenarios including low-rank structures, noisy oracles, membership queries, and multi-head non-identifiability.

**Key Insight**: Leverage the unique advantage of attention's "controllable length": when $N=1$, the softmax weight is constant at 1, yielding $f(X) = x^\top v$, allowing $v$ to be recovered independently. When $N=2$, the softmax reduces to a sigmoid, and $\sigma^{-1}(\cdot)$ can be solved from oracle outputs to obtain linear equations regarding the columns of $W$.

**Core Idea**: Utilize a combination of "single-token queries to recover $v$" and "dual-token queries to recover $W$ column-wise via sigmoid inversion" for a total of $d^2+d$ queries to precisely recover $(W^\star, v^\star)$. This approach is combined with compressed sensing, Lipschitz clipping, and antisymmetric queries to cover variants including low-rank, noisy, and ReLU FFN-integrated models.

## Method

### Overall Architecture
The attacker faces a value oracle $f_{W^\star,v^\star}(X)=\text{softmax}(x_1^\top W^\star x_N,\dots,x_N^\top W^\star x_N)^\top(Xv^\star)$ and aims to recover $(W^\star,v^\star)\in\mathbb R^{d\times d}\times\mathbb R^d$. The algorithm proceeds in two stages: (1) Use length-1 inputs $X=[e_i^\top]$ to read $v^\star_i$ directly in $d$ queries. (2) For each column $w_j=W^\star e_j$, construct length-2 inputs $X=[(u+e_j)^\top; e_j^\top]$ to reduce softmax to $\sigma(u^\top w_j)$. From the output $y=v^\star_j+\sigma(u^\top w_j)\cdot u^\top v^\star$, solve for $u^\top w_j=\sigma^{-1}((y-v^\star_j)/(u^\top v^\star))$. Using $d$ linearly independent $u$ allows for solving the column. A full Transformer layer uses antisymmetric queries $\widetilde{\text{VQ}}(X)=\text{VQ}(X)-\text{VQ}(-X)$ to eliminate ReLUs before calling the attention learner; the FFN part is handled by existing learners $\mathcal A_{\text{FFN}}$ (e.g., Milli 2019).

### Key Designs

1.  **Length-2 Probing + Sigmoid Inversion (Thm 4.1)**:
    - **Function**: Precisely transforms the non-linear attention structure into a system of linear equations to recover $W^\star$ column by column.
    - **Mechanism**: Fixing column $j$, set $X=[(u+e_j)^\top; e_j^\top]$. The two scores are $s_1=(u+e_j)^\top W^\star e_j$ and $s_2=e_j^\top W^\star e_j$. The difference $s_1-s_2=u^\top w_j$ cancels out the $e_j^\top W^\star e_j$ term. The attention weight for position 1 is $\alpha=\sigma(u^\top w_j)$. The oracle returns $y=v^\star_j+\alpha\,(u^\top v^\star)$. Provided $u^\top v^\star \neq 0$, $\alpha$ is calculated and $u^\top w_j = \sigma^{-1}(\alpha)$ is obtained.
    - **Design Motivation**: While the softmax global normalization is a hurdle, at $N=2$ it collapses into a sigmoid—a globally invertible, smooth scalar function. By isolating $v^\star$ with $N=1$, $W^\star$ can be solved via the sigmoid inverse. Complexity is $O(d^2)$ based on columns $\times$ probes per column.

2.  **Low-Rank Recovery via Compressed Sensing (Thm 5.1)**:
    - **Function**: Reduces query complexity from $d^2$ to $O(rd)$ when the head dimension $r \ll d$ (e.g., 128 vs 4096), where $W^\star = K^\top Q$ is rank-$r$.
    - **Mechanism**: Use i.i.d. Gaussian probes $a, b \sim \mathcal{N}(0, I_d)$ with $X=[(a+b)^\top; b^\top]$. Similarly, $\alpha = \sigma(a^\top W^\star b)$ yields a rank-1 measurement $t = \langle ab^\top, W^\star \rangle$. After collecting $m = O(rd)$ such ROP (rank-one projection) measurements, solve the convex program $\min \|W\|_\ast \text{ s.t. } \langle a_k b_k^\top, W \rangle = t_k$.
    - **Design Motivation**: Instead of querying every entry of the $d \times d$ matrix, each query provides a "rank-1 linear snapshot" of $W^\star$. Applying compressed sensing theory to these snapshots is a methodological highlight of this work.

3.  **Stable Recovery under Noisy Oracles (Thm 6.1)**:
    - **Function**: Addresses real-world APIs with output noise where $\sigma^{-1}$ is not Lipschitz near $\{0, 1\}$, preventing naive algorithm failure.
    - **Mechanism**: Scale probes to $a=1/2, b=1/W$ and construct $X=[(b u+ae_j)^\top; (ae_j)^\top]$ to bound logits $|ab\,W^\star_{ij}| \leq 1/2$. This ensures $\alpha^\star$ stays within the Lipschitz interval $[\sigma(-1/2), 1-\sigma(-1/2)]$. During estimation, clip the observed $\hat{\alpha}$ and apply $\sigma^{-1}$. Error propagation remains linear: $|\sigma^{-1}(\text{clip}(\hat{\alpha}))-\sigma^{-1}(\alpha^\star)| \leq 5|\hat{\alpha}-\alpha^\star|$.
    - **Design Motivation**: To prevent error explosion near the boundaries of the sigmoid inverse, the probes are designed to "lock" logits into $[-1/2, 1/2]$. This embeds smoothness analysis directly into the algorithm design.

### Loss & Training
This paper is proof-based and does not involve training. Proofs center on query complexity and probabilistic accuracy. The low-rank recovery uses a convex objective guided by the nuclear norm $\|W\|_\ast$. Non-identifiability for multi-head attention is shown in Prop 7.1 by constructing distinct $\{(W_h, v_h)\}$ that induce the same mapping.

## Key Experimental Results
This is a theoretical paper; "data" refers to query and precision complexities.

### Main Results

| Setting | Query Complexity | Guarantee | Assumptions |
| :--- | :--- | :--- | :--- |
| Exact Single-Head Recovery (Thm 4.1) | $O(d^2)$ | Exact | $v^\star \neq 0$ |
| Low-Rank Single-Head (Thm 5.1) | $O(rd)$ | Exact, prob $1-e^{-\Omega(m)}$ | $\text{rank}(W^\star) \leq r$, $v^\star \neq 0$ |
| Noisy Oracle (Thm 6.1) | $O(d^2)$ | $\|\hat{W}-W^\star\|_F \leq \epsilon_W$ | $\|W^\star\|_F \leq W$, $\min v^\star \geq \mu$ |
| Single Transformer Layer | $Q_{\text{FFN}}(d,m)+O(d^2)$ | Exact, depends on $\mathcal A_{\text{FFN}}$ | $A^\star w_o^\star \neq 0$ |
| Multi-head Attention | Unidentifiable | No algorithm exists | No extra structure |

### Ablation Study

| Variant | Queries / Accuracy | Notes |
| :--- | :--- | :--- |
| Value query | $O(d^2)$, Exact | Baseline |
| Membership query (App. B) | poly + bisection | Only $\pm 1$ labels; higher complexity |
| Antisymmetric query | $2\times$ single query | Used to cancel ReLU in Transformer layers |

### Key Findings
- While single-head attention and single-hidden-layer ReLU MLPs both contain "one non-linearity + one matrix + one vector," the former is significantly easier to learn via queries due to the global invertibility of softmax compared to ReLU.
- The choice of probe scales (Gaussian for low-rank, $b=1/W$ for noise) is critical for algorithm closure.
- Multi-head attention is inherently unidentifiable due to arbitrary permutations and linear combinations between heads unless additional structural assumptions (e.g., orthogonality) are made.
- The $N=1$ query to read $v^\star$ is a necessary prerequisite for the sigmoid inversion in $N=2$ queries.
- Under membership queries (binary labels), bisection reduces the sigmoid inversion to multiple comparison queries; complexity remains polynomial but constants increase significantly.

## Highlights & Insights
- Using "controllable sequence length" to reduce softmax to sigmoid is an attention-specific attack surface, placing a single Transformer layer in a much more vulnerable position than an MLP.
- The use of antisymmetric queries $f(X)-f(-X)$ to cancel ReLUs and transform the problem into a linear equivalence can be reused for analyzing any architecture with "odd transformation + even non-linearity."
- By framing model extraction within PAC-style query complexity, the paper bridges the security and learning theory communities.
- Reducing query complexity from $O(d^2)$ to $O(rd)$ aligns perfectly with modern LLM dimensions ($r \sim 128$ vs $d \sim 4096$), implying that state-of-the-art scale models are vulnerable.
- Locking the sigmoid inverse into a Lipschitz region using specific probe scales is a masterclass in embedding smoothness analysis into algorithm design.

## Limitations & Future Work
- The single-head + linear MLP setup is a simplified version; real Transformers involve multiple layers, LayerNorm, and position encodings.
- The noisy case requires a margin $\mu > 0$, which may not hold for sparse or near-zero weights common in LLMs.
- Multi-head non-identifiability is demonstrated via counterexamples but the boundaries of identifiability under structural assumptions (e.g., FFN gates) are not fully explored.
- The assumption $v^\star \neq 0$ is required; in the degenerate case where $v^\star = 0$, $W^\star$ is unidentifiable.

## Related Work & Insights
- **vs Chen et al. 2021**: Their recovery of 2-layer ReLU MLPs requires Gaussian inputs and distribution-dependent proofs; this work shows attention recovery requires no distribution assumptions.
- **vs Daniely-Granot 2023**: This paper's Transformer layer FFN subroutine directly utilizes their algorithms, showcasing a modular approach.
- **vs Carlini et al. 2024**: While they perform empirical extraction on industrial LLMs, this work provides the minimal provable passive version of the same problem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First provable parameter recovery for softmax attention.
- Experimental Thoroughness: ⭐⭐ Purely theoretical, no empirical demos.
- Writing Quality: ⭐⭐⭐⭐⭐ Proofs are concise and motivations are clear.
- Value: ⭐⭐⭐⭐ Provides new tools for both security research and attention theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](../../AAAI2026/model_compression/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)

</div>

<!-- RELATED:END -->
