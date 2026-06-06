---
title: >-
  [Paper Note] SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel
description: >-
  [ICML 2026][LLM Efficiency][Yat-kernel] SLAY linearizes the Yat-kernel, inspired by the physical "inverse-square interaction," through four steps: (1) spherical normalization…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Yat-kernel"
  - "Spherical Normalization"
  - "Bernstein Theorem"
  - "Positive Random Features"
  - "Gauss–Laguerre Quadrature"
date: 2026-05-08
content_hash: 26e2294e301373e3
---

# SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel

**Conference**: ICML 2026  
**arXiv**: [2602.04915](https://arxiv.org/abs/2602.04915)  
**Code**: None  
**Area**: Linear Attention / Transformer Efficiency / Long Context Modeling / Kernel Methods  
**Keywords**: Yat-kernel, Spherical Normalization, Bernstein Theorem, Positive Random Features, Gauss–Laguerre Quadrature

## TL;DR
SLAY linearizes the Yat-kernel, inspired by the physical "inverse-square interaction," through four steps: (1) spherical normalization, (2) Laplace integral representation via Bernstein theorem, (3) Gauss-Laguerre quadrature, and (4) tensor product positive random features for polynomial+exponential kernels. This yields an $O(L)$ attention mechanism nearly indistinguishable from softmax.

## Background & Motivation

**Background**: Standard Transformers use softmax attention, requiring construction of an $L \times L$ matrix, leading to $O(L^2)$ time and space complexity—prohibitive for long contexts. Many efficient attention methods exist: clustering/hashing (Reformer), kernel approximation + random features (Performer/FAVOR+), low-rank (Linformer), sliding windows, etc.

**Limitations of Prior Work**: (1) Early Rahimi-Recht style trigonometric random features can produce **negative values**, causing unstable training; Performer addresses this with positive random features (PRFs) but still only approximates softmax-like kernels. (2) Softmax couples "alignment" and "magnitude" in $\exp(\mathbf{q}^\top \mathbf{k})$, requiring careful normalization/stabilization. (3) The emerging Yat-kernel $\text{Yat}(\mathbf{q}, \mathbf{k}) = (\mathbf{q}^\top \mathbf{k})^2 / (\|\mathbf{q} - \mathbf{k}\|^2 + \epsilon)$ (inspired by inverse-square forces) is inherently geometry-aware—rewarding alignment and penalizing distance—but **not factorizable**: $\|\mathbf{q} - \mathbf{k}\|^2 = \|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2\mathbf{q}^\top \mathbf{k}$ couples q/k in the denominator, making Performer-style "decompose-rearrange" infeasible; naive implementation remains $O(L^2)$.

**Key Challenge**: The goal is to retain the geometry-awareness (self-regularization + free movement) of the Yat-kernel while achieving linear time complexity—yet the distance term in the denominator is inherently non-separable.

**Goal**: (1) Design a kernel variant that preserves Yat-kernel's geometric properties but is **separable**; (2) derive its linear-time approximation; (3) maintain softmax-level performance with controllable random feature count; (4) rigorously ensure non-negativity to avoid instability from negative attention.

**Key Insight**: By constraining q/k to the unit sphere $\mathbb{S}^{d-1}$, $\|\mathbf{q}-\mathbf{k}\|^2 = 2 - 2\mathbf{q}^\top\mathbf{k}$, so the kernel depends only on angular alignment $x = \mathbf{q}^\top \mathbf{k} \in [-1, 1]$, denoted $\text{Yat}_{\text{sph}}(\mathbf{q}, \mathbf{k}) = x^2 / (C - 2x)$, $C = 2 + \epsilon$. **This decouples q/k**, but $1/(C-2x)$ is still not factorizable. The Bernstein theorem expresses $1/y$ as $\int_0^\infty e^{-sy} ds$; discretizing with Gauss-Laguerre quadrature, each node yields a **polynomial × exponential** kernel—both with existing positive random features.

**Core Idea**: Spherical normalization for decoupling + Bernstein Laplace integral to express the non-separable kernel as a positive mixture of exponentials + Gauss-Laguerre quadrature + tensor product positive random features—these four steps jointly deliver "geometry-awareness + linear time + training stability."

## Method

### Overall Architecture
A four-step pipeline: (1) **Spherical normalization**—L2-normalize each row of q/k to the unit sphere, reducing the Yat-kernel to a scalar function of angular alignment $x$, $\text{Yat}_{\text{sph}}(\hat{\mathbf{q}}, \hat{\mathbf{k}}) = x^2 / (C - 2x)$; (2) **Bernstein Laplace integral**—express the denominator $1/(C-2x)$ as $\int_0^\infty e^{-s(C-2x)} ds$, so the kernel becomes $\int_0^\infty e^{-sC} \cdot [x^2 e^{2sx}] ds$, a **positive-weighted mixture** of "second-order polynomial × exponential dot-product kernels"; (3) **Gauss-Laguerre quadrature**—discretize the integral with $R$ nodes $\{s_r, w_r\}$; (4) **Tensor product positive random features**—for each node $s_r$, construct $\Psi_r(\mathbf{u}) = \sqrt{w_r} \cdot \mathcal{S}(\phi_{\text{poly}}(\mathbf{u}) \otimes \phi_{\text{PRF}}(\mathbf{u}; s_r))$, where $\mathcal{S}$ is a sketching operator; concatenate all node features to obtain $\widetilde{\Psi}(\mathbf{u})$. Attention is computed as $\hat{\mathbf{Y}} = \widetilde{\Psi}(\mathbf{Q}) (\widetilde{\Psi}(\mathbf{K})^\top \mathbf{V}) / \widetilde{\Psi}(\mathbf{Q})(\widetilde{\Psi}(\mathbf{K})^\top \mathbf{1})$, never explicitly constructing the $L \times L$ matrix.

### Key Designs

1. **Spherical Yat Kernel + Bernstein Integral Linearization**:

    - Function: Rewrites the non-separable geometry-aware kernel into a linearizable "positive mixture" form.
    - Mechanism: Unit sphere normalization yields $\|\hat{\mathbf{q}} - \hat{\mathbf{k}}\|^2 = 2(1 - \hat{\mathbf{q}}^\top \hat{\mathbf{k}})$, reducing the kernel to $x^2/(C-2x)$, interpretable as a chordal distance-regularized alignment score $\text{Yat}_{\text{sph}} = (\hat{\mathbf{q}}^\top \hat{\mathbf{k}})^2 / (d_{\mathbb{S}^{d-1}}^2 + \epsilon)$. The function $g(y) = 1/y$ is **completely monotonic** on $(0, \infty)$ (Bernstein theorem), so $1/y = \int_0^\infty e^{-sy} ds$. Substituting $y = C - 2x$ (with $y \ge \epsilon > 0$ for $x \in [-1, 1]$), yields $\text{Yat}_{\text{sph}} = \int_0^\infty e^{-sC} \cdot x^2 e^{2sx} ds$, a **positive-weighted mixture** of polynomial × exponential dot-product kernels.
    - Design Motivation: Decomposing a non-separable kernel into a "positive weighted sum of basic linearizable kernels" is an alternative to Performer—so long as each atomic kernel has a positive random feature approximation, the mixture remains non-negative.

2. **Gauss-Laguerre Quadrature + Tensor Product Positive Features (Anchor + PRF)**:

    - Function: Discretizes the integral into finitely many kernel products, then approximates each factor with existing RF tools.
    - Mechanism: Use $R$-point Gauss-Laguerre quadrature $\int_0^\infty e^{-sC} h(s) ds \approx \sum_r w_r h(s_r)$ (nodes $s_r = t_r/C$, weights $w_r = \alpha_r/C$). The **polynomial factor** $(\hat{\mathbf{q}}^\top \hat{\mathbf{k}})^2$ uses anchor features $\phi_{\text{anc}}(\mathbf{x}) = \frac{1}{\sqrt{P}}[(\mathbf{x}^\top \mathbf{a}_i)^2]_{i=1}^P$ (default; non-negative and no Gram matrix inversion needed); the **exponential factor** $e^{2s \hat{\mathbf{q}}^\top \hat{\mathbf{k}}}$ uses Choromanski's PRF $\phi_{\text{PRF}}(\mathbf{u}; s) = \frac{1}{\sqrt{D}}[\exp(\sqrt{2s} \omega_i^\top \mathbf{u} - s)]_{i=1}^D$ ($\omega_i \sim \mathcal{N}(0, I_d)$, unbiased estimator of the exponential kernel on the unit sphere). **Tensor Product Sketch** $\mathcal{S}$ fuses both and reduces dimensionality, avoiding explicit $D_p \cdot D_r$-dimensional Kronecker vectors.
    - Design Motivation: Anchor features are slightly slower than TensorSketch/Random Maclaurin but **preserve non-negativity**—crucial for avoiding denominator cancellation and numerical collapse. Table 1 compares four polynomial kernel approximations; only explicit $\text{vec}(uu^\top)$ ($d^2$-dimensional, too large) and anchor are both "unbiased + non-negative." Table 2 empirically shows anchor achieves the lowest relative L2 error in 489ms, outperforming Laplace-only (1906ms), with TensorSketch/RM/Nystrom errors 3–4 orders of magnitude higher.

3. **Linear-Time Attention Computation + Numerical Stabilization**:

    - Function: After assembling all features, uses standard linear attention rearrangement, avoiding explicit $L \times L$ matrices.
    - Mechanism: Concatenate all node features to obtain $\widetilde{\Psi}(\mathbf{Q}), \widetilde{\Psi}(\mathbf{K}) \in \mathbb{R}^{L \times m}$ ($m = O(R D_t)$); compute attention as $\hat{\mathbf{Y}} = \widetilde{\Psi}(\mathbf{Q})(\widetilde{\Psi}(\mathbf{K})^\top \mathbf{V}) / \widetilde{\Psi}(\mathbf{Q})(\widetilde{\Psi}(\mathbf{K})^\top \mathbf{1})$, with numerator $L \times d_V$ and denominator $L \times 1$ broadcast row-wise. Overall time complexity is $O(L m d_V)$, space $O(Lm)$, never $L^2$. A small stabilizer $\delta$ is added to the denominator to prevent division by zero.
    - Design Motivation: This follows the standard Performer approach; the innovation lies in the first three steps (how $\widetilde{\Psi}$ is constructed), with the fourth step being straightforward.

### Loss & Training
The training loss is unchanged; only the attention mechanism is replaced. SLAY serves as a drop-in replacement for other attention mechanisms (softmax / Performer FAVOR+ / Cosformer / Linear ELU+1), keeping other hyperparameters unchanged for fair comparison.

## Key Experimental Results

### Main Results
Five evaluation dimensions: (1) Comparison of polynomial factor approximation methods; (2) Computational scalability; (3) 22 synthetic tasks (core capability tests); (4) Extreme classification; (5) Full Transformer model training.

| Evaluation Scenario | Metric | SLAY (Anchor) | Comparison | Notes |
|---------------------|--------|--------------|------------|-------|
| Polynomial Kernel Approximation Quality | Rel. L2↓ | 0.527 | Laplace-only 0.544; Nystrom 70.3; TensorSketch 24823 | anchor has lowest error |
| Polynomial Kernel Approximation Latency | Latency (ms)↓ | **489** | Laplace-only 1906; Hadamard 1932 | anchor is 4× faster |
| Polynomial Kernel Approximation Cosine | Cos↑ | 0.850 | Hadamard 0.732; Nystrom -0.009 | anchor highest alignment |
| Long Sequence Scaling (A100) | Max Sequence Length | 131K | Standard OOM much earlier | $O(L)$ memory/compute |
| Transformer End-to-End | Performance Gap vs softmax | "Almost indistinguishable" | Performer/Cosformer degrade significantly | Core finding |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Full SLAY (Spherical + Bernstein + GL + Anchor + PRF) | Optimal | — |
| w/o Spherical Normalization | Non-separable, still $O(L^2)$ | Spherical is prerequisite for linearization |
| TensorSketch instead of Anchor | 4 orders of magnitude error | Loses non-negativity, denominator collapse |
| Nystrom instead of Anchor | Unacceptable error | Requires Gram inversion, loses non-negativity |
| Laplace-only (no polynomial factor) | Slightly higher error, 4× slower | Polynomial factor is key for geometry-awareness |
| Hadamard (shared $\omega$) | Error close to exact softmax but 1932ms latency | Impractical |

### Key Findings
- **Anchor features are the sweet spot**: Non-negative, unbiased, $O(dP)$ efficient, more stable than Nystrom, more accurate than TensorSketch/RM.
- **Non-negativity is fundamental for stability**: Signed approximations (TensorSketch/RM/Nystrom) can yield negative denominators, causing division by zero or cancellation; see Appendix L.2 for proof.
- **Spherical normalization + Bernstein** is a generalizable template for linearizing "non-separable kernels"—any "distance-regularized alignment score" can use this approach.
- SLAY runs stably at sequence length 131K, while standard attention OOMs much earlier.

## Highlights & Insights
- **Bernstein theorem transmutes non-separable kernels into positive mixtures**: A beautiful example of mathematical tools bridging numerical linear algebra and probabilistic kernel methods.
- **"Geometry-awareness" and "linear time" are no longer mutually exclusive**: Yat-kernel's physical intuition (inverse-square interaction) is preserved, while enjoying $O(L)$ complexity.
- **Anchor features as the polynomial kernel sweet spot**: Table 1 compares four competing methods by dimension/cost/unbiasedness/non-negativity—a clear engineering decision showcase.
- The "sphericalization + Bernstein + GL + tensor-product RF" pipeline is a general template, portable to other physics- or geometry-inspired kernels (e.g., multipole/Coulomb forms).

## Limitations & Future Work
- **Quadrature node count $R$ and PRF count $D$ are hyperparameters**, requiring tuning; no automatic selection strategy is provided.
- The polynomial factor is fixed as quadratic ($( \hat{\mathbf{q}}^\top \hat{\mathbf{k}} )^2$); higher-order polynomial control would require redesigning anchor features.
- Current experiments focus on transformer encoder-style tasks; further validation is needed for autoregressive LM, code, and multimodal tasks.
- Spherical normalization removes the magnitude information of q/k—potentially losing some "weight magnitude" signals; authors argue this is analogous to softmax normalization.
- The optimal configuration of anchor count $P$, PRF count $D$, and quadrature order $R$ varies with $d$ and $L$, requiring case-by-case engineering.

## Related Work & Insights
- **vs Performer / FAVOR+ (Choromanski 2021)**: They linearize softmax; this work linearizes Yat-kernel. Both use PRF, but this work adds the nontrivial Bernstein step.
- **vs Cosformer (Qin 2022)**: Cosformer redesigns the similarity function for $O(L)$, but loses softmax expressiveness; SLAY uses Yat-kernel to achieve both geometry-awareness and $O(L)$.
- **vs Reformer (LSH-based)**: Hashing yields sparse approximations with complexity depending on bucket collisions; SLAY uses dense low-rank approximations with more predictable complexity.
- **vs ELU+1 linear attention**: The simplest feature mapping, with limited performance; SLAY uses a more refined "polynomial + exponential" combination, achieving softmax-level performance.
- **vs Hadamard shared $\omega$ variant**: Same error but 4× higher latency, impractical in engineering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Bernstein theorem into attention linearization is a genuinely novel mathematical contribution; sphericalization + Yat geometry-awareness is a new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five evaluation dimensions (polynomial approximation/scaling/synthetic tasks/extreme classification/end-to-end Transformer) are comprehensive, but lacks large-scale autoregressive LM validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Each derivation is supported by theorems and remarks; mathematically rigorous and implementation is clearly presented.
- Value: ⭐⭐⭐⭐ Provides a high-quality softmax-level, $O(L)$ complexity alternative; the anchor features summary is directly valuable for the linear attention community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prism: Spectral-Aware Block-Sparse Attention](prism_spectral-aware_block-sparse_attention.md)
- [\[ICML 2026\] Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](../../ICLR2026/llm_efficiency/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)

</div>

<!-- RELATED:END -->
