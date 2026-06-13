---
title: >-
  [Paper Note] SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel
description: >-
  [ICML 2026][LLM/NLP][Yat-kernel] SLAY linearizes the Yat-kernel, which is inspired by physical "inverse-square interactions," through a four-step sequence: (1) spherical normalization…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Yat-kernel"
  - "Spherical Normalization"
  - "Bernstein’s Theorem"
  - "Positive Random Features"
  - "Gauss–Laguerre Quadrature"
date: 2026-05-08
content_hash: 7cf5e9dfc4910fa7
---

# SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel

**Conference**: ICML 2026  
**arXiv**: [2602.04915](https://arxiv.org/abs/2602.04915)  
**Code**: None  
**Area**: Linear Attention / Transformer Efficiency / Long Context Modeling / Kernel Methods  
**Keywords**: Yat-kernel, Spherical Normalization, Bernstein’s Theorem, Positive Random Features, Gauss–Laguerre Quadrature

## TL;DR
SLAY linearizes the Yat-kernel, which is inspired by physical "inverse-square interactions," through a four-step sequence: (1) spherical normalization, (2) Laplace integral representation via Bernstein’s theorem, (3) Gauss-Laguerre quadrature, and (4) tensor product positive random features (PRFs) combining polynomial and exponential kernels. This achieves $O(L)$ time complexity while remaining nearly indistinguishable from softmax attention.

## Background & Motivation

**Background**: Standard Transformers utilize softmax attention, necessitating the construction of an $L \times L$ matrix, which results in $O(L^2)$ time and space complexity. This becomes cost-prohibitive in long-context scenarios. Numerous efficient attention mechanisms have been proposed, including clustering/hashing (Reformer), kernel approximation with random features (Performer/FAVOR+), low-rank projections (Linformer), and sliding windows.

**Limitations of Prior Work**: (1) Early Rahimi-Recht style trigonometric random features often generate **negative values**, leading to training instability. While Performer utilizes positive random features (PRFs) to resolve stability, it remains limited to approximating the softmax kernel family. (2) Softmax couples "alignment" and "magnitude" within $\exp(\mathbf{q}^\top \mathbf{k})$, requiring careful normalization and stabilization. (3) The emerging Yat-kernel $\text{Yat}(\mathbf{q}, \mathbf{k}) = (\mathbf{q}^\top \mathbf{k})^2 / (\|\mathbf{q} - \mathbf{k}\|^2 + \epsilon)$ (inspired by inverse-square forces) is naturally geometry-aware—rewarding alignment while penalizing distance. However, it is **non-factorizable**: the $\|\mathbf{q} - \mathbf{k}\|^2 = \|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2\mathbf{q}^\top \mathbf{k}$ term couples q and k in the denominator, preventing the "decompose-and-rearrange" approach used by Performer. A naive implementation remains $O(L^2)$.

**Key Challenge**: The goal is to retain the geometric awareness of the Yat-kernel (self-regularization and flexibility) while achieving linear time complexity, despite the inherently inseparable distance term in the denominator.

**Goal**: (1) Design a factorizable variant of the Yat-kernel that preserves its geometric properties; (2) Derive its linear time approximation; (3) Maintain softmax-level performance while keeping the number of random features manageable; (4) Rigorously guarantee non-negativity to avoid instabilities associated with negative attention weights.

**Key Insight**: The authors observe that if q and k are constrained to the unit sphere $\mathbb{S}^{d-1}$, then $\|\mathbf{q}-\mathbf{k}\|^2 = 2 - 2\mathbf{q}^\top\mathbf{k}$. Consequently, the entire kernel depends solely on the angular alignment $x = \mathbf{q}^\top \mathbf{k} \in [-1, 1]$, denoted as $\text{Yat}_{\text{sph}}(\mathbf{q}, \mathbf{k}) = x^2 / (C - 2x)$, where $C = 2 + \epsilon$. **This decouples the norms of q and k**. To handle the $1/(C-2x)$ term, Bernstein’s theorem is applied to express $1/y$ as $\int_0^\infty e^{-sy} ds$. This integral is then discretized using Gauss-Laguerre quadrature, where each node represents a **polynomial × exponential** kernel—both of which possess known positive random feature approximations.

**Core Idea**: The approach combines spherical normalization for decoupling, Bernstein’s Laplace integral to represent the non-factorizable kernel as a positive mixture of exponential families, Gauss-Laguerre quadrature for discretization, and tensor product PRFs to integrate "geometric awareness, linear time, and training stability" into a single package.

## Method

### Overall Architecture
The pipeline consists of four steps: (1) **Spherical Normalization**—Each row of Q and K is L2-normalized to the unit sphere, reducing the Yat-kernel to a scalar function of angular alignment $x$, $\text{Yat}_{\text{sph}}(\hat{\mathbf{q}}, \hat{\mathbf{k}}) = x^2 / (C - 2x)$; (2) **Bernstein Laplace Integral**—The denominator $1/(C-2x)$ is rewritten as $\int_0^\infty e^{-s(C-2x)} ds$, transforming the kernel into $\int_0^\infty e^{-sC} \cdot [x^2 e^{2sx}] ds$, which is a **positively weighted mixture** of "second-order polynomial × exponential dot-product kernels"; (3) **Gauss-Laguerre Quadrature**—The integral is discretized using $R$ nodes $\{s_r, w_r\}$; (4) **Tensor Product PRF**—For each node $s_r$, features are constructed as $\Psi_r(\mathbf{u}) = \sqrt{w_r} \cdot \mathcal{S}(\phi_{\text{poly}}(\mathbf{u}) \otimes \phi_{\text{PRF}}(\mathbf{u}; s_r))$, where $\mathcal{S}$ is a sketching operator. The final features $\widetilde{\Psi}(\mathbf{u})$ are formed by concatenating features from all nodes. Attention is computed as $\hat{\mathbf{Y}} = \widetilde{\Psi}(\mathbf{Q}) (\widetilde{\Psi}(\mathbf{K})^\top \mathbf{V}) / \widetilde{\Psi}(\mathbf{Q})(\widetilde{\Psi}(\mathbf{K})^\top \mathbf{1})$, avoiding the explicit construction of the $L \times L$ matrix.

### Key Designs

1. **Spherical Yat-Kernel + Bernstein Integral Linearization**:
    - **Function**: Transforms the non-factorizable geometry-aware kernel into a linearizable "positive mixture" form.
    - **Mechanism**: Unit sphere normalization ensures $\|\hat{\mathbf{q}} - \hat{\mathbf{k}}\|^2 = 2(1 - \hat{\mathbf{q}}^\top \hat{\mathbf{k}})$, simplifying the kernel to $x^2/(C-2x)$, which represents an alignment score regularized by spherical chordal distance $\text{Yat}_{\text{sph}} = (\hat{\mathbf{q}}^\top \hat{\mathbf{k}})^2 / (d_{\mathbb{S}^{d-1}}^2 + \epsilon)$. Since $g(y) = 1/y$ is **completely monotonic** on $(0, \infty)$ (Bernstein’s Theorem), it admits the Laplace representation $1/y = \int_0^\infty e^{-sy} ds$. Substituting $y = C - 2x$ (where $y \ge \epsilon > 0$ for $x \in [-1, 1]$) yields $\text{Yat}_{\text{sph}} = \int_0^\infty e^{-sC} \cdot x^2 e^{2sx} ds$, a positively weighted mixture of polynomial × exponential dot-product kernels.
    - **Design Motivation**: Decomposing non-factorizable kernels into positive weighted sums of linearizable "atomic" kernels provides a stable pathway to linear complexity, provided each atomic kernel has a positive random feature approximation.

2. **Gauss-Laguerre Quadrature + Tensor Product PRFs (Anchor + PRF)**:
    - **Function**: Discretizes the integral into a finite sum of kernel products and approximates the factors using RF tools.
    - **Mechanism**: $R$-point Gauss-Laguerre quadrature calculates $\int_0^\infty e^{-sC} h(s) ds \approx \sum_r w_r h(s_r)$. The **polynomial factor** $(\hat{\mathbf{q}}^\top \hat{\mathbf{k}})^2$ is approximated using anchor features $\phi_{\text{anc}}(\mathbf{x}) = \frac{1}{\sqrt{P}}[(\mathbf{x}^\top \mathbf{a}_i)^2]_{i=1}^P$ (default; ensures non-negativity without Gram matrix inversion). The **exponential factor** $e^{2s \hat{\mathbf{q}}^\top \hat{\mathbf{k}}}$ uses Choromanski’s PRF $\phi_{\text{PRF}}(\mathbf{u}; s) = \frac{1}{\sqrt{D}}[\exp(\sqrt{2s} \omega_i^\top \mathbf{u} - s)]_{i=1}^D$. A **Tensor Product Sketch** $\mathcal{S}$ combines these factors while reducing dimensionality to avoid explicit $D_p \cdot D_r$ Kronecker vectors.
    - **Design Motivation**: Anchor features are slower than TensorSketch/Random Maclaurin but **guarantee non-negativity**, which is critical for avoiding numerical collapse caused by denominator cancellations. Table 1 shows that anchor features are the only method that is both "unbiased and non-negative." Table 2 demonstrates that anchor features achieve the lowest relative L2 error in 489ms, outperforming Laplace-only (1906ms) and other methods by several orders of magnitude.

3. **Linear Time Attention Computation + Numerical Stabilization**:
    - **Function**: Computes attention using the rearranged linear attention formulation to avoid $O(L^2)$ complexity.
    - **Mechanism**: Concatenating features yields $\widetilde{\Psi}(\mathbf{Q}), \widetilde{\Psi}(\mathbf{K}) \in \mathbb{R}^{L \times m}$ ($m = O(R D_t)$). Attention is computed as $\hat{\mathbf{Y}} = \widetilde{\Psi}(\mathbf{Q})(\widetilde{\Psi}(\mathbf{K})^\top \mathbf{V}) / \widetilde{\Psi}(\mathbf{Q})(\widetilde{\Psi}(\mathbf{K})^\top \mathbf{1})$. The numerator is $L \times d_V$ and the denominator is an $L \times 1$ vector. Total time complexity is $O(L m d_V)$ with $O(Lm)$ space. A small stability constant $\delta$ is added to the denominator to prevent division by zero.
    - **Design Motivation**: This follows the standard Performer/Linear Attention paradigm; the novelty lies in the derivation of $\widetilde{\Psi}$.

### Loss & Training
The training loss remains unchanged. SLAY acts as a drop-in replacement for other attention mechanisms (Softmax, Performer FAVOR+, Cosformer, Linear ELU+1), maintaining consistent hyperparameters for fair comparison.

## Key Experimental Results

### Main Results
Evaluation was conducted across five dimensions: (1) Polynomial factor approximation comparison; (2) Computational cost scalability; (3) 22 synthetic tasks; (4) Extreme classification; (5) Full Transformer model training.

| Evaluation Scenario | Metric | SLAY (Anchor) | Comparison | Notes |
|---------|------|--------------|------|------|
| Poly-Kernel Approx Quality | Rel. L2↓ | 0.527 | Laplace-only 0.544; Nystrom 70.3; TensorSketch 24823 | Anchor has lowest error |
| Poly-Kernel Approx Latency | Latency (ms)↓ | **489** | Laplace-only 1906; Hadamard 1932 | Anchor is 4× faster |
| Poly-Kernel Approx Cosine | Cos↑ | 0.850 | Hadamard 0.732; Nystrom -0.009 | Anchor has highest alignment |
| Long Sequence Scaling (A100) | Max Seq Length | 131K | Standard Attention OOMs earlier | $O(L)$ Memory/Compute |
| Transformer End-to-End | Perf vs Softmax | "Almost Indistinguishable" | Performer/Cosformer degrade significantly | Core Conclusion |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full SLAY (Spherical + Bernstein + GL + Anchor + PRF) | Optimal | — |
| w/o Spherical Normalization | Non-factorizable $O(L^2)$ | Spherical norm is requisite for linearization |
| TensorSketch instead of Anchor | Error $10^4$ higher | Loses non-negativity; denominator collapse |
| Nystrom instead of Anchor | Unacceptable Error | Requires Gram inversion; loses non-negativity |
| Laplace-only (no poly factor) | Higher error; 4× slower | Polynomial factor is key to geometry awareness |
| Hadamard (shared $\omega$) | Near exact softmax error | High latency (1932ms); impractical |

### Key Findings
- **Anchor features are the sweet spot**: They are non-negative, unbiased, and computationally efficient ($O(dP)$), providing better stability than Nystrom and higher accuracy than TensorSketch/RM.
- **Non-negativity is fundamental to stability**: Signed approximations (TensorSketch/RM/Nystrom) can produce negative values in the denominator, leading to division by zero or cancellation errors (verified in Appendix L.2).
- **Spherical Normalization + Bernstein** provides a general template for linearizing "distance-regularized alignment" kernels.
- SLAY remains functional at 131K sequence length on hardware where standard attention encounters OOM.

## Highlights & Insights
- **Bernstein’s Theorem for Inseparable Kernels**: A mathematically elegant application for converting non-factorizable geometric kernels into positive mixtures of factorizable ones.
- **Breaking the "Geometry vs. Efficiency" Trade-off**: Retains the physical intuition of the Yat-kernel (inverse-square behavior) while benefiting from $O(L)$ complexity.
- **Anchor features as the Poly-kernel Sweet Spot**: The systematic comparison of four approximation methods in Table 1 clarifies the engineering trade-offs between dimensionality, cost, bias, and non-negativity.
- **Generality**: The "Spherical + Bernstein + GL + Tensor PRF" framework is portable to other physics-inspired or geometrically-grounded kernels (e.g., multipole or Coulomb forms).

## Limitations & Future Work
- **Quadrature nodes $R$ and PRF count $D$** are hyperparameters requiring manual tuning; no automated selection strategy is provided.
- The polynomial factor is fixed to the second order ($(\hat{\mathbf{q}}^\top \hat{\mathbf{k}})^2$); adjusting sharpness via higher-order polynomials would require re-designing the anchor features.
- Experiments focused on encoder-style tasks; further validation is needed for auto-regressive LMs, coding tasks, and multimodal domains.
- Spherical normalization ignores the magnitude of q and k; while justified as being similar to softmax normalization, it may lose "importance" signals represented by norm scale.

## Related Work & Insights
- **vs. Performer / FAVOR+ (Choromanski 2021)**: While both use PRFs, Performer linearizes softmax, whereas SLAY linearizes Yat-kernel via the non-trivial Bernstein step.
- **vs. Cosformer (Qin 2022)**: Cosformer redesigns the similarity function for $O(L)$, but sacrifices softmax expressivity; SLAY preserves geometric awareness.
- **vs. Reformer (LSH-based)**: LSH relies on sparse approximations with complexity dependent on hash collisions; SLAY provides a dense low-rank approximation with predictable complexity.
- **vs. ELU+1 Linear Attention**: ELU+1 is a simple feature map with limited expressivity; SLAY yields performance closer to softmax through refined polynomial-exponential combinations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Bernstein's theorem for attention linearization is a genuinely novel mathematical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five-dimensional evaluation is comprehensive, though large-scale autoregressive LM validation is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations supported by theorems and remarks; engineering decisions are clearly articulated.
- Value: ⭐⭐⭐⭐ Provides a high-quality $O(L)$ alternative with softmax-level performance; the study of anchor features is valuable for the linear attention community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](../../NeurIPS2025/llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[ICML 2026\] Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](scheduling_llm_inference_with_uncertainty-aware_output_length_predictions.md)
- [\[NeurIPS 2025\] MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention](../../NeurIPS2025/llm_nlp/monarchattention_zero-shot_conversion_to_fast_hardware-aware_structured_attentio.md)
- [\[ICML 2026\] In-Context Routing (ICR): Train Once, Use Everywhere via Attention-Level Implicit ICL](train_once_reuse_everywhere_generalizable_implicit_in-context_learning_by_routin.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](../../NeurIPS2025/llm_nlp/adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)

</div>

<!-- RELATED:END -->
