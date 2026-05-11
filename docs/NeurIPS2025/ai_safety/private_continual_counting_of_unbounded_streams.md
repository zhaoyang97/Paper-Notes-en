---
title: >-
  [Paper Note] Private Continual Counting of Unbounded Streams
description: >-
  [NeurIPS 2025][AI Safety][Differential Privacy] This paper proposes a novel matrix factorization method based on logarithmic perturbation…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Differential Privacy"
  - "Continual Observation"
  - "Matrix Factorization"
  - "Streaming Computation"
  - "Unbounded Input"
date: 2026-05-08
content_hash: 12d0117f1ed2e6d8
---

# Private Continual Counting of Unbounded Streams

**Conference**: NeurIPS 2025
**arXiv**: [2506.15018](https://arxiv.org/abs/2506.15018)
**Code**: [GitHub](https://github.com/ben-jacobsen/central-dpolo/)
**Area**: AI Safety
**Keywords**: Differential Privacy, Continual Observation, Matrix Factorization, Streaming Computation, Unbounded Input

## TL;DR

This paper proposes a novel matrix factorization method based on logarithmic perturbation, achieving for the first time a differentially private continual counting algorithm that simultaneously satisfies the three properties of "unbounded streams," "smooth error," and "near-optimal asymptotic error," with variance $O(\log^{2+2\alpha}(t))$ at time step $t$ for any $\alpha > 0$.

## Background & Motivation

- **Background**: Differentially private continual counting refers to maintaining an accurate running sum over a sensitive data stream. This problem has broad applications in privacy-preserving optimization (e.g., federated learning, private gradient descent), online learning, and streaming data analysis. Large-scale private deployments such as Google's private next-word prediction and Smart Select models rely on this primitive.

- **Limitations of Prior Work**: Matrix factorization-based algorithms—particularly the $M_{\text{count}}^{1/2}$ factorization—have substantially improved the privacy–utility trade-off over classical binary tree mechanisms. However, these state-of-the-art algorithms share a fundamental assumption: **the input size $n$ must be known in advance.** While this assumption is natural in model training scenarios, it is unrealistic in applications such as public health surveillance. Apple's deployment of differentially private user data collection is a notable example: due to this limitation, only single-day privacy could be guaranteed, causing cumulative privacy loss to grow without bound over time—illustrating the practical impact of the unbounded input problem.

- **Key Challenge**: All existing solutions face a fundamental trilemma:
  - **Doubling trick**: Preserves asymptotic performance but causes non-smooth error growth.
  - **Fixed budget**: Maintains performance and smoothness but is not truly unbounded—computation halts once the privacy budget is exhausted.
  - **Independent noise**: Naturally unbounded and smooth, but error is highly suboptimal.

## Method

### Overall Architecture

The core idea is to identify special analytic functions for constructing lower-triangular Toeplitz (LTToep) matrix factorizations $L \cdot R = M_{\text{count}}$, such that the column norms of $R$ are bounded (enabling unbounded use), while the row norms of $L$ grow as slowly as possible (approaching optimal error). The key innovation is introducing a logarithmic perturbation to $(1-z)^{-1/2}$—the function corresponding to $M_{\text{count}}^{1/2}$—to render it $L^2$-integrable, thereby enabling unbounded inputs.

### Key Designs

1. **Logarithmic Perturbation Matrix Factorization**:

    - Define the generating function family: $f(z;\gamma,\delta) = f_1(z) \cdot f_2(z;\gamma) \cdot f_3(z;\delta)$
    - where $f_1(z) = \frac{1}{\sqrt{1-z}}$, $f_2(z;\gamma) = \left(\frac{1}{z}\ln\frac{1}{1-z}\right)^\gamma$, $f_3(z;\delta) = \left(\frac{2}{z}\ln\frac{1}{z}\ln\frac{1}{1-z}\right)^\delta$
    - Set $R = \text{LTToep}(f(z; -\frac{1}{2}-\alpha, \delta))$, $L = \text{LTToep}(f(z; \frac{1}{2}+\alpha, -\delta))$
    - **Design Motivation**: Directly adjusting the exponent of $(1-z)^{-1/2}$ (e.g., to $(1-z)^{-1/2+\alpha}$) yields $O(t^\alpha)$ error, since $(1-z)^{-\alpha}$ diverges too rapidly as $z \to 1^-$. The factor $\ln(1/(1-z))$ diverges more slowly, keeping the error at logarithmic scale.

2. **Four Key Properties (Theorem 1)**:

    - **Joint validity**: $LR = \text{LTToep}(f_L \cdot f_R) = \text{LTToep}(\frac{1}{1-z}) = M_{\text{count}}$
    - **Bounded sensitivity**: The Flajolet–Odlyzko asymptotic expansion is used to show that the sum of squares of $[z^n]f_R$ converges (when $\gamma < -1/2$), and Parseval's theorem is applied to compute $\|R\|_{1 \to 2}^2 = \frac{1}{2\pi}\int_0^{2\pi}|f_R(e^{i\theta})|^2 d\theta$ exactly.
    - **Near-optimal asymptotic error**: $\|[L]_n\|_{2 \to \infty} = O(\log^{1+\alpha}(n))$
    - **Computability**: Achieves $O(t)$ space and amortized $O(\log t)$ time per step via FFT.

3. **Parameter Selection Strategy**:

    - Choice of $\alpha$: Setting $\alpha = 0.01$ introduces only a $1.03\times$ overhead for the global population ($8.2\text{B}$ individuals $\times$ 100 data points).
    - Three options for $\delta$: $\delta = 0$ (fastest), $\delta = -\gamma$ (better error, approximately $1.5\times$ slower), and $\delta = -6\gamma/5$ (matches the first two Taylor coefficients of $M_{\text{count}}^{1/2}$ exactly, optimal for $t > 2^{20}$).
    - **Design Motivation**: Although the parameter space is large, both theoretical analysis and experiments point to a small number of optimal configurations.

### Loss & Training

This is a theoretical work. The core algorithm (Algorithm 1):
- At initialization, computes the sensitivity $\Delta$ precisely via numerical integration.
- At each $t = 2^m$, uses FFT to batch-compute the next $t$ coefficients of $L$ and the product $Lz$.
- Samples Gaussian noise $z_s \sim \mathcal{N}(0, C_{\varepsilon,\delta}^2 \Delta^2)$.
- Outputs $S_t + (Lz)_t$.

## Key Experimental Results

### Main Results: Algorithm Variance Comparison

| Algorithm | Variance $\mathbb{V}(\mathcal{M}(X_t))$ | Time | Space | Smooth | Unbounded |
|---|---|---|---|---|---|
| Binary Tree | $O(\log t \cdot \log n)$ | $O(t)$ | $O(\log t)$ | No | No |
| Hybrid Binary Tree | $O(\log^2 t + \log t)$ | $O(t)$ | $O(\log t)$ | No | Yes |
| Smooth Binary Tree | $O((\log n + \log\log n)^2)$ | $O(t)$ | $O(\log t)$ | Yes | No |
| Sqrt Matrix | $O(\log t \cdot \log n)$ | $O(n\log n)$ | $O(n)$ | Yes | No |
| **Algorithm 1** | $O(\log^{2+2\alpha} t)$ | $O(t\log t)$ | $O(t)$ | **Yes** | **Yes** |

### Ablation Study: Practical Performance Comparison

| Configuration | Variance Ratio at $t=2^{12}$ (vs. Sqrt) | Variance Ratio at $t=2^{24}$ | Notes |
|---|---|---|---|
| $\delta=0$ | ~$1.3\times$ | ~$1.4\times$ | Fastest but slightly higher error |
| $\delta=-\gamma$ | ~$1.1\times$ | ~$1.2\times$ | Intermediate |
| $\delta=-6\gamma/5$ | ~$1.15\times$ | ~$1.1\times$ | Optimal for $t>2^{20}$ |
| Algorithm 2 (approx.) | Indistinguishable from Alg. 1 | Indistinguishable from Alg. 1 | $>5\times$ speedup |

### Key Findings

- For $t$ up to $2^{24}$ (approximately 16.8 million), Algorithm 1's variance never exceeds $1.5\times$ that of the optimal bounded algorithm.
- Algorithm 2 (approximate variant) is virtually indistinguishable from the exact version for $t \geq 2^{11}$, while reducing computational overhead by over 80%.
- The hybrid mechanism (using Algorithm 1 as the unbounded sub-component) can achieve exact $O(\log^2 t)$ variance, but error fluctuates sharply when $t$ approaches powers of 2.

## Highlights & Insights

- The paper elegantly resolves the long-standing "unbounded–smooth–optimal" trilemma in continual private counting.
- It approaches the matrix factorization problem from the perspective of analytic function theory, converting an algebraic problem into one of complex analysis.
- The Flajolet–Odlyzko classical asymptotic expansion theorem is employed to precisely control coefficient growth rates—a theoretical tool rarely seen in the privacy literature.
- An intriguing open question is raised: does there exist an LTToep factorization achieving exact $O(\log^2 n)$ variance?

## Limitations & Future Work

- Space complexity of $O(t)$ is significantly higher than the $O(\log t)$ of binary tree methods, making it unsuitable for extremely long streams.
- Variance is $O(\log^{2+2\alpha} t)$ rather than the optimal $O(\log^2 t)$, though the practical difference is minimal.
- Only additive Gaussian noise mechanisms are discussed; other noise types are not considered.
- The authors conjecture that no LTToep factorization can achieve exact $O(\log^2 t)$ variance, but this remains unproven.

## Related Work & Insights

- The paper directly builds upon the rational approximation approach of Dvijotham et al., but in the opposite direction: their work seeks simpler functions for computational efficiency, whereas this paper seeks more complex functions to enable unboundedness.
- A detailed theoretical and empirical comparison is made against the $M_{\text{count}}^{1/2}$ method of Henzinger et al.
- The approach has direct application potential for privacy-preserving computation in federated learning, such as model aggregation in DP-SGD.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Addresses a combinatorial optimization problem from the perspective of complex analysis; theoretical contribution is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Numerical experiments accurately demonstrate theoretical results, though no real-world application validation is provided.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretically rigorous yet clearly written; the technical overview section provides excellent intuitive explanations.
- Value: ⭐⭐⭐⭐ Resolves an important theoretical problem with practical implications for large-scale private system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](private_zeroth-order_optimization_with_public_data.md)
- [\[NeurIPS 2025\] Differentially Private High-dimensional Variable Selection via Integer Programming](differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)

</div>

<!-- RELATED:END -->
