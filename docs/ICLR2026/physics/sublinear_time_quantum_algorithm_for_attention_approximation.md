---
title: >-
  [Paper Note] Sublinear Time Quantum Algorithm for Attention Approximation
description: >-
  [ICLR 2026][Physics & Scientific Computing][Quantum Computing] This paper proposes the first quantum data structure with **sublinear** time complexity in sequence length $n$ for approximating row queries of the Transform…
tags:
  - "ICLR 2026"
  - "Physics & Scientific Computing"
  - "Quantum Computing"
  - "Attention Approximation"
  - "Sublinear Algorithm"
  - "Nyström Approximation"
  - "Quantum Sampling"
date: 2026-05-08
content_hash: d22be61fb4325847
---

# Sublinear Time Quantum Algorithm for Attention Approximation

**Conference**: ICLR 2026
**arXiv**: [2602.00874](https://arxiv.org/abs/2602.00874)  
**Code**: None  
**Area**: Physics
**Keywords**: Quantum Computing, Attention Approximation, Sublinear Algorithm, Nyström Approximation, Quantum Sampling
**Authors**: Zhao Song (UC Berkeley/Simons), Jianfei Xue (NYU), Jiahao Zhang, Lichen Zhang (MIT)

## TL;DR

This paper proposes the first quantum data structure with **sublinear** time complexity in sequence length $n$ for approximating row queries of the Transformer attention matrix. The preprocessing time is $\widetilde{O}(\epsilon^{-1} n^{0.5} \cdot \text{poly}(d, s_\lambda, \alpha))$ and each row query takes $\widetilde{O}(s_\lambda^2 + s_\lambda d)$, achieving a quadratic speedup over classical algorithms with respect to $n$.

## Background & Motivation

**The Quadratic Bottleneck of Attention**: The core of the Transformer is the attention module $\text{Att}(Q,K,V) = D^{-1}AV$, where $A = \exp(QK^\top / \sqrt{d}) \in \mathbb{R}^{n \times n}$ and $D = \text{diag}(A \mathbf{1}_n)$. Explicitly constructing the softmax matrix $D^{-1}A$ requires $\Omega(n^2)$ time, which becomes a severe bottleneck for long-sequence LLM inference.

**Limits of Classical Approximations**: A substantial body of work has accelerated attention computation to $\widetilde{O}(nd)$ (sparse attention, linearized kernel methods, KDE, etc.), but $\widetilde{O}(nd)$ is already optimal on classical computers, since the output matrix itself has size $n \times d$.

**The Row-Query Model as a Breakthrough**: During inference, one often only needs to query **specific rows** of $\text{Att}(Q,K,V)$ rather than the full matrix. This "row-query model" circumvents the $\Omega(nd)$ output lower bound. Nevertheless, since each row is a convex combination of rows of $V$, classical algorithms still struggle to achieve sublinear complexity in $n$.

**Potential for Quantum Speedup**: Quantum primitives such as Grover search can provide quadratic speedups for sampling problems. The key insight of this paper is to decompose attention approximation into three subproblems—approximating $A$, approximating $D$, and approximating $V$—each of which can be accelerated by a factor of $\sqrt{n}$ using quantum techniques.

**Limitations of Prior Quantum Approaches**: The prior quantum attention method of [Gao et al.] requires structural assumptions (a bounded number $k$ of highly correlated keys per query), yielding no speedup when $k = n$. This paper makes no structural assumptions.

**Theoretical Significance**: This is the first quantum attention approximation algorithm achieving sublinear complexity in $n$ under the row-query model, bridging the theoretical gap between quantum computing and efficient Transformer inference.

## Method

### Overall Architecture

The algorithm approximates each component of $D^{-1}AV$ in three steps:

1. **Approximating the attention matrix $A$**: via quantum Nyström kernel approximation
2. **Approximating the normalization factor $D$**: via quantum multivariate mean estimation
3. **Approximating the value matrix $V$**: via quantum leverage score sampling

These components are integrated into a quantum data structure `QAttention` supporting preprocessing and row queries.

### Key Design 1: Quantum Nyström Kernel Approximation (Approximating $A$)

**Core Challenge**: $A = \exp(QK^\top)$ is not a symmetric PSD matrix and cannot be directly subjected to Nyström approximation.

**Solution**: An extended dataset $X = \{q_1, \ldots, q_n, k_1, \ldots, k_n\}$ is constructed, forming a $2n \times 2n$ exponential kernel matrix:

$$E = \begin{bmatrix} \exp(QQ^\top) & \exp(QK^\top) \\ \exp(KQ^\top) & \exp(KK^\top) \end{bmatrix}$$

$E$ is PSD and amenable to Nyström approximation, with the upper-right block being exactly $A$.

**Source of Quantum Speedup**: Classical recursive ridge leverage score sampling requires $O(ns_\lambda)$ kernel evaluations. This paper employs the quantum sampler `QSample` based on Grover search to accelerate the sampling step to $\widetilde{O}(n^{0.5} s^{0.5})$ oracle calls, with overall time:

$$\widetilde{O}(n^{0.5} s^{1.5}(d + s) + s^\omega)$$

where $s = O(s_\lambda \log(s_\lambda / \delta))$, $s_\lambda$ is the $\lambda$-statistical dimension defined as $s_\lambda(E) := \text{tr}[E(E + \lambda I)^{-1}]$, and $\omega \approx 2.37$ is the matrix multiplication exponent.

**Approximation Guarantee**: $E \preceq \widetilde{E} \preceq E + \lambda I$, which implies $\|A - \widetilde{A}\| \leq \lambda$ and $\|A - \widetilde{A}\|_F \leq \lambda \sqrt{n}$.

### Key Design 2: Quantum Mean Estimation (Approximating $D$)

**Challenge**: $\widetilde{E}$ cannot be constructed explicitly (it has size $\Omega(n)$), so the normalization factor must be computed implicitly.

**Approach**: Writing $\widetilde{E} = UU^\top$ and partitioning as $U = [U_1; U_2]$, one obtains $\widetilde{A} = U_1 U_2^\top$. The normalization factor $\widetilde{A} \mathbf{1}_n = U_1 (U_2^\top \mathbf{1}_n)$.

The key is to approximate $U_2^\top \mathbf{1}_n$ as a multivariate mean estimation problem. Using quantum multivariate mean estimation (Lemma 2.6), one obtains $\widetilde{\mu}$ satisfying:

$$\|\widetilde{\mu} - U_2^\top \mathbf{1}_n\|_{(U_2^\top U_2)^{-1}} \leq \epsilon$$

with $\widetilde{O}(\epsilon^{-1} n^{0.5} s^{0.5})$ row queries to $U_2$. Querying the normalization factor for the $i$-th row then requires only $O(s(s+d))$ time.

### Key Design 3: Leverage Score Sampling (Approximating $V$)

**Challenge**: Classical methods (e.g., joint row-norm sampling) require reading all entries of $V$ to estimate the Frobenius norm.

**Approach**: Quantum leverage score sampling is employed. The **row incoherence** parameter is introduced:

$$\alpha(V) := \frac{d}{\|V\|_F^2} \cdot \max_{i \in [n]} \frac{\|v_i\|_2^2}{\tau_i}$$

where $\tau_i$ is the leverage score of the $i$-th row of $V$. This parameter satisfies $\alpha \leq d / \text{srank}(V)$.

By sampling $\widetilde{O}(\epsilon^{-2} \alpha)$ rows, an approximate matrix product guarantee (in Frobenius norm) is obtained. The quantum algorithm completes in $\widetilde{O}(\epsilon^{-1} n^{0.5} \alpha^{0.5} d)$ time.

### Main Theorem (Theorem 3.1 / 9.2)

Combining the three components yields the final guarantee:

$$\|\widetilde{D}^{-1}\widetilde{A}\widetilde{V} - \text{Att}(Q,K,V)\|_F \leq \epsilon \cdot (\beta \cdot \|D^{-1}\|) \cdot (\|A\|_F + \lambda\sqrt{n}) \cdot \|V\|_F$$

where $\beta = \frac{1}{1 - (\epsilon\|A\| + \lambda\sqrt{n})\|D^{-1}\|}$.

| Phase | Time Complexity |
|-------|----------------|
| Preprocessing | $\widetilde{O}(\epsilon^{-1} n^{0.5}(s_\lambda^{2.5} + s_\lambda^{1.5}d + \alpha^{0.5}d))$ |
| Row Query | $\widetilde{O}(s_\lambda^2 + s_\lambda d)$ |

### Trade-off in $\lambda$

| Change in $\lambda$ | $s_\lambda$ | Slack in $\|D^{-1}\|$ Constraint | $\beta$ |
|:-:|:-:|:-:|:-:|
| $\uparrow$ | $\downarrow$ | $\downarrow$ | $\uparrow$ |
| $\downarrow$ | $\uparrow$ | $\uparrow$ | $\downarrow$ |

Increasing $\lambda$ reduces the statistical dimension $s_\lambda$ (accelerating computation) but increases $\beta$ (degrading accuracy), necessitating a careful balance.

## Experiments

The experiments primarily validate the key theoretical parameters on real-world models, using pretrained checkpoints of OLMo2-1B and OLMo2-7B.

### Experimental Setup

| Parameter | OLMo2-1B | OLMo2-7B |
|-----------|----------|----------|
| Sequence length $n$ | 4096 | 4096 |
| Value dimension $d$ | 128 | 128 |
| Number of layers $L$ | 16 | 32 |
| Number of attention heads $H$ | 16 | 32 |

The pretrained dataset is used with batch size 2 over 16 batches; statistics are averaged across all heads and layers.

### Main Results: Parameter Validation

| Metric | OLMo2-1B | OLMo2-7B | Theoretical Worst Case | Practical Meaning |
|--------|----------|----------|----------------------|-------------------|
| $\epsilon_{\max}$ | 0.1708 | 0.1685 | — | Maximum $\epsilon$ satisfying $\|D^{-1}\| \leq \frac{1}{\epsilon\|A\|}$ |
| $\|A\|_F / \|A\|$ | 1.3769 | 1.3586 | $\sqrt{n} = 64$ | Ratio of Frobenius to spectral norm |
| $\|V\|_F / \|V\|$ | 11.3137 | 11.3137 | $\sqrt{d} \approx 11.3$ | Related to stable rank of value matrix |
| $d / \text{srank}(V)$ | 1.7126 | 2.1345 | $d = 128$ | Upper bound on row incoherence $\alpha$ |
| $\|A\|_\infty / \|A\|$ | 2.7439 | 2.7439 | $\sqrt{n} = 64$ | Ratio of infinity norm to spectral norm |

### Key Findings

1. **$\epsilon_{\max} \approx 0.17$**: Substantially larger than the commonly used $\epsilon \approx 0.1$, indicating that the $\|D^{-1}\|$ condition is easily satisfied in practice with ample room for parameter tuning.

2. **$\|A\|_F / \|A\| \approx 1.4$**: Far below the theoretical worst case of $\sqrt{n} = 64$. This implies that the Frobenius norm error guarantee is close to a spectral norm guarantee, and the algorithm's accuracy exceeds theoretical predictions.

3. **$d / \text{srank}(V) \approx 1.7\text{-}2.1$**: Far below the upper bound $d = 128$, indicating $\alpha = O(1)$. The row incoherence of the value matrix is very mild and does not cause additional runtime inflation.

4. **$\|A\|_\infty / \|A\| \approx 2.7$**: Far below $\sqrt{n} = 64$, meaning the additive error term $\lambda\sqrt{n}$ in the theoretical analysis is effectively $O(\lambda)$ in practice, substantially widening the feasible range of $\lambda$.

5. **Bit Complexity**: The average value of $\log(\kappa(V))$ is below 4 and at most below 20, which is acceptable in the QRAM model.

## Highlights & Insights

- **First Sublinear Quantum Attention Algorithm**: Under the row-query model, preprocessing time is $O(n^{0.5})$, achieving a quadratic speedup over the classical $O(n)$—a pioneering contribution to this research direction.
- **No Structural Assumptions**: No structural constraints (e.g., sparsity, low-rankness) are imposed on $Q$, $K$, or $V$, making the approach broadly applicable.
- **Elegant Kernel Matrix Construction**: The asymmetric matrix $\exp(QK^\top)$ is embedded into a symmetric PSD kernel matrix $E$, enabling the application of Nyström approximation theory.
- **Mutual Validation of Theory and Experiment**: Empirical results demonstrate that the theoretical worst-case bounds are highly loose in practice, suggesting the algorithm's actual performance may substantially exceed its theoretical guarantees.
- **Introduction of the Row Incoherence Parameter $\alpha$**: This generalizes the classical approximate matrix multiplication requirement for orthonormal columns, enabling efficient sampling for non-orthogonal $V$.

## Limitations & Future Work

1. **Frobenius Norm Error Guarantee Only**: The current analysis provides only a Frobenius norm guarantee (sum of squared errors over all rows), without per-row worst-case spectral norm guarantees.
2. **Additive $\lambda\sqrt{n}$ Term**: The theoretical analysis bounds the infinity norm by $\sqrt{n}$ times the spectral norm, introducing this additive error term—although experiments confirm it is much smaller in practice, it theoretically forces the selection of small $\lambda$.
3. **No Support for Causal Masking**: The Nyström approximation implicitly assumes full query-key interactions and cannot directly accommodate the causal mask required in autoregressive decoding.
4. **QRAM Model Assumption**: The algorithm relies on Quantum Random Access Memory (QRAM), which is not supported by current quantum hardware and is not practically deployable in the near term.
5. **Incomplete Bit Complexity Analysis**: Only a preliminary analysis is provided; the bit complexity of numerical linear algebra in the QRAM model warrants more systematic investigation.

## Related Work & Insights

| Direction | Representative Methods | Relation to This Work |
|-----------|----------------------|----------------------|
| Sparse Attention | Sliding window, BigBird, ETC | Classical $O(n)$, relies on structural priors, no quantum speedup |
| Linear Attention | Random feature maps, Performer | Classical $O(nd)$, limited approximation accuracy |
| Data-Structure Attention | KDE [Alman & Song], Hashing [Chen et al.] | Classical optimal $\widetilde{O}(nd)$; this work achieves further improvement under quantum computation |
| Nyströmformer | [Xiong et al.] | Also uses Nyström, but convergence requires landmarks over all rows; this work performs spectral approximation on the kernel matrix |
| Quantum Attention | [Gao et al.] | Requires $k$-sparsity assumption, no speedup when $k=n$; this work is assumption-free |
| General Quantum ML Methods | Grover search, QMatVec | Fundamental quantum primitives used in this work |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First assumption-free sublinear quantum attention algorithm; the kernel matrix embedding approach is elegant
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Rigorous and complete analysis across three components: Nyström spectral approximation, mean estimation, and leverage score sampling
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited to parameter validation on OLMo2; no end-to-end speedup or accuracy comparison experiments
- **Writing Quality**: ⭐⭐⭐⭐ — Technical overview is clear, proof steps are thorough, and the roadmap is well-organized
- **Value**: ⭐⭐ — Depends on QRAM and cannot be deployed on real quantum hardware in the near term; theoretical contribution outweighs practical impact
- **Overall**: ⭐⭐⭐⭐ — A high-quality theoretical contribution at the intersection of quantum computing and Transformers

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](../../CVPR2026/physics/physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[CVPR 2026\] Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis](../../CVPR2026/physics/continuous_exposure-time_modeling_for_realistic_atmospheric_turbulence_synthesis.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/physics/hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[ICLR 2026\] Feedback-driven Recurrent Quantum Neural Network Universality](feedback-driven_recurrent_quantum_neural_network_universality.md)
- [\[ICML 2026\] Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](../../ICML2026/physics/understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)

</div>

<!-- RELATED:END -->
