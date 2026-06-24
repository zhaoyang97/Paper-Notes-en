---
title: >-
  [Paper Note] On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training
description: >-
  [ICLR 2026][Learning Theory][Kolmogorov-Arnold Networks] This paper proves that under the overparameterized setting with training restricted to the first-layer coefficients, two-layer KANs using gradient descent converge to the global optimum (zero training error). It provides a fine-grained convergence rate determined by the "projection of labels onto the eigenstructure of the KAN Tangent Kernel" and demonstrates that KANs require only $m=O(n^2)$ hidden layer width to guaran…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Optimization Convergence"
  - "Kolmogorov-Arnold Networks"
  - "Neural Tangent Kernel (NTK)"
  - "Overparameterization"
  - "Global Convergence"
  - "lazy training"
date: 2026-05-08
content_hash: b33cf7745c9eb4c8
---

# On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=buuwRBYfrP](https://openreview.net/forum?id=buuwRBYfrP)  
**Code**: TBC  
**Area**: Learning Theory / Optimization Convergence  
**Keywords**: Kolmogorov-Arnold Networks, Neural Tangent Kernel (NTK), Overparameterization, Global Convergence, lazy training

## TL;DR
This paper proves that under the overparameterized setting with training restricted to the first-layer coefficients, two-layer KANs using gradient descent converge to the global optimum (zero training error). It provides a fine-grained convergence rate determined by the "projection of labels onto the eigenstructure of the KAN Tangent Kernel" and demonstrates that KANs require only $m=O(n^2)$ hidden layer width to guarantee convergence—a polynomial improvement over the $m=O(n^6)$ required for classic ReLU two-layer networks.

## Background & Motivation

**Background**: Kolmogorov-Arnold Networks (KANs), which place learnable univariate spline functions on edges while nodes perform only summation, are considered more interpretable and parameter-efficient than MLPs. Recent empirical results in vision, time-series, and PDE solving tasks are promising. However, the theoretical understanding of "why they can be trained effectively via gradient descent" remains largely unexplored.

**Limitations of Prior Work**: The training dynamics of overparameterized networks are well-studied within the Neural Tangent Kernel (NTK) framework. For two-layer ReLU networks, Du et al. (2019) proved that if the width is polynomially larger than the number of samples $n$, gradient descent converges to the global optimum. However, this analysis does not directly apply to KANs: their edge functions are linear combinations of basis functions (e.g., RBF, polynomials), resulting in a different NTK form. Previous closed-form expressions and width requirements were unquantified. Gao & Tan (2025) proved convergence for KANs with "all layers trained," but the width requirement is high ($\tilde O(g^9 n^3/\lambda_0^4)$) and the dependence on the minimum eigenvalue $\lambda_0$ is poor ($\lambda_0^{-4}$).

**Key Challenge**: Clarifying training dynamics requires a "stable and analyzable" tangent kernel. Training both layers simultaneously complicates the analysis and increases the cost of width and stability. Can a more restricted yet cleaner setting provide tighter width bounds and more robust guarantees?

**Key Insight**: The authors adopt a classic simplification from standard neural network analysis—**training only the first layer while fixing the second layer with random initialization** (as used by Du et al. 2019 for MLPs). In this setting, the tangent kernel depends only on the derivatives of the first-layer coefficients, making the dynamics closer to kernel regression, which allows for closed-form NTK derivation and precise analysis.

**Core Idea**: Under the "first-layer training + overparameterization" regime, the authors establish the **KAN Tangent Kernel (KAN-TK)**. They use it to prove global convergence, provide label-dependent convergence rates, and quantify the parameter efficiency advantage of KANs ($O(n^2)$ vs. $O(n^6)$ width compared to MLPs).

## Method

### Overall Architecture

This is a theoretical analysis focusing on the following two-layer KAN: input $x\in[0,1]^d$, hidden layer width $m$, and scalar output:

$$f(x)=\frac{1}{\sqrt m}\sum_{p=1}^{m}\sum_{l=1}^{g}\beta_{pl}\,\phi_l(z_p),\qquad z_p=\sum_{k=1}^{d}\sum_{j=1}^{g}\alpha_{pjk}\,\phi_j(x_k)$$

where $\{\phi_j\}_{j=1}^g$ are $g$ basis functions (RBFs are used for analysis), $\alpha_{pjk}$ are learnable first-layer coefficients, $\beta_{pl}$ are second-layer coefficients, and $\tfrac{1}{\sqrt m}$ is the standard scaling for overparameterization. Training protocol: $\alpha$ is initialized from $\mathcal N(0,\sigma^2)$ and updated via full-batch gradient descent; $\beta$ is initialized uniformly from $\{-1,+1\}$ and **frozen throughout training**; the loss is MSE $L=\tfrac12\|y-u\|_2^2$.

The logical chain is: **define and derive the KAN Tangent Kernel** $H_{ij}=\langle\nabla_\alpha f(x_i),\nabla_\alpha f(x_j)\rangle$ in the infinite-width limit (for RBF basis), **prove kernel stability** at sufficient width (lazy training), derive **linear convergence of the loss to zero** from stability, **refine convergence rates** via the kernel spectrum to obtain label-dependent bounds, and finally **compare** width, parameters, and learning rates with baselines to quantify KAN advantages.

### Key Designs

**1. Closed-form derivation of KAN-TK: Defining the metric for dynamics**

NTK theory posits that in the lazy training regime, network output can be approximated by a first-order Taylor expansion near initialization, reducing complex non-linear dynamics to kernel regression. For this KAN, where only $\alpha$ is trained, the kernel is the inner product of gradients w.r.t. $\alpha$. For 1D inputs and RBF bases $\phi_j(x)=\exp\!\big(-\tfrac{(x-\mu_j)^2}{2\sigma^2}\big)$, the authors derive the closed-form $H^\infty$ in the infinite-width limit $m\to\infty$ (Proposition 3.1). While mathematically complex, its value lies in enabling the calculation of **eigenvalues and eigenvectors of the KAN-TK**, which is essential for analyzing label alignment and convergence behavior.

**2. First-layer training + lazy training for global convergence: Tighter bounds through cleaner settings**

Under three standard assumptions (bounded basis derivatives, $\phi_l(0)=0$; positive definite infinite-width kernel $\lambda_0=\lambda_{\min}(H^\infty)>0$; bounded labels $|y_i|\le1$), Theorem 4.2 proves that given sufficient width and small initialization variance:

$$m\gtrsim\max\!\Big(\frac{d^2 g^6 n^2}{\lambda_0^2}\log\frac n\delta,\;n\Big),\qquad \sigma=O\!\Big(\frac{\delta}{\sqrt{mng^3 d}}\Big),$$

then with probability at least $1-O(\delta)$, gradient descent satisfies linear convergence $L(t+1)\le\big(1-\tfrac{\eta\lambda_0}{2}\big)L(t)$ for a learning rate $\eta=O\!\big(\tfrac{\lambda_0}{n^3 d^2 g^6}\big)$. **First-layer training** is the critical simplification, keeping weights almost static and the kernel nearly constant, thereby reducing width requirements to $O(n^2)$ and improving the dependence on $\lambda_0$ from $\lambda_0^{-4}$ to $\lambda_0^{-2}$.

**3. Label-dependent convergence rates: Eigenvector alignment**

Theorem 4.6 refines the rate along the KAN-TK spectrum: let $H^\infty=\sum_{i=1}^n\lambda_i v_i v_i^\top$, then the error vector satisfies:

$$\|y-u(t)\|_2\le\sqrt{\sum_{i=1}^{n}(1-\eta\lambda_i)^{2t}(v_i^\top y)^2}\;\pm\;\epsilon,$$

where $\epsilon \to 0$ as $m\to\infty$. Error decays fastest in the directions of eigenvectors corresponding to **large eigenvalues** $\lambda_i$. If the label $y$ projects strongly onto these top eigenvectors, overall convergence is significantly faster than with random labels.

**4. Efficiency comparison: KAN vs. MLP advantages and costs**

The paper compares the proposed method with two baselines: classic two-layer ReLU NN (Du et al. 2019) and all-layer trained KAN (Gao & Tan 2025). Regarding width, ReLU NN requires $O(n^6)$, and all-layer KAN requires $\tilde O(n^3)$, whereas the proposed first-layer training requires only $O(n^2)$. This is a polynomial improvement in $n$ and a reduction in $\lambda_0$ dependence from fourth-order to second-order. The mechanism is explained in Remark 2: KANs replace neuron-level activations with **smooth univariate splines**, where the NTK depends only on bounded derivatives and **pairwise interactions** between samples, leading to the $O(n^2)$ width; in contrast, ReLU networks must control **high-order interactions** to maintain discrete activation stability, requiring $O(n^6)$. The trade-off is a smaller learning rate $\eta$, leading to **slower single-step convergence**.

## Key Experimental Results

Experiments use a FastKAN-based two-layer RBF-KAN with first-layer training and full-batch GD.

### Main Results: Effect of Width on Convergence (Validating Thm 4.2)

| Setting | Observation | Theoretical Correspondence |
|------|------|---------|
| $n=100, d=100$, labels $\sim\mathcal N(0,1)$, $m\in\{500,\dots,32000\}$ | Greater $m$ leads to faster training error decay | Thm 4.2: Wider networks converge faster |
| Same as above, tracking $\|\alpha(t)-\alpha(0)\|_\infty$ | Greater $m$ results in smaller weight shift | Lemma 4.3: Lazy training behavior |

### Ablation Study: Effect of Label Structure on Convergence (Validating Thm 4.6)

| Label Type | Spectral Projection Feature | Convergence Speed |
|----------|-----------|---------|
| Structured ($y=\sin^2(0.7x/2)/\sin^2(x/2)$) | Energy concentrated on top eigenvectors | Fastest |
| Random (i.i.d. $\mathcal N(0,1)$) | Energy distributed uniformly across spectrum | Moderate |
| Anti-structured (Eigenvector of $\lambda_{\min}(H^\infty)$) | Concentrated on smallest eigenvalue direction | Slowest |

### Key Findings
- **Lazy training confirmed**: Wider networks converge faster while weights shift less, supporting the "static weights, static kernel" assumption.
- **Convergence dominated by label-spectrum alignment**: The progression from structured to random to anti-structured tasks demonstrates that error decays fastest in large eigenvalue directions.
- **Positive definiteness holds in practice**: The minimum eigenvalue of $H^\infty$ was observed to be strictly positive across various distributions, supporting the $\lambda_0>0$ assumption.

## Highlights & Insights
- **Theoretically grounding parameter efficiency**: The "KAN is more parameter-efficient" intuition is formalized as an $O(n^2)$ vs. $O(n^6)$ width requirement comparison. 
- **KAN-TK as a reusable tool**: The closed-form derivation under RBF basis provides a foundation for future analytical work on KAN dynamics and kernel regression.
- **Robust $\lambda_0$ dependence**: Reducing dependence from $\lambda_0^{-4}$ to $\lambda_0^{-2}$ is significant when $\lambda_0$ is small ($10^{-4}$ range), offering a more stable theoretical framework.

## Limitations & Future Work
- **Restricted setting**: The analysis is limited to first-layer training, frozen second-layer, and RBF bases; results may not generalize immediately to deep KANs or B-splines.
- **Trade-off between efficiency and speed**: While KAN is width-efficient, the smaller learning rate implies slower per-step convergence.
- **Computational scalability of NTK**: The closed-form KAN-TK scales polynomially with $n$, making it suitable for theoretical validation but not for large-scale practical computation.
- **Conservative baseline**: THE $O(n^2)$ advantage is compared to classic $O(n^6)$ bounds; using more advanced tools (like Clarke subdifferentials) might narrow the gap between KAN and refined ReLU bounds.

## Related Work & Insights
- **vs. Du et al. (2019) (Two-layer ReLU NN)**: Using similar NTK stability analysis, this work improves the width requirement from $O(n^6)$ to $O(n^2)$ for KANs.
- **vs. Gao & Tan (2025) (All-layer KAN)**: By restricting training to the first layer, this paper reduces width from $O(n^3)$ to $O(n^2)$ and improves $\lambda_0$ sensitivity, trading off per-step convergence speed.

## Rating
- Novelty: ⭐⭐⭐⭐ First closed-form NTK and global convergence proof for first-layer KAN training.
- Experimental Thoroughness: ⭐⭐⭐ Focused on theoretical validation with synthetic data rather than real-world tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain and insightful Remarks on mechanisms.
- Value: ⭐⭐⭐⭐ Establishes a solid theoretical foundation for KAN training efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Two-Layer Convolutional Autoencoders Trained on Normal Data Provably Detect Unseen Anomalies](two-layer_convolutional_autoencoders_trained_on_normal_data_provably_detect_unse.md)
- [\[ICML 2026\] Two-Layer Linear Auto-Regressive Models Estimate Latent States](../../ICML2026/learning_theory/two-layer_linear_auto-regressive_models_estimate_latent_states.md)
- [\[ICLR 2026\] High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification](high-dimensional_analysis_of_single-layer_attention_for_sparse-token_classificat.md)
- [\[ICLR 2026\] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks](tractability_via_low_dimensionality_the_parameterized_complexity_of_training_qua.md)
- [\[ICLR 2026\] Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape](saddle-to-saddle_dynamics_in_deep_relu_networks_low-rank_bias_in_the_first_saddl.md)

</div>

<!-- RELATED:END -->
