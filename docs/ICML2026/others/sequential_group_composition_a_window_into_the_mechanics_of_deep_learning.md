---
title: >-
  [Paper Note] Sequential Group Composition: A Window into the Mechanics of Deep Learning
description: >-
  [ICML 2026][Group Composition] The authors utilize the task of computing the "cumulative product of a sequence of group elements" as a unified lens. Using Fourier analysis on groups and the AGF framework…
tags:
  - "ICML 2026"
  - "Group Composition"
  - "Irreducible Representations"
  - "Fourier Analysis on Groups"
  - "Alternating Gradient Flow"
  - "Architectural Expressivity"
date: 2026-05-08
content_hash: ce96286bf35fb9b0
---

# Sequential Group Composition: A Window into the Mechanics of Deep Learning

**Conference**: ICML 2026  
**arXiv**: [2602.03655](https://arxiv.org/abs/2602.03655)  
**Code**: None  
**Area**: Mechanistic Interpretability / Learning Dynamics  
**Keywords**: Group Composition, Irreducible Representations, Fourier Analysis on Groups, Alternating Gradient Flow, Architectural Expressivity

## TL;DR
The authors utilize the task of computing the "cumulative product of a sequence of group elements" as a unified lens. Using Fourier analysis on groups and the AGF framework, they prove that two-layer networks learn irreducible representations (irreps) sequentially according to their Fourier energy. They further characterize the expressivity gaps where two-layer networks, RNNs, and deep MLPs require $2^k$ width, $k$ steps, and $\log k$ layers respectively for a sequence of length $k$.

## Background & Motivation

**Background**: Mechanistic interpretability has recently focused on "modular addition" in small algebraic tasks, observing that networks spontaneously learn Fourier features. Learning dynamics research indicates that loss curves often exhibit a "staircase" pattern, corresponding to the sequential learning of increasingly complex features. However, a unified experimental platform for analytical derivation is missing.

**Limitations of Prior Work**: Modular addition is a specific case of cyclic groups where $k=2$. These findings are difficult to generalize to non-Abelian groups, longer sequences ($k>2$), or comparisons between architectures. Current analyses are often post-hoc observations, lacking the ability to derive feature emergence order from first principles.

**Key Challenge**: Does the network memorize compositions as a whole, or does it discover the underlying algebraic structure? If it is the latter, which representation does it learn, in what order, and what are the efficiency differences between architectures? These questions cannot be isolated within the modular addition framework.

**Goal**: (i) Provide a unified task covering any finite group and arbitrary sequence length $k$; (ii) Derive a rigorous theorem for the feature learning order in two-layer networks; (iii) Quantify the width/depth efficiency gaps between two-layer networks, RNNs, and deep MLPs.

**Key Insight**: Treat "learning cumulative products" as a regression problem $(g_1,\dots,g_k)\mapsto\prod_i g_i$. By leveraging regular representations and group Fourier transforms, the task can be block-diagonalized into irrep subspaces. This allows the behavior of gradient descent to be analytically solved within each subspace.

**Core Idea**: Training dynamics follow a "greedy" activation of irreps along a Fourier spectrum based on an "importance ruler" defined by $\|\hat{x}[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$. Furthermore, depth utilizes the associative property to compress $k$-item products into pairwise combinations, reducing width requirements from $2^k$ down to $k$ or $\log k$.

## Method

### Overall Architecture

The task is defined by a finite group $G$ and an encoding vector $x\in\mathbb{R}^{|G|}$. Each element $g$ is encoded via a regular representation $x_g=\lambda(g)^\top x$ (reducing to one-hot if $x=e_1$). The network $f:\mathbb{R}^{k|G|}\to\mathbb{R}^{|G|}$ receives a sequence $x_{\mathbf g}=(x_{g_1},\dots,x_{g_k})$ and outputs a regression estimate for the product $x_{g_1\cdots g_k}$ using MSE. Lemma 3.5 proves that if $x$ is non-trivial and zero-mean, no linear map can solve this task, necessitating non-linear interactions. The simplest instance is a two-layer network $f=W_\text{out}\sigma(W_\text{in}x_{\mathbf g})$ with $\sigma(z)=z^k$. Under vanishing initialization ($\theta_i(0)\sim\mathcal N(0,\alpha^2)$, $\alpha\to0$) and a neuron-dependent learning rate $\eta_{\theta_i}=\|\theta_i\|^{1-k}\log(1/\alpha)$, the training follows the Alternating Gradient Flow (AGF).

The analysis framework, based on Kunin et al. 2025, shows that neurons switch between a "dormant" state and an "activated" state. The training process consists of plateaus (dormant neurons competing to maximize utility) and drops (activated neurons minimizing residuals). This work applies AGF to the Fourier domain of groups, mapping each activation to the emergence of an irrep.

### Key Designs

1.  **Sigma-pi-sigma Decomposition + Utility Maximization Theorem**:
    - **Function**: Decomposes the polynomial output of each neuron $f(x_{\mathbf g};\theta_i)$ into a "full interaction term" $f^{(\times)}=w_i\cdot k!\cdot\prod_j\langle u_{i,j},x_{g_j}\rangle$ and redundant "additive terms" $f^{(+)}$. It proves that utility (the inner product of dormant neurons and residuals) is contributed solely by $f^{(\times)}$.
    - **Mechanism**: By writing $f^{(\times)}$ in the Fourier domain, utility is shown to be a tensor product of Fourier coefficients and $\hat x[\rho]$. Maximizing under $\|\theta\|=1$, the optimal solution concentrates all Fourier energy onto a single irrep $\rho_*$ selected by the criterion $\rho_*=\arg\max_{\rho\notin\mathcal I^{t-1}}\|\hat x[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$ (Theorem 4.1).
    - **Design Motivation**: Converts the empirical observation of "which feature is learned next" into a computable closed-form ranking for the first time.

2.  **Alignment and Residual Updates under Cost Minimization**:
    - **Function**: Once $N$ neurons activate and align with $\rho_*$, they jointly minimize the loss within the $\rho_*$ subspace, effectively eliminating the Fourier component of the target.
    - **Mechanism**: The output of activated neurons is represented in the Fourier domain as $f(x_{\mathbf g};\Theta_\mathcal A)[h]=\tfrac{1}{|G|}\sum_{\rho\in\mathcal I^{t-1}}\langle\rho(g_1\cdots g_k h)^\dagger,\hat x[\rho]\rangle_\rho$. After each AGF step, the learned set $\mathcal I^{t-1}$ incorporates the next $\rho_*$ until all irreps are learned and the task is perfectly fitted.
    - **Design Motivation**: This link between the "emergence order" and "residual updates" allows the learning path of any finite group to be predicted in advance.

3.  **Architecture-Expressivity Scaling**:
    - **Function**: Compares the capacity and depth required for two-layer MLPs, RNNs, and deep feedforward MLPs.
    - **Mechanism**: A two-layer network must fit all $k$-ary products simultaneously, requiring enough neurons to cancel $f^{(+)}$ terms, leading to $2^k$ width scaling. RNNs leverage associativity to compute cumulative products serially in $k$ steps with fixed-size networks. Deep MLPs merge sequences in pairs (binary tree), requiring only $\log_2 k$ layers.
    - **Design Motivation**: Transforms the empirical claim of "why transformers/deep structures are efficient" into an analytical statement regarding group composition.

### Loss & Training
The regression loss is $\mathcal L(\Theta)=\tfrac{1}{2|G|^k}\sum_{\mathbf g\in G^k}\|x_{g_1\cdots g_k}-f(x_{\mathbf g};\Theta)\|^2$. It uses vanishing initialization $\alpha\to 0$ and adaptive learning rates $\eta_{\theta_i}\propto\|\theta_i\|^{1-k}$. Assumptions for encoding $x$ include zero-mean, invertibility of $\hat x[\rho]$, and separation of utility criteria.

## Key Experimental Results

### Main Results

| Setting | Group / Architecture | Phenomenon | Key Observation |
|:---|:---|:---|:---|
| Binary $k=2$ | Abelian $C_p$ / 2-layer MLP | Staircase loss; plateaus correspond to $\rho$ activation | Activation order matches Theorem 4.1 exactly |
| Binary $k=2$ | Non-Abelian $D_3$ / 2-layer MLP | 1D and 2D irreps learned sequentially | Теоретическая generalizability to non-Abelian groups |
| Sequence $k$ Scan | Fixed $G$, vary $k=2,3,4,\dots$ | Exponential width growth for 2-layer | Required width scales as $2^k$ |
| Arch. Comparison | 2-layer vs RNN vs Deep MLP | RNN learns in $k$ steps; Deep MLP in $\log k$ | Associativity enables depth-width trade-off |

### Ablation Study

| Configuration / Contrast | Result | Description |
|:---|:---|:---|
| Encoding $x=e_1$ (one-hot) | Default baseline | All irreps weighted equally |
| Spectrum shaping of $x$ | Learning order re-ranks | Confirms $\rho_*$ selection depends on the criterion |
| Poly vs Smooth Activation | Similar behavior after Taylor expansion | Smooth activations inherit interaction terms via Taylor expansion |
| Single vs Multiple $N$ Alignment | Multiple neurons required to cancel $f^{(+)}$ | Explains the necessity of redundant width |

### Key Findings
- The learning order is **fully determined** by the Fourier energy spectrum of $x$. This is the first analytical formula for the emergence of Fourier circuits.
- Sequence length $k$ amplifies the advantage of dominant irreps via the exponent $k+1$, leading to sharper "greedy" behavior in longer sequences.
- The exponential width requirement for two-layer networks stems from the need to cancel $f^{(+)}$ interference, a burden bypassed by deep structures via associativity.

## Highlights & Insights
- Combines group Fourier analysis and AGF to prescribe when and how irreps are selected, moving mechanistic interpretability from "post-hoc naming" to "a priori prediction."
- Theorem 4.1 shows that $k$ explicitly influences feature ranking; this effect is invisible in modular addition ($k=2$) literature, revealing that simplicity bias in long sequences is an algebraic fact.
- The "Associativity = Depth Advantage" intuition is quantified: $2^k$ for 2-layer, $k$ for RNN, and $\log k$ for deep architectures, providing non-heuristic evidence for parameter efficiency.
- Spectrum shaping of $x$ provides a new control knob to program the curriculum of a network by designing its Fourier profile.

## Limitations & Future Work
- The theoretical rigor is restricted to two-layer quadratic networks with vanishing initialization; generalizations to ReLU or Adam are qualitative.
- Experiments are conducted on small finite groups ($C_p, D_3$); validation on large-scale models or realistic arithmetic LLMs is missing.
- The AGF assumption of "one feature at a time" depends on the separation of criteria, which may blur if the spectrum is near-uniform or $G$ is extremely large.
- The $\log k$ scaling is an existence proof; it does not guarantee that SGD will find this specific solution in practice.

## Related Work & Insights
- **vs Nanda et al. 2023 (grokking modular addition)**: While Nanda observed Fourier circuits empirically, this work provides a closed-form theorem for why, when, and in what order these circuits emerge.
- **vs Chughtai / Stander et al. 2023**: Expands upon observations in binary compositions by providing analytical formulas (Theorem 4.1) and extending the analysis to sequential tasks.
- **vs Kunin et al. 2025 (AGF)**: This work is the first application of the AGF framework to the group Fourier domain, providing concrete sigma-pi-sigma forms for AGF's abstract utility-cost steps.
- **vs Liu / Sanford et al. (Expressivity)**: Quantifies the "Transformer vs RNN" debate within the group composition task as a three-tier scaling gap: $2^k$ vs $k$ vs $\log k$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Synthesizes group theory, AGF, and expressivity into a unique analytical testbed.
- Experimental Thoroughness: ⭐⭐⭐ Strong on small groups, but lacks validation on real-world large-scale tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical statements and well-structured proofs.
- Value: ⭐⭐⭐⭐⭐ Establishes a benchmark task for algorithmic interpretability with significant potential for future extensions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)
- [\[ICLR 2026\] CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](../../ICLR2026/others/chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)

</div>

<!-- RELATED:END -->
