---
title: >-
  [Paper Note] Sequential Group Composition: A Window into the Mechanics of Deep Learning
description: >-
  [ICML 2026][Others][Alternating Gradient Flow] The authors use the unified task of "calculating the cumulative product of a sequence of group elements" as a microscope. Using Fourier analysis on groups and the AGF framework, they prove that two-layer networks learn irreducible representations (irreps) sequentially according to their Fourier energy. They further cha
tags:
  - ICML 2026
  - Others
  - Alternating Gradient Flow
date: 2026-05-08
content_hash: a34f050dbcca03bc
---
# Sequential Group Composition: A Window into the Mechanics of Deep Learning

**Conference**: ICML 2026  
**arXiv**: [2602.03655](https://arxiv.org/abs/2602.03655)  
**Code**: None  
**Area**: Mechanistic Interpretability / Learning Dynamics  
**Keywords**: Group composition, Irreducible representations, Fourier analysis on groups, Alternating Gradient Flow, Architecture expressivity

## TL;DR
The authors use the unified task of "calculating the cumulative product of a sequence of group elements" as a microscope. Using Fourier analysis on groups and the AGF framework, they prove that two-layer networks learn irreducible representations (irreps) sequentially according to their Fourier energy. They further characterize the expressivity gap across sequence length $k$, showing requirements of $2^k$ width for two-layer networks, $k$ steps for RNNs, and $\log k$ layers for deep MLPs.

## Background & Motivation

**Background**: Recent work in mechanistic interpretability has focused on small algebraic tasks like "modular addition," observing that networks spontaneously learn Fourier features (implementing addition via trigonometric identities). In learning dynamics, loss curves often exhibit "staircase" patterns, corresponding to the network progressively learning increasingly complex features. Both lines of research lack a unified, analytically derivable testbed.

**Limitations of Prior Work**: Modular addition is a specific case of cyclic groups with $k=2$, making it difficult to generalize findings to non-Abelian groups, sequence lengths $k>2$, or comparisons between architectures. Existing analyses are often "observe first, name later," lacking the ability to derive the emergence order of features from first principles.

**Key Challenge**: Does the network memorize the composition operation as a whole, or does it truly discover the algebraic structure of the group? If the latter, which representation does it learn, in what order, and how does the efficiency differ across architectures? These questions cannot be disentangled within the modular addition framework.

**Goal**: (i) Provide a unified task covering any finite group (Abelian/non-Abelian) and arbitrary sequence length $k$; (ii) Provide a rigorous theorem for the feature learning order in two-layer networks; (iii) Quantify the width/depth disparities for two-layer, RNN, and deep MLP architectures in expressing such compositions.

**Key Insight**: Treat "learning cumulative products" as a regression problem $(g_1,\dots,g_k)\mapsto\prod_i g_i$. By using the regular representation and group Fourier transform, the task can be block-diagonalized into irrep bases, allowing the gradient descent behavior within each irrep subspace to be solved analytically.

**Core Idea**: Training dynamics = greedy activation of irreps according to an "importance ruler" defined by $\|\hat{x}[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$. Depth leverages associativity to compress "calculating $k$ items at once" into "pairwise compositions," reducing width from $2^k$ down to $k$ or $\log k$.

## Method

### Overall Architecture

Task Definition: Fix a finite group $G$ and an encoding vector $x\in\mathbb{R}^{|G|}$. Each group element $g$ is encoded via the regular representation as $x_g=\lambda(g)^\top x$ (degenerating to one-hot when $x=e_1$). The network $f:\mathbb{R}^{k|G|}\to\mathbb{R}^{|G|}$ receives a sequence $x_{\mathbf g}=(x_{g_1},\dots,x_{g_k})$ and outputs a regression estimate of the product $x_{g_1\cdots g_k}$, trained with MSE. Lemma 3.5 proves that if $x$ is non-trivial and zero-mean, no linear map can solve this task—non-linear interaction is required. The simplest instance is a two-layer network $f=W_\text{out}\sigma(W_\text{in}x_{\mathbf g})$ with $\sigma(z)=z^k$ (monic polynomial), trained under vanishing initialization ($\theta_i(0)\sim\mathcal N(0,\alpha^2)$, $\alpha\to0$) using neuron-specific learning rates $\eta_{\theta_i}=\|\theta_i\|^{1-k}\log(1/\alpha)$.

The analysis framework is based on the Alternating Gradient Flow (AGF) by Kunin et al. 2025: under vanishing initialization, each hidden neuron switches between a "dormant" state ($\|\theta_i\|\approx 0$, no impact on output) and an "active" state ($\|\theta_i\|\gg 0$, dominating the fit). Training is step-like—plateaus correspond to dormant neurons competing to maximize utility, while descents correspond to active neurons collectively minimizing the residual. This paper applies AGF to the group Fourier domain, where each activation corresponds to the "emergence" of an irrep.

### Key Designs

**1. Sigma-pi-sigma Decomposition + Utility Maximization Theorem: Closed-form Ranking for the Next Feature**

The interpretability field has relied on probes to "observe then name" Fourier features, but cannot explain why or in what order they appear. This work analytizes this through the polynomial output of neurons: each neuron $f(x_{\mathbf g};\theta_i)$ decomposes into a critical "full interaction term" $f^{(\times)}=w_i\cdot k!\cdot\prod_j\langle u_{i,j},x_{g_j}\rangle$ and irrelevant "additive terms" $f^{(+)}$. Utility (the inner product of dormant neurons and the residual) is contributed solely by $f^{(\times)}$. In the group Fourier domain, utility becomes a tensor product of input/output weights and $\hat x[\rho]$. Under the $\|\theta\|=1$ constraint, the optimal solution concentrates all Fourier energy into a single irrep $\rho_*$, selected by a computable "importance ruler":

$$\rho_*=\arg\max_{\rho\notin\mathcal I^{t-1}}\frac{\|\hat x[\rho]\|_\text{op}^{k+1}}{(C_\rho n_\rho)^{(k-1)/2}}\quad(C_\rho=1\ \text{if}\ \rho\ \text{is real, otherwise}\ 2)$$

Theorem 4.1 rigorously proves that networks greedily learn irreps by Fourier importance, with sequence length $k$ explicitly in the exponent—larger $k$ amplifies the advantage of high-energy irreps ($k+1$ vs $(k-1)/2$), leading to sharper steps for longer sequences.

**2. Alignment and Residual Update under Cost Minimization: One-to-One Correspondence between Steps and Irreps**

Utility maximization only explains "who shows up first." The second step of AGF explains what happens after activation. Once $N$ neurons activate and align to the same $\rho_*$, they jointly minimize the loss within the constrained subspace of $\rho_*$, effectively "canceling out" the target $x_{g_1\cdots g_k}$'s Fourier component on $\rho_*$. Representing the active neurons' output in the Fourier domain:

$$f(x_{\mathbf g};\Theta_\mathcal A)[h]=\frac{1}{|G|}\sum_{\rho\in\mathcal I^{t-1}}\langle\rho(g_1\cdots g_k h)^\dagger,\hat x[\rho]\rangle_\rho$$

Each AGF round merges the next $\rho_*$ into the learned set $\mathcal I^{t-1}$ (maintaining conjugate closure) until $\mathcal I^{t-1}=\mathcal I(G)$, at which point the network fits the task perfectly. This "order + update" mechanism allows every step in the loss curve to be pre-mapped to a specific irrep, making the learning path predictable for any finite group (including non-Abelian $D_3$). Furthermore, the order can be rewritten by spectral shaping of the encoding $x$.

**3. Architecture → Expressivity: Three Scaling Regimes for Associativity**

The same cumulative product task requires vastly different capacities across architectures. This work provides three comparable scalings. To fit all $k$-ary products at once, a two-layer network needs enough neurons per Fourier mode to cancel out interference terms $f^{(+)}$, leading to a hidden width explosion of order $2^k$. Note that the bottleneck is not "learning the irrep" but "the redundancy cost of canceling $f^{(+)}$." RNNs use associativity to write the product as $k$ sequential updates, requiring only a fixed-size network per step. Deep MLPs merge pairs in a binary tree, requiring only $\log_2 k$ layers. Thus, the heuristic claim that "deep/recurrent structures are more parameter-efficient for algorithmic tasks" is formalized as an analytical statement: "2-layer $2^k$, RNN $k$, deep $\log k$."

### Loss & Training
Regression loss $\mathcal L(\Theta)=\tfrac{1}{2|G|^k}\sum_{\mathbf g\in G^k}\|x_{g_1\cdots g_k}-f(x_{\mathbf g};\Theta)\|^2$; vanishing initialization $\alpha\to 0$; adaptive learning rate $\eta_{\theta_i}\propto\|\theta_i\|^{1-k}$. Assumptions on encoding $x$: zero-mean, $\hat x[\rho]$ is either 0 or invertible, and utility criterion values for different $\rho$ are distinct (ensuring clear steps).

## Key Experimental Results

### Main Results

| Setting | Group / Architecture | Phenomenon | Key Observation |
|------|----------|------|---------|
| Binary Composition $k=2$ | Abelian $C_p$ / 2-layer quadratic MLP | Loss shows steps; each plateau corresponds to a $\rho$ activation | Activation order matches Theorem 4.1 exactly |
| Binary Composition $k=2$ | Non-Abelian $D_3$ / 2-layer quadratic MLP | 1D and 2D irreps learned sequentially | Validates generalization to non-Abelian groups |
| Sequence Length $k$ sweep | Fixed $G$, vary $k=2,3,4,\dots$ | 2-layer width requirement grows exponentially | Width for perfect fit scales as $2^k$ |
| Architecture Comparison | 2-layer vs RNN vs Deep MLP | RNN uses $k$ steps; Deep MLP uses $\log k$ layers | Associativity allows depth to substitute width |

### Ablation Study

| Configuration / Comparison | Result | Note |
|------------|------|------|
| Encoding $x=e_1$ (one-hot) | Default baseline | All irreps equally weighted |
| Spectral shaping of $x$ | Learning order reordered by criterion | Validates $\rho_*$ selection depends on $\|\hat x[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$ |
| Polynomial vs Smooth activation | Similar behavior after Taylor expansion | Appendix discusses how non-polynomial activations inherit same interaction terms |
| Single vs Multiple neurons $N$ per $\rho_*$ | Multiple neurons required to cancel $f^{(+)}$ | Explains why redundant width is necessary |

### Key Findings
- The learning order is **entirely determined** by the Fourier energy spectrum of the encoding $x$—the first analytical formula for the "emergence order of Fourier circuits" in interpretability.
- Sequence length $k$ amplifies the advantage of the primary irrep at the $k+1$ exponent, meaning longer sequences $\to$ more "greedy" behavior and sharper steps.
- The root cause of the $2^k$ width explosion for two-layer networks is not the inability to learn irreps, but the requirement for redundant neurons to cancel $f^{(+)}$ interference; deep architectures bypass this via associativity.

## Highlights & Insights
- Combines group Fourier analysis and AGF into a tool that strictly predicts when and how each irrep is selected, elevating mechanistic interpretability from "post-hoc naming" to "a priori prediction."
- The criterion $\|\hat x[\rho]\|_\text{op}^{k+1}/(C_\rho n_\rho)^{(k-1)/2}$ shows $k$ explicitly affecting feature ranking, a fact invisible in $k=2$ modular addition literature, suggesting that stronger simplicity biases in long-sequence tasks are algebraic facts of the task itself.
- "Associativity = depth advantage" is precisely quantified: 2-layer $2^k$, RNN $k$, deep $\log k$, providing non-heuristic evidence for parameter efficiency in deep models for algorithmic tasks (e.g., Rubik's Cube, path integrals).
- Spectral shaping of encoding $x$ provides a new "knob": controlling what a network learns first or last by designing the Fourier shape of inputs, which has implications for curriculum learning and pedagogical design.

## Limitations & Future Work
- Theoretical rigor is focused on 2-layer quadratic networks with vanishing initialization; generalizations to ReLU networks, Adam optimizers, and finite initialization are discussed only qualitatively.
- Experiments are primarily on small finite groups ($C_p$, $D_3$, etc.), and have not been validated for Fourier circuit emergence in large-scale models or real tasks (e.g., arithmetic LLMs).
- The AGF assumption of "one feature at a time" depends on criterion values being "well-separated"; when $G$ is large or the spectrum is near-uniform, steps become blurred, reducing the utility of the theoretical characterization.
- The $\log k$ scaling for deep networks is an existence proof and does not guarantee that SGD will find this solution in practice; there remains a gap between theory and whether Transformers truly use $\log k$ layers for composition.

## Related Work & Insights
- **vs Nanda et al. 2023 (grokking modular addition)**: That work empirically discovered Fourier addition circuits; this work provides a closed-form theorem for "why, when, and in what order" such circuits appear, generalizing to non-Abelian groups and any $k$.
- **vs Chughtai et al. 2023 / Stander et al. 2023 (small group composition)**: They observed similar Fourier structures in binary compositions; this work provides the analytical formula for the emergence order (Theorem 4.1) and extends analysis to sequential tasks.
- **vs Kunin et al. 2025 AGF**: This work is the first application of the AGF framework to the "group Fourier domain"—writing out the utility-cost steps explicitly within representation theory and providing a specific form via sigma-pi-sigma decomposition.
- **vs Liu et al. 2022 / Sanford et al. 2023-2024 (Transformer vs RNN expressivity)**: That line of work discusses Transformers outperforming RNNs on algorithmic tasks with $\log$ depth; this work quantifies the same comparison as "2-layer $2^k$ / RNN $k$ / Deep $\log k$" in the context of group composition, anchoring expressivity gaps to specific constants.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Seamlessly integrates representation theory, AGF, and expressivity into an analytically tractable testbed; a rare "prove then observe" work in mechanistic interpretability.
- Experimental Thoroughness: ⭐⭐⭐ Solid verification on small groups, but lacks transfer validation on large models or real-world arithmetic tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptional clarity in mathematical statements with a logical hierarchy of lemmas and theorems; appendix covers proofs and background thoroughly.
- Value: ⭐⭐⭐⭐⭐ Establishes a benchmark task for "algorithmic interpretability," likely to influence future work on complex group structures and realistic architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)
- [\[ICLR 2026\] CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning](../../ICLR2026/others/chlu_the_causal_hamiltonian_learning_unit_as_a_symplectic_primitive_for_deep_lea.md)
- [\[CVPR 2026\] Towards Knowledge-augmented Bayesian Deep Learning For Computer Vision](../../CVPR2026/others/towards_knowledge-augmented_bayesian_deep_learning_for_computer_vision.md)

</div>

<!-- RELATED:END -->
