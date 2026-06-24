---
title: >-
  [Paper Note] Why High-Rank Neural Networks Also Generalize?: An Algebraic Framework Based on RKHS
description: >-
  [ICLR 2026][Learning Theory][Generalization bounds] Ours uses Koopman operators, group representations, and Reproducing Kernel Hilbert Spaces (RKHS) to formulate deep networks as an algebraic "product of operators," deriving a new Rademacher complexity bound. The denominator features the determinant of the weight matrix $\det(W_l^*W_l)^{1/4}$, theoretically explaining the empirical phenomenon that "high-rank weight matrices with large singular values generalize well." This wo…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Generalization Theory"
  - "Generalization bounds"
  - "Rademacher complexity"
  - "Koopman operator"
  - "Reproducing Kernel Hilbert Space"
  - "Group representation"
date: 2026-05-08
content_hash: b82d9183051d0602
---

# Why High-Rank Neural Networks Also Generalize?: An Algebraic Framework Based on RKHS

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=nCE7Sli461](https://openreview.net/forum?id=nCE7Sli461)  
**Area**: Learning Theory / Generalization Theory  
**Keywords**: Generalization bounds, Rademacher complexity, Koopman operator, Reproducing Kernel Hilbert Space, Group representation

## TL;DR
Ours uses Koopman operators, group representations, and Reproducing Kernel Hilbert Spaces (RKHS) to formulate deep networks as an algebraic "product of operators," deriving a new Rademacher complexity bound. The denominator features the determinant of the weight matrix $\det(W_l^*W_l)^{1/4}$, theoretically explaining the empirical phenomenon that "high-rank weight matrices with large singular values generalize well." This work also extends Koopman theory to non-smooth activations (like tanh, sigmoid, Leaky ReLU) and bounded data spaces for the first time.

## Background & Motivation

**Background**: Understanding the generalization capability of deep networks is a core challenge in machine learning. Classical approaches involve controlling generalization error using VC dimension, norm-based bounds, or compression-based bounds. Norm-based bounds depend on $(p,q)$-norms of weight matrices, while compression-based bounds examine how much a network can be compressed.

**Limitations of Prior Work**: Both types of bounds suggest that "low-rank, small-singular-value (approximately low-rank) weight matrices are beneficial for generalization." However, experiments repeatedly observe a contradictory phenomenon—**networks with high-rank weight matrices and large singular values often generalize very well** (Goldblum et al., 2020). Norm and compression bounds focus solely on low-rank cases and fail to explain these phenomena.

**Key Challenge**: To explain "why high-rank generalizes," a bound is needed where the "determinant/product of singular values" appears in the denominator—the larger the singular values and the determinant, the smaller the bound. The Koopman bound proposed by Hashimoto et al. (2024) takes this form: the bound is proportional to $\prod_l G_l\|K_{\sigma_l}\|_{H_l}\|W_l\|^{s_l-1}/(\sqrt{S}\,\det(W_l^*W_l)^{1/4})$. However, this bound **strongly depends on model smoothness and unbounded data spaces**: it defines the Koopman operator norm in Sobolev space $H_l$, excluding non-smooth activations like tanh, sigmoid, or ReLU acting on bounded domains. Furthermore, factors like $\|K_{\sigma_l}\|_{H_l}$ and $G_l$ are extremely difficult to evaluate in most cases, leaving the exact impact of activation functions on complexity unclear.

**Goal**: Retain the advantage of the Koopman bound (determinant in the denominator to explain high-rank generalization) while extending it from idealized smooth, unbounded models to realistic models using tanh/sigmoid/Leaky ReLU and bounded data spaces, while ensuring every factor in the bound can be explicitly calculated.

**Key Insight**: Ours finds that by switching the workspace of Koopman operators from Sobolev space $H_l$ to the larger $L^2$ space $L_l$ ($H_l\subset L_l$), non-smooth activations and bounded domains can be accommodated. The "reproducibility" required for deriving generalization bounds is obtained by constructing an additional RKHS on the **parameter space**.

**Core Idea**: Algebrize deep networks into a "product of group representations/Koopman operators" $\rho(g_1)A_1\cdots A_{L-1}\rho(g_L)v$. By constructing an RKHS on the parameter space using a kernel function, the Rademacher complexity bound is controlled by a sequence of operator norms through reproducibility, resulting in a more broadly applicable and computable high-rank generalization bound.

## Method

### Overall Architecture

This paper provides a pure theoretical derivation: the input is an $L$-layer deep network (a sequence of weight matrices and activation functions), and the output is an upper bound on the Rademacher complexity of this function class (which is then converted to a generalization error bound via the theorem by Mohri et al.). The derivation chain follows four steps:

1. **Algebrization**: The "composition of linear transformations and non-linear activations" is unified as a product of operators $f(g_1,\dots,g_L)=\rho(g_1)A_1\rho(g_2)A_2\cdots A_{L-1}\rho(g_L)v$ on a Hilbert space $H$, where $\rho$ is a unitary representation of group $G$ (describing the action of weights), $A_l$ are Koopman operators corresponding to activation functions, and $v$ is the terminal non-linear transformation.
2. **Regularized Approximation**: Since the space $H$ (typically $L^2(\mathbb{R}^d)$) where the model resides lacks reproducibility, the model is first inner-producted with a kernel $p_{c,x}$ to obtain a regularized model $F_c(g,x)=\langle\rho(g_1)A_1\cdots A_{L-1}\rho(g_L)v,\,p_{c,x}\rangle$. As $c\to\infty$, $p_{c,x}$ tends toward a Dirac delta centered at $x$, and $F_c$ converges back to the original model $f$.
3. **Constructing RKHS on Parameter Space**: Define a kernel $k\big((g_1,\dots,g_L),(\tilde g_1,\dots,\tilde g_L)\big)=\langle\rho(g_1)A_1\cdots\rho(g_L)v,\,\rho(\tilde g_1)A_1\cdots\rho(\tilde g_L)v\rangle_H$. It spans an RKHS $R_k$ over the parameter space $G^L$, and an isometric isomorphism $\iota$ maps the model subspace $K\subseteq H$ to $R_k$ (Proposition 3.4). This allows the model on the data space to be treated as a function on the parameter space, utilizing the reproducibility of RKHS.
4. **Deriving the Bound**: Using reproducibility, the Rademacher complexity is expanded into a product of operator norms, leading to the main theorem $\hat R(F_c)\le \|A_1\|\cdots\|A_{L-1}\|\,\|v\|\,E(c)/\sqrt{S}$ (Theorem 4.1). When applied to specific networks (invertible, varying width, CNNs), $\det(W_l^*W_l)^{1/4}$ naturally emerges in the denominator, resulting in smaller bounds for high-rank weights.

### Key Designs

**1. Algebrizing networks as operator products, unifying linear layers and activations using group representations and Koopman operators**

To bring the determinant into the denominator, the network must first be written in an operator form where norms can be decomposed. Ours expresses an $L$-layer network as $f(g_1,\dots,g_L)=\rho(g_1)A_1\cdots A_{L-1}\rho(g_L)v$. Two types of linear operators are used: **group representations** $\rho:G\to B(H)$ describe the effect of weights on the model (e.g., using the affine group $G=GL(d)\ltimes\mathbb{R}^d$ and $\rho(g)h(x)=|\det W|^{1/2}h(W(x-b))$ corresponds to an invertible weighted network—note that $|\det W|^{1/2}$ is the source of the determinant in the denominator); **Koopman operators** $K_\sigma$ describe the composition of activation functions, defined as $K_\sigma h(x)=h(\sigma(x))$. This linearizes "non-linear composition" into a linear operator, which is the most critical property of Koopman operators for network analysis.

**2. Constructing RKHS on parameter space to control Rademacher complexity via operator norms**

The space $H$ (e.g., $L^2(\mathbb{R}^d)$) lacks reproducibility, preventing direct application of RKHS bounds. The core innovation of Ours is **constructing an RKHS on the parameter space instead**: using kernel $k$ to span $R_k$ and proving that $\iota:K\to R_k$ is an isometric isomorphism (Proposition 3.4) allows the model function to be "moved" from the data space to the parameter space. With reproducibility, the regularized model satisfies $F_c(\cdot,x)=\iota(p_{c,x})\in R_k$, allowing the Rademacher complexity to be decomposed layer-by-layer:

$$\hat R(F_c,x_1,\dots,x_S)\le\frac{\|A_1\|\cdots\|A_{L-1}\|\,\|v\|\,E(c)}{\sqrt{S}}.$$

Applying this to Example 3.1 for invertible networks, since $\rho$ is irreducible and $A_l$ is invertible, the bound simplifies to $\hat R(\overline{NN}_c)\le E(c)\|v\|\prod_l\|A_l\|/\sqrt{S}\cdot\sup \prod_l|\det W_l|^{-1/2}$. **Because $\det W_l$ is the product of all singular values and resides in the denominator**, the bound remains small even if $W_l$ has large singular values and high rank.

**3. Using $L^2$ space instead of Sobolev space for non-smooth activations and explicit factor evaluation**

Prior Koopman bounds defined operator norms in Sobolev space $H_l$, requiring smooth activations and unbounded data spaces. Ours measures operator norms in the larger $L^2$ space $L_l$ ($H_l\subset L_l$). This change provides two benefits: first, it **increases applicability** to non-smooth activations and bounded domains; second, it makes **factors explicitly computable**. Ours provides three lemmas to bound Koopman operator norms: Lemma 2.3 uses the Jacobian of $\sigma^{-1}$ to give $\|K_\sigma\|\le\sup_x|J_{\sigma^{-1}}(x)|^{1/2}$; Lemma 2.4 gives explicit bounds for tanh/sigmoid on bounded domains; Lemma 2.5 gives $\|K_\sigma\|\le\max\{1,1/a^d\}^{1/2}$ for Leaky ReLU. As noted in Remark 5.4, saturating activations (tanh/sigmoid) increase complexity by increasing $\|A_l\|$ when $\det W_l$ and volume $X_l$ are large, whereas unbounded activations (Leaky ReLU) increase the norm of the terminal transform $v$.

**4. Generalization to varying widths, non-injective weights, and CNNs using weighted Koopman operators**

While the initial assumption uses a single Hilbert space $H$ (constant width), Ours introduced layer-specific spaces $H_0,\dots,H_{L-1},\tilde H_1,\dots,\tilde H_L$. Weights $W_l$ are also represented as Koopman operators $\eta_l(W_l)=K_{W_l}$, yielding a bound for varying widths (Theorem 5.1). When $W$ is not injective, $K_W$ is unbounded; Ours instead uses **weighted Koopman operators** $\tilde K_{\psi,W}h(x)=\psi(x)h(\sigma(x))$, where the weight $\psi$ zeros out contributions in the kernel space directions (Theorem 5.5). Remark 5.6 provides an intuition: if $\ker(W_{l+1})$ aligns with noise directions and is filtered out by $W_{l+1}$, the factor $\mu_{\ker(W_{l+1})}(Y_l)$ is small and the model generalizes well, aligning with the noise stability conclusions of Arora et al. (2018). For CNNs, convolutions are expressed as linear operators diagonalized on Fourier components $\gamma_m(\theta)$ (Proposition 5.7).

## Key Experimental Results

As this is a theoretical work, numerical experiments are used to verify the validity of the bounds (Appendix B contains details):

### Main Results

| Experiment | Model / Data | Setting | Conclusion |
|------|------------|------|------|
| Bound Validity | Synthetic regression, $X_0=[-1,1]^3$, tanh, 2 layers, $S=1000$ | Target $t(x)=e^{-\|2x-1\|^2}$, added $0.1r$ regularization ($r$ proportional to Ours bound) | Generalization error **decreases proportionally** with Ours bound (Fig 4a). |
| Comparison | MNIST 4-layer dense, smooth Leaky ReLU + softmax, $S=1000$ | Ours regularization $0.01(r_1+r_2+r_3)$ vs Hashimoto 2024 regularization | Ours regularization achieves **higher test accuracy** (Fig 4b). |
| Practical Model (LeNet) | LeNet on MNIST, tanh + softmax | Ours regularization $0.1(r_1+r_2+r_3)$ vs no regularization | Adding regularization yields **higher accuracy**; prior bounds fail for tanh+softmax (Fig 4c). |

The three regularization terms correspond to factors in the bound: $r_1$ controls $\|A_l\|$, $r_2$ controls $1/\det(W_l^*W_l)^{1/2}$ (increasing the minimum singular value $s_{\min}(W_l)$), and $r_3$ controls $\|v\|$.

### Key Findings
- **The determinant in the denominator is key to explaining high-rank generalization**: The bound is proportional to $1/\prod_l\det(W_l^*W_l)^{1/4}$. Larger singular values result in larger determinants and smaller bounds, which complements the "low-rank preference" of norm/compression bounds.
- **The space transition ($H_l\to L_l$) is the core increment over prior work**: It extends the **applicability** of the bound from idealized models to realistic ones and makes activation factors computable.
- **The role of activation functions is clarified**: Saturated activations (tanh/sigmoid) increase complexity by scales of $\|A_l\|$, while unbounded activations (Leaky ReLU) do so by scales of $\|v\|$ (Remark 5.4).

## Highlights & Insights
- **"Constructing an RKHS on parameter space"** is the most ingenious step: It bypasses the lack of reproducibility in the model's $L^2$ space by mapping it to $R_k$ via isometric isomorphism $\iota$.
- **Algebrizing "why high-rank generalizes" into a single formula**: The $1/\det(W^*W)^{1/4}$ term makes the "large singular value $\to$ small bound" relationship explicit.
- **Linearizing non-linear composition via Koopman operators** is a transferable concept: This framework of operator algebra + operator norm complexity control could potentially be applied to structural compositions like RNNs, Normalizing Flows, or Diffusion sampling chains.

## Limitations & Future Work
- **Exact ReLU is not covered**: The framework fails when the derivative of the activation is zero over an interval (like exact ReLU). The authors suggest weighted Koopman operator variants might be a solution.
- **Trade-off between $E(c)$ and approximation accuracy**: For $p_{c,x}=(c/\pi)^{d/2}e^{-c\|y-x\|^2}$, $E(c)=(2c/\pi)^{d/2}$. Larger $c$ improves the approximation of $F_c$ to the original model but loosens the bound (Remark 4.3).
- **Experimental scale is limited**: Verification is limited to synthetic regression and MNIST/LeNet with $S=1000$ using indirect regularization validation. Tightness on large-scale models and data remains to be tested.

## Related Work & Insights
- **vs Norm-based / Compression-based Bounds** (Bartlett 2017, Arora 2018): These rely on $(p,q)$-norms or compressibility, favoring low-rank weights. Ours is complementary, specifically explaining high-rank / large-singular-value scenarios.
- **vs Hashimoto et al. (2024)**: While both use determinants to explain high-rank generalization, the prior work was limited to smooth, unbounded activations in Sobolev space. Ours extends this to tanh/sigmoid/Leaky ReLU, bounded data, and varying widths in $L^2$ space.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses RKHS + Group Rep + Koopman to extend high-rank bounds to realistic models.
- Experimental Thoroughness: ⭐⭐⭐ Small-scale verification on synthetic/MNIST/LeNet.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation chain, though mathematically dense.
- Value: ⭐⭐⭐⭐ Substantive advancement in generalization theory by theoretically justifying high-rank generalization.

## Related Papers

- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] Resurfacing the Instance-only Dependent Label Noise Model through Loss Correction](resurfacing_the_instance-only_dependent_label_noise_model_through_loss_correctio.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Derandomization Framework for Structure Discovery: Applications in Neural Networks and Beyond](a_derandomization_framework_for_structure_discovery_applications_in_neural_netwo.md)
- [\[ICLR 2026\] The Logical Expressiveness of Topological Neural Networks](the_logical_expressiveness_of_topological_neural_networks.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)

</div>

<!-- RELATED:END -->
