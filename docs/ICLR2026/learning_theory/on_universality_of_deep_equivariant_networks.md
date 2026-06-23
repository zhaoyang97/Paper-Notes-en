---
title: >-
  [Paper Note] On Universality of Deep Equivariant Networks
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper establishes the theorem of "universality under separation constraints" for deep invariant/equivariant networks, identifying depth and readout layers as the decisive mechanisms for achieving universality. It introduces a finer "entry-wise separability" criterion for the equivariant case than standard separati
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: c2baeb9447454482
---
# On Universality of Deep Equivariant Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q2D1PI6zY1](https://openreview.net/forum?id=Q2D1PI6zY1)  
**Code**: None (Theoretical paper, no experimental code)  
**Area**: Learning Theory / Equivariant Networks / Expressive Power  
**Keywords**: Equivariant networks, invariant networks, universal approximation, separation power, entry-wise separability

## TL;DR
This paper establishes the theorem of "universality under separation constraints" for deep invariant/equivariant networks, identifying depth and readout layers as the decisive mechanisms for achieving universality. It introduces a finer "entry-wise separability" criterion for the equivariant case than standard separation, unifying and generalizing previous conclusions limited to shallow or specific architectures.

## Background & Motivation

**Background**: Symmetry has become a core organizing principle in deep learning. Equivariant networks (CNNs, GNNs, PointNet, SE(3)-Transformer, etc.) encode inductive biases by ensuring "input transformation $\to$ output synchronous transformation," achieving widespread success on tasks involving molecules, point clouds, graphs, and manifolds. However, there has been concern: besides symmetry, does this inductive bias secretly impose **additional, unwanted** expressive limitations?

**Limitations of Prior Work**: The two mainstream routes for characterizing equivariant network expressivity both have flaws. The first studies **universality** directly (i.e., whether it can approximate all target functions compatible with the symmetry), but results from Ravanbakhsh, Maron, etc., require hidden layers to use regular representations or high-order tensors, causing the intermediate representation dimension to explode with group size, making them **impractical**. The second studies **separation power** (i.e., how many pairs of inputs can be distinguished), which is well-studied in graph learning via the Weisfeiler-Lehman test, but it is only a **necessary condition** for approximation.

**Key Challenge**: Separation is necessary for approximation but not necessarily sufficient. Pacini et al. (2025b) provided a counterexample: two shallow invariant architectures with the **same separation power but different approximation powers**, showing that "equal separation power" does not imply "equal class of approximable functions." This stands in sharp contrast to classical neural network theory, where depth only affects parameter efficiency and does not change the class of approximable functions under standard settings.

**Goal**: In both invariant and equivariant scenarios, clarify how **depth and readout layers** exactly fix the "separation power = approximation power" gap, and provide a unified framework that transcends previous architecture-specific results.

**Key Insight**: Zaheer, Qi, Segol & Lipman, among others, observed that adding a fully connected readout layer or increasing depth can turn restricted architectures into models that are "universal under separation constraints." The authors delve deeper into this: are depth and readout layers universal mechanisms for universality under separation constraints?

**Core Idea**: Reformulate "universality" as approximating the entire function class $C_\rho$ under a separation constraint $\rho$. For invariant networks, prove that "adding one fully connected readout layer achieves universality." For equivariant networks, identify that standard separation is too coarse and introduce the sharper **entry-wise separability**, proving universality can be achieved via either "sufficient depth" or "specific readout layers."

## Method

### Overall Architecture

The paper presents an approximation theory for the expressive power of equivariant networks. There is no training procedure or network pipeline to illustrate; instead, the logic advances from "invariant" to "equivariant" cases.

The authors unify all objects under the language of **permutation representations + layer spaces**. A finite group $G$ acting on a finite set $X$ yields a permutation representation $\mathbb{R}^X$. A **layer space** $M \subseteq \mathrm{Aff}_G(V, \mathbb{R}^X)$ is a family of affine maps satisfying equivariant constraints (linear layers, invariant layers $I$, convolutional layers $C$, and PointNet layers $P$ are all special cases). Stringing multiple layer spaces together with point-wise activation $\tilde\sigma$ yields a **neurospace**. Allowing intermediate widths to vary freely and taking the uniform convergence closure yields the **universal class** $U_\sigma(M_1,\dots,M_d)$ of an architecture.

Criteria are then defined. A family of functions $U$ induces an equivalence relation $\rho(U)$ (pairs of inputs that cannot be separated). If $U$ can approximate all continuous functions $C_\rho$ that "respect $\rho$," it is called **universal under separation constraints**. The three core parts of the paper are: (1) Invariant networks achieve $C_\rho$ universality via a fully connected readout; (2) The equivariant target class $C_\rho$ is imprecise and must be replaced by the entry-wise version $C_{\boldsymbol\rho}$; (3) Equivariant networks are driven to $C_{\boldsymbol\rho}$ universality via two different means (depth or convolutional readout).

### Key Designs

**1. Framework of Universality under Separation + FC Readout Theorem for Invariant Nets: Fixing the "Separation $\neq$ Approximation" Pathology with One Readout Layer**

Addressing the anomaly revealed by Pacini et al. (2025b) where invariant architectures with the same separation power have strictly unequal approximation power (e.g., $U_\sigma(C,I) \subsetneq U_\sigma(P,I) \subsetneq C_{S_n}(\mathbb{R}^n,\mathbb{R})$ despite identical separation). The authors formalize "universality" as equality under separation constraints. The separation relation of a function family $U$ is:

$$\rho(U) = \{(\alpha,\beta)\in V\times V \mid f(\alpha)=f(\beta)\ \forall f\in U\},$$

and the target function class consists of all continuous functions respecting this relation $C_\rho(V,W)=\{f\in C(V,W)\mid f(\alpha)=f(\beta)\text{ whenever }(\alpha,\beta)\in\rho\}$. **Theorem 1** proves: for any network ending with an invariant layer $I$, adding one standard fully connected readout $L$ (concatenation of $L=\mathrm{Aff}(\mathbb{R},\mathbb{R})$) yields:

$$U_\sigma(M_1,\dots,M_d,I,L) = C_\rho(V),\qquad \rho=\rho\big(U_\sigma(M_1,\dots,M_d,I)\big).$$

The insight is that the FC readout $L$ **does not change** the separation relation $\rho$ (fixed by previous layers) but allows arbitrary component functions $f_1,\dots,f_h$ to be combined non-linearly, filling the gap where separation was sufficient but approximation was not. This generalizes invariant universality results from Joshi et al. and Chen et al.

**2. Entry-wise Separability: Standard Separation is Too Coarse for Equivariance**

In the equivariant case, the authors find a counterexample (**Example 3**): for a convolutional layer space $C$ of width 1, a pure convolutional network of depth $d \ge 2$ satisfies:

$$U_\sigma^d(C) = \{(x_1,\dots,x_n)\mapsto(f(x_1),\dots,f(x_n))\mid f\in C(\mathbb{R})\}\ \subsetneq\ C_{S_n}(\mathbb{R}^n, \mathbb{R}^n).$$

Since the identity map is in $U_\sigma^d(C)$, its separation relation $\rho$ is **trivial** (everything can be separated), so the standard target is $C_{S_n,\rho}=C_{S_n}(\mathbb{R}^n,\mathbb{R}^n)$. However, the equation shows this target is unreachable regardless of depth. Standard universality under separation constraints **fails** in the equivariant case because "separation power" merges all output coordinates, whereas the real constraint is **what each output coordinate can separate individually**.

The authors introduce **entry-wise separability** (Definition 6). Let $\pi_x:\mathbb{R}^X\to\mathbb{R}$ be the projection to the $x$-th coordinate. Define the separation relation for each coordinate in neurospace $N$:

$$\rho_x(N) = \{(\alpha,\beta)\in V\times V \mid \pi_x f(\alpha)=\pi_x f(\beta)\ \forall f\in N\},$$

and package them into a **relation family** $\boldsymbol\rho(N)=(\rho_{x_1}(N),\dots,\rho_{x_n}(N))$. The target class $C_{\boldsymbol\rho}$ requires each coordinate to respect its own $\rho_x$. Since $\rho(N)=\bigcap_x \rho_x(N)$, entry-wise separation **implies** standard separation and may be strictly stronger. It reduces to standard separation in the invariant case ($G$ acting trivially on $\mathbb{R}$) or when all $\rho_x$ are equal. Proposition 2 proves the convolutional network in Example 3 satisfies $U_\sigma^d(C)=C_{\boldsymbol\rho}(\mathbb{R}^n,\mathbb{R}^n)$.

**3. Two Paths to Equivariant Universality: Depth Stability (Theorem 2) and Convolutional Readout (Theorem 3)**

With entry-wise separability, the authors provide two theorems for equivariant universality. **Theorem 2 (Depth Path)**: Let an output layer space $M$ (containing the identity) be stacked repeatedly. Once the depth reaches a threshold where **entry-wise separation stabilizes**—i.e.,

$$\rho := \rho\big(U_\sigma(M_1,\dots,M_f,\underbrace{M,\dots,M}_{d})\big)=\rho\big(U_\sigma(M_1,\dots,M_f,\underbrace{M,\dots,M}_{d+1})\big),$$

then one more layer achieves entry-wise universality $U_\sigma(\dots,\underbrace{M,\dots,M}_{d+1})=C_{\boldsymbol\rho}(V_0,\mathbb{R}^X)$. Combined with Pacini et al. (2025a), **Corollary 1** guarantees a threshold $D$ exists, beyond which the universal class **saturates**. This theoretically rules out "infinite gain from infinite depth."

**Theorem 3 (Readout Path)**: If the output layer is replaced by a convolutional filter $C$ of width 1, **no depth condition is required** to achieve $U_\sigma(M_1,\dots,M_f,C)=C_{\boldsymbol\rho}(V)$. The key difference: adding $C$ **does not change** the model's entry-wise separation power (acting as an "equivariant proxy" for the FC readout in the invariant case), whereas stacking $M$ layers may **increase** separation power—Theorem 2 pays the price of "waiting for stability" for this latter effect. Remark 1 recovers PointNet universality from Segol & Lipman (2020) and notes the depth threshold in Theorem 2 is sufficient but not always necessary.

## Key Experimental Results

This paper contains no experiments. The theoretical results are summarized below.

### Summary of Main Theorems

| Theorem / Conclusion | Setting | Mechanism | Form of Conclusion |
|------|------|------|------|
| Theorem 1 | Invariant Nets | Add FC Readout $L$ | $U_\sigma(M_1,\dots,M_d,I,L)=C_\rho(V)$ |
| Example 3 / Prop. 2 | Equivariant (Conv) | — (Standard Separation Fails) | $U_\sigma^d(C)=C_{\boldsymbol\rho}\subsetneq C_{S_n}$ |
| Theorem 2 + Cor. 1 | Equivariant Nets | Depth reaches stability | $U_\sigma(\dots,M^{d+1})=C_{\boldsymbol\rho}$, saturates at finite depth |
| Theorem 3 | Equivariant Nets | Width-1 Conv Readout $C$ | $U_\sigma(M_1,\dots,M_f,C)=C_{\boldsymbol\rho}(V)$ |

### Comparison with Prior Work

| Prior Work | Original Conclusion | Unification in Ours |
|------|------|------|
| Pinkus (1999) | Classical Universality | $U_\sigma(L,L)=C(\mathbb{R},\mathbb{R})$ as a trivial $\rho$ case |
| Segol & Lipman (2020) | 3-layer PointNet Universal | Recovered by Theorem 3 + Remark 1 |
| Joshi et al. (2023) | Head after G-orbit separation | Covered by Theorem 1 |
| Chen / Geerts / Maron | GNN Universality $\leftrightarrow$ WL | Integrated into the $C_\rho$ framework |
| Pacini et al. (2025b) | Same separation, diff approx | Theorem 1 / Entry-wise separation fixes this |

### Key Findings
- **Depth and readout layers are decisive mechanisms for universality**: Given constant separation power, they change the approximable function class—a fundamental contrast to classical networks where depth only improves parameter efficiency.
- **Equivariance requires finer separation criteria**: Standard separation merges output coordinates and overestimates capability; entry-wise separation looks at coordinates individually to accurately characterize the universal class.
- **The two paths are not equivalent**: Increasing depth (Theorem 2) may improve separation and requires waiting for stability; a convolutional readout (Theorem 3) does not change separation and achieves universality directly.

## Highlights & Insights
- **Reformulating universality** from "approximating everything" to "approximating $C_\rho$ under separation constraints" provides a unified language for what were previously architecture-specific proofs.
- **Entry-wise separability is a transferable analytical tool**: For any equivariant model where outputs are structured objects, one can analyze expressivity via "per-coordinate projections + individual separation relations" without heavy differential operator characterizations.
- **The Analogy "Readout = Equivariant Proxy of FC Readout"**: Explains why Theorem 3 specializes to Theorem 1 in the 1D case, stitching invariant and equivariant theories together.
- **Finite Depth Saturation** (Corollary 1) has practical implications: it tells engineers that expressivity hits a ceiling at a certain depth, discouraging blind stacking of layers.

## Limitations & Future Work
- **Limited to point-wise activations + permutation representations**: Generalizing to other representation types or general non-linearities may require entirely new methods.
- **Asymptotic conclusions without quantitative rates**: The paper does not provide approximation rates or sample complexity bounds, which are critical for understanding practical expressivity.
- **Does not address trainability**: While depth guarantees universality existence, whether such deep networks can be efficiently trained remains an open question—theoretical "existence" is not optimization "reachability."
- The depth threshold in Theorem 2 depends on "separation stability," which is difficult to determine generally; practical heuristics for estimating this stable depth are lacking.

## Related Work & Insights
- **vs. Ravanbakhsh (2020) / Maron et al. (2019b)**: They use regular representations/high-order tensors to prove universality at the cost of dimensions exploding with group size; this paper uses permutation representations to characterize **practical architectures** (CNN/GNN/PointNet).
- **vs. Joshi et al. (2023) / Chen et al. (2019)**: They established invariant universality via orbit separation; Theorem 1 covers this and extends the front to equivariant cases.
- **vs. Pacini et al. (2025b)**: They identified the "same separation, different approximation" pathology using differential operators; this paper provides an accurate characterization using entry-wise separation **without differential operators**.
- **vs. Classical Theory (Telgarsky 2016, Yarotsky 2017/2018)**: In classical results, depth only improves parameter efficiency; this paper reveals that under equivariant/separation constraints, depth actually expands the class of approximable functions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Entry-wise separation is a genuinely new criterion that fills gaps in equivariant universality theory.
- Experimental Thoroughness: ⭐⭐⭐ Theoretical paper with no experiments, but proofs are complete and self-consistent.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and good grounding in existing results, though concept-dense for non-theoreticians.
- Value: ⭐⭐⭐⭐ Provides a general framework and transferable tools for equivariant network expressivity analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Quasi-Equivariant Metanetworks](quasi-equivariant_metanetworks.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] Variational Deep Learning via Implicit Regularization](variational_deep_learning_via_implicit_regularization.md)

</div>

<!-- RELATED:END -->
