---
title: >-
  [Paper Note] On the Convergence Direction of Gradient Descent
description: >-
  [ICLR 2026][Optimization][Gradient Descent] This paper demonstrates that when Gradient Descent (GD) converges to a local strongly convex minimum, its trajectory does not approach from an arbitrary direction. Instead, it either **aligns to a fixed direction** (small learning rate) or **converges while oscillating back and forth along a straight line** (large learning rate). The boundary is exactly $\eta = 2/(\lambda_1+\lambda_n)$. This discrete version of the "Gradient Conject…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Gradient Descent"
  - "Convergence Direction"
  - "Edge of Stability"
  - "sharpness"
  - "Gradient Conjecture"
date: 2026-05-08
content_hash: 2fe2444c5d9d0263
---

# On the Convergence Direction of Gradient Descent

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3U6wH7uAPZ](https://openreview.net/forum?id=3U6wH7uAPZ)  
**Code**: TBD  
**Area**: Optimization Theory / Gradient Descent Dynamics  
**Keywords**: Gradient Descent, Convergence Direction, Edge of Stability, sharpness, Gradient Conjecture

## TL;DR
This paper demonstrates that when Gradient Descent (GD) converges to a local strongly convex minimum, its trajectory does not approach from an arbitrary direction. Instead, it either **aligns to a fixed direction** (small learning rate) or **converges while oscillating back and forth along a straight line** (large learning rate). The boundary is exactly $\eta = 2/(\lambda_1+\lambda_n)$. This discrete version of the "Gradient Conjecture" also provides an explanation for the sharpness oscillation observed in the Edge of Stability phenomenon.

## Background & Motivation

**Background**: GD is the most fundamental and thoroughly researched optimization algorithm in deep learning. Classical analysis (convex + $L$-smooth) dictates that as long as the learning rate satisfies $0<\eta<2/L$, GD converges stably with standard convergence rates. Based on second-order approximations near a local minimum, a more refined conclusion is that convergence usually requires $0<\eta<2/\lambda_n$, where $\lambda_n$ is the maximum eigenvalue of the Hessian at the minimum.

**Limitations of Prior Work**: These classical frameworks only answer "whether it will converge" and "how fast it converges," while providing almost no characterization of **what the trajectory looks like**—especially the direction from which GD approaches a local minimum. More awkwardly, experiments often show GD continuing to progress even when $2/\eta$ is smaller than the current sharpness (violating classical stability conditions). This is the Edge of Stability (EoS) phenomenon observed by Cohen et al., where the loss fluctuates non-monotonically over short time scales while the sharpness hovers around $2/\eta$. Classical theory is unable to explain these dynamics of "unstable yet progressing" optimization.

**Key Challenge**: For continuous-time gradient flow, a definitive answer exists—the **Gradient Conjecture** proposed by René Thom and proven by Parusinski et al. (2000) asserts that the normalized secant $\frac{x(t)-x_0}{\|x(t)-x_0\|}$ of a gradient flow trajectory of an analytic function must converge near a critical point (the direction exists). However, discrete GD introduces a learning rate $\eta$, meaning step sizes are no longer infinitesimal. Whether such directions still exist or if large $\eta$ values trigger behaviors absent in continuous time has remained unanswered.

**Goal**: To "translate" the continuous-time Gradient Conjecture to discrete GD and characterize what determines the convergence direction.

**Key Insight**: The authors observe behavior in a simple quadratic example. For $f(x,y)=x^2/2+2y^2$, the analytical GD solution is $x_k=(1-\eta)^k x_0,\ y_k=(1-4\eta)^k y_0$. Examining the normalized vector $v_k=\frac{(x_k,y_k)}{\|(x_k,y_k)\|}$: when $0<\eta<2/5$, $v_k\to(\mathrm{sign}(x_0),0)$ (locking onto the small eigenvalue direction, the $x$-axis); when $2/5<\eta<1/2$, $(-1)^k v_k\to(0,\mathrm{sign}(y_0))$ (alternating oscillation along the large eigenvalue direction, the $y$-axis). This pattern persists even with higher-order perturbations. This "phase transition" suggests the direction is determined by the relative relationship between the learning rate and the eigenvalues.

**Core Idea**: Characterize the asymptotic direction of GD using "eigenvalues + learning rate"—**small learning rates follow the flattest direction (minimum eigenvalue), while large learning rates oscillate along the steepest direction (maximum eigenvalue)**, with the boundary at $2/(\lambda_1+\lambda_n)$.

## Method

As pure theoretical work, the core consists of a theorem (Theorem 1) and its proof, along with an explanation for EoS and validation experiments. There is no engineering pipeline to visualize; thus, no architecture diagram is provided. The logical chain is: establish intuition with a quadratic example $\to$ provide the main theorem under general $C^3$ strongly convex minima $\to$ complete the proof via Hessian diagonalization, component separation, and invariant set arguments $\to$ combine the "direction + eigenvalue perturbation" to explain EoS sharpness oscillation $\to$ verify the ubiquity of directional alignment in real networks (SGD/Adam).

### Overall Architecture

Let the loss $f\in C^3(\mathbb{R}^n)$, GD iterations $x_{k+1}=F(x_k)=x_k-\eta\nabla f(x_k)$, and $x^*$ be an isolated local minimum. Let $V_\eta=\{x_0:\lim_k x_k=x^*\}$ be the set of initial values converging to it. A technical assumption is added: the preimage $F^{-1}(W)$ of any zero-measure set $W$ is also zero-measure (a standard assumption in GD dynamics analysis to exclude "pathological" initial values). At $x^*$, the Hessian eigenvalues satisfy $0<\lambda_1<\lambda_2\le\cdots\le\lambda_{n-1}<\lambda_n$ (local strong convexity). Under this setting, the paper presents a bifurcated main conclusion, centered on the rigorous proof and explanatory power of this bifurcation. The key design points follow a step-by-step progression: the main theorem is the skeleton, the proof technique is the support, and the EoS explanation and modern optimizer experiments are the two outward-reaching applications.

### Key Designs

**1. Main Theorem: Dichotomous Phase Transition of Convergence Direction with Learning Rate**

This is the framework's core, directly addressing the direction GD takes near a minimum. For almost all initial values $x_0\in V_\eta$:

- When $0<\eta<\dfrac{2}{\lambda_1+\lambda_n}$ (small learning rate), the **convergence direction exists**: $\displaystyle\lim_{k\to\infty}\frac{x_k-x^*}{\|x_k-x^*\|}$ exists and aligns with the eigenvector corresponding to the **minimum eigenvalue** $\lambda_1$ (the flattest direction).
- When $\dfrac{2}{\lambda_1+\lambda_n}<\eta<\dfrac{2}{\lambda_n}$ (large learning rate), an **alternating convergence direction exists**: $\displaystyle\lim_{k\to\infty}(-1)^k\frac{x_k-x^*}{\|x_k-x^*\|}$ exists, and the trajectory oscillates along the eigenvector corresponding to the **maximum eigenvalue** $\lambda_n$ (the steepest direction), flipping signs at each iteration.

The boundary at $2/(\lambda_1+\lambda_n)$ can be intuitively understood through the linear main term $1-\eta\lambda_i$. When $\eta<2/(\lambda_1+\lambda_n)$, the term $1-\eta\lambda_1$ is the largest and positive for the minimum eigenvalue direction, resulting in the slowest contraction; thus, this direction "survives" to dominate the trajectory. When $\eta$ crosses $2/(\lambda_1+\lambda_n)$, the factor $1-\eta\lambda_n$ for the maximum eigenvalue direction becomes the negative value with the largest magnitude, yielding the slowest contraction while flipping signs every step. This generalizes the "direction must converge" property of the continuous-time Gradient Conjecture to discrete GD and reveals a new branch—oscillatory convergence—that does not exist in continuous time.

**2. Proof Technique: Diagonalization + Component Separation + Forward Invariant Sets**

The challenge lies in $f$ being a general nonlinear function, preventing direct use of quadratic analytical solutions. The approach is as follows: without loss of generality, let $x^*=0$ and $\nabla^2 f(0)=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$. The GD iteration is written component-wise as a linear term + high-order remainder:

$$x_{k+1,i}=(1-\eta\lambda_i)x_{k,i}+g_i(x_k),\qquad g_i(x)=\eta(\lambda_i x_i-\partial_i f(x)),$$

where $g_i$ is a second-order quantity encapsulating the nonlinear "residual." Let $a=1-\eta\lambda_1$ and $b=\max_{2\le i\le n}|1-\eta\lambda_i|$. From $\eta<2/(\lambda_1+\lambda_n)<1/\lambda_1$, it follows that $a>b\ge0$, meaning the contraction factor of the minimum eigenvalue direction is strictly larger than the others.

The proof has three layers. **Layer 1 (Lemma 1)** uses the forward invariance theorem from the authors' previous work (Chen et al., 2024), constructing an open set $\Omega\subset V_\eta$ containing $0$ and a constant $C$ such that the trajectory never escapes, providing key estimates $|g_{k,i}|\le C\|x_k\|^2$ and $\left|\partial g_{k,i}/\partial x_{k,j}\right|\le C\|x_k\|$; thus, high-order terms are controlled relative to the linear term. **Layer 2 (Lemma 2)** selects a sufficiently small $\varepsilon$ to make the ball $B(0,\varepsilon)$ forward invariant. **Layer 3** defines a "bad initial set" $S=\{x_0\in B(0,\varepsilon):\forall k,\ |x_{k,1}|<\sum_{i\ge2}|x_{k,i}|\}$ and proves it has zero measure. For other initial values, once $|x_{k^*,1}|\ge\sum_{i\ge2}|x_{k^*,i}|$ is satisfied at some step, use $a>b$ to inductively show the first component dominates thereafter with a constant sign, leading to:

$$\frac{|x_{k,i}|}{|x_{k,1}|^\alpha}\le C_1\quad(1<\alpha<2)\ \Longrightarrow\ \lim_{k\to\infty}\frac{x_{k,i}}{x_{k,1}}=0.$$

Consequently, $\frac{x_k}{\|x_k\|}\to(\mathrm{sign}(x_{k^*,1}),0,\dots,0)$, locking the direction to the $\lambda_1$ axis. The proof for the large learning rate case (oscillating along $\lambda_n$) follows a similar structure and is detailed in the appendix. This "diagonalization $\to$ separation of dominant component $\to$ invariant set + ratio vanishing" technique effectively compresses the asymptotic behavior of general functions back to that of quadratic terms.

**3. Sharpness Oscillation: Explaining Edge of Stability via Directional Conclusions**

The hallmark of EoS is that the training sharpness (maximum Hessian eigenvalue) $\lambda_n(x_k)$ fluctuates around the threshold $2/\eta$. This paper explains this by combining directional conclusions with first-order eigenvalue perturbations. According to matrix perturbation theory, $\lambda_n(x)$ is differentiable near the minimum: $\lambda_n(x)=\lambda_n+\omega^\top x+o(\|x\|)$. Substituting the directional conclusions:

- Small learning rate: $\lambda_n(x_k)=\lambda_n+C_\eta\|x_k\|+o(\|x_k\|)$. Since $\|x_k\|$ monotonically decreases, sharpness **monotonically** converges to $\lambda_n$ (increasing or decreasing based on the sign of $C_\eta$).
- Large learning rate: Since the trajectory flips signs along the dominant direction, the perturbation term acquires a $(-1)^k$ factor:
$$\lambda_n(x_k)=\lambda_n+(-1)^k C_\eta\|x_k\|+o(\|x_k\|),$$
Thus, even and odd steps approach $\lambda_n$ from opposite sides, causing sharpness to **oscillate** towards convergence. Crucially, because $\lambda_n<2/\eta<\lambda_1+\lambda_n$, early in the trajectory (when $\|x_k\|$ is still large), $\lambda_n(x_k)$ can easily **overshoot** $2/\eta$. This corresponds exactly to the phenomenon in EoS where sharpness occasionally crosses the $2/\eta$ limit, while the long-term envelope still converges to $\lambda_n$. This elevates EoS sharpness fluctuations from "empirical observation" to a corollary of the directional theorem.

**4. Directional Alignment in Modern Optimizers: Verification via Cosine Similarity**

As the theory only covers deterministic vanilla GD, the authors investigated whether this directional alignment holds for practical optimizers. They used a CNN on CIFAR-10, running SGD with momentum and Adam, tracking the cosine similarity of adjacent parameter updates:

$$\cos\langle\Delta x_{k+1},\Delta x_k\rangle=\frac{\langle\Delta x_k,\Delta x_{k+1}\rangle}{\|\Delta x_k\|\,\|\Delta x_{k+1}\|},\qquad \Delta x_k=x_{k+1}-x_k.$$

Experiments found that as the loss converges stably, the cosine value for all three methods (GD/SGD/Adam) tends toward $1$. This indicates that the update direction gradually aligns to a stable vector, consistent with theoretical predictions. This extends a conclusion proven only for ideal GD to real-world training, suggesting directional alignment may be a more universal geometric property of optimization.

### A Complete Example

Consider $f(x,y)=x^2+y^2/2+x^2y+x^3$ (local minimum at $(0,0)$, $\lambda_1=1, \lambda_2=2$, boundary at $2/(\lambda_1+\lambda_2)=2/3$):

- For $\eta=0.1<2/3$ (small learning rate): The trajectory converges to the minimum along the $v=(0,\mp1)$ direction. Sharpness $\lambda_2(x_k)$ **monotonically** converges to $\lambda_2=2$, matching the small learning rate prediction.
- For $\eta=0.95\in(2/3,\,1)$ (large learning rate, $2/\lambda_2=1$): The trajectory **oscillates** along the steepest direction $v=(1,0)$. Sharpness fluctuates around the asymptotic value $\lambda_2$ and occasionally crosses the theoretical upper bound $2/\eta\approx2.105$ (red dashed line in the plot)—this is the EoS phenomenon, but the long-term convergence of the oscillation envelope to $\lambda_2$ confirms the theorem.

## Key Experimental Results

As a theoretical paper, these "experiments" serve to validate the theorem rather than achieve state-of-the-art metrics.

### Directional Phase Transition (Synthetic Functions)

| Function | Learning Rate $\eta$ | Theoretical Boundary | Observed Convergence Direction |
|------|------|------|------|
| $x^2/2+2y^2$ | $0.1$ | $2/5$ | Locks on $x$-axis (Minimum eigenvalue direction) |
| $x^2/2+2y^2$ | $0.42$ | $2/5$ | Oscillates along $y$-axis (Maximum eigenvalue direction) |
| $x^2/2+2y^2+xy^2+y^3$ | $0.1$ | — | Still locks on $x$-axis (Robust to high-order perturbations) |
| $x^2/2+2y^2+xy^2+y^3$ | $0.42$ | — | Still oscillates along $y$-axis |

### Sharpness Behavior and Modern Optimizers

| Configuration | Phenomenon | Relationship to Theory |
|------|------|------|
| $f$ example, $\eta=0.1$ | $\lambda_2(x_k)$ monotonic $\to 2$ | Small $\eta$: Fixed direction $\implies$ Monotonic sharpness |
| $f$ example, $\eta=0.95$ | $\lambda_2(x_k)$ oscillates $\to 2$, crosses $2/\eta$ | Large $\eta$: Sign flip $\implies$ Sharpness oscillation (EoS) |
| CIFAR-10 + CNN, SGD-m | $\cos\langle\Delta x_{k+1},\Delta x_k\rangle\to 1$ | Directional alignment occurs in stochastic optimizers |
| CIFAR-10 + CNN, Adam | $\cos\langle\Delta x_{k+1},\Delta x_k\rangle\to 1$ | Directional alignment occurs in adaptive optimizers |

### Key Findings
- The "phase transition" of the convergence direction is determined entirely by the position of $\eta$ relative to $2/(\lambda_1+\lambda_n)$ and is robust to high-order perturbations, indicating it is an asymptotic property dominated by the local quadratic term.
- Sharpness exceeding $2/\eta$ in EoS is not an anomaly but an inevitable early behavior caused by directional oscillation under large learning rates, with the long-term envelope still converging—unifying empirical observation under the theorem.
- Directional alignment is not limited to ideal GD: SGD and Adam also exhibit high alignment of adjacent updates during stable descent, suggesting a universal geometric regularity in optimization.

## Highlights & Insights
- **Bridging the Gradient Conjecture to Discrete GD**: While "direction must converge" is a classic result for continuous gradient flow, this work replicates it for discrete GD using diagonalization, invariant sets, and ratio vanishing techniques, while discovering the "oscillatory convergence" branch absent in continuous time.
- **A Clean Threshold $2/(\lambda_1+\lambda_n)$ Unifies Behaviors**: The value precisely separates flattest-direction convergence from steepest-direction oscillation, and conveniently separates EoS sharpness oscillation from classical stable convergence, offering theoretical elegance.
- **Direction + First-order Perturbation = EoS Explanation**: Characterizing $\lambda_n(x_k)=\lambda_n+(-1)^k C_\eta\|x_k\|+o(\|x_k\|)$ explains sharpness oscillation in a single step—a reusable analytical pattern: if the trajectory direction is known, the oscillation pattern of any smooth scalar along the path can be predicted.
- **Cosine Similarity as an Observable Proxy**: $\cos\langle\Delta x_{k+1},\Delta x_k\rangle\to1$ is simple to measure and serves as a practical signal to determine when optimization has entered the "directional locking" phase.

## Limitations & Future Work
- **Limited to Local Strongly Convex Minima**: The Hessian eigenvalues are required to be strictly positive and distinct ($0<\lambda_1<\cdots<\lambda_n$). Saddle points, flat minima, degenerate Hessians, and general non-convex landscapes are not covered. Extension to more general functions (e.g., those satisfying the KL condition) is labeled for future work.
- **Strict Proof Only for Vanilla GD**: Directional alignment for SGD/Adam currently relies only on empirical evidence from CIFAR-10 without formal proof. How momentum and adaptive step sizes modify the phase transition remains an open question.
- **Directional Information Only, No Rate**: The theorem answers "**which direction**," but does not quantify "**how fast**" it converges along that direction. Refined rate guarantees for directional convergence could lead to better theoretical bounds or help design direction-aware optimizers.
- **Reliance on Technical Assumptions**: The proof depends on the assumption that $F^{-1}$ preserves zero-measure sets and the conclusion holds for "almost all" initial values, leaving room for pathological counterexamples of zero measure.

## Related Work & Insights
- **vs. Gradient Conjecture (Parusinski et al., 2000)**: They proved normalized secant convergence for continuous gradient flow. This paper moves to discrete GD, where the learning rate introduces a new freedom, resulting in both "directional convergence" and "oscillatory convergence" branches.
- **vs. EoS (Cohen et al., 2021)**: They empirically observed sharpness hovering near $2/\eta$. This paper provides an analytical explanation via the directional theorem and first-order perturbations, showing that crossing $2/\eta$ is a natural consequence of sign-flipping at large learning rates.
- **vs. Authors' Prior Work (Chen et al., 2024, Unstable Convergence)**: This paper builds upon and progresses that line of research by reusing the forward invariance theorem (Theorem 2) as Lemma 1 to characterize the specific direction.
- **vs. SAM / Long Steps / Momentum in EoS (Long & Bartlett 2024; Grimmer 2024; Phunyaphibarn et al. 2024)**: These works study large learning rate behavior via regularization, acceleration, or the catapult effect. This paper provides a complementary "asymptotic direction" perspective, potentially pointing toward a unified framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to strictly generalize the Gradient Conjecture to discrete GD and discover the learning-rate-driven phase transition and oscillatory convergence.
- **Experimental Thoroughness**: ⭐⭐⭐ Theoretical focus; synthetic examples and CIFAR-10 verification are adequate, though modern optimizers lack formal proof.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from intuitive examples to theorem, proof, and EoS explanation (minor typos do not hinder understanding).
- **Value**: ⭐⭐⭐⭐ Provides a clean theoretical perspective on long-term GD dynamics and EoS, potentially inspiring new optimizer designs leveraging directional properties.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime](on_the_convergence_behavior_of_preconditioned_gradient_descent_toward_the_rich_l.md)
- [\[ICLR 2026\] Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region](gradient_descent_with_large_step_sizes_chaos_and_fractal_convergence_region.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICLR 2026\] Fast Convergence of Natural Gradient Descent for Over-parameterized Physics-Informed Neural Networks](fast_convergence_of_natural_gradient_descent_for_over-parameterized_physics-info.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->
