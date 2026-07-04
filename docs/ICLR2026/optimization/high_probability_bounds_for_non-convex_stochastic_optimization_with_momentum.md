---
title: >-
  [Paper Note] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum
description: >-
  [ICLR 2026][Optimization][SGDM] Ours completes the **high-probability** convergence and generalization bounds for Stochastic Gradient Descent with Momentum (SGDM) under non-convex settings: by relaxing noise to sub-Weibull heavy-tailed distributions and sequentially layering PL and Bernstein structural assumptions, a complete hierarchy of bounds is derived, ranging from $\tilde{O}(1/\sqrt{T})$ and $\tilde{O}(1/T)$ to dimension-independent $\tilde{O}(1/n^2)$…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "SGDM"
  - "high-probability bounds"
  - "non-convex optimization"
  - "generalization bounds"
  - "heavy-tailed noise"
date: 2026-05-08
content_hash: 7bb3d4f9946ac104
---

# High Probability Bounds for Non-Convex Stochastic Optimization with Momentum

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KirKWFPYJA](https://openreview.net/forum?id=KirKWFPYJA)  
**Code**: None (Not released)  
**Area**: Non-Convex Stochastic Optimization / Learning Theory  
**Keywords**: SGDM, high-probability bounds, non-convex optimization, generalization bounds, heavy-tailed noise

## TL;DR
Ours completes the **high-probability** convergence and generalization bounds for Stochastic Gradient Descent with Momentum (SGDM) under non-convex settings: by relaxing noise to sub-Weibull heavy-tailed distributions and sequentially layering PL and Bernstein structural assumptions, a complete hierarchy of bounds is derived, ranging from $\tilde{O}(1/\sqrt{T})$ and $\tilde{O}(1/T)$ to dimension-independent $\tilde{O}(1/n^2)$, marking the first generalization bounds for SGDM in the industry.

## Background & Motivation

**Background**: Stochastic Gradient Descent with Momentum (SGDM, specifically the Polyak Heavy-Ball method) is essentially the default optimizer in deep learning. Its update rule adds a momentum term to SGD:
$$x_{t+1} = x_t - \eta_t \nabla f(x_t; z_{j_t}) + \gamma(x_t - x_{t-1}),$$
relying on the inertia of historical directions to smooth stochastic gradients. However, the theoretical side lags significantly behind practice: analyzing "how well SGDM learns" requires two complementary perspectives—**convergence bounds** (how well it optimizes the empirical risk $F_S$) and **generalization bounds** (how the learned model performs on unseen data).

**Limitations of Prior Work**: Existing analyses of SGDM are almost entirely **in-expectation** bounds. Expectation bounds have two major flaws: first, they cannot exclude the possibility of "extremely poor results"; second, large-scale training is often a single run, and the performance of a single execution is better characterized by **high-probability bounds**. Currently, there are only two papers on high-probability convergence bounds: Li & Orabona (2020) provided $\tilde{O}(1/\sqrt{T})$ for the squared gradient norm under sub-Gaussian noise, and Cutkosky & Mehta (2021) provided $\tilde{O}(T^{-(\theta-1)/(3\theta-2)})$ under $\theta$-th moment conditions (using a momentum variant with clipping/normalization, not pure Polyak momentum). Both rates are relatively slow, and **neither provided any generalization bounds**.

**Key Challenge**: The difficulty on the generalization side is more fundamental. The mainstream tool for proving SGD generalization bounds is **algorithmic stability** (uniform stability). However, Ramezani-Kebrya et al. (2024) constructed counterexamples showing that even if the loss is convex, the uniform stability gap of multi-round SGDM may **diverge**. This means the standard path of "stability $\to$ generalization" is largely infeasible for SGDM—a trade-off exists between fast convergence and stability, and SGDM leans toward the fast convergence end.

**Goal**: To simultaneously provide high-probability convergence and generalization bounds for SGDM under non-convex settings, while covering more realistic heavy-tailed noise than sub-Gaussian.

**Key Insight**: The authors bypass stability and instead use the **uniform convergence** proof route. Simultaneously, by "layering structural assumptions," they map different assumption strengths to different tiers of rates, allowing the reader to see the full picture: "stronger assumptions yield faster rates."

**Core Idea**: Under sub-Weibull heavy-tailed noise, following a three-level structure of "General Non-Convex $\to$ PL $\to$ PL + Bernstein," ours proves a hierarchy of bounds from $\tilde{O}(1/\sqrt{T})$ convergence and $\tilde{O}(\sqrt{d/n})$ generalization, all the way to dimension-independent $\tilde{O}(1/n^2)$ generalization under low noise.

## Method

### Overall Architecture

This is a purely theoretical paper with no algorithmic modules requiring a pipeline diagram. Its core consists of "problem setup + assumption system + a set of main theorems + proof techniques." The overall logic is a main line of **progressing structural assumptions leading to faster rates**: the structure of the objective function is gradually tightened from "general non-convex" to "satisfying the PL condition," and then to "PL + gradients satisfying the Bernstein condition." Each level of tightening yields a faster tier of convergence/generalization rates. Throughout, gradient noise is relaxed to sub-Weibull heavy-tailed distributions, with a tail parameter $\theta$ uniformly characterizing the degradation rule where "heavier tails lead to slower rates."

**Problem Setup**: Parameter space $\mathcal{X} \subseteq \mathbb{R}^d$, loss $f(x;z) \ge 0$ may be non-convex. The goal is to minimize the population risk $F(x) = \mathbb{E}_{z \sim P}[f(x;z)]$, but since $P$ is unknown, one can only optimize the empirical risk $F_S(x) = \frac{1}{n} \sum_{i=1}^n f(x;z_i)$ composed of $n$ i.i.d. samples. The algorithm is the SGDM in Algorithm 1: at each step, a sample $j_t$ is drawn uniformly, the momentum $m_t = \gamma m_{t-1} + \eta_t \nabla f(x_t; z_{j_t})$ is updated, and then $x_{t+1} = x_t - m_t$ is set, which is equivalent to the expanded form with $\gamma(x_t - x_{t-1})$ mentioned above. Note that the batch size is set to 1, which is the "worst-case" scenario where momentum finds it hardest to show an advantage over SGD.

**Function**: In general non-convex settings where reaching a global minimum cannot be guaranteed, the **average squared gradient norm** is used for measurement—optimization considers $\frac{1}{T} \sum_{t=1}^T \|\nabla F_S(x_t)\|^2$, and generalization considers $\frac{1}{T} \sum_{t=1}^T \|\nabla F(x_t)\|^2$. After entering the PL regime, one can directly discuss **function value error**—optimization looks at $F_S(x_{T+1}) - F_S(x(S))$, and generalization looks at excess risk $F(x_{T+1}) - F(x^*)$. These bounds are provided for the **last iterate** rather than the average iterate, which is closer to practical deployment.

The four key designs below correspond to: how noise models are relaxed (Design 1), how the three-level structure yields three tiers of rates (Design 2), why generalization bounds switch to uniform convergence (Design 3), and how Bernstein is used to eliminate the dimension $d$ (Design 4).

### Key Designs

**1. sub-Weibull Noise Model: Unified characterization from light-tailed to heavy-tailed using a tail parameter**

Existing high-probability analyses of SGDM (Li & Orabona, 2020) only cover sub-Gaussian noise, but increasing evidence suggests that stochastic gradient noise is often heavier-tailed than Gaussian. Ours relaxes the noise assumption (Assumption 2.9) to sub-Weibull: requiring gradient noise $e_t = \nabla f(x_t; z_{j_t}) - \nabla F_S(x_t)$ to satisfy
$$\mathbb{E}_{j_t} \!\left[ \exp \! \Big( \big( \|e_t\|/K \big)^{1/\theta} \Big) \right] \le 2, \qquad \theta \ge \tfrac{1}{2}.$$
Here $\theta$ is the tail parameter: $\theta = 1/2$ reduces to sub-Gaussian, $\theta = 1$ corresponds to sub-exponential, and $\theta > 1$ represents exponential heavy-tailed distributions; the larger the $\theta$, the heavier the tail. The value of this design lies in its ability to explicitly write the rates of all main theorems as a function of $\theta$ (typically appearing as factors like $\log^{2\theta}(1/\delta)$ in convergence bounds), thus **quantitatively** answering the vague question of "how much convergence and generalization slow down as noise moves from light-tailed to heavy-tailed"—the conclusion being that larger $\theta$ leads to worse rates, consistent with intuition.

**2. Three-level Structural Assumptions $\to$ Three Tiers of Convergence and Generalization Rates**

This is the backbone of the paper: by imposing increasingly strong curvature structures on the objective function, increasingly fast rates are obtained.

*Level 1 (General Non-convex)*: Requires only $L$-smoothness (Assumption 2.1) + sub-Weibull noise. Theorem 3.1 gives the convergence bound
$$\frac{1}{T} \sum_{t=1}^T \|\nabla F_S(x_t)\|^2 = \tilde{O} \! \Big( \tfrac{1}{\sqrt{T}} \Big),$$
and Theorem 3.3 gives the generalization bound $\frac{1}{T} \sum_t \|\nabla F(x_t)\|^2 = \tilde{O} \big( \sqrt{d/n} \big)$ when the number of iterations is $T \asymp n/d$. Here $\tilde{O}(1/\sqrt{T})$ touches the lower bound for smooth non-convex first-order oracles (Arjevani et al., 2019). Compared to Li & Orabona (2020) under similar assumptions, ours refined the $\log(T/\delta)$ factor to $\log(1/\delta)$, and it is **dimension-independent** (whereas the competitor's delayed AdaGrad variant includes $d$).

*Level 2 (PL Condition)*: Further assumes that $F_S$ (and $F$ for generalization) satisfies the Polyak–Łojasiewicz inequality
$$f(x) - f^* \le \tfrac{1}{2\mu} \|\nabla f(x)\|^2,$$
meaning the gradient norm can inversely control the function value suboptimality gap—this is one of the weakest curvature conditions to replace strong convexity, satisfied locally by two-layer networks, matrix completion, phase recovery, and over-parameterized shallow networks. Paired with a decaying step size $\eta_t = \frac{1}{\mu(S)(t+t_0)}$, Theorem 3.5 **accelerates the convergence rate from $\tilde{O}(1/\sqrt{T})$ to $\tilde{O}(1/T)$** (last-iterate, function value error), and Theorem 3.7 improves generalization (excess risk) from $\tilde{O}(\sqrt{d/n})$ to
$$F(x_{T+1}) - F(x^*) = \tilde{O} \! \Big( \tfrac{d + \log(1/\delta)}{n} \Big),$$
improving the dependence on $n$ from $1/\sqrt{n}$ to $1/n$. To the authors' knowledge, the $\tilde{O}(1/T)$ high-probability rate for SGDM under PL has not been seen before.

*Level 3 (PL + Bernstein)*: See Design 4. Layering these three levels constitutes a complete spectrum from "slow but universal" to "fast but structured," and this **hierarchical** presentation itself is a highlight of the paper's organization.

**3. Switching to Uniform Convergence rather than Algorithmic Stability for Generalization Proofs**

As stated in the motivation, the uniform stability of SGDM can diverge, making the stability route for proving generalization a dead end. The generalization bounds in ours switch to a **uniform convergence** route (Xu & Zeevi, 2020; Lei & Tang, 2021): directly proving that "empirical risk for all hypotheses in the hypothesis class converges uniformly to their population risk," then substituting the algorithmic iteration points. This approach bypasses the problem of stability failing for SGDM, at the cost of uniform convergence naturally carrying a $\sqrt{d}$ dimension factor in general non-convex settings (Feldman, 2016), which is the source of $d$ in the $\tilde{O}(\sqrt{d/n})$ of Theorem 3.3. In other words, the authors exchanged "uncalculable stability" for "calculable dimension dependence"—a key technical choice that allowed the first generalization bounds for SGDM.

**4. Bernstein Condition for Localization: Eliminating dimension $d$ and hitting $\tilde{O}(1/n^2)$ under low noise**

The $\sqrt{d}$ dependence brought by uniform convergence is fatal in high-dimensional neural networks. To remove it, ours adds a mild Bernstein moment condition (Assumption 2.3): at the optimal point $x^*$, higher-order moments of the gradient norm are controlled by the second-order moment,
$$\mathbb{E}_z \big[ \|\nabla f(x^*;z)\|^k \big] \le \tfrac{1}{2} k! \, \mathbb{E}_z \big[ \|\nabla f(x^*;z)\|^2 \big] B_*^{k-2}, \quad k \ge 2.$$
The Bernstein condition is essentially equivalent to sub-exponential and is weaker than "uniformly bounded gradients," serving as a standard assumption in learning theory. Its role is to perform **localization** analysis: anchoring the generalization error to the variance structure near $x^*$, thereby escaping the dimension term generated during uniform coverage of the entire hypothesis class. After $T \asymp n^2$ and satisfying a mild sample size lower bound, Theorem 3.9 improves the excess risk to
$$F(x_{T+1}) - F(x^*) = \tilde{O} \! \Big( \tfrac{F(x^*)}{n} + \tfrac{1}{n^2} \Big),$$
where $F(x^*)$ is the optimal population risk, which is usually minimal. In **low-noise** regimes ($F(x^*) = O(1/n)$, common in interpolation/over-parameterization), this reduces to $\tilde{O}(1/n^2)$—a rare fast rate in learning theory featuring quadratic dependence on sample size. More importantly, this bound **does not contain dimension $d$**, making it directly applicable to high-dimensional large models. To the authors' knowledge, this is the first high-probability $\tilde{O}(1/n^2)$ generalization guarantee for SGDM.

### Mechanism

The high-probability bounds on the optimization side rely on expanding the momentum updates and organizing the stochastic gradient noise $e_t$ into a **martingale difference sequence**, then applying concentration inequalities (Fan & Giraudo, 2019; Li, 2021 class of weak exponential tail martingale concentration) to vector-valued martingales with sub-Weibull tails—the tail parameter $\theta$ enters the final rate at this step, manifesting as factors like $\log^{2\theta}(1/\delta)$. The treatment of the momentum term relies on coordinating step sizes like $\eta_t = ct^{-1/2}$ (or $\frac{1}{\mu(S)(t+t_0)}$ under PL) with the constant $C_\gamma = 1 + \frac{2}{\ln^2\gamma} - \frac{3}{\ln\gamma}$ (depending only on $\gamma$) to control inertia accumulation. Generalization follows uniform convergence in Design 3 and localization under PL+Bernstein in Design 4. Notably, Assumption 2.5 relaxes the bounded gradient assumption to $\eta_t \|\nabla F_S(x_t)\| \le G$—as step size decays, this allows $\|\nabla F_S(x_t)\|$ to grow with $t$, which is only required in the heavy-tailed regime of $\theta > 1/2$.

## Key Experimental Results

This is a theoretical paper; numerical experiments are only used to **corroborate the degradation trend of rates with the tail parameter $\theta$** and do not seek SOTA. Experiments were conducted on binary classification across 6 LIBSVM datasets (Heart, Fourclass, German, Australian, Diabetes, Phishing), with an 8:2 train/test split. Test set empirical risk $F_{S'}$ served as a proxy for population risk $F$. At each step, sub-Weibull noise was independently injected into each coordinate of the gradient, normalized to zero mean and unit variance, with $\theta \in \{1/2, 1, 5\}$, repeated 100 times for averaging.

### Main Results: Heavier Tails, Poorer Generalization

| Setting | Metric | Observed Trend | Theoretical Correspondence |
| :--- | :--- | :--- | :--- |
| Huber Loss (Fig 1, 6 datasets) | $\frac{1}{T} \sum_t \|\nabla F_{S'}(x_t)\|^2$ vs epochs | Curve gets higher as $\theta$ increases; $\theta=5$ is significantly worst | Corroborates Theorem 3.3 ($\theta > 1$ heavy-tail degradation) |
| Square Loss (Fig 2, 6 datasets) | Same as above | Systemic degradation with increasing $\theta$ | Consistent with Theorem 3.3 |

Hyperparameters were fixed at $\gamma = 0.9$, $\eta_t = 0.1 \, t^{-1/2}$, and Huber threshold $\tau = 0.1$.

### Key Findings
- Across both loss types and 6 datasets, the **average squared gradient norm monotonically worsens with $\theta$**, and $\theta = 5$ (heavy-tailed $\theta > 1$ region) is significantly inferior to $\theta = 1/2, 1$, perfectly matching the predicted "tail parameter entering the rate" theory.
- The experiment used a batch size of 1, which is the worst-case scenario where momentum is least likely to outperform SGD; this echoes the claim in the text—under this mechanism, SGDM and SGD share the same order of high-probability rates (Kidambi et al., 2018 also observed that momentum may not accelerate in this case).
- The paper does not conduct horizontal comparisons across different optimizers, as the goal is only to verify the qualitative "tail parameter $\to$ rate" relationship rather than benchmark speed.

## Highlights & Insights
- **Hierarchical narrative of "assumption progression ↔ rate tiers"**: Matching the three levels of assumptions (General Non-convex, PL, PL+Bernstein) with various rates ($\tilde{O}(1/\sqrt{T})$/$\tilde{O}(1/T)$, $\tilde{O}(\sqrt{d/n})$/$\tilde{O}(1/n)$/$\tilde{O}(1/n^2)$) allows the reader to see exactly what each additional structural assumption "buys." This is a commendable way to organize theoretical papers.
- **The "Bypassing Stability" move is crucial**: Since the uniform stability of SGDM diverges, forcing a stability route is destined for failure. Switching to uniform convergence introduces $\sqrt{d}$ but yields the first set of generalization bounds for SGDM—this "change the tool rather than the fight" strategy is transferable to other algorithms where stability fails (like Nesterov acceleration).
- **Bernstein Localization to Eliminate Dimension**: Dimension-independent generalization bounds of $\tilde{O}(1/n^2)$ are particularly friendly to high-dimensional neural networks, suggesting that "variance structure at the optimum + low noise" is a realistic path to fast rates. These localization techniques can be reused in other over-parameterized scenarios.
- **sub-Weibull Parametrizing Noise within Assumptions**: Using a single $\theta$ throughout all theorems transforms the "cost of heavy tails" from a vague intuition into an explicit factor in the rate, providing a clean paradigm for handling heavy-tailed stochastic optimization.

## Limitations & Future Work
- **Batch size fixed at 1**: The authors explicitly leave large-batch analysis for future work. It is precisely under this worst-case mechanism that ours only proves "SGDM is of the same order as SGD," failing to reflect theoretical acceleration from momentum; the benefits of momentum might rely precisely on noise reduction from large batches, which hasn't been characterized yet.
- **Failure to truly separate SGD and SGDM**: The paper admits that clear theoretical separation between SGD and SGDM under non-convex stochastic settings remains an open problem—ours proves that "SGDM is not inferior to SGD" rather than "momentum is superior."
- **Realism of PL/Bernstein Structural Assumptions**: Fast rates ($\tilde{O}(1/T)$, $\tilde{O}(1/n^2)$) rely on relatively strong conditions like PL and low noise $F(x^*) = O(1/n)$, where PL parameters $\mu(S)$ and $\mu$ must hold respectively. These conditions are often only locally or approximately satisfied in real deep networks, and the actual tightness of the bounds remains to be tested.
- **Experimental Focus on "Trend Verification"**: Validation was limited to qualitative effects of $\theta$ on small-scale LIBSVM with artificial noise; the tightness of the bounds was not quantified on real deep models, nor were horizontal optimizer comparisons provided.

## Related Work & Insights
- **vs Li & Orabona (2020)**: Also analyzed Polyak momentum, but they only covered sub-Gaussian noise, provided a convergence rate of $\tilde{O}(1/\sqrt{T})$ (with $\log(T/\delta)$), and had no generalization bounds. Ours extends noise to sub-Weibull heavy-tailed distributions, refines the logarithmic factor to $\log(1/\delta)$, and provides the first generalization bounds.
- **vs Cutkosky & Mehta (2021)**: They used a momentum variant with gradient clipping and normalization, providing $\tilde{O}(T^{-(\theta-1)/(3\theta-2)})$ under $\theta$-th moments ($\tilde{O}(T^{-1/4})$ when $\theta=2$). Ours analyzes **pure Polyak momentum without clipping**, deriving the same $\tilde{O}(T^{-1/4})$ rate via Jensen's inequality while adding generalization bounds.
- **vs Ramezani-Kebrya et al. (2024) / Chen et al. (2018)**: They pointed out that the uniform stability of SGDM diverges, causing the stability route to fail. Ours uses this as a starting point to switch to uniform convergence, thereby bypassing stability to prove generalization bounds.
- **vs Li & Liu (2022) (SGD without momentum)**: Under the same assumptions, the high-probability convergence/generalization bounds for SGDM in ours are of the same order (differing by constants and log factors) as their SGD results, thereby **closing the theoretical gap between SGDM and SGD**—proving that the widely used momentum method also enjoys high-probability guarantees equivalent to SGD under non-convex/PL/Bernstein assumptions.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to provide high-probability generalization bounds for SGDM and achieves rare dimension-independent $\tilde{O}(1/n^2)$ fast rates under low noise.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments only corroborate the $\theta$ trend; small scale with no horizontal comparisons, but sufficient for a theoretical paper.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The hierarchical narrative of "three-level assumptions — three tiers of rates" is clear, and Remarks clarify comparisons with prior work effectively.
- **Value**: ⭐⭐⭐⭐ Closes a long-standing gap in high-probability analysis for SGDM, offering solid reference value for stochastic optimization and learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] High-Probability Bounds for the Last Iterate of Clipped SGD](high-probability_bounds_for_the_last_iterate_of_clipped_sgd.md)
- [\[ICLR 2026\] Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization](derandomized_online-to-non-convex_conversion_for_stochastic_weakly_convex_optimi.md)
- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)

</div>

<!-- RELATED:END -->
