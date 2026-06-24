---
title: >-
  [Paper Note] Dimension-Free Decision Calibration for Nonlinear Loss Functions
description: >-
  [ICLR2026][learning theory][Decision calibration] When downstream decision-makers utilize model predictions for decision-making, "decision calibration" requires predictions to be unbiased with respect to decision-relevant events. This paper extends this concept from linear to nonlinear loss functions. It proves that auditing calibration under deterministic optimal response inevitably requires $\Omega(\sqrt{m})$ samples (where $m$ is the feature dimension). However…
tags:
  - "ICLR2026"
  - "learning theory"
  - "Decision calibration"
  - "nonlinear loss"
  - "RKHS"
  - "quantal response"
  - "sample complexity"
date: 2026-05-08
content_hash: 59803c126eb109a4
---

# Dimension-Free Decision Calibration for Nonlinear Loss Functions

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=vAU1fo1zRV](https://openreview.net/forum?id=vAU1fo1zRV)  
**Code**: To be confirmed  
**Area**: learning theory  
**Keywords**: Decision calibration, nonlinear loss, RKHS, quantal response, sample complexity

## TL;DR
When downstream decision-makers utilize model predictions for decision-making, "decision calibration" requires predictions to be unbiased with respect to decision-relevant events. This paper extends this concept from linear to nonlinear loss functions. It proves that auditing calibration under deterministic optimal response inevitably requires $\Omega(\sqrt{m})$ samples (where $m$ is the feature dimension). However, by adopting smoothed quantal response, the authors provide auditing and post-processing algorithms with sample complexity $\mathrm{poly}(|A|,1/\epsilon)$ that is **completely independent of dimension $m$**, covering a wide range of loss classes including piecewise linear, Cobb–Douglas, and any Lipschitz differentiable functions.

## Background & Motivation
**Background**: In high-risk scenarios such as medical diagnosis and financial forecasting, model predictions $p(x)$ are used by downstream decision-makers to optimize their utility—choosing an action $a$ according to a loss function $\ell(a,y)$. A fundamental question is: when can a decision-maker "treat the prediction as the ground truth"? Classic **calibration** provides an answer: if $\mathbb{E}[Y\mid p(X)=v]=v$, the prediction is unbiased and can safely substitute the true value. However, in high-dimensional outcome spaces, full calibration requires verifying unbiasedness for an exponential number of $\{p(x)=v\}$ events, leading to a sample complexity that explodes exponentially with the dimension.

**Limitations of Prior Work**: Zhao et al. (2021) proposed a weaker notion called **decision calibration** to bypass the curse of dimensionality—requiring predictions to be unbiased only on "decision-relevant" events, i.e., for a loss $\ell$ and decision rule $k$:
$$\mathbb{E}_{(x,y)\sim D}\mathbb{E}_{a\sim k(x)}[\ell(a,y)]=\mathbb{E}_{(x,y)\sim D}\mathbb{E}_{a\sim k(x)}[\ell(a,p(x))].$$
Both auditing and post-processing for this notion are polynomial-time in high dimensions. However, **all existing works only handle linear losses**—linearity is critical for the guarantee that "taking the optimal response to a calibrated prediction is optimal for the true value."

**Key Challenge**: In reality, many loss functions are nonlinear—risk-averse investors penalize large losses more heavily, and clinicians assign higher weights to severe medical outcomes. A natural approach to nonlinearity is **feature expansion** $\phi:Y\to H$, expressing the loss as a linear operator in the feature space: $\ell(a,y)=\langle r_\ell(a),\phi(y)\rangle_H$. The problem is that even for simple nonlinear functions, the required feature dimension $m=\dim(H)$ can be exponentially large or infinite. Since existing decision calibration algorithms have sample complexities polynomial in $m$, they become infeasible when applied to such high-dimensional expansions.

**Goal**: Is it possible to achieve decision calibration with complexity **completely independent** of the feature dimension $m$? I.e., "dimension-free decision calibration."

**Key Insight**: The authors first investigated the sample complexity required for auditing itself and found that the path of deterministic optimal response (hard-max) is blocked by a lower bound. Consequently, they turned to **quantal response**, a stochastic, smoothed optimal decision rule well-studied in decision theory that naturally characterizes bounded rationality. Smoothing is the key to keeping covering numbers bounded, thereby escaping dimensional dependency.

**Core Idea**: Use smoothed quantal response instead of deterministic optimal response, and utilize a pseudo-metric that projects high-dimensional loss vectors onto a one-dimensional space to control the covering number. This allows both auditing and patching for decision calibration to be dimension-free.

## Method

### Overall Architecture
This paper addresses a "test + post-processing" problem: given an arbitrary initial predictor $p_0:X\to H$ and a tolerance $\epsilon$, process it into a predictor that is **decision-calibrated** for a large class of nonlinear losses without increasing its mean squared error, using a sample size independent of the feature dimension $m$.

The pipeline consists of three components. **First is problem reduction**: the loss class $L$ is linearized into $\hat L_\phi$ via a feature mapping $\phi$ and $(\dim H,\lambda,\epsilon/2)$-uniform approximation. The predictor $p$ is not directly exposed to the decision-maker; instead, a loss estimator $f_p(x,a,\ell)=\langle r_\ell(a),p(x)\rangle_H$ is constructed for queries. Lemma 2.1 guarantees: as long as $\epsilon/2$-decision calibration is achieved for the linearized class $\hat L_\phi$, $\epsilon$-decision calibration is achieved for the original nonlinear class $L_\phi$. **Second is auditing**: an oracle is used to detect if the current $p_t$ is calibrated; if not, it outputs a pair of violating loss functions $(\ell, \ell')$. **Third is patching**: decision calibration is equivalently rewritten as "weighted calibration," and the predictor is updated along the witness direction until calibration is achieved.

The lower bound result ($\Omega(\sqrt m)$ under deterministic response) is not a step in the pipeline but a motivational conclusion: it proves that dimension-free algorithms are impossible under deterministic optimal response, forcing the framework to switch to quantal response.

### Key Designs

**1. $\Omega(\sqrt{m})$ Lower Bound under Deterministic Optimal Response: Why hard-max must be replaced**

The deterministic optimal response (hard-max) rule assumes the outcome is perfectly predicted and selects the action that minimizes loss. The first result of this paper is "bad news": even in the simplest setting—number of actions $|A|=2$, $\phi(y)=y$, and linear loss class $L_{\text{LIN}}$—to distinguish whether a predictor is $0$-decision calibrated or $\epsilon$-violating, **any algorithm requires at least $\Omega(\sqrt d)$ samples** (Theorem 3.1, where $d$ is result dimension). This is the **first statistical complexity lower bound** for decision calibration known to the authors. The proof follows an "indistinguishability" route: constructing two nearly identical distributions $D_1, D_2$, where only $D_1$ satisfies decision calibration. $D_2$ injects a tiny bias $(y-p(x))$ into the outcome, making it statistically difficult to distinguish from zero-mean label noise, while utilizing VC theory "shattering" arguments—the optimal response region for two actions is a half-space $\mathbb{1}[\langle r,p(x)\rangle>0]$, and one can find a loss $\ell$ such that the corresponding half-space perfectly covers the biased region. Since all known decision calibration algorithms rely on iterative auditing, and auditing naturally inherits this $\Omega(\sqrt d)$ lower bound, this proves that no non-trivial dimension-free algorithm exists under deterministic response.

**2. Quantal Response Smoothed Decision Rule: Making the impossible possible**

Since hard-max is blocked, the paper turns to **quantal response**—an optimal rule smoothed by a temperature parameter $\beta>0$:
$$\tilde k_{f_p,\ell}(x,a)=\frac{e^{-\beta\langle r_\ell(a),p(x)\rangle_H}}{\sum_{a'}e^{-\beta\langle r_\ell(a'),p(x)\rangle_H}}.$$
It selects actions stochastically based on loss estimates and is a classic model in economics and decision theory for "bounded rationality." The key lies not in its realism, but in its **smoothness**: action probabilities are continuous (softmax) functions of loss estimates. Small perturbations in the prediction vector result in only small changes in the action distribution. This allows the covering number argument to hold—under hard-max, actions are discontinuous functions (half-space indicators) of predictions, causing covering numbers to explode with dimension. With quantal response, the calibration gap function becomes Lipschitz, and the covering number is controlled by a value independent of $m$.

**3. Projection Pseudo-metric and Dimension-Free Covering Number: Why auditing is dimension-independent**

The goal of auditing is to witness a pair $(\ell, \ell')$ such that the empirical calibration error exceeds $\epsilon/2$ with high probability when $\mathrm{decCE}(f_p,D)\ge\epsilon$. Lemma 4.1 rewrites decision calibration in a more intuitive form: $f_p$ is $(L_H,\tilde K_{L_H},\epsilon)$-calibrated if and only if
$$\sup_{\ell,\ell'\in L_H}\Big|\mathbb{E}_{(x,y)\sim D}\Big[\sum_{a=1}^{|A|}\langle r_\ell(a),\phi(y)-p(x)\rangle\,\tilde k_{f_p,\ell'}(x,a)\Big]\Big|\le\epsilon.$$
The authors collect all "calibration gap functions" $g_{\ell,\ell'}$ into a class $\mathcal G$ and prove that its **uniform convergence upper bound** (Theorem 4.1) is proportional to $\big(C_1\log(C_2 n)+\log(1/\delta)\big)/\sqrt n$, where $C_1=|A|^{3/2}\beta^2R_1^3R_2^3$ and $C_2=R_1R_2$—**completely independent of dimension $m$**. The technical core is a carefully designed **pseudo-metric**: it first projects high-dimensional loss vectors onto a 1D space along a random direction (given by prediction $p(X)$ from random samples), then measures distance in this 1D space,
$$d_P(r_{\ell_1}(a),r_{\ell_2}(a))=\sqrt{\mathbb{E}_{X\sim P}[\langle r_{\ell_1}(a)-r_{\ell_2}(a),X\rangle^2]}.$$
Using Dudley's chaining method, the covering number is reduced to a finite covering number on the Hilbert ball $B(R_1)$. While this is infinite under standard metrics when $m$ is unbounded, it remains bounded under this projected pseudo-metric (combined with quantal response smoothness). This immediately leads to an ERM oracle serving as an auditing algorithm (Theorem 4.2).

**4. Decision Calibration $\equiv$ Weighted Calibration + Implicit Patching: The DimFreeDeCal Main Algorithm**

Given an auditing oracle, the authors designed the post-processing algorithm **DimFreeDeCal** (Algorithm 1). The first step **reduces decision calibration to weighted calibration**: by the $W$-calibration error $\mathrm{CE}_W(p)=\sup_{w\in W}|\mathbb{E}_D[\langle w(p(x)),p(x)-\phi(y)\rangle_H]|$ from Gopalan et al. (2022b), decision calibration is exactly a special case with weight class $W_{\text{dec}}=\{w_{\ell,\ell'}:w_{\ell,\ell'}(p(x))=\sum_a r_\ell(a)\tilde k_{f_p,\ell'}(x,a)\}$. This allows the use of iterative templates for weighted calibration.

However, the infinite-dimensional setting poses a challenge: the update direction $w_{\ell_t,\ell'_t}$ containing $r_{\ell_t}(a)$ cannot necessarily be written as a linear combination of $\phi(y)$, making it impossible to use the representer property for loss estimation. The authors resolve this with **implicit patching**: by replacing $\ell_t$ with $r_{\ell^*_t}(a)=R_1\mathbb{E}[(\phi(y)-p_t(x))\tilde k_{\ell'_t}(x,a)]/\|\cdot\|_H$, the violation becomes more severe (benefiting the auditor), and $r_{\ell^*_t}(a)$ naturally becomes a linear combination of $\phi(y)$ (approximated via empirical expectation). Thus, the predictor always maintains a finite linear representation of the form $p_t(x)=\sum_{i=1}^{N_t}\alpha_{ti}(x)\phi(y_{ti})$ (Proposition 5.1). One only needs to track the coefficients $\alpha_{ti}$ and outcomes $y_{ti}$ to compute $f_{p_t}$ and complete the patching without explicitly touching the infinite-dimensional $p_t$.

### Loss & Training
DimFreeDeCal iterations use step size $\eta$ to update $p_{t+1}(x)\mapsto p_t(x)+\sum_a d_{ta}\tilde k_{\ell'_t}(x,a)$, where $d_{ta}=\eta R_1\hat{\mathbb{E}}[(\phi(y)-p_t(x))\tilde k_{\ell'_t}(x,a)]/\|\hat{\mathbb{E}}[(\phi(y)-p_t(x))\tilde k_{\ell'_t}(x,a)]\|_H$, followed by a projection back to $B(R_2)$. This is an "audit-repair" cycle without an explicit loss function, with the stopping criterion being the decision calibration error falling below $\epsilon$.

## Key Experimental Results
This is a purely theoretical work with no empirical experiments; "experimental results" are presented as provable **complexity bounds**.

### Main Results

| Result | Setting | Key Bound | Implication |
|------|------|--------|------|
| Theorem 3.1 (Lower Bound) | Deterministic Response, $\phi(y)=y$, $|A|=2$ | Audit needs $n\ge\Omega(\sqrt d)$ | Dimension-free algorithms impossible under hard-max |
| Theorem 4.1 (Uniform Conv.) | Quantal Response, RKHS | Error $\le O\!\big(\tfrac{C_1\log(C_2n)+\log(1/\delta)}{\sqrt n}\big)$, $C_1,C_2$ excl. $m$ | Auditing generalization bound is dimension-independent |
| Theorem 4.2 (ERM Auditor) | Quantal Response | $n\ge\tilde O(|A|^3\beta^4R_1^6R_2^6\epsilon^{-2})$ | Provides a realizable $\epsilon$-auditor |
| Theorem 5.1 (Main Algorithm) | DimFreeDeCal | $T=O(R_1^2R_2^2/\epsilon^2)$ rounds, $\tilde O(|A|^3R_1^8R_2^8/\epsilon^4)$ samples | Dimension-free, no increase in MSE |

### Complexity Comparison with Zhao et al. (2021)

| Method | Sample Complexity (w.r.t. $\epsilon$) | Dimension Dependency | Loss Class |
|------|------------------------------|----------|--------|
| Zhao et al. (2021) (finite sample variant) | $\tilde O(1/\epsilon^6)$ | Depends on $m$ | Linear |
| **Ours DimFreeDeCal** | $\tilde O(1/\epsilon^4)$ | **Independent of $m$** | Piecewise Linear / Cobb–Douglas / Lipschitz diff., etc. |

### Key Findings
- **Smoothness is the watershed**: For the same auditing problem, deterministic response is blocked by an $\Omega(\sqrt m)$ lower bound, while quantal response yields bounded covering numbers and complexity decoupled from $m$—showing dimension dependency is not inherent to the problem but caused by the discontinuity of the hard-max decision rule.
- **Post-processing does not lose accuracy**: Theorem 5.1 guarantees that the $\ell_2$ error of the output predictor does not exceed that of the initial predictor: $\mathbb{E}[\|p_T(x)-\phi(y)\|_H^2]\le\mathbb{E}[\|p_0(x)-\phi(y)\|_H^2]$. "Calibration" is a free addition to any existing predictor.
- **Superior $\epsilon$ dependency**: Even comparing only the linear case, the $1/\epsilon^4$ of this paper outperforms the $1/\epsilon^6$ complexity obtained by adapting Zhao et al.'s algorithm to finite samples.

## Highlights & Insights
- **Projection pseudo-metric is the highlight**: Converting the "covering number of high-dimensional vector sets" into a finite covering number on a Hilbert ball by "projecting along random prediction directions" is the core trick to bypass infinite dimensions. This can be transferred to other statistical learning problems involving high-dimensional features and smooth decisions.
- **Negative lower bound as a design guide**: Proving hard-max is impossible and then precisely switching to quantal response—the lower bound is not just "bad news," it identifies which assumption (smoothness of the decision rule) needs to change.
- **Decision calibration = Special case of weighted calibration**: This equivalence connects a seemingly new objective to the existing weighted calibration framework, allowing the reuse of iterative post-processing templates.
- **Implicit patching enables infinite-dimensional computation**: By maintaining $p_t(x)=\sum_i\alpha_{ti}(x)\phi(y_{ti})$ as a finite linear representation and only tracking coefficients and sample points, updates in infinite-dimensional space are grounded in finite computation—a beautiful application of the kernel method’s representer property.

## Limitations & Future Work
- **Dependency on norm $R_1$ and temperature $\beta$**: The sample complexity includes factors like $R_1^8R_2^8\beta^4$. For loss classes that can only be approximated by large norm functions or require very large $\beta$ (to approximate hard-max), the constants may be very high; dimension independence is traded for dependency on other parameters.
- **Quantal response as a modeling assumption**: The dimension-free guarantee only holds when the decision-maker acts according to smoothed quantal response. When they actually use deterministic optimal response, these results do not directly apply, and the $\Omega(\sqrt m)$ lower bound suggests that path is inherently difficult.
- **Purely theoretical, no empirical verification**: The paper lacks numerical experiments. The number of iterations on real data, constant sizes, and sensitivity to $\beta$ remain untested.
- **Future Directions**: Tightening the dependency on $R_1, R_2, \beta$ or characterizing the trade-off of "how well a quantal response with a specific $\beta$ can approximate a deterministic decision" would make these guarantees more practical for deployment.

## Related Work & Insights
- **vs. Classic Full Calibration (Foster–Vohra; Gopalan et al. 2024a)**: Full calibration requires unbiasedness for all output values, leading to exponential sample complexity in high dimensions. This paper follows Zhao et al.'s weaker concept of decision calibration, which only requires unbiasedness on decision-relevant events, compressing complexity to polynomial and further achieving dimension independence. The $\Omega(\sqrt m)$ lower bound proof in this paper draws from the indistinguishability arguments of Gopalan et al. (2024a), but their work targets stronger full calibration; this paper's distribution construction utilizes the geometry of optimal response regions (half-spaces).
- **vs. Zhao et al. (2021)**: They pioneered decision calibration but only for linear losses, with sample complexity depending on outcome dimension. This paper generalizes it to a wide range of nonlinear losses in RKHS, achieving dimension independence through quantal response and projection pseudo-metrics, while improving $\epsilon$ dependency from $1/\epsilon^6$ to $1/\epsilon^4$.
- **vs. Weighted Calibration (Gopalan et al. 2022b)**: The patching framework here follows the template of weighted calibration, but while they patched in finite-dimensional settings, this paper handles (potentially infinite-dimensional) prediction spaces, breaking limitations via implicit patching and finite linear representations.
- **vs. Quantal Response Models (McFadden; McKelvey–Palfrey)**: Quantal response originated as an economics tool for modeling bounded rationality. This paper treats its smoothness as a statistical advantage—smoothness ensures Lipschitz calibration gap functions and bounded covering numbers, supporting dimension-free guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First lower bound for decision calibration + first dimension-free nonlinear decision calibration algorithm; both the problem and techniques are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical without experiments, but the theorem system (lower bound, uniform convergence, auditor, main algorithm) is complete and self-consistent.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation following the "lower bound → changing assumptions → positive algorithm" line, with sufficient technical details.
- Value: ⭐⭐⭐⭐ Extending decision calibration from linear to broad nonlinear losses has practical significance for trustworthy prediction in high-risk decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs](an_improved_model-free_decision-estimation_coefficient_with_applications_in_adve.md)
- [\[ICLR 2026\] Decision Aggregation under Quantal Response](decision_aggregation_under_quantal_response.md)
- [\[ICLR 2026\] A Faster Parameter-Free Regret Matching Algorithm](a_faster_parameter-free_regret_matching_algorithm.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](../../ICML2026/learning_theory/expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[ICLR 2026\] Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](deep_flexqp_accelerated_nonlinear_programming_via_deep_unfolding.md)

</div>

<!-- RELATED:END -->
