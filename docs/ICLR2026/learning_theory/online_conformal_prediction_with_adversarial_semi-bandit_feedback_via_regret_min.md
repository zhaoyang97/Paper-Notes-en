---
title: >-
  [Paper Note] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization
description: >-
  [ICLR 2026][Learning Theory][Online conformal prediction] This paper reformulates the online conformal prediction problem—where the true label is revealed only if it falls within the prediction set—as an adversarial multi-armed bandit problem by treating threshold candidates as arms. By designing a specialized loss function that explicitly links regret to the miscoverage rate and adapting the EXP3.P algorithm into OCP-Unlock+, the authors provide the first long-term coverage…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Online Learning"
  - "Conformal Prediction"
  - "Online conformal prediction"
  - "adversarial semi-bandit feedback"
  - "EXP3.P"
  - "regret minimization"
  - "long-term coverage guarantee"
date: 2026-05-08
content_hash: 3aa6bd2765ea87f5
---

# Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RMWcdp5IUy](https://openreview.net/forum?id=RMWcdp5IUy)  
**Code**: TBD  
**Area**: Learning Theory / Online Learning / Conformal Prediction  
**Keywords**: Online conformal prediction, adversarial semi-bandit feedback, EXP3.P, regret minimization, long-term coverage guarantee

## TL;DR
This paper reformulates the online conformal prediction problem—where the true label is revealed only if it falls within the prediction set—as an adversarial multi-armed bandit problem by treating threshold candidates as arms. By designing a specialized loss function that explicitly links regret to the miscoverage rate and adapting the EXP3.P algorithm into OCP-Unlock+, the authors provide the first long-term coverage guarantee under adversarial data streams without relying on i.i.d. assumptions.

## Background & Motivation
**Background**: Conformal prediction (CP) is a model-agnostic uncertainty quantification method that constructs a "prediction set" for each input, guaranteeing that the set contains the true label with a user-specified probability (coverage guarantee). The online version (online conformal prediction) handles scenarios where data points arrive sequentially, providing long-term coverage guarantees even under adversarial data streams: after sufficient steps, the empirical coverage rate approaches the target level $1-\alpha$.

**Limitations of Prior Work**: Almost all existing online conformal prediction methods assume **full feedback**—where the true label $y_t$ is observed at every step. They use $y_t$ either to estimate quantiles or to evaluate miscoverage loss across multiple candidate prediction sets. However, in reality, labels are often unavailable: in human-in-the-loop systems, the label is revealed only when the prediction set contains the correct answer and is deemed worth checking by a human.

**Key Challenge**: Ge et al. (2025) proposed a **semi-bandit feedback** setting—where the true label $y_t$ is revealed only if it falls within the selected prediction set—but their coverage guarantee holds only under i.i.d. data streams. Thus, one must currently assume either full feedback or i.i.d. data; no existing work addresses both "restricted feedback" and "adversarial data streams," which are the most realistic constraints.

**Goal**: Design an algorithm under **adversarial semi-bandit feedback** such that the miscoverage rate $\mathrm{MC}(T)=\frac1T\sum_t m_t(\pi_t)$ satisfies $\mathrm{MC}(T)\le\alpha+\varepsilon(T)$ where $\varepsilon(T)\to0$, while simultaneously controlling the average prediction set size $\mathrm{Ineff}(T)=\frac1T\sum_t|\hat C_{\pi_t}(x_t)|$ (since outputting the entire label space $\mathcal Y$ would trivially achieve coverage).

**Key Insight**: The authors observe that the core decision in online conformal prediction is selecting a threshold $\pi\in[0,1]$ to parameterize the prediction set $\hat C_\pi(x)=\{\tilde y: f_t(x,\tilde y)\ge\pi\}$. By discretizing the continuous threshold space and treating each candidate threshold as an "arm," the problem becomes an adversarial multi-armed bandit problem—for which algorithms like EXP3.P already exist that provide high-probability sublinear regret under adaptive adversaries.

**Core Idea**: Replace "quantile estimation" with "adversarial bandit + regret minimization" for online conformal prediction, and translate the regret into a coverage guarantee through a custom-tailored loss function.

## Method

### Overall Architecture
The paper addresses the construction of prediction sets with long-term coverage guarantees on data streams that are adversarial and where labels are visible only when captured by the prediction set. The approach follows a "problem translation + algorithm adaptation" pipeline: first, the threshold space $[0,1]$ is discretized into $K$ candidates $\Pi$, with each candidate treated as an arm (**Problem Reformulation**). Next, a loss function $\ell_t(\pi)$ consisting of a miscoverage term and an inefficiency term is designed (**Loss Design**), ensuring the learner is penalized for both missed coverage and excessively large sets. Then, a learner-agnostic bridge lemma is proven (**Regret→Coverage**), showing that any learner achieving sublinear regret on this loss also achieves the target coverage. Finally, EXP3.P is modified to leverage additional information from semi-bandit feedback and threshold monotonicity, resulting in the faster-converging **OCP-Unlock+** algorithm.

### Key Designs

**1. Reformulating Online CP as an Adversarial Multi-armed Bandit**

The challenge is that continuous threshold spaces cannot directly use standard bandit algorithms. The authors discretize $[0,1]$ into $K=|\Pi|$ candidate thresholds, where each $\pi\in\Pi$ corresponds to a prediction set $\hat C_\pi$. Selecting a threshold each step is then equivalent to "pulling an arm" from $K$ candidates. Performance is measured by regret against the best fixed arm:

$$\mathrm{Reg}(T):=\sum_{t=1}^{T}\ell_t(\pi_t)-\min_{\pi\in\Pi}\sum_{t=1}^{T}\ell_t(\pi).$$

A key difference from typical adversarial bandits (Table 1 in the paper) is that while the adversary controls the samples $(x_t, y_t)$, the **loss function** $\ell_t:\Pi\to[\ell_{\min},\ell_{\max}]$ is designed by the learner. This allows the loss to be shaped specifically to favor coverage control for an **adaptive adversary**.

**2. Custom Loss Function for Conformal Prediction**

Monitoring miscoverage alone is insufficient; a "lazy" learner outputting $\pi_t=0$ (the whole $\mathcal Y$) would get $\mathrm{MC}(T)=0$ but provide uninformative sets. Thus, the loss must penalize both "missed coverage" and "set size":

$$\ell_t(\pi;\alpha):=d_t(\pi;\alpha)+a_t(\pi),$$

where the miscoverage term $d_t$ is designed based on the feedback $m_t(\pi)=\mathbb 1(y_t\notin\hat C_\pi(x_t))$ as:

$$d_t(\pi;\alpha)=m_t(\pi)+\mathbb 1(m_t(\pi)=0)\,(\alpha+\alpha(1-\alpha))+\mathbb 1(m_t(\pi)=1)\,(-\alpha(1-\alpha)).$$

The terms $g(\alpha)>0$ and $h(\alpha)<0$ adjust the loss in an $\alpha$-adaptive way. When coverage is successful ($m_t=0$), a small positive term is added; when it fails ($m_t=1$), it is subtracted. This ensures $d_t$ is smaller for successful thresholds, and the difference prioritized coverage as $\alpha$ decreases. The inefficiency term $a_t(\pi)$ penalizes large sets (e.g., using an $o(T)^{-1}$ term as shown in Eq. 7 of the paper).

**3. Bridge from Regret to Coverage**

Minimizing regret does not naturally equate to controlling $\mathrm{MC}(T)$. To bridge this, the authors prove a learner-agnostic lemma (Lemma 1): for any learner generating regret $\mathrm{Reg}(T)$ on the loss in Eq. 5 with $a_t(\pi) = o(T)^{-1}$, the following holds:

$$\mathrm{MC}(T)-\alpha\le\frac1T\mathrm{Reg}(T)+C_{\mathrm{MC}}(T),$$

where $C_{\mathrm{MC}}(T)$ is an $o(T)^{-1}$ constant. This implies that if regret is sublinear ($\mathrm{Reg}(T)/T\to0$), long-term coverage reaches at least $1-\alpha$.

**4. OCP-Unlock+: Leveraging Monotonicity to Accelerate Coverage**

Standard EXP3.P (termed OCP-Bandit) achieves sublinear regret but ignores the structural information in semi-bandit feedback. The key observation is that when $m_t(\pi_t)=0$ (successful coverage), the true label $y_t$ is revealed, allowing $m_t(\pi)$ to be evaluated for **all** $\pi\in\Pi$ (full unlocking). Even if $m_t(\pi_t)=1$ (failure), the monotonicity of $m_t(\pi)$ implies that for any $\pi\ge\pi_t$, $m_t(\pi)=1$, allowing evaluation for a subset (partial unlocking). The **unlocking set** is defined as:

$$\Pi_t(\pi_t):=\begin{cases}\Pi & \text{if } m_t(\pi_t)=0\\ \{\pi\in\Pi:\pi\ge\pi_t\} & \text{if } m_t(\pi_t)=1.\end{cases}$$

OCP-Unlock+ uses a biased gain estimator $\tilde g_t(\pi\mid\Pi_t(\pi_t))$ that adapts to the available feedback. For areas outside the unlocking set during failures, a **pseudo-gain** is used based on $\tilde\ell_t(\pi)$, constructed to preserve the monotonicity in the estimator. Theorem 1 provides the high-probability coverage deviation bound:

$$\mathrm{MC}(T)-\alpha\le\ell_{\mathrm{diff}}\!\left(\sqrt{\tfrac{C\ln K}{T}}+4.15\sqrt{\tfrac{K\ln K}{T}}+\sqrt{\tfrac{K}{T\ln K}\ln(\delta^{-1})}+2o(T)^{-1}\right)+C_{\mathrm{MC}}(T),$$

where $\ell_{\mathrm{diff}}=\ell_{\max}-\ell_{\min}$. Unlocking does not change the asymptotic order of regret but significantly speeds up the convergence to target coverage.

### Loss & Training
The core objective is the loss $\ell_t(\pi)=d_t(\pi)+a_t(\pi)$. The strategy update follows the EXP3.P structure using exponential weights and uniform mixing: $p_t(\pi)\propto(1-\gamma)\frac{\exp(\eta\tilde G_{t-1}(\pi))}{\sum_{\tilde\pi}\exp(\eta\tilde G_{t-1}(\tilde\pi))}+\gamma\frac1K$, where cumulative gain $\tilde G_t(\pi)$ uses the biased estimator on the unlocking set. Hyperparameters $\beta, \gamma, \eta$ are set as per Theorem 1.

## Key Experimental Results

### Main Results
Experiments were conducted on ImageNet (classification) and UCI Airfoil (regression) under both i.i.d. and non-i.i.d. settings. OCP-Unlock+ was compared against SPS (semi-bandit but i.i.d. only) and MVP (Oracle with full feedback).

| Task / Setting | Config | Long-term MC | Inefficiency Ineff |
|------|------|------|------|
| ImageNet i.i.d. | MVP (Oracle, full feedback) | Approx. $\alpha$ | Minimum |
| ImageNet i.i.d. | SPS | $\le\alpha$ | High |
| ImageNet i.i.d. | **OCP-Unlock+** | Approx. $\alpha$ | Medium |
| ImageNet non-i.i.d. (evolving scoring) | SPS | Fails to adapt | Volatile |
| ImageNet non-i.i.d. | **OCP-Unlock+** | Approx. $\alpha$ | Controlled |
| Airfoil Covariate Shift | SPS | Unreliable | — |
| Airfoil Covariate Shift | **OCP-Unlock+** | Conservative | High |

Key Conclusion: Under non-i.i.d. or covariate shift, **SPS's coverage guarantee collapses**, whereas OCP-Unlock+ consistently pulls the MC back to the target level, performing comparably to the full-feedback Oracle MVP.

### Ablation Study
- **Discretization Level $K$**: As $K$ increases, it affects both the convergence speed and the granularity of inefficiency (Experiment 3).
- **Feedback Utilization**: Moving from OCP-Bandit → OCP-Unlock → OCP-Unlock+ (utilizing full and partial unlocking) shows that more feedback leads to faster convergence to target coverage (Experiment 4).
- **Target $\alpha$**: The algorithm is robust to different coverage levels (Experiment 5).

### Key Findings
- **Unlocking is crucial for speed**: Fully utilizing structural information like monotonicity does not change the asymptotic regret order but drastically reduces the number of steps needed to reach the target coverage.
- **Robustness is the differentiator**: While all methods perform adequately under i.i.d. conditions, OCP-Unlock+ remains stable under adversarial shifts where i.i.d.-dependent methods like SPS fail.
- **Price of limited feedback**: OCP-Unlock+ is generally more conservative (larger set size) than MVP, which is the expected trade-off for ensuring coverage when labels are rarely seen.

## Highlights & Insights
- **Elegant Problem Translation**: Converting continuous CP into a discrete arm bandit problem allows the use of decades of cumulative research in sublinear-regret algorithms like EXP3.P.
- **Loss Design as a Lever**: Unlike standard bandits where the adversary sets the loss, here the learner designs the loss to bake "coverage priority" directly into the optimization objective.
- **Monotonicity as "Free" Feedback**: The fact that miscoverage is monotonic with respect to the threshold allows a "failure" to still provide useful information for a large subset of thresholds.
- **Regret-to-Target Paradigm**: Lemma 1 follows the paradigm of converting abstract regret bounds into domain-specific guarantees (like MC or FDR), providing a template for other online decision problems.

## Limitations & Future Work
- **Limitations**: The algorithm is context-free and does not utilize input-side covariate information. It is also more conservative than full-feedback methods under heavy drift.
- **Discretization**: Setting $K$ involves a trade-off; a large $K$ slows convergence, while a small $K$ limits threshold precision. There is currently no adaptive method for $K$.
- **Adversarial Strength**: The "adversarial" components in experiments are synthetic (scoring and covariate shifts). Real-world human-in-the-loop adversarial behavior may differ.
- **Future Work**: Extending this to contextual bandits, exploring continuous-arm bandit versions to remove the dependence on $K$, and tighter analysis of the pseudo-gain approximation error.

## Related Work & Insights
- **vs SPS (Ge et al., 2025)**: Both use semi-bandit feedback, but SPS requires i.i.d. data. This work is the first to provide guarantees under **adversarial** streams.
- **vs MVP (Bastani et al., 2022)**: MVP is the full-feedback Oracle. This work approaches MVP's coverage performance with significantly less information.
- **vs Online Gradient Descent (OGD) approaches**: Traditional online CP often uses OGD for quantile tracking under full feedback; this work uses the bandit framework, which is naturally suited for "revealed-on-demand" labels.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Bandit Learning in Matching Markets Robust to Adversarial Corruptions](bandit_learning_in_matching_markets_robust_to_adversarial_corruptions.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Bandit Learning in Matching Markets Robust to Adversarial Corruptions](bandit_learning_in_matching_markets_robust_to_adversarial_corruptions.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)

</div>

<!-- RELATED:END -->
