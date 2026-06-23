---
title: >-
  [Paper Note] Efficiently Learning Drifting Halfspaces with Massart Noise
description: >-
  [ICML 2026][learning_theory][Paper Note] In online learning scenarios where the distribution drifts over time and labels are corrupted by Massart noise, this paper provides the first **polynomial-time** algorithm for learning $\gamma$-margin halfspaces with an error of $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$. Using low-degree polynomial lower bounds, it further
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: f4a0960a22322a1e
---
# Efficiently Learning Drifting Halfspaces with Massart Noise

**Conference**: ICML2026  
**arXiv**: [2606.11149](https://arxiv.org/abs/2606.11149)  
**Code**: None (Purely theoretical)  
**Area**: Learning Theory  
**Keywords**: Distribution Drift, Halfspace Learning, Massart Noise, Information-Computation Gap, Low-degree Polynomial Lower Bounds

## TL;DR
In online learning scenarios where the distribution drifts over time and labels are corrupted by Massart noise, this paper provides the first **polynomial-time** algorithm for learning $\gamma$-margin halfspaces with an error of $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$. Using low-degree polynomial lower bounds, it further demonstrates that the $\Delta^{1/3}$ exponent is unavoidable for efficient algorithms, thereby revealing an information-computation gap.

## Background & Motivation
**Background**: Classical statistical learning theory assumes training samples are drawn from a fixed distribution and testing occurs on the same distribution. However, in applications such as financial forecasting, consumer preferences, weather modeling, or language models, the data distribution **drifts** over time—models are trained on historical snapshots but deployed on new data. This is formalized as "binary classification under distribution drift," where the Total Variation (TV) distance between distributions in adjacent rounds is bounded by a drift rate $d_{\mathrm{TV}}(D^{(t)},D^{(t+1)})\le\Delta$. The learner operates online, outputting a hypothesis in each round before receiving a noisy sample.

**Limitations of Prior Work**: Early research (HLO91, Bar92, HL94, etc.) characterized **statistically optimal** errors—$\Theta((d\Delta)^{1/2})$ in the realizable case and $\Theta((d\Delta)^{1/3})$ in the agnostic case. However, these results rely on Empirical Risk Minimization (ERM), which is **generally exponential-time**. In other words, while the statistical limits of "how well we can learn" are known, the computational limits of "how well efficient algorithms can learn" remain unclear. For halfspaces, the most fundamental class, existing efficient algorithms either suffer from poor error bounds (e.g., HL94's $\tilde{O}(d\sqrt{\Delta})$, lacking a factor of $\sqrt{d}$) or require stringent marginal distribution assumptions (e.g., CMEDV10 and HKY15 require samples to be uniformly distributed on the unit sphere).

**Key Challenge**: Drift itself does not increase the "informational" threshold for learnability (the statistically optimal error under Massart/RCN remains in the $\Delta^{1/2}$ range, comparable to the realizable case). However, it is unknown whether **efficient** algorithms can achieve this statistical optimum. Furthermore, under unstructured marginal distributions, a small $d_{\mathrm{TV}}$ does not necessarily imply a small geometric distance between the true weight vectors $w^{*(i)}$ and $w^{*(i+1)}$, which presents a major obstacle to porting noise-free analyses.

**Goal**: To design polynomial-time (in $d, 1/\gamma, 1/\Delta$) drifting halfspace learners under **minimal distribution assumptions** (only requiring a margin $\gamma$ for the target halfspace) and **Massart label noise**, and to identify the error limits achievable by efficient algorithms.

**Key Insight**: Massart noise (where each label is flipped with an unknown probability $\eta(x)\le\eta<1/2$) is selected because efficient learners exist for the drift-free case (DGT19), whereas agnostic (adversarial) noise is proven computationally hard even with a margin. The authors adapt the Leaky ReLU gradient from DKTZ25, designed for fixed halfspaces, and develop a strategy to migrate it to the drifting setting.

**Core Idea**: An "Online Projected Gradient Descent + Leaky ReLU noise-robust gradient" scheme is executed in epoch-based blocks. The technical linchpin is a **Regret-to-Error Lemma**, which bypasses the difficulty of tracking the drift of $w^*$ and instead bounds the prediction error directly through the "per-round reward plus a drift error term," pinning the efficient algorithm's error at $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$.

## Method

### Overall Architecture
The algorithm `DriftedMassart` (Algorithm 1) operates in units of **epochs of length $W$**. Within the $i$-th epoch, the learner uses the hypothesis $\hat{h}^{(i-1)}$ learned in the previous epoch for predictions while collecting all noisy samples into a set $S^{(i)}$. At the end of the epoch ($t=iW$), it calls the subroutine `DriftPerceptron` (Algorithm 2) to relearn a halfspace on $S^{(i)}$, which serves as the hypothesis for the next epoch. The epoch length is critical and is set to $W=2\lfloor\Delta^{-2/3}\log(1/\Delta)\rfloor$.

Internally, `DriftPerceptron` splits $S^{(i)}$ into two halves: the first half $S_1$ is used for **online updates** of the weight vector $w$ (producing a trajectory $w^{(1)}, w^{(2)}, \dots$ via projected gradient descent), and the second half $S_2$ is used to **select the hypothesis with the minimum empirical error** from this trajectory. The technical core lies not in the algorithm's form (which is online PGD with a specific gradient) but in the **analysis**: proving that this trajectory contains a hypothesis with an error as low as $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$ and that it can be reliably selected using $S_2$.

The total error budget is partitioned into three parts: optimization error $\tfrac{1}{2T\mu}+\tfrac{2\mu}{\gamma^2}$, drift error $O(T\Delta/\gamma)$, and concentration error $\sum_i\xi_i$. The first two are balanced by tuning $W$ and the step size $\mu=\gamma/\sqrt{T}$, while the third is controlled via martingale concentration. All three are suppressed to $\tilde{O}(\Delta^{1/3}/\gamma)$ when $T=\tilde{\Theta}(\Delta^{-2/3})$.

### Key Designs

**1. Epoch Block Partitioning + Drift-Estimation Trade-off: Balancing "More Samples for Better Estimation" and "Older Samples for Higher Distortion"**

The fundamental tension in drift learning is that reducing estimation/concentration error requires more historical samples, but older samples deviate further from the current target distribution, increasing drift error. This paper quantifies this trade-off via the epoch length $W$. Within an epoch of $T=W/2$ samples, the optimization error $\sim 1/(\sqrt{T}\gamma)$ decreases as $T$ increases, while the drift error $\sim T\Delta/\gamma$ increases. Equating the two yields $T=\Theta(\Delta^{-2/3})$, thus $W=\Theta(\Delta^{-2/3}\log(1/\Delta))$. This explains the arithmetic origin of the $\Delta^{1/3}$ exponent: it is the value at the equilibrium point between $1/\sqrt{T}$ and $T\Delta$.

**2. Leaky ReLU Noise-Robust Gradient + Online PGD: Ensuring Updates Directional Correctness under Massart Noise**

For each sample $(x,y)$, a "gradient" vector is constructed:
$$g(w^{(i)};x,y)=\frac{\big((1-2\eta)\,\mathrm{sign}(w^{(i)}\cdot x)-y\big)\,x}{\max\{|w^{(i)}\cdot x|,\gamma\}},$$
followed by a projected gradient descent step $w^{(i+1)}=\mathrm{proj}_{\mathbb{B}(1)}(w^{(i)}-\mu g)$. This $g$ can be viewed as the gradient of the Leaky ReLU loss $\ell_\eta(w;x,y)=(1-2\eta)|yw\cdot x|+yw\cdot x$, weighted by $\max\{|w\cdot x|,\gamma\}^{-1}$. The weighting ensures different penalties for misclassified samples at various margins, preventing samples near the decision boundary from dominating the update. The $(1-2\eta)$ factor is key to handling Massart noise: it cancels out the expected bias from the flip probability $\le\eta$, ensuring the expected update direction points toward the true halfspace.

**3. Regret-to-Error Lemma: Bypassing "$w^*$ Drift Tracking" to Control Per-round Prediction Error**

The primary hurdle in extending fixed-halfspace analysis to the drifting setting is that under unstructured marginals, $d_{\mathrm{TV}}(D^{(i)},D^{(i+1)})$ being small does **not** imply $\|w^{*(i)}-w^{*(i+1)}\|$ is small; the true weights can fluctuate wildly. Prior works (CMEDV10, HKY15) relied on the uniform spherical assumption to bound weight changes. This paper circumvents this via **Lemma 3.1**, which proves the existence of a bounded function $F_i(w^*,w,x)$ such that:
$$\mathbf{E}_{y\sim D^{(i)}\mid x}[g(w;x,y)\cdot(w-w^*)]\ge 2(\mathrm{err}^{(i)}(w,x)-\eta)-F_i,$$
where $|F_i|\le 1/\gamma$ and $\mathbf{E}_x F_i=O(\Delta m/\gamma)$. The intuition is that even if $w^*$ drifts significantly, the **per-round excess prediction error $(\mathrm{err}-\eta)$ can be bounded by that round's expected reward term plus a "drift error" $F_i$**. Thus, tracking the geometric evolution of $w^*$ is unnecessary. The expectation of $F_i$ is further attributed to two factors: changes in the **disagreement region** $\{x:(w^*\cdot x)(w^{*(i)}\cdot x)<0\}$ and changes in the **noise rate** $\eta_i(x)-\eta_{i+m}(x)$. Both grow only with the total drift $m\Delta$.

**4. Information-Computation Gap: Using Low-degree Polynomial Lower Bounds to Prove the Unavoidability of $\Delta^{1/3}$**

The algorithmic error $\Delta^{1/3}$ is qualitatively **worse** than the statistically optimal $\Delta^{1/2}$, which might suggest a loose analysis. However, this paper proves this is an intrinsic cost for efficient algorithms. In the specific case of Random Classification Noise (RCN, where $\eta(x)\equiv\eta$), the authors establish a lower bound (Theorem 1.2) for the **low-degree polynomial test**—a widely studied computational model covering many learning algorithms. Under the low-degree hardness conjecture, no polynomial-time algorithm can achieve an error of $\mathrm{opt}_T+\Delta^{1/3}\gamma^{-1/6}$ across a family of instances. This clean conclusion shows that while bounded noise (Massart/RCN) is information-theoretically equivalent to the realizable case ($\Delta^{1/2}$), it is **computationally** pushed to the limits of the agnostic case ($\Delta^{1/3}$).

### Loss & Training
The algorithm uses no trainable parameters in the traditional sense; "training" refers to the selection of hyperparameters for online PGD: step size $\mu=\gamma/\sqrt{T}$, using the first half $S_1$ of an epoch for the update trajectory, and selecting the optimal hypothesis from the second half $S_2$ based on empirical error $\hat{\mathrm{err}}(\hat{h}_j)=\frac{1}{|S_2|}\sum_{(x,y)\in S_2}\mathbb{1}\{\hat{h}_j(x)\ne y\}$. Hoeffding’s inequality and a union bound ensure that the empirical error reliably identifies a high-quality hypothesis with probability at least $0.9$. For RCN, since there exists some $\eta^*$ such that $\mathrm{opt}_t\le\eta^*+O(\Delta W)$, repeatedly running Algorithm 1 with different $\eta$ values upgrades the guarantee from $\eta+\dots$ to $\mathrm{opt}_t+\tilde{O}(\Delta^{1/3}/\gamma)$ (Theorem 3.1).

## Key Experimental Results

### Main Results: Error Guarantees of Efficient Algorithms

| Setting | Our Efficient Alg. Error | Prev. Best (Efficient) | Description |
|------|------------------|------------------|------|
| Massart Drift + $\gamma$-margin | $\eta+\tilde{O}(\Delta^{1/3}/\gamma)$ | None (First with noise guarantee) | Theorem 1.1, $T=\tilde\Omega(\Delta^{-2/3})$ |
| RCN Drift + $\gamma$-margin | $\mathrm{opt}_t+\tilde{O}(\Delta^{1/3}/\gamma)$ | None | Theorem 3.1 |
| Realizable + $\gamma$-margin | $\tilde{O}(\sqrt{\Delta}\,\gamma^{-3/2})$ | $\tilde{O}(\sqrt{\Delta}\,\gamma^{-2})$ (HL94) | Margin dependency improved from $\gamma^{-2}$ to $\gamma^{-3/2}$ |
| Realizable (Gen. VC dim $d$) | $\tilde{O}(d\sqrt{\Delta})$ (HL94) | — | Baseline for comparison (ERM is statistical opt) |

### Information vs. Computation: The $\Delta$-Scale Gap

| Noise Model | Statistically Optimal (Exp. Time) | Efficient Alg. (Upper + Lower Bound) | Conclusion |
|----------|----------------------------|------------------------------|------|
| Realizable | $\Theta((d\Delta)^{1/2})$ | $\Delta^{1/2}$ scale | No gap |
| Massart / RCN | $\tilde\Theta((d\Delta)^{1/2})$ (Thm 2.1/2.2) | $\Delta^{1/3}$ (Thm 1.1 upper; Thm 1.2 lower) | **Information-Computation Gap exists** |
| Agnostic | $\Theta((d\Delta)^{1/3})$ | No efficient alg. even with margin (Dan16) | Computationally unlearnable |

### Key Findings
- The $\Delta^{1/3}$ exponent is not an artifact of loose analysis but a consequence of balancing $1/\sqrt{T}$ and $T\Delta$; this is confirmed by low-degree polynomial lower bounds as unavoidable for efficient algorithms.
- Bounded label noise (Massart/RCN) is information-theoretically as easy as the realizable case ($\Delta^{1/2}$), but efficient algorithms are forced to pay the price of the agnostic setting ($\Delta^{1/3}$).
- Under unstructured marginals, the Regret-to-Error Lemma is the essential tool for analysis, as one cannot assume $\|w^{*(i)}-w^{*(i+1)}\|$ remains small.

## Highlights & Insights
- **Replacing Drift Tracking with Regret-Bounded Error**: Lemma 3.1 is the most ingenious step, enabling drift analysis under unstructured marginals. This approach is transferable to other online learning problems involving drift and noise.
- **Transparent Origin of Epoch Length**: The exponent $\Delta^{-2/3}$ is derived directly from the equilibrium between optimization error and drift error.
- **Symmetry between Upper and Lower Bounds**: By providing matching lower bounds using a standard computational model, the paper establishes the algorithm as essentially optimal in scale.

## Limitations & Future Work
- The guarantee takes the form $\eta+\tilde{O}(\dots)$ rather than $\mathrm{opt}_t+\tilde{O}(\dots)$ for general Massart noise, reflecting the known computational hardness of achieving error below $\eta-o(1)$ even without drift (NT22).
- The margin dependency remains $1/\gamma$ (for noise) or $\gamma^{-3/2}$ (for realizable). Whether this is optimal or can be improved to $\gamma^{-1}$ remains an open question.
- The lower bounds depend on the low-degree hardness conjecture; upgrading these to unconditional bounds or broader algorithm classes is a natural next step.
- Only halfspaces are addressed; extending these results to more general geometric concept classes (e.g., polytopes, low-degree PTFs) remains open.

## Related Work & Insights
- **vs. HL94 (LP-based efficient drift learning)**: They achieved $\tilde{O}(d\sqrt{\Delta})$ in the realizable case but did not handle noise. Ours handles Massart noise and improves the realizable dependency to $\gamma^{-3/2}$.
- **vs. CMEDV10 / HKY15 (Drift learning under uniform sphere assumptions)**: They controlled $\|w^{*(i)}-w^{*(i+1)}\|$ using specific distributions. Ours uses the Regret-to-Error Lemma under minimal distribution assumptions.
- **vs. DGT19 / DKTZ25 (Fixed Massart halfspace learning)**: While borrowing their Leaky ReLU gradients, this work uniquely addresses the challenges of shifting targets where no single ground-truth $w^*$ exists.
- **vs. Lon99 ($\Theta((d\Delta)^{1/3})$ statistical optimality for agnostic drift)**: This paper connects the dots by showing bounded noise forces efficient algorithms into the $\Delta^{1/3}$ scaling typically associated with agnostic settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First efficient learning guarantee under both drift and label noise, with a matching information-computation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Purely theoretical, but the matrix of results is comprehensive and well-matched.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear decomposition of error budgets and intuitive explanation of the $\Delta^{1/3}$ origin.
- Value: ⭐⭐⭐⭐⭐ Provides a computational benchmark for learnability on non-stationary data, fundamental for drift learning theory.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](robustness_of_mixtures_of_experts_to_feature_noise.md)
- [\[ICML 2026\] Performative Learning Theory](performative_learning_theory.md)
- [\[ICML 2026\] Bandit Social Learning with Exploration Episodes](bandit_social_learning_with_exploration_episodes.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](towards_optimal_robustness_in_learning-augmented_paging.md)

</div>

<!-- RELATED:END -->
