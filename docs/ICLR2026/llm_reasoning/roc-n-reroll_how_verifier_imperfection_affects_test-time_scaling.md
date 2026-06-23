---
title: >-
  [Paper Note] ROC-n-Reroll: How Verifier Imperfection Affects Test-Time Scaling
description: >-
  [ICLR 2026][LLM Reasoning][Best-of-N] This paper utilizes the classical ROC curve to provide a precise theoretical characterization of how Best-of-N and Rejection Sampling scale under imperfect verifiers. It proves two counter-intuitive conclusions: Rejection Sampling outperforms Best-of-N at fixed compute budgets, and high-compute performance cannot be ex
tags:
  - ICLR 2026
  - LLM Reasoning
  - Best-of-N
date: 2026-05-08
content_hash: cb5aafa5d650b1d0
---
# ROC-n-Reroll: How Verifier Imperfection Affects Test-Time Scaling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3Gy5mmyuxn](https://openreview.net/forum?id=3Gy5mmyuxn)  
**Code**: https://github.com/socialfoundations/roc-n-reroll  
**Area**: LLM Reasoning  
**Keywords**: Test-time scaling, Verifier, ROC curve, Rejection sampling, Best-of-N  

## TL;DR
This paper utilizes the classical ROC curve to provide a precise theoretical characterization of how Best-of-N and Rejection Sampling scale under imperfect verifiers. It proves two counter-intuitive conclusions: Rejection Sampling outperforms Best-of-N at fixed compute budgets, and high-compute performance cannot be extrapolated from low-compute observations.

## Background & Motivation
**Background**: Test-time scaling has become a mainstream approach for improving LLM performance following pre-training, gaining explosive popularity since OpenAI o1. The "resampling" route is the most direct: generating multiple candidate answers for the same query and selecting the best one using a **verifier** (a mechanism that scores answers). The two most common resampling methods are Best-of-N (BoN, sampling $N$ candidates and picking the highest score) and Rejection Sampling (RS, repeatedly sampling until a candidate exceeds a verifier score threshold).

**Limitations of Prior Work**: Previous theoretical analyses almost exclusively assumed **oracle verifiers**, where BoN accuracy simply equals pass@N. However, in reality, perfect verifiers only exist in specific domains like code (unit tests) or math (standard numerical answers), and even then, false negatives/positives occur—unsafe code might pass static tests, and flawed reasoning might yield the correct final digit. More generally, LLMs are increasingly used as verifiers themselves, whose accuracy is far from perfect.

**Key Challenge**: To what extent does verifier imperfection determine the scaling curves of these two methods? This relationship has lacked theoretical grounding. Verifiers involve a trade-off between two types of errors: false negatives (rejecting correct answers) and false positives (accepting incorrect answers). This trade-off is precisely what the Receiver Operating Characteristic (ROC) curve characterizes in classification—yet test-time scaling had not been linked to ROC curves until now.

**Goal**: For a fixed query, precisely characterize how "accuracy vs. compute" curves for RS and BoN depend on verifier imperfection, and determine whether performance at high compute can be extrapolated from low-compute observations.

**Core Idea**: Encode the full error trade-off of any verifier for a specific query into its **ROC curve** $T(F)$ (true positive rate as a function of false positive rate). The paper proves that the instance-level accuracy of RS and BoN depends implementation details **only** through the "generator's base accuracy $\pi$ + the verifier's ROC curve"—all implementation specifics are absorbed into this single curve.

## Method

### Overall Architecture
This is a purely theoretical analysis. It does not propose a new algorithm but establishes a precise mathematical characterization of "verifier imperfection → test-time scaling potential." The argumentation chain follows: abstracting the verifier using ROC curves (formal setup), deriving closed-form "accuracy-compute" expressions for RS and BoN, and finally extracting three key conclusions—the superiority of RS at fixed compute, the convergence of both methods at infinite compute, and the impossibility of performance extrapolation.

The formal setup fixes a query $q$ and a generator $g_{\text{base}}$ with a single-sample success probability $\pi := \text{ACC}(g_{\text{base}}) = P[y(X)=1]$, where $y(X)\in\{0,1\}$ is the unknown correctness label. A verifier is a scoring function $f:X\mapsto[0,1]$, which yields a classifier $h_\tau(X)=\mathbb{I}[f(X)\ge\tau]$ via thresholding. Each classifier has a true positive rate $T = P[h(X)=1\mid y(X)=1]$ and a false positive rate $F = P[h(X)=1\mid y(X)=0]$. Connecting the maximum $T$ achievable for any given $F$ defines the ROC curve $T(F)$. All conclusions in this paper take this curve as the sole input.

### Key Designs

**1. ROC Curve: Compressing "Verifier Quality" into a Single Curve**

This is the foundation. The paper proves (Proposition 1, 4) that for a fixed query, the accuracy of RS and BoN depends **only** on the generator's initial accuracy $\pi$ and the verifier's ROC curve $T(F)$, remaining agnostic to all other implementation details (e.g., which model is the verifier or its scoring mechanism). The value of this step lies in mapping the vague concept of "verifier imperfection" onto an object with well-understood geometric properties: the bottom-left corner ($F\approx0$, high threshold, corresponding to high compute) and top-right corner ($F\approx1$, low threshold, corresponding to low compute) encode behaviors at either end of the compute spectrum. Whether the curve is **concave** determines if scaling is monotonic or subject to "overoptimization."

**2. Rejection Sampling: Precision and Compute Determined by Local Slopes**

Since RS continues sampling until acceptance, its accuracy is simply the **precision** (positive predictive value) of the classifier $h_F$. The paper provides the closed-form expression:

$$\bar{A}_f(F) = \text{ACC}(g_{h_F}) = \frac{T(F)\cdot\pi}{T(F)\cdot\pi + F\cdot(1-\pi)}$$

The average compute (expected number of samples) follows a geometric distribution: $C(F)=\dfrac{1}{T(F)\cdot\pi + F\cdot(1-\pi)}$. By substituting $F$ for compute $C$, one obtains the accuracy-compute curve $A(C)$. A key finding (Proposition 1) is that the slope of $A(C)$ is proportional to the slope of the ROC curve $T'(F)$; thus, **a concave ROC curve implies monotonic RS scaling**. Specifically, scaling at **low compute** ($F\approx1$) is dictated by $T'(1)$, while the **infinite compute limit** ($F\to0$) is locked by the slope at the origin $T'(0)$ (Proposition 2):

$$\lim_{C\to\infty}A(C) = \frac{T'(0)\cdot\pi}{T'(0)\cdot\pi + (1-\pi)}$$

This means the ultimate ceiling of RS depends entirely on whether the verifier can reliably identify rare correct answers near the origin.

**3. Best-of-N: Global Integration and Inferiority to RS at Fixed Compute**

Unlike RS, which is determined by local geometry, BoN selects the highest score among $N$ samples, meaning its accuracy depends on the **global** shape of the ROC curve. Under a regularity assumption on score distributions, the paper expresses BoN accuracy as an integral along the ROC curve (Proposition 4):

$$\text{ACC}(g_N) = 1 - (1-\pi)^N\int_0^1 \psi(F)^{N-1}\,dF,\quad \psi(F)=(1-\pi)(1-F)+\pi(1-T(F))$$

For an oracle verifier where $T(F)=1$, this collapses to the well-known pass@N $=1-(1-\pi)^N$. Two further results serve as practical guidelines: (i) **RS dominates** at fixed compute (Proposition 5). For concave ROC curves, setting the RS threshold such that the expected sample count equals $N$ yields RS accuracy $\ge$ BoN. (ii) However, **both converge at infinite compute** (Theorem 1). The high-compute limit for BoN is identical to RS, as the integral for large $N$ is dominated by the region near the origin.

**4. Impossibility of Extrapolation: Low-Compute Cannot Predict High-Compute**

RS performance at low and high compute is determined by $T'(1)$ and $T'(0)$, respectively. Since the geometry at these two ends is **independent**, one cannot infer high-compute performance from low-compute observations without knowing the full ROC curve (Proposition 3). Even with a concavity assumption, one can only guarantee performance will not degrade, but cannot distinguish between "saturating quickly" and "approaching perfection." This directly challenges the practice of fitting power laws to early scaling points for extrapolation.

### A Complete Example
Using Figure 1 (GSM8K Q58, generator Qwen3-1.7B): When using Qwen3 32B/14B/4B as verifiers, all three ROC curves overlap at the **top-right** but have different slopes at the **origin**. The results match the theory: at low compute, RS and BoN curves for all three verifiers are nearly identical, but they diverge significantly as compute increases. The verifier with the steeper slope at the origin climbs higher.

## Key Experimental Results

Experiments used Qwen3 and LLaMA series models as both generators and verifiers, validated on GSM8K and MATH500. Generations were 5-shot with temperature $t=1$. Verifiers provided Chain-of-Thought followed by a 0–10 score, averaged over 5 runs to get $f(X) \in [0,1]$.

### Main Results

| Experiment | Setting | Key Results |
|------|------|----------|
| Instance-level Prediction | GSM8K Q58 (Qwen3-1.7B Gen) | Theoretical curves almost perfectly match measured points. |
| Instance-level Prediction | GSM8K Q2 (LLaMA-3.2-3B Gen) | RS consistently outperforms BoN; absolute prediction error < 1pp. |
| Aggregated Performance | GSM8K / MATH500 | Controlled for compute, RS > BoN, but the gap disappears at high compute. |
| Capped RS | GSM8K / MATH500 | Capped RS matches Bo25 performance but requires significantly fewer average samples. |

### Key Findings
- **RS is consistently superior to BoN at fixed compute**: This holds at both instance and aggregate levels.
- **Low-compute overlap, high-compute divergence**: Confirms that high-compute performance cannot be predicted via extrapolation.
- **Capped RS (CRS) is a practical compromise**: By sampling with a fixed threshold up to a maximum (e.g., 25) and falling back to the best-so-far, CRS captures RS's compute efficiency while avoiding the "infinite sampling" risk of pure RS.

## Highlights & Insights
- **Repurposing the ROC curve for LLM test-time scaling** is an elegant "old tool, new problem" application. It absorbs implementation complexity into a single geometric object.
- **"Local vs. Global" Duality**: RS performance is local (slopes at endpoints), while BoN is global (integral), yet they share the same scaling ceiling.
- **Extrapolation Warning**: The paper provides a rigorous warning against fitting power laws to early scaling data, proving such fits are ungrounded for imperfect verifiers.

## Limitations & Future Work
- Theoretical conclusions are strictly proven for **individual queries**; aggregate behavior on datasets is an empirical observation without the same level of theoretical guarantee.
- Analysis is limited to RS and BoN, excluding more complex strategies like majority voting, beam search, or Process Reward Models (PRMs).
- Estimating the ROC curve requires significant sampling (1000 samples per query in this study).
- Exploring "customized verifiers" optimized specifically for the origin or the top-right of the ROC curve depending on the targeted compute budget.

## Related Work & Insights
- **vs. Brown et al. (2024)**: They analyze scaling under **perfect verifiers** (BoN = pass@N). This paper relaxes that assumption using ROC curves.
- **vs. Huang et al. (2025b)**: They use MSE between scores and ground truth to provide **bounds** on BoN; this paper provides an **exact characterization** for binary labels.
- **vs. Song et al. (2025)**: They empirically study RS; this paper provides the precise theory and proves RS's compute efficiency advantage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CaTS: Calibrated Test-Time Scaling for Efficient LLM Reasoning](cats_calibrated_test-time_scaling_for_efficient_llm_reasoning.md)
- [\[ICLR 2026\] T1: Tool-Integrated Verification for Test-Time Compute Scaling in Small Language Models](t1_tool-integrated_verification_for_test-time_compute_scaling_in_small_language_.md)
- [\[ICLR 2026\] Strategic Scaling of Test-Time Compute: A Bandit Learning Approach](strategic_scaling_of_test-time_compute_a_bandit_learning_approach.md)
- [\[ICLR 2026\] Mode-conditioning unlocks superior test-time compute scaling](mode-conditioning_unlocks_superior_test-time_compute_scaling.md)
- [\[ICLR 2026\] Optimal Aggregation of LLM and PRM Signals for Efficient Test-Time Scaling](optimal_aggregation_of_llm_and_prm_signals_for_efficient_test-time_scaling.md)

</div>

<!-- RELATED:END -->
