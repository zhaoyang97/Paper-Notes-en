---
title: >-
  [Paper Note] Fixing the Loose Brake: Exponential-Tailed Stopping Time in Best Arm Identification
description: >-
  [ICML2025][best arm identification] This paper reveals that classic fixed-confidence best arm identification algorithms (Successive Elimination, KL-LUCB) have non-zero probability events where they never stop. It proposes two schemes, FC-DSH and the meta-algorithm BrakeBooster, achieving the first guaranteed exponential tail decay for stopping time without losing instance-dependent complexity (only up to log factors).
tags:
  - "ICML2025"
  - "best arm identification"
  - "multi-armed bandit"
  - "fixed confidence"
  - "stopping time"
  - "exponential tail"
date: 2026-05-08
content_hash: 008d321c1a775017
---

# Fixing the Loose Brake: Exponential-Tailed Stopping Time in Best Arm Identification

**Conference**: ICML2025  
**arXiv**: [2411.01808](https://arxiv.org/abs/2411.01808)  
**Code**: To be confirmed  
**Area**: Others  
**Keywords**: best arm identification, multi-armed bandit, fixed confidence, stopping time, exponential tail

## TL;DR
This paper reveals that classic fixed-confidence best arm identification algorithms (Successive Elimination, KL-LUCB) have non-zero probability events where they never stop. It proposes two schemes, FC-DSH and the meta-algorithm BrakeBooster, achieving the first guaranteed exponential tail decay for stopping time without losing instance-dependent complexity (only up to log factors).

## Background & Motivation

### Best Arm Identification (BAI) in Multi-Armed Bandits

Under the fixed-confidence setting, an algorithm must stop as quickly as possible and output the best arm with a $\delta$-correctness guarantee. The stopping time $\tau$ is random, and existing literature focuses on two types of guarantees:

**Asymptotic Expected Sample Complexity** (AE): $\liminf_{\delta \to 0} \frac{\mathbb{E}[\tau]}{\ln(1/\delta)} \leq T_\delta^*$

**High-Probability Sample Complexity**: $\mathbb{P}(\tau \geq T_\delta^*) \leq \delta$

### Limitations of Prior Work

Both types of guarantees **fail to characterize the tail behavior of the stopping time**:

- **High-Probability Guarantee**: It only states that $\mathbb{P}(\tau \geq T^*) \leq \delta$, but with probability $\delta$, the algorithm might **never stop**.
- **AE Guarantee**: It hides non-asymptotic behavior, and $\tau$ can be heavy-tailed.

This paper theoretically and experimentally confirms that this concern is not merely hypothetical but real:

- **Theorem 2.4**: There exist instances for Successive Elimination where it never stops with probability $\Omega(\delta^{118})$.
- **Theorem 2.5**: There exist instances for KL-LUCB where it never stops with **constant probability**.

In experiments (3 arms, $\Delta=0.1$, $\delta=0.01$), a significant number of SE runs did not terminate within 30,000 steps and had already eliminated the best arm, indicating they would likely never stop.

## Method

### Formal Definition of Exponentially-Tailed Stopping Time

$$(T_\delta, \kappa)\text{-exponential stopping tail}: \quad \forall T \geq T_\delta, \quad \mathbb{P}(\tau \geq T) \leq \exp\left(-\frac{T}{\kappa \cdot \text{polylog}(T)}\right)$$

This property is strictly stronger than high-probability sample complexity and implies:
- High-probability stopping guarantee (Proposition 2.9(i))
- Finite expected stopping time (Proposition 2.9(ii))
- Almost sure stopping (Proposition 2.9(iii))

### Scheme 1: FC-DSH (Fixed-Confidence Doubling Sequential Halving)

**Design Motivation**: Combine Sequential Halving (SH) with the doubling trick and design careful stopping rules.

**Algorithm Flow**:
1. Iterate by phase $m = 1, 2, \ldots$, with budget $T_m = T_1 \cdot 2^{m-1}$ ($T_1 = \lceil K \log_2 K \rceil$) in each phase.
2. Run an independent SH instance inside each phase: $L = \lceil \log_2 K \rceil$ stages, eliminating half of the arms in each stage.
3. In each stage $\ell$, sample each arm $N^{(m,\ell)} = \lfloor T_m / (K 2^{-\ell+1} \lceil \log_2 K \rceil) \rfloor$ times.

**Stopping Rule**: At the end of phase $m$, if the selected arm $J_m$ satisfies:

$$L_{J_m}^{(m)} \geq \max_{i \neq J_m} U_i^{(m)}$$

namely, the lower confidence bound of the best arm exceeds the upper confidence bound of all other arms, then stop and output $J_m$.

Here, the confidence width is $b_i^{(m)} = \sqrt{\frac{2}{N^{(m,\ell_i)}} \log\frac{6K \lceil \log_2 K \rceil m^2}{\delta}}$.

**Theoretical Guarantees**:
- **Theorem 3.1** (Correctness): FC-DSH is $\delta$-correct.
- **Theorem 3.2** (Exponential Tail): FC-DSH satisfies a $(\tilde{\Theta}(H_2 \log(1/\delta)),\ \tilde{O}(H_2))$-exponential stopping tail.

### Scheme 2: BrakeBooster (Meta-Algorithm)

**Core Idea**: Take any FC-BAI algorithm $\mathcal{A}$ that satisfies high-probability stopping guarantees as input, and transform it into an algorithm with exponential tail guarantees via repeated invocation + majority voting + a 2D doubling trick.

**BudgetedIdentification Subroutine**:
1. Run $\mathcal{A}$ in $L$ trials under a budget $T$.
2. If more than half of the trials are forced to terminate -> return failure (0).
3. Otherwise -> return the majority vote of non-zero outcomes.

**BrakeBooster Main Loop** (2D doubling trick):
- Iterate through stages indexed by $(r, c)$, where $r = 1, 2, \ldots$ and $c = 1, \ldots, r$.
- Number of trials $L_{r,c} = r \cdot 2^{r-c} L_1$, budget per trial $T_{r,c} = 2^{c-1} T_1$.
- The total budget within the same row $r$ remains constant: $L_{r,c} T_{r,c} = r \cdot 2^{r-1} L_1 T_1$.
- The column direction increases the single-trial budget while decreasing the number of trials, and the row direction doubles the overall budget.

**Theoretical Guarantees**:
- **Theorem 4.1** (Correctness): BrakeBooster is $\delta$-correct.
- **Theorem 4.2** (Exponential Tail): It satisfies a $(\tilde{\Theta}(T_{\delta_0}^*(\mathcal{A}) \ln(1/\delta)),\ T_{\delta_0}^*(\mathcal{A}))$-exponential stopping tail.
- **Corollary 4.3**: When using SE as the base algorithm, BrakeBooster achieves a $(\tilde{\Theta}(H_1 \ln(1/\delta)),\ \tilde{O}(H_1))$-exponential stopping tail.

## Key Experimental Results

### Table 1: Comparison of Algorithm Guarantees

| Algorithm | Exponential-Tailed Stopping | High-Probability Complexity | Asymptotic Expected Complexity | Meta-Algorithm |
|------|----------|------------|-------------|-------|
| Successive Elimination | ✗ | ✓ | ✗ | ✗ |
| LUCB | Unknown | ✓ | ✓ | ✗ |
| Track-and-Stop | Unknown | Unknown | ✓ | ✗ |
| Top Two | Unknown | Unknown | ✓ | ✗ |
| **FC-DSH** | **✓** | ✓ | ✓ | ✗ |
| **BrakeBooster** | **✓** | ✓ | ✓ | **✓** |

### Experiment on SE Failing to Stop

- Setting: 3 arms, means $\{1.0, 0.9, 0.9\}$, Gaussian noise $\mathcal{N}(0,1)$, $\delta=0.01$.
- 1,000 independent trials, forced termination at 30,000 steps.
- A large number of trials did not stop and had already eliminated the best arm -> expected to never stop.

### Performance of BrakeBooster + SE

- Setting: 4 arms, means $\{1.0, 0.9, 0.9, 0.9\}$, $\delta=0.01$, forced termination at 1M steps.
- BrakeBooster enabled **all trials to successfully stop**.
- The CDF curve catches up with vanilla SE at around $\sim 0.05 \times 10^6$ steps, showing an acceptable overhead.

## Highlights & Insights
1. **Reveals a fundamental theoretical blind spot**: The fact that mainstream FC-BAI algorithms might never stop was not formally proven before.
2. **Exponential tail is a more essential guarantee**: It simultaneously implies high-probability complexity, finite expectation, and almost sure stopping.
3. **FC-DSH is simple and effective**: Based on Sequential Halving + doubling trick, it is simple to implement and theoretically elegant.
4. **BrakeBooster is a universal upgrader**: Any FC-BAI algorithm satisfying basic conditions can be "upgraded with one click" to gain exponential tail guarantees.
5. **Clever 2D doubling trick design**: Search for the optimal budget and optimal number of trials simultaneously, ensuring logarithmic overhead.

## Limitations & Future Work
1. **Sample complexity has an extra logarithmic factor**: Compared to asymptotically optimal algorithms like Track-and-Stop, the instance-dependent complexity of FC-DSH and BrakeBooster has an extra polylog term.
2. **Reset mechanisms affect practicality**: FC-DSH runs SH independently in each phase, and BrakeBooster repeatedly calls sub-algorithms, both involving resets and wasting info across phases.
3. **FC-DSH complexity depends on $H_2$ instead of the better $H_1$**: Although $H_2 \leq H_1 \leq \log(2K) H_2$, there is still room for improvement.
4. **Whether modern practical algorithms (such as the Top Two family) already possess exponential tails remains unverified**: This is an important open problem.
5. **Whether the polylog(T) in the exponential tail can be eliminated**: Matching lower bounds are needed to answer this.

## Related Work & Insights
- **Successive Elimination** (Even-Dar et al., 2006): High-probability complexity of $\tilde{O}(H_1 \ln(1/\delta))$, but can potentially never stop.
- **Sequential Halving** (Karnin et al., 2013): A classic algorithm for the fixed-budget setting, serving as the basis for FC-DSH.
- **LUCB** (Kalyanakrishnan et al., 2012): Known to have a polynomial tail guarantee $\mathbb{P}(\tau \geq T) \leq 4\delta/T^2$.
- **Track-and-Stop** (Garivier & Kaufmann, 2016): Asymptotically optimal but lacks tail guarantees.
- **Hyperband** (Li et al., 2018): A practical extension of SH; the 2D doubling in BrakeBooster can be seen as its generalization.
- **Insights**: The tail behavior of stopping time distribution is an under-researched direction in BAI and broader sequential decision-making problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Discovers and formalizes an overlooked fundamental problem, with the proposed exponential tail concept being a brand new theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ — The experiments are mainly of a theoretical verification nature, with no large-scale or complex scenario experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clean motivation, precise definitions, and intuitive comparison tables.
- Value: ⭐⭐⭐⭐ — Reveals a significant theoretical flaw in the BAI field and offers the first class of solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Fixed-Confidence Multiple Change Point Identification under Bandit Feedback](fixed-confidence_multiple_change_point_identification_under_bandit_feedback.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[ICML 2025\] DRO-BAS: Decision Making under the Exponential Family DRO with Bayesian Ambiguity Sets](decision_making_under_the_exponential_family_distributionally_robust_optimisatio.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[ICML 2025\] Discrepancy Minimization in Input-Sparsity Time](discrepancy_minimization_in_input-sparsity_time.md)

</div>

<!-- RELATED:END -->
