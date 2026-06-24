---
title: >-
  [Paper Note] Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation
description: >-
  [ACL 2025][Reasoning][self-consistency] Reinterprets Self-Consistency as a dynamic alignment problem between the sampling distribution and the true answer distribution, revealing that temperature not only controls sampling randomness but also directly shapes the true answer distribution. Based on this, a confidence-driven three-stage dynamic temperature adjustment mechanism is proposed (with theoretical derivation of the FSD threshold), improving both average and best perform…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "self-consistency"
  - "temperature calibration"
  - "distributional alignment"
  - "confidence-driven"
date: 2026-05-08
content_hash: 2c43da4a182059ec
---

# Revisiting Self-Consistency from Dynamic Distributional Alignment Perspective on Answer Aggregation

**Conference**: ACL 2025  
**arXiv**: [2502.19830](https://arxiv.org/abs/2502.19830)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: self-consistency, temperature calibration, distributional alignment, confidence-driven, reasoning

## TL;DR
Reinterprets Self-Consistency as a dynamic alignment problem between the sampling distribution and the true answer distribution, revealing that temperature not only controls sampling randomness but also directly shapes the true answer distribution. Based on this, a confidence-driven three-stage dynamic temperature adjustment mechanism is proposed (with theoretical derivation of the FSD threshold), improving both average and best performance with zero training overhead across 10 models on GSM8K/MATH.

## Background & Motivation

**Background**: Self-Consistency (SC) is a classic approach to improving LLM reasoning—voting for the final answer after sampling CoT multiple times. Its effectiveness has been verified across various tasks, but the underlying dynamical mechanisms have lacked theoretical understanding.

**Limitations of Prior Work**:
   - **Rigid, Fixed Temperature**: The sampling temperature in SC is a fixed hyperparameter, whereas questions of varying difficulty require different temperatures—simple questions converge with low temperatures, while difficult questions require high temperatures to explore better answer distributions.
   - **Lack of Theoretical Understanding**: Present works merely understand SC as "multiple samplings fitting the true distribution", ignoring the shaping effect of temperature on the true distribution itself.
   - **Wasted Sampling Budget**: Under a fixed temperature, continuing to sample for simple questions is wasteful, whereas insufficient sampling for difficult questions leads to excessive noise.

**Key Challenge**: Low temperature = high certainty + low diversity (fast convergence but potentially stuck in suboptimal distributions), while high temperature = high diversity + high noise (enables exploration of better distributions but requires more sampling to stabilize). Temperature affects both convergence speed and final accuracy, but in opposing directions.

**Goal**:  
(1) How does temperature influence the convergence behavior and final accuracy of SC?  
(2) Can temperature be dynamically adjusted based on real-time confidence to accelerate convergence and explore superior distributions?

**Key Insight**: Leveraging a distributional alignment perspective—the essence of SC is to align the top-1 answer of the sampling distribution with that of the true distribution, where the temperature parameter dictates the efficiency and endpoint of this alignment process.

**Core Idea**: Dynamically regulate temperature driven by real-time confidence (FSD), lowering temperature for convergence under low confidence and raising temperature for exploration under high confidence, achieving dynamic synchronization of the sampling and true distributions.

## Method

### Overall Architecture
Building upon the standard SC workflow (multiple sampling $\rightarrow$ voting), the sampling process is modified into a three-stage adaptive temperature strategy. The input is a question $\mathbf{x}$, and the output is the final voted answer $\hat{y}_{SC}$. The core change is that the temperature $T$ is no longer fixed but dynamically adjusted based on the confidence of existing sampling results.

### Key Designs

1. **Distributional Alignment Theoretical Framework**:

    - **Function**: Reinterprets why SC is effective and how temperature affects it from a probability theory perspective.
    - **Mechanism**: SC is a Monte Carlo estimation of the true answer distribution $p(y|\mathbf{x})$, with the sampling distribution $\hat{p}_{SC}(y) \to p(y|\mathbf{x})$ as $n \to \infty$. However, the key new finding is that **temperature modifies the true distribution itself**—high temperatures make the distribution flatter (more answers have non-zero probabilities), while low temperatures make it sharper.
    - **Key Findings**: (1) Convergence speed is positively correlated with accuracy and negatively correlated with temperature; (2) under infinite sampling, the optimal temperature is relatively high (~1.0), but under finite sampling, the optimal temperature decreases as the number of samples decreases; (3) CoT increases confidence by narrowing the output space.
    - **Design Motivation**: To break the conception of "temperature as merely a hyperparameter" and reveal it as the key knob governing the "exploration-exploitation" trade-off.

2. **Confidence-Driven Dynamic Temperature Adjustment Mechanism**:

    - **Function**: Adaptively regulates temperature based on the First-Second Distance (FSD) of current sampling results.
    - **Mechanism**: FSD is defined as the difference between the top-1 and top-2 answer probabilities: $\text{FSD}^{(t)} = p_1^{(t)} - p_2^{(t)}$, reflecting the model's certainty in distinguishing the dominant candidate answers. The temperature update rules are as follows:
        - $\text{FSD} < \tau - \epsilon$ (low confidence) $\rightarrow$ lower temperature by 0.1 (converging to current optimum)
        - $\text{FSD} > \tau + \epsilon$ (high confidence) $\rightarrow$ raise temperature by 0.1 (exploring better distributions)
        - Intermediate zone $\rightarrow$ remains unchanged (dead-zone design to ensure stability)
    - The temperature range is constrained to $[0.1, 1.0]$, and $\epsilon = 0.05$.
    - **Design Motivation**: Low confidence indicates that finite sampling cannot reliably determine the top-1 answer, requiring a lower temperature to concentrate sampling; high confidence implies that the current distribution has stabilized, allowing a higher temperature to explore (as the high-temperature distribution offers higher accuracy under infinite sampling).

3. **Three-Stage Sampling Protocol**:

    - **Phase 1 - Exploration**: Sample $n_1 = 5$ items with a preset temperature $T^{(1)}$ to estimate the initial FSD.
    - **Phase 2 - Adaptation**: Adjust the temperature to $T^{(2)}$ based on the FSD and continue to sample $n_2 = 0.5N - n_1$ items.
    - **Phase 3 - Exploitation**: Re-adjust the temperature to $T^{(3)}$ and sample the remaining $n_3 = 0.5N$ items.
    - **Design Motivation**: Progressively shifting from exploration to exploitation—estimating difficulty with a small sample size early on, and concentrating sampling efforts later.

4. **Theoretical Derivation of FSD Threshold**:

    - **Function**: Derives the reasonable value of the FSD threshold $\tau$ via hypothesis testing theory.
    - **Mechanism**: Formulates a one-tailed z-test where the null hypothesis is "the current top-1 is not the top-1 of the true distribution". Through multinomial distribution and Jensen's inequality derivation, it yields $z \geq \hat{d}\sqrt{2N}$. Setting $z = 1.64$ ($p < 0.05$) gives $\tau = \frac{1.16}{\sqrt{N}}$.
    - **Design Motivation**: Providing statistically backed thresholds rather than arbitrary default and heuristic parameters; the threshold naturally decreases as the sample size $N$ increases (the more samples, the smaller the FSD difference required to confirm top-1).

## Key Experimental Results

### Main Results
Evaluated on two mathematical reasoning datasets, GSM8K and MATH, across 10 models (Qwen2.5 1.5B/7B $\times$ {base, instruct, math, math-instruct} + Llama-3-8B $\times$ {base, instruct}). Comparing fixed temperature SC (Fix) vs. dynamic temperature SC (Dynamic), this reports the average performance (Mean) and the best single temperature performance (Max) under varying sampling budgets $N=\{10, 20, 40\}$.

| Model | Dataset | N | Fix Mean | Dynamic Mean | Fix Max | Dynamic Max |
|------|--------|---|----------|-------------|---------|-------------|
| Qwen2.5-7B | GSM8K | 10 | 84.6 | **84.7** | 86.1 | **86.3** |
| Qwen2.5-7B | GSM8K | 40 | 86.3 | **86.8** | 88.9 | **89.0** |
| Qwen2.5-7B | MATH | 10 | 48.7 | **49.6** | 52.0 | **52.3** |
| Qwen2.5-7B | MATH | 40 | 51.8 | **53.2** | 54.9 | **55.1** |
| Llama-3-8B | GSM8K | 40 | 62.5 | **64.3** | 67.4 | **67.6** |
| Llama-3-8B | MATH | 40 | 21.7 | **23.6** | 25.1 | **25.5** |
| Qwen2.5-Math-7B | MATH | 40 | 56.3 | **57.7** | 59.4 | **59.7** |

### Ablation Study

| Configuration | Description | Impact |
|------|------|------|
| Fixed low temp ($T=0.4$) | Fast convergence but low accuracy upper bound | Fast convergence but lower Max |
| Fixed high temp ($T=0.8$) | High accuracy upper bound but slow convergence | Requires more samples to outperform low temp |
| Dynamic temperature | Adaptive adjustment | Simultaneous improvement in Mean and Max |
| Varying initial temperatures | $T \in \{0.4, 0.6, 0.8, 1.0\}$ | Consistent improvements across all starting points |

### Key Findings
- **Simultaneous Improvement in Mean and Max**: The dynamic strategy simultaneously improves both average and best-temperature performance across nearly all model/dataset/sample size combinations, indicating it is not a simplistic trade-off between temperatures.
- **Weaker Models Benefit More**: Llama-3-8B achieves a 1.9 percentage point gain in Mean (21.7 $\rightarrow$ 23.6) on MATH with $N=40$, indicating that weaker models are more sensitive to temperature.
- **Benefits Are More Significant with Smaller $N$**: The dynamic strategy yields greater advantages under limited sampling budgets, as the noise issue of fixed high temperatures is more pronounced in such scenarios.
- **Effective Theoretical Threshold**: The theoretically derived threshold $\tau = 1.16/\sqrt{N}$ matches the empirically optimal threshold.
- **Positive Correlation of Convergence Speed to Accuracy and Negative Correlation to Temperature**: This explains why weaker models perform better at lower temperatures.

## Highlights & Insights
- **Theoretical Depth of Distributional Alignment**: Instead of a simple "adaptive temperature" trick, this work theoretically uncovers the overlooked fact that temperature alters the true distribution itself, and derives a statistically-guaranteed threshold from it.
- **Highly Practical Zero-Overhead Improvement**: Requires no training, no extra data, models, or modules. It dynamically adapts temperatures based on existing sampling outputs during inference, rendering it truly plug-and-play.
- **Rational Three-Stage Design**: They progress through Exploration $\rightarrow$ Adaptation $\rightarrow$ Exploitation. This progressive strategy is more stable than adjusting the temperature at every single step, avoiding frequent oscillations.

## Limitations & Future Work
- **Inaccurate Initial Confidence**: The FSD estimated from the first 5 samples may suffer from large deviations, particularly when the answer space is large.
- **Fixed Temperature Step Size of 0.1**: Future work could consider continuous temperature adjustments based on the degree of FSD deviation rather than discrete step intervals.
- **Evaluation Limited to Mathematical Reasoning**: Only validated on GSM8K and MATH, where the answer space is restricted (primarily numbers). Generalizability to open-ended generation tasks remains unverified.
- **Interaction with CoT Lacks Depth**: While the paper finds that CoT enhances confidence, it is left as future work. The interaction between CoT quality and dynamic temperature merits deeper exploration.
- **Optimal Temperature Upper Bound**: The authors hypothesize that the optimal temperature upper bound of ~1.0 is correlated with the base training temperature, but this remains unverified.

## Related Work & Insights
- **vs. Standard Self-Consistency (Wang et al., 2022)**: Standard SC employs a fixed temperature with voting, while this work demonstrates that the temperature is dynamically adjustable and should adapt to question difficulty.
- **vs. Adaptive Consistency (Li et al., 2024)**: Li et al. save computation by predicting the required number of samples. This work optimizes along the temperature dimension; the two methods are orthogonal and can be combined.
- **vs. Inference-Time Scaling**: The dynamic temperature here can be viewed as an inference-time compute allocation strategy—reducing sampling for simple questions (fast convergence via low temperature) and boosting exploration for hard questions (high temperature), which aligns with the philosophy of inference-time compute scaling.

## Rating
- Novelty: ⭐⭐⭐⭐ The distributional alignment perspective and theoretical derivation show depth, though the core idea of dynamic temperature tuning is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10 models $\times$ 2 datasets $\times$ multiple sampling counts, offering wide coverage, but is limited to mathematical reasoning tasks.
- Writing Quality: ⭐⭐⭐⭐ The theoretical analysis is clear, and the logical progression from Findings to Insights is coherent.
- Value: ⭐⭐⭐⭐ A plug-and-play improvement to SC, applicable to any system employing SC.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ranked Voting based Self-Consistency of Large Language Models](ranked_voting_based_self-consistency_of_large_language_models.md)
- [\[ICML 2025\] Self-Consistency Preference Optimization](../../ICML2025/llm_reasoning/self-consistency_preference_optimization.md)
- [\[ACL 2025\] CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis](cot-based_synthesizer_enhancing_llm_performance_through_answer_synthesis.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](../../NeurIPS2025/llm_reasoning/a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[ACL 2026\] Does Self-Consistency Improve the Recall of Encyclopedic Knowledge?](../../ACL2026/llm_reasoning/does_self-consistency_improve_the_recall_of_encyclopedic_knowledge.md)

</div>

<!-- RELATED:END -->
