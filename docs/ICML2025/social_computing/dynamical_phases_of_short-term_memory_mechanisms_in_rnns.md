---
title: >-
  [Paper Note] Dynamical Phases of Short-Term Memory Mechanisms in RNNs
description: >-
  [ICML 2025][Social Computing][Short-term memory] This work discovers two distinct underlying dynamical mechanisms supporting short-term memory in RNNs—slow-point manifolds and limit cycles. It analytically derives the power-law scaling laws of their maximum learnable learning rates using toy models (SP: $\beta$ approx. 4-5 vs LC: $\beta$ approx. 2-3), and provides large-scale empirical validation by training approximately 80,000 RNNs.
tags:
  - "ICML 2025"
  - "Social Computing"
  - "Short-term memory"
  - "RNN"
  - "dynamical phase transition"
  - "slow-point manifold"
  - "limit cycle"
date: 2026-05-08
content_hash: 9bf1d0822ed9e677
---

# Dynamical Phases of Short-Term Memory Mechanisms in RNNs

**Conference**: ICML 2025  
**arXiv**: [2502.17433](https://arxiv.org/abs/2502.17433)  
**Code**: [https://github.com/fatihdinc/dynamical-phases-stm](https://github.com/fatihdinc/dynamical-phases-stm)  
**Area**: Social Computing  
**Keywords**: Short-term memory, RNN, dynamical phase transition, slow-point manifold, limit cycle

## TL;DR
This work discovers two distinct underlying dynamical mechanisms supporting short-term memory in RNNs—slow-point manifolds and limit cycles. It analytically derives the power-law scaling laws of their maximum learnable learning rates using toy models (SP: $\beta$ approx. 4-5 vs LC: $\beta$ approx. 2-3), and provides large-scale empirical validation by training approximately 80,000 RNNs.

## Background & Motivation
**Background**: Short-term memory is a core function of cognitive processing, but its neural mechanisms are still incompletely understood in systems neuroscience. Prior studies have linked memory maintenance to sequential activation patterns.  

**Limitations of Prior Work**: Recurrent connections are believed to drive sequential dynamics, but a mechanistic understanding is lacking. Three key questions remain: Q1: What mechanism supports memory maintenance of sequential activity? Q2: What determines the choice of mechanism? Q3: How do mechanisms change with delay duration?  

**Key Challenge**: Different internal mechanisms can produce identical activities within the trial period but exhibit completely different out-of-trial behaviors, making them difficult to distinguish solely from behavioral data.  

**Goal**: To systematically identify and classify dynamical mechanisms in RNNs, and to establish quantitative relationships between task and optimization parameters and mechanism selection.  

**Key Insight**: Analytical analysis of low-dimensional toy models combined with large-scale full-rank RNN training, from a dynamical systems theory perspective.  

**Core Idea**: There exist two equivalent neural sequence generation mechanisms in RNNs, whose emergence is determined by the power-law relationship between delay duration and learning rate, forming a predictable phase diagram.

## Method

### Overall Architecture
A four-step progression: (1) observing the two mechanisms on a rank-2 RNN; (2) studying the impact of task design on mechanism selection; (3) deriving scaling laws from a toy model; (4) training approximately 80,000 full-rank RNNs to validate the theory.

### Key Designs

1. **Delayed Activation Task**: 

    - **Function**: The simplest short-term memory task—suppressing the output during $T_{\text{delay}}$, and then producing the output during $T_{\text{resp}}$.
    - **Mechanism**: Stripping away all unnecessary complexity, leaving only the core challenge of "delayed output".
    - Variants: Adding a post-response period ($T_{\text{post}}$) fundamentally changes the learned mechanism.

2. **Slow-Point Manifold Mechanism**: 

    - **Function**: Creating a slow region in the state space to achieve delay.
    - **Mechanism**: The system slowly traverses the slow-point region to achieve a time delay.
    - Features: Converging to a fixed point after the trial, without repetition.
    - Scaling: $\alpha_{\text{SP}} \sim \mathcal{O}(T_{\text{delay}}^{-\beta_{\text{SP}}})$, where $\beta_{\text{SP}} \in [4,5]$—showing a very steep decay.

3. **Limit Cycle Mechanism**: 

    - **Function**: Creating a closed periodic orbit to achieve delay.
    - **Mechanism**: The half-period corresponds to the transition from "suppression to activation".
    - Features: Continuing to oscillate after the trial.
    - Scaling: $\alpha_{\text{LC}} \sim T_{\text{delay}}^{-\beta_{\text{LC}}}$, where $\beta_{\text{LC}} \in [2,3]$—showing a milder decay compared to SP.

4. **Toy Model Analytical Analysis**: 

    - SP Model: Saddle-node bifurcation $dx/dt = x^2 + r$.
    - LC Model: $x(t) = \sin(2\pi rt)$, where $r$ is learnable.
    - Key Findings: $\beta_{\text{LC}} \le \beta_{\text{SP}}$, meaning limit cycles allow a larger learning rate under large delays.

5. **Mechanism Discrimination Index**: 

    - Automated classification based on spectral analysis: Limit cycles have low energy in low frequencies, while slow-point manifolds have high energy in low frequencies.

### Loss & Training
- RNN Equation: $\tau \frac{dr}{dt} = -r(t) + \tanh(Wr(t) + W_{\text{in}}u(t) + b + \epsilon)$
- Standard MSE Loss
- SGD, systematically scanning learning rates and delay durations.
- Scale: Approximately 80,000 RNNs ($N=100$ neurons), with carbon emissions of about 230 kg $\text{CO}_2$.

## Key Experimental Results

### Main Results

| Mechanism Type | Theoretical Scaling Exponent | Experimental Scaling Exponent | Match |
|----------|------------|------------|------|
| Slow-Point Manifold (SP) | $\beta \in [4, 5]$ | 4.05 ± 0.10 | Consistent |
| Limit Cycle (LC) | $\beta \in [2, 3]$ | 2.72 ± 0.07 | Consistent |

| Task Variant | SP Occurrence Rate | LC Occurrence Rate | Description |
|---------|---------|---------|------|
| No post-response period, short delay | High | Low | SP dominates |
| No post-response period, long delay | Low | High | LC dominates |
| With post-response period | Approx. 0 | High | Post-response period strongly biases towards LC |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| No-memory task | $\beta = 0.38 \pm 0.02$ | Delay-dependent scaling almost disappears |
| Increasing time constant $\tau$ | Similar scaling | Robust to different intrinsic dynamics |
| Mechanism evolution during training | SP shifts to LC | Corresponds to jumps in the loss curve |

### Key Findings
- The two mechanisms produce nearly identical sequential activity within the trial window, but exhibit completely different out-of-trial behaviors.
- Adding a post-response period fundamentally changes the mechanism—shifting from potentially SP to almost always LC.
- Around 80,000 RNNs precisely reproduce the theoretical scaling predictions of the toy model.
- Mechanism selection forms a phase diagram determined by (delay duration $\times$ learning rate).

## Highlights & Insights
- A complete closed loop from theory to large-scale empirical validation.
- An important warning for systems neuroscience: minor choices in experimental design can fundamentally alter the dynamical mechanisms.
- Open release of 80,000 pre-trained RNN models.
- Mechanism evolution during training reveals "algorithmic phase transitions" in the optimization process.

## Limitations & Future Work
- Toy models cannot exhaust all possible memory mechanisms.
- Only SGD optimizer is considered.
- The RNN scale of 100 neurons is relatively limited.
- Lack of direct comparison with real neural data.

## Related Work & Insights
- Rajan et al. (2016) linked sequential activity with short-term memory; this work reveals the two underlying mechanisms that generate sequences.
- Drawing an interesting analogy to works on grokking/algorithmic phase transitions.
- Insight: Dynamical systems theory provides a powerful tool to understand the qualitative changes in neural network training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](../../ACL2026/social_computing/probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[ICML 2025\] Raising the Bar: Investigating the Values of Large Language Models via Generative Evolving Testing](raising_the_bar_investigating_the_values_of_large_language_models_via_generative.md)
- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[ICML 2025\] When Bad Data Leads to Good Models](when_bad_data_leads_to_good_models.md)
- [\[ICML 2025\] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)

</div>

<!-- RELATED:END -->
