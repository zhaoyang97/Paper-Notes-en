---
title: >-
  [Paper Note] Meek Models Shall Inherit the Earth
description: >-
  [ICML 2025][LLM Pretraining][scaling laws] Based on mathematical modeling of Chinchilla scaling laws, this work demonstrates that under a next-token prediction objective with a fixed distribution, the diminishing returns of compute scaling will eventually cause the capabilities gap of state-of-the-art (SOTA) large models relative to low-compute-budget "meek models" to converge to zero. This argues that the democratization of AI capabilities is an inevitable trend under the cu…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "scaling laws"
  - "diminishing returns"
  - "compute scaling"
  - "AI democratization"
  - "AI governance"
date: 2026-05-08
content_hash: a4ed58d64ef560fa
---

# Meek Models Shall Inherit the Earth

**Conference**: ICML 2025  
**Authors**: Hans Gundlach, Jayson Lynch, Neil Thompson (MIT CSAIL)  
**arXiv**: [2507.07931](https://arxiv.org/abs/2507.07931)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: scaling laws, diminishing returns, compute scaling, AI democratization, AI governance  

## TL;DR

Based on mathematical modeling of Chinchilla scaling laws, this work demonstrates that under a next-token prediction objective with a fixed distribution, the diminishing returns of compute scaling will eventually cause the capabilities gap of state-of-the-art (SOTA) large models relative to low-compute-budget "meek models" to converge to zero. This argues that the democratization of AI capabilities is an inevitable trend under the current scaling paradigm, and existing compute-based AI governance strategies require a fundamental redesign.

## Background & Motivation

**Background**: Over the past decade, the scale of AI systems has grown dramatically—with training compute increasing at a rate of 5x per year between 2010 and 2022, and large corporations dominating the training of SOTA models such as GPT, LLaMA, and Gemini. Common intuition suggests that more compute equals stronger performance and a greater competitive advantage.

**Limitations of Prior Work**: This intuition overlooks a critical issue—diminishing returns from scaling. Chinchilla scaling laws show that the relation between loss and compute is $L_{opt}(C) = AC^{-\alpha} + L_0$. An exponent of $\alpha \approx 0.155$ means that for every 10x increase in compute, loss only decreases by about 30%. When compute is already massive, the marginal gain from further doubling is extremely small.

**Key Challenge**: While corporate compute investments grow exponentially (approx. 3.6x/year), all organizations share the benefits of hardware improvements (Moore's law ~1.4x/year) and algorithmic advances (~2.8x/year). These shared advancements exponentially increase the effective compute for all players—causing fixed-budget "meek models" to strengthen rapidly as well. When the diminishing returns of scaling are prominent enough, the loss advantage gained from proprietary investment growth starts to shrink after a certain inflection point.

**Key Insight**: This study models the difference in training loss between SOTA models and fixed-budget meek models, deriving the inflection time and convergence trends of the loss advantage, and translates training loss differences into observable differences in capability.

**Core Idea**: Under the current next-token scaling paradigm, diminishing returns + shared algorithmic/hardware advancements = the capability gap between SOTA and meek models will eventually converge.

## Method

### Overall Architecture

The paper constructs three progressive theoretical models: (1) training inequality model—the training loss difference between SOTA and meek models; (2) inference inequality model—the performance difference under a fixed inference budget; (3) loss-capability translation—mapping loss differences to benchmark performance and information-theoretic distinguishability. These predictions are then validated with empirical data.

### Key Designs

1. **Training Loss Difference Model**:

    - **Function**: Quantifies the change over time in the training loss gap between SOTA and meek models
    - **Mechanism**: Let a meek model have a fixed \$1000 training budget (approx. $10^{17}$ GPU FLOPs), while SOTA model compute investment grows at $g_i = 3.57$ times/year. Both benefit from algorithmic advances $g_{alg} = 2.8$ times/year and hardware improvements $g_h = 1.4$ times/year. Based on Chinchilla's law, the loss difference is $\Delta L = A(g_{alg} \cdot g_h)^{-\alpha t} C_0^{-\alpha} - A(g_{alg} \cdot g_h \cdot g_i)^{-\alpha t} C_0^{-\alpha}$. Crucial derivation: the inflection time for the loss advantage is $t^* = \frac{1}{\alpha \ln g_i} \ln\left[\frac{\ln(g_h g_{alg} g_i)}{\ln(g_h g_{alg})}\right]$
    - **Design Motivation**: To demonstrate that even with exponential growth in SOTA compute investment, the compounding effects of shared algorithmic/hardware improvements will eventually drive the marginal advantage to zero

2. **Inference Inequality Model**:

    - **Function**: Analyzes the gap between the SOTA and models that can run under a fixed inference budget (e.g., \$10^-8/token)
    - **Mechanism**: Inference cost is influenced by three factors: hardware FLOPs/\$ growth, parameter/FLOP efficiency (KV-cache, sparse attention, etc.), and effective-to-actual parameter ratio (distillation, overtraining, etc.). Cottier et al. 2025 data shows inference costs fall at ~9x/year, which means the size of effective models runnable at a fixed budget grows at 9x/year, much faster than training investment growth. Consequently, the inference performance gap converges even faster
    - **Design Motivation**: For most users, inference cost is more relevant than training cost—if near-SOTA performance can be obtained with cheap inference, practical democratization of AI capabilities is already achieved

3. **Loss-Capability Translation Analysis**:

    - **Function**: Proves that the convergence of loss differences indeed signifies a meaningful convergence in capabilities
    - **Mechanism**: Argued from two perspectives: (a) **Sigmoid Benchmark Mapping**—Benchmark performance (like MMLU) has a sigmoid relationship with loss, $Perf = \frac{A}{1+e^{-k(L-x_0)}} + b$; thus, a narrowing loss gap shrinks the benchmark performance gap. (b) **Hypothesis Testing Perspective**—Based on SPRT (Sequential Probability Ratio Test), the number of tokens required to distinguish two models is $E[N] = \frac{(1-\alpha)\log\frac{1-\alpha}{\alpha} + \alpha\log\frac{\alpha}{1-\alpha}}{\Delta L}$. As $\Delta L \to 0$, the required tokens approach infinity, rendering the two models practically indistinguishable
    - **Design Motivation**: Responds to the skepticism that "a narrowing loss difference does not imply a narrowing of actual capability gaps"

### Loss & Training

This is a theoretical analytical paper. It utilizes the Chinchilla loss formulation $L_{opt}(C) = 1070 \cdot C^{-0.155} + 1.7$ as the foundation for the analysis.

## Key Experimental Results

### Main Results——Model Prediction Comparison

| Dimension | SOTA Model | Meek Model (\$1000 budget) | Trend of Difference |
|---------|----------|---------------------|---------|
| Training Loss Difference | Exponential decrease | Slower decrease | Widens first, then converges; inflection point around mid-2020s |
| MMLU Performance Difference | Stable at ~80% | Rapidly catching up | Gap narrows from ~20% to ~5% within 5 years |
| Inference Loss Difference | - | - | Converges faster than training loss difference (inference cost drops by 9x/year) |

### Empirical Validation——MMLU-Pro Score Gaps

| Year | Best Model MMLU-Pro | Best MMLU-Pro within Fixed Inference Budget (\$0.5-1/1M tokens) | Gap |
|------|-----------------|---------------------------------------------|------|
| Mid 2023 | ~55 | ~35 | ~20 points |
| Early 2024 | ~62 | ~50 | ~12 points |
| Late 2024 | ~72 | ~65 | ~7 points |
| Trend | Slow growth | Rapidly catching up | Continuously narrowing |

### Key Findings

- There is a clear inflection point for the training loss advantage—using a \$1000 $C_0$ baseline, the inflection occurs around the mid-2020s, after which the SOTA's relative advantage continuously shrinks.
- The inference performance gap converges far quicker than the training gap—since improvements in inference efficiency (~9x/year) vastly outpace training investment growth (~3.6x/year).
- Empirical data from the Artificial Analysis LLM Leaderboard is qualitatively consistent with the model predictions—the MMLU-Pro scores of the best models within a fixed inference price range are rapidly catching up to the unconstrained best models.
- Even if SOTA investment growth rates change (e.g., from 3.6x to 5x or 10x per year), the qualitative conclusion remains unchanged—it only delays the inflection point.
- Multi-step tasks (requiring $p$ consecutive correct steps) extend the advantage window for SOTA models, but they still eventually converge.

## Highlights & Insights

- Counter-intuitive yet mathematically rigorous core conclusion—more money $\neq$ enduring advantage, overturning the simplistic "compute is king" narrative.
- Disentangles three computing growth factors (investment growth, hardware progress, algorithmic improvement), clearly explaining why shared advancements "drown out" the advantages of proprietary investments.
- The information-theoretic perspective (SPRT distinction cost) provides an exact quantification of the practical significance of loss differences.
- The discussion on AI governance addresses real-world pain points—pointing out that FLOP-threshold-based regulatory strategies (e.g., US >$10^{26}$, EU >$10^{25}$) might become obsolete under the trend of capability democratization.

## Limitations & Future Work

- The core assumption relies on a fixed distribution next-token prediction—new paradigms like RL, synthetic data, and test-time compute scaling might disrupt the diminishing returns of scaling laws.
- In adversarial scenarios (e.g., security contests, game theory), small capability gaps can be magnified into massive advantages; the paper acknowledges this but does not explore it in depth.
- Using parameter count as a proxy for training compute is not perfectly accurate (distillation, overtraining, etc., make the relationship non-linear).
- Empirical data is sparse and heavily sourced from commercial LLM leaderboards, which may be influenced by pricing strategies rather than pure technical capability.
- Does not account for data walls—exclusive access to high-quality data may provide durable advantages beyond those of scaling laws.

## Related Work & Insights

- **Chinchilla Scaling Laws** (Hoffmann et al., 2022): The core instrument of this paper; the power-law form $L = AC^{-\alpha} + L_0$ directly governs the strength of diminishing returns.
- **Algorithmic Progress** (Ho et al., 2024): Quantifies the staggering rate at which effective compute of language models doubles every 8 months, which is a major driver for the rapid catch-up of meek models.
- **Thompson et al., 2021**: "The Diminishing Returns of Deep Learning" is a spiritual predecessor to this paper, though this work extends the formulation to competitive dynamics and governance implications.
- **Insights**: Although the paper openly concedes that new computing paradigms (inference scaling, RL, synthetic data) might change the conclusions, the core framework provides a valuable baseline for evaluating future trends.

## Rating

| Dimension | Score | Reason |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First systematic modeling of the capability convergence trend between SOTA and meek models. |
| Technical Depth | ⭐⭐⭐⭐ | Rigorous derivation of scaling laws; creative information-theoretic arguments. |
| Experimental Thoroughness | ⭐⭐⭐ | Dominated by theoretical analysis; empirical validation data is relatively sparse. |
| Writing Quality | ⭐⭐⭐⭐ | Highly coherent core arguments; deep discussion on governance implications. |
| Value | ⭐⭐⭐⭐⭐ | Holds direct reference value for AI strategy, investment decisions, and governance policies. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)
- [\[ICML 2025\] Language Models over Canonical Byte-Pair Encodings](language_models_over_canonical_byte-pair_encodings.md)
- [\[ICML 2025\] Large Language Models are Demonstration Pre-Selectors for Themselves](large_language_models_are_demonstration_pre-selectors_for_themselves.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](../../ACL2025/llm_pretraining/leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ICML 2025\] Provable Maximum Entropy Manifold Exploration via Diffusion Models](provable_maximum_entropy_manifold_exploration_via_diffusion_models.md)

</div>

<!-- RELATED:END -->
