---
title: >-
  [Paper Note] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning
description: >-
  [ICML 2026][Pretraining][PAC-Bayes] This paper systematically investigates the role of data difficulty in SFT, discovering that there is no "universally optimal difficulty." Instead, an optimal difficulty exists that **drifts toward harder samples as the data scale increases**. This is explained through a PAC-Bayes framework as a tradeoff between the "in
tags:
  - ICML 2026
  - Pretraining
  - PAC-Bayes
date: 2026-05-08
content_hash: 5b005c1766c765b4
---
# Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2605.12906](https://arxiv.org/abs/2605.12906)  
**Code**: None  
**Area**: LLM Pre-training / SFT Data Selection  
**Keywords**: Data Difficulty, Supervised Fine-Tuning, Generalization-Extrapolation Tradeoff, PAC-Bayes, Data Scale

## TL;DR
This paper systematically investigates the role of data difficulty in SFT, discovering that there is no "universally optimal difficulty." Instead, an optimal difficulty exists that **drifts toward harder samples as the data scale increases**. This is explained through a PAC-Bayes framework as a tradeoff between the "in-distribution generalization gap" and the "extrapolation gap."

## Background & Motivation

**Background**: Various heuristics for SFT data selection currently coexist—some advocate for removing "too easy" samples (LIMO, s1, Marion et al.), some prefer "simple data close to the base model distribution" (BERTIN, DFT, Anchored-SFT), and others argue "medium difficulty is best." Each paper provides competitive results, yet their conclusions frequently contradict one another.

**Limitations of Prior Work**: These conclusions lack a unified explanatory framework, making the choice between hard and easy data an empirical "black art." Table 1 shows that on OpenR1-Math-94k, medium is optimal; on OpenMath, easy is optimal; on OpenScience, easy/medium perform similarly while hard performance plunges. Conclusions for the same model and evaluation metric can flip across different datasets.

**Key Challenge**: Previous studies mostly compared difficulty at a "fixed data scale," but **difficulty and data scale are not independent variables**—they jointly determine model performance after SFT. Figure 2 provides a key observation: excluding "hard" samples is beneficial at small scales but harmful at large scales; excluding "easy" samples shows the opposite trend.

**Goal**: (1) Establish a 2D experimental map of (data scale $n$, data difficulty); (2) provide a mechanism to explain both the "non-monotonicity of difficulty" and its "drift with $n$"; (3) derive an interpretable theoretical upper bound.

**Key Insight**: Test risk is decomposed into an **in-distribution generalization gap** $G_{\mathrm{gen}}$ and an **extrapolation gap** $G_{\mathrm{ext}}$. The former increases with difficulty (harder to fit) and decreases with $n$; the latter decreases with difficulty (harder training distributions better cover harder test distributions). The opposing movement of these two gaps produces a **unimodal** "optimal difficulty."

**Core Idea**: The binary logic of "hard vs. easy" is replaced by a "tradeoff between the TV/KL gap (train vs. test distribution) and the posterior-prior KL gap." It is noted that increasing $n$ primarily compresses $G_{\mathrm{gen}}$, causing the optimal difficulty to monotonically shift rightward as $n$ grows.

## Method

### Overall Architecture
Rather than proposing a new "method," this work builds a mechanism through theoretical bounds and extensive controlled experiments. The framework consists of three layers: first, a 2D SFT scan (scale $\times$ difficulty) on real data (Qwen2.5-Math-1.5B/7B $\times$ OpenMath buckets/sizes); second, precise difficulty control using synthetic iGSM data to isolate failure modes of "in-distribution fit collapse" and "extrapolation failure"; finally, a PAC-Bayes interpretable bound (Proposition 4.1) to unify all observed phenomena.

### Key Designs

**1. CoT-length Difficulty Metric: Side-stepping Circular Dependency via Task Attributes**

The most natural way to measure problem difficulty is using the model's own perplexity. However, perplexity depends on the specific model being evaluated and drifts during SFT, making it an inconsistent yardstick. This paper uses the length of the ground-truth Chain-of-Thought (CoT) as a proxy for difficulty. CoT length is a task-side attribute, comparable across models, and facilitates controlled experiments with the same difficulty but different base models. Figure 1 validates this metric: longer CoTs strongly correlate with lower pass rates from external LLMs, allowing for a reliable easy/medium/hard tripartite split.

**2. 2D Map + Decomposed Evaluation: Diagnosing Score Changes**

Previous work often viewed difficulty through local slices (fixed $n$ or fixed difficulty). This paper plots a complete (size $\times$ difficulty) heatmap and, crucially, slices the test set by operation count to calculate the improvement of the SFT model on each test difficulty level. This step is the primary diagnostic tool: summary scores only show if performance rose or fell, whereas Figure 6 reveals specific failure modes—easy training improves in-domain test results but fails on hard tests (extrapolation failure), while hard training at small $n$ causes declines across all test slices (generalization failure).

**3. Two-gap PAC-Bayes Decomposition: Difficulty Adjustment as KL-TV Regularization**

The paper formulates the test risk upper bound as $\mathbb{E}_{\theta\sim\pi_\mathrm{train}}[R_{\mathcal D_\mathrm{test}}(\theta)]\le \mathbb E[\hat R_S(\theta)] + G_\mathrm{gen}+G_\mathrm{ext}+\epsilon$, where the generalization term $G_\mathrm{gen}=\mathcal O(\sqrt{\mathrm{KL}(\pi_\mathrm{train}\|\pi_\mathrm{pre})/n})$ and the extrapolation term $G_\mathrm{ext}=\mathcal O(\mathrm{TV}(\mathcal D_\mathrm{test},\mathcal D_\mathrm{train}))$. Physically, the pre-trained model acts as the prior $\pi_\mathrm{pre}$ and the SFT parameters act as the posterior $\pi_\mathrm{train}$. Increasing difficulty pushes the posterior further from the prior (increasing $G_\mathrm{gen}$) but aligns the training distribution closer to difficult test sets (decreasing $G_\mathrm{ext}$). The sum of these opposing gaps naturally creates a unimodal optimal difficulty. Since $n$ compresses $G_{\mathrm{gen}}$, the optimal difficulty shifts right as $n$ increases.

## Key Experimental Results

### Main Results

| Dataset | Base Model | Easy | Medium | Hard | Optimal Difficulty |
|---|---|---|---|---|---|
| OpenR1-Math-94k (Math500) | Qwen2.5-Math-1.5B | 61.1 | **68.3** | 61.7 | medium |
| OpenMath 200k subset (Math500) | Qwen2.5-Math-1.5B | **71.7** | 70.1 | 69.0 | easy |
| OpenScience 200k subset (MMLU) | Qwen2.5-Math-1.5B | **53.4** | 53.0 | 41.2 | easy |

2D scanning conclusions (Figure 3-4): For a fixed $n$, the performance-difficulty curve is an inverted U-shape. For a fixed difficulty, performance vs. scale shows logarithmic saturation. The optimal difficulty drifts harder as $n$ increases.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---|---|---|
| Ops[2–8]2k, Hard train + Small $n$ | Performance drops across all test slices | $G_\mathrm{gen}$ dominates (failure to fit) |
| Ops[2–8]2k, Easy train + Any $n$ | Easy test rises, hard test drops | $G_\mathrm{ext}$ dominates (failure to cover) |
| Ops[2–8]2k, Medium train | Highest overall improvement | $G_\mathrm{gen}+G_\mathrm{ext}$ sum is minimized |
| Strong base vs. Weak base | Optimal difficulty shifts right for strong base | Stronger prior $\to$ smaller $G_\mathrm{gen}$ term |
| DFT vs. SFT, Small $n$ + Hard data | DFT outperforms SFT | DFT biases toward high-prob tokens, implicitly reducing difficulty |
| DFT vs. SFT, Large $n$ | SFT outperforms DFT | DFT's easy-bias hinders $G_\mathrm{ext}$ improvement |

### Key Findings
- The "optimal difficulty" is an increasing function of $n$: small data scales favor simple samples (to reduce $G_\mathrm{gen}$), while large scales favor difficult samples (to reduce $G_\mathrm{ext}$). This holds across math, science, and iGSM data.
- Difficulty is **relative**: a "hard" sample for a weak base model might be "medium" for a strong one. Data selection must consider base model capability, not just absolute token length.
- The non-universal gains of DFT are explained—it acts as an implicit easy-shift, benefiting scenarios with high training difficulty and low $n$, but underperforming when data is abundant.

## Highlights & Insights
- The paper unifies contradictory claims about "easy vs. hard" data into a single $G_\mathrm{gen}$-$G_\mathrm{ext}$ framework—a rare instance of theoretical clarity resolving empirical confusion.
- The "decomposed evaluation" used on iGSM is a powerful diagnostic tool; slicing graphs by test difficulty immediately reveals why a model is losing performance.
- Treating SFT as a dual-source risk (posterior deviation + distribution shift) provides a clear physical explanation for adjusting difficulty based on data scale.

## Limitations & Future Work
- The theoretical bounds are worst-case; the exact values of TV and KL are nearly impossible to estimate on real text, serving only as qualitative guidance.
- Experiments focused on the Qwen2.5-Math and Llama math families; CoT-length may be less stable as a difficulty metric in areas like code or general dialogue.
- The DFT analysis is an extension; a size-dependent token-weighting algorithm based directly on this theory has not yet been designed.

## Related Work & Insights
- **vs. LIMO / s1 (Ye et al. 2025)**: These works advocate for selecting the "hardest" data. This paper shows this is only optimal when $n$ is large; it can be disastrous at small scales.
- **vs. BERTIN**: Advocates for "simple samples close to base distribution." This is shown to be optimal only at small data scales.
- **vs. Curriculum Learning**: This work identifies why curriculum learning is often, but not always, effective—it follows the "optimal difficulty drift" curve, but fails if the schedule is misaligned.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifying contradictory conclusions under a single framework is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across real/synthetic data, multiple base models, and 2D heatmap analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear logic; the PAC-Bayes explanation aligns well with observations.
- Value: ⭐⭐⭐⭐ Actionable guidance for SFT data selection—difficulty should be chosen based on base model capability and data budget.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICML 2026\] Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](../../ICML2025/llm_pretraining/dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)

</div>

<!-- RELATED:END -->
