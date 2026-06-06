---
title: >-
  [Paper Note] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning
description: >-
  [ICML 2026][LLM Pretraining][Data Difficulty] This work systematically investigates the role of data difficulty in SFT, finding that there is no "universally optimal difficulty." Instead…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Data Difficulty"
  - "Supervised Fine-Tuning"
  - "Generalization-Extrapolation Tradeoff"
  - "PAC-Bayes"
  - "Data Scale"
date: 2026-05-08
content_hash: 63b5d2bfe7390992
---

# Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2605.12906](https://arxiv.org/abs/2605.12906)  
**Code**: None  
**Area**: LLM Pre-training / SFT Data Selection  
**Keywords**: Data Difficulty, Supervised Fine-Tuning, Generalization-Extrapolation Tradeoff, PAC-Bayes, Data Scale

## TL;DR
This work systematically investigates the role of data difficulty in SFT, finding that there is no "universally optimal difficulty." Instead, the optimal difficulty **shifts toward harder data as data scale increases**. This is explained via a trade-off between the "in-distribution generalization gap" and the "extrapolation gap," with a PAC-Bayes interpretation.

## Background & Motivation

**Background**: Heuristics for data selection in SFT are diverse—some advocate removing "too easy" samples (LIMO, s1, Marion et al.), others suggest retaining "easy samples similar to the base model distribution" (BERTIN, DFT, Anchored-SFT), and some claim "medium difficulty is best." Each paper presents compelling comparative tables, yet their conclusions are often contradictory.

**Limitations of Prior Work**: These conclusions lack a unified explanatory framework, making the engineering decision of "whether to select hard or easy data" somewhat mystical. In Table 1, the authors show that medium is optimal on OpenR1-Math-94k, easy is optimal on OpenMath, and on OpenScience, easy/medium are close while hard drops sharply. The same model and evaluation can yield opposite conclusions on different datasets.

**Key Challenge**: Previous work almost always compares difficulty at a "fixed data scale," but **difficulty and data scale are not independent variables**—they jointly determine SFT model performance. Figure 2 provides a key observation: removing "hard" samples is beneficial with small data, but harmful with large data; removing "easy" samples has the opposite effect.

**Goal**: (1) Establish a two-dimensional experimental map of (data scale $n$, data difficulty); (2) Use a single mechanism to explain both "non-monotonicity of difficulty" and "optimal difficulty drifting with $n$"; (3) Provide an interpretable theoretical upper bound.

**Key Insight**: Decompose test risk into **in-distribution generalization gap** $G_{\mathrm{gen}}$ and **extrapolation gap** $G_{\mathrm{ext}}$—the former increases with difficulty (harder to fit) and decreases with $n$, while the latter decreases with difficulty (harder training distributions better cover hard test distributions). The opposing movement of these two gaps produces a **unimodal** "optimal difficulty."

**Core Idea**: Replace the binary "hard/easy" logic with a "trade-off between the TV/KL gap between train-test distributions and the posterior-prior KL gap," and point out that increasing $n$ mainly compresses $G_{\mathrm{gen}}$, so the optimal difficulty monotonically shifts rightward with $n$.

## Method

This work is not about "proposing a method," but rather about "building a mechanism + theoretical upper bound + extensive controlled experiments."

### Overall Architecture
The paper is structured in three layers: (a) Two-dimensional SFT scans on real data (Qwen2.5-Math-1.5B/7B × OpenMath across difficulty buckets and sizes); (b) Controlled experiments on synthetic iGSM data, using the number of operations as a proxy for difficulty, and evaluating by test set difficulty slices to separate "in-distribution fit collapse" vs "extrapolation stagnation" failure modes; (c) A PAC-Bayes theoretical framework providing an interpretable two-gap decomposition upper bound (Proposition 4.1).

### Key Designs

1. **CoT-length as a Difficulty Metric**:

    - Function: Uses ground-truth Chain-of-Thought length as a proxy for problem difficulty.
    - Mechanism: Figure 1 shows CoT length is strongly negatively correlated with external LLM pass rate—the longer the CoT, the lower the pass rate, indicating higher difficulty. This avoids the circularity of "using the model's own perplexity to measure the difficulty it needs to learn."
    - Design Motivation: Perplexity-based difficulty depends on the evaluated model and drifts with SFT; CoT length is a task-side attribute, comparable across models, and facilitates "same difficulty, different model" experiments.

2. **Two-dimensional Experimental Map (size × difficulty) + Decomposed Evaluation**:

    - Function: Moves beyond the "fix $n$ and sweep difficulty" or "fix difficulty and sweep $n$" local perspectives, drawing a complete heatmap and slicing the test set by operation count to observe SFT model improvements at each test difficulty.
    - Mechanism: Figure 6 shows two typical failure modes—easy training improves in-domain test but drops on hard test (extrapolation failure); hard training with small $n$ causes all slices to drop (generalization failure). This decomposition is key to diagnosing "where things break down."
    - Design Motivation: Aggregate scores can obscure mechanisms; slicing reveals which end of the trade-off is problematic, supporting the subsequent PAC-Bayes upper bound's physical interpretation.

3. **Two-gap PAC-Bayes Decomposition (Proposition 4.1)**:

    - Function: Expresses the test risk upper bound as $\mathbb{E}_{\theta\sim\pi_\mathrm{train}}[R_{\mathcal D_\mathrm{test}}(\theta)]\le \mathbb E[\hat R_S(\theta)] + G_\mathrm{gen}+G_\mathrm{ext}+\epsilon$, where $G_\mathrm{gen}=\mathcal O(\sqrt{\mathrm{KL}(\pi_\mathrm{train}\|\pi_\mathrm{pre})/n})$, $G_\mathrm{ext}=\mathcal O(\mathrm{TV}(\mathcal D_\mathrm{test},\mathcal D_\mathrm{train}))$.
    - Mechanism: Treats pre-training as the prior $\pi_\mathrm{pre}$ and SFT-trained parameter distribution as the posterior $\pi_\mathrm{train}$; PAC-Bayes provides a posterior-prior KL complexity term, and the TV term captures the shift from training to test distribution. As difficulty increases, the posterior moves further from the prior, raising $G_\mathrm{gen}$; but the training distribution better covers hard test data, lowering $G_\mathrm{ext}$. Their opposing trends produce a unimodal optimum.
    - Design Motivation: Previous SFT data selection lacked theoretical anchors; this upper bound explains four key observations (data scale, non-monotonicity of difficulty, optimal difficulty drift, model-relative difficulty), and gives geometric intuition that "tuning difficulty is equivalent to regularizing between KL and TV."

### Loss & Training

SFT uses standard CE; difficulty slices are defined by dividing CoT length into three (easy/medium/hard) or by equal operation count intervals (iGSM experiments). In the DFT case study, token weights are $\mathrm{sg}(p_\theta)\cdot \nabla\log p_\theta$, using "token probability" as an implicit difficulty signal—this extension validates that the theory also applies to "token-level data selection."

## Key Experimental Results

### Main Results

| Dataset | Base Model | Easy | Medium | Hard | Optimal Difficulty |
|---|---|---|---|---|---|
| OpenR1-Math-94k (Math500) | Qwen2.5-Math-1.5B | 61.1 | **68.3** | 61.7 | medium |
| OpenMath 200k subset (Math500) | Qwen2.5-Math-1.5B | **71.7** | 70.1 | 69.0 | easy |
| OpenScience 200k subset (MMLU) | Qwen2.5-Math-1.5B | **53.4** | 53.0 | 41.2 | easy |

Two-dimensional scan conclusions (Figures 3-4, OpenMath/Qwen2.5-Math-7B): At fixed $n$, the performance-difficulty curve is inverted-U shaped; at fixed difficulty, performance vs. scale shows logarithmic saturation; optimal difficulty shifts toward harder data as $n$ increases.

### Ablation Study (Synthetic iGSM Controlled Experiments, Section 4-5)

| Configuration | Phenomenon | Explanation |
|---|---|---|
| Base Ops[2–8]2k, hard training + small $n$ | All test slices drop | $G_\mathrm{gen}$ dominates (cannot fit) |
| Base Ops[2–8]2k, easy training + any $n$ | Easy test improves, hard test drops | $G_\mathrm{ext}$ dominates (not covered) |
| Base Ops[2–8]2k, medium training | Overall best improvement | $G_\mathrm{gen}+G_\mathrm{ext}$ minimized |
| Strong base (Ops[2–8]2k vs Ops[2–6]2k) | Stronger base shifts optimal difficulty right | Stronger prior → smaller $G_\mathrm{gen}$ |
| DFT vs SFT, small $n$ + hard data | DFT outperforms SFT | DFT biases toward high-probability tokens, equivalent to implicit difficulty reduction |
| DFT vs SFT, large $n$ | SFT overtakes | DFT's high-probability token bias suppresses $G_\mathrm{ext}$ improvement |

### Key Findings
- "Optimal difficulty" is an increasing function of $n$: small data favors easy samples (reducing $G_\mathrm{gen}$), large data favors hard samples (reducing $G_\mathrm{ext}$). This holds for both real math/science data and synthetic iGSM.
- Difficulty is **relative**: the same "hard" is medium for a strong base, ultra-hard for a weak base; thus, data selection must consider base model capability, not just absolute token length.
- The non-universal gain of DFT is naturally explained by the theory—it is essentially an implicit easy-shift, thus beneficial when "training difficulty is too high and $n$ is insufficient," but hindered by $G_\mathrm{ext}$ when "data is sufficient."

## Highlights & Insights
- Unifies previously contradictory "easy/medium/hard is best" claims into a single $G_\mathrm{gen}$-$G_\mathrm{ext}$ trade-off framework—an exemplary case of "theory resolving experimental chaos."
- Decomposed evaluation on iGSM is a powerful diagnostic tool for "seeing the mechanism"—once test difficulty is sliced, the reason for model performance drops becomes immediately clear, and this should be adopted in all SFT data ablation studies.
- Viewing SFT as "posterior deviation from prior + train-test distribution shift" is the most natural application of traditional PAC-Bayes to LLM SFT, providing a clear physical explanation for "tuning difficulty according to data scale."

## Limitations & Future Work
- The theoretical upper bound is still worst-case; actual TV and KL values are nearly impossible to estimate on real text, so guidance is qualitative. How to make "optimal difficulty" a computable, actionable criterion remains open, as acknowledged by the authors.
- Experiments are mainly on Qwen2.5-Math and Llama math families; when extending to code, agents, or general dialogue SFT, "CoT length" as a difficulty metric may not be stable.
- The DFT case is only an extension for validation; no concrete algorithm is provided for "token-level adaptive adjustment according to this theory." Future work could design size-dependent token weighting to directly implement the theory.

## Related Work & Insights
- **vs LIMO / s1 (Ye et al. 2025, Muennighoff et al. 2025)**: These always remove "easy problems already solved by the base," corresponding to "blindly selecting the hardest." This is only reasonable when $n$ is large; with small data, it is disastrous.
- **vs BERTIN / Zhang et al. 2025**: Advocates "easy samples close to the base distribution," i.e., "blindly selecting the easiest." This is only reasonable with small data.
- **vs DFT (Wu et al. 2025)**: Implicitly biases toward easy at the token level; this work absorbs it as a token-level case, unifying the explanation for DFT's variable performance across settings.
- **vs Curriculum Learning**: This work reveals the root cause of why curriculum is "often but not always effective"—curriculum is equivalent to moving along the "optimal difficulty curve as $n$ increases" during training, but if the schedule is misaligned, it can go astray.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new method, but unifies many contradictory conclusions under a single framework—a rare and valuable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Real data + synthetic iGSM + multiple base models + multiple evaluations, with comprehensive 2D heatmaps and slice analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logical structure, PAC-Bayes explanations tightly linked to four key observations; formula layout is dense but readable.
- Value: ⭐⭐⭐⭐ Directly actionable guidance for teams working on SFT data selection—do not fixate on easy/hard, but jointly decide based on base capability and data budget.

## Related Papers

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICML 2025\] DipLLM: Fine-Tuning LLM for Strategic Decision-Making in Diplomacy](../../ICML2025/llm_pretraining/dipllm_fine-tuning_llm_for_strategic_decision-making_in_diplomacy.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[ICLR 2026\] Block-Sample MAC-Bayes Generalization Bounds](../../ICLR2026/llm_pretraining/block-sample_mac-bayes_generalization_bounds.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)

</div>

<!-- RELATED:END -->
