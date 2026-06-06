---
title: >-
  [Paper Note] Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning
description: >-
  [ICML 2026][LLM Pretraining][Data Difficulty] This paper systematically investigates the role of data difficulty in SFT, discovering that there is no "universal optimal difficulty." Instead…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Data Difficulty"
  - "Supervised Fine-Tuning"
  - "Generalization-Extrapolation Tradeoff"
  - "PAC-Bayes"
  - "Data Scale"
date: 2026-05-08
content_hash: 1ffe39449b0e70f7
---

# Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2605.12906](https://arxiv.org/abs/2605.12906)  
**Code**: None  
**Area**: LLM Pre-training / SFT Data Selection  
**Keywords**: Data Difficulty, Supervised Fine-Tuning, Generalization-Extrapolation Tradeoff, PAC-Bayes, Data Scale

## TL;DR
This paper systematically investigates the role of data difficulty in SFT, discovering that there is no "universal optimal difficulty." Instead, there exists an optimal difficulty that **drifts toward harder samples as data scale increases**. This is explained through a PAC-Bayes framework as a trade-off between the "in-distribution generalization gap" and the "extrapolation gap."

## Background & Motivation

**Background**: Heuristics for data selection in SFT are diverse—some suggest removing "too simple" samples (LIMO, s1, Marion et al.), some advocate for retaining "easy" data close to the base model distribution (BERTIN, DFT, Anchored-SFT), while others claim "medium difficulty is best." Each paper provides compelling comparison tables, yet they frequently contradict one another.

**Limitations of Prior Work**: The aforementioned conclusions lack a unified explanatory framework, making the choice between "hard" or "easy" data an empirical mystery in engineering. In Table 1, the authors show that "medium" is optimal for OpenR1-Math-94k, "easy" is optimal for OpenMath, and for OpenScience, "easy/medium" performs well while "hard" causes performance to crash—conclusions flip for the same model and evaluation across different datasets.

**Key Challenge**: Prior works mostly compare difficulty at a "fixed data scale," but **difficulty and data scale are not independent variables**—they jointly determine model performance after SFT. Figure 2 provides a key observation: removing "hard" samples is beneficial at small scales but harmful at large scales; removing "easy" samples shows the opposite trend.

**Goal**: (1) Establish a 2D experimental map of (data scale $n$, data difficulty); (2) Use a single mechanism to simultaneously explain "non-monotonic difficulty" and the "drift of optimal difficulty with $n$"; (3) Provide an interpretable theoretical upper bound.

**Key Insight**: Decompose test risk into an **in-distribution generalization gap** $G_{\mathrm{gen}}$ and an **extrapolation gap** $G_{\mathrm{ext}}$. The former increases with higher difficulty (harder to fit) and decreases with $n$, while the latter decreases as difficulty increases (harder training distributions better cover harder test distributions). The opposing movements of these two gaps produce a **unimodal** "optimal difficulty."

**Core Idea**: Replace the binary "easy/hard logic" with a trade-off between the "TV/KL gap between training and test distributions" and the "posterior-prior KL gap." It is noted that increasing $n$ primarily compresses $G_{\mathrm{gen}}$, causing the optimal difficulty to monotonically shift right as $n$ grows.

## Method

This paper is not primarily about "proposing a method" but rather "building a mechanism + theoretical bounds + extensive controlled experiments."

### Overall Architecture
The work is structured in three layers: (a) A 2D scan of SFT on real data (Qwen2.5-Math-1.5B/7B × OpenMath difficulty buckets × various sizes); (b) Precise difficulty control experiments on synthetic iGSM data, where difficulty is represented by the number of operations, evaluated across test set difficulty slices to isolate failure modes ("in-distribution fitting failure" vs. "extrapolation failure"); (c) A PAC-Bayes theoretical framework providing an interpretable two-gap decomposition bound (Proposition 4.1).

### Key Designs

1. **CoT-length as a Difficulty Metric**:

    - **Function**: Uses the ground-truth Chain-of-Thought length as a proxy for problem difficulty.
    - **Mechanism**: Figure 1 verifies that CoT length strongly correlates negatively with external LLM pass rates—longer CoT leads to lower pass rates, indicating higher difficulty. This avoids the circular dependency of "measuring difficulty based on the model's own perplexity."
    - **Design Motivation**: Perplexity-based difficulty depends on the evaluated model and drifts during SFT; CoT length is a task-side attribute, comparable across models, facilitating experiments with "consistent difficulty across different models."

2. **2D Experimental Map (size × difficulty) + Decomposed Evaluation**:

    - **Function**: Moves beyond the local perspective of "fixed $n$ scanning difficulty" or "fixed difficulty scanning $n$" by plotting a full heatmap and slicing the test set by operation count to observe SFT gains on each test difficulty level.
    - **Mechanism**: Figure 6 illustrates two typical failure modes—training on "easy" data improves in-domain test scores but fails on "hard" test slices (extrapolation failure); training on "hard" data at small $n$ leads to a drop across all slices (generalization failure). This is the key diagnostic for identifying "where it broke."
    - **Design Motivation**: A single total score masks underlying mechanisms; slicing reveals where the trade-off ends fail, supporting the physical interpretation of the PAC-Bayes bound.

3. **Two-gap PAC-Bayes Decomposition (Proposition 4.1)**:

    - **Function**: Formulates the test risk upper bound as $\mathbb{E}_{\theta\sim\pi_\mathrm{train}}[R_{\mathcal D_\mathrm{test}}(\theta)]\le \mathbb E[\hat R_S(\theta)] + G_\mathrm{gen}+G_\mathrm{ext}+\epsilon$, where $G_\mathrm{gen}=\mathcal O(\sqrt{\mathrm{KL}(\pi_\mathrm{train}\|\pi_\mathrm{pre})/n})$ and $G_\mathrm{ext}=\mathcal O(\mathrm{TV}(\mathcal D_\mathrm{test},\mathcal D_\mathrm{train}))$.
    - **Mechanism**: Views pre-training as the prior $\pi_\mathrm{pre}$ and the SFT parameter distribution as the posterior $\pi_\mathrm{train}$. PAC-Bayes provides a complexity term for the posterior-prior KL. The TV term captures the shift from the training distribution to the test distribution. Increasing difficulty → posterior moves further from prior → $G_{\mathrm{gen}}$ rises; however, the training distribution becomes closer to a hard test set → $G_{\mathrm{ext}}$ falls. The opposing movements create a unimodal peak.
    - **Design Motivation**: Previous SFT data selection lacked theoretical anchors. This bound explains four major observations (data scale, non-monotonic difficulty, optimal difficulty drift, and relative model difficulty) and provides a geometric intuition of "tuning difficulty as an equivalence to regularizing between KL and TV."

### Loss & Training
Standard CE loss is used for SFT. Difficulty slices are partitioned by tertiles of CoT length (easy/medium/hard) or equidistant intervals of operation counts (iGSM experiments). In the DFT case study, token weights are $\mathrm{sg}(p_\theta)\cdot \nabla\log p_\theta$, using "token probability" as an implicit difficulty signal—this part serves as an extended case to verify that the theory holds for "token-level data selection."

## Key Experimental Results

### Main Results

| Dataset | Base Model | Easy | Medium | Hard | Optimal Difficulty |
|---|---|---|---|---|---|
| OpenR1-Math-94k (Math500) | Qwen2.5-Math-1.5B | 61.1 | **68.3** | 61.7 | medium |
| OpenMath 200k subset (Math500) | Qwen2.5-Math-1.5B | **71.7** | 70.1 | 69.0 | easy |
| OpenScience 200k subset (MMLU) | Qwen2.5-Math-1.5B | **53.4** | 53.0 | 41.2 | easy |

2D scanning conclusions (Figure 3-4, OpenMath/Qwen2.5-Math-7B): At a fixed $n$, the performance-difficulty curve is inverted U-shaped; at a fixed difficulty, performance-scale follows logarithmic saturation; the optimal difficulty drifts toward "harder" as $n$ increases.

### Ablation Study (Synthetic iGSM Controlled Experiments, Sections 4-5)

| Configuration | Phenomenon | Explanation |
|---|---|---|
| Base Ops[2–8]2k, hard training + small $n$ | Performance drops across all test slices | $G_\mathrm{gen}$ dominates (failure to fit) |
| Base Ops[2–8]2k, easy training + any $n$ | Easy test improves, hard test drops | $G_\mathrm{ext}$ dominates (failure to cover) |
| Base Ops[2–8]2k, medium training | Highest overall improvement | Sum of $G_\mathrm{gen} + G_\mathrm{ext}$ is minimized |
| Stronger base (Ops[2–8]2k vs Ops[2–6]2k) | Optimal difficulty shifts right with strong base | Stronger prior → smaller $G_\mathrm{gen}$ term |
| DFT vs SFT, small $n$ + hard data | DFT outperforms SFT | DFT biases toward high-prob tokens, implicitly lowering difficulty |
| DFT vs SFT, large $n$ | SFT overtakes DFT | DFT's bias toward high-prob tokens hinders $G_\mathrm{ext}$ improvement |

### Key Findings
- "Optimal difficulty" is an increasing function of $n$: Small data scales prefer easy samples (reducing $G_{\mathrm{gen}}$), while large data scales prefer difficult samples (reducing $G_{\mathrm{ext}}$). This assertion is replicated across real math/science data and synthetic iGSM data.
- Difficulty is **relative**: A "hard" sample for a weak base might be "medium" for a strong base and "ultra-hard" for another; thus, data selection must consider base model capability rather than absolute token length.
- The non-universal gains of DFT are naturally explained by theory—it is essentially an implicit "easy-shift," benefiting when "training difficulty is too high and $n$ is insufficient" but being held back by $G_{\mathrm{ext}}$ when "data is abundant."

## Highlights & Insights
- It unifies the seemingly contradictory "easy vs. medium vs. hard" debate from previous papers into a single $G_{\mathrm{gen}}$-$G_{\mathrm{ext}}$ framework—a rare and excellent example of using theory to resolve experimental chaos.
- The decomposed evaluation on iGSM is a powerful diagnostic tool for "seeing the mechanism"—once results are plotted by test difficulty slices, "why the model failed" is immediately revealed. This should be adopted in all SFT data ablation studies.
- Viewing SFT as a dual-source risk of "posterior deviating from prior + train-test distribution shift" is the most natural application of PAC-Bayes to LLM SFT, providing a clear physical explanation for "tuning difficulty according to data scale."

## Limitations & Future Work
- Theoretical bounds remain in a worst-case form; specific values of TV and KL are nearly impossible to estimate on real text, offering only qualitative guidance. How to transform "optimal difficulty" into a computable, executable criterion remains an open question, as the authors acknowledge.
- Experiments focused on Qwen2.5-Math and Llama math families; "CoT length" as a difficulty metric may not be stable when extended to code, agents, general dialogue, and other SFT domains.
- The DFT case is an extended validation; it does not provide a specific algorithm for "how to adaptively adjust at the token level using this theory." Future work could design size-dependent token weighting to implement the theory directly.

## Related Work & Insights
- **vs LIMO / s1 (Ye et al. 2025, Muennighoff et al. 2025)**: These exclude "easy problems the base already knows," corresponding to "blindly selecting the hardest." Ours proves this is only reasonable when $n$ is large; it is disastrous for small data scales.
- **vs BERTIN / Zhang et al. 2025**: Advocates for "easy samples close to the base distribution," corresponding to "blindly selecting the easiest." Ours proves this is only reasonable for small data scales.
- **vs DFT (Wu et al. 2025)**: Implicitly biases toward "easy" at the token level; our work absorbs this as a token-level case study, explaining why DFT performance varies across settings.
- **vs Curriculum Learning**: Ours provides the root cause of why curriculum is "often effective but not always"—curriculum is equivalent to moving along the "optimal difficulty grows with $n$" curve during training, but if the schedule is misaligned, it deviates from the optimum.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new method, but a rare contribution that uses a unified framework to reconcile a series of contradictory conclusions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Real data + iGSM synthetic data + multiple base models + multiple evaluations, with comprehensive 2D heatmaps and slice analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logical layers; the PAC-Bayes explanation is tightly coupled with the four major observations; formulas are dense but readable.
- Value: ⭐⭐⭐⭐ Directly actionable guidance for teams doing SFT data filtering—do not fixate on easy/hard selection; decide based on the joint relationship of base capability and data budget.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](../../ICLR2026/llm_pretraining/token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICML 2026\] Tuning the Implicit Regularizer of Masked Diffusion Language Models: Enhancing Generalization via Insights from k-Parity](tuning_the_implicit_regularizer_of_masked_diffusion_language_models_enhancing_ge.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)

</div>

<!-- RELATED:END -->
