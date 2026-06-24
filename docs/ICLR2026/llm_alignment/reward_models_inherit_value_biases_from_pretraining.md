---
title: >-
  [Paper Note] Reward Models Inherit Value Biases from Pretraining
description: >-
  [ICLR 2026][LLM Alignment][Reward Model] This paper employs an interpretability method of "exhaustive token search + psycholinguistic corpora" to systematically examine 10 mainstream open-source Reward Models (RMs). It finds that RM preferences across multiple human value dimensions—such as "agency vs. communion"—highly depend on the base LLM (Llama series prefers agency; Gemma series prefers communion). These biases are traced back to the log-probabilities of the base models…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Reward Model"
  - "Value Bias"
  - "Pretraining"
  - "Interpretability"
  - "Psycholinguistics"
date: 2026-05-08
content_hash: 0cd4b8e62e8e3016
---

# Reward Models Inherit Value Biases from Pretraining

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=dT399j1Azv](https://openreview.net/forum?id=dT399j1Azv)  
**Area**: Alignment RLHF  
**Keywords**: Reward Model, Value Bias, Pretraining, Interpretability, Psycholinguistics

## TL;DR
This paper employs an interpretability method of "exhaustive token search + psycholinguistic corpora" to systematically examine 10 mainstream open-source Reward Models (RMs). It finds that RM preferences across multiple human value dimensions—such as "agency vs. communion"—highly depend on the base LLM (Llama series prefers agency; Gemma series prefers communion). These biases are traced back to the log-probabilities of the base models, proving that they are difficult to "wash away" during the preference fine-tuning process.

## Background & Motivation

**Background**: Reward models are core components of RLHF / DPO for aligning LLMs with human values. However, compared to pre-trained and post-trained LLMs themselves, RMs remain relatively under-studied. With the recent emergence of open-source preference datasets, RM weights, and public benchmarks like RewardBench, interpretability research specifically targeting RMs has begun to gain traction.

**Limitations of Prior Work**: Existing interpretability work on RMs mostly focuses on two aspects: either "proactively" using RMs to push post-trained models toward specific preferences (personalization), or how RMs "unintentionally" introduce biases into post-trained LLMs. However, all such works treat RMs as "transmitters" of bias. No one has addressed a more fundamental question: since an RM is initialized from an LLM and then preference fine-tuned, **does the base LLM transmit its own value biases to the RM**?

**Key Challenge**: The design goal of an RM is to represent "human preference"; theoretically, its output should only reflect the preference data. However, RMs structurally inherit all representations of the base LLM (often directly reusing its weights). This creates a contradiction: How much of an RM's score comes from preference data, and how much comes from the base model's "innate tendencies"? If the latter constitutes a significant proportion, then "changing the base = changing the values," a value-laden consideration almost entirely ignored by the open-source community when selecting base models.

**Goal**: This study decomposes the problem into three progressive sub-questions: (1) Do "wild" open-source RMs exhibit systematic value differences based on their base models? (2) If so, can these differences be traced back to the base LLMs (instruct-tuned or even pre-trained versions)? (3) Under controlled training, how much preference data is required to dilute this "innate bias," or is it impossible to eliminate?

**Key Insight**: The authors adapt the "exhaustive token search" proposed by Christian et al. (2025). For a value-oriented prompt, the method traverses the entire vocabulary to score every token with the RM, identifying which tokens receive the highest/lowest scores to extract the RM's "value preference." By mapping these tokens to value dimensions annotated by psychology experts (e.g., Agency/Communion in the Big Two), vague "values" are quantified into statistically rankable data.

**Core Idea**: Use psycholinguistic corpora to perform a "value check-up" on RM token-level scoring. Applying the same methodology to the log-probabilities of base LLMs proves that RM value bias is a "hereditary condition" established during the pre-training phase and stubbornly retained through preference fine-tuning.

## Method

### Overall Architecture
This is an analytical paper that does not introduce a new model for training; its "method" consists of a progressive diagnostic pipeline corresponding to three sections of the text: **First, measure value differences at the base level in wild RMs → Trace these differences to base LLM log-probabilities (and construct an "implicit RM" to characterize the delta between two bases) → Finally, conduct controlled experiments with RMs trained from scratch to observe how bias evolves and whether it can be washed away by more data.**

The input for the entire pipeline is a "value-oriented prompt + a reward model/language model," and the output is the "preference intensity (median rank) of the model on a specific value dimension (e.g., agency/communion)." The core measurement tool remains consistent: traverse the vocabulary for a prompt to calculate scores (reward scores for RMs, log-probabilities for LLMs), aggregate token-level rankings into value categories based on psycholinguistic corpora, and perform mixed-effects statistical tests. The three stages build upon each other, tracing the "existence of the phenomenon" to its "root cause in pre-training and its resistance to elimination."

### Key Designs

**1. Exhaustive Token Search + Psycholinguistic Corpora: Quantifying "Values" as Statistical Rankings**

Since value preferences in wild RMs are latent, the authors traverse the entire vocabulary for a value-oriented prompt (e.g., "What, in one word, is the greatest thing ever?") and score each token as a response to get a full ranking. These rankings are then mapped to two expert-validated corpora: the Big Two (263 words encoding "agency" such as freedom/success and "communion" such as love/family) and the Moral Foundations Dictionary (MFD2, encoding authority/care/fairness/loyalty/sanctity). By aggregating "token-level rewards" into "category-level rewards," value preference becomes a statistically measurable quantity—the median rank of specific categories.

To ensure robustness, the authors evaluated 10 leading RMs from RewardBench (based on Gemma or Llama) using 54 prompt variants (27 positive + 27 negative formulations). Using mixed-effects linear models with "base choice" as a key factor, the results were clear: under positive prompts, Llama-based RMs ranked agency words higher, while Gemma-based RMs ranked communion words higher (results inverted for negative prompts). The interaction between category, base, and valence was significant ($p < .001$) with a medium effect size (Cohen's $d \approx 0.40\text{–}0.43$).

**2. Implicit Reward Models and MWLR Scores: Reading the "Difference Between Bases"**

After proving differences in wild RMs, the authors traced the root cause to base LLMs. First, direct analysis of log-probabilities for Big Two nouns in instruct-tuned Gemma 2 2B and Llama 3.2 3B revealed the same agency/communion split ($p < .001$). Crucially, this held true for **pre-trained** versions ($p < .001$), indicating the bias was embedded during pre-training.

Furthermore, the authors constructed the "difference between two bases" as an implicit reward model. Based on the mathematics of RLHF, a fine-tuned model can be expressed as $\pi_r(y|x) = \frac{1}{Z_x}\,\pi_{\text{base}}(y|x)\exp(\beta\cdot r(x,y))$. Conversely, any two models $\pi_1, \pi_2$ can be viewed as having an implicit reward defined by their log-probability ratio: $r_{1\to2}(x,y) = c(x) + \beta\cdot\log\frac{\pi_2(y|x)}{\pi_1(y|x)}$. By performing exhaustive token search on this log-ratio, one can identify which tokens are most rewarded/punished when moving from Gemma to Llama. To filter out noise from low-probability tail tokens, the authors used **Mixed-Weight Log Ratio (MWLR)**:

$$\text{MWLR} = \tfrac{1}{2}(p+q)\cdot(\log q - \log p),$$

where $p \equiv \pi_1(\cdot|x)$ and $q \equiv \pi_2(\cdot|x)$. The weight $\frac{1}{2}(p+q)$ ensures that only tokens assigned significant probability by at least one model are amplified. In a Gemma→Llama implicit RM, the optimal token was "Freedom" and the worst (excluding formatting) was "Love," consistent with RM results. This "Freedom > Love" pattern held across 21 pairs of Llama 3 and Gemma 2 models.

**3. Controlled RM Training Experiments: Bias Evolves but Persists**

Finally, the authors trained RMs from scratch. To exclude dataset-specific effects, they used two non-overlapping datasets (Skywork ≈77k, Unified Feedback ≈850k) to initialize RMs from Llama 3.2 3B Instruct and Gemma 2 IT 2B using **identical hyperparameters**. Checkpoints were saved every 1000 steps for exhaustive token search to plot the evolution of bias.

Three findings: first, Llama RMs consistently favored agency while Gemma RMs favored communion. Second, the gap was largest at the start of training and narrowed over time. Third, **the gap narrowed but never closed**, stabilizing around one-third of the way through training. Data volume ablations showed that while more data helps (requiring >100k preference pairs to align Gemma and Llama on these two dimensions), the gap remained unclosed for other bases like Qwen or when using methods like GRM that preserve language modeling capabilities.

## Key Experimental Results

### Main Results

| Target of Analysis | Phenomenon | Key Statistic |
|--------|------|---------|
| 10 Wild RMs (Big Two, Positive) | Llama favors agency; Gemma favors communion; inverted for negative prompts | 3-way interaction $p<.001, d\approx0.40\text{–}0.43$ |
| Instruct Gemma 2 2B vs Llama 3.2 3B (Logprob) | Identical agency/communion split | $F(1,208)=58.3, p<.001$ |
| Pre-trained Gemma 2 2B vs Llama 3.2 3B | Bias already exists in the pre-training stage | $F(1,208)=43.2, p<.001$ |
| Implicit Gemma→Llama RM (MWLR) | Best token = "Freedom", Worst = "Love" | Freedom > Love in 21/21 comparisons |

### Ablation Study

| Configuration | Key Phenomenon | Note |
|------|---------|------|
| Training Dynamics (Skywork) | Gap starts max → narrows → stabilizes without closing at ~1/3 | Bias is persistent |
| Data Source (UF vs Skywork) | Source has minimal impact | Bias is not dataset-specific |
| Data Volume | ~100k+ pairs needed to bridge Gemma/Llama gap | Partial mitigation via data volume |
| 3rd-party Base (Qwen) | Gap remains unclosed even after 100k training | Generalizability warning |
| GRM (630k+ pairs) | Gap remains significant | Methodological choices prolong bias life |

### Key Findings
- **Base models are the true source of value bias**: Given the same data and pipeline, changing the base model reliably alters the RM's value preference, turning base selection into a value-laden decision.
- **Bias is established in pre-training and is stubborn**: A consistent agency/communion split is traceable from wild RMs back to pre-trained LLMs. The gap stabilizes and does not close during training.
- **MWLR weights are critical**: Direct log-ratios are dominated by noise; the $\frac{1}{2}(p+q)$ weight focuses analysis on relevant tokens.

## Highlights & Insights
- **Interpreting the "difference between models" as an implicit RM** is an elegant perspective shift: it leverages the mathematical definition of RLHF as a transformation of the base model.
- **Using psycholinguistic corpora for "value health checks"** provides a quantifiable, expert-validated framework that can be extended to test political, moral, or cultural orientations.
- **The "Freedom vs Love" contrast** provides a powerful visual: Llama answers "Freedom" and Gemma answers "Love" to the same prompt, representing a fundamental divergence between model families.

## Limitations & Future Work
- The study is limited to Big Two dimensions and two primary model families; the "~100k data to bridge" conclusion may not hold in multi-dimensional spaces or for other bases like Qwen.
- Experiments used standard Bradley-Terry loss; the interaction between regularization methods (like GRM) and base bias requires deeper investigation.
- As a diagnostic study, it proves the source of bias but does not propose a specific "de-biasing" training method.

## Related Work & Insights
- **vs. Existing RM Interpretability**: While prior work views RMs as conveyors of bias, this paper identifies them as sources inheriting bias from their base, complicating regularization-based solutions.
- **vs. LLM Value Quantification (Surveys)**: Whereas others use questionnaires to test post-trained LLMs, this study uses expert corpora to scrutinize RMs.
- **vs. Model Multiplicity**: Supports the idea that models with similar performance can have vastly different internal representations, showing these are systematic "family-level" differences rather than random seed variations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to trace RM value bias to pre-training with quantifiable evidence.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of wild RMs and controlled training, though value dimensions could be expanded.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical progression with powerful contrasts.
- Value: ⭐⭐⭐⭐⭐ Elevates base model selection to a core safety and alignment consideration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pretrain Value, Not Reward: Decoupled Value Policy Optimization](pretrain_value_not_reward_decoupled_value_policy_optimization.md)
- [\[ICLR 2026\] Group-Normalized Implicit Value Optimization for Language Models](group-normalized_implicit_value_optimization_for_language_models.md)
- [\[ICLR 2026\] Cognitive models can reveal interpretable value trade-offs in language models](cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)
- [\[ICLR 2026\] StoryAlign: Evaluating and Training Reward Models for Story Generation](storyalign_evaluating_and_training_reward_models_for_story_generation.md)
- [\[ICLR 2026\] Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment](evaluating_and_improving_cultural_awareness_of_reward_models_for_llm_alignment.md)

</div>

<!-- RELATED:END -->
