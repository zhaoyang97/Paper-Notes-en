---
title: >-
  [Paper Note] Confidence Estimation for LLMs in Multi-turn Interactions
description: >-
  [ACL 2026 Findings][LLM (Other)][Multi-turn dialogue] This paper presents the first systematic study of LLM confidence estimation in multi-turn dialogue scenarios. It proposes two core desiderata (per-turn calibration and monotonicity with increasing information), the corresponding InfoECE metric and Kendall’s $\tau$ evaluation, and the Hinter-Guesser dataset construction paradigm. A novel P(SUFFICIENT) logit probe is introduced. Findings indicate that existing methods (verba…
tags:
  - "ACL 2026 Findings"
  - "LLM (Other)"
  - "Multi-turn dialogue"
  - "Confidence estimation"
  - "P(SUFFICIENT)"
  - "InfoECE"
  - "Monotonicity"
date: 2026-05-08
content_hash: cbdc81d6ee8af047
---

# Confidence Estimation for LLMs in Multi-turn Interactions

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.02179](https://arxiv.org/abs/2601.02179)  
**Code**: Mentioned GitHub in the paper (see original text for link)  
**Area**: LLM Calibration / Dialogue / Confidence Estimation  
**Keywords**: Multi-turn dialogue, Confidence estimation, P(SUFFICIENT), InfoECE, Monotonicity

## TL;DR
This paper presents the first systematic study of LLM confidence estimation in multi-turn dialogue scenarios. It proposes two core desiderata (per-turn calibration and monotonicity with increasing information), the corresponding InfoECE metric and Kendall’s $\tau$ evaluation, and the Hinter-Guesser dataset construction paradigm. A novel P(SUFFICIENT) logit probe is introduced. Findings indicate that existing methods (verbalized / SC / P(TRUE)) exhibit poor calibration and monotonicity in multi-turn settings. In contrast, P(SUFFICIENT) reduces InfoECE to 5.27 on the GUESS task (vs. 79.97 for P(TRUE)) and achieves a $\tau$ of 81.51, although the task remains far from solved.

## Background & Motivation
**Background**: Confidence estimation is a central direction for mitigating LLM hallucinations. However, the vast majority of existing work (FActScore, Tian 2023, Xiong 2024) focuses on single-turn QA, assuming the input is a complete, one-time question.

**Limitations of Prior Work**: (1) Real human-computer interaction is multi-turn and incremental—users clarify needs over multiple turns, models ask follow-up questions, and the hypothesis space gradually narrows. Confidence behavior under this dynamic information accumulation remains unstudied. (2) It is unknown whether existing methods maintain calibration in multi-turn settings or reflect the intuition that "more information leads to higher certainty." (3) There is a lack of evaluation metrics and datasets tailored for multi-turn scenarios—single-turn ECE cannot handle differences in dialogue length, and standard QA datasets lack incremental information structures.

**Key Challenge**: In multi-turn scenarios, confidence should not be a "fixed attribute of a single response" but a "dynamic signal that increases as the dialogue evolves and information accumulates." Whether this can be achieved, by what methods, and how to measure it are questions without systematic answers.

**Goal**: (1) Formalize two desiderata for multi-turn confidence: per-turn calibration and monotonicity. (2) Design the length-normalized InfoECE metric and the Kendall’s $\tau$ monotonicity metric. (3) Construct datasets suitable for multi-turn evaluation (under-specified using the Hinter-Guesser paradigm to generate 20Q / GUESS; fully-specified using existing GRACE / TrickMe). (4) Evaluate mainstream confidence methods and propose the new P(SUFFICIENT) method.

**Key Insight**: The authors observe that in under-specified scenarios, "a correct answer does not imply sufficient information." A model might guess the correct answer by chance while multiple candidates remain unexcluded; in such cases, confidence should remain low. P(TRUE) only asks "is the answer correct," failing to capture this lack of identifiability.

**Core Idea**: Shift the semantics of the confidence probe from "is the answer correct (P(TRUE))" to "is the current information sufficient to uniquely determine the answer (P(SUFFICIENT))." This aligns confidence with identifiability rather than incidental correctness.

## Method

### Overall Architecture
For each turn $i$ of every dialogue $d$, the model outputs an answer $\hat{y}_{d,i}$ and a confidence score $c_{d,i} \in [0, 1]$, and the correctness $z_{d,i} = \mathbb{I}[\hat{y}_{d,i} = y_d]$ is recorded. To eliminate the impact of dialogue length differences, turn $i$ is normalized into an information level $s_{d,i} = i / L_d \in (0, 1]$, and divided into $B$ bins. Two core metrics are used: (a) **InfoECE** = $\frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|$, which measures calibration at each information level; (b) **Kendall’s $\tau$**, which measures the degree to which confidence monotonically increases within a dialogue. This is evaluated across 5 confidence estimation methods (3 existing + 1 proposed) and 2 types of datasets (under-spec / fully-spec).

### Key Designs

**1. Hinter-Guesser Paradigm: Ensuring Strict Information Accumulation via Structured Roles**

If two LLMs play 20Q / GUESS freely, early turns often involve "hitting a wall" or asking irrelevant questions, leading to fluctuating or regressive confidence—this would contaminate the experimental basis for multi-turn confidence evaluation. The Hinter-Guesser paradigm addresses this with structured roles: the Hinter (LLM) is assigned a secret entity and provides "helpful but non-trivial" hints each turn; the Guesser makes a best guess and performs **uniqueness probing**—even if the guess is correct, it marks "whether other candidates still fit the current evidence." This distinguishes "incidental correctness" from "information-based unique identification." Dialogues continue until the Guesser both guesses correctly and certifies uniqueness; non-convergent trajectories are discarded. The resulting data satisfies C1-C3 (monotonic information increase + step-by-step answerability + monotonic confidence). The final 20Q set contains 1848 turns / 226 entities, and GUESS contains 1625 turns / 223 entities. Since information is strictly cumulative, any observed confidence failures can be attributed to the method rather than data noise.

**2. InfoECE Metric: Normalizing Turn Indices into Information Progress to Enable Dialogue Comparison**

Dialogues vary in length, and using absolute turn indices $i$ for ECE makes short and long dialogues incomparable. InfoECE first normalizes each turn position into a fractional information level $s_{d,i} = i/L_d \in (0, 1]$ within the dialogue, then splits these into $B$ equal-width bins. Within each bin, the average confidence $\text{conf}_b$ and average accuracy $\text{acc}_b$ across all dialogues are computed:

$$\text{InfoECE} = \frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|.$$

After normalization to $[0,1]$, dialogues of different lengths are aligned to the same "information progress bar," allowing questions like "how confident should the model be at 50% information level" to have well-defined answers. Monotonicity is measured using Kendall’s $\tau = \frac{1}{N}\sum_d \frac{N^{(d)}_{con} - N^{(d)}_{dis}}{\binom{L_d}{2}}$, reflecting the increase of confidence over turns.

**3. P(SUFFICIENT) logit probe: Shifting Confidence Semantics from Accuracy to Sufficiency**

P(TRUE) is structurally flawed in under-specified scenarios—it only asks "is this answer correct." A best guess in turn 1 might be correct purely by luck among many candidates; P(TRUE) would assign high confidence despite low epistemic certainty. P(SUFFICIENT) retains the logit-based binary probe format of P(TRUE) but changes the prompt to "Based on the information above, is it sufficient to infer that the correct answer is $\hat{y}$?" The output is constrained to a single uppercase letter A (Sufficient) or B (Insufficient). Confidence is derived as $\text{confidence} = \Pr[\text{A} \mid p_{d,i}, \hat{y}_{d,i}]$. Thus, even if the model accidentally guesses correctly, as long as the hints have not excluded other candidates, confidence remains low—linking confidence to identifiability rather than incidental correctness. While the paper provides no formal proof, experiments validate this semantic shift: on GUESS, InfoECE drops from 79.97 (P(TRUE)) to 5.27, and $\tau$ rises from 3.29 to 81.51.

### Loss & Training
No models were trained; experiments used off-the-shelf open-source LLMs: Llama3.1 Instruct (8B / 70B) and Qwen2.5 Instruct (7B / 72B). Generation temperature was set to 1, and confidence estimation temperature to 0. Self-Consistency used $m=20$ samples. For fairness, each method first let the model answer to obtain $a$, then estimated the confidence of $a$.

## Key Experimental Results

### Main Results (InfoECE↓ / $\tau$↑, Example: Llama3.1-70B)

| Method | 20Q InfoECE | 20Q $\tau$ | GUESS InfoECE | GUESS $\tau$ | GRACE InfoECE | TRICKME InfoECE |
|--------|-------------|------------|---------------|---------------|---------------|-----------------|
| Vanilla-Verb | 59.63 | 17.60 | 65.52 | 16.92 | 39.06 | 47.47 |
| CoT-Verb | 58.39 | 34.49 | 70.16 | 18.24 | 96.04 | 80.97 |
| SC (m=20) | 32.99 | 28.98 | 56.88 | 2.59 | 15.91 | 19.90 |
| P(TRUE) | 67.82 | 40.82 | 79.97 | 3.29 | 37.04 | 35.62 |
| **P(SUFFICIENT)** | **13.05** | **48.43** | **5.27** | **81.51** | **11.52** | 23.16 |

P(SUFFICIENT) achieved the best InfoECE across most of the 4 datasets × 4 models. Notably, on GUESS, InfoECE was only **5.27** (vs. 79.97 for P(TRUE), a 15× improvement) and $\tau$ reached **81.51** (vs. 3.29 for P(TRUE)). When evaluating ground-truth answers, $\tau$ for all methods improved significantly; P(SUFFICIENT) reached $\tau = 93.91$ with Qwen2.5-72B on GUESS, proving that models can partially recognize whether hints align with the correct answer.

### Control Experiment: Placebo vs Informative turn (Llama3.1-70B on GUESS)

| Method | Conf at $i-1$ | Conf placebo at $i'$ | Conf informative at $i$ |
|------|---------------|----------------------|--------------------------|
| Vanilla-Verb | 71.30 | 73.70 (+2.40) | 83.83 (+12.53) |
| CoT-Verb | 78.39 | 78.77 (+0.38) | 88.41 (+10.02) |
| SC | 52.42 | 53.18 (+0.76) | 72.33 (+19.91) |
| P(TRUE) | 88.16 | 88.14 (−0.02) | 95.17 (+7.01) |
| **P(SUFFICIENT)** | 14.27 | **2.97 (−11.30)** | 27.58 (+13.31) |

P(SUFFICIENT) is the only method that **significantly decreases** confidence in the presence of a placebo (non-informative) hint, proving it tracks information rather than turn count. P(TRUE) showed increases of +11.75 / +14.61 on placebo turns for Llama3.1-8B / Qwen2.5-72B on GUESS, revealing a length artifact.

### Key Findings
- **Existing methods are generally uncalibrated in multi-turn settings**: Verbalized methods (VANILLA-VERB / COT-VERB) and P(TRUE) often show InfoECE between 40-80, far beyond reasonable levels. SC is the best default for calibration in fully-specified scenarios but often has single-digit $\tau$ in under-specified tasks.
- **P(SUFFICIENT) is overwhelmingly superior in under-specified tasks**: This is because the "information accumulation → candidate reduction" semantics in GUESS / 20Q perfectly match the sufficiency probe; while the advantage narrows in fully-specified datasets, it still leads (GRACE InfoECE 11.52 vs. SC 15.91).
- **Placebo experiments are a critical diagnostic tool**: Across 40 comparisons (5 methods × 4 models × 2 datasets), informative turns caused significant changes 27 times vs. only 18 for placebo turns, proving that confidence growth is **partially** driven by real information but also by turn count artifacts. P(SUFFICIENT) distinguishes these two best.
- **Multi-turn vs. Single-turn summary**: Summarizing multi-turn hints into a single-turn prompt showed <1% accuracy difference, suggesting no "get lost in conversation" effect like Laban et al. (2025) (due to the lack of complex reasoning). However, confidence behavior differed greatly—P(SUFFICIENT) dropped sharply in single-turn mode (e.g., Qwen2.5-7B on 20Q dropped from 63.13 to 13.23), indicating reliance on dialogue structure cues.
- **Model scale effects**: Increasing parameters significantly improved $\tau$ (Qwen2.5-72B on GUESS reached $\tau = 83.76$ for P(SUFFICIENT) vs. 51.44 for the 7B model), but the improvement in InfoECE was subtle, and sometimes smaller models showed better absolute calibration.

## Highlights & Insights
- **Shifting from "Correctness" to "Sufficiency"**: This is a profound semantic shift—P(TRUE) measures outcome correctness, while P(SUFFICIENT) measures epistemic identifiability. In scenarios where information is revealed step-by-step, the latter is the true signal for "rational confidence." Redefining probe semantics rather than stacking complexity is an elegant approach.
- **Hinter-Guesser Paradigm + Uniqueness Probing**: The use of structured roles + uniqueness certification elegantly solves the noise problem in multi-turn dataset construction, allowing evaluation to focus on the confidence methods themselves. This paradigm is generalizable to any multi-turn eval requiring strict info accumulation (e.g., multi-step reasoning, medical diagnosis).
- **Normalization Design of InfoECE**: Substituting absolute turn indices with fractional information levels makes ECE comparable across varying dialogue lengths. This normalization can be applied to any calibration evaluation for variable-length sequences (e.g., CoT steps).
- **Placebo Control Experiment**: Using "non-informative turns" as an adversarial baseline to separate "information-driven confidence" from "turn count artifacts" is a clever experimental design that should be standard in "dynamic confidence" research.
- **Cross-model Variance Reveals Differences in Robustness**: The Qwen2.5 series occasionally showed the highest $\tau$ on verbalized methods but poor absolute calibration, reminding researchers that "high $\tau$ does not equal usability"—InfoECE and $\tau$ must be viewed together.

## Limitations & Future Work
- The datasets are simplified information retrieval games, lacking real-world dialogue phenomena like topic switching, error recovery, or mixed intents; generalizability to open-domain dialogue is limited.
- Multi-turn evaluation is restricted to information-seeking tasks; confidence dynamics in creative collaboration or open generation remain unexplored.
- Evaluation only covers calibration and monotonicity; the actual value for downstream applications (e.g., triggering clarification or tool use) was not quantified.
- Only confidence was studied, not uncertainty—which is harder but potentially more important in agentic applications.
- Author observation: The advantage of P(SUFFICIENT) narrows in fully-specified scenarios (GRACE / TRICKME), suggesting its strength lies in leveraging "candidate reduction." If the initial candidate space is small, sufficiency becomes equivalent to truth, and the advantage diminishes.
- The impact of model scale + finetuning was not evaluated—how would SFT for sufficiency judgments affect results?
- Evaluation was limited to 4 open-source models; closed-source LLMs (GPT-4 / Claude) might exhibit different behaviors.

## Related Work & Insights
- **vs. Tian et al. (2023) Verbalized Confidence**: Proved that classic single-turn self-reported confidence is severely uncalibrated in multi-turn settings (InfoECE often 50+).
- **vs. Kadavath et al. (2022) P(TRUE)**: Proved that this standard logit probe is neither calibrated nor monotonic in under-specified multi-turn scenarios, as "correctness != sufficiency." P(SUFFICIENT) addresses this specific semantic flaw.
- **vs. Self-Consistency (Manakul et al. 2023)**: Best calibration baseline in fully-specified settings but has very low $\tau$ in under-specified tasks (single digits on GUESS), as sampling repeatedly confirms the same incorrect guess.
- **vs. Laban et al. (2025) "LLMs get lost in multi-turn"**: No "lost in conversation" effect was observed here since the task does not involve complex math, suggesting the effect is highly task-specific.
- **vs. Zhang et al. (2026) Conformity in Multi-turn Persuasion**: While Zhang discusses confidence resistance under persuasion, this work is complementary, focusing on confidence growth under cooperative information accumulation.
- **vs. Sung et al. (2025) GRACE / Wallace et al. (2019) TrickMe**: Directly used these incremental QA datasets as benchmarks for the fully-specified regime.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic multi-turn confidence evaluation. Methodology (InfoECE + Hinter-Guesser + P(SUFFICIENT) + placebo control) contains original contributions in every area.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 5 methods × 4 datasets + ground-truth $\tau$ + placebo control + multi/single-turn comparison + model scaling scan. Very thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear definitions, strong motivation, full prompts in appendix, well-organized charts.
- Value: ⭐⭐⭐⭐⭐ Establishes a foundational methodology for multi-turn LLM calibration. The "sufficiency vs. truth" insight has far-reaching implications for future confidence research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](../../AAAI2026/llm_nlp/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](../../ACL2025/llm_nlp/revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](../../ACL2025/llm_nlp/controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](../../ACL2025/llm_nlp/cer_confidence_enhanced_reasoning.md)

</div>

<!-- RELATED:END -->
