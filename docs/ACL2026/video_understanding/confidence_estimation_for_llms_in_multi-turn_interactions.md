---
title: >-
  [Paper Note] Confidence Estimation for LLMs in Multi-turn Interactions
description: >-
  [ACL 2026][Video Understanding][P(SUFFICIENT)] This paper presents the first systematic study of LLM confidence estimation in multi-turn dialogue scenarios. It proposes two core desiderata (per-turn calibration + monotonicity as information increases), the corresponding InfoECE metric and Kendall's $\tau$ evaluation, and the Hinter-Guesser dataset construction para
tags:
  - ACL 2026
  - Video Understanding
  - P(SUFFICIENT)
  - InfoECE
date: 2026-05-08
content_hash: ccebe6f3e377cff6
---
# Confidence Estimation for LLMs in Multi-turn Interactions

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.02179](https://arxiv.org/abs/2601.02179)  
**Code**: Paper mentions GitHub (link available in original text)  
**Area**: LLM Calibration / Dialogue / Confidence Estimation  
**Keywords**: Multi-turn dialogue, Confidence estimation, P(SUFFICIENT), InfoECE, Monotonicity

## TL;DR
This paper presents the first systematic study of LLM confidence estimation in multi-turn dialogue scenarios. It proposes two core desiderata (per-turn calibration + monotonicity as information increases), the corresponding InfoECE metric and Kendall's $\tau$ evaluation, and the Hinter-Guesser dataset construction paradigm. The authors introduce a novel P(SUFFICIENT) logit probe—finding that existing methods (verbalized / SC / P(TRUE)) exhibit poor calibration and monotonicity in multi-turn settings, while P(SUFFICIENT) reduces InfoECE to 5.27 on GUESS (vs. 79.97 for P(TRUE)) and achieves a $\tau$ of 81.51, although the task remains far from solved.

## Background & Motivation
**Background**: Confidence estimation is a core direction for mitigating LLM hallucinations. However, the vast majority of work (FActScore, Tian 2023, Xiong 2024) focuses on single-turn QA, assuming the input is a complete, one-off question.

**Limitations of Prior Work**: (1) Real-world human-computer interaction is multi-turn and incremental—users clarify needs over time, models ask follow-up questions, and the hypothesis space narrows gradually. Confidence behavior under such dynamic information accumulation remains unstudied. (2) It is unknown whether existing methods can maintain calibration or reflect the intuition that "more information leads to higher certainty" in multi-turn settings. (3) There is a lack of evaluation metrics and datasets tailored for multi-turn scenarios—standard ECE cannot handle differences in dialogue length, and standard QA datasets lack incremental information structures.

**Key Challenge**: In multi-turn scenarios, confidence should not be a "fixed attribute of a single response" but rather a "dynamic signal that increases as the dialogue evolves and information accumulates." Whether this can be achieved, by what methods, and how to measure it are questions without systematic answers.

**Goal**: (1) Formalize two desiderata for multi-turn confidence: per-turn calibration + monotonicity; (2) Design the length-normalized InfoECE metric and Kendall’s $\tau$ monotonicity metric; (3) Construct datasets suitable for multi-turn (under-specified scenarios using the Hinter-Guesser paradigm for 20Q / GUESS; fully-specified using existing GRACE / TrickMe); (4) Evaluate mainstream confidence methods and propose a new method, P(SUFFICIENT).

**Key Insight**: The authors observe that in under-specified scenarios, "correctness does not equal sufficient information"—a model might guess the correct answer by chance while multiple candidates remain unshared. In such cases, confidence should be low. P(TRUE) only asks "is the answer correct," failing to capture this lack of identifiability.

**Core Idea**: Change the semantics of the confidence probe from "is the answer correct (P(TRUE))" to "is the current information sufficient to uniquely determine the answer (P(SUFFICIENT))," aligning confidence with identifiability rather than incidental correctness.

## Method

### Overall Architecture
For each turn $i$ of every dialogue $d$, the model outputs an answer $\hat{y}_{d,i}$ and a confidence score $c_{d,i} \in [0, 1]$, and the correctness $z_{d,i} = \mathbb{I}[\hat{y}_{d,i} = y_d]$ is recorded. To eliminate variances in dialogue length, turn $i$ is normalized to an information level $s_{d,i} = i / L_d \in (0, 1]$ and divided into $B$ bins. Two core metrics are used: (a) **InfoECE** = $\frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|$, measuring calibration at each information level; (b) **Kendall's $\tau$**, measuring the degree to which confidence monotonically increases within a dialogue. These are tested across 5 confidence estimation methods (3 existing + 1 proposed) and 2 types of datasets (under-spec / fully-spec).

### Key Designs

**1. Hinter-Guesser Paradigm: Ensuring strict information accumulation with structured roles to make confidence failures attributable.**

If two LLMs play 20Q / GUESS freely, early turns often involve redundant questions or irrelevant clues, causing confidence to fluctuate or regress—which would contaminate the experimental basis for multi-turn evaluation. The Hinter-Guesser paradigm solves this with structured roles: The Hinter (LLM) is assigned a secret entity and provides "helpful but non-trivial" hints each turn; the Guesser provides a best guess and performs **uniqueness probing**—even if the guess is correct, it must mark "whether other candidates still fit the current evidence." This distinguishes "lucky guesses" from "information-based identification." Dialogues continue until the Guesser is both correct and confirms uniqueness; failed trajectories are discarded. The resulting data satisfies C1-C3 (monotonic information growth + step-by-step answerability + expected monotonic confidence), yielding 1848 turns/226 entities for 20Q and 1625 turns/223 entities for GUESS. Because information accumulation is strictly enforced, observed confidence failures can be attributed to the method itself rather than data noise.

**2. InfoECE Metric: Normalizing absolute turn indices into an information progress bar for comparable variable-length dialogues.**

Different dialogues have different lengths; using the raw turn index $i$ for ECE would make short and long dialogues incomparable. InfoECE first normalizes each turn position to a fractional information level $s_{d,i} = i/L_d \in (0, 1]$ and divides these into $B$ equal-width bins. Within each bin, average confidence $\text{conf}_b$ and average accuracy $\text{acc}_b$ are calculated across all turns from all dialogues, resulting in:

$$\text{InfoECE} = \frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|.$$

Once normalized to $[0,1]$, dialogues of varying lengths are aligned to the same "information progress bar," allowing questions like "how confident should the model be at 50% information level" to have well-defined answers. Monotonicity is measured using Kendall's $\tau = \frac{1}{N}\sum_d \frac{N^{(d)}_{con} - N^{(d)}_{dis}}{\binom{L_d}{2}}$.

**3. P(SUFFICIENT) Logit Probe: Switching confidence semantics from "correctness" to "sufficiency."**

P(TRUE) has a structural flaw in under-specified scenarios—it only asks "is this answer right," but a best guess at turn 1, even if correct, is merely a lucky pick among candidates. P(TRUE) would assign high confidence here, whereas epistemically, confidence should be low. P(SUFFICIENT) maintains the logit-based binary probe format of P(TRUE) but changes the prompt to "Based on the information above, is it sufficient to infer that the correct answer is $\hat{y}$?", forcing an output of a single uppercase letter A (Sufficient) or B (Insufficient). Confidence is then calculated as confidence $= \Pr[\text{A} \mid p_{d,i}, \hat{y}_{d,i}]$. Thus, even if the model accidentally guesses the right answer, as long as hints cannot yet exclude other candidates, confidence remains low—linking confidence to identifiability rather than incidental correctness. Experiments validate this semantic switch: on GUESS, InfoECE drops from 79.97 for P(TRUE) to 5.27, while $\tau$ rises from 3.29 to 81.51.

### Loss & Training
No models are trained; all experiments use off-the-shelf open-source LLMs: Llama3.1 Instruct (8B / 70B), Qwen2.5 Instruct (7B / 72B). Generation temperature is 1, and confidence estimation temperature is 0. Self-Consistency uses $m=20$ samples. To ensure fairness, all methods first allow the model to answer to get $a$, then estimate the confidence of $a$.

## Key Experimental Results

### Main Results (InfoECE↓ / $\tau$↑, Example: Llama3.1-70B)

| Method | 20Q InfoECE | 20Q $\tau$ | GUESS InfoECE | GUESS $\tau$ | GRACE InfoECE | TRICKME InfoECE |
|--------|-------------|------------|---------------|---------------|---------------|-----------------|
| Vanilla-Verb | 59.63 | 17.60 | 65.52 | 16.92 | 39.06 | 47.47 |
| CoT-Verb | 58.39 | 34.49 | 70.16 | 18.24 | 96.04 | 80.97 |
| SC (m=20) | 32.99 | 28.98 | 56.88 | 2.59 | 15.91 | 19.90 |
| P(TRUE) | 67.82 | 40.82 | 79.97 | 3.29 | 37.04 | 35.62 |
| **P(SUFFICIENT)** | **13.05** | **48.43** | **5.27** | **81.51** | **11.52** | 23.16 |

P(SUFFICIENT) achieves the best InfoECE in most cases across 4 datasets and 4 models, especially on GUESS where InfoECE is only **5.27** (vs. 79.97 for P(TRUE), a 15x improvement) and $\tau$ reaches **81.51** (vs. 3.29 for P(TRUE)). When evaluated on ground-truth answers, $\tau$ for all methods increases significantly, with P(SUFFICIENT) reaching $\tau = 93.91$ on Qwen2.5-72B (GUESS), proving models can partially recognize whether hints align with the correct answer.

### Control Experiment: Placebo vs. Informative turns (Llama3.1-70B on GUESS)

| Method | Conf at $i-1$ | Conf placebo at $i'$ | Conf informative at $i$ |
|------|---------------|----------------------|--------------------------|
| Vanilla-Verb | 71.30 | 73.70 (+2.40) | 83.83 (+12.53) |
| CoT-Verb | 78.39 | 78.77 (+0.38) | 88.41 (+10.02) |
| SC | 52.42 | 53.18 (+0.76) | 72.33 (+19.91) |
| P(TRUE) | 88.16 | 88.14 (−0.02) | 95.17 (+7.01) |
| **P(SUFFICIENT)** | 14.27 | **2.97 (−11.30)** | 27.58 (+13.31) |

P(SUFFICIENT) is the only method that **significantly decreases** confidence when presented with a placebo (uninformative "filler" hint), proving it tracks actual information rather than just turn count. P(TRUE) showed increases of +11.75 / +14.61 on GUESS for Llama3.1-8B and Qwen2.5-72B, revealing a length artifact.

### Key Findings
- **Existing methods are generally uncalibrated in multi-turn settings**: Verbalized methods (VANILLA-VERB / COT-VERB) and P(TRUE) often have InfoECE values between 40-80, far beyond reasonable levels. SC is the best default for calibration in fully-specified scenarios but often shows single-digit $\tau$ in under-specified ones.
- **P(SUFFICIENT) is overwhelmingly superior in under-specified scenarios**: This is because the semantics of "information accumulation → candidate reduction" in GUESS / 20Q perfectly match the sufficiency probe. Its advantage narrows but remains on fully-specified datasets (GRACE InfoECE 11.52 vs. SC 15.91).
- **Placebo experiments are a critical diagnostic tool**: Out of 40 comparisons (5 methods × 4 models × 2 datasets), informative turns caused significant changes 27 times vs. only 18 for placebos, proving confidence growth is **partly** driven by real information but also by turn-count artifacts. P(SUFFICIENT) distinguishes these two factors most effectively.
- **Multi-turn vs. Single-turn summary**: Summarizing multi-turn hints into a single-turn prompt showed <1% difference in accuracy, avoiding the "get lost in conversation" effect noted by Laban et al. (2025) (as these tasks are not complex mathematical reasoning). However, confidence behavior differed greatly—P(SUFFICIENT) dropped sharply in single-turn mode (e.g., Qwen2.5-7B on 20Q dropped from 63.13 to 13.23), indicating a reliance on dialogue structure cues.
- **Model scale effects**: Increasing parameters significantly improves $\tau$ (Qwen2.5-72B on GUESS reached $\tau = 83.76$ vs. 51.44 for 7B), but the improvement in InfoECE is more subtle; sometimes smaller models even show better absolute calibration.

## Highlights & Insights
- **Switching from "is it right" to "is it sufficient"**: This is a profound semantic shift—P(TRUE) measures outcome correctness, while P(SUFFICIENT) measures epistemic identifiability. In scenarios where information is revealed step-by-step, the latter is the true signal of "rational confidence." This approach of redefining probe semantics rather than stacking complexity is elegantly simple.
- **Hinter-Guesser Paradigm + Uniqueness Probing**: Using structured roles and uniqueness certification solves the messiness of multi-turn dataset construction, allowing evaluation to focus on the confidence methods themselves. This paradigm is generalizable to any multi-turn evaluation requiring strict information accumulation (e.g., multi-step reasoning, medical diagnostic dialogues).
- **InfoECE’s Length-Normalization**: Replacing absolute turn indices with fractional information levels within a dialogue makes ECE comparable across variable-length dialogues. This normalization can be transferred to calibration evaluations of any variable-length sequence (e.g., chain-of-thought steps).
- **Placebo Control Experiments**: Using the insertion of uninformative filler turns as an adversarial baseline to separate "information-driven growth" from "turn-count artifacts" is a clever experimental design that should be standardized for any study on "dynamic confidence."
- **Cross-model Variance Reveals Robustness Gaps**: The Qwen2.5 series occasionally shows high $\tau$ on verbalized methods but poor absolute calibration, reminding readers that "high $\tau \neq$ usability"; InfoECE and $\tau$ must be considered jointly.

## Limitations & Future Work
- The datasets reflect simplified information retrieval games, lacking phenomena like topic switching, error correction, or mixed intentions found in real dialogues; thus, transferability to open-domain dialogues is limited.
- Multi-turn confidence evaluation is restricted to information-seeking tasks; confidence dynamics in open-ended generation or creative collaboration are unstudied.
- Evaluation covers only calibration and monotonicity; the actual value for downstream applications (e.g., triggering a clarification question or a tool call) is not yet quantified.
- The study focuses on confidence rather than uncertainty—the latter is harder but potentially more important in agentic applications.
- Author's observation: The advantage of P(SUFFICIENT) narrows in fully-specified scenarios (GRACE / TRICKME), suggesting it inherently relies on "candidate reduction" for calibration. If the initial candidate space is already small, sufficiency becomes equivalent to truth, and the method's advantage diminishes.
- The impact of model scale + finetuning is not evaluated—what if models were SFT'd for sufficiency judgment?
- Evaluation is limited to 4 open-source models; closed-source LLMs (GPT-4 / Claude) might behave differently.

## Related Work & Insights
- **vs. Tian et al. (2023) Verbalized Confidence**: The classic method for single-turn self-reported confidence; this paper proves it is severely miscalibrated in multi-turn settings (InfoECE often 50+).
- **vs. Kadavath et al. (2022) P(TRUE)**: The benchmark for single-turn logit probes; this paper proves it is neither calibrated nor monotonic in under-specified multi-turn settings because "correctness $\neq$ sufficiency." P(SUFFICIENT) is designed specifically to fix this semantic gap.
- **vs. Self-Consistency (Manakul et al. 2023)**: The best calibration baseline for fully-specified scenarios but shows extremely low $\tau$ in under-specified ones (single digits on GUESS), as multiple samples simply confirm the same incorrect guess.
- **vs. Laban et al. (2025) "LLMs get lost in multi-turn"**: This paper finds **no** "lost in conversation" effect in its incremental information datasets, suggesting the effect is highly task-specific and potentially absent in non-mathematical tasks.
- **vs. Zhang et al. (2026) (Concurrent) Conformity in Multi-turn Persuasion**: Discusses confidence resistance under adversarial persuasion; this paper is complementary, focusing on confidence growth under cooperative information accumulation.
- **vs. Sung et al. (2025) GRACE / Wallace et al. (2019) TrickMe**: Directly adopts these incremental QA datasets as benchmarks for the fully-specified regime.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic multi-turn confidence evaluation; every methodology component (InfoECE + Hinter-Guesser + P(SUFFICIENT) + placebo control) is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 5 methods × 4 datasets + ground-truth $\tau$ + placebo controls + multi/single-turn comparisons + model scale scanning; highly exhaustive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formal definitions, strong motivation, full prompts provided in the appendix, and well-organized figures.
- Value: ⭐⭐⭐⭐⭐ Lays the methodological foundation for multi-turn LLM calibration; the "sufficiency vs. truth" semantic shift is a profound insight for future confidence research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)
- [\[ECCV 2024\] Benchmarks and Challenges in Pose Estimation for Egocentric Hand Interactions with Objects](../../ECCV2024/video_understanding/benchmarks_and_challenges_in_pose_estimation_for_egocentric_hand_interactions_wi.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](../../ICML2026/video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](../../NeurIPS2025/video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](../../NeurIPS2025/video_understanding/when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)

</div>

<!-- RELATED:END -->
