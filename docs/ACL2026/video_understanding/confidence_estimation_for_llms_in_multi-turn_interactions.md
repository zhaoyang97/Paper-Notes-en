---
title: >-
  [Paper Note] Confidence Estimation for LLMs in Multi-turn Interactions
description: >-
  [ACL 2026][Video Understanding][multi-turn dialogue] This paper presents the first systematic study of LLM confidence estimation in multi-turn dialogue scenarios. It proposes two core desiderata—per-turn calibration and…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "multi-turn dialogue"
  - "confidence estimation"
  - "P(SUFFICIENT)"
  - "InfoECE"
  - "monotonicity"
date: 2026-05-08
content_hash: d6329cc7b6dc96f8
---

# Confidence Estimation for LLMs in Multi-turn Interactions

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.02179](https://arxiv.org/abs/2601.02179)  
**Code**: GitHub mentioned in the paper (Link in original text)  
**Area**: LLM Calibration / Dialogue / Confidence Estimation  
**Keywords**: multi-turn dialogue, confidence estimation, P(SUFFICIENT), InfoECE, monotonicity

## TL;DR
This paper presents the first systematic study of LLM confidence estimation in multi-turn dialogue scenarios. It proposes two core desiderata—per-turn calibration and monotonicity as information increases—along with the corresponding InfoECE metric and Kendall's $\tau$ evaluation. It introduces the Hinter-Guesser data construction paradigm and a novel P(SUFFICIENT) logit probe. Results indicate that existing methods (verbalized / SC / P(TRUE)) exhibit poor calibration and monotonicity in multi-turn settings, while P(SUFFICIENT) reduces InfoECE to 5.27 on GUESS (vs. 79.97 for P(TRUE)) and achieves a $\tau$ of 81.51, although the task remains far from solved.

## Background & Motivation
**Background**: Confidence estimation is a central direction for mitigating LLM hallucinations, but the majority of work (FActScore, Tian 2023, Xiong 2024) focuses on single-turn QA, assuming the input is a complete, one-time question.

**Limitations of Prior Work**: (1) Real human-computer interaction is multi-turn and incremental—users clarify needs over multiple steps and models query repeatedly—yet confidence behavior under dynamic information accumulation remains unstudied. (2) It is unknown whether existing methods maintain calibration or reflect the intuition that "more information leads to higher certainty" in multi-turn contexts. (3) There is a lack of evaluation metrics and datasets tailored for multi-turn scenarios; single-turn ECE cannot handle varying dialogue lengths, and standard QA datasets lack incremental information structures.

**Key Challenge**: In multi-turn scenarios, confidence should not be a "fixed attribute of a single response" but rather a "dynamic signal that increases as the dialogue evolves and information accumulates." Whether this can be achieved, by what method, and how to measure it are questions without systematic answers.

**Goal**: (1) Formalize two desiderata for multi-turn confidence: per-turn calibration and monotonicity; (2) Design the length-normalized InfoECE metric and Kendall’s $\tau$ monotonicity metric; (3) Construct datasets suitable for multi-turn interactions (using the Hinter-Guesser paradigm for under-specified tasks like 20Q/GUESS, and existing GRACE/TrickMe for fully-specified tasks); (4) Benchmark mainstream confidence methods and propose the new P(SUFFICIENT) method.

**Key Insight**: The authors observe that in under-specified scenarios, "a correct answer does not imply sufficient information." A model might guess the correct answer while multiple candidates remain unexcluded, in which case confidence should remain low. P(TRUE) only asks "is the answer correct," failing to capture this lack of identifiability.

**Core Idea**: Shift the semantics of the confidence probe from "is the answer correct (P(TRUE))" to "is the current information sufficient to uniquely determine the answer (P(SUFFICIENT))," aligning confidence with identifiability rather than incidental correctness.

## Method

### Overall Architecture
For each turn $i$ of every dialogue $d$, the model outputs an answer $\hat{y}_{d,i}$ and a confidence score $c_{d,i} \in [0, 1]$, and records correctness $z_{d,i} = \mathbb{I}[\hat{y}_{d,i} = y_d]$. To eliminate dialogue length variance, turn $i$ is normalized into an information level $s_{d,i} = i / L_d \in (0, 1]$, divided into $B$ bins. Two core metrics are used: (a) **InfoECE** = $\frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|$, measuring calibration at each information level; (b) **Kendall’s $\tau$**, measuring the degree to which confidence increases monotonically within a dialogue. This is applied across 5 confidence estimation methods and 2 categories of datasets.

### Key Designs

1.  **Hinter-Guesser Paradigm (Dataset Construction)**:
    - **Function**: Addresses the "confidence backtracking" seen in naive 20Q setups where irrelevant early turns cause confidence to drop. It constructs dialogues satisfying C1-C3 (monotonic info increase, answerability at each step, and expected monotonic confidence).
    - **Mechanism**: (a) **QA Phase**—A Hinter (LLM) is assigned a secret entity and provides "helpful but non-trivial" hints each turn; the Guesser makes a best guess and performs uniqueness probing; (b) **Uniqueness Probing**—Even if the guess is correct, the model must flag "whether other candidates still fit the evidence," distinguishing "lucky guesses" from "information-based identification"; (c) **Stop + Filter**—The dialogue ends when the Guesser is both correct and confirms uniqueness. Only successful, converging trajectories are kept.
    - **Design Motivation**: Free-form 20Q simulations often lead to early turns hitting a wall or fluctuating confidence, undermining the evaluation. This paradigm uses structured roles and uniqueness signals to ensure strict info accumulation, ensuring method failures are attributable to the method rather than data noise.

2.  **InfoECE Metric (Length-normalized Multi-turn Calibration)**:
    - **Function**: Enables fair comparison across dialogues of different lengths within a single ECE framework.
    - **Mechanism**: Normalizes each turn index $i$ to a fractional info level $s_{d,i} = i/L_d \in (0, 1]$ and bins them into $B$ equal widths. Average confidence $\text{conf}_b$ and accuracy $\text{acc}_b$ are calculated across all dialogues for each bin. Monotonicity is measured via Kendall's $\tau = \frac{1}{N}\sum_d \frac{N^{(d)}_{con} - N^{(d)}_{dis}}{\binom{L_d}{2}}$.
    - **Design Motivation**: Directly using turn index $i$ makes long and short dialogues incomparable. Normalization to $[0, 1]$ aligns different dialogues on the same "information progress bar," making questions like "how confident should the model be at 50% info?" well-defined.

3.  **P(SUFFICIENT) Logit Probe (Core Contribution)**:
    - **Function**: Reflects whether information is sufficient to lock in a unique answer rather than simply whether the current guess is correct.
    - **Mechanism**: Similar to P(TRUE) as a logit-based binary probe, but the prompt is changed to "Based on the information above, is it sufficient to infer that the correct answer is $\hat{y}$?" The output is forced to a single letter 'A' (Sufficient) or 'B' (Insufficient). Confidence is $\Pr[\text{A} \mid p_{d,i}, \hat{y}_{d,i}]$.
    - **Design Motivation**: P(TRUE) has structural flaws in under-specified scenarios—the best guess at turn 1, even if correct, is just a "guess among many." P(TRUE) would yield high confidence, but epistemic confidence should be low. P(SUFFICIENT) shifts the semantics to identifiability. Experiments show InfoECE on GUESS drops from 79.97 (P(TRUE)) to 5.27, and $\tau$ rises from 3.29 to 81.51.

### Loss & Training
No models were trained; experiments used off-the-shelf Llama3.1 Instruct (8B / 70B) and Qwen2.5 Instruct (7B / 72B). Generation temperature was 1; confidence estimation temperature was 0. Self-Consistency used $m=20$ samples. For fairness, models provided an answer $a$ before their confidence in $a$ was estimated.

## Key Experimental Results

### Main Results (InfoECE↓ / $\tau$↑, Example: Llama3.1-70B)

| Method | 20Q InfoECE | 20Q $\tau$ | GUESS InfoECE | GUESS $\tau$ | GRACE InfoECE | TRICKME InfoECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Vanilla-Verb | 59.63 | 17.60 | 65.52 | 16.92 | 39.06 | 47.47 |
| CoT-Verb | 58.39 | 34.49 | 70.16 | 18.24 | 96.04 | 80.97 |
| SC (m=20) | 32.99 | 28.98 | 56.88 | 2.59 | 15.91 | 19.90 |
| P(TRUE) | 67.82 | 40.82 | 79.97 | 3.29 | 37.04 | 35.62 |
| **P(SUFFICIENT)** | **13.05** | **48.43** | **5.27** | **81.51** | **11.52** | 23.16 |

P(SUFFICIENT) achieved the best InfoECE in most cases across 4 datasets and 4 models. On GUESS, InfoECE was only **5.27** (a 15x improvement over P(TRUE)) and $\tau$ reached **81.51**.

### Placebo vs Informative turn (Llama3.1-70B on GUESS)

| Method | Conf at $i-1$ | Conf placebo at $i'$ | Conf informative at $i$ |
| :--- | :--- | :--- | :--- |
| Vanilla-Verb | 71.30 | 73.70 (+2.40) | 83.83 (+12.53) |
| CoT-Verb | 78.39 | 78.77 (+0.38) | 88.41 (+10.02) |
| SC | 52.42 | 53.18 (+0.76) | 72.33 (+19.91) |
| P(TRUE) | 88.16 | 88.14 (−0.02) | 95.17 (+7.01) |
| **P(SUFFICIENT)** | 14.27 | **2.97 (−11.30)** | 27.58 (+13.31) |

P(SUFFICIENT) was the only method that **significantly decreased** confidence upon receiving a placebo (no-info) hint, proving it tracks information rather than turn counts.

### Key Findings
- **Existing methods are generally uncalibrated in multi-turn settings**: Verbalized methods and P(TRUE) typically had InfoECE scores between 40-80. SC is the best default for fully-specified scenarios but often shows single-digit $\tau$ in under-specified tasks.
- **P(SUFFICIENT) is overwhelmingly superior in under-specified tasks**: Its "info accumulation → candidate reduction" semantics align perfectly with sufficiency probes. Although its lead narrows in fully-specified sets, it remains competitive.
- **Placebo experiments are a vital diagnostic tool**: Across 40 comparisons, informative turns caused significant changes 27 times vs. 18 for placebo turns, proving confidence growth is partially driven by information and partially by turn count artifacts.
- **Model Size Effects**: Increasing parameters significantly improves $\tau$ (Qwen2.5-72B reached $\tau=83.76$ on GUESS), but the improvement in InfoECE is more subtle; sometimes smaller models exhibit better absolute calibration.

## Highlights & Insights
- **Shifting from "Is it correct?" to "Is it enough?"**: This is a profound semantic shift—P(TRUE) measures outcome correctness, whereas P(SUFFICIENT) measures epistemic identifiability. In incremental info settings, the latter is the true signal of "rational confidence."
- **Hinter-Guesser Paradigm + Uniqueness Probing**: Solves the noise issue in multi-turn data construction and allows evaluation to focus on the confidence methods themselves. This paradigm is generalizable to any "strict info accumulation" task.
- **Length Normalization in InfoECE**: Using fractional information levels allows comparison across variable-length dialogues, a concept transferable to chain-of-thought step calibration.
- **Placebo Control**: Using nonsense turns as an adversarial baseline to decouple information-driven confidence from turn-count artifacts is an elegant experimental design that should be standard for "dynamic confidence" research.

## Limitations & Future Work
- The datasets are simplified information retrieval games, lacking real-world topic switching, error correction, or mixed intents.
- Restricted to information-seeking tasks; dynamic confidence in creative or open-ended generation is unstudied.
- Only covers calibration and monotonicity; the downstream utility (e.g., triggering clarification questions) is not quantified.
- Own Insight: The advantage of P(SUFFICIENT) diminishes in fully-specified scenarios, suggesting it relies on candidate reduction. If the initial candidate space is small, sufficiency and truth converge, and the method's advantage fades.
- Only evaluated on 4 open-source models; closed-source LLMs (GPT-4/Claude) may behave differently.

## Related Work & Insights
- **vs. Tian et al. (2023) Verbalized Confidence**: Proven to be severely uncalibrated in multi-turn settings (InfoECE often 50+).
- **vs. Kadavath et al. (2022) P(TRUE)**: Standard single-turn logit probe fails multi-turn calibration/monotonicity due to the "Correctness $\neq$ Sufficiency" gap.
- **vs. Self-Consistency (Manakul et al. 2023)**: Best baseline for fully-specified calibration, but low $\tau$ in under-specified tasks as sampling merely confirms the same incorrect guess.
- **vs. Laban et al. (2025) "LLMs get lost in multi-turn"**: This study found **no** "lost in conversation" effect in these specific tasks, suggesting the effect is highly task-dependent (e.g., complex math vs. info retrieval).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic evaluation of multi-turn confidence; every methodological component (InfoECE, Hinter-Guesser, P(SUFFICIENT)) is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 5 methods × 4 datasets + ground-truth $\tau$ + placebo controls + scale analysis; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear formal definitions, strong motivation, and transparent presentation of prompts and data.
- Value: ⭐⭐⭐⭐⭐ Establishes a foundational methodology for multi-turn LLM calibration; the "sufficiency vs. truth" insight is profound for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](../../ICML2026/video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](../../NeurIPS2025/video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](../../NeurIPS2025/video_understanding/when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)

</div>

<!-- RELATED:END -->
