---
title: >-
  [Paper Note] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection
description: >-
  [ACL 2026][Social Computing][Hate speech detection] The authors evaluated 5 LLMs (strongly aligned vs. weakly aligned) under 4 political personas on the Latent Hatred benchmark using zero-shot classification. They found that strongly aligned models achieved higher strict accuracy (69.0%) compared to weakly aligned ones (64.1%) and were nearly immune to persona manipulation. However, all models exhibited systematic failures in handling irony, target group fairness…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Hate speech detection"
  - "safety alignment"
  - "political persona"
  - "calibration"
  - "fairness"
date: 2026-05-08
content_hash: 9023202a92d69368
---

# Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection

**Conference**: ACL 2026  
**arXiv**: [2509.00673](https://arxiv.org/abs/2509.00673)  
**Code**: None (Data + reproduction bundle available in paper appendix)  
**Area**: LLM Safety / Alignment RLHF / Content Moderation  
**Keywords**: Hate speech detection, safety alignment, political persona, calibration, fairness

## TL;DR
The authors evaluated 5 LLMs (strongly aligned vs. weakly aligned) under 4 political personas on the Latent Hatred benchmark using zero-shot classification. They found that strongly aligned models achieved higher strict accuracy (69.0%) compared to weakly aligned ones (64.1%) and were nearly immune to persona manipulation. However, all models exhibited systematic failures in handling irony, target group fairness, and confidence calibration.

## Background & Motivation

**Background**: Automated hate speech detection is a critical capability for content moderation. RLHF alignment makes LLMs more "deployable" for such tasks. The community often uses "uncensored/censored" labels to frame discussions, but this description conflates "upstream training interventions" with "deployment-time guardrails + refusal heuristics + post-filtering."

**Limitations of Prior Work**: (1) Evaluation often excludes model refusals from accuracy calculations, making high-refusal models appear more "accurate"; (2) Prior work studied persona prompting or alignment failure separately, without cross-evaluating the two axes; (3) Self-reported confidence is used as a default threshold for human-in-the-loop triggers, yet its reliability is rarely rigorously examined.

**Key Challenge**: What users actually experience is "censorship-as-deployed" (the entire suite of guardrails during deployment), not isolated RLHF. Conflating "alignment" with "alignment + deployment filters" leads to incorrect conclusions—one makes the model more stable, while the other turns it into an ideological anchor.

**Goal**: Using a unified strict accuracy measure (counting refusals, truncations, and content filtering as errors), the study answers four RQs: the relationship between alignment and accuracy, directional bias from personas, alignment × persona interaction, and confidence calibration.

**Key Insight**: Using UGI leaderboard scores as a proxy for deployment-time alignment strength and LMArena Elo to control for general capability. All failure modes (refusal / null / filtering) are preserved in decomposition to avoid the "high refusal = high accuracy" illusion.

**Core Idea**: Treating "censorship-as-deployed" as the unit of analysis. The study jointly measures accuracy, persona sensitivity, target group fairness, and confidence calibration across a grid of 5 models × 4 personas × 3267 samples (65,340 responses total).

## Method

### Overall Architecture
The research is an observational audit rather than a new model: using a balanced subset of Latent Hatred (1089 explicit / 1089 implicit / 1089 not hate). Five LLMs (censored: o3-mini, Llama-3.1-405B; uncensored: GPT-4o, Mistral Medium, Mistral Large) were tasked with outputting JSON structured classifications (HATE / NOT_HATE / CANNOT_CLASSIFY) + confidence $\in [0,1]$ + reasoning text under zero-shot settings with 4 political personas (Progressive / Conservative / Libertarian / Centrist) as system prompts. Each sample was sampled once at $T=0.7$. All failure modes (regex fallback failure, truncation, content filtering, API errors, in-schema refusal) were retained as null predictions for strict accuracy calculations.

### Key Designs

**1. Censorship-as-deployed as a unit of analysis: Treating "actual guardrail strength experienced by the user" as a first-order variable**

The community often uses binary "uncensored/censored" labels, conflating upstream RLHF with deployment-time heuristics and post-filtering. The authors used UGI (Uncensored General Intelligence) scores as a continuous proxy for alignment strength (o3-mini=22.8 vs. Mistral Medium=56.77). General capabilities were matched using LMArena Elo (1317–1401) to isolate the alignment axis. As models differ in architecture and data, this observational design uses "associated with" instead of declaring causality.

**2. Strict accuracy + Dual-axis decomposition: Counting any response that cannot map to a binary label as an error**

Prior evaluations silently dropped unparseable rows, making high-refusal models appear more "accurate" while masking their avoidance of difficult problems. Ours counts any response failing to map to $\{\text{HATE}, \text{NOT\_HATE}\}$ (regex failure, truncation, filtering, API error, in-schema refusal) as an error for a fair strict accuracy headline. It reports misclassification rates and refusal/null rates separately (Fig. 1 + Tables 5/9) and provides "answered accuracy." This reveals that uncensored models' disadvantages stem from 24.2% refusal rather than incorrect answers.

**3. Persona × Alignment interaction measurement: Treating persona as ideological perturbation**

Prior work studied persona prompting or alignment failure separately. A Wald $\chi^2$ joint test was performed within a post-clustered logistic regression framework: within the censored group, the persona main effect $\chi^2(3)=3.34$ (non-significant); within the uncensored group, $\chi^2(3)=207.6$ ($p < 0.001$). The UGI × persona interaction $\chi^2(3)=101.3$ ($p < 0.001$) quantitatively confirms that ideological plasticity is concentrated at the weakly aligned end. Calibration was characterized using Expected Calibration Error $\text{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\,|\text{acc}(B_m)-\text{conf}(B_m)|$ and per-class overconfidence.

### Loss & Training
No models were trained—all LLMs were accessed via API inference with temperature 0.7, single sampling, and strict JSON schema constraints. Audit code, responses, and reproduction scripts were archived in the publication bundle (2026-04-20).

## Key Experimental Results

### Main Results
65,340 responses (5 models × 4 personas × 3267 posts), overall strict accuracy 66.1%, null rate 19.5%.

| Content Type | Censored Acc | Uncensored Acc | Difference |
| :--- | :--- | :--- | :--- |
| Explicit Hate | 0.760 | 0.914 | **uncensored +0.154** |
| Implicit Hate | 0.747 | 0.673 | censored +0.074 |
| Not Hate | 0.562 | 0.337 | censored +0.225 |
| **Overall** | **0.690** | **0.641** | censored +0.049 |

Error decomposition: Uncensored total error 35.9% (refusal 24.2% + misclass 11.7%), censored total error 31.0% (refusal 12.6% + misclass 18.5%). Conditioned on the answered subset, censored models actually made more errors (21.1% vs 15.4%).

### Ablation Study (Persona × Alignment Interaction)

| Configuration | Strict Acc | Description |
| :--- | :--- | :--- |
| Censored × Progressive | 0.688 | Strongly aligned stays stable |
| Censored × Libertarian | 0.686 | Nearly identical; fluctuation only 0.7pp |
| Uncensored × Progressive | 0.672 | Best persona for weakly aligned |
| Uncensored × Libertarian | 0.605 | Worst persona for weakly aligned; 6.7pp fluctuation |
| Implicit irony sub-class | 0.644 | Sub-class with lowest strict acc (35.6% total error) |
| Not-specified target | 0.363 | Target group bucket with worst fairness |
| Non-whites target | 0.912 | Target group bucket with best fairness (54.8pp gap) |

### Key Findings
- Censored models are not "better at judging" but "more willing to answer"—conditional accuracy was 5.7pp lower than uncensored models. The overall advantage comes entirely from lower null rates; deployment filters turn models into stable but ideologically fixed anchors.
- Persona steering is only significant in uncensored models (6.7pp vs. 0.7pp fluctuation). Progressive personas favor liberal bias (high FP), while Libertarian personas favor conservative bias (high FN).
- Irony is the biggest weakness for all models: 19.5% of the 35.6% total error rate is true misjudgment rather than refusal, indicating a failure in understanding rather than conservative refusal.
- Calibration disaster: Mean confidence for incorrect predictions remains as high as 80.1%–84.1%. In the not_hate class, 57.0% of incorrect answers have confidence > 0.80. Aggregate ECE (0.060) hides local calibration catastrophes.

## Highlights & Insights
- The **Strict accuracy + failure mode decomposition** framework is reusable: it separates "format fragility" from "ideological shift" without obscuring deployment experience.
- The concept of "**Censorship-as-deployed**" is key—separating "training alignment" from "deployment guardrails" explains contradictory phenomena (strongly aligned being more stable but more error-prone).
- The full grid design (5 models × 4 personas × 3267 samples) with all nulls retained is a pragmatic trade-off for large-scale social auditing under compute budgets.
- Quantifying a 54.8pp gap in target group fairness and a 22.5% refusal rate for conservatives turns LLM moderation unfairness into concrete "avoidance bias" metrics.

## Limitations & Future Work
- Observational design cannot prove causality: Models differ across many dimensions beyond alignment.
- Personas cover a narrow English Western political scope, missing socialist, populist, or non-Western divisions.
- Single sampling ($T=0.7$): Most 4–6pp gaps should be read as "conservative estimates" rather than precise values.
- Strict accuracy aggregates truncation, content filters, and API errors; future work should separate these failure semantics and include qualitative reasoning analysis.

## Related Work & Insights
- **vs. Zhang et al. 2024 (Latent Hatred + RLHF hypersensitivity)**: They noted RLHF makes models "hypersensitive" to implicit hate; Ours further quantifies this via strict accuracy decomposition.
- **vs. Yuan et al. 2025 (MBTI persona effects)**: They study steerability; Ours finds steerability is concentrated in uncensored models.
- **vs. Walsh & Joshi 2024 (Calibration vs. Accuracy)**: Ours provides evidence that self-reports cannot serve as human-in-the-loop thresholds in hate detection due to high-confidence errors.
- **vs. Dash et al. 2026 (Motivated reasoning)**: Ours is among the first to quantify motivated reasoning across the censorship axis.

## Rating
- Novelty: ⭐⭐⭐⭐ "Censorship-as-deployed" concept + triple-axis audit + full null retention are new, though individual method innovations are modest.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid 65,340 response grid, but lacks seed CI / paired tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptualization, honest limitations, and well-organized tables.
- Value: ⭐⭐⭐⭐⭐ Audit framework is reusable; fairness/calibration findings have actionable implications for trust & safety teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)
- [\[ACL 2026\] Explain the Flag: Contextualizing Hate Speech Beyond Censorship](explain_the_flag_contextualizing_hate_speech_beyond_censorship.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](../../ACL2025/social_computing/implihatevid_video_hate.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ACL 2026\] Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking](diagnosing_llm_arbitration_behavior_over_pre-evidence_epistemic_states_in_rag-ba.md)

</div>

<!-- RELATED:END -->
