---
title: >-
  [Paper Note] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework
description: >-
  [ICLR 2026][LLM/NLP][human mobility generation] This paper proposes ELLMob, a self-aligned LLM framework grounded in Fuzzy-Trace Theory (FTT), which generates human mobility trajectories that balance everyday routines with event-driven responses by extracting and iteratively aligning "habitual pattern gists" with "event constraint gists."
tags:
  - ICLR 2026
  - LLM/NLP
  - human mobility generation
  - event-driven
  - LLM
  - self-alignment
  - Fuzzy-Trace Theory
date: 2026-05-08
content_hash: 6d34bd6ecdc5c28d
---

# ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework

**Conference**: ICLR 2026
**arXiv**: [2603.07946](https://arxiv.org/abs/2603.07946)
**Code**: [https://github.com/deepkashiwa20/ELLMob](https://github.com/deepkashiwa20/ELLMob)
**Area**: LLM/NLP
**Keywords**: human mobility generation, event-driven, LLM, self-alignment, Fuzzy-Trace Theory

## TL;DR

This paper proposes ELLMob, a self-aligned LLM framework grounded in Fuzzy-Trace Theory (FTT), which generates human mobility trajectories that balance everyday routines with event-driven responses by extracting and iteratively aligning "habitual pattern gists" with "event constraint gists."

## Background & Motivation

Human mobility trajectory generation aims to synthesize realistic spatiotemporal movement data, with broad applications in urban planning, traffic management, and public health. Existing LLM-based methods perform well for routine trajectory generation but exhibit two critical shortcomings when confronted with large-scale social events such as typhoons, pandemics, and the Olympics:

**Data Gap**: The absence of event-annotated mobility datasets makes it impossible to reliably evaluate existing models under irregular scenarios.

**Irreconcilable Decision Conflicts**: During events, human mobility is the result of a "competition" between habitual routines and event constraints—for example, during a typhoon, people still need to commute (habit) but avoid coastal areas (event constraint). Existing methods either follow habitual patterns entirely or are dominated by event constraints.

The core insight is drawn from Fuzzy-Trace Theory (FTT) in cognitive psychology: human decision-making under uncertainty is not based on precise probabilities but on the "gist" (i.e., the core meaning) of information. For instance, evacuating from a tsunami is not driven by a calculated 15% probability estimate, but by the gist that "the risk is very high."

## Method

### Overall Architecture

ELLMob consists of three core modules:

1. **Event Schema Construction**: Transforms unstructured event descriptions into structured representations.
2. **Trajectory Generation**: Generates candidate trajectories based on user history and event context.
3. **Reflection-based Alignment**: Reconciles conflicts through gist extraction and iterative alignment.

### Key Designs

**Event Schema Construction**: Raw event text (news reports, policy documents, etc.) is converted by the LLM into a four-dimensional structured representation:
- Event overview (type, name, time, affected area)
- Intensity and scale (quantitative indicators such as wind speed and rainfall)
- Infrastructure impact (transportation and public venue operational status)
- Official directives (government orders, travel advisories, and applicable scope)

**Three-Type Gist Extraction** (Core Innovation):
- **Pattern Gist**: Distills core behavioral patterns from a user's historical trajectories, including inertial anchors (e.g., returning home every evening) and vulnerable dependencies (e.g., reliance on a subway line that may be suspended).
- **Event Gist**: Distills primary impact intentions, behavioral implications (e.g., seek shelter away from the coast), and risk-benefit assessments from event context.
- **Action Gist**: Distills the main purpose, degree of habit adherence, and event compliance from LLM-generated candidate trajectories.

**Reflection-based Alignment Mechanism** (conflict reconciliation, not generic error correction):
- **Alignment Audit**: Candidate trajectories are evaluated along two dimensions—internal alignment (whether they reflect the user's habitual patterns) and external alignment (whether they reasonably respond to event constraints). A trajectory is accepted only when both criteria are satisfied.
- **Corrective Refinement**: If the audit fails, specific failure reasons are fed back to the trajectory generator to produce a revised version.
- A maximum of $K=3$ iterations are performed; if alignment is still not achieved, the last available trajectory in the buffer is accepted and unsatisfied constraints are reported.

### Dataset Construction

The paper constructs the first event-annotated mobility dataset, covering 1,100 users in the Tokyo metropolitan area across four scenarios:
- Typhoon Hagibis (2019.10.12–13): short-term natural disaster
- COVID-19 pandemic (2020.04.07–13): public health emergency
- Tokyo 2021 Olympics (2021.07.23–29): large-scale event during the pandemic
- Normal period (2019.09.01–30): baseline comparison

Data are sourced from Twitter and Foursquare check-in records, containing multi-dimensional information including timestamps, geographic coordinates, venue categories, and user comments.

### Loss & Training

ELLMob involves no model training and uses GPT-4o-mini as the inference engine:
- Temperature 0.1, Top-p=1, 10-minute temporal resolution
- Grid size $S=10$, maximum alignment iterations $K=3$
- Evaluation uses JSD (Jensen-Shannon Divergence) to measure the discrepancy between generated and real distributions

## Key Experimental Results

### Main Results

JSD is evaluated across four dimensions (↓ lower is better):

| Method | Typhoon SI↓ | Typhoon SD↓ | COVID SI↓ | COVID SD↓ | Olympics SI↓ | Olympics SD↓ |
|--------|------------|------------|-----------|-----------|-------------|-------------|
| DeepMove | 0.1697 | 0.0826 | 0.1838 | 0.0834 | 0.1667 | 0.0492 |
| LLMOB | 0.0949 | 0.1195 | 0.1013 | 0.1051 | 0.0973 | 0.0274 |
| LLM-Move | 0.1267 | 0.0392 | 0.1408 | 0.0567 | 0.1967 | 0.0298 |
| **ELLMob** | **0.0642** | **0.0200** | **0.1003** | **0.0444** | **0.0617** | **0.0061** |

ELLMob outperforms all baselines across all event scenarios: SI improves by 32.3% in the typhoon scenario, SD improves by 16.5% in the COVID scenario, and the overall average exceeds the strongest baseline by 46.9%.

### Ablation Study

| Variant | Typhoon SI↓ | COVID SI↓ | Olympics SI↓ | Notes |
|---------|------------|-----------|-------------|-------|
| w/o I.A.&E.A. | 0.1304 | 0.2331 | 0.1465 | Both alignments removed |
| w/o I.A. | 0.0835 | 0.1235 | 0.1355 | Internal alignment removed |
| w/o E.A. | 0.0680 | 0.2237 | 0.1392 | External alignment removed |
| w/o Eve. Ext. | 0.0736 | 0.2037 | 0.0686 | Event schema removed |
| **ELLMob** | **0.0642** | **0.1003** | **0.0617** | **Full model** |

Key finding: Removing external alignment degrades SI by 132.4% in the COVID scenario, demonstrating its critical role in scenarios requiring significant deviation from habitual patterns.

### Key Findings

1. **LLM-based methods consistently outperform traditional deep learning**: They show clear advantages in spatial consistency metrics (SD, SGD) due to their ability to incorporate event context.
2. **Two failure modes in existing LLM methods**: Either defaulting to habitual patterns (LLM-Move, LLMOB) or over-correcting (LLM-ZS completely suppresses social activities).
3. **Dual alignment is indispensable**: Internal alignment provides baseline plausibility; external alignment provides scenario-specific correction. The two modules operate in complementary directions.
4. **Disaster scenario application**: In a binary classification task identifying "active users" during the typhoon, ELLMob achieves the highest F1-Score with a recall of 59.3%.

## Highlights & Insights

1. **Cognitive theory-driven system design**: FTT is not merely a post-hoc explanation but guides the multi-gist decision framework, unified gist space, and the selection of interpretable attributes at the architectural level.
2. **Contribution in problem formulation**: This is the first work to formally define the task of "event-driven human mobility generation" and to provide the first multi-event annotated dataset.
3. **Innovation in self-alignment paradigm**: The framework reframes general LLM self-alignment from "error correction" to "conflict reconciliation," better reflecting the nature of event-driven mobility.
4. **Clear practical value**: The approach has direct applications in emergency response planning and traffic management.

## Limitations & Future Work

1. **Limited geographic scope**: Validation is confined to the Tokyo metropolitan area; despite supplementary experiments in Osaka, global generalizability remains to be examined.
2. **Data source bias**: Reliance on Twitter/Foursquare check-in data may introduce user population bias.
3. **Limited event types**: Only three event categories are covered; performance on other crisis types such as wars or economic crises remains unknown.
4. **Inference cost**: Iterative alignment requires multiple LLM calls; although $K=3$ represents a deliberate trade-off, it remains a bottleneck for large-scale trajectory generation.
5. **Fixed temporal granularity**: The 10-minute resolution may fail to capture finer-grained behavioral changes.

## Related Work & Insights

- **LLM-MOB / LLM-Move / LLMOB**: Prior LLM-based trajectory generation works that do not address event-driven scenarios.
- **FTT (Reyna & Brainerd, 1995)**: Provides the cognitive-theoretical foundation and inspires the gist extraction design.
- Insight: The "conflict reconciliation" paradigm in self-alignment is transferable to other LLM generation tasks involving multi-objective trade-offs (e.g., balancing safety and helpfulness).

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4.5 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4.0 |
| Writing Quality | 4.0 |
| Value | 4.0 |
| Overall | 4.0 |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Rethinking Uncertainty Estimation in LLMs: A Principled Single-Sequence Measure](rethinking_uncertainty_estimation_in_llms_a_principled_single-sequence_measure.md)
- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [\[ICLR 2026\] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [\[ICLR 2026\] AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer](assetformer_modular_3d.md)

<!-- RELATED:END -->
