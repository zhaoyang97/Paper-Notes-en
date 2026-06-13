---
title: >-
  [Paper Note] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection
description: >-
  [ACL 2026][Multi-Agent][Half-truth detection] The RADAR framework is proposed to detect half-truths based on omitted context through role-anchored (Politician vs. Scientist) multi-agent debate. Combined with a dual-thres…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Half-truth detection"
  - "Multi-agent debate"
  - "Omission reasoning"
  - "Role anchoring"
  - "Adaptive termination"
date: 2026-05-08
content_hash: c05229468a996179
---

# Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection

**Conference**: ACL 2026  
**arXiv**: [2604.19005](https://arxiv.org/abs/2604.19005)  
**Code**: [https://github.com/tangyixuan/RADAR](https://github.com/tangyixuan/RADAR)  
**Area**: Fact-checking / Misinformation Detection  
**Keywords**: Half-truth detection, Multi-agent debate, Omission reasoning, Role anchoring, Adaptive termination

## TL;DR

The RADAR framework is proposed to detect half-truths based on omitted context through role-anchored (Politician vs. Scientist) multi-agent debate. Combined with a dual-threshold adaptive early stopping mechanism, it consistently outperforms single-agent and traditional multi-agent baselines under noisy retrieval conditions.

## Background & Motivation

**Background**: Fact-checking systems have made progress in detecting explicit misinformation but remain blind to "half-truths"—statements that are factually correct but misleading due to the omission of key context. For example, "A politician reduced national debt by 15%" is correct but hides the fact that it was first increased by 20% during the same period.

**Limitations of Prior Work**: (1) Single-agent methods (encoder classifiers, instruction LLMs) perform single-pass reasoning and are prone to misjudgment when key context is missing. (2) Traditional multi-agent debate (MAD) uses fixed pro/con roles designed for explicit contradictions, which is unsuitable for omission reasoning—the core issue is missing context rather than opposing statements. (3) TRACER explicitly modeled omission for the first time but assumed gold evidence and used a single-agent pipeline.

**Key Challenge**: Omission detection requires reasoning about "what was not said" rather than "what is wrong"—existing verification systems look for contradictions rather than absences.

**Goal**: To design a fact-checking framework capable of discovering missing context under realistic noisy retrieval conditions.

**Key Insight**: Modeling verification as a structured debate between complementary roles—one constructing the best narrative (exposing motives for selective framing) and the other probing for omissions (revealing missing context).

**Core Idea**: Replace pro/con debates with "Politician" and "Scientist" role anchoring to transform omission detection from finding contradictions into active probing for missing context.

## Method

### Overall Architecture

RADAR consists of two stages: (1) constructing a shared evidence pool under noisy retrieval conditions; (2) reasoning through role-anchored multi-turn debate with adaptive early stopping. Three agents participate: Politician (constructs supportive narrative), Scientist (probes for omissions), and Judge (adjudicates and controls termination).

### Key Designs

1. **Role-Anchored Debate Protocol**:

    - **Function**: Discovers omitted context through complementary reasoning roles.
    - **Mechanism**: The Politician agent constructs the most persuasive supportive narrative from the evidence (tending toward confirmatory reasoning), while the Scientist agent examines missing, weak, or selectively presented information in the same evidence (tending toward analytical reasoning). The protocol includes opening statements $\rightarrow$ rebuttal rounds $\rightarrow$ concluding summaries. The Judge makes a ternary judgment (true/half-true/false) based on debate records and evidence.
    - **Design Motivation**: The Politician's role naturally tends toward selective presentation, and the Scientist's role naturally tends toward questioning omissions; their confrontation simulates the generation and detection mechanisms of half-truths.

2. **Dual-Threshold Adaptive Early Stopping Controller**:

    - **Function**: Ensures reasoning depth while reducing unnecessary debate rounds.
    - **Mechanism**: At the end of each round, the Judge calculates a stop margin $s = p(\text{STOP}) - p(\text{CONTINUE})$ and maximum label confidence $c = \max_y p(y)$. Debate terminates only when $s \geq \tau_s$ and $c \geq \tau_v$. Both thresholds are calibrated on the development set.
    - **Design Motivation**: A single threshold may stop prematurely in uncertain cases (especially half-truths). Dual thresholds require both "sufficient information" and "high-confidence judgment."

3. **Retrieval-Anchored Evidence Sharing**:

    - **Function**: Constrains the foundation of the debate under realistic retrieval conditions.
    - **Mechanism**: All agents share the same evidence pool (top-m retrieval results). Arguments must cite retrieved evidence rather than internal model knowledge. Divergent conclusions stem from reasoning differences rather than information asymmetry.
    - **Design Motivation**: Unlike traditional MAD which relies on internal knowledge, retrieval anchoring improves transparency and traceability.

### Loss & Training

RADAR is an unsupervised reasoning framework involving no training. Thresholds are calibrated on the development set.

## Key Experimental Results

### Main Results

Results on the PolitiFact-Hidden benchmark (under retrieval evidence conditions):

| Method | Accuracy | F1_macro | F1_HalfTrue |
|------|----------|---------|-------------|
| FIRE | 60.3 | 46.9 | 34.1 |
| D2D (MAD) | 63.0 | 50.9 | 39.7 |
| RADAR_single | 58.4 | 51.0 | 41.5 |
| **RADAR_multi (Ours)** | **77.7** | **63.3** | **56.5** |

### Ablation Study

| Configuration | Accuracy | Description |
|------|----------|------|
| Gold Evidence + RADAR | 83.6 | Upper bound with perfect retrieval |
| Retrieval Evidence + RADAR | 77.7 | Strong performance in realistic conditions |
| No Early Stopping | ~76 | Slight decline with increased cost |
| Fixed Pro/Con Roles | ~65 | Role design is critical |

### Key Findings

- RADAR achieves a 14.7% accuracy Gain over the best traditional method D2D under retrieval conditions, with significant advantages in half-truth detection (F1 improved from 39.7 to 56.5).
- Role anchoring is the core contribution: performance drops significantly when replaced with traditional pro/con roles, validating the necessity of complementary reasoning designs.
- Adaptive early stopping reduces the average number of debate rounds by approximately 30% without performance loss.
- Results consistently exceed baselines under both gold and retrieval evidence settings, demonstrating the robustness of the framework.

## Highlights & Insights

- The "Politician-Scientist" role metaphor is ingenious: half-truths are common in political discourse; using roles that simulate these discourse strategies to detect them creates a highly effective design philosophy.
- The dual-threshold early stopping mechanism is a practical engineering innovation: it balances reasoning cost and quality, which is particularly important for the inherently uncertain category of half-truths.
- The paradigm shift from "finding contradictions" to "discovering omissions" opens a new direction for the field of fact-checking.

## Limitations & Future Work

- Tested only on political fact-checking datasets; half-truth detection in other domains (science, healthcare) remains to be verified.
- Role design is effective but relies on manually defined prompt templates, which may limit generalizability.
- Retrieval quality remains a bottleneck—the ~6% gap between gold and retrieval evidence indicates that improvements in retrieval could lead to further Gains.
- Ternary classification (true/half-true/false) might be too coarse; degrees of half-truth in reality should ideally be continuous.

## Related Work & Insights

- **vs TRACER**: First omission detection framework but assumes gold evidence and uses a single-agent; RADAR achieves stronger performance under noisy retrieval via multi-agent debate.
- **vs D2D/TED**: Traditional MAD use fixed pro/con roles for explicit contradictions; RADAR's role anchoring targets omission reasoning, improving F1 by over 12 points.
- **vs FIRE**: Uses iterative search-verify loops but remains single-agent; RADAR achieves deeper reasoning through structured debate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ New paradigm for role anchoring and omission reasoning
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baseline comparisons, ablation studies, and efficiency analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation and intuitive role design
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in half-truth detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/multi_agent/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)

</div>

<!-- RELATED:END -->
