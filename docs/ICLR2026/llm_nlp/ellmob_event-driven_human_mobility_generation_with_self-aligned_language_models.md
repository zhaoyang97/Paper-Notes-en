---
title: >-
  [Paper Note] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework
description: >-
  [ICLR 2026][LLM/NLP][Human mobility generation] This paper proposes ELLMob, a framework grounded in Fuzzy-Trace Theory (FTT) from cognitive psychology. By extracting and iteratively aligning "habit gist" and "event gist…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "Human mobility generation"
  - "event-driven trajectory"
  - "LLM self-alignment"
  - "fuzzy-trace theory"
  - "cognitive decision-making"
date: 2026-05-08
content_hash: ae2400dc60e8287a
---

# ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework

**Conference**: ICLR 2026
**arXiv**: [2603.07946](https://arxiv.org/abs/2603.07946)
**Code**: [GitHub](https://github.com/deepkashiwa20/ELLMob)
**Area**: LLM/NLP
**Keywords**: Human mobility generation, event-driven trajectory, LLM self-alignment, fuzzy-trace theory, cognitive decision-making

## TL;DR

This paper proposes ELLMob, a framework grounded in Fuzzy-Trace Theory (FTT) from cognitive psychology. By extracting and iteratively aligning "habit gist" and "event gist," the framework reconciles the competition between users' routine patterns and social event constraints, enabling interpretable event-driven trajectory generation.

## Background & Motivation

Human mobility generation aims to synthesize plausible spatiotemporal trajectory data, with broad applications in urban planning, traffic management, and public health. While LLMs have achieved success in routine trajectory generation, two critical challenges remain:

1. **Evaluation bias from data scarcity**: Existing methods are predominantly developed and evaluated on non-event-day (stable-period) data, raising doubts about their reliability under sudden social events (natural disasters, public health emergencies).
2. **Lack of competing-decision reconciliation**: Real-world mobility during events combines habitual regularity with shock-induced deviation—users retain routine activities at key anchor points (e.g., workplaces) while adjusting other behaviors. Existing methods either default to habitual patterns or are dominated by event constraints.

Concrete manifestations:
- Typhoon: moving away from coastal areas, canceling non-essential commutes
- COVID-19: self-restricting activity range
- Olympics: restricted zones and traffic congestion

## Method

### Overall Architecture

ELLMob consists of three interconnected modules: (1) Event Schema Construction, which structures raw event narratives; (2) a trajectory generation module that leverages LLMs to produce candidate trajectories; and (3) gist-based reflective self-alignment, which iteratively reconciles competing decisions.

### Key Designs

**Event Schema Construction:**

Free-text event descriptions are transformed into structured representations along four dimensions:
- Event Profile: type, name, occurrence time, affected area
- Intensity & Scale: quantitative indicators such as wind speed and precipitation
- Infrastructure Impact: transportation and public facility operational status
- Official Directives: government orders, applicable populations, and geographic scope

**Three-Type Gist Extraction Based on Fuzzy-Trace Theory:**

| Gist Type | Attribute | Description | Example |
|-----------|-----------|-------------|---------|
| Pattern Gist | Core behavior | Primary behavioral pattern | Daily commute to office |
| | Inertial anchors | Deeply embedded, non-negotiable components | Returning home to a specific neighborhood at night |
| | Vulnerability points | Critical dependencies and single points of failure | Reliance on a single railway line that may be suspended |
| Event Gist | Primary intent | Core impact of the event on mobility decisions | High outdoor risk; strong incentive to stay home |
| | Behavioral influence | Survival, social dynamics, and compliance | Evacuating from coastal areas, seeking indoor shelter |
| | Risk-benefit assessment | Cost-benefit analysis of event-related risks | Injury risk outweighs benefit of non-essential outings |
| Action Gist | Primary intent | Main purpose driving trajectory choices | Procuring necessities from a nearby store |
| | Habit adherence | Degree to which habitual patterns are retained | Low: deviating from usual work commute |
| | Event compliance | Degree to which event constraints are followed | High: short trips avoiding hazardous areas |

**Reflection-based Alignment:**

A two-stage iterative process:

1. **Alignment Auditing:** Candidate trajectories are examined along two binary dimensions:
    - Internal Alignment: Does the trajectory reflect the user's intrinsic habitual mobility patterns?
    - External Alignment: Does the trajectory represent a reasonable, compliant response to event constraints?
    - A trajectory is accepted only when both criteria are satisfied.

2. **Corrective Refinement:** Upon failure, precise failure reasons are provided as feedback to guide regeneration. A maximum of $K=3$ iterations is allowed; upon timeout, the most recent valid trajectory from a buffer is used and unmet constraints are reported.

### Loss & Training

- Primary backbone: GPT-4o-mini (2025-01-01-preview)
- Temperature 0.1, Top-p 1
- Trajectory modeling at 10-minute temporal resolution
- Spatial grid parameter $S = 10$
- Maximum iterations $K = 3$ (determined via parameter study)

**Problem Formulation:**

$$F: (D_{\text{long-term}}^{(u)}, D_{\text{short-term}}^{(u)}, E_{ctx}) \mapsto \tau$$

- Long-term trajectory $D_{\text{long-term}}^{(u)}$: historical trajectories from an earlier pre-event period
- Short-term trajectory $D_{\text{short-term}}^{(u)}$: recent pre-event trajectories
- Event context $E_{ctx}$: structured event schema

## Key Experimental Results

### Main Results

**Method comparison across three events (JSD↓, lower is better):**

| Model | Typhoon SI | Typhoon SD | Typhoon CD | Typhoon SGD |
|-------|-----------|-----------|-----------|------------|
| LSTM | 0.1336 | 0.1039 | 0.0555 | 0.1111 |
| DeepMove | 0.1697 | 0.0826 | 0.0266 | 0.0759 |
| LLM-MOB | 0.1214 | 0.0468 | 0.0285 | 0.0344 |
| LLM-Move | 0.1267 | 0.0392 | 0.0136 | 0.0303 |
| LLMOB | 0.0949 | 0.1195 | 0.0123 | 0.0256 |
| **ELLMob** | **0.0642** | **0.0200** | **0.0041** | **0.0173** |

| Model | COVID SI | COVID SD | COVID CD | COVID SGD |
|-------|---------|---------|---------|----------|
| LLM-MOB | 0.1166 | 0.0532 | 0.0234 | 0.0353 |
| LLM-Move | 0.1408 | 0.0567 | 0.0127 | 0.0503 |
| LLMOB | 0.1013 | 0.1051 | 0.0186 | 0.0286 |
| **ELLMob** | **0.1003** | **0.0444** | **0.0080** | **0.0268** |

| Model | Olympics SI | Olympics SD | Olympics CD | Olympics SGD |
|-------|------------|------------|------------|-------------|
| LLMOB | 0.0973 | 0.0274 | 0.0110 | 0.0051 |
| LLM-Move | 0.1967 | 0.0298 | 0.0101 | 0.0057 |
| **ELLMob** | **0.0617** | **0.0061** | **0.0022** | **0.0035** |

**Key figures:** ELLMob outperforms the strongest baseline by 32.3% on SI under the typhoon scenario and by 16.5% on SD under COVID-19, with an average improvement of 46.9% over the strongest baseline.

### Ablation Study

| Variant | Typhoon SI | Typhoon SD | COVID SI | COVID SD |
|---------|-----------|-----------|---------|---------|
| Full ELLMob | 0.0642 | 0.0200 | 0.1003 | 0.0444 |
| w/o I.A.&E.A. | 0.1304 | 0.1270 | 0.2331 | 0.1077 |
| w/o I.A. (E.A. only) | 0.0835 | 0.0720 | 0.1235 | 0.0950 |
| w/o E.A. (I.A. only) | 0.0680 | 0.0258 | 0.2237 | 0.0860 |
| w/o Eve. Ext. | 0.0736 | 0.0273 | 0.2037 | 0.0741 |

Key ablation findings:
- Removing external alignment causes a 132.4% degradation on SI in the COVID-19 scenario—external alignment is critical for handling significant behavioral deviations.
- Removing internal alignment causes the model to over-correct (e.g., unreasonably increasing health-care-related trips).
- Cognitive self-alignment improves non-aligned variants by an average of 69.5%.

### Key Findings

1. **LLM-based methods consistently outperform deep learning methods**, particularly on spatial consistency metrics (SD, SGD), owing to their ability to integrate event context.
2. **Existing LLM baselines fail severely in event scenarios**: they either default to habitual patterns (underestimating health-related mobility) or over-respond to event constraints (completely suppressing social activity).
3. **Disaster activity classification**: ELLMob achieves the highest F1-score in identifying active users during typhoons (binary classification), with a recall of 59.3%.
4. **Internal and external alignment serve distinct roles**: internal alignment provides foundational plausibility, while external alignment delivers scenario-specific correction.

## Highlights & Insights

1. **Cognitively-grounded AI framework design**: Fuzzy-Trace Theory (FTT) is incorporated into LLM-based trajectory generation—this represents principled architectural design with a cognitive science foundation rather than simple prompt engineering.
2. **First event-annotated mobility dataset**: Covering three distinct event types (natural disaster / public health emergency / major sporting event), filling a significant data gap.
3. **Explicit reconciliation of competing decisions**: Trajectory generation is reframed from "maximizing statistical likelihood" to "cognitive plausibility," making the decision process traceable through gist alignment.
4. **Comprehensive experimental coverage**: 12 baseline methods (6 deep learning + 4 LLM + ablation variants), 4 evaluation metrics, and 4 scenarios.
5. **The average improvement of 46.9% is substantial**, maintained across all three event types.

## Limitations & Future Work

1. **Geographic limitation of data**: Only Twitter/Foursquare check-in data from the Greater Tokyo Area is used; generalizability remains to be verified (though supplementary experiments in Osaka are provided in the appendix).
2. **LLM API cost**: The iterative alignment process requires multiple API calls, resulting in high inference overhead.
3. **Manual design of event schema**: The four-dimensional event schema relies on domain expertise, limiting automation.
4. **Sparsity and bias of check-in data**: Social media check-ins cannot fully reflect real-world mobility.
5. **Coarse temporal resolution**: A 10-minute resolution may fail to capture fine-grained behavioral changes.

## Related Work & Insights

- **LLM-MOB** (Wang et al., 2023), **LLM-Move** (Feng et al., 2024), and **LLMOB** (Wang et al., 2024) serve as the primary LLM baselines.
- **Fuzzy-Trace Theory** (Reyna & Brainerd, 1995) provides the cognitive theoretical foundation—the fact that gist is expressible in language makes the integration of FTT with LLMs feasible.
- **Self-alignment/self-reflection**: Unlike general self-alignment methods for hallucination correction, the self-alignment in this paper focuses specifically on reconciling competing decisions.
- Insight: Cognitive science theories can provide principled guidance for LLM application architecture design, rather than relying solely on large-scale prompt engineering.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First event-driven mobility generation framework; FTT-gist alignment is a distinctively original design concept.
- **Technical Depth**: ⭐⭐⭐⭐ — The integration of cognitive theory with LLMs is rigorous, and the problem formalization is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 12 baselines, 4 scenarios, multi-dimensional evaluation, and comprehensive ablation.
- **Practicality**: ⭐⭐⭐⭐ — Direct application value for emergency management and urban planning.
- **Writing Quality**: ⭐⭐⭐⭐ — Framework diagrams are clear; the introduction of cognitive theory is well-executed.

**Overall**: ⭐⭐⭐⭐ (4.5/5) — A highly creative interdisciplinary work that organically integrates cognitive psychology with LLM-based trajectory generation. The problem definition is novel, experimental performance is outstanding, and the paper stands as an excellent representative of the LLM-for-Science research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [\[ICLR 2026\] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)
- [\[AAAI 2026\] IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization](../../AAAI2026/llm_nlp/irote_human-like_traits_elicitation_of_large_language_model_via_in-context_self-.md)
- [\[CVPR 2026\] GUIDE: Guided Updates for In-context Decision Evolution in LLM-Driven Spacecraft Operations](../../CVPR2026/llm_nlp/guide_guided_updates_for_in-context_decision_evolution_in_llm-driven_spacecraft_.md)
- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](gasp_guided_asymmetric_self-play_for_coding_llms.md)

</div>

<!-- RELATED:END -->
