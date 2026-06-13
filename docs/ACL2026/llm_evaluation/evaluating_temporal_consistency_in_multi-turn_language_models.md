---
title: >-
  [Paper Note] Evaluating Temporal Consistency in Multi-Turn Language Models
description: >-
  [ACL 2026][LLM Evaluation][Temporal Consistency] This paper introduces ChronoScope, an automatically synthesized evaluation set of 1.46 million multi-turn QA chains based on Wikidata. It specifically assesses whether LLM…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Temporal Consistency"
  - "Multi-turn QA"
  - "ChronoScope"
  - "Wikidata"
  - "present-day bias"
date: 2026-05-08
content_hash: 32926c808bb2e702
---

# Evaluating Temporal Consistency in Multi-Turn Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.23051](https://arxiv.org/abs/2604.23051)  
**Code**: <https://github.com/yashkumaratri/ChronoScope>  
**Area**: LLM Evaluation / Temporal Reasoning / Multi-turn Dialogue  
**Keywords**: Temporal Consistency, Multi-turn QA, ChronoScope, Wikidata, present-day bias

## TL;DR
This paper introduces ChronoScope, an automatically synthesized evaluation set of 1.46 million multi-turn QA chains based on Wikidata. It specifically assesses whether LLMs can "maintain previously implied temporal scopes" during multi-turn interactions. The study finds that strong models, including GPT-4 and Gemini-2.5, systematically exhibit "present-day drift," which worsens as interactions lengthen and cannot be eliminated even with oracle context.

## Background & Motivation

**Background**: Single-turn temporal question answering (such as TempQuestions, TimeQA, TimeR1, and PAT-Questions) has been extensively studied. However, these benchmarks typically provide "explicit time markers for every question"—models can correctly recall facts simply by identifying markers like "in 2010" in the prompt. In real-world multi-turn dialogues, users often set the temporal framework only in the first turn, expecting follow-up queries to default to that scope without repeating the year.

**Limitations of Prior Work**: LLMs exhibit highly unstable performance in these "implicit temporal inheritance" scenarios. While a model may possess the correct factual knowledge (e.g., answering correctly about the UK Prime Minister in 2010 in a single turn), it often switches to current 2024 answers once the context requires implicitly carrying the 2010 scope to the next sentence (e.g., "What policies did he lead?"). No existing benchmark systematically quantifies this failure mode of being "factually correct but temporally mismatched."

**Key Challenge**: Single-turn factual accuracy $\neq$ multi-turn temporal consistency. With unchanged parameters and knowledge bases, the model's interpretation of a query drifts during cross-turn inference. This reveals a failure in context binding at inference-time rather than a knowledge gap.

**Goal**: (i) Formalize "temporal scope stability" as a measurable multi-turn property; (ii) construct a benchmark to isolate this failure mode under controlled conditions; (iii) systematically quantify failure rates of SOTA models across four temporal patterns: implicit carryover, explicit switch, cross-entity transfer, and long trajectories.

**Key Insight**: The authors draw on the classic framework of Reichenbach (1947) featuring "speech time / event time / reference time" and Discourse Representation Theory. They treat the "temporal scope" as an implicit discourse state variable maintained across turns, which can be explicitly overridden, implicitly inherited, or transferred to related entities.

**Core Idea**: Utilizing time-qualified facts from the Wikidata knowledge graph combined with deterministic templates, the authors generate 1.46 million chains. Each chain is explicitly labeled with its scope transition pattern (11 chain families) and evaluated under three context settings to make "present-day bias" an independently measurable metric (Drift).

## Method

### Overall Architecture
The construction pipeline for ChronoScope is a two-stage, fully deterministic process without human writing or LLM generation: (i) Anchored Truth Table — for each snapshot year and anchor date, facts valid at that anchor are extracted from Wikidata claims (filtered by start time / end time / point-in-time) and deduplicated by QID; (ii) Chain Generation — attribute-specific templates convert anchored facts into natural language QA, which are then combined into multi-turn chains according to 11 chain family templates. Evaluation runs models under three context settings (Gold Context / Self-Conditioned / Questions Only) and measures performance using four metrics (Acc@1 / Final@1 / Chain@1 / Drift).

### Key Designs

1.  **Formalization and Tri-state Classification of Temporal Scope**:
    *   **Function**: Converts the vague concept of "implicit context carryover" into discrete states suitable for grading.
    *   **Mechanism**: Defines a chain as $\{(q_1,a_1),\dots,(q_L,a_L)\}$, where the first turn explicitly provides an anchor year (e.g., "In 2010"). Subsequent turns follow three evolution patterns: Persist (inheritance), Override (replaced by new time), or Transfer (migration to a related entity while retaining time). Each chain is categorized into one of 11 families, with quantitative measures for "Avg Scope Shift" and "Implicit Turns %."
    *   **Design Motivation**: Previous multi-turn QA benchmarks (HotpotQA / CoQA / Parrot) implicitly assume factual stability across turns. This paper explicitly distinguishes "scope states" to allow evaluation to precisely attribute failure modes—whether it is a failure to inherit time, failure to switch, or failure to transfer across entities.

2.  **11 Chain Families Covering the Full Temporal Pattern Space**:
    *   **Function**: Uses minimal yet comprehensive chain templates to probe each type of failure mode.
    *   **Mechanism**: *Carryover / Carryover-Then* test basic implicit inheritance; *Scope Switch* tests explicit override; *Cross-Entity Then* tests entity switching with invariant time; *Multi-Turn Chain* (3-6 turns) tests long-range stability; *Change Point* tests sudden explicit switches after multiple implicit turns; *Interval Reasoning / Interval Change / Distinct Count* test interval-based temporality; *Temporal Narrative* simulates chronicles; *Bridged Multi-PID* tests multiple attributes with fixed time.
    *   **Design Motivation**: A single template might be "overfitted" by models. The 11-family design enables both stress testing and attribution analysis—for instance, high failure rates in *Bridged Multi-PID* indicate model weakness under multi-hop constraints.

3.  **Three Context Settings + Drift Metric to Isolate Failure Modes**:
    *   **Function**: Decouples "knowledge absence" from "temporal drift" as mixed causes of failure.
    *   **Mechanism**: *Gold Context* provides the model with gold-standard answers for each turn (eliminating the impact of single-turn factual errors); *Self-Conditioned* uses the model's own previous predictions as context (compounding error propagation); *Questions Only* provides no context (the most severe). The *Drift* metric specifically measures cases where the model "answers incorrectly but provides the correct fact for the present day (2025)."
    *   **Design Motivation**: If a model exhibits high *Drift* even under *Gold Context*, it proves the issue is not error accumulation but the model's inherent inability to maintain an implicit temporal state—the most diagnostic experimental design in this paper.

### Loss / Training
This paper does not train any models; it is a dedicated evaluation study. All models are evaluated in a zero-shot setting. Details regarding prompting, decoding, sampling, and matching are provided in Appendix A.2.1. Higher values are better for Acc@1 / Final@1 / Chain@1, while lower values are better for Drift.

## Key Experimental Results

### Main Results

| Model | Gold Acc@1 | Gold Final@1 | Gold Drift | Self Final@1 | Self Drift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ChatGPT-4 | 0.441 | 0.516 | 0.163 | 0.353 | 0.215 |
| Gemini-2.5-Flash | 0.384 | 0.446 | 0.197 | 0.264 | 0.254 |
| ChatGPT-3.5 | 0.323 | 0.384 | 0.226 | 0.226 | 0.284 |
| Qwen-2.5-7B | 0.306 | 0.387 | 0.042 | 0.286 | 0.007 |
| Qwen-3-4B | 0.292 | 0.382 | 0.066 | 0.130 | 0.013 |
| DeepSeek-V3 | 0.247 | 0.276 | 0.081 | 0.291 | 0.008 |
| LLaMA-3.1-8B | 0.253 | 0.306 | 0.022 | 0.249 | 0.008 |

The strongest model, GPT-4, achieves a Final@1 of only 0.516 under Gold Context, with Drift as high as 0.163. Gemini-2.5-Flash exhibits a Drift near 0.20. While commercial LLMs are generally stronger, they show higher Drift, suggesting that RLHF training may bias them toward "answering with the most recent facts."

### Ablation Study

| Configuration / Failure Mode | Performance | Key Observation |
| :--- | :--- | :--- |
| Gold Context (Oracle) | GPT-4 Final@1=0.516, Drift=0.163 | Systematic drift persists even with gold-standard history. |
| Self-Conditioned | Final@1 generally drops by 0.1-0.2 | Error propagation amplifies scope drift. |
| Questions Only | Acc@1 < 0.2 across all models | Models almost entirely switch to present-day facts without context. |
| Open-source vs. Commercial | Drift: Qwen/LLaMA < 0.05, GPT-4/Gemini > 0.15 | Stronger models exhibit more severe present-day bias. |
| Long Chains | Final@1 monotonically decreases | Drift worsens as interaction length increases. |

### Key Findings
*   **Counter-intuitive Correlation Between Capability and Drift**: Small-to-medium open-source models (Qwen 7B / LLaMA 8B) generally have Drift < 0.05, whereas GPT-4 and Gemini-2.5 exceed 0.15-0.20. The authors hypothesize that larger models undergo more aggressive RLHF to "always be helpful and current," leading to a preference for 2024 data.
*   **Oracle Context Cannot Eliminate Drift**: Even when previous answers are injected as ground truth (Gold Context), Drift remains between 0.04 and 0.23. This is the most significant diagnostic conclusion—proving the issue is not retrieval or memory, but a defect in scope-binding capability during inference.
*   **Chain Length vs. Monotonic Drift**: From 2-turn to 11-turn chains, Final@1 continues to decline, indicating that the problem is not mitigated by the model "digesting more context" but is instead amplified.
*   **Extremely Low Chain@1 (generally < 0.10)**: Metrics requiring an entire chain to be correct show nearly total failure, indicating substantial room for improvement in multi-turn temporal consistency.

## Highlights & Insights
*   **Decoupling Present-day Bias from Hallucination**: Previous studies often categorized incorrect answers regarding recent facts as hallucinations. ChronoScope proves through the Drift metric that this is not a case of "not knowing," but "choosing the wrong time." This distinction is vital for future mitigation—fixing inference-time scope binding rather than just updating knowledge.
*   **Full Determinism + 1.46M Scale**: All QA pairs are derived from Wikidata and fixed templates without human or LLM intervention. This ensures perfect reproducibility, zero labeling costs, and scalability to any new snapshot.
*   **Activating Reichenbach's Framework**: Applying the "speech/event/reference" time framework to LLM evaluation is a sophisticated move that re-activates linguistic theory from half a century ago, suggesting the NLP community should refer back to discourse theories rather than reinventing concepts.

## Limitations & Future Work
*   The evaluation only covers factual QA and does not address non-factual information (e.g., opinions or plans) that changes over time, where scope binding might differ.
*   The Drift metric relies on finding a corresponding present-day answer; it cannot be measured for attributes without 2025 equivalents (e.g., the current occupation of a deceased individual).
*   No mitigation is provided—the authors identify the problem without offering a solution. Future work could explore explicitly injecting "today is 2010" in prompts or adding temporal-anchored preference data during RLHF.
*   The 11 chain families are manually designed and might miss certain scope evolution patterns in real conversations (e.g., negative temporality such as "before 2010 but not in 2009").

## Related Work & Insights
*   **vs. TimeQA / TimeR1 / PAT-Questions**: These are single-turn benchmarks with explicit time markers. ChronoScope treats temporal scope as an implicit cross-turn state variable, capturing entirely different failure modes.
*   **vs. Laban et al. (2025) / Parrot / MTSA**: These multi-turn follow-up evaluations focus on underspecified queries. This work focuses on the specific dimension of time, making it more controlled and diagnostic.
*   **vs. Continual Learning Perspective**: The authors relate "temporal scope stability" to an inference-time version of "continual knowledge retention." While continual learning binds old knowledge at the parameter level, this work focuses on binding context time at the inference level, providing a bridge for migrating mitigation methods.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ The first multi-turn evaluation set to treat "implicit temporal scope stability" as an independent measurement target.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluates 9 SOTA models across 3 settings and 11 chain families, though only in zero-shot settings without few-shot or CoT comparisons.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear concepts, logical progression of motivation, and the Drift metric is highly intuitive.
*   **Value**: ⭐⭐⭐⭐⭐ 1.46M automatically generated pairs, fully reproducible, exposing new failure modes in powerful models. Highly likely to be widely cited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference](march_evaluating_the_intersection_of_ambiguity_interpretation_and_multi-hop_infe.md)

</div>

<!-- RELATED:END -->
