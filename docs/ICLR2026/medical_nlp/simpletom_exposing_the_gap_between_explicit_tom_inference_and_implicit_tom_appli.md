---
title: >-
  [Paper Note] SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs
description: >-
  [ICLR 2026][Medical NLP][Theory of Mind] SimpleToM exposes a critical gap in LLMs' Theory of Mind capabilities: frontier models can accurately infer others' mental states (explicit ToM)…
tags:
  - "ICLR 2026"
  - "Medical NLP"
  - "Theory of Mind"
  - "ToM"
  - "LLM Social Reasoning"
  - "Explicit vs. Applied ToM"
  - "Information Asymmetry"
date: 2026-05-08
content_hash: 74fc4a7e4a0dfdbc
---

# SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs

**Conference**: ICLR 2026
**arXiv**: [2410.13648](https://arxiv.org/abs/2410.13648)  
**Code**: [https://github.com/yulinggu-cs/SimpleToM](https://github.com/yulinggu-cs/SimpleToM)  
**Area**: Human Understanding
**Keywords**: Theory of Mind, ToM, LLM Social Reasoning, Explicit vs. Applied ToM, Information Asymmetry

## TL;DR
SimpleToM exposes a critical gap in LLMs' Theory of Mind capabilities: frontier models can accurately infer others' mental states (explicit ToM), but performance drops sharply when this knowledge must be applied to behavior prediction and behavior judgment (applied ToM), revealing a substantial divide between "knowing what" and "knowing how to use what is known."

## Background & Motivation

**Background**: LLMs are widely deployed as conversational agents, and understanding others' beliefs (Theory of Mind) is essential for avoiding catastrophic responses—such as ignoring emotional distress, taking sarcasm literally, or providing inappropriate advice in sensitive situations.

**Limitations of Prior Work**:
- Existing ToM evaluations are largely limited to Sally-Anne tasks or templated variants—scenarios are narrow and cover only a restricted range of information asymmetry types.
- Explicit perception/mentalization verbs such as "sees" and "thinks" are used as trigger words, allowing models to answer without genuine commonsense reasoning.
- Nearly all evaluations measure only "explicit ToM" (inferring mental states), without testing whether models can apply that knowledge to behavior prediction or judgment.

**Key Challenge**: LLMs can correctly answer "Does Mary know the chips are moldy?" (explicit ToM), yet fail to correctly infer "Will Mary pay or report the mold?" (applied ToM)—suggesting that LLMs' ToM knowledge is "decoupled" and cannot be reliably applied.

**Goal**:
- Construct a benchmark covering multiple levels of ToM reasoning (mental state → behavior prediction → behavior judgment).
- Evaluate across diverse everyday scenarios rather than restricting assessment to classic toy tasks.
- Reveal and quantify the capability gap between explicit ToM and applied ToM.

**Key Insight**: Ten naturalistic information-asymmetry scenarios (supermarket, hospital, second-hand market, etc.); each story consists of only two sentences yet requires implicit commonsense reasoning; three question tiers progressively increase reasoning depth.

**Core Idea**: LLMs exhibit a "knowing–doing split" in ToM—they can identify what another agent does not know (explicit), but cannot leverage that knowledge to predict and judge behavior (applied), even in simple everyday scenarios.

## Method

### Overall Architecture
The benchmark comprises 1,147 concise two-sentence stories, each paired with three questions: (a) a mental state question (explicit ToM) → (b) a behavior prediction question (applied ToM) → (c) a behavior judgment question (deep applied ToM). Ten daily information-asymmetry scenarios are covered; the dataset is constructed via LLM-assisted generation followed by rigorous human filtering.

### Key Designs

1. **Three-Tier Question Design**:

    - **Mental State (MS)**: "Does Mary know the chips are moldy?" — directly queries mental state (yes/no).
    - **Behavior Prediction**: "Will Mary pay or report the mold?" — requires implicit mental-state inference before predicting behavior.
    - **Judgment**: "Mary paid. Is this reasonable?" — requires first implicitly predicting the expected behavior, then evaluating the rationality of the actual behavior (two layers of implicit reasoning).
    - **Design Motivation**: Each tier incrementally increases reasoning depth—(a) infer mental state only; (b) map mental state to behavior; (c) infer mental state → predict expected behavior → compare with actual behavior → judge rationality.

2. **Implicit Information Asymmetry**:

    - **Function**: In all stories, a character's lack of knowledge is conveyed implicitly, without explicit mentalization verbs such as "sees" or "thinks."
    - **Mechanism**: The first sentence introduces key information ("The chips are moldy"); the second introduces the character's action ("Mary picks up the chips and walks to the checkout")—the character's ignorance must be inferred through commonsense reasoning (e.g., one cannot see through a sealed can).
    - **Design Motivation**: Prevents models from exploiting trigger words, requiring genuine commonsense inference.

3. **Ten Everyday Scenarios**:

    - Supermarket food, medical information, false labeling, behind-the-scenes service industry, container contents, unethical behavior, personal item containers, second-hand markets, concealed physical traits, and locked devices.
    - **Design Motivation**: Each scenario corresponds to a distinct type of information asymmetry (physical occlusion, knowledge barriers, deception, etc.), ensuring evaluative diversity.

### Dataset Construction
- **Step 1**: One seed story is hand-written per scenario.
- **Step 2**: GPT-4 generates diverse stories based on each seed (varied entities and contexts).
- **Step 3**: Rigorous human filtering—annotators who pass a qualification test review each story for quality and answer correctness.
- **Final Scale**: 1,147 stories × 3 questions = 3,441 evaluation instances.

## Key Experimental Results

### Main Results (Accuracy Across Three Question Tiers)

| Model | Mental State↑ | Behavior↑ | Judgment↑ | Gap (MS−Judgment) |
|-------|--------------|-----------|-----------|-------------------|
| GPT-5 | ~95% | ~75% | ~65% | −30% |
| o1-preview | ~93% | ~70% | ~60% | −33% |
| Claude-3.5 | ~92% | ~72% | ~62% | −30% |
| Llama-3-70B | ~85% | ~60% | ~50% | −35% |
| GPT-4 | ~90% | ~68% | ~58% | −32% |

### Cross-Scenario Analysis

| Scenario | MS Accuracy | Behavior Accuracy | Gap |
|----------|-------------|-------------------|-----|
| Supermarket food | ~95% | ~80% | −15% |
| Medical information | ~90% | ~55% | −35% |
| Unethical behavior | ~88% | ~50% | −38% |
| Locked devices | ~92% | ~65% | −27% |

### Key Findings
- **Sharp drop of 25–35% from explicit to applied ToM**: Even GPT-5, which achieves ~95% on mental state inference, reaches only ~65% on behavior judgment—a striking gap.
- **Performance decreases monotonically with tier depth**: MS > Behavior > Judgment is a consistent pattern across all models.
- **High cross-scenario variance**: Behavior accuracy for the same model can vary by more than 30% across scenarios, indicating that ToM capability is highly context-dependent.
- **Scaling does not fundamentally resolve the issue**: GPT-5 outperforms Llama-70B, yet the gap remains large—this is not a simple scaling problem.
- **Chain-of-thought prompting provides limited benefit**: Instructing models to first infer mental state before predicting behavior yields only single-digit accuracy gains.

## Highlights & Insights
- **Conceptualizing the "knowing–doing split"** is highly valuable: LLMs' ToM is not a binary capability but exists on a continuum—knowing what another agent does not know (easy), applying that knowledge to prediction (harder), and applying it to judgment (hardest). This tiered evaluation framework is generalizable to assessments of other cognitive capabilities.
- **Implicit design that avoids trigger-word exploitation** is a key methodological contribution: many ToM benchmarks inadvertently provide cues (e.g., "sees," "thinks"), whereas SimpleToM mandates genuine commonsense inference.
- **A warning for safe LLM deployment**: If models cannot reliably predict and evaluate human behavior, their deployment in sensitive social applications—mental health support, customer service, education—warrants extreme caution.

## Limitations & Future Work
- Only English-language scenarios are evaluated; cross-cultural and cross-lingual ToM variation remains unexplored.
- The multiple-choice format may underestimate problems that would emerge in open-ended generation.
- The fixed two-sentence story format does not cover more complex, multi-turn conversational ToM.
- Whether fine-tuning or RLHF can close the explicit-to-applied ToM gap has not been explored.
- Although the ten scenarios are diverse, certain types of information asymmetry may still be absent.

## Related Work & Insights
- **vs. Sally-Anne / BigToM**: Classical ToM tests evaluate only explicit mental-state inference in narrow scenarios; SimpleToM extends evaluation to applied ToM across ten diverse scenario types.
- **vs. SocialIQA**: A social reasoning benchmark that does not focus on information asymmetry or the hierarchical structure of Theory of Mind.
- **vs. FANToM**: Also a ToM benchmark, but uses a dialogue format with explicitly annotated mental states; SimpleToM requires implicit reasoning throughout.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First systematic distinction between explicit and applied ToM, revealing a striking capability gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive multi-model, cross-scenario, ablation, and CoT analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from motivation to design to results is exceptionally clear, with vivid illustrative examples.
- Value: ⭐⭐⭐⭐⭐ — A milestone contribution to evaluating LLMs' social reasoning capabilities; dataset is publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](../../ACL2026/medical_nlp/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)
- [\[AAAI 2026\] MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](../../AAAI2026/medical_nlp/mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](../../ACL2026/medical_nlp/promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)

</div>

<!-- RELATED:END -->
