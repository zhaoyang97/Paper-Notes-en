---
title: >-
  [Paper Note] SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs
description: >-
  [ICLR 2026][Medical NLP][Theory of Mind] SimpleToM reveals a critical deficiency in LLM Theory of Mind: while frontier models accurately infer others' mental states (Explicit ToM), their performance drops sharply when applying this knowledge to predict or judge behaviors (Applied ToM), exposing a significant gap between "knowing what" and "how to use what is
tags:
  - ICLR 2026
  - Medical NLP
  - Theory of Mind
date: 2026-05-08
content_hash: 46edd3d880cfff8f
---
# SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs

**Conference**: ICLR 2026  
**arXiv**: [2410.13648](https://arxiv.org/abs/2410.13648)  
**Code**: [https://github.com/yulinggu-cs/SimpleToM](https://github.com/yulinggu-cs/SimpleToM)  
**Area**: Human Understanding  
**Keywords**: Theory of Mind, LLM Social Reasoning, Explicit vs. Applied ToM, Information Asymmetry

## TL;DR
SimpleToM reveals a critical deficiency in LLM Theory of Mind: while frontier models accurately infer others' mental states (Explicit ToM), their performance drops sharply when applying this knowledge to predict or judge behaviors (Applied ToM), exposing a significant gap between "knowing what" and "how to use what is known."

## Background & Motivation

**Background**: LLMs are widely deployed as conversational agents; understanding others' beliefs (Theory of Mind) is essential to avoid catastrophic responses (e.g., ignoring emotional distress, misinterpreting sarcasm, or giving inappropriate advice in sensitive scenarios).

**Limitations of Prior Work**:
   - Existing ToM evaluations are often limited to Sally-Anne tasks or templated variants featuring narrow scenarios and limited types of information asymmetry.
   - Frequent use of explicit perceptual/mentalizing verbs like "sees" or "thinks" as triggers allow models to answer without genuine commonsense reasoning.
   - Most benchmarks only measure "Explicit ToM" (inferring mental states) while neglecting the application of that knowledge to behavior prediction or judgment.

**Key Challenge**: LLMs can correctly answer "Does Mary know the chips are moldy?" (Explicit ToM) but fail to infer "Will Mary pay for them or report them?" (Applied ToM). This suggests that LLM ToM knowledge is "decoupled" and cannot be reliably applied.

**Goal**:
   - Construct a benchmark covering multiple levels of ToM reasoning (Mental State $\rightarrow$ Behavior Prediction $\rightarrow$ Behavior Judgment).
   - Evaluate models across diverse daily scenarios beyond classic toy tasks.
   - Reveal and quantify the capability gap between explicit and applied ToM.

**Key Insight**: Leveraging 10 natural categories of information asymmetry (supermarkets, hospitals, second-hand markets, etc.), each story contains only two sentences but requires implicit commonsense reasoning across three questions of increasing depth.

**Core Idea**: A "separation of knowledge and action" exists in LLM ToM capabilities—models know what others do not know (Explicit) but fail to utilize this knowledge to predict and judge behaviors (Applied), even in simple daily contexts.

## Method

### Overall Architecture
SimpleToM seeks to decompose the question of "whether a model can use ToM" into measurable layers of capability. It consists of 1,147 minimalist stories, each only two sentences long, which embed a specific information asymmetry: a key fact known to the reader but unknown to a character (e.g., mold inside an opaque chip bag). Each story is paired with three progressive questions: first, the character's mental state (Explicit ToM: Does the character know?); second, the character's subsequent action (Applied ToM: What will they do next?); and third, a judgment of the character's actual behavior (Applied ToM: Is this reasonable?). This structure isolates "inference" from "application," allowing each to be scored independently. The dataset was produced via a pipeline of "seed stories + LLM expansion + rigorous human filtering" to ensure both scale and quality.

### Key Designs

**1. Three-Level Questions: Decomposing a story into "Inference $\rightarrow$ Prediction $\rightarrow$ Judgment"**

Using the moldy chips story as an example: The Mental State (MS) level asks "Does Mary know the chips are moldy?", requiring a simple Yes/No based on state inference. The Behavior Prediction level asks "Will Mary pay or report it?", requiring mapping the inferred state to a logical action (paying normally). The Behavior Judgment level asks "Mary paid; is this reasonable?", providing the character's action and asking for an evaluation. Judgment requires two layers of implicit reasoning: predicting what Mary *should* do based on her state, and matching it against the actual behavior.

**2. Implicit Information Asymmetry: Avoidance of "sees" or "thinks" trigger words**

Many ToM benchmarks include explicit cues like "character saw..." or "character thinks...", allowing models to bypass reasoning. SimpleToM avoids such clues: the first sentence states a fact ("The chips are moldy"), and the second describes an objective action ("Mary walks to the register"). The character's ignorance is entirely implicit—the model must use commonsense to realize humans cannot see through opaque bags.

**3. 10 Daily Scenarios: Covering physical occlusion, knowledge barriers, deception, etc.**

The dataset spans diverse domains: supermarket food, doctor-patient info, false labels, service industry "behind the scenes," container contents, unethical behavior, personal item containers, second-hand markets, hidden physical traits, and locked devices. High scenario variance ensures that the measured gap represents a general ToM phenomenon rather than an artifact of a specific context.

**4. Data Construction: LLM Expansion with Rigorous Human Filtering**

To balance scale and quality, SimpleToM uses a pipeline starting with human-written seed stories. Models like GPT-4 and Claude-3.5-Sonnet expanded these into ~3,600 candidates. Qualified human annotators then audited each story and answer key, eliminating ambiguous samples to produce 1,147 high-quality stories (3,441 evaluation instances).

## Key Experimental Results

Evaluated on 21 models using binary-choice questions (chance level = 50%).

### Main Results (Accuracy of representative models across levels, %)

| Model | Mental State (Explicit)↑ | Behavior Prediction (Applied)↑ | Behavior Judgment (Applied)↑ |
|------|------|------|------|
| GPT-3.5 | 36.5 | 7.6 | 29.1 |
| GPT-4o | 95.6 | 49.5 | 15.3 |
| GPT-4 | 96.6 | 63.0 | 19.5 |
| Llama-3.1-405B | 97.8 | 58.2 | 10.0 |
| Claude-3.5-Sonnet | 97.9 | 67.0 | 24.9 |
| GPT-4.5-preview | 97.0 | 67.8 | 26.7 |
| GPT-5 | 98.5 | 64.4 | 40.0 |
| DeepSeek-R1 | 97.3 | 73.8 | 65.8 |
| o1-preview | 95.6 | 84.1 | 59.5 |

### Test-Time Interventions (Accuracy on Behavior Prediction/Judgment, %)

| Model | Predict (Ref) | Predict (CoT) | Predict (MS Remind) | Judge (Ref) | Judge (CoT) | Judge (MS Remind) |
|------|------|------|------|------|------|------|
| GPT-4o | 49.5 | 62.8 | 82.8 | 15.3 | 39.2 | 42.2 |
| Llama-3.1-405B | 58.2 | 57.2 | 89.5 | 10.0 | 35.2 | 25.8 |
| Claude-3.5-Sonnet | 67.0 | 77.2 | 96.9 | 24.9 | 39.4 | 84.1 |

### Key Findings
- **Strong Inference, Weak Application**: Frontier models consistently score >95% on Mental State tasks, but performance drops to 50–70% for Behavior Prediction and as low as 10–25% for Behavior Judgment (e.g., Llama-3.1-405B at 10.0%, far below chance).
- **Depth Correlation**: Performance follows a strict MS > Behavior > Judgment pattern; Judgment is the hardest as it requires nested implicit reasoning.
- **Reasoning Models Help but Do Not Solve**: While o1-preview and DeepSeek-R1 achieve the best scores in applied tasks, they still show a significant gap compared to their mental state accuracy, indicating this is not a simple scaling issue.
- **High Scenario Variance**: Models behave inconsistently across scenarios. For instance, performance is high in medical scenarios (likely due to safety training regarding health), but fails in daily logistics or containers.
- **Interventions are Insufficient**: CoT and system prompts do not effectively close the gap; even with directly provided mental state answers (MS remind), Behavior Judgment often remains below 45% for several top models.

## Highlights & Insights
- **Conceptualization of "Decoupled ToM"**: The distinction between knowing a state (easy) and applying it to prediction (hard) or judgment (harder) provides a valuable hierarchical framework for evaluating cognitive abilities.
- **Methodological Innovation in Implicit Design**: By excluding mental state trigger words, SimpleToM forces models to rely on genuine commonsense reasoning rather than surface-level linguistic patterns.
- **Warning for Safety Deployment**: If models cannot reliably predict or judge human behavior based on what humans know, their deployment in sensitive social domains (counseling, customer service, education) warrants extreme caution.

## Limitations & Future Work
- Evaluation is limited to English; cross-cultural/linguistic ToM differences are unexplored.
- The multiple-choice format may underestimate issues present in open-ended generation.
- The two-sentence story format does not cover more complex, multi-turn conversational ToM.
- The impact of fine-tuning or RLHF specifically on closing the Explicit-Applied gap remains to be studied.

## Related Work & Insights
- **vs. Sally-Anne / BigToM**: Classical tests focus on explicit inference in simple toy tasks; SimpleToM extends this to diverse scenarios and applied tasks.
- **vs. SocialIQA**: General social reasoning benchmarks do not specifically target the information asymmetry or the hierarchical structure of ToM.
- **vs. FANToM**: While FANToM uses conversational formats and explicit labels, SimpleToM emphasizes implicit reasoning without linguistic mentalizing cues.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically distinguishes explicit/applied ToM for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive cross-model, cross-scenario, and intervention analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear logic from motivation to methodology.
- Value: ⭐⭐⭐⭐⭐ A milestone for evaluating LLM social reasoning with a publicly available dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](../../ACL2026/medical_nlp/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[ICML 2026\] Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference](../../ICML2026/medical_nlp/exploring_accurate_and_transparent_domain_adaptation_in_predictive_healthcare_vi.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](../../ACL2026/medical_nlp/promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)

</div>

<!-- RELATED:END -->
