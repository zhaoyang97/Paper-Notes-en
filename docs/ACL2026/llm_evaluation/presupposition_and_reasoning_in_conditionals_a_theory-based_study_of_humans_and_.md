---
title: >-
  [Paper Note] Presupposition and Reasoning in Conditionals: A Theory-Based Study of Humans and LLMs
description: >-
  [ACL2026][LLM Evaluation][Presupposition projection] This paper compares humans and four LLMs using a conditional presupposition projection task based on linguistic theory. It finds that humans jointly utilize probabilit…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Presupposition projection"
  - "conditional reasoning"
  - "pragmatic evaluation"
  - "human-AI comparison"
  - "LLM-as-a-Judge"
date: 2026-05-08
content_hash: 7309d9a1b24fc74d
---

# Presupposition and Reasoning in Conditionals: A Theory-Based Study of Humans and LLMs

**Conference**: ACL2026  
**arXiv**: [2605.18352](https://arxiv.org/abs/2605.18352)  
**Code**: https://github.com/proviso-bench/Presupposition-and-Reasoning-in-Conditionals  
**Area**: LLM Evaluation / Semantic-Pragmatic Reasoning  
**Keywords**: Presupposition projection, conditional reasoning, pragmatic evaluation, human-AI comparison, LLM-as-a-Judge  

## TL;DR
This paper compares humans and four LLMs using a conditional presupposition projection task based on linguistic theory. It finds that humans jointly utilize probability, antecedent-presupposition relevance, and contextual cues, whereas LLM rating similarity is significantly decoupled from theoretical reasoning quality; many human-like judgments from models may stem from surface-level pattern matching.

## Background & Motivation
**Background**: LLM evaluation for semantic and pragmatic capabilities is shifting from simple NLI or classification tasks toward more granular linguistic phenomena, such as implicature, presupposition, reference, and discourse context. Presupposition projection is particularly suitable as a stress test because it requires models to simultaneously handle formal semantics, pragmatic accommodation, world knowledge, and probabilistic reasoning.

**Limitations of Prior Work**: Existing presupposition benchmarks mostly focus on entailment or simple trigger identification, rarely comparing human behavioral data and model behavior within the same controlled experiment. More importantly, an LLM providing a final score close to human averages does not imply it follows linguistic theory; it might simply be capturing lexical co-occurrence or commonsense associations.

**Key Challenge**: The "proviso problem" in conditionals lacks a simple answer. For sentences like "If A, Bp", a listener may interpret the presupposition $p$ in the consequent as unconditionally true, or as true only when $A$ holds. This choice depends on the relationship between $Pr(p\mid c)$ and $Pr(p\mid A,c)$, the relevance between the antecedent and the presupposition, and how the context constrains possible worlds.

**Goal**: The paper aims to answer four questions: how humans use antecedent-presupposition relevance in conditional judgments; how close LLM Likert judgments are to humans; how minimal context affects both; and whether model explanations truly reflect presupposition projection and pragmatic reasoning.

**Key Insight**: The authors combine traditional psycholinguistic experiments with LLM benchmarks. They first conduct a norming study to construct three levels of probability (low/mid/high) and three levels of relationship (relevant/somewhat relevant/irrelevant). Subsequently, 120 human participants and four LLMs provide ratings (0-7) across the same 90 items, followed by a theory-driven checklist to evaluate model reasoning traces.

**Core Idea**: Instead of only asking "Are LLM answers human-like?", the study simultaneously compares behavioral distributions and theoretical reasoning processes to check for consistency.

## Method
The method constitutes a two-layer evaluation: the behavioral layer compares likelihood ratings for the target presupposition between humans and models; the explanatory layer uses LLM-as-a-Judge to verify if the model-generated reasoning satisfies key constraints from semantic/pragmatic theory.

### Overall Architecture
First, the authors construct 30 base propositions centered on possessive pronoun triggers (e.g., "Someone has a guitar / apron / boat / smartphone / sibling"). These cover high, medium, and low probability possession relations, paired with neutral contextual constraints.

Then, each proposition is expanded into four norming conditions: a baseline to measure $Pr(p\mid c)$, and high/medium/low antecedent relevance conditions to measure $Pr(p\mid A,c)$. Norming was completed by 30 native English speakers. After confirming monotonic separation between conditions, 90 conditional items were selected for the main experiment.

In the main experiment, 120 humans and four models (GPT-5, Gemini-2.5-flash, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) judged the likelihood of the target presupposition being true on a scale of 0 to 7. The experiment included "without-context" and "with-context" conditions (the latter providing a minimal background, such as where a person is from).

Beyond numerical judgments, LLMs were required to output step-by-step reasoning. Subsequently, Claude-Haiku-4 acted as a judge, using an expert-designed checklist to evaluate whether reasoning traces met theoretical standards. Finally, 5% of the outputs were sampled for manual verification by two Linguistics PhDs to validate the judge results.

### Key Designs
1. **Controlled Item Construction for Probability and Relevance**:
	- **Function**: Operationalizes the proviso problem (whether the antecedent makes the presupposition more likely) into experimentally controlled variables.
	- **Mechanism**: For each presupposition $p$, the baseline $Pr(p\mid c)$ and conditional probability $Pr(p\mid A,c)$ are estimated. If $A$ significantly increases the likelihood of $p$, a relevant condition is constructed; otherwise, somewhat relevant or irrelevant conditions are created.
	- **Design Motivation**: This moves beyond researcher intuition regarding "relevance," allowing human-LLM comparisons to be built on normed stimuli.

2. **Parallel Human and Model Behavioral Experiments**:
	- **Function**: Compares humans and models under the same tasks, scales, and contextual conditions.
	- **Mechanism**: Both groups provide 0-7 likelihood ratings for 90 "If A, Bp" sentences. The contrast between without-context and with-context conditions reveals how minimal information alters the integration of probability and relevance.
	- **Design Motivation**: Many pragmatic evaluations only look at binary correctness; using continuous ratings with Spearman/MAE is better suited for capturing nuanced behavioral patterns.

3. **Theory-driven Checklist for LLM-as-a-Judge**:
	- **Function**: Evaluates whether model reasoning aligns with semantic and pragmatic theory, rather than just the final score.
	- **Mechanism**: The checklist covers dimensions like Accuracy, Context, Pragmatic, Presupposition Handling, and Coherence (59 yes/no questions for with-context; 52 for without-context). The score $S$ for each response is the average proportion of satisfied checklist items: $S=\frac{1}{|K|}\sum_j J((c,s,\tau,r),\kappa_j)$.
	- **Design Motivation**: Final scores and explanation quality can decouple. This design tests if the model is performing theoretically sound presupposition reasoning or just outputting plausible-looking values.

### Loss & Training
The paper focuses on evaluation design rather than training new models. Statistical analysis utilizes linear mixed-effects models, with fixed effects including proposition probability, A-p relevance, and their interaction, using participant as a random intercept. For LLM generation, no self-consistency or repeated sampling was used; open-source models used temperature 0.7, top_p 0.9, and max_new_tokens 1024 with bfloat16 inference; closed-source models used identical settings via APIs. The judge model output binary judgments for each checklist item; manual validation showed an 89% exact match between human annotators and 79.46% agreement between humans and the judge.

## Key Experimental Results

### Main Results
The norming study first validated the stimuli: human ratings for low/mid/high probability items rose monotonically. Main experiment results showed that humans consider both the commonality of the proposition and whether the antecedent provides a reason for the presupposition to hold.

| Experimental Part | Key Result | Explanation |
|----------|----------|------|
| Norming low probability | M = 2.76 | Lowest ratings for low-probability possession |
| Norming mid probability | M = 4.38 | Median ratings |
| Norming high probability | M = 5.52 | Highest ratings |
| low → mid | $\beta=1.62$, $p<.001$ | Probability levels significantly increase ratings |
| low → high | $\beta=2.75$, $p<.001$ | High probability significantly separated from low |

Human mixed-effects results indicate that in without-context settings, irrelevant relationships strongly depress low/mid probability items; with-context, humans more stably distinguish between probability and relevance cues.

| Condition | Intercept / Clear Case | Negative Effect | Interaction Effect | Implication |
|------|---------------|------------|----------|------|
| With-context | Intercept 5.377 | low: -0.977, irrelevant: -0.377, somewhat relevant: -0.258 | low × irrelevant: -0.310, mid × irrelevant: -0.493 | Context allows for finer use of probability and relevance |
| Without-context | Intercept 5.340 | low: -0.347, irrelevant: -0.356 | low × irrelevant: -0.734, mid × irrelevant: -0.572 | Irrelevance acts more like a gating factor without context |

### Ablation Study
A clear decoupling between model behavior and explanation emerged. Qwen2.5-7B-IT was closest to humans in Likert ratings but had the lowest checklist reasoning compliance; larger closed-source models had more theoretically sound reasoning but were not necessarily closer to human ratings.

| Model | Human Alignment | MAE / Correlation | With-context checklist total | Observation |
|------|--------------|--------------|------------------------------|------|
| Qwen2.5-7B-IT | Most consistent human ranking | $\rho=0.25$ (w/o), $\rho=0.38$ (w/); MAE 1.32 | 39.08% | Human-like behavior, but weakest theoretical reasoning |
| Llama3.1-8B-IT | Moderate/stable alignment | $\rho=0.21$ (w/o), $\rho=0.30$ (w/); MAE 1.14 | 40.53% | Lowest MAE, yet checklist remains low |
| Gemini-2.5-flash | Dependent on context for alignment | With-context $\rho=0.26$; MAE 1.90 | 60.81% | Stronger reasoning, larger behavioral error |
| GPT-5 | Overall correlation insignificant | MAE 1.34 | 63.18% | Highest checklist score, but Likert distribution deviates from humans |

Checklist dimension results also indicate that LLMs do not improve across all dimensions; with-context settings sometimes introduce reasoning noise.

| Model | Accuracy | Pragmatic | Presupposition | Context Util. | Total |
|------|----------|-----------|----------------|---------------|-------|
| Gemini-2.5-flash | 61.39% | 82.84% | 61.11% | 30.10% | 60.81% |
| GPT-5 | 60.74% | 78.89% | 71.02% | 28.18% | 63.18% |
| Llama3.1-8B-IT | 16.67% | 54.94% | 49.35% | 14.75% | 40.53% |
| Qwen2.5-7B-IT | 22.96% | 56.42% | 42.04% | 12.42% | 39.08% |

### Key Findings
- Human judgment is not purely based on prior probability or relevance alone, but an interaction of both. Without context, irrelevant antecedents lead humans to fall back on the prior probability of the presupposition; with context, they integrate probability and relevance more consistently.
- Qwen2.5-7B-IT and Llama3.1-8B-IT are closer to humans in behavioral distribution, yet their checklist compliance is only about 39%-41%. This suggests "scoring like a human" does not equate to "reasoning via human-interpretable pragmatic theory."
- GPT-5 and Gemini show more compliant reasoning but do not mirror human Likert judgments more closely. The paper identifies this as a dissociation between behavior alignment and explicit reasoning quality.
- LLM-as-a-Judge costs are manageable: evaluating ~40,000 checklist items using Claude-Haiku-4 cost approximately 55 CAD, proving fine-grained pragmatic evaluation is accessible.

## Highlights & Insights
- The primary highlight is the deep integration of linguistic theory into benchmark design, operationalizing probability, relevance, and context variables.
- The behavior-explanation decoupling is insightful. While most LLM evaluations assume better CoT is always superior, this study shows that human-like behavior and theoretically sound explanations can be mutually exclusive.
- The checklist evaluation breaks down presupposition handling, context integration, and coherence rather than providing a single score. This design is transferable to tasks involving implicature, anaphora, or discourse coherence.
- The study cautions that small models might approximate human scores via surface-level distributional knowledge, while large models might achieve high checklist scores via fluent formal explanations, yet neither necessitates stable pragmatic competence.

## Limitations & Future Work
- The task scope is narrow, covering only "If A, Bp" structures and possessive pronoun triggers (simple existence presuppositions). Results may not generalize to factive verbs, change-of-state verbs, or complex nested structures.
- Human data is limited to English speakers; cross-linguistic and cross-cultural generalization remains unknown, as strategies for accommodation may vary.
- The LLM-as-a-Judge checklist primarily encodes satisfaction-theoretic and formal pragmatic principles, which may not capture the heuristic, probabilistic processes humans actually use.
- Human validation covered only 5% of outputs. Future work could expand manual verification, especially regarding judge bias across different checklist dimensions.
- CoT may be post-hoc rationalization. Low explanation quality doesn't prove an absence of capability, just as high quality doesn't guarantee the actual reasoning process was correct.

## Related Work & Insights
- **vs IMPPRES / NOPE / PROPRES**: These focus on entailment or trigger identification in classification; this paper moves toward graded likelihood judgments and human behavioral alignment.
- **vs CONFER / Proviso problem work**: These highlight LLM failures in complex conditionals; this paper uses normed probability/relevance and checklist analysis to explain these failures.
- **vs Pragmatics Understanding Benchmark**: While PUB offers broader coverage, this study is narrower and deeper, focusing on conditional projection with parallel human experiments.
- **vs General LLM-as-a-Judge**: Instead of overall preference scores, this study uses an expert-designed yes/no checklist to reduce evaluation variance and serve as a diagnostic tool.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines linguistic theory, human experiments, and reasoning diagnostics effectively.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Norming, human trials, multi-model comparisons, and judge validation are comprehensive, though triggers are limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear research questions and detailed methodology; tables strongly support conclusions.
- Value: ⭐⭐⭐⭐☆ Highly relevant for discussions on semantic-pragmatic evaluation and "behavior vs. reasoning" alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)

</div>

<!-- RELATED:END -->
