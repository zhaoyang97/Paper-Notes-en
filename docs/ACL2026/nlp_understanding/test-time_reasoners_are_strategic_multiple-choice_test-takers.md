---
title: >-
  [Paper Note] Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers
description: >-
  [ACL2026][NLP Understanding][Multiple-choice evaluation] This paper systematically compares the performance of 12 reasoning LLMs on full multiple-choice questions (MCQs) versus choices-only MCQs. It finds that test-time…
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Multiple-choice evaluation"
  - "test-time reasoning"
  - "partial-input"
  - "choices-only"
  - "reasoning traces"
date: 2026-05-08
content_hash: 4b1aa48c488980ca
---

# Test-Time Reasoners Are Strategic Multiple-Choice Test-Takers

**Conference**: ACL2026  
**arXiv**: [2510.07761](https://arxiv.org/abs/2510.07761)  
**Code**: https://github.com/nbalepur/mcqa-shortcuts  
**Area**: NLP Understanding / LLM Evaluation  
**Keywords**: Multiple-choice evaluation, test-time reasoning, partial-input, choices-only, reasoning traces

## TL;DR
This paper systematically compares the performance of 12 reasoning LLMs on full multiple-choice questions (MCQs) versus choices-only MCQs. It finds that test-time reasoning (TTR) does enable models to perform above chance in choices-only scenarios. However, reasoning traces reveal that this is not entirely due to shallow cheating, but involves "strategic test-taking" behaviors such as inferring missing questions, eliminating incorrect options, and invoking factual knowledge.

## Background & Motivation
**Background**: Multiple-choice questions remain one of the most common formats in LLM evaluation, with benchmarks from ARC and MMLU to Super GPQA relying on the "question + options + single correct answer" structure. With the rise of reasoning models, LLMs no longer just output options directly but generate long reasoning traces at test time before providing a final answer.

**Limitations of Prior Work**: Previous partial-input research found that models can significantly exceed random chance on MCQs even without seeing the question stem. This is typically interpreted as the presence of dataset artifacts or the model utilizing shallow cues like "longest option," "most specific option," or "unique numerical format." However, these conclusions mostly derive from non-reasoning models or simple perturbation experiments, making it difficult to discern if the model is opportunistic or genuinely reverse-engineering the question from the options.

**Key Challenge**: Success in choices-only scenarios may expose benchmark writing flaws on one hand, but on the other, it might reflect a legitimate capability for reasoning with partial information. For instance, a student who forgets the question during an exam might still increase their hit rate by eliminating obviously wrong items, identifying option categories, or guessing the intent of the original question. Categorizing all partial-input success as "cheating" might unfairly penalize non-shallow capabilities; ignoring it entirely allows genuine option artifacts to go unchecked.

**Goal**: The authors aim to answer two questions: whether test-time reasoning amplifies choices-only success, and what strategies the model's reasoning traces actually use when it answers correctly with only options—and whether these strategies necessarily imply a flaw in the benchmark or the model.

**Key Insight**: The paper looks beyond choices-only accuracy by combining the two axes of full/choices-only and base/reasoning, and further performs manual coding of choices-only reasoning traces. This approach allows for quantifying the impact of TTR while decomposing "correct answers" into different mechanisms such as shallow cues, factual recall, elimination, option pattern recognition, and inferring the missing question.

**Core Idea**: Treat reasoning traces as soft evidence to distinguish between harmful MCQ artifacts and less harmful strategic partial-information reasoning, rather than using choices-only accuracy as a single metric to declare a benchmark invalid.

## Method
The research design is controlled: rather than training new models, it performs controlled evaluations of existing reasoning LLMs on three MCQ benchmarks. Each item consists of a question stem $q$, a set of options $C$, and a correct answer $a$. Two input conditions are constructed: the "full" condition provides $q$ and $C$, while the "choices-only" condition provides only $C$, prompting the model to "use any strategy necessary" to guess the correct option. Each condition is further split into "base" and "reason" prompts: base requires a direct answer, and reason requires generating a step-by-step reasoning trace before selecting an answer.

### Overall Architecture
The evaluation covers three benchmarks of varying difficulty: ARC, MMLU, and Super GPQA. Models include Gemini 1.5 Lite / Flash / Pro, GPT-4o Mini / GPT-4o, Claude Haiku / Sonnet, Cohere Command R / R+, DeepSeek-V3, and Qwen2.5-72B-Instruct. For models supporting API reasoning effort, base is set to none and reason is set to medium; for those without it, reasoning is toggled via explicit CoT prompts. All settings measure accuracy based on whether the final selected option matches the gold label.

The experiment proceeds in three layers. First, it compares the gain in accuracy provided by TTR in full vs. choices-only settings. Second, it varies the reasoning effort of models like GPT-4o Mini to observe if longer reasoning further increases choices-only accuracy. Third, it performs qualitative coding of choices-only reasoning traces and uses regression to analyze which strategies correlate with correct or incorrect answers.

### Key Designs
1.  **2D Control of Full/Choices-only × Base/Reason**:
    *   **Function**: Separates normal MCQ capability, partial-input capability, and the gains brought by TTR.
    *   **Mechanism**: The same model and item are run under four conditions. If reasoning only improves the full setting and not choices-only, it suggests reasoning primarily utilizes the question stem; if choices-only reasoning also significantly improves, it suggests TTR may be enhancing option-level strategies.
    *   **Design Motivation**: Testing choices-only accuracy alone cannot determine if "reasoning models are more prone to cheating." A 2D comparison helps judge whether TTR generally improves MCQA or specifically amplifies partial-input shortcuts.

2.  **Trace Faithfulness Checks**:
    *   **Function**: Confirms that traces have a usable correlation with model answers before strategy analysis.
    *   **Mechanism**: Three types of faithfulness sanity checks were performed: the model mostly maintains the same answer after adding TTR; GPT-4o can predict the model's final choice with over 90% accuracy by looking only at the trace; and when the authors manually introduce duplicate, synonymous, nonsensical, or factually incorrect options, the model changes its answer or explicitly mentions the perturbation in the trace.
    *   **Design Motivation**: While CoT is not necessarily a true causal explanation, a trace that cannot even support the final answer is useless for strategy analysis. These checks position the trace as "informative soft evidence."

3.  **Choices-only Strategy Coding & Regression Analysis**:
    *   **Function**: Distinguishes which choices-only successes resemble benchmark flaws and which resemble legitimate partial-information reasoning.
    *   **Mechanism**: 180 correct/incorrect choices-only traces from ARC for Gemini Pro, Claude Sonnet, and Qwen-Instruct were qualitatively coded into six labels: FACT, ELIM, PATTERNS, INFER Q, SHALLOW, and INCONS. Logistic regression was then used to see if a strategy predicted success or failure, and INFER Q was analyzed to determine if the guessed question semantically matched the original.
    *   **Design Motivation**: The meaning of partial-input success depends on the strategy used. If a model is correct only because "1.5 looks the messiest," it is a benchmark flaw; if it infers "this is probably asking about renewable resources" from the options and chooses "trees," it is closer to abductive reasoning.

### Loss & Training
No new models were trained. All experiments are zero-shot prompting evaluations with temperature set to 1.0 and maximum output tokens at 8192. The appendix includes a comparison between SFT and GRPO using Qwen-2.5 Instruct 3B: SFT directly optimizes for the answer, while GRPO rewards traces leading to the correct answer; results consistently show that reasoning training does not drastically amplify choices-only success.

## Key Experimental Results

### Main Results
Test-time reasoning helps full MCQA significantly more than choices-only MCQA. In 36 model-dataset combinations, TTR improved accuracy in most full settings (significant in 25/36); in choices-only settings, improvement occurred in only 15/36. The gap between full and choices-only varies by task: it is larger on ARC/MMLU, indicating the question stem remains important, while on Super GPQA, some base models show smaller gaps, suggesting option cues are more prominent in high-difficulty problems.

| Evaluation Question | Key Findings | Interpretation |
| :--- | :--- | :--- |
| Does TTR improve full MCQA? | Improved in 25/36 combinations significantly (raw improvement in 27/36). | Reasoning is generally helpful in standard MCQA. |
| Does TTR improve choices-only? | Improved in 15/36 combinations. | Reasoning enhances partial-information answering, but much less than full. |
| Is choices-only above random? | All LLMs are significantly above random; GPT-4o reaches ~0.57 on ARC. | Modern models utilize option info or reverse-engineer questions. |
| Is reasoning length critical? | Increasing reasoning effort for GPT-4o Mini/Gemini/Claude lengthens traces but only slightly shifts choices-only accuracy. | Longer thinking $\neq$ stronger partial-input success; strategy is more key. |
| Do traces support answers? | GPT-4o predicts model options from traces with >90% accuracy. | Traces serve as soft evidence for strategy analysis. |

Prompt ablation shows that choices-only success is not an accidental byproduct of a specific prompt. Adding an "I don't know" (IDK) option slightly reduces accuracy, but models still consistently exceed the 0.25 random baseline. Accuracy persists even when using standard InspectAI prompts.

| Dataset | Model | Choices-only Base | + IDK | Choices-only Reason | Reason + IDK | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ARC | G-Flash | 0.5010 | 0.4880 | 0.5350 | 0.5075 | Remains significantly above random with IDK. |
| ARC | GPT-4o Mini | 0.4640 | 0.4273 | 0.5290 | 0.4848 | Reasoning helps, but IDK makes model conservative. |
| ARC | GPT-4o | 0.4910 | 0.4945 | 0.5180 | 0.5080 | Limited impact from prompt variations. |
| MMLU | G-Flash | 0.4650 | 0.4515 | 0.4530 | 0.4698 | IDK does not eliminate choices-only ability. |
| MMLU | GPT-4o Mini | 0.4258 | 0.3907 | 0.4920 | 0.4432 | Reasoning retains above-random option inference. |
| MMLU | Command-R | 0.3880 | 0.3700 | 0.4037 | 0.3840 | Even weaker models are not purely random. |

### Ablation Study
Qualitative coding reveals that choices-only reasoning does not rely solely on shallow cues. FACT, ELIM, PATTERNS, and INFER Q all potentially invoke the knowledge or explanation capabilities that MCQs are intended to measure; only SHALLOW represents a strategy closest to traditional artifacts. Regression results support this: on ARC, SHALLOW significantly predicts failure. On MMLU, no single strategy significantly predicts success/failure, suggesting a mix of non-shallow strategies in both correct and incorrect traces.

| Strategy | Meaning | Inherently Harmful? | Observation |
| :--- | :--- | :--- | :--- |
| FACT | Recalling facts related to options | Not necessarily | e.g., determining an option is a universal scientific fact. |
| ELIM | Eliminating obviously incorrect options | Not necessarily | Similar to partial knowledge guessing in humans. |
| PATTERNS | Identifying categories/patterns in options | Depends on item | Can help infer the question or expose poor homogeneity. |
| INFER Q | Guessing the original question | Mostly not | In success cases, ~80% of guessed questions were semantically close. |
| SHALLOW | Using surface cues like "messy number" | Yes | ARC regression coeff -0.701 (p=0.002); significantly predicts failure. |
| INCONS | Trace does not support final answer | Yes | Rare, but negative impact on ARC (p=0.067). |

Comparison of training strategies in the appendix further indicates that "inducing reasoning" does not simply equal "more shortcut utilization." Both SFT and GRPO for Qwen-2.5 3B allow choices-only performance to exceed random, but GRPO does not provide the same massive advantage in choices-only as it does in the full setting.

| Analysis Item | Result | Insight |
| :--- | :--- | :--- |
| ARC Policy Regression | SHALLOW significant negative; ELIM, FACT, INFER Q non-significant. | Shallow cues and inconsistency are failure signals, not success signals. |
| MMLU Policy Regression | No strategy significantly predicted success/failure. | For high-knowledge tasks, strategy alone doesn't explain performance. |
| Semantic Matching | Successful INFER Q matched original ~80% of time; failures only ~10%. | Successful choices-only often involves genuine abductive reasoning. |
| SFT vs GRPO | Both exceed random, but GRPO does not vastly outperform SFT here. | Reasoning training does not inevitably amplify partial-input shortcuts. |

### Key Findings
- Choices-only accuracy itself is not a sufficient diagnostic. Success can stem from shallow artifacts or from legitimate option knowledge, elimination, and question inference.
- TTR improves performance more stably in full MCQA, while only showing improvement in about half of choices-only scenarios, suggesting reasoning models are not simply "better at cheating."
- While reasoning traces are not perfect causal explanations, they are sufficiently informative for qualitative strategy analysis.
- Benchmark remediation should be strategy-oriented. If traces use SHALLOW outliers, options should be rewritten for homogeneity; if models succeed via INFER Q, the item itself is not necessarily "broken."

## Highlights & Insights
- The most valuable contribution is shifting partial-input research from "accuracy trials" to "strategy diagnosis." Success based on shallow outliers versus success based on reverse-engineering the context has opposite implications.
- Using reasoning traces for benchmark item debugging is practical. The authors demonstrate an outlier case where rewriting an option to match its category eliminated choices-only success, providing a direct workflow for MCQ revision.
- The paper takes a robust stance on CoT faithfulness. It does not claim traces are internal processes but demonstrates that, after sanity checks, they serve as actionable soft evidence.
- Impact on evaluation design: Future MCQA reports should not only show full accuracy but also report choices-only strategy distributions to isolate shallow artifacts from abductive reasoning.

## Limitations & Future Work
- The sample size for trace analysis is limited. Qualitative coding focused on three models with strong choices-only performance on ARC/MMLU, which may not cover all model families or subjects.
- Items are restricted to English MCQs. The role of option cues and context inference may differ in other languages or open-ended Q&A.
- Faithfulness remains a sanity check. Models might generate post-hoc rationalizations that look plausible; GPT-4o's ability to predict answers from traces doesn't prove the trace determined the answer.
- The work focuses more on evaluation analysis and does not propose an automated repair pipeline. Combining strategy classifiers with automated rewriting to mitigate shallow shortcuts is a logical next step.

## Related Work & Insights
- **vs. partial-input / hypothesis-only baselines**: Traditional studies use partial-input success to identify artifacts; this work further argues that such success is not monolithic and requires distinguishing between shallow cues and legitimate inference.
- **vs. MCQA benchmark research**: MCQs are often criticized for susceptibility to answer matching or outlier heuristics; this work provides a path to locate specific "bad" options using reasoning traces.
- **vs. CoT faithfulness research**: Even if CoT is not fully faithful, this study shows it can still aid in understanding model behavior and benchmark deficiencies as soft evidence.
- **vs. human test-taking strategy**: Comparing LLM choices-only behavior to human student "partial knowledge guessing" suggests that strategic answering shouldn't be automatically equated with cheating.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Problem isn't new, but the combination of TTR, choices-only, and trace coding is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 12 models, 3 benchmarks, and multiple faithfulness checks; manual coding scale remains a limitation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and restrained claims; some data is presented graphically rather than as raw tables.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for MCQA evaluation and benchmark remediation, preventing the misdiagnosis of legitimate strategic reasoning as simple flaws.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)
- [\[ACL 2026\] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](can_llms_estimate_cognitive_complexity_of_reading_comprehension_items.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)
- [\[ACL 2026\] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?](can_llms_estimate_cognitive_complexity_of_reading_comprehension_items.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)

</div>

<!-- RELATED:END -->
