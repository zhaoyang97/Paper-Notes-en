---
title: >-
  [Paper Note] Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual Classification] This SemEval-2026 Task 9 system paper utilizes Gemma3-27B and 12 categories of English prompt variants to conduct online polarization detection…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual Classification"
  - "Polarization Detection"
  - "Prompt Engineering"
  - "SemEval"
  - "Social Issues"
date: 2026-05-08
content_hash: c4632b5f6b1c0570
---

# Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection

**Conference**: ACL 2026  
**arXiv**: [2606.03334](https://arxiv.org/abs/2606.03334)  
**Code**: None  
**Area**: Multilingual NLP / Social Issue Text Classification / Prompting  
**Keywords**: Multilingual Classification, Polarization Detection, Prompt Engineering, SemEval, Social Issues

## TL;DR
This SemEval-2026 Task 9 system paper utilizes Gemma3-27B and 12 categories of English prompt variants to conduct online polarization detection across 22 languages. It finds that prompt-only methods effectively handle coarse-grained binary classification but exhibit significant degradation in fine-grained multi-label tasks such as identifying polarization targets and manifestations.

## Background & Motivation
**Background**: Polarization detection in online socio-political discussions requires identifying whether text exhibits in-group/out-group opposition, hostility, exclusion, stigmatization, or denial of different groups. SemEval-2026 Task 9 extends this problem to multilingual, multi-cultural, and multi-event scenarios, decomposed into three task layers: presence of polarization, type of polarized target, and manifestation form.

**Limitations of Prior Work**: Polarization does not always manifest as direct hate or explicit attacks; it is often expressed through irony, cultural allusions, rhetorical questions, or ideological dog-whistles. Multilingual scenarios further amplify these difficulties due to varying discourse patterns, cultural backgrounds, and label distributions across different languages.

**Key Challenge**: LLMs can perform complex classification via prompting, yet prompt-only methods lack task-specific fine-tuning or language-specific adaptation. A distinct trade-off exists between high-precision conservative judgment and high-recall fine-grained identification. While coarse-grained judgment needs to minimize false positives, fine-grained multi-label tasks must capture implicit targets and rhetorical expressions.

**Goal**: Instead of training a new model, the authors systematically compare the impact of prompt design on multilingual polarization detection and submit a SemEval system covering all three subtasks.

**Key Insight**: The paper designs 12 short prompts ranging from simple to complex, progressively increasing terminological clarity, task definitions, reasoning steps, and in-context examples. It compares aya-101 and Gemma3-27B, ultimately selecting Gemma3-27B with the most instructional prompt for the test submission based on development phase performance.

**Core Idea**: Unify multilingual polarization analysis into an instruction-following classification problem, using prompt variants as control variables to observe how task complexity and linguistic differences affect the classification boundaries of the LLM.

## Method
The methodology is lightweight: no fine-tuning, no external data, and no data augmentation. Instead, all three subtasks are encapsulated as prompt-based inference. Its value lies in the systematic comparison of prompt forms and the reporting of cross-lingual error patterns.

### Overall Architecture
The input is a social media text $x$. Subtask 1 outputs a binary label determining if the text expresses attitude polarization; Subtask 2 outputs a multi-label vector tagging targets such as political, racial/ethnic, religious, gender/sexual orientation, or others; Subtask 3 outputs a multi-label vector tagging manifestations like stereotype, vilification, dehumanization, extreme language, lack of empathy, or invalidation.

The system first evaluates different prompt and model combinations on training data, then fixes the selected prompt and Gemma3-27B for submission on the official held-out test set. Evaluation metrics use macro-averaged F1 because all three tasks exhibit class imbalance, especially in multi-label tasks where minority labels significantly impact macro F1.

### Key Designs
1.  **12 Categories of Prompt Variants as Control Variables**:
    -   **Function**: Systematically analyze the impact of definitions, reasoning instructions, and examples within the prompt on classification results.
    -   **Mechanism**: Prompts 1-2 provide minimal context; Prompts 3-4 provide short task definitions; Prompts 5-6 explicitly define polarized/non-polarized boundaries; Prompts 7-8 require step-by-step analysis before answering; Prompts 9-12 include in-context examples.
    -   **Design Motivation**: Polarization judgment relies heavily on boundary definitions; terminological clarity in prompts directly shifts when the model predicts a positive class.

2.  **Unified English Prompt for Cross-lingual Reasoning**:
    -   **Function**: Use a single set of English prompts to cover 22 languages to test the cross-lingual generalization of multilingual LLMs.
    -   **Mechanism**: Prompts are written in English without per-language customization; the model relies on its multilingual capabilities to understand input text and output binary/multi-label sets.
    -   **Design Motivation**: Language-specific prompts are costly, and SemEval systems require rapid coverage of many languages. A unified prompt facilitates the comparison of linguistic differences and exposes cross-lingual weaknesses of prompt-only methods.

3.  **Conservative Decision-based Final Prompt**:
    -   **Function**: Reduce false positives in coarse-grained binary classification.
    -   **Mechanism**: The final prompt is biased towards predicting polarization only when there is clear, unambiguous evidence; this improves macro F1 in Subtask 1 but suppresses recall for implicit targets and subtle manifestations in Subtasks 2/3.
    -   **Design Motivation**: Official macro F1 is sensitive to both false positives and false negatives. A conservative boundary is more stable in coarse tasks, though this strategy contributes to performance degradation in finer tasks.

### Loss & Training
No model training or loss functions are used in this paper. The experimental strategy fixes the generation configuration, selects the prompt and model using training data, and then evaluates on the official test set. No external data or augmentation was used. Subtask 1 outputs a single binary label, while Subtasks 2/3 output a set of binary judgments for each category.

## Key Experimental Results

### Main Results
Official test set results indicate that performance decreases as task granularity increases. Subtask 1 covers binary classification for 22 languages; Subtask 2 covers multi-label target identification for 22 languages; Subtask 3 covers multi-label manifestation identification for 18 languages.

| Subtask | Task Format | Languages | Avg. Macro F1 | Avg. Accuracy | Main Conclusion |
|---------|-------------|-----------|---------------|---------------|-----------------|
| Subtask 1 | Polarization Binary | 22 | 0.762 | 0.819 | Prompt-only is effective for coarse detection |
| Subtask 2 | Target Multi-label | 22 | 0.587 | 0.678 | Significant drop when moving to target ID |
| Subtask 3 | Manifestation Multi-label | 18 | 0.444 | 0.498 | Most significant drop; requires abstract rhetoric ID |

### Ablation Study
The paper does not report individual development set scores for all 12 prompts, preventing a precise numerical ablation. Differences in prompt groups can be inferred from the analysis:

| Prompt Group | Information Added | Observed Effect in Paper | Specific Score |
|--------------|-------------------|--------------------------|----------------|
| Prompts 1-2 | Minimal/No context | Weakest baseline definition | Not reported |
| Prompts 3-4 | Short task definition | Helps model understand polarized label | Not reported |
| Prompts 5-6 | Explicit boundaries | Improves boundary judgment, reduces FPs | Not reported |
| Prompts 7-8 | Step-by-step reasoning | Strengthens analysis, but no recall guarantee | Not reported |
| Prompts 9-12 | In-context examples | Can improve performance, but introduces bias | Not reported |
| Final Prompt | Clear def + examples + conservative requirement | Best for Subtask 1; suppressed recall for Subtask 2/3 | See Main Results |

### Key Findings
- In Subtask 1, Chinese macro F1 reached 0.9211, Nepali reached 0.9180, while Italian was only 0.3459; the paper notes large cross-lingual variance in Subtask 1 (from ~0.92 to 0.35).
- For Hausa in Subtask 1, accuracy was 0.8936, but positive F1 was 0.0000 and macro F1 was 0.4719, indicating high accuracy may stem from systemic negative predictions.
- In Subtask 2, Chinese macro F1 was 0.8250, Urdu 0.7978, Hindi 0.7704; Hausa was 0.3022 and Italian 0.3409.
- In Subtask 3, Hindi macro F1 reached 0.7186, among the highest; Hausa was 0.1111 and Bengali 0.1609, showing manifestation identification is highly sensitive to language and label sparsity.
- Analysis suggests English performance is not necessarily dominant because English polarization often relies on irony, ideological shorthand, and implicit framing, which conservative prompts easily miss.

## Highlights & Insights
- The practical value of this system paper lies in demonstrating the boundaries of prompt-only multilingual classification: binary classification is feasible, but multi-label refinement degrades rapidly.
- The "conservative prompt" is a double-edged sword. It reduces false positives in Subtask 1 but misses weak explicit signals like irony, cultural allusions, and implicit exclusion in Subtasks 2/3.
- Linguistic performance is not solely determined by pre-training resource volume. Nepali and Chinese are strong in Subtask 1/2, likely because polarization is expressed through clear group references and topical markers; Hindi is strong in Subtask 3, potentially due to more direct mapping of rhetoric to labels in the corpus.
- This paper serves as a reminder for multilingual social NLP tasks: accuracy is misleading. Macro F1 and per-language recall are essential to reveal whether a system actually identifies the positive class.

## Limitations & Future Work
- The authors admit the final prompt is conservative; while precision is optimized, recall—especially in fine-grained tasks—is compromised.
- All prompts are in English without localization; future work could compare native-language prompts, bilingual prompts, or culture-specific instructions.
- Translation-based reasoning might stabilize model behavior but could weaken emotional and rhetorical intensity, particularly detrimental to manifestation identification.
- In-context examples improve performance but may introduce representation bias if they primarily originate from English or Western contexts.
- No complete numerical ablation for each prompt variant was provided, making it difficult to judge which prompt elements contribute most.

## Related Work & Insights
- **vs task-specific fine-tuning**: While fine-tuning relies on annotation costs, this paper demonstrates a rapid no-tuning prompt-only baseline; the advantage is scalability, while the disadvantage is insufficient fine-grained recall.
- **vs prompt tuning / PEFT**: This work involves no parameter updates, only natural language prompt modifications, leading to lower deployment costs but an inability to learn task-specific boundaries.
- **vs multilingual hate-speech detection**: Polarization detection is more covert than hate speech; many samples lack explicit insults, requiring focus on framing, group boundaries, and rhetorical dismissal.
- **Inspiration for future work**: One could ensemble conservative prompts with recall-enhanced prompts or automatically select native prompts by language to mitigate recall suppression in Subtasks 2/3.

## Rating
- **Novelty**: ⭐⭐⭐ (Primarily a system submission; limited methodological innovation.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Covers 3 subtasks and 22 languages, though prompt ablation lacks full numerical data.)
- **Writing Quality**: ⭐⭐⭐ (Structure is complete, but some expressions and table descriptions are coarse.)
- **Value**: ⭐⭐⭐⭐ (Highly referential for prompt baselines, error analysis, and task difficulty assessment in multilingual social NLP.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[ACL 2026\] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources](bhashasutra_a_task-centric_unified_survey_of_indian_nlp_datasets_corpora_and_res.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)

</div>

<!-- RELATED:END -->
