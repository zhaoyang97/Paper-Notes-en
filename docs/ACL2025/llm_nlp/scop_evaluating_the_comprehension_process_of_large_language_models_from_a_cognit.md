---
title: >-
  [Paper Note] SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View
description: >-
  [ACL 2025][LLM (Other)][Reading Comprehension Evaluation] From a cognitive science perspective, SCoP decomposes the document comprehension process of LLMs into five progressive skills (Locating, Inferring, Connecting, Organizing, and Selecting). It constructs a test set containing 4,682 samples to evaluate the comprehension "process" rather than just the "answers." The study reveals that LLMs generally perform significantly better in local comprehension (~94%) than in global…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Reading Comprehension Evaluation"
  - "Cognitive Process"
  - "Comprehension Skills"
  - "LLM Reliability"
  - "Supporting Sentence Locating"
date: 2026-05-08
content_hash: b18c963278d25067
---

# SCoP: Evaluating the Comprehension Process of Large Language Models from a Cognitive View

**Conference**: ACL 2025  
**arXiv**: [2506.05000](https://arxiv.org/abs/2506.05000)  
**Code**: [SCUNLP/SCOP](https://github.com/SCUNLP/SCOP)  
**Area**: LLM / NLP  
**Keywords**: Reading Comprehension Evaluation, Cognitive Process, Comprehension Skills, LLM Reliability, Supporting Sentence Locating

## TL;DR

From a cognitive science perspective, SCoP decomposes the document comprehension process of LLMs into five progressive skills (Locating, Inferring, Connecting, Organizing, and Selecting). It constructs a test set containing 4,682 samples to evaluate the comprehension "process" rather than just the "answers." The study reveals that LLMs generally perform significantly better in local comprehension (~94%) than in global comprehension (~31%), and that their comprehension processes can be flawed even when the final answers are correct.

## Background & Motivation

**Background**: Reading comprehension is one of the most fundamental capabilities of LLMs. Existing evaluations (such as SQuAD, HotpotQA, and RACE) measure comprehension capability by matching model outputs with standard answers. On these benchmarks, top LLMs have approached or even surpassed human performance. However, in high-stakes scenarios like law, medicine, and education, researchers and practitioners still hesitate to fully trust the comprehension results of LLMs.

**Limitations of Prior Work**: Existing evaluations only focus on the "results" of comprehension (whether the answer is correct), completely ignoring the "process" of comprehension. Models might generate correct answers by memorizing training data, exploiting surface shortcuts, or guessing options, rather than truly understanding the document content. A few studies focusing on the process (e.g., Sugawara et al.) target linguistic skills (coreference resolution, named entity recognition, etc.) "prior to" comprehension, rather than the comprehension process itself.

**Key Challenge**: High answer-matching scores do not equal genuine document comprehension. A model that achieves a correct answer via a "shortcut" can completely fail when encountering new scenarios uncovered by training data. However, there is a lack of a systematic framework to evaluate whether the comprehension "process" of LLMs aligns with that of human experts.

**Goal**: Build a framework based on cognitive science theories to decompose the document comprehension process into testable skill hierarchies, construct evaluation datasets to assess LLM performance on each skill, and reveal the relationship between process correctness and answer correctness.

**Key Insight**: Drawing on Bloom's Taxonomy and reading cognitive theory (Krathwohl 2002, Afflerbach et al. 2015), the authors divide the comprehension process from local to global into three levels and five skills: Locating (single-sentence fact retrieval), Inferring (multi-sentence information integration), Connecting (inter-sentence logical relationship), Organizing (discourse structure comprehension), and Selecting (full-text main point extraction). Each skill corresponds to an independent test task.

**Core Idea**: Evaluate the document comprehension "process" rather than "results" of LLMs using a three-level, five-skill framework from cognitive science, showing that even the strongest LLMs fall far short of expert-level comprehension processes.

## Method

### Overall Architecture

SCoP consists of three components: (1) Definitions of five comprehension skills based on cognitive theory—Locating (sentence-level fact retrieval), Inferring (multi-sentence information integration), Connecting (inter-sentence logical filling), Organizing (paragraph segmentation by subheadings), and Selecting (key sentence extraction); (2) 4,682 test samples strictly selected and constructed from 12 existing datasets; (3) A systematic evaluation of 4 top-tier LLMs (Qwen2-72B, Llama3.1-70B, Claude-3.5, GPT-4o). The core innovation lies in requiring models not only to provide answers but also to output supporting sentences (evidence of the comprehension process), thereby detecting the inconsistency of "correct answer but incorrect process."

### Key Designs

1. **从认知理论到可测试任务的映射**:

    - **Function**: Decomposes the abstract cognitive comprehension process into five concrete, automatically evaluable NLP tasks.
    - **Mechanism**: Locating $\rightarrow$ given a question and a document, output supporting sentences + the answer; Inferring $\rightarrow$ given a complex question, output multiple supporting sentences + the answer; Connecting $\rightarrow$ mask several sentences in the document, and choose the correct sentence from candidates to fill in (similar to cloze, but at the sentence level); Organizing $\rightarrow$ given a scrambled document and subheadings, predict the correct position of each subheading; Selecting $\rightarrow$ given a document and the required number of key sentences, extract key sentences.
    - **Design Motivation**: These five skills form a progressive hierarchy from local to global. Cognitive science research indicates that human reading also follows this hierarchy, so evaluating LLMs using the same framework allows direct comparison of human-machine differences.

2. **严格的数据构建与噪声过滤框架**:

    - **Function**: Ensures that test samples genuinely require document comprehension to answer, eliminating memorization and shortcuts.
    - **Mechanism**: For the Locating and Inferring tasks, supporting sentences are first automatically annotated using semantic similarity (z-score threshold), and then syntax tree analysis is used to determine whether the question requires multi-step reasoning. A critical step is "de-memorization filtering"—questions are sent to the LLM (without the document), and any sample where the model answers correctly solely based on memory is removed. This step filtered out over 90% of the samples (37,023 $\rightarrow$ 4,682).
    - **Design Motivation**: The exceptionally high filtering rate of 90% is a striking finding in itself—indicating that a vast number of questions in existing reading comprehension benchmarks can actually be answered by LLMs through memorization, with absolutely no need to comprehend the document. This validates the necessity of evaluating the comprehension process.

3. **不一致性分析（Inconsistency Analysis）**:

    - **Function**: Detects the inconsistency phenomenon of "correct answer but incorrect comprehension process."
    - **Mechanism**: Calculates the inconsistency score—in Locating and Inferring tasks, measuring the proportion of samples where the supporting sentences are predicted incorrectly but the final answer is correct. GPT-4o's inconsistency rate at the Inferring level reaches 18.29%, which is 8 times higher than that at the Locating level (2.32%).
    - **Design Motivation**: If a model arrives at the correct answer through an incorrect comprehension process, it implies dependency on memory, guessing, or surface shortcuts rather than true comprehension of the document. Such "false accuracy" is unacceptable in high-stakes scenarios.

### Loss & Training

SCoP is an evaluation framework rather than a model training method. All models are evaluated in a zero-shot manner with temperature set to 0.

## Key Experimental Results

### Main Results

Accuracy (%) of four LLMs across five skills:

| Skill | Level | GPT-4o | Claude-3.5 | Llama3.1-70B | Qwen2-72B |
|------|------|--------|-----------|-------------|-----------|
| Locating | Local | 95.10 | 92.78 | 93.09 | 93.23 |
| Inferring | Intermediate | 33.89 | 37.49 | 46.54 | 31.51 |
| Connecting | Global | 41.79 | 65.04 | 25.36 | 52.81 |
| Organizing | Global | 31.70 | 49.50 | 30.43 | 17.78 |
| Selecting | Global | 17.46 | 14.18 | 13.31 | 12.96 |

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Inference performance after providing correct supporting sentences | GPT-4o: +15.97%, Claude: +8.03%, Qwen: +18.22% |
| Inconsistency rate (Locating) | GPT-4o: 2.32%, Claude: 4.77%, Llama: 4.45%, Qwen: 4.28% |
| Inconsistency rate (Inferring) | GPT-4o: **18.29%**, Claude: 12.80%, Llama: 19.68%, Qwen: 22.50% |
| De-memorization filtering rate | 37,023 → 4,682 (87.4% filtered) |

### Key Findings

- **Local comprehension >> Global comprehension**: All models exceed 92% in the Locating skill, but do not exceed 17.5% in the Selecting skill, showing a gap of approximately 76%. This aligns with findings in human cognitive research—local information processing is easier than global synthesis.
- **Open-source models can outperform closed-source models**: Llama3.1-70B outperforms GPT-4o in the Inferring skill (46.54% vs 33.89%), and Qwen2 beats GPT-4o in the Connecting skill (52.81% vs 41.79%), indicating that parameter scale and training data volume are not the sole determinants of comprehension process quality.
- **Comprehension process correctness directly affects answer quality**: After providing correct supporting sentences, inference accuracy is significantly improved (Qwen +18.22%), proving that the correctness of the comprehension process is a prerequisite for high-quality answers.
- **Up to 22.5% "false accuracy"**: In the Inferring task, Qwen2 exhibited incorrect supporting sentences but correct answers for 22.5% of the samples, indicating heavy reliance on shortcuts.
- **De-memorization exposes evaluation illusion**: 87.4% of the samples in existing benchmarks can be correctly answered by LLMs without even reading the document, indicating that conventional evaluations severely overestimate of models' comprehension capabilities.

## Highlights & Insights

- **Paradigmatic significance of the "process evaluation" perspective**: Shifting from evaluating "answers" to evaluating "processes" provides inspiration for all NLP evaluations. SCoP demonstrates that a correct answer does not equal correct comprehension, serving as a critical warning for deploying LLMs in high-stakes scenarios.
- **Organic integration of cognitive science and NLP evaluation**: The five skills are not designed arbitrarily but are grounded in cognitive science theories (Bloom's Taxonomy, Kintsch's Model), lending theoretical soundness to the framework. Correlation analysis between skills further validates the internal consistency of the design.
- **The 90% filtering rate is the most shocking finding**: This implies that almost all current reading comprehension benchmarks test memorization rather than actual comprehension to a large extent, posing a severe challenge to the evaluation methodology as a whole.

## Limitations & Future Work

- **Automated quality of supporting sentence annotation**: Supporting sentences for Locating and Inferring tasks are labeled automatically based on semantic similarity. Though the F1 score reaches 0.91-0.96, noise still exists.
- **Only evaluating English documents**: Cross-lingual comprehension processes may involve different cognitive patterns.
- **Limited model scale**: Only 4 models at the 70B scale were evaluated, lacking evaluations of larger-scale models (e.g., GPT-4-turbo, Claude-3-Opus).
- **No mitigation method proposed**: The framework reveals the problems but does not propose specific training or optimization schemes to improve the comprehension process of LLMs.

## Related Work & Insights

- **vs SQuAD/HotpotQA answer evaluation**: These benchmarks only evaluate the correctness of answers. SCoP evaluates both supporting sentences (comprehension process) and answers, discovering many cases of "correct answer but incorrect process."
- **vs Sugawara et al. (2017) skill analysis**: They analyzed linguistic skills needed "prior to" comprehension (NER, coreference, etc.), whereas SCoP evaluates cognitive skills "during" comprehension.
- **vs MMLU/SuperGLUE comprehensive evaluation**: These benchmarks mix memory with comprehension. SCoP ensures pure comprehension evaluation through de-memorization filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ Evaluating comprehension from the perspective of cognitive processes is a unique and valuable angle; the discovery of the 90% filtering rate is striking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Spanning 12 datasets, 4 models, 5 skills, with multi-dimensional analyses, but lacks larger-scale models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, natural mapping between cognitive theories and NLP tasks.
- Value: ⭐⭐⭐⭐ Poses fundamental questions about LLM evaluation methodology, offering critical warnings for deploying LLMs in high-risk scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[ACL 2025\] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](cognibench_cognitive_faithfulness.md)
- [\[ACL 2025\] Catching Shortcuts: A Framework for Evaluating Shortcuts in Large Language Models](catching_shortcuts_a_framework_for_evaluating_shortcuts_in_large_language_models.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)

</div>

<!-- RELATED:END -->
