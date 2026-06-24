---
title: >-
  [Paper Note] UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization
description: >-
  [ACL 2025][LLM (Other)][Time-sensitive question-answering] This work proposes UnSeenTimeQA, a time-sensitive question-answering benchmark based on synthetic facts (rather than real-world events) that eliminates the risk of data contamination by avoiding web-searchable queries. It designs three categories of temporal reasoning questions to evaluate the true temporal reasoning capabilities of LLMs, revealing that LLMs perform poorly on long-range event dependencies and parallel…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Time-sensitive question-answering"
  - "temporal reasoning"
  - "data contamination"
  - "synthetic facts"
  - "benchmark evaluation"
date: 2026-05-08
content_hash: e70b138128dca64e
---

# UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization

**Conference**: ACL 2025  
**arXiv**: [2407.03525](https://arxiv.org/abs/2407.03525)  
**Code**: None (dataset on HuggingFace: nurakib/UnSeenTimeQA)  
**Area**: LLM/NLP  
**Keywords**: Time-sensitive question-answering, temporal reasoning, data contamination, synthetic facts, benchmark evaluation

## TL;DR

This work proposes UnSeenTimeQA, a time-sensitive question-answering benchmark based on synthetic facts (rather than real-world events) that eliminates the risk of data contamination by avoiding web-searchable queries. It designs three categories of temporal reasoning questions to evaluate the true temporal reasoning capabilities of LLMs, revealing that LLMs perform poorly on long-range event dependencies and parallel event reasoning.

## Background & Motivation

**Background**: Time-Sensitive Question-Answering (TSQA) is a crucial task for evaluating the temporal reasoning capabilities of LLMs. Existing benchmarks, such as TimeQA, SituatedQA, and TempReason, test models using real-world time-dependent facts—for example, questions like "Who was the President of the United States in 2020?" which require linking time with facts.

**Limitations of Prior Work**: Existing TSQA benchmarks suffer from a fundamental problem: they are based on real-world facts, almost all of which can be searched online. LLMs are highly likely to have "seen" these questions or related information during pre-training, allowing them to answer via memorization rather than true temporal reasoning. This leads to data contamination—where test data leaks into training data, preventing the evaluation results from accurately reflecting the models' reasoning capacity.

**Key Challenge**: The goal is to evaluate the temporal reasoning capability of LLMs (i.e., whether the model can correctly reason the answer given the temporal relationships of a sequence of events); however, existing benchmarks may actually test memorization (i.e., whether the model saw the fact during training). These two capabilities are confounded and cannot be disentangled.

**Goal**: To construct a TSQA benchmark that completely eliminates the risk of data contamination, forcing LLMs to rely on true temporal reasoning capabilities to answer questions rather than the factual knowledge memorized during pre-training.

**Key Insight**: Using synthetically generated fictitious scenarios instead of real-world events. These scenarios do not exist on any web page, making it impossible for LLMs to have encountered them during pre-training. Thus, answering them must rely solely on reasoning. Additionally, the data generation framework supports on-demand generation of new samples, allowing regeneration even if the test set leaks.

**Core Idea**: Building a temporal reasoning benchmark where LLMs cannot "cheat" by using synthetic fact scenarios combined with systematic temporal question types, thereby truly evaluating their temporal reasoning capabilities.

## Method

### Overall Architecture

UnSeenTimeQA consists of two core components: (1) a data generation framework that automatically generates time-sensitive event scenarios and corresponding questions based on synthetic facts; and (2) an evaluation framework that assesses the performance of various LLMs on different types of temporal reasoning questions. The input is a timeline describing fictitious characters/events along with related questions, and the output is the LLM's answer, which is compared with the ground truth to evaluate accuracy.

### Key Designs

1. **Synthetic Fact Generation Framework**:

    - **Function**: Generating data-contamination-immune time-sensitive Q&A data.
    - **Mechanism**: The framework constructs timeline narratives using fictitious names of people, organizations, locations, and events. Each scenario contains multiple events and their occurrences (precise to year/month/day), with sequential or overlapping temporal relations. For instance, scenarios like a fictitious character "X worked at company A from 2015 to 2018 and lived in city B from 2017 to 2020." The key design is that all entities and facts are completely fabricated and do not correspond to any real-world person or event. The generation process is parameterized, allowing control over the number of events, temporal span, proportion of parallel events, etc., enabling the on-demand generation of arbitrary numbers of new samples.
    - **Design Motivation**: The fundamental issue with existing benchmarks is their reliance on real-world facts, which fails to exclude data contamination. Synthetic facts solve this problem at its root, while parameterized generation ensures reproducibility and scalability.

2. **Three Categories of Time-Sensitive Questions**:

    - **Function**: Comprehensively testing different levels of temporal reasoning capabilities in LLMs.
    - **Mechanism**: Three question types with ascending difficulty are designed: (a) **Easy/Simple temporal questions**: directly asking for the occurrence time of an event or the state at a certain time point (e.g., "When did X start working at company A?"), which only requires locating the corresponding event; (b) **Sequential event questions**: involving reasoning over multiple chronologically ordered events, which requires comparing temporal relations between different events (e.g., "Did X join company A first or move to city B first?"), requiring comparison of start times; (c) **Parallel event questions**: involving reasoning over temporally overlapping events (e.g., "Was X still working at company A while living in city B?"), which requires determining whether the time windows of two events overlap, as well as handling long-range event dependencies.
    - **Design Motivation**: Real-world temporal reasoning involves various complexities—from simple event localization to complex multi-event sequence reasoning. The stratified design allows for precise identification of the difficulty levels encountered by LLMs.

3. **Data Contamination Immunity Mechanism**:

    - **Function**: Ensuring evaluation results reflect true reasoning capabilities.
    - **Mechanism**: In addition to using synthetic facts, the framework incorporates multiple defenses: (a) support for on-demand dataset regeneration, ensuring any leaked version can be replaced with a new one; (b) randomized generation (names, times, event combinations) to prevent complete overlap between matches; (c) "freshness validation"—if model performance on synthetic facts is significantly higher than expected, it may indicate data leakage, signaling a need to update the data.
    - **Design Motivation**: Data contamination is one of the greatest threats to current LLM evaluations. Even if the current version is secure, future leaks are possible. On-demand generation ensures the long-term validity of the benchmark.

### Loss & Training

As a benchmarking study, this work does not involve model training. Evaluations are conducted on multiple LLMs under zero-shot and few-shot settings.

## Key Experimental Results

### Main Results

| Model | Easy Subset | Sequential Subset | Parallel Subset | Overall |
|------|---------|----------------|-------------|------|
| GPT-4o | High (~70%+) | Medium (~55%) | Low (~40%) | ~55% |
| GPT-3.5 | Medium (~60%) | Medium (~45%) | Low (~30%) | ~45% |
| Llama3-70B | Medium (~55%) | Low-Medium (~40%) | Low (~25%) | ~40% |
| Llama3-8B | Low-Medium (~45%) | Low (~30%) | Very Low (~20%) | ~30% |
| Real-Fact TSQA | High (~80%+) | High (~70%+) | Moderately High (~60%+) | ~70% |

### Ablation Study

| Configuration | Accuracy | Description |
|------|--------|------|
| Synthetic Fact TSQA | ~55% | Core evaluation results of this work |
| Real-Fact TSQA | ~70% | Performance of the same model on traditional TSQA |
| Difference | ~15% | Estimated contribution of memorization |
| Short-range event dependency | Relatively High | Reasoning involving 2-3 events |
| Long-range event dependency | Significant Drop | Reasoning involving 5+ events |
| Single event time window | Relatively High | Simple scenarios with no overlap |
| Multi-event temporal overlap | Significant Drop | Complex scenarios with overlap |

### Key Findings

- **LLM performance on synthetic facts is significantly lower than on real-fact TSQA**: This discrepancy directly proves the existence of data contamination in prior TSQA benchmarks—the high performance of LLMs on traditional benchmarks is partially driven by memorization rather than reasoning.
- **Parallel event reasoning is the primary bottleneck**: Reasoning over temporally overlapping events (e.g., "Did X also study in B during their employment at A?") poses a severe challenge to all LLMs, with accuracy significantly lower than that of simple questions.
- **Long-range event dependencies lead to performance degradation**: When the reasoning chain involves multiple sequential events, LLM performance decreases as the number of events increases, suggesting limited temporal reasoning capabilities.
- **Model scale helps but does not resolve the root issue**: GPT-4o outperforms GPT-3.5, which in turn outperforms smaller models; however, even the strongest models perform poorly on parallel events.
- **Few-shot demonstrations provide limited help**: Providing a small number of exemplars helps with simple questions, but yields limited improvements for complex temporal reasoning.

## Highlights & Insights

- The design of using **synthetic facts to eliminate data contamination** is the most core contribution of this paper. This concept is not only applicable to temporal reasoning but can also be generalized to other reasoning evaluation scenarios that need to exclude the confounding effects of memorization (e.g., spatial reasoning, causal reasoning).
- The **three-tiered difficulty question design** cleverly diagnoses the specific weaknesses of LLMs in temporal reasoning: it is not that they lack an understanding of temporal concepts, but rather that they encounter difficulties when handling complex multi-event temporal relationships (especially parallel and overlapping events).
- The **on-demand generation design** imparts an "evergreen" property to the dataset, which is particularly valuable in the current context of rampant LLM data contamination.

## Limitations & Future Work

- Synthetic fact scenarios are relatively simple and lack real-world complexity and ambiguity, which may underestimate the requirements of real-world temporal reasoning for LLMs.
- Templatized generation of event scenarios may introduce pattern bias—LLMs might learn patterns such as "which answers typically correspond to this text format."
- Only 5–6 LLMs were evaluated, leaving out more recent models (e.g., Claude 3.5, Gemini, etc.).
- Reasoning enhancement methods such as Chain-of-Thought (CoT) were not explored regarding their effectiveness in improving temporal reasoning.
- Future work can incorporate more types of temporal reasoning (e.g., periodic events, vague temporal expressions, time-interval calculations) or integrate knowledge graphs to build richer event networks.

## Related Work & Insights

- **vs. TimeQA**: TimeQA is based on real-world events (e.g., Wikipedia timelines) and cannot exclude data contamination. UnSeenTimeQA fundamentally addresses this issue through synthetic facts.
- **vs. TempReason**: TempReason focuses on understanding temporal expressions (e.g., which day "last Friday" refers to), emphasizing linguistic expression rather than temporal reasoning. The two focus on different levels of temporal reasoning.
- **vs. SituatedQA**: SituatedQA covers context-dependent QA across both temporal and spatial dimensions, which is broader in scope but less focused on fine-grained temporal reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Utilizing synthetic facts to eliminate data contamination is an excellent idea, though the paradigm of synthetic benchmarking itself has prior precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model evaluation and error analysis are relatively detailed, although the model coverage could be broader.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is clear and the method is well-described, but some details (e.g., implementation of the generation framework) could be more detailed.
- **Value**: ⭐⭐⭐⭐ Provides a more reliable benchmark for temporal reasoning evaluation and uncovers the true bottlenecks in LLM temporal reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] ScaleQuest: Unleashing LLM Reasoning Capability via Scalable Question Synthesis from Scratch](unleashing_llm_reasoning_capability_via_scalable.md)

</div>

<!-- RELATED:END -->
