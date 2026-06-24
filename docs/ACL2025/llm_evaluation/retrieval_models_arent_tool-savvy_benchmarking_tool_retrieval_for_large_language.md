---
title: >-
  [Paper Note] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models
description: >-
  [LLM Evaluation] This work proposes ToolRet, the first large-scale tool retrieval benchmark (comprising 7.6k retrieval tasks and 43k tools), revealing that existing strong Information Retrieval (IR) models perform poorly on tool retrieval tasks (with the strongest model achieving an nDCG@10 of only 33.83). It also contributes the ToolRet-train dataset, featuring over 200k training instances, which significantly improves the tool retrieval capabilities of IR models and enhance…
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 5d299f15072e4d95
---

# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models

## Paper Information

- **Conference**: ACL 2025
- **arXiv**: [2503.01763](https://arxiv.org/abs/2503.01763)
- **Code**: [https://github.com/shizhl/Tool-Retrieval-Benchmark](https://github.com/shizhl/Tool-Retrieval-Benchmark)
- **Area**: LLM Evaluation
- **Keywords**: Tool Retrieval, Tool Learning, Information Retrieval, Benchmark, LLM Agent

## TL;DR

This work proposes ToolRet, the first large-scale tool retrieval benchmark (comprising 7.6k retrieval tasks and 43k tools), revealing that existing strong Information Retrieval (IR) models perform poorly on tool retrieval tasks (with the strongest model achieving an nDCG@10 of only 33.83). It also contributes the ToolRet-train dataset, featuring over 200k training instances, which significantly improves the tool retrieval capabilities of IR models and enhances the success rate of end-to-end tool-use tasks.

## Background & Motivation

- **Background**: Tool learning aims to equip LLMs with external tools to solve practical tasks. In scenarios with large-scale toolsets, using IR models to retrieve useful tools is a crucial first step.
- **Limitations of Prior Work**: Existing tool-use benchmarks (e.g., ToolBench, ToolACE) simplify the retrieval step by manually pre-annotating only 10-20 relevant tools per task, which is far removed from the challenges in real-world application scenarios involving tens of thousands of tools.
- **Key Challenge**: Pilot experiments indicate that after replacing officially annotated toolsets with retrieved tools, the task success rate of agents drops significantly; even strong retrievers such as ColBERTv2 struggle to retrieve target tools effectively.
- **Design Motivation**: There is a critical need to (1) systematically evaluate the performance of IR models across diverse tool retrieval scenarios; and (2) analyze the impact of retrieval quality on end-to-end tool-use task success rates.

## Method

### Overall Architecture

The construction of the ToolRet benchmark consists of three stages: (1) Data Collection—gathering over 30 tool-use datasets from top AI conference papers, venue resources, and open-source communities; (2) Data Sampling—de-deduplicating and merging toolsets using K-means clustering; (3) Instruction Construction—automatically generating retrieval instructions using GPT-4o to support instruction-based retrieval evaluation.

### Key Designs

1. **Heterogeneous Tool Corpus**: The 43k tools encompass three types—Web APIs (36,978), code functions (3,794), and custom applications (2,443), covering diverse tool document types and domains.
2. **Clustering-based Task Sampling**: Using NV-embed-v1 to encode tasks and performing K-means clustering, where the number of clusters is set to the minimum of the toolset size and the query count. One task is randomly sampled from each cluster to ensure diversity while reducing redundancy.
3. **Target-Aware Instruction Generation**: Three experts were invited to handwrite 100 seed instructions, and GPT-4o was then used to automatically generate instructions for each task via in-context learning, enabling the instructions to bridge query intentions and target tool functionalities.

### Training Dataset: ToolRet-train

Data collection is expanded to the training sets of ToolACE, APIGen, and ToolBench, constructing training data with over 200k retrieval tasks. Each training sample contains a query, a generated instruction, target tools, and 10 negative tools retrieved by NV-embed-v1. The training adopts a contrastive learning framework, employing hard negative mining to enhance the model's discriminative ability regarding tool similarity.

### Benchmark Statistics

- **Total Retrieval Tasks**: 7,615 (Web API retrieval: 4,916; code function retrieval: 950; custom application retrieval: 1,749)
- **Total Tools**: 43,215
- **Average Query/Instruction Length**: 46.87 / 43.43 tokens
- **Average Tool Document Length**: 174.56 tokens

## Experiments

### Main Results

| Model Type | Representative Model | nDCG@10 |
|---------|---------|---------|
| Sparse Retrieval | BM25 | 18.72 |
| Dense Retrieval | NV-embed-v1 | 33.83 |
| Dense Retrieval | E5-Mistral | 24.46 |
| ColBERT | ColBERTv2 | 19.82 |
| Cross-Encoder | MiniLM-L6 | 28.60 |
| LLM Re-ranking | RankGPT | 30.56 |

- Even though the strongest model NV-embed-v1 performs excellently on traditional IR benchmarks, its nDCG@10 on ToolRet is only 33.83, indicating that the tool retrieval task is significantly more difficult than traditional retrieval.

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| Vocabulary Overlap Rate | The vocabulary overlap rate between the query and the target tool in tool retrieval tasks is much lower than in traditional retrieval tasks, requiring IR models to possess stronger semantic representation capabilities. |
| Task Shift | The task shift from information-seeking tasks to tool retrieval leads to a performance drop in IR models. |
| Training Effects | After fine-tuning on ToolRet-train, the retrieval performance of IR models is significantly improved, and the end-to-end task success rate increases accordingly. |
| Impact of Retrieval on Agents | Retrieval Recall@10 is strongly positively correlated with the agent task success rate, verifying the critical impact of tool retrieval quality on downstream performance. |

### Key Findings

1. Existing IR models perform far worse in tool retrieval than on traditional retrieval benchmarks, showing a significant task gap.
2. Low-quality tool retrieval directly causes a drop in the end-to-end task success rate of LLM Agents.
3. IR models fine-tuned with ToolRet-train not only exhibit substantially improved retrieval performance but also effectively enhance the overall task performance of tool-use LLMs.

## Highlights & Insights

- The first systematic tool retrieval benchmark, covering 7.6k tasks and 43k heterogeneous tools, filling an evaluation gap in this area.
- Quantitatively reveals the strong correlation between retrieval quality and end-to-end agent performance.
- Contributes a large-scale training dataset (200k+ instances), providing a practical resource for the community to improve tool retrieval models.
- The benchmark design follows the MTEB/BEIR formatting standards, facilitating rapid integration and replication by the community.
- Provides a systematic evaluation covering 6 kinds of IR models, offering direct reference for practitioners selecting tool retrieval solutions.

## Limitations & Future Work

- The quality of tool documentation is uneven; documentation descriptions from some sources are too brief or lack structured information.
- The evaluation only covers English tool retrieval tasks, without considering multilingual tool retrieval scenarios.
- Instructions are automatically generated by GPT-4o, which may introduce a distribution shift compared to human-written instructions.
- The toolset merging strategy may lose some tool features specific to certain datasets.
- Retrieval latency and efficiency performance under online-deployment scenarios are not evaluated.

## Related Work & Insights

- **Tool Learning**: Frameworks like ToolBench (Qin et al., 2023), ToolACE, and APIGen train LLMs to use tools via synthetic data, but are limited by context window length when facing large-scale toolsets.
- **IR Benchmarks**: Traditional IR benchmarks, such as MS-MARCO and BEIR, primarily target information-seeking tasks and lack evaluations for tool retrieval scenarios.
- **Related Systems**: Systems like ToolGen (Wang et al.) and COLT (Qu et al.) use semantic retrieval to aid tool selection, but their approaches are ad-hoc and lack systematic cross-scenario evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Exceptionally defines and systematically evaluates the tool retrieval task for the first time, filling a clear gap.
- **Value**: ⭐⭐⭐⭐⭐ — The benchmark and training dataset directly facilitate the development of the tool-use agent ecosystem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The evaluated model types are comprehensive, and end-to-end task success rate analyses are included.
- **Writing Quality**: ⭐⭐⭐⭐ — The problem definition is clear, and the experimental design is rational.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)

</div>

<!-- RELATED:END -->
