---
title: >-
  [Paper Note] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models
description: >-
  [ACL 2025][LLM Efficiency][Long-context referencing] This paper proposes the Ref-Long benchmark to evaluate long-context language models (LCLMs) from the overlooked dimension of "referencing attribution" (identifying which documents reference a given key and returning their indices). It contains 3 subsets (ranging from synthetic to real) with a total of 4,300 tasks. The findings reveal that even GPT-4o achieves only 19% ExAcc on Multi-Hard-24K…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long-context referencing"
  - "benchmark"
  - "document index identification"
  - "human comparison"
  - "LCLM evaluation"
  - "error analysis"
date: 2026-05-08
content_hash: 1d1ac6f1fe0766af
---

# Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.09506](https://arxiv.org/abs/2507.09506)  
**Code**: [github.com/wujunjie1998/Ref-Long](https://github.com/wujunjie1998/Ref-Long)  
**Area**: LLM Efficiency / Long-Context Evaluation  
**Keywords**: Long-context referencing, benchmark, document index identification, human comparison, LCLM evaluation, error analysis

## TL;DR

This paper proposes the Ref-Long benchmark to evaluate long-context language models (LCLMs) from the overlooked dimension of "referencing attribution" (identifying which documents reference a given key and returning their indices). It contains 3 subsets (ranging from synthetic to real) with a total of 4,300 tasks. The findings reveal that even GPT-4o achieves only 19% ExAcc on Multi-Hard-24K, far below the human baseline of 92%, and neither prompt engineering nor specialized fine-tuning can fundamentally resolve this issue.

## Background & Motivation

**Background**: Long-context language models (LCLMs) claim to support input windows of 128K-1M tokens. Existing benchmarks primarily evaluate them from two dimensions: general long-text understanding (LongBench, L-Eval, NOCHA, HELMET) and retrieval capabilities (Needle-in-a-Haystack, Counting-Stars, RULER).

**Limitations of Prior Work**: (1) General benchmarks either create "artificial length" by concatenating unrelated texts, leading to distribution shifts, or require expensive human annotation (e.g., NOCHA spent $3,330 to annotate 1,001 QA pairs). (2) Retrieval-based benchmarks focus only on "whether the key can be found." GPT-4o still achieves over 70% accuracy on 128K inputs, indicating these tasks are too simple—they ignore the localization and attribution relationship between the key and the context documents.

**Key Challenge**: Existing evaluations cover "retrieval" (finding text containing specific information) but completely ignore "referencing" (identifying which documents reference a specific key entity and returning their position indices). Referencing attribution is crucial in scenarios such as legal clause retrieval, financial report attribution, and academic paper citation tracing.

**Goal**: Build a benchmark specifically for evaluating the capability of "long-context referencing attribution," consisting of progressive subsets from synthetic to real-world data, and systematically reveal the deficiencies of LCLMs in this dimension through human baselines, prompt variants, fine-tuning experiments, and error analysis.

**Key Insight**: Formalize the referencing capability as: given $M$ indexed documents and a key $k$, the LCLM must return the set of indices of all documents that reference $k$. This requires the model not only to retrieve the key but also to understand the mapping and attribution relationship between the key and the document.

**Core Idea**: Referencing attribution is one of the core capabilities of long-context understanding, but existing LCLMs are severely deficient in this aspect—even the strongest GPT-4o achieves only a 19% exact match rate on a 24K input, whereas humans reach 92%.

## Method

### Overall Architecture

The task setting of Ref-Long is unified: $M$ documents are randomly sampled from a candidate document set and numbered, with each document containing approximately 1,000 tokens. A key $k$ that appears within these documents is selected, and the LCLM is required to output the indices of all documents that reference $k$. The evaluation metrics are Exact Match Accuracy (ExAcc, order-insensitive exact match) and F1 score. Under this framework, three subsets are constructed to cover the gradient from synthetic to semi-synthetic to real-world data, totaling around 4,300 tasks. Thirteen LCLMs (4 closed-source and 9 open-source) are evaluated.

### Key Designs

#### Key Design 1: Ref-Long-A (Synthetic Obtrusive Key, 1,800 Tasks)

- **Function**: Generate 100 documents of approximately 1,000 tokens each by randomly concatenating essays from the Paul Graham Essays dataset, and randomly insert 1 (Single) / 5 (Multi) template sentences "The little penguin counted {num} ★" in each essay, where num is an integer key.
- **Mechanism**: Adjust the difficulty by controlling the range of values for num—Easy: $\text{num} \in [0,100)$, Medium: $\text{num} \in [0,60)$, Hard: $\text{num} \in [0,20)$. A smaller range increases the probability of the same num appearing across multiple documents, leading to higher confusion. $M \in \{8, 16, 24\}$ corresponds to input lengths of 8K/16K/24K tokens, with 100 tasks per configuration. Cumulative sampling (where 8K tasks are prefixes of 16K tasks) is adopted to reduce randomness.
- **Design Motivation**: The abrupt insertion makes the key highly conspicuous in the context. If LCLMs cannot perform well even in this scenario, it proves that referencing attribution itself is the bottleneck rather than semantic understanding.

#### Key Design 2: Ref-Long-F (Fluent Natural Key, 2,100 Tasks) and Ref-Long-Paper (Real Citation, 400 Tasks)

- **Ref-Long-F**: Based on the SummHay benchmark, 3 news topics are selected, with 100 coherent documents generated by GPT-4o for each topic. The key is an insight sentence (natural language) embedded in the text, and $M \in \{8, ..., 56\}$ corresponds to 8K-56K. The key is no longer obtrusive but semantically integrated into the context, closely resembling real-world scenarios.
- **Ref-Long-Paper**: Manually collect 47 computer science arXiv seed papers published after March 2024, along with 34 distractor papers from early 2024 (which cannot physically cite the seed papers due to publication time). $M \in \{8, 12, 16, 20\}$ corresponds to 30K-75K. The key is the title of a seed paper, and the LCLM is required to identify the indices of documents that cite this paper.
- **Design Motivation**: The three subsets establish a difficulty gradient of "synthetic $\rightarrow$ semi-synthetic $\rightarrow$ real-world," validating basic referencing capabilities, semantically integrated scenarios, and real academic citation scenarios respectively.

#### Key Design 3: Multi-Angle Analytical Experiments

- **Human Evaluation**: 2 PhD students annotated 50 Multi-Hard-24K tasks, with an average time of 124 seconds per task, achieving 92% ExAcc and an inter-annotator agreement rate of 84%.
- **Prompt Variants**: (1) Human strategy prompting—instructing the LCLM to build a dictionary while reading; (2) Natural language key—substituting integers with fruit names. Results: only GPT-4o showed improvement under strategy prompting (19% $\rightarrow$ 34%), while weaker models showed no effect or even degraded.
- **Fine-Tuning Experiments**: Fine-tuning Llama-3.1-8B on Multi-Easy-8K (500 samples) led to some improvements on 16K/24K, but results remained far below the human level.
- **Error Analysis**: Failure cases of GPT-4o are categorized into three types—Type I (missing citation, accounting for 85% in Ref-Long-A), Type II (over-citation, accounting for 50-54% in Ref-Long-F/Paper), and Type III (both).

## Key Experimental Results

### Main Results: Ref-Long-A Multi-24K (Table 1)

| Model | Easy ExAcc | Medium ExAcc | Hard ExAcc | Hard F1 |
|------|-----------|-------------|-----------|---------|
| GPT-4o | 75% | 61% | **19%** | 75.38 |
| Gemini-1.5-Pro | 67% | 44% | 9% | 65.24 |
| GPT-4o mini | 67% | 52% | 7% | 68.64 |
| Llama-3.3-70B | 43% | 19% | 4% | 56.23 |
| Qwen2.5-72B | 39% | 22% | 5% | 60.90 |
| Llama-3.1-8B | 2% | 0% | 0% | 38.85 |
| **Human** | — | — | **92%** | 99.08 |

### Ref-Long-F Twitter 24K (Selected Table 6)

| Model| F1 | ExAcc |
|------|-----|-------|
| GPT-4o | 83.50 | 41% |
| Gemini-1.5-Pro | 80.47 | 39% |
| Llama-3.1-70B | 80.08 | 34% |

### Ref-Long-Paper (Selected Table 7, $M=20$, ~75K)

| Model | F1 | ExAcc |
|------|-----|-------|
| GPT-4o | 71.13 | 17% |
| Qwen2.5-72B | 76.69 | **31%** |
| Gemini-1.5-Pro | 70.89 | 20% |

### Ablation Study

| Experiment | GPT-4o Hard-24K ExAcc |
|------|----------------------|
| Default prompt | 19% |
| + Human Strategy Prompt | **34%** |
| + Natural Language Key (Fruit Names) | 19% |
| Fine-tuned Llama-8B (Easy-8K) | Easy: 22% $\rightarrow$ 22%, 16K: 5% $\rightarrow$ 20%, 24K: 2% $\rightarrow$ 8% |

### Key Findings
- **Referencing attribution is a blind spot for current LCLMs**: Even GPT-4o achieves only 19% ExAcc on the Hard setting with just 24K input (well below its 128K window), compared to 92% for humans. This substantial gap indicates that this is not a simple computation capacity issue.
- **Model scale is positively correlated, but the ceiling is very low**: Larger models consistently outperform smaller ones, yet even the largest closed-source models do not exceed 20% ExAcc under the Hard setting.
- **Prompt engineering provides limited help**: Human strategy prompting is only effective for GPT-4o (19% $\rightarrow$ 34%), and is ineffective or even detrimental for weaker models; changing the key format has almost no impact.
- **Fine-tuning yields limited improvements**: After fine-tuning on Easy-8K, the model still fails on longer inputs and higher difficulty levels—referencing capability is not a generalizable skill that can be acquired merely through a small amount of task-specific data.
- **Error patterns vary across scenarios**: Under synthetic keys, "under-citation" (85%) is dominant, whereas under natural language keys, "over-citation" (50-54%) is primary. This suggests models tend to be conservative with numerical keys but prone to over-matching with natural language keys.
- **Models fine-tuned for long contexts do not necessarily perform better**: ProLong-8B and LongCite-8B do not significantly outperform their base models on Ref-Long, indicating that existing long-context fine-tuning strategies do not effectively cover referencing attribution.

## Highlights & Insights

- **Precise distinction between "referencing vs. retrieval"**: Ref-Long specifies an ignored yet important capability dimension—the model must not only locate the information but also know which document the information "belongs to." This is crucial for practical applications such as legal, financial, and academic domains.
- **Criticality of the human baseline**: The massive gap of 92% vs. 19% provides a clear upper bound for room for improvement and demonstrates that the task design is reasonable (rather than being an unreasonable task that "even humans cannot perform").
- **Synthetic-to-real progressive benchmark design**: The three subsets establish a difficulty gradient, avoiding the limitations of a single evaluation dimension. The controlled variable design of Ref-Long-A allows for precise isolation of confounding factors.
- **Practical value of error analysis**: The categorization of Type I/II/III errors directly points to directions for model improvement—enhancing recall for numerical keys and precision for natural language keys.

## Limitations & Future Work

- **Maximum input length limited to 75K**: The maximum length in Ref-Long-Paper is around 75K tokens, failing to test extreme scenarios in the 128K-1M context window.
- **Small-scale real-world subset**: Ref-Long-Paper contains only 400 tasks and 47 seed papers, offering limited data diversity.
- **Lack of comprehensive evaluation on reasoning models**: o1 was tested on only 50 tasks (yielding extremely poor results with 0% ExAcc), while newer models like o3 or o4-mini have not been evaluated.
- **Abnormally high scores of Qwen2.5-72B on the Paper subset**: This may be due to potential data leakage since its training data might contain some arXiv papers, weakening the evaluation reliability of this subset.

## Related Work & Insights

- **vs. Needle-in-a-Haystack / RULER**: These benchmarks only test "whether the key can be found," which GPT-4o easily handles with over 70% accuracy. Ref-Long additionally requires identifying "where" the key is, causing the difficulty to skyrocket, with ExAcc dropping to 19%.
- **vs. LongBench / NOCHA**: General benchmarks evaluate comprehensive understanding but suffer from high construction costs. Ref-Long only requires annotating the key locations, resulting in extremely low construction costs and high controllability.
- **vs. SummHay**: SummHay also contains referencing elements but lacks a systematic approach. Ref-Long-F is directly built upon its data but redesigns the evaluation paradigm.
- **Insights**: The referencing attribution capability might need explicit modeling during pre-training or long-context fine-tuning phases (such as introducing document-level position awareness into attention mechanisms) rather than expecting it to emerge naturally solely by expanding the context length.

## Rating
- Novelty: ⭐⭐⭐⭐ Precisely defines the overlooked dimension of "referencing attribution," with a clean and controllable benchmark design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive and systematic, involving 13 models, 3 subsets, human baseline, prompt variants, fine-tuning experiments, and error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, building progressively from synthetic to real-world scenarios.
- Value: ⭐⭐⭐⭐ Highlights the severe deficiencies in the referencing capabilities of current LCLMs; the 92% vs. 19% gap provides a clear direction for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)

</div>

<!-- RELATED:END -->
