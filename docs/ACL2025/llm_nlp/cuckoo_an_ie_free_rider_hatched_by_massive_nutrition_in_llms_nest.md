---
title: >-
  [Paper Note] Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest
description: >-
  [ACL 2025][LLM (Other)][Information Extraction] This paper proposes the Next Tokens Extraction (NTE) paradigm, converting next-token prediction in LLM pre-training data into a BIO-tagged extraction task. By pre-training a RoBERTa tagger (Cuckoo) on 102.6 million instances derived from C4 and TuluV3, it comprehensively outperforms existing IE pre-training models in few-shot information extraction tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Information Extraction"
  - "Pre-training"
  - "Next Tokens Extraction"
  - "BIO Tagging"
  - "Free Rider"
date: 2026-05-08
content_hash: ce1d3d0c7494b95f
---

# Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest

**Conference**: ACL 2025  
**arXiv**: [2502.11275](https://arxiv.org/abs/2502.11275)  
**Code**: [github](https://github.com/KomeijiForce/Cuckoo)  
**Area**: LLM/NLP  
**Keywords**: Information Extraction, Pre-training, Next Tokens Extraction, BIO Tagging, Free Rider

## TL;DR

This paper proposes the Next Tokens Extraction (NTE) paradigm, converting next-token prediction in LLM pre-training data into a BIO-tagged extraction task. By pre-training a RoBERTa tagger (Cuckoo) on 102.6 million instances derived from C4 and TuluV3, it comprehensively outperforms existing IE pre-training models in few-shot information extraction tasks.

## Background & Motivation

The biggest bottleneck in pre-training for Information Extraction (IE) lies in data scale. The core success of LLMs is driven by Next Token Prediction (NTP), which utilizes every token in text as a label, training on trillions of tokens. In contrast, IE pre-training consistently requires annotated spans with label names, making its data acquisition efficiency far lower than NTP.

For example, MultiNERD required substantial effort to collect only 164,000 English NER instances from Wikipedia and Wikinews, whereas NTP can easily acquire supervised signals from trillions of tokens in raw text. This massive gap in data scale has driven an increasing number of researchers to transition to using LLMs as the core model for IE.

The authors propose an extremely elegant solution: IE models can "free ride" on LLM training resources. The key insight is that many "next tokens" in text have actually already occurred in the context. By converting the prediction of these recurring spans into tag-based extraction tasks, one can directly exploit the massive pre-training and post-training datasets of LLMs.

## Method

### Overall Architecture

The training of the Cuckoo model consists of two stages, mimicking the training workflow of LLMs:

1. **Pre-training**: Extract 100 million NTE instances from the C4 dataset to learn extraction relationships from raw text.
2. **Post-training**: Extract 2.6 million chat-formatted NTE instances from the TuluV3 dataset to impart instruction-following capabilities.

The final Cuckoo model is obtained by continuing training on RoBERTa-large.

### Key Designs

**Next Tokens Extraction (NTE) Paradigm:**

The goal of NTP is to predict the next token $x_{t+1}$ given the context $[x_1, ..., x_t]$. NTE modifies this as follows: when the next $n$ tokens $[x_{t+1}, ..., x_{t+n}]$ have already appeared in the context (i.e., there exists $k$ such that $[x_{k+1}, ..., x_{k+n}] = [x_{t+1}, ..., x_{t+n}]$), the corresponding positions in the context are annotated with BIO tags—the starting position is tagged with B, subsequent positions with I, and other positions with O.

**Three Advantages of NTE over NTP:**

1. **Parameter Efficiency**: NTP requires extra parameters to store knowledge for generating tokens not present in the input, whereas NTE only needs to focus on tagging input tokens, making it suitable for smaller models.
2. **Inference Efficiency**: NTE taggers are not only smaller but can also extract multiple tokens in a single forward pass using the BIO scheme.
3. **Transferability**: NTE taggers can directly adapt to IE tasks, as those tasks typically use the BIO tagging scheme as well.

**Data Construction Details:**

- Use SpaCy to parse noun phrases, filtering out stop words and punctuation.
- Sample 5% of spans that recur in the context as NTE instances.
- Pre-training Stage: The full C4 dataset can be converted into approximately 5 billion NTE instances, from which 100 million (2%) are sampled for experiments.
- Post-training Stage: 939,000 TuluV3 conversations are converted into 2.6 million NTE instances (retaining only spans that appear in the user prompt and match in the assistant response).
- Additionally, sample 5% of spans not present in the context as negative instances labeled with all O's.

**Data Quality Validation:**

Using GPT-4o to evaluate 20,000 sampled data points, 93.39% of the pre-training data and 96.20% of the post-training data contain genuine extraction relationships, proving the high data efficiency of the automated tagging strategy.

### Loss & Training

- Base Model: RoBERTa-large (~300M parameters)
- Optimizer: AdamW, initial learning rate $10^{-5}$
- Batch size: 64
- Total steps: ~1.6 million steps
- Pre-trained on C4 NTE data, followed by post-training on TuluV3 NTE data.

To exclude the influence of base model advantages, the baseline NTP model uses an OPT model of equivalent scale (~300M parameters), whose pre-training resources cover the RoBERTa training data.

## Key Experimental Results

### Main Results

**Evaluation is divided into three levels of comprehension:**

**1. Basic IE (Entity/Relation Recognition, 5-shot):**

| Method | NER Avg | RE Avg |
|------|---------|---------|
| RoBERTa | 46.80 | 18.15 |
| MultiNERD | 60.59 | 51.31 |
| NuNER | 65.99 | 64.42 |
| MetaIE | 65.57 | 64.61 |
| MRQA | 65.83 | 66.84 |
| **Cuckoo** | **66.34** | **70.63** |
| Rainbow Cuckoo | **68.91** | **73.26** |

**2. Query-based IE (MRC, 32-shot):**

| Method | SQuAD | SQuAD-V2 | DROP | Average |
|------|-------|----------|------|------|
| MetaIE | 74.59 | 62.54 | 30.73 | 55.95 |
| **Cuckoo** | **77.47** | **64.06** | **54.25** | **65.26** |
| MRQA (In-domain) | 80.07 | 66.22 | 54.46 | 66.92 |
| Rainbow Cuckoo (In-domain) | **86.57** | **69.41** | **64.64** | **73.54** |

**3. Instruction-following IE:**

| Method | Disambiguation | Preference | Miscellaneous |
|------|------|------|------|
| MultiNERD | 31.71 | 30.84 | 44.68 |
| NuNER | 31.40 | 51.01 | 44.32 |
| MetaIE | 29.77 | 56.12 | 47.35 |
| **Cuckoo**| **34.97** | 62.53 | **49.17** |
| Rainbow Cuckoo | **37.75** | **70.95** | **51.86** |

### Ablation Study

**Contributions of Pre-training vs. Post-training:**

| Variant | NER Avg | RE Avg | MRC Avg | Instruction Following |
|------|---------|---------|---------|---------|
| Only Pre-train (C4) | 65.61 | 68.77 | 63.94 | Moderate |
| Only Post-train (TuluV3) | 65.51 | 69.21 | 64.75 | Stronger |
| **Cuckoo (Both Combined)** | **66.34** | **70.63** | **65.26** | Strong |

- Basic IE tasks: C4 contributes more (the tasks are simple, and raw text is sufficient for learning).
- Query-based and Instruction-following IE: TuluV3 contributes more (requiring higher instruction awareness).

**NTE vs. NTP Comparison:**

OPT-C4-TuluV3 (NTP paradigm with equivalent data and parameter scale) underperforms Cuckoo (NTE paradigm) significantly across all levels, validating the programmatic advantage of NTE in IE tasks.

### Key Findings

1. **Evolving with LLM Data**: Utilizing different versions of post-training data (Tulu V1 $\rightarrow$ V2 $\rightarrow$ V3), Cuckoo's performance improves continuously across most dimensions. This means that as LLM data preparation advances, Cuckoo can naturally evolve without additional manual effort.

2. **Emergence of In-context Tagging**: When provided 5 examples in context (CoNLL2003) or 1 example (SQuAD), only Cuckoo exhibits an improvement (or at least maintains) in IE capability, while other pre-trained models (including NuNER and MRQA) show significant performance degradation. This is because sporadic burstiness in the raw text facilitates the emergence of in-context learning capabilities.

3. **Data Scaling Trend**: In the early stage of 4.1 million instances, performance shows a clear upward trend with increased data volume. Scaling up to 100 million instances stabilizes the macro-growth trend but introduces volatility, suggesting that the capacity of RoBERTa might be reaching its limit.

4. **NTE Data Efficiency**: The sentence-to-NTE instance conversion rate is 332% for C4 and 235% for TuluV3. Although the full C4 can yield approximately 5 billion NTE instances, only 4.06% of tokens are utilized for NTE learning, indicating that there is still room for further strengthening the supervised signal.

5. **Instruction Sensitivity Analysis**: In preference instruction testing (long answers vs. short answers), Cuckoo’s Dual Exact Match (DualEM) score (11.67) falls short of MRQA’s (12.32), but its Answer Similarity (AnsSim) is 40.48—far lower than MRQA’s 48.17. This indicates that Cuckoo exhibits higher distinctiveness across different instructions. Rainbow Cuckoo achieves a DualEM of 18.95, significantly outperforming MRQA.

## Highlights & Insights

1. **Extremely Elegant and Concise Core Idea**: The "free rider" concept—IE models can directly leverage the pre-training and post-training data of LLMs without any additional annotation effort. The conversion from NTP to NTE requires only detecting repetitive spans in the context and assigning BIO tags.

2. **Clever Metaphorical Naming**: Cuckoo is famous for laying eggs in other birds' nests, which perfectly mirrors how this model "free rides" on LLM training resources.

3. **Potential for Paradigm Shift**: NTE breaks the assumption that IE pre-training must rely on specialized annotated data, reducing the cost of data acquisition from "expensive human/LLM-synthesized annotations" to "zero-cost transformation". The scale of 102.6 million instances far exceeds all previous IE pre-training datasets.

4. **Natural Co-evolution with the LLM Ecosystem**: As the LLM community continuously improves pre-training/post-training data quality, Cuckoo automatically benefits without requiring any extra IE-specific work.

5. **Small Model, Large Capability**: RoBERTa-large (300M parameters) pre-trained with NTE approaches or even surpasses much larger LLMs on IE tasks, showcasing the parameter efficiency advantage of task specialization.

## Limitations & Future Work

1. **Label Embedding**: The current approach requires enumerating label names (similar to generative IE). Future work could explore label embedding variations to improve efficiency.
2. **Diversity of Data Sources**: Only C4 was utilized as the pre-training data source; specific data sources (such as textbooks) might be more beneficial for certain IE skills.
3. **Backbone Scaling**: Currently validated only on RoBERTa-large. Future studies could explore model scaling laws, multilingual backbones, and other architectures.
4. Only 4.06% of tokens are utilized in NTE learning, which represents low utilization and leaves room to further strengthen the supervised signal.
5. Performance volatility occurred at the 100-million instance stage, implying that RoBERTa's capacity might be a bottleneck, necessitating larger models to fully exploit the NTE data.

## Related Work & Insights

Cuckoo bridges two previously relatively independent research directions: IE pre-training and LLM data engineering. Its primary insight—"extraction is a special case of predicting recurring spans"—is remarkably clever.

Compared to earlier IE pre-training efforts (e.g., MultiNERD relying on Wikipedia links, NuNER and MetaIE relying on LLM-synthesized data), the data acquisition for NTE is essentially free. This "paradigm free-riding" strategy could inspire other NLP tasks: can custom-task learning be reframed as an extraction format of contextually available information within LLM training data?

The emergence of in-context tagging is also insightful: similar to the emergence mechanisms of in-context learning in LLMs, the burstiness distributional characteristics in raw text may play a crucial role.

## Rating

- **Novelty**: ★★★★★ — The paradigm shift from NTP to NTE is incredibly clever and concise.
- **Value**: ★★★★★ — Zero-cost data acquisition + small model + strong performance, possessing immense practical value.
- **Experimental Thoroughness**: ★★★★☆ — Three-tiered evaluation + ablation + evolutionary analysis + scaling trends, highly comprehensive.
- **Writing Quality**: ★★★★★ — Clear motivation, vivid metaphors, and tight arguments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SkillAggregation: Reference-free LLM-Dependent Aggregation](skillaggregation_reference-free_llm-dependent_aggregation.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ACL 2025\] Meta-Reflection: A Feedback-Free Reflection Learning Framework](meta-reflection_a_feedback-free_reflection_learning_framework.md)
- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)

</div>

<!-- RELATED:END -->
