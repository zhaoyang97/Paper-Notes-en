---
title: >-
  [Paper Note] NExtLong: Toward Effective Long-Context Training without Long Documents
description: >-
  [ICML 2025][LLM Efficiency][Long-context training] This paper proposes the NExtLong framework, which synthesizes long-context training data by segmenting documents into meta-chunks and inserting hard negative distractor texts retrieved from a pre-training corpus between these chunks. This forces the model to distinguish long-range dependency information from distractors, achieving an average improvement of 7.33% over the prior state-of-the-art long-context synthesis method…
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Long-context training"
  - "data synthesis"
  - "negative sample extension"
  - "long-range dependency"
  - "hard negative"
date: 2026-05-08
content_hash: 66755ffe85bb5a5c
---

# NExtLong: Toward Effective Long-Context Training without Long Documents

**Conference**: ICML 2025  
**arXiv**: [2501.12766](https://arxiv.org/abs/2501.12766)  
**Code**: [https://github.com/caskcsg/longcontext/tree/main/NExtLong](https://github.com/caskcsg/longcontext/tree/main/NExtLong)  
**Area**: LLM Efficiency  
**Keywords**: Long-context training, data synthesis, negative sample extension, long-range dependency, hard negative

## TL;DR
This paper proposes the NExtLong framework, which synthesizes long-context training data by segmenting documents into meta-chunks and inserting hard negative distractor texts retrieved from a pre-training corpus between these chunks. This forces the model to distinguish long-range dependency information from distractors, achieving an average improvement of 7.33% over the prior state-of-the-art long-context synthesis method, Quest, on the HELMET and RULER benchmarks.

## Background & Motivation
**Background**: The context length of LLMs has grown rapidly (from 4K in Llama 2 to 128K in Llama 3.1), where long-context capability is key to unlocking tasks such as document summarization, long-form QA, and code planning.

**Limitations of Prior Work**: Mainstream methods require a large volume of high-quality long documents for continued pre-training, but high-quality long documents are extremely scarce in most domains, and this problem becomes more severe as the target context length increases.

**Key Challenge**: Existing synthetic methods (KNN-based similar document concatenation, random concatenation, Quest keyword-retrieval-based concatenation) lack explicit mechanisms for modeling long-range dependencies—although the concatenated documents are long, the associations between different parts are weak or random.

**Goal**: How to synthesize long-context data that can effectively train long-range dependency modeling capabilities in the absence of natural long documents.

**Key Insight**: Inspired by the hard negative technique in contrastive learning—inserting semantically similar but unrelated distractor texts between dependent segments not only extends the dependency distance but also forces the model to learn to identify truly relevant context amidst noise.

**Core Idea**: Segment short documents into meta-chunks and insert retrieved hard negatives between adjacent meta-chunks to create "Negative Extended" long documents, forcing the model to capture cross-chunk, long-range dependencies amid a large number of distractors.

## Method

### Overall Architecture
NExtLong consists of two stages: (1) **Negative Document Extension**: Segmenting short documents into meta-chunks, retrieving hard negatives for each chunk, and concatenating them to generate long documents; (2) **Long-Range Dependency Modeling**: Training with NTP loss, where full loss is computed on meta-chunk tokens, and loss weights for hard negative tokens are downgraded or set to zero.

### Key Designs

1. **Document Chunking**:

    - **Function**: Segmenting a meta-document into multiple meta-chunks according to a maximum length $s$.
    - **Mechanism**: Segmenting paragraphs by newlines and then sequentially concatenating paragraphs until the maximum length $s$ is reached, ensuring sentence integrity.
    - **Design Motivation**: Controlling the granularity of each chunk while preserving semantic coherence.
    - A meta-document is divided into $p$ meta-chunks: $r \to \{m_1, m_2, \dots, m_p\}$.

2. **Hard Negative Mining**:

    - **Function**: Retrieving chunks that are semantically similar but different in content from the pre-training corpus for each meta-chunk to serve as distractors.
    - **Mechanism**:
        - Building a chunk-level index of the pre-training corpus using FAISS (also chunked at granularity $s$).
        - Retrieving the top-$k$ most similar chunks for each meta-chunk as hard negatives.
        - Concatenating the hard negatives after the corresponding meta-chunk to form an extended chunk: $l_i = [m_i, n_{i,1}, n_{i,2}, \dots, n_{i,k}]$.
    - **Design Motivation**: Since the pre-training corpus is thoroughly deduplicated, the retrieved chunks are semantically similar to the meta-chunks but do not duplicate their content—which is precisely the definition of a "hard" negative.
    - Placing meta-chunks before hard negatives (validated as superior by ablation studies).

3. **Long Document Synthesis and Dependency Modeling**:

    - **Function**: Concatenating all extended chunks to form a long document $t = [l_1, l_2, \dots, l_p]$ for training.
    - **Mechanism**: The originally adjacent meta-chunks ($m_i$ and $m_{i+1}$) are separated by a large number of hard negatives, stretching short-range dependencies into long-range dependencies.
    - Distinguishing between meta-chunk tokens and hard negative tokens during training: the model needs to "traverse" distractor information to locate the true contextual dependencies.
    - **Design Motivation**: Numerous studies show that LLMs are easily distracted by irrelevant context, and this distraction worsens as the context grows longer—NExtLong specifically leverages this vulnerability for reinforcement training.

### Loss & Training
- Standard NTP loss is used for continued training.
- Different loss weights can be applied to meta-chunk tokens and hard negative tokens during training.
- The target context length is jointly determined by the meta-document length, the chunking granularity $s$, and the number of hard negatives $k$.

## Key Experimental Results

### Main Results (HELMET Benchmark, Multi-Task Multi-Length Average)

| Model | Recall | RAG | ICL | Re-rank | LongQA | Summ | Average |
|------|--------|-----|-----|---------|--------|------|------|
| Llama-3.1-8B (128K Long Doc Training) | Reference | Reference | Reference | Reference | Reference | Reference | Reference |
| Quest (Long-Context Synthesis SOTA) | -- | -- | -- | -- | -- | -- | Baseline |
| **NExtLong** | Better | Better | Better | Better | Better | Better | **+7.33%** |

### Comparison with Reputable Models (HELMET + RULER, 8K to 128K Average Length)

| Model | HELMET Average | RULER Average |
|------|-----------|-----------|
| Llama-3.1-8B-Instruct | Reference | Reference |
| Qwen-2.5-7B | Reference | Reference |
| **NExtLong (Llama-3-8B-Base)** | Outstanding | Outstanding |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| meta-chunk position: front vs. back | Front is superior | Placing meta-chunks before hard negatives yields better performance |
| Chunking granularity $s$ | Significant impact | Needs to be adjusted based on document characteristics and target length |
| No hard negatives (pure concatenation) | Poor performance | Validates the necessity of hard negatives for modeling long-range dependencies |
| Random negatives vs. hard negatives | Hard negatives are superior | Semantically similar distractors better enhance the discriminating ability of the model |

### Key Findings
- NExtLong improves over Quest by an average of 7.33%, showing even greater improvements compared to random concatenation and KNN-based concatenation.
- It comprehensively outperforms models trained on natural long documents across multiple task types, proving that synthetic data can substitute for scarce real long documents.
- The "semantically similar but unrelated" property of hard negatives is key to their effectiveness—random negatives perform significantly worse.
- The method is effective across a range of context lengths from 8K to 128K, showing strong context-length generalization.

## Highlights & Insights
- It ingeniously introduces the concept of hard negatives from contrastive learning into long-context data synthesis, providing a unique perspective.
- Kills two birds with one stone: it both addresses the scarcity of long documents and enhances long-range dependency modeling through robust distractor training.
- Crucially, turning the known weakness "LLMs are easily distracted by irrelevant context" into a training signal serves as a highly inspiring design principle.
- Experiments demonstrate that synthetic data can match or even exceed training with real long documents, carrying significant practical implications for long-context LLM development.

## Limitations & Future Work
- Building FAISS indexes for hard negative retrieval increases the engineering overhead of data preparation.
- Chunking granularity $s$ and the number of hard negatives $k$ require hyperparameter tuning based on target length and document characteristics.
- The quality of hard negatives heavily depends on the deduplication quality of the pre-training corpus.
- Scalability in ultra-long context scenarios (256K+) is not yet fully discussed in the paper.
- The effectiveness of hard negatives may vary across different domains (e.g., code, mathematics).

## Related Work & Insights
- Difference from Quest: Quest balances semantic relevance and diversity via keyword retrieval but lacks an explicit long-range dependency mechanism; NExtLong directly strengthens long-range dependencies through the insertion of hard negatives.
- Orthogonal to train-free methods like LM-Infinite and StreamingLLM: NExtLong resolves training data limitations.
- The philosophy of hard negatives can be generalized to other scenarios requiring enhanced model discrimination capabilities.
- The meta-chunk level retrieval and concatenation framework is highly flexible and can be integrated with other synthesis strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ (The idea of introducing hard negatives to long-context synthesis is novel and intuitive)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Dual benchmarks with HELMET and RULER, comprehensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, detailed methodological description)
- Value: ⭐⭐⭐⭐⭐ (Highly practical, resolving the core bottleneck in long-context LLM development—the scarcity of training data)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ICLR 2026\] EntropyLong: Effective Long-Context Training via Predictive Uncertainty](../../ICLR2026/llm_efficiency/entropylong_effective_long-context_training_via_predictive_uncertainty.md)
- [\[ACL 2025\] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](../../ACL2025/llm_efficiency/adaptive_grouped_pe_context_window.md)
- [\[ACL 2025\] What are the Essential Factors in Crafting Effective Long Context Multi-Hop Instruction Datasets? Insights and Best Practices](../../ACL2025/llm_efficiency/what_are_the_essential_factors_in_crafting_effective_long_context_multi-hop_inst.md)
- [\[ICML 2025\] Curse of High Dimensionality Issue in Transformer for Long-context Modeling](curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)

</div>

<!-- RELATED:END -->
