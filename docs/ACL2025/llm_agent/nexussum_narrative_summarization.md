---
title: >-
  [Paper Note] NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization
description: >-
  [ACL 2025][LLM Agent][Long-Form Summarization] Proposes NexusSum, a three-stage multi-agent LLM framework (dialogue-to-description $\to$ hierarchical summarization $\to$ iterative compression) that generates summaries for long narrative texts like books, movies, and TV series without fine-tuning, achieving up to a 30% BERTScore improvement on BookSum.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Long-Form Summarization"
  - "Narrative Understanding"
  - "Multi-Agent Framework"
  - "Dialogue Transformation"
  - "Iterative Compression"
date: 2026-05-08
content_hash: 2970609c1a214c02
---

# NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization

**Conference**: ACL 2025  
**arXiv**: [2505.24575](https://arxiv.org/abs/2505.24575)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: Long-Form Summarization, Narrative Understanding, Multi-Agent Framework, Dialogue Transformation, Iterative Compression  

## TL;DR
Proposes NexusSum, a three-stage multi-agent LLM framework (dialogue-to-description $\to$ hierarchical summarization $\to$ iterative compression) that generates summaries for long narrative texts like books, movies, and TV series without fine-tuning, achieving up to a 30% BERTScore improvement on BookSum.

## Background & Motivation
**Background**: Long-form narrative summarization (books, movie scripts, TV scripts) requires capturing complex plotlines, evolving character relationships, and thematic consistency. Existing methods mainly fall into three categories: long-context modeling (e.g., extended windows), extractive-generative pipelines, and multi-agent frameworks.

**Limitations of Prior Work**: Even with a 200K-token context window, LLMs still lose information when processing long narratives; extractive pipelines miss critical details, disrupting narrative coherence; and zero-shot LLMs perform significantly worse on narrative summarization compared to fine-tuned models.

**Key Challenge**: Narrative text mixes descriptive prose and multi-speaker dialogues. The fragmented structure makes it difficult for LLMs to generate coherent summaries. Additionally, output length control is challenging, resulting in summaries that are either too long or miss key events.

**Goal**: (1) How to summarize long-form text while maintaining narrative structure and coherence? (2) How does dialogue-to-description transformation improve consistency? (3) How does iterative compression balance length control and content retention?

**Key Insight**: Observing that the fragmentation of dialogues in narrative texts is the key reason for incoherent summaries, it is proposed to first unify dialogues into third-person narrative prose before hierarchical processing.

**Core Idea**: Formulate a hierarchical pipeline with three specialized LLM agents (preprocessor $\to$ summarizer $\to$ compressor) to achieve high-quality long-form narrative summarization without fine-tuning, through dialogue-to-description transformation, scene chunking, and iterative compression.

## Method

### Overall Architecture
NexusSum is a three-stage sequential pipeline: the input is the full narrative text (40K-160K tokens), which goes through (1) a Preprocessor Agent that transforms dialogues into descriptive prose, (2) a Narrative Summarizer Agent that chunks the preprocessed text to generate initial summaries, and (3) a Compressor Agent that iteratively compresses them to the target length. All three stages utilize a chunk-and-concat strategy without requiring fine-tuning.

### Key Designs

1. **Dialogue-to-Description Transformation (Preprocessor Agent $P$)**:

    - **Function**: Transforms character dialogue in narrative texts into structured, third-person narrative prose.
    - **Mechanism**: Splits the input text into chunks of 8 scenes each, i.e., $N = n_1 \oplus n_2 \oplus \cdots \oplus n_k$. The LLM then rewrites the dialogue in each chunk, preserving the speakers' intents while unifying them into a descriptive format, outputting $N' = P(n_1) \oplus \cdots \oplus P(n_k)$.
    - **Design Motivation**: Multi-speaker dialogues in narratives lead to fragmented summaries. Converting them into a unified prose format makes it easier for the LLM to capture semantic coherence. Ablation studies show this step contributes +2.45 BERTScore.

2. **Hierarchical Narrative Summarization (Summarizer Agent $S$)**:

    - **Function**: Generates initial summaries for the preprocessed text.
    - **Mechanism**: Chunks $N'$ by scene and generates summaries for each chunk independently, then concatenates them: $S_0 = S(n'_1) \oplus S(n'_2) \oplus \cdots \oplus S(n'_j)$. Unlike traditional single-pass generation, hierarchical chunking preserves long-range information.
    - **Design Motivation**: Directly processing long texts causes context loss. By chunking, each agent only processes text of a manageable length, ensuring information retention. This contributes +4.86 BERTScore, making it the highest contributor among the three modules.

3. **Iterative Compression (Compressor Agent $C$)**:

    - **Function**: Iteratively compresses the initial summary to a target word count $\theta$.
    - **Mechanism**: Splits $S_0$ at the sentence level (with chunk size $\delta$ tokens) and compresses iteratively: $S_i = C_i(s_{i-1,1}) \oplus \cdots \oplus C_i(s_{i-1,l_{i-1}})$. If $S_i$ still exceeds $\theta$, it continues to the next round, up to 10 rounds.
    - **Design Motivation**: $\delta$ controls the compression ratio (smaller input chunks yield lower compression rates). Through multiple iterations, precise length control is achieved, with LAR (Length Adherence Rate) close to 1.0.

### Loss & Training
Completely training-free and fine-tuning-free. Mistral-Large-Instruct-2407 (123B) is used as the base model, with inference run via vLLM (temperature=0.3, top-p=1.0). The framework can be further adapted to different dataset styles through CoT reasoning and few-shot learning.

## Key Experimental Results

### Main Results
Comparison of BERTScore (F1) across four long-form narrative summarization benchmarks:

| Dataset | NexusSum | Prev. SOTA | Gain |
|--------|----------|-----------|------|
| BookSum (Book) | **70.70** | 54.4 (CachED) | +30.0% |
| MovieSum (Movie) | **63.53** | 59.32 (HM-SR) | +7.1% |
| MENSA (Script) | **65.73** | 64.6 (CachED) | +1.7% |
| SummScreenFD (TV) | **61.59** | 61.59 (CachED) | Flat |

### Ablation Study
Contribution of each module on the MENSA dataset (gradually accumulated):

| Configuration | BERTScore (F1) | Gain |
|------|---------------|------|
| Zero-Shot baseline | 54.81 | - |
| + Preprocessor ($P$) | 57.26 | +2.45 |
| + Summarizer ($S$) | 62.12 | +4.86 |
| + Compressor ($C$) = NexusSum | 65.73 | +1.83 |

### Key Findings
- The Summarizer Agent has the largest contribution (+4.86), indicating that chunk-based hierarchical summarization is core to the performance improvement.
- Regarding length control, NexusSum's LAR reaches 0.99 for target lengths of 900 and 1200 words, far exceeding Zero-Shot's ~0.5.
- CoT + Few-Shot prompt engineering on SummScreenFD yields an additional +5.0 BERTScore, showing strong framework adaptability.
- Human evaluation shows that NexusSum outperforms Zero-Shot in key event coverage (4.17) and factual accuracy (4.0), but has noticeably poorer readability (2.17 vs. 4.17); adding a Rewrite Agent restores readability to 3.67.

## Highlights & Insights
- **Ingenious pre-processing idea of dialogue-to-description**: The fragmentation of narrative texts inherently stems from the dialogue format. Unifying the format before summarization is a simple yet effective design that can be transferred to other dialogue-containing text processing tasks.
- **Iterative compression achieves precise length control**: Precise control over output length is enabled via two parameters, $\delta$ (chunk size) and $\theta$ (target words). This strategy is valuable for any generation task with length constraints.
- **Fine-tuning-free multi-agent collaboration**: The three agents perform distinct functions and adapt to different datasets/domains solely through prompt engineering, demonstrating the flexibility of LLM agent frameworks.

## Limitations & Future Work
- **Significant readability gap**: In human evaluation, NexusSum's readability score is only 2.17/5, far below Zero-Shot's 4.17, indicating that highly dense summaries are not necessarily preferred by humans.
- **Limitations of evaluation metrics**: A 30% increase in BERTScore while humans prefer Zero-Shot exposes the misalignment between automatic metrics and human preferences.
- **High computational cost**: Running the 123B model requires 4 A100 GPUs, and the inference time of the three-stage pipeline is not fully discussed.
- **Small-scale human evaluation**: Only 3 evaluators assessed 3 Korean TV shows, which lacks statistical significance.

## Related Work & Insights
- **vs. CoA (Chain of Agents)**: CoA is a general-purpose multi-agent summarization framework. NexusSum adds a dialogue-to-description preprocessor tailored for narrative texts, outperforming CoA by 4.6% in ROUGE.
- **vs. CachED**: CachED uses gradient caching for efficient fine-tuning. NexusSum is completely training-free but significantly outperforms CachED on BookSum, suggesting that in narrative domains, domain-specific preprocessing can be more effective than model fine-tuning.
- **vs. HM-SR**: HM-SR performs hierarchical merging + refinement but lacks precise length control. NexusSum's iterative compression mechanism is its core differentiating advantage.

## Rating
- Novelty: ⭐⭐⭐ Dialogue-to-description is a new contribution, but multi-agent chunk-based summarization frameworks are not novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four datasets with ablation and human evaluations, though the human evaluation scale is too small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and reasonable layout of diagrams and tables.
- Value: ⭐⭐⭐ Narrative summarization is a relatively narrow application area. The substantial BERTScore increase versus poor readability exposes evaluation issues.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/llm_agent/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[ACL 2026\] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](../../ACL2026/llm_agent/timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)
- [\[ACL 2025\] Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)

</div>

<!-- RELATED:END -->
