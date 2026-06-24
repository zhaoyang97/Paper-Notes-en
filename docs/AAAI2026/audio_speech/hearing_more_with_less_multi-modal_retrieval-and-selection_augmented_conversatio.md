---
title: >-
  [Paper Note] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR
description: >-
  [AAAI 2026][Audio & Speech][Conversational ASR] MARS proposes a multi-modal retrieval-and-selection approach to select the most relevant historical context for conversational LLM-ASR (instead of using a fixed set of preceding sentences or the entire history). It outperforms the SOTA system TEA-ASLP, which is trained on 179K hours of data, using only 1.5K hours of training data.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Conversational ASR"
  - "LLM-ASR"
  - "Multi-modal Retrieval"
  - "RAG"
  - "Historical Context Selection"
date: 2026-05-08
content_hash: 640443c4f6b842bf
---

# Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR

**Conference**: AAAI 2026  
**arXiv**: [2508.01166](https://arxiv.org/abs/2508.01166)  
**Code**: None  
**Area**: Speech Recognition / Audio Processing  
**Keywords**: Conversational ASR, LLM-ASR, Multi-modal Retrieval, RAG, Historical Context Selection

## TL;DR

MARS proposes a multi-modal retrieval-and-selection approach to select the most relevant historical context for conversational LLM-ASR (instead of using a fixed set of preceding sentences or the entire history). It outperforms the SOTA system TEA-ASLP, which is trained on 179K hours of data, using only 1.5K hours of training data.

## Background & Motivation

**Background**: Conversational speech recognition requires utilizing historical context to address challenges such as speaking styles, filler words, and contextual links. Recent LLM-ASR methods have demonstrated potential in leveraging long contexts.

**Limitations of Prior Work**: Existing conversational LLM-ASR approaches use context in two extremes: (1) Fixed preceding N sentences: assuming that the most relevant context lies in the most recent sentences, whereas the actual target history might reside earlier in the conversation, and recent sentences might be cluttered with filler words or irrelevant content; (2) Full conversation history: providing rich context but introducing substantial redundant information that disrupts recognition and incurs high computational overhead.

**Key Challenge**: The position of historical context is not fixed, and the most relevant context may occur early in the conversation, while the full history contains too much irrelevant information. There is a need for a mechanism that precisely locates the most relevant historical context.

**Goal**: How to retrieve and select the single most helpful historical context from the entire conversation history to enhance conversational LLM-ASR performance.

**Key Insight**: Taking inspiration from the retrieval concepts in RAG but customized for ASR scenarios. While RAG aims to generate new content, ASR focuses on mapping speech to text, which has distinct objectives. MARS performs retrieval across both speech and text modalities and uses a near-ideal ranking method to select the best single context.

**Core Idea**: Utilizing speech and text multi-modal retrieval to fetch candidate historical context, followed by a TOPSIS-style near-ideal ranking to integrate both similarity scores to choose the best single context for LLM input, achieving a "less is more" paradigm of context utilization.

## Method

### Overall Architecture

The pipeline of MARS is as follows: (1) Construct a database with fine-tuned Whisper storing the ID, speech embedding, and hypothesis triplet for each utterance; (2) For the current utterance, the multi-modal retrieval module retrieves the Top-K similar historical contexts separately in speech and text modalities; (3) The multi-modal selection module determines the single best historical context from the retrieval results; (4) The hypothesis of the selected context, the speech embedding and hypothesis of the current utterance, and language prompts are fed into the LLM to generate the final transcription.

### Key Designs

1. **Multi-Modal Retrieval**:

    - **Function**: Retrieves the Top-K similar historical contexts from the entire conversation history separately using speech and text modalities.
    - **Mechanism**: The speech modality uses FastDTW to calculate frame-level acoustic similarity (the cumulative distance between aligned speech embeddings) plus the cosine similarity after pooling as the utterance-level similarity. A weighted sum is computed to obtain the final speech retrieval similarity, and the Top-K are selected. The text modality uses an embedding model (Qwen3-Embedding-0.6B) to compute semantic similarity between hypotheses and select Top-K. Speech retrieval alleviates pronunciation variations and reduces mispronunciations, while text retrieval resolves word ambiguity.
    - **Design Motivation**: Uni-modal retrieval cannot comprehensively measure similarity. Speech similarity captures pronunciation and prosody, while text similarity captures semantic relevance; the two are complementary.

2. **Near-Ideal Ranking Multi-Modal Selection**:

    - **Function**: Selects the single context that optimizes both speech and text similarity from the $2K$ candidates in the retrieval results.
    - **Mechanism**: First, both similarity scores are calculated for all $2K$ candidates (filling in missing scores that were absent in individual modality retrieval). Since the two similarity metrics have different scales, they cannot be directly summed for ranking. A TOPSIS-style method is adopted: (a) Normalize both similarities to eliminate scale differences: $sr_i = sw_i / \sqrt{\sum sw_j^2}$; (b) Define an ideal point (where both similarities are optimal) and a negative-ideal point (where both are worst); (c) Calculate the Euclidean distance of each candidate to the ideal point and negative-ideal point, denoted as $d_i^+$ and $d_i^-$, respectively; (d) Calculate the relative closeness $c_i = d_i^- / (d_i^+ + d_i^-)$, and select the one with the maximum value as the best context.
    - **Design Motivation**: The two similarity metrics are computed using different methods and scales, making them impossible to add together directly. The TOPSIS approach is naturally suited for multi-criteria decision-making across heterogeneous scales.

3. **Adaptive Context Decoding Strategy**:

    - **Function**: Randomly decides whether to use the retrieved historical context during training.
    - **Mechanism**: Randomly masks the selected best historical context with a 50% probability to prevent the model from over-relying on context while ignoring the current utterance itself. During inference, three decoding options are supported: direct decoding (without history), MARS decoding (one-pass), and two-pass decoding (first performing direct decoding to obtain the initial hypothesis, then re-retrieving and decoding with MARS).
    - **Design Motivation**: To enhance generalization capability so that the model maintains robust performance even in the absence of suitable historical context.

### Loss & Training

Using Qwen2.5-7B-Instruct as the LLM, LoRA (rank=64, alpha=256) fine-tunes seven projection layers. The projector consists of two linear layers with ReLU. Training is conducted for 3 epochs using the Adam optimizer with a peak learning rate of 0.0001. All checkpoints are averaged for inference.

## Key Experimental Results

### Main Results

| Method | Training Data | MER (Dev) | MER (Test) |
|------|---------|-----------|------------|
| Vanilla Whisper-large-v3 | Pre-trained | 16.82 | 17.33 |
| Fine-tuned Whisper | 1.5K hr | 11.87 | 10.15 |
| Qwen2-Audio | Pre-trained | 51.90 | 53.47 |
| TEA-ASLP (Prev. SOTA) | **179K hr** | 10.62 | 9.60 |
| **MARS** | **1.5K hr** | **8.97** | **8.35** |

Using only 1.5K hours of training data, MARS achieves a MER that is 1.25 points lower (representing a 13% relative improvement) compared to TEA-ASLP, which is trained on 179K hours of data.

### Ablation Study

| Configuration | MER (Dev) | MER (Test) | Explanation |
|------|-----------|------------|------|
| LLM-ASR (No Context) | 12.75 | 11.04 | Baseline |
| + Hypothesis | 11.15 | 9.89 | Text hypothesis is helpful |
| + Speech Retrieval | 10.24 | 9.41 | Speech retrieval is effective |
| + Text Retrieval | 10.33 | 9.23 | Text retrieval is effective |
| + Multi-modal Selection | 9.77 | 8.96 | Multi-modal selection further improves performance |
| + Two-pass Decoding | **8.97** | **8.35** | Two-pass decoding achieves the best results |

### Key Findings

- The effect of a fixed preceding N-sentence context is limited, and performance even decreases as N increases (MER is 13.49 when N=5 vs. 9.74 when N=1), validating that redundant information is detrimental.
- Even if the ground-truth transcriptions are used as context, the improvement from the Bi-context method is not as substantial as using context retrieved by MARS. This indicates that "selecting the right context" is more critical than the "quality of the context".
- Two-pass decoding significantly outperforms one-pass decoding because the first-pass hypothesis is more accurate, providing a higher-quality database.

## Highlights & Insights

- **Astonishing Data Efficiency**: Surpassing a system trained on 179K hours with only 1.5K hours of training data demonstrates that precise context exploitation is far more effective than brute-force data scaling. This offers profound insights for low-resource scenarios.
- **Near-Ideal Ranking**: An ingenious multi-criteria decision-making method (TOPSIS) that normalizes and compares speech and text similarities from heterogeneous scales. This approach can be transferred to any scenario requiring the synthesis of multiple heterogeneous metrics.
- **"Less is More" Design Philosophy**: Selecting only a single optimal context instead of multiple ones helps avoid information redundancy, standing in contrast to the common practice in RAG of "filling up the context length".

## Limitations & Future Work

- Retrieval depends on the quality of the first-pass Whisper decoding. If the initial hypothesis has severe errors, subsequent retrieval and selection might be affected.
- The near-ideal ranking assumes that speech and text similarities are equally important (equal weighting) and has not explored adaptive weights.
- Only a single historical context is selected, whereas multiple complementary contexts might be required in certain situations.
- FastDTW still incurs certain computational overheads in large-scale conversations; more efficient approximation methods could be considered.
- Validated only on the MLC-SLM dataset; further experiments are required to generalize to other conversational scenarios.

## Related Work & Insights

- **vs. TEA-ASLP**: The previous SOTA relied on 179K hours of large-scale data and MoE architectures. MARS achieves superior performance with just 1/100 of the data by leveraging context accurately.
- **vs. Seewo/Bi-context**: Strategies utilizing a fixed N preceding sentences show limited improvement even when using ground-truth transcriptions. MARS demonstrates the superiority of a retrieval-based context selection.
- **vs. RAG**: While RAG focuses on retrieving external knowledge to generate new content, MARS constrains retrieval to internal dialogue history with the goal of aiding transcription rather than generation.

## Rating

- Novelty: ⭐⭐⭐⭐ Creatively adapts the RAG approach to conversational ASR; near-ideal ranking is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts large-scale multi-lingual evaluation, detailed ablations, and comprehensive comparisons with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions, rich illustrations and tables, and smooth logic.
- Value: ⭐⭐⭐⭐⭐ Phenomenal data efficiency offers high practical value, setting a new SOTA on MLC-SLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[ACL 2025\] Soundwave: Less is More for Speech-Text Alignment in LLMs](../../ACL2025/audio_speech/soundwave_less_is_more_for_speech-text_alignment_in_llms.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)

</div>

<!-- RELATED:END -->
