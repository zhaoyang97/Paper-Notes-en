---
title: >-
  [Paper Note] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR
description: >-
  [AAAI 2026][Audio & Speech][Conversational ASR] MARS proposes a multimodal retrieval-and-selection approach to identify the most relevant historical context for conversational LLM-ASR—rather than relying on a fixed numbe…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Conversational ASR"
  - "LLM-ASR"
  - "Multimodal Retrieval"
  - "RAG"
  - "Historical Context Selection"
date: 2026-05-08
content_hash: 962677a3dffb6e70
---

# Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR

**Conference**: AAAI 2026
**arXiv**: [2508.01166](https://arxiv.org/abs/2508.01166)
**Code**: None
**Area**: Speech Recognition / Audio Processing
**Keywords**: Conversational ASR, LLM-ASR, Multimodal Retrieval, RAG, Historical Context Selection

## TL;DR

MARS proposes a multimodal retrieval-and-selection approach to identify the most relevant historical context for conversational LLM-ASR—rather than relying on a fixed number of preceding utterances or the entire history—achieving state-of-the-art performance with only 1.5K hours of training data, surpassing TEA-ASLP trained on 179K hours.

## Background & Motivation

**Background**: Conversational ASR must leverage historical context to handle challenges such as speaking style, filler words, and discourse coherence. Recent LLM-ASR approaches have demonstrated the potential for exploiting long contexts.

**Limitations of Prior Work**: Existing conversational LLM-ASR systems fall into two extremes when utilizing context: (1) *Fixed preceding N utterances*: assumes the most relevant context lies in the recent turns, whereas in practice the most relevant history may appear much earlier, and recent turns may be dominated by filler words and irrelevant content; (2) *Full dialogue history*: provides rich context but introduces substantial redundancy, disrupting recognition and incurring high computational cost.

**Key Challenge**: The position of the most relevant historical context is not fixed—it may occur early in the dialogue—while the full history contains excessive irrelevant information. A mechanism that precisely locates the most relevant historical context is therefore needed.

**Goal**: Retrieve and select the single most helpful historical utterance from the entire dialogue history for a given current utterance, thereby enhancing conversational LLM-ASR performance.

**Key Insight**: Inspired by the retrieval paradigm of RAG, but customized for ASR—RAG aims at generating new content, whereas ASR maps speech to text, representing fundamentally different objectives. MARS performs retrieval from both speech and text modalities, then applies a near-ideal ranking strategy to select the optimal single context.

**Core Idea**: Retrieve candidate historical contexts via speech-and-text dual-modal retrieval, and apply a TOPSIS-style near-ideal ranking that integrates both similarity scores to select the single best context for the LLM, realizing a "less is more" approach to context utilization.

## Method

### Overall Architecture

The MARS pipeline proceeds as follows: (1) A fine-tuned Whisper model is used to construct a database storing triples of utterance ID, speech embedding, and hypothesis for each utterance; (2) For the current utterance, the multimodal retrieval module retrieves the Top-$K$ most similar historical contexts from the database using speech and text modalities separately; (3) The multimodal selection module identifies the single best historical context from the retrieved candidates; (4) The hypothesis of the best context, the speech embedding and hypothesis of the current utterance, and a language prompt are jointly fed into the LLM to generate the transcription.

### Key Designs

1. **Multimodal Retrieval**:

    - *Function*: Retrieve the Top-$K$ most similar historical contexts from the entire dialogue history using speech and text modalities independently.
    - *Mechanism*: The speech modality computes frame-level acoustic similarity via FastDTW (minimum cumulative alignment distance between two speech embedding sequences) combined with pooled cosine similarity as utterance-level similarity; a weighted sum yields the speech retrieval score for Top-$K$ selection. The text modality uses an embedding model (Qwen3-Embedding-0.6B) to compute semantic similarity between hypotheses for Top-$K$ selection. Speech retrieval captures pronunciation variants to reduce phonetic errors, while text retrieval resolves lexical ambiguity.
    - *Design Motivation*: Single-modality retrieval cannot comprehensively measure similarity—speech similarity captures pronunciation and prosody, while text similarity captures semantic relatedness; the two are complementary.

2. **Near-Ideal Ranking Multimodal Selection**:

    - *Function*: Select the single context from the $2K$ retrieved candidates that optimally combines speech and text similarity.
    - *Mechanism*: Both similarity scores are first computed for all $2K$ candidates (supplementing scores missing from each unimodal retrieval set). Since the two scores have different scales, they cannot be directly summed for ranking. A TOPSIS-style procedure is adopted: (a) normalize both scores to eliminate scale differences, $sr_i = sw_i / \sqrt{\sum sw_j^2}$; (b) define the ideal point (both scores maximized) and the negative-ideal point (both scores minimized); (c) compute the Euclidean distances from each candidate to the ideal and negative-ideal points, $d_i^+$ and $d_i^-$; (d) compute the relative closeness $c_i = d_i^- / (d_i^+ + d_i^-)$ and select the candidate with the highest value as the best context.
    - *Design Motivation*: The two similarity measures are derived from different methods and operate on different scales, precluding simple summation. The TOPSIS method is naturally suited to multi-criteria ranking with heterogeneous metrics.

3. **Adaptive Context Decoding Strategy**:

    - *Function*: Randomly determine during training whether to use the retrieved historical context.
    - *Mechanism*: The best historical context is randomly masked with 50% probability to prevent the model from over-relying on history at the expense of the current utterance. At inference, three decoding modes are supported: direct decoding (no history), MARS decoding (single-pass), and two-pass decoding (first pass generates an initial hypothesis; MARS then re-retrieves and re-decodes using this hypothesis).
    - *Design Motivation*: Enhances generalization, enabling the model to maintain strong performance when no suitable historical context is available.

### Loss & Training

Qwen2.5-7B-Instruct is used as the LLM with LoRA (rank=64, alpha=256) fine-tuning applied to seven projection layers. The projector consists of two linear layers with ReLU activations. Training runs for 3 epochs using the Adam optimizer with a peak learning rate of 0.0001. All checkpoints are averaged for inference.

## Key Experimental Results

### Main Results

| Method | Training Data | MER (Dev) | MER (Test) |
|--------|--------------|-----------|------------|
| Vanilla Whisper-large-v3 | Pretrained | 16.82 | 17.33 |
| Fine-tuned Whisper | 1.5K hr | 11.87 | 10.15 |
| Qwen2-Audio | Pretrained | 51.90 | 53.47 |
| TEA-ASLP (Prev. SOTA) | **179K hr** | 10.62 | 9.60 |
| **MARS** | **1.5K hr** | **8.97** | **8.35** |

Using only 1.5K hours of training data, MARS achieves a MER 1.25 points lower than TEA-ASLP trained on 179K hours (relative improvement of ~13%).

### Ablation Study

| Configuration | MER (Dev) | MER (Test) | Notes |
|---------------|-----------|------------|-------|
| LLM-ASR (no context) | 12.75 | 11.04 | Baseline |
| + Hypothesis | 11.15 | 9.89 | Text hypothesis is beneficial |
| + Speech Retrieval | 10.24 | 9.41 | Speech retrieval is effective |
| + Text Retrieval | 10.33 | 9.23 | Text retrieval is effective |
| + Multi-modal Selection | 9.77 | 8.96 | Joint selection yields further gains |
| + Two-pass Decoding | **8.97** | **8.35** | Two-pass decoding is optimal |

### Key Findings

- Fixed preceding-$N$ context provides limited benefit, and performance degrades as $N$ increases (MER 13.49 at $N$=5 vs. 9.74 at $N$=1), confirming that redundant information is harmful.
- Even when ground-truth transcriptions are used as context, the Bi-context approach yields smaller improvements than MARS with retrieved context, demonstrating that *selecting the right context* matters more than *context quality*.
- Two-pass decoding substantially outperforms single-pass, as the first-pass hypothesis is more accurate and thus produces a higher-quality database for subsequent retrieval.

## Highlights & Insights

- **Remarkable data efficiency**: A system trained on 1.5K hours outperforms one trained on 179K hours, demonstrating that precise context utilization is more effective than brute-force data scaling—a finding with significant implications for low-resource scenarios.
- **Near-ideal ranking**: An elegant multi-criteria decision-making method (TOPSIS) that enables unified comparison of speech and text similarities with heterogeneous scales; this approach is transferable to any setting requiring the joint ranking of multiple heterogeneous criteria.
- **"Less is more" design philosophy**: Selecting a single best context rather than multiple candidates avoids information redundancy, contrasting with the common RAG practice of filling the context window.

## Limitations & Future Work

- Retrieval quality depends on the first-pass Whisper output; severely erroneous initial hypotheses may degrade subsequent retrieval and selection.
- Near-ideal ranking assumes equal importance of speech and text similarity (uniform weights); adaptive weighting remains unexplored.
- Only a single historical context is selected; certain scenarios may benefit from multiple complementary contexts.
- FastDTW still incurs non-trivial computational overhead for large-scale dialogues; more efficient approximation methods could be considered.
- Evaluation is conducted solely on the MLC-SLM dataset; generalization to other conversational settings requires further investigation.

## Related Work & Insights

- **vs. TEA-ASLP**: The previous SOTA relies on 179K hours of large-scale data and a MoE architecture; MARS surpasses it using approximately 1/100 of the data through precise context utilization.
- **vs. Seewo/Bi-context**: Fixed preceding-$N$ strategies show limited gains even with ground-truth transcriptions; MARS demonstrates the superiority of retrieval-based context selection.
- **vs. RAG**: RAG focuses on external knowledge retrieval for new content generation, whereas MARS restricts retrieval to intra-dialogue history with the goal of assisting transcription rather than generation.

## Rating

- Novelty: ⭐⭐⭐⭐ Creatively adapts the RAG paradigm to conversational ASR; near-ideal ranking is a genuinely novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale multilingual evaluation, detailed ablations, and comprehensive comparison against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, figures and tables are informative, and the overall presentation is logically coherent.
- Value: ⭐⭐⭐⭐⭐ Exceptional data efficiency offers high practical value; establishes a new state of the art on MLC-SLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)
- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](../../ICLR2026/audio_speech/incentive-aligned_multi-source_llm_summaries.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[AAAI 2026\] Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning](towards_authentic_movie_dubbing_with_retrieve-augmented_director-actor_interacti.md)

</div>

<!-- RELATED:END -->
