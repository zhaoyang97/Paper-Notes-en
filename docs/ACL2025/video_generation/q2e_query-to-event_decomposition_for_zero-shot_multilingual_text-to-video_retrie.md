---
title: >-
  [Paper Note] Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval
description: >-
  [ACL 2025][Video Generation][Text-to-Video Retrieval] Q2E proposes a zero-shot query-to-event decomposition method. It leverages the parameterized world knowledge of LLMs and VLMs to decompose simple queries into prequel, current, and sequel events. Combining these with dense video descriptions and speech transcriptions, it achieves SOTA multilingual text-to-video retrieval performance through inverse entropy fusion ranking.
tags:
  - "ACL 2025"
  - "Video Generation"
  - "Text-to-Video Retrieval"
  - "Event Decomposition"
  - "Zero-shot"
  - "LLM knowledge transfer"
  - "Multimodal fusion"
date: 2026-05-08
content_hash: b0d8c9ba15066977
---

# Q2E: Query-to-Event Decomposition for Zero-Shot Multilingual Text-to-Video Retrieval

**Conference**: ACL 2025  
**arXiv**: [2506.10202](https://arxiv.org/abs/2506.10202)  
**Code**: [Available](https://dipta007.github.io/Q2E/)  
**Area**: Video Generation  
**Keywords**: Text-to-Video Retrieval, Event Decomposition, Zero-shot, LLM knowledge transfer, Multimodal fusion

## TL;DR

Q2E proposes a zero-shot query-to-event decomposition method. It leverages the parameterized world knowledge of LLMs and VLMs to decompose simple queries into prequel, current, and sequel events. Combining these with dense video descriptions and speech transcriptions, it achieves SOTA multilingual text-to-video retrieval performance through inverse entropy fusion ranking.

## Background & Motivation

### Background

Text-to-video retrieval is an important multimedia task, but it faces several core challenges:

**Queries are overly concise**: Users typically input short queries (such as "2025 LA fire") but expect the system to understand all aspects of the event.

**Information is scattered across multiple videos**: A single video may only show a part of the event.

**Multilingual barriers**: Videos may not be in the language known to the user.

### Core Motivation

Existing video retrieval systems usually rely on platform metadata or manual annotations (such as titles and search-optimized descriptions), which cannot handle complex real-world event queries. Classic datasets like MSR-VTT and MSVD only contain generic, high-level queries (e.g., "a person is explaining something") rather than complex events.

### Three Key Insights

1. Query decomposition via LLMs can enhance the understanding of coarse-grained queries, retrieving prequel/sequel-related videos that would otherwise be ignored.
2. Although VLM captions and ASR outputs are noisy and redundant, LLM refiners can effectively denoise them.
3. When aggregating rankings from multiple similarity/relevance judgments, entropy-based fusion methods outperform simple methods.

## Method

### Overall Architecture

The Q2E system (Figure 2) consists of four core modules:

1. **Event Decomposition Module (blue part)**: Decomposes queries into prequel/current/sequel events.
2. **Video Decomposition Module (green part)**: Extracts multimodal descriptions from videos.
3. **Audio Decomposition Module (orange part)**: Processes speech through a multi-layer translation pipeline.
4. **Fusion Ranking Module (purple part)**: Fuses all scores via inverse entropy.

### Key Designs

1. **Event Decomposition**:

    - Uses LLMs to decompose a query into three types of sub-events: **Prequel**—prior events that might cause the current event; **Current**—specific observable sub-events during the occurrence; and **Sequel**—subsequent outcomes that might result from the event.
    - Generates 5 sub-events for each category.
    - Uses the same LLM to extract temporal, spatial, and subject event information for **decomposition refinement** to generate more natural queries.

2. **Video Decomposition**:

    - **Contextualized Frame Descriptions**: Uniformly samples 16 frames and uses a sliding window (window=2) VLM to generate contextualized captions for each frame conditioned on the previous frame's description.
    - **Video Descriptions**: Feeds all 16 frame descriptions into an LLM to summarize them into a single dense video caption, preserving temporal information and focusing on the global scene.

3. **Audio Decomposition**: Multi-layer translation pipeline—

    - First layer: Whisper-v3 multilingual ASR (transcribes the original language + English translation).
    - Second layer: NLLB translator (translates the original transcription to English).
    - Third layer: Llama-70B refiner (refines both English translation results).

4. **Scoring and Fusion**:

    - Computes 5 types of scores: (a) Query vs. Video, (b-d) Prequel/Current/Sequel vs. Multimodal descriptions, and (e) Query vs. Multimodal descriptions.
    - Query-to-Video uses the cosine similarity of the MultiCLIP image encoder.
    - Query-to-Description uses ColBERT text similarity (outperforming SBERT, as ColBERT's token-level max-aggregation reduces the impact of noise).
    - Event-to-Description uses many-to-many global maximum similarity.
    - **Inverse Entropy Fusion Ranking**: Converts each score into a softmax distribution, where low entropy indicates high confidence, and fuses them weighted by inverse entropy: $\hat{S} = \sum_{i}^{5} \frac{1}{H(P_i)} \cdot P_i$

### Loss & Training

- **Fully Zero-shot**: Does not fine-tune any model; instead, it leverages the parameterized knowledge of existing LLMs/VLMs.
- The method is highly adaptable across datasets, domains, LLMs, and VLMs.

## Key Experimental Results

### Main Results (Table 1, NDCG Metric)

| Dataset | Encoder | Baseline | +Event | +ASR+Event |
|--------|--------|----------|--------|------------|
| MultiVENT | MultiCLIP | 75.34 | 80.04 | **83.24** |
| MultiVENT | InternVideo2-1B | 50.43 | 69.15 | **76.10** |
| MSR-VTT-1kA | MultiCLIP | 59.72 | 61.51 | **63.59** |
| MSR-VTT-1kA | InternVideo2-1B | 66.07 | 67.16 | **69.53** |
| MSVD | MultiCLIP | 71.69 | **74.10** | - |
| MSVD | InternVideo2-1B | 77.51 | **77.84** | - |

### Ablation Study

| Fusion Method | NDCG↑ |
|---------|-------|
| Neg. Exp. Entropy | 73.20 |
| RRF | 76.29 |
| Max | 80.04 |
| Mean | 82.44 |
| **Inv. Entropy (Q2E)** | **83.24** |

| LLM Size | NDCG↑ |
|----------|-------|
| Baseline (w/o decomposition) | 75.34 |
| 1B | 82.50 |
| 3B | 83.03 |
| 8B | 82.91 |
| **70B** | **83.24** |

### Key Findings

1. **Event decomposition is effective**: Adding only event decomposition (without ASR) improves MultiVENT NDCG by 5-19 points; incorporating ASR yields an additional 3-7 point improvement, reaching a total gain of 8-26 points.
2. **Consistent multilingual improvements**: Low-resource languages like Arabic, Chinese, and Korean experience larger improvements (+6, +9, and +10 NDCG, respectively).
3. **Smaller models are also effective**: Even a 1B parameter LLM can improve the baseline by at least 8 NDCG points.
4. **Inverse entropy fusion is optimal**: It outperforms other methods like Mean, Max, and RRF.
5. **Complementary components**: Removing the video score has the most severe impact (-9.28 NDCG), followed by events (-1.49) and queries (-1.70).

## Highlights & Insights

- **Causal Event Knowledge Transfer**: Creatively leverages the world knowledge of LLMs to expand simple queries into prequel/current/sequel event structures, representing a novel query augmentation paradigm.
- **Multi-layer ASR Pipeline**: Three-layer processing (ASR + Translation + Refinement) effectively resolves speech quality issues in multilingual videos.
- **Zero-shot Plug-and-Play**: Requires no fine-tune operations and can directly replace underlying encoders (such as MultiCLIP/InternVideo2), showing high practical utility.
- **Global Maximum Strategy**: Employs global maximum instead of mean calculation for event-to-description matching, effectively mitigating the negative impacts of LLM hallucination.

## Limitations & Future Work

- **High Computational Overhead**: Requires running multiple large-scale models (LLM, VLM, ASR, Translator), leading to high inference time and cost.
- **Risk of LLM Hallucinated Information**: Fictional information may be generated during event decomposition and caption refinement.
- **Bias Propagation**: Relies on parameterized model knowledge, potentially propagating the models' inherent biases.
- **Focus Only on Text-to-Video**: Video-to-text retrieval has not been explored.
- Future research can explore leveraging factual and counterfactual information for positive/negative alignment.

## Related Work & Insights

- Continues the application of causal event reasoning in NLP, introducing temporal causal reasoning (prequel/sequel) to the video retrieval field for the first time.
- Inspired by Yin & Jiang (2024) to use inverse entropy fusion.
- Insights for RAG systems: The paradigm of query decomposition + multi-source information fusion can be transferred to other retrieval scenarios.
- The multi-layer translation pipeline concept can be applied to other multilingual and multimodal tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Event decomposition + inverse entropy fusion is a novel combination, creatively transferring LLM world knowledge into video retrieval.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering 3 datasets, 2 encoders, 5 languages, and multiple ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear construction and intuitive illustrations, though some mathematical layouts feel slightly crowded.
- **Value**: ⭐⭐⭐⭐ High practical value, with the zero-shot method achieving significant improvements in event-dense video retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](../../CVPR2025/video_generation/zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[CVPR 2025\] Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](../../CVPR2025/video_generation/video-colbert_contextualized_late_interaction_for_text-to-video_retrieval.md)
- [\[ICCV 2025\] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization](../../ICCV2025/video_generation/quantifying_and_narrowing_the_unknown_interactive_text-to-video_retrieval_via_un.md)
- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](../../CVPR2025/video_generation/identity-preserving_text-to-video_generation_by_frequency_decomposition.md)

</div>

<!-- RELATED:END -->
