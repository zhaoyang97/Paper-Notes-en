---
title: >-
  [Paper Note] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation
description: >-
  [ACL2026][Audio & Speech][Video Retrieval-Augmented Generation] MARQUIS decomposes multi-video retrieval-augmented article generation into three stages: "Query Decomposition and Re-ranking Retrieval…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Video Retrieval-Augmented Generation"
  - "Query Decomposition"
  - "Evidence Extraction"
  - "Uncertainty Calibration"
  - "RLM Controller"
date: 2026-05-08
content_hash: 55121312c2f19c3d
---

# MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation

**Conference**: ACL2026  
**arXiv**: [2605.17640](https://arxiv.org/abs/2605.17640)  
**Code**: https://github.com/debashishc/marquis  
**Area**: Video RAG / Multimodal Retrieval / Evidence-Attributed Generation  
**Keywords**: Video Retrieval-Augmented Generation, Query Decomposition, Evidence Extraction, Uncertainty Calibration, RLM Controller

## TL;DR
MARQUIS decomposes multi-video retrieval-augmented article generation into three stages: "Query Decomposition and Re-ranking Retrieval," "Calibrated Structured Evidence Extraction," and "Cited Article Generation." It can also employ an RLM controller for iterative evidence management. On MAGMaR2026, it improved retrieval nDCG@10 from 0.195 to 0.759, and the Iter-QA-Base achieved a human score of 3.83 on the generation side.

## Background & Motivation
**Background**: Video corpora record a vast number of real-world events. However, transforming audio-visual evidence from multiple videos into a cited, attributable, and structurally clear analytical article still largely relies on manual effort. Traditional RAG is primarily designed for text; video RAG must also process visual frames, audio, transcripts, cross-video evidence synthesis, and citation attribution.

**Limitations of Prior Work**: Problems exist in both retrieval and generation. In retrieval, queries for tasks like MAGMaR are often long, containing professional personas, backgrounds, and multiple implicit/explicit information needs; a single dense embedding tends to compress multi-faceted requirements into one vector, missing relevant videos. In generation, models encounter issues such as excessive context length, insufficient cross-video reasoning, disordered citations, and unclear factual support when facing multiple long videos.

**Key Challenge**: Directly feeding a large volume of video content into a long-context VLM to write an article is both expensive and unreliable. However, simply performing video retrieval is insufficient because article generation requires fine-grained, citable, and calibratable evidence units. The system needs to bridge "finding videos" and "writing articles" with an evidence management layer.

**Goal**: The authors aim to construct a modular pipeline: first, decompose complex queries into retrievable atomic sub-queries to improve recall; then, transform videos into structured evidence with estimated support probabilities; and finally, generate cited articles based only on filtered evidence. An additional MARQUIS-RLM uses structured memory and tool calls to control evidence collection and organization.

**Key Insight**: The paper treats video RAG as an evidence-management problem. The key is not to have a single model "watch all videos and summarize" in one go, but to decompose query, retrieval, extraction, calibration, and citation generation into checkable steps.

**Core Idea**: Repair retrieval with query decomposition and rank fusion; repair evidence granularity with query-agnostic notes, query-conditioned claims, and QA evidence; filter unsupported claims using CLUE support probabilities; and compare different evidence synthesis methods using strategies such as Bullet, CAG, GINGER, and RLM.

## Method

### Overall Architecture
MARQUIS consists of three stages. Stage 1, **Video Retrieval**: Decomposes complex queries into multiple atomic sub-queries, each independently retrieved using OmniEmbed, followed by fusion of ranked lists using strategies like RRF and similarity aggregation, and video-native reranking of Top-100 candidates using RankVideo. Stage 2, **Information Extraction**: Extracts query-agnostic notes, query-conditioned claims, and question-answer evidence from retrieved videos in parallel, and uses CLUE to calibrate whether each piece of evidence is supported by the source video. Stage 3, **Article Generation**: Hands filtered evidence artifacts to a generator, comparing Bullet, CAG, GINGER, QA-based generation, and the MARQUIS-RLM controller.

MARQUIS-RLM is an optional high-level control system. It wraps retrieval, extraction, QA, calibration, and generation modules as tools called by a Root LM in a persistent Python sandbox via a Think-Act-Observe loop, maintaining a structured memory bank to reduce evidence forgetting, cross-source confusion, and missed conflicts during long processes.

### Key Designs
1. **Query Decomposition, Fusion, and Video Re-ranking Retrieval**:
	- **Function**: Transforms complex, multi-faceted queries into short queries better suited for dense retrievers and recovers overall ranking.
	- **Mechanism**: The LLM decomposes the original query into $N$ atomic sub-queries, each retrieving the Top-1000 video candidates. Fusion strategies include RRF, Sum/Max/Mean similarity, Weighted RRF, etc. For instance, RRF uses $RRF_K(v)=\sum_i 1/(K+rank(v,q_i))$ to aggregate multiple ranked lists. RankVideo is then used to rerank the fused Top-100.
	- **Design Motivation**: Dense retrievers are typically trained on short query-document pairs; long persona + multi-requirement queries are out-of-distribution inputs. Decomposing them into atomic information needs makes it easier for the retriever to find videos covering different facets.

2. **Three-Way Evidence Extraction and Video Support Calibration**:
	- **Function**: Converts video candidates into fine-grained evidence artifacts that can be selected, filtered, cited, and synthesized.
	- **Mechanism**: Query-agnostic note extraction records directly observable visual events, on-screen text, and speech; query-conditioned claim extraction only extracts claims relevant to the current query and directly supported by the video; QA extraction decomposes information needs into questions answered by the VLM based on video content and transcripts. All outputs are scored by CLUE to obtain a support probability $s_\theta(v,x) \in [0,1]$, used to filter unsupported claims.
	- **Design Motivation**: Separating extraction and support estimation prevents the model from confidently misjudging the credibility of evidence while generating it. Multi-way evidence also covers broad observations, task-related claims, and targeted QA.

3. **Evidence-Driven Article Generation and RLM Controller**:
	- **Function**: Generates cited articles from structured evidence and compares different evidence organizations.
	- **Mechanism**: Bullet directly lists evidence, being conservative but non-prose; CAG synthesizes a cited article in one shot; GINGER performs facet clustering, cluster ranking, and per-cluster summarization before polishing into prose; MARQUIS-RLM allows the Root LM to iteratively call tools in a persistent environment, utilizing a memory bank to search, reuse, and revise evidence records while explicitly handling conflicts and information gaps.
	- **Design Motivation**: The most error-prone part of multi-video generation is not linguistic fluency but evidence organization and citation maintenance. RLM turns "filling gaps, resolving conflicts, and organizing facts" from a one-time prompt into observable state transitions.

### Loss & Training
MARQUIS is primarily a system pipeline rather than an end-to-end training method. Experiments use OmniEmbed for video/query encoding, Qwen3.5-9B for query decomposition and extraction, Qwen3.5-27B for QA and article generation, and Qwen2.5-Omni-7B with Whisper medium.en for multimodal embedding and transcription. Claim-based extraction/generation does not use audio; the QA pipeline and RLM can access audio via transcription tools. Evaluation is conducted on the MAGMaR2026 Test Set using nDCG and Recall for retrieval, and MiRAGE automatic metrics plus 1-5 scale ratings from 3 human annotators for generation.

## Key Experimental Results

### Main Results
The improvement in the retrieval stage on MAGMaR2026 is highly significant:

| Method | nDCG@10 | nDCG@20 | R@10 | R@20 | Note |
|------|---------|---------|------|------|------|
| OmniEmbed | 0.195 | 0.229 | 0.190 | 0.276 | Single-query dense retrieval baseline |
| Max Sim | 0.722 | 0.743 | 0.639 | 0.731 | Strongest first-stage nDCG |
| RRF K=10 | 0.700 | 0.739 | 0.612 | 0.735 | Balanced recall |
| Sum Sim + RankVideo | 0.747 | 0.758 | 0.636 | 0.711 | Significant gain after reranking |
| RRF K=10 + RankVideo | 0.759 | 0.771 | 0.652 | 0.735 | Best overall nDCG@10 |

The generation stage compares 8 systems under the oracle relevant videos setting:

| System | Human Score | Best Votes | Best % | Info P/R | Cite P/R | Key Observations |
|------|-------------|------------|--------|----------|----------|----------|
| CAG baseline | 3.09 | 1 | 1.8% | 76.4 / 41.0 | 61.7 / 22.8 | One-shot synthesis baseline |
| Bullet | 2.67 | 0 | 0.0% | 71.1 / 39.4 | 60.4 / 23.7 | Conservative but not an article |
| GINGER | 3.12 | 6 | 10.5% | 77.6 / 40.4 | 64.3 / 22.6 | Strongest prose among claim-based |
| MARQUIS-RLM | 3.30 | 3 | 5.3% | 70.8 / 38.5 | 59.2 / 27.2 | Strongest citation recall among non-QA |
| SS-QA-GINGER | 3.42 | 10 | 17.5% | 54.4 / 32.4 | 32.6 / 23.8 | Most best votes |
| Iter-QA-Base | 3.83 | 8 | 14.0% | 34.7 / 31.3 | 26.8 / 25.8 | Highest average human score |
| Iter-QA-GINGER | 3.69 | 5 | 8.8% | 34.5 / 29.0 | 25.7 / 22.6 | QA + GINGER remains strong |

### Ablation Study
| Comparison Point | Result | Implication |
|--------|------|------|
| Original OmniEmbed vs query expansion | nDCG@10 from 0.195 to max 0.722 | Query decomposition is the main driver of retrieval gain |
| RRF K=10 first-stage vs +RankVideo | nDCG@10 from 0.700 to 0.759 | Video reranking further improves ranking quality |
| Max Sim vs Max Sim + RankVideo | nDCG@10 from 0.722 dropped to 0.399 | RankVideo failed significantly with Max Sim fusion |
| CAG vs GINGER | Human Score 3.09 to 3.12, Best % 1.8% to 10.5% | Facet clustering/staged generation improves readability |
| CAG vs MARQUIS-RLM | Human Score 3.09 to 3.30, Citation Recall 22.8 to 27.2 | RLM evidence management improves attribution recall |

### Key Findings
- In retrieval, the most critical factor is decomposing complex queries into atomic sub-queries; this relocates the retriever back to the familiar short-query distribution.
- While RRF methods may not have the highest first-stage nDCG, they cover multiple facets better and become optimal after reranking.
- There is no single winner on the generation side: QA systems receive high human scores but tend to conservatively refuse to write when they fail, pulling down automatic metrics; claim-based systems have good precision but may lack overall synthesis capability; RLM produces better citation recall with structured memory but may include more irrelevant facts.

## Highlights & Insights
- **Breaking down video RAG into an evidence pipeline is correct**: Video article generation is not just "watching more videos before writing," but requires five steps: retrieval, extraction, calibration, organization, and citation. MARQUIS's modularity makes each step replaceable and evaluable.
- **Calibration is independent of extraction**: Extracting evidence first and then independently judging support from the source video reduces the circular self-validation where the "model hallucinates evidence and grants itself high confidence."
- **RLM acts as a state management tool rather than a gimmick**: In long multi-video tasks, a memory bank is more important than a longer context. The value of RLM lies in allowing the Root LM to explicitly search and revise evidence records.
- **Tension between human ratings and automatic metrics**: Iter-QA-Base has the highest human rating, but its automatic info/citation precision is not high. This serves as a reminder that video RAG evaluation cannot rely solely on automatic metrics.

## Limitations & Future Work
- The severe degradation of Max Sim + RankVideo suggests interaction failures between different fusion strategies and rerankers, which the paper does not explain in depth.
- QA systems refuse to write articles when sub-question decomposition or VLM answering fails; while this avoids hallucinations, it results in zero-score outputs for topics that originally contained information.
- MARQUIS-RLM has high citation recall but lower information and citation precision, indicating that iterative evidence collection introduces more noise and requires stronger evidence filtering or integration with GINGER.
- Claim-based extraction/generation does not use audio, potentially missing speech-only information; though QA/RLM can access transcripts, system complexity is higher.
- The paper primarily validates on MAGMaR2026; cross-domain video libraries, real-time video streams, non-English speech, and longer event chains still require further testing.

## Related Work & Insights
- **vs Text RAG**: Text RAG usually retrieves passages; MARQUIS retrieves videos and extracts citable evidence, addressing additional challenges in modality, timestamps, and citation support.
- **vs Single-video caption/QA**: Traditional video understanding focuses on captions or entity-centric QA; this task requires synthesizing analytical articles across multiple videos, which is harder for evidence organization.
- **vs Long-context VLM**: Extending context can accommodate more videos but does not guarantee attribution or conflict handling; MARQUIS replaces "blindly stuffing context" with structured evidence management.
- **Insights**: Future multimodal RAG systems should use "evidence units" as the core intermediate representation, preserving source IDs, timestamps, support probabilities, and claim dependencies, rather than just saving natural language summaries.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The three-stage video RAG framework is not a single-point breakthrough, but the combination of evidence calibration + RLM control is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Systematic experiments, human scores, and automatic metrics are included for both retrieval and generation; the limitation is the concentration on MAGMaR2026.
- Writing Quality: ⭐⭐⭐⭐☆ The pipeline is clear, and tables are info-dense; however, some module details are scattered in the appendix.
- Value: ⭐⭐⭐⭐⭐ Highly referential for designing video RAG, evidence-attributed generation, and multimodal retrieval systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](../../CVPR2026/audio_speech/echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)

</div>

<!-- RELATED:END -->
