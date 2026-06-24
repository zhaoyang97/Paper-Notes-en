---
title: >-
  [Paper Note] What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations
description: >-
  [ACL 2025][Text Generation][video summarization] This paper proposes the VISTA dataset consisting of 18,599 AI conference presentation videos paired with paper abstracts, and introduces a plan-based summarization framework that guides structured summary generation for scientific videos by generating intermediate question sequences, significantly improving factual consistency.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "video summarization"
  - "scientific presentations"
  - "plan-based generation"
  - "multimodal"
  - "dataset"
date: 2026-05-08
content_hash: fffe408d65731364
---

# What Is That Talk About? A Video-to-Text Summarization Dataset for Scientific Presentations

**Conference**: ACL 2025  
**arXiv**: [2502.08279](https://arxiv.org/abs/2502.08279)  
**Code**: [dongqi.me/projects/VISTA](https://dongqi.me/projects/VISTA)  
**Area**: Video-to-Text Summarization  
**Keywords**: video summarization, scientific presentations, plan-based generation, multimodal, dataset  

## TL;DR

This paper proposes the VISTA dataset consisting of 18,599 AI conference presentation videos paired with paper abstracts, and introduces a plan-based summarization framework that guides structured summary generation for scientific videos by generating intermediate question sequences, significantly improving factual consistency.

## Background & Motivation

**Problem Definition:** Transforming recorded academic presentation videos into concise and accurate text summaries (i.e., video-to-text summarization) is an increasingly important challenge in multimodal learning. Existing datasets primarily target generic content (YouTube, movies, news) and lack specialized datasets and methods for academic scientific videos.

**Limitations of Prior Work:** Large Multimodal Models (LMMs) experience performance degradation in scientific scenarios, particularly when dealing with technical terminology and scientific visual elements (diagrams, charts). End-to-end summary generation methods struggle to capture the inherent structure of scientific summaries (background-methodology-results-conclusion), leading to generated content that lacks organization and factual accuracy.

**Design Motivation:** Scientific summaries typically follow a relatively fixed structure, making them well-suited for structured generation strategies. By introducing an intermediate plan to explicitly model the latent structure of the summary, the generation process can be better-guided.

## Method

### Overall Architecture

The system comprises two independently trained modules:
1. **Plan Generation (PG) Module:** Given a video $v$, it generates a plan $p = \{q_1, q_2, \ldots, q_m\}$ (a sequence of questions).
2. **Summary Generation (SG) Module:** Given the concatenation of the video and the plan $[v; p]$, it generates the final summary $s$.

During inference, the PG module first predicts a plan $\hat{p}$, and then $[v; \hat{p}]$ is fed into the SG module to generate the summary. The learning objective is expanded from $P(s|v)$ to $P(s|v,p)$.

### Key Designs

1. **VISTA Dataset Construction:** Presentation videos paired with paper abstracts from 2020-2024 are collected from the ACL Anthology (ACL, EMNLP, NAACL, EACL, etc.) and ML conferences (ICML, NeurIPS). Tutorials, invited talks, and excessively short or long videos (<1 minute or >30 minutes) are excluded, resulting in a final collection of 18,599 samples. On average, videos are 6.76 minutes long with 16.36 shots; summaries average 192.6 tokens and 7.19 sentences.

2. **Plan Generation based on QUD Theory:** Inspired by the Question Under Discussion (QUD) theory, this work assumes that each sentence in the summary can be viewed as an answer to a specific question. GPT-o1 is utilized to generate a silver-standard question sequence based on the reference summary sentences and preceding contexts. The order of questions aligns with the original sequence of summary sentences, ensuring plan coherence.

3. **Comprehensive Benchmark Evaluation System:** The framework covers three modal baselines: text-to-text (LLaMA-3.1 + transcript/OCR), audio-to-text (Qwen2-Audio), and video-to-text (various LMMs), as well as three training settings: zero-shot, QLoRA, and full fine-tuning.

### Loss & Training

Both PG and SG modules employ the standard autoregressive language modeling loss. PG is trained on $(v, p)$ pairs, while SG is trained on $([v;p], s)$ tuples. Both modules share the same backbone but are trained independently.

## Experiments

### Main Results: Zero-Shot Evaluation (Selected)

| Method | Open-source | R1 | R2 | RLsum | BLEU | BERTscore | FactVC |
|------|------|------|------|-------|------|-----------|--------|
| LLaMA-3.1 (transcript) | ✓ | 23.68 | 4.22 | 21.39 | 2.70 | 80.93 | 34.32 |
| Claude 3.5 Sonnet | ✗ | 27.71 | 5.59 | 24.14 | 3.14 | 82.57 | 50.11 |
| GPT-o1 | ✗ | 27.90 | 5.69 | 24.37 | 4.38 | 82.63 | 51.36 |
| mPLUG-Owl3 | ✓ | 25.57 | 4.82 | 22.84 | 2.99 | 81.39 | 42.07 |
| **Plan-mPLUG-Owl3** | ✓ | **25.62** | **4.95** | **22.97** | **3.14** | **81.45** | **47.37** |

### QLoRA Fine-Tuning Results (Selected)

| Method | R1 | R2 | RLsum | BLEU | BERTscore | FactVC |
|------|------|------|-------|------|-----------|--------|
| mPLUG-Owl3 | 33.40 | 12.82 | 30.66 | 8.29 | 83.49 | 70.08 |
| **Plan-mPLUG-Owl3** | **33.52** | **13.01** | **31.10** | 8.33 | **83.53** | **73.11** |
| LLaVA-NeXT-Interleave | 33.37 | 12.77 | 30.56 | 8.30 | 83.47 | 66.14 |

### Key Findings

1. **Consistent Improvements with Plan-based Methods:** Under both zero-shot and fine-tuning settings, Plan-mPLUG-Owl3 consistently and significantly outperforms the end-to-end mPLUG-Owl3. Notably, the improvement in factual consistency (FactVC) is the most prominent (zero-shot: 42.07 -> 47.37; QLoRA: 70.08 -> 73.11).
2. **Significant Domain Fine-Tuning Multiplier:** After QLoRA fine-tuning, R1 improves from ~25 to ~33, and R2 improves from ~5 to ~13, demonstrating that domain-specific data is crucial for scientific video summarization.
3. **Video Models Outperform Text-only/Audio-only Models:** Under identical conditions, video LMMs generally outperform models that only use transcribed text or audio, indicating that visual information (slides, charts) is valuable for understanding scientific presentations.
4. **Closed-source Models Lead but the Gap Can Be Narrowed:** Under the zero-shot setting, GPT-o1/Gemini 2.0 significantly lead open-source models, but fine-tuning allows open-source models to substantially narrow the gap.
5. **Significant Gap Remains Between Models and Humans:** Even the best performing models remain significantly below human-level performance across multiple metrics.

## Highlights & Insights

- The first large-scale scientific presentation video summarization dataset, containing 18,599 samples covering top conferences such as ACL, EMNLP, NAACL, ICML, and NeurIPS.
- The plan-based framework cleverly leverages the structured nature of academic summaries, explicitly modeling the summary structure as a question sequence via QUD theory.
- Comprehensive experimental design: covering three modal inputs, three training settings, and over ten models, providing an exhaustive benchmark.
- Includes human evaluations and error analysis, thoroughly diagnosing the key issues in model-generated summaries.

## Limitations & Future Work

- Only covers conferences in computational linguistics and machine learning, and has not yet been extended to other disciplines (e.g., biology, physics).
- Plan generation relies on silver-standard questions generated by GPT-o1, whose quality upper bound is limited by the comprehension ability of GPT-o1.
- Using paper abstracts as a proxy for video summaries may result in information emphasis discrepancies between the two.
- The dataset only covers English, without considering multilingual scientific presentation scenarios.

## Related Work & Insights

- **Video-to-Text Summarization Datasets:** VideoXum (Lin et al., 2023), MMSum (Qiu et al., 2024), Instruct-V2Xum (Hua et al., 2024)
- **Scientific Text Summarization:** TalkSumm (Lev et al., 2019) academic video transcript summarization, ACLSum (Takeshita et al., 2024) ACL paper summarization
- **Plan-based Summarization:** Entity/QA-based planning framework by Narayan et al. (2021, 2023), Blueprint models (Liu et al., 2023) blueprint framework for visual storytelling
- **Scientific Multimodality:** M3AV (Chen et al., 2024) supporting ASR/TTS/slide script generation

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Text Summarization via Global Structure Awareness](../../ICLR2026/nlp_generation/text_summarization_via_global_structure_awareness.md)
- [\[ACL 2025\] DTCRS: Dynamic Tree Construction for Recursive Summarization](dtcrs_dynamic_tree_construction_for_recursive_summarization.md)
- [\[ACL 2025\] Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)
- [\[ACL 2025\] Decomposed Opinion Summarization with Verified Aspect-Aware Modules](decomposed_opinion_summarization_with_verified_aspect-aware_modules.md)
- [\[ACL 2025\] An Empirical Study of Many-to-Many Summarization with Large Language Models](an_empirical_study_of_manytomany_summarization.md)

</div>

<!-- RELATED:END -->
