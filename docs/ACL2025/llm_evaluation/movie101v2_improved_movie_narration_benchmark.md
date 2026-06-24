---
title: >-
  [Paper Note] Movie101v2: Improved Movie Narration Benchmark
description: >-
  [ACL 2025][LLM Evaluation][movie narration] Proposes Movie101v2, a large-scale bilingual movie narration benchmark (203 movies, 46K Chinese-English video-narration pairs). It decomposes automatic movie narration into a three-stage progressive goal: L1 visual factual description $\rightarrow$ L2 plot narration $\rightarrow$ L3 deployable AD. It designs an LLM-based hierarchical evaluation framework, systematically benchmarks multiple LVLMs, and provides an in-depth analysis of…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "movie narration"
  - "video description"
  - "benchmark dataset"
  - "vision-language model"
  - "accessibility technology"
date: 2026-05-08
content_hash: a877f1fc577aa56c
---

# Movie101v2: Improved Movie Narration Benchmark

**Conference**: ACL 2025  
**arXiv**: [2404.13370](https://arxiv.org/abs/2404.13370)  
**Code**: [Movie101 Project](https://movie101-dataset.github.io)  
**Area**: LLM Evaluation  
**Keywords**: movie narration, video description, benchmark dataset, vision-language model, accessibility technology

## TL;DR

Proposes Movie101v2, a large-scale bilingual movie narration benchmark (203 movies, 46K Chinese-English video-narration pairs). It decomposes automatic movie narration into a three-stage progressive goal: L1 visual factual description $\rightarrow$ L2 plot narration $\rightarrow$ L3 deployable AD. It designs an LLM-based hierarchical evaluation framework, systematically benchmarks multiple LVLMs, and provides an in-depth analysis of the core bottlenecks in visual perception and text generation.

## Background & Motivation

**Task Background**: Audio Description (AD) is a core accessibility technology that helps visually impaired individuals understand movies by inserting voiceover descriptions of visual content during gaps in character dialogue. Professional AD production is expensive and capacity-limited, making it unable to cover the vast amount of film and television resources. Therefore, automatic movie narration generation holds significant social value.

**Limitations of Prior Work**: Existing datasets suffer from systematic limitations. The average clip length in M-VAD is only 6.2 seconds, and in MAD, it is only 4.1 seconds. Such short clips restrict the model's capability for coherent narration in complex plots. LSMDC replaces character names with "someone", downgrading movie narration to generic video description. Although Movie101 includes longer clips and character information, it only contains 101 movies and 14K samples, is limited to Chinese, and contains errors in metadata caused by automatic scraping.

**Key Challenge**: Prior works lack a unified task objective for movie narration: AutoAD emphasizes context dependency (introducing historical narration and subtitles), while AutoAD II requires predicting narration timestamps. However, experiments demonstrate that the basic visual understanding capability remains unresolved, making these extra requirements an unnecessary distraction at the current stage.

**Unreasonable Evaluation**: Mainstream evaluation methods directly match model outputs with reference narrations (CIDEr / BLEU / ROUGE). However, reference narrations are written by human experts based on rich contexts (plot history, character background, audio), while models only have access to a single video clip. Such direct comparison is inherently unfair and fails to provide effective optimization feedback.

## Method

### Overall Architecture

Movie101v2 advances along three axes: **Data, Task, and Evaluation**. It expands the data scale based on Movie101 to 203 movies and 46K bilingual samples. It decomposes the ultimate goal into three progressive stages (L1 $\rightarrow$ L2 $\rightarrow$ L3) and proposes LLM-based L1-Score and L2-Score hierarchical evaluation metrics to replace direct reference matching.

### Key Designs

**1. Automated Data Construction Pipeline**

The entire pipeline utilizes expert models and LLMs to achieve low-cost automation:

| Step | Tool | Function |
|------|------|------|
| Voice Transcription | Whisper | Converts movie audio to text (including narration + dialogue) |
| Dialogue Removal | PaddleOCR + GPT-4 | OCR detects subtitles to identify dialogue time intervals; GPT-4 removes remaining dialogue |
| Text Correction | GPT-3.5-turbo | Corrects typos, punctuation errors, and meaningless phrases |
| Clip Merging | Heuristic dynamic threshold | Merges adjacent narration segments into coherent paragraphs to avoid over-segmentation |
| Chinese-to-English translation | GPT-3.5-turbo | Works with manually constructed English cast lists to ensure correct character names |
| Character Name Unification | Manual + GPT-3.5-turbo | Completes the cast list $\rightarrow$ automatically aligns character names in narration with official names |

Quality control strategies: The LLM processes only one refinement step at a time (avoiding multi-step combination that reduces quality); references adjacent contexts during batch processing; provides In-Context Learning (ICL) examples; human verification of 300 samples shows the quality is comparable to the crowdsourced refinement of Movie101.

**2. Three-Stage Task Roadmap**

Decomposing the ultimate goal of "automatically generating deployable AD" into a progressive route:

| Stage | Goal | Input | Core Capability Requirement |
|------|------|------|-------------|
| **L1** Visual Factual Description | Accurately describe scenes, objects, and character actions in the clip | Single video clip | Basic visual perception |
| **L2** Plot Narration | Reason across multiple shots to describe plot development | Single video clip | Cross-shot reasoning, story understanding |
| **L3** Deployable AD | Generate timely, properly paced narration scripts | Full movie + multi-modal context | Long-sequence modeling, multi-modal alignment |

The key difference where L2 surpasses L1 is: movies convey plots through shot sequences, requiring the model to piece together information fragments into a coherent story rather than simply listing visual facts.

**3. LLM-Based Hierarchical Evaluation Metrics**

- **L1-Score** (0–5 points): Evaluates the coverage of visual facts in the narration, divided into the environment sub-dimension (scenes/objects/events) and the character sub-dimension (names/actions/emotions).
- **L2-Score** (0–5 points): Evaluates the consistency between the generated narration and the reference narration in terms of plot delivery, disregarding differences in linguistic phrasing.
- Evaluator LLMs: DeepSeek-V2.5 is used for Chinese, and Llama-3.1-70B-Instruct is used for English to ensure reproducibility.

### Loss & Training

Open-source models are fine-tuned on the Movie101v2 training set for 3 epochs: freezing the visual encoder and only training the visual projector and the LLM's LoRA adapters. Since GPT-4V cannot be fine-tuned, ICL is performed using carefully designed task instructions combined with randomly retrieved training examples. Video models fuse character portraits as extra frames with video features early on (which performs better than independent encoding); multi-image models evenly divide the video into $K$ concatenated frames, while character portraits are concatenated into a single image with text annotations added.

## Key Experimental Results

### Dataset Comparison

| Dataset | No. of Movies | No. of Clips | Avg. Duration (s) | Avg. Text Length | No. of Characters | Bilingual |
|--------|-------|-------|-------------|-------------|-------|------|
| M-VAD | 92 | 49K | 6.2 | 9.1 words | — | ✗ |
| MAD | 650 | 385K | 4.1 | 12.7 words | — | ✗ |
| Movie101 | 101 | 14K | 20.4 | 80.7 characters | 2.0 | ✗ |
| **Movie101v2** | **203** | **46K** | **12.8** | **60.0 characters / 39.1 words** | **1.9** | **✓** |

### Baseline Results

Baseline models include VideoGPT+, VideoChat-2, VideoLLaMA 2, InternVL2, CogVLM2-Video, Qwen-VL, Qwen2-VL, and GPT-4V.

**Core Findings**:
- GPT-4V without fine-tuning performs the strongest in the Chinese setup, showing excellent cross-lingual generalization capability.
- Among open-source models, VideoGPT+, VideoLLaMA 2, InternVL2, and Qwen2-VL show comparable performance, with varying strengths in L1 and L2.
- All models consistently improve after incorporating external character knowledge (portraits + names), highlighting the critical importance of character understanding.
- However, even with character knowledge, the performance of all models remains far below deployable levels.

### Visual Perception Analysis

| Analysis Dimension | Key Data | Conclusion |
|---------|---------|------|
| Input Capacity | GPT-4V recalls 77.8% of visual facts at 2 FPS; drops sharply at <1 FPS; 23.1% of clips have insufficient information at 1 FPS | Visual context window is the primary bottleneck |
| Visual Understanding | Good recognition of objects and scenes; difficulty in recognizing character actions and expressions | Character behavior understanding is a core shortcoming |
| Face Recognition | GPT-4V precision is only 43.6%, underperforming ArcFace (47.8%) | General LVLMs find it difficult to perform character re-identification |
| Text Generation | Movie101v2 achieves the highest GPT-2 perplexity and n-gram diversity; perplexity of VideoChat-2 remains much higher than on other datasets even after training | Narration text complexity far exceeds that of generic video description |

### Context Dependency Validation

Human evaluation of 1,000 clips indicates that generating reference-quality narration generally requires multi-modal information such as plot history and character contexts. As contexts are step-by-step removed, the ability to generate accurate narration declines significantly. This validates the irrationality of direct reference-matching evaluation.

## Highlights & Insights

- **Practical and clear three-stage roadmap**: Acknowledges current technical limitations, focusing L1/L2 on single-clip understanding, and leaving L3 for future breakthroughs in long-sequence modeling and multi-modal alignment.
- **Evaluation framework innovation**: LLM-based hierarchical scoring avoids the unfairness of reference narration matching—since reference narrations are generated based on rich human contexts, directly comparing them with model outputs based solely on video clips is inherently unfair.
- **Reproducible data construction pipeline**: The LLM automates most steps, offering a low-cost and scalable approach that provides a data expansion paradigm for the community.
- **Bilingual design** fills the gap of lacking English counterparts in Chinese movie narration, while also enabling non-Chinese researchers to re-use this benchmark.

## Limitations & Future Work

- Simplifying to single-clip understanding to accommodate technical limitations might constrain more radical research on entire movie understanding.
- The data only contains Chinese movies, lacking cultural diversity.
- The quality of English narrations translated by LLMs has not been verified by large-scale human evaluation.
- The technical path for L3 remains unclear, with long-sequence modeling and multi-modal alignment pending breakthroughs.
- The consistency between LLM scores and human judgements requires deeper validation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The three-stage task decomposition and hierarchical LLM evaluation are systematic contributions.
- **Practicality**: ⭐⭐⭐⭐ — The large-scale bilingual benchmark + reproducible data construction pipeline hold direct value for the community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dimensional analysis (perception/understanding/recognition/generation) is in-depth and highly informative.
- **Writing Quality**: ⭐⭐⭐⭐ — Complete structure, clear arguments, and intuitive roadmap presentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](../../ICCV2025/llm_evaluation/discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ACL 2025\] CFBench: A Comprehensive Constraints-Following Benchmark for LLMs](cfbench_a_comprehensive_constraints-following_benchmark_for_llms.md)
- [\[ACL 2025\] SwiLTra-Bench: The Swiss Legal Translation Benchmark](swiltra-bench_the_swiss_legal_translation_benchmark.md)

</div>

<!-- RELATED:END -->
