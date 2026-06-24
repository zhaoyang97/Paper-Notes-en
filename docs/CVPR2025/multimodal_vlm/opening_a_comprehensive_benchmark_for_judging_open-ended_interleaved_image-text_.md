---
title: >-
  [Paper Note] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation
description: >-
  [CVPR 2025][Multimodal VLM][Interleaved Image-Text Generation] This paper proposes the OpenING benchmark (5,400 human-annotated instances, 56 real-world tasks) and the IntJudge evaluation model (82.42% agreement rate with human judgments), filling the vacuum in open-ended interleaved image-text generation evaluation. It finds that current integrated pipelines (e.g., Gemini+Flux) significantly outperform end-to-end models, yet all methods still fall far short of human annotati…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Interleaved Image-Text Generation"
  - "Benchmark Evaluation"
  - "Judge Model"
  - "Multimodal Generation"
  - "Human Alignment"
date: 2026-05-08
content_hash: 3e2b8363581c6672
---

# OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation

**Conference**: CVPR 2025  
**arXiv**: [2411.18499](https://arxiv.org/abs/2411.18499)  
**Code**: [https://opening-benchmark.github.io](https://opening-benchmark.github.io)  
**Area**: Multimodal VLM  
**Keywords**: Interleaved Image-Text Generation, Benchmark Evaluation, Judge Model, Multimodal Generation, Human Alignment

## TL;DR

This paper proposes the OpenING benchmark (5,400 human-annotated instances, 56 real-world tasks) and the IntJudge evaluation model (82.42% agreement rate with human judgments), filling the vacuum in open-ended interleaved image-text generation evaluation. It finds that current integrated pipelines (e.g., Gemini+Flux) significantly outperform end-to-end models, yet all methods still fall far short of human annotation quality.

## Background & Motivation

**Background**: Multimodal large models have made rapid progress in visual understanding and geneneration, and interleaved image-text generation has become an essential capability towards artificial general intelligence. Early models likes DALL-E and Stable Diffusion focused on unidirectional tasks (text-to-image or image understanding). Recently, native autoregressive models such as Emu-3 and Chameleon, as well as two-stage models like SEED-X, have emerged, which are capable of alternately generating text and images.

**Limitations of Prior Work**: The evaluation systems lag seriously behind model development. Existing benchmarks (OpenLEAF has only 660 instances, InterleavedBench has only 815 instances) suffer from small scale, narrow scene coverage, and insufficient query diversity. Crucially, existing evaluations heavily rely on GPT-based scoring, which suffers from biases toward self-generated content, risk of data leakage, and API privacy concerns. Traditional metrics like BLEU/ROUGE fail to measure visual quality, FID/IS ignore textual elements, and CLIPScore cannot comprehensively evaluate open-ended interleaved content.

**Key Challenge**: The community lacks an interleaved image-text generation benchmark of sufficient scale, rich tasks, and a reliable offline judge model. Without effective evaluation, the path for model advancement remains unclear.

**Goal**: (1) Construct a large-scale, high-quality interleaved image-text generation benchmark covering real-world scenarios; (2) Train an offline judge model with high agreement to human judgment; (3) Systematically evaluate the strengths and weaknesses of current methods.

**Key Insight**: Starting from real-world daily scenarios (travel guides, design, brainstorming, etc.), a top-down approach is used to design 23 meta-themes and 56 specific tasks, supported by a 50-person team for high-quality annotation.

**Core Idea**: Construct the OpenING benchmark, a large-scale interleaved image-text benchmark covering 56 real-world tasks, and the IntJudge evaluation model with 82.42% human alignment, systematically evaluating interleaved image-text generation methods.

## Method

### Overall Architecture

The OpenING project consists of three core contributions: (1) The OpenING benchmark, featuring 5,400 human-annotated multi-step interleaved instances; (2) IntJudge, an evaluation model trained on Qwen2-VL-7B; and (3) Interleaved Arena, a pairwise comparison evaluation framework. The data is collected from 20+ sources and constructed through five stages: conceptualization $\rightarrow$ collection $\rightarrow$ annotation $\rightarrow$ filtering $\rightarrow$ processing.

### Key Designs

1. **Top-down Task Conceptualization and Data Annotation**:

    - **Function**: Ensure the breadth and depth of the benchmark in covering real-world scenarios.
    - **Mechanism**: Determine 23 meta-themes (fashion, cooking, travel, design, etc.) via AI agent brainstorming, which are subdivided into 56 specific tasks. Data is collected from 20+ sources such as Xiaohongshu, YouTube, Google, and OpenDataLab. Twenty-eight professional annotators, supervised by 14 data experts, perform annotation using the self-developed IntLabel tool, with each instance restricted to within 10 steps. Cross-checking ensures consistency; unqualified data is discarded and replenished using content generated by GPT-4o+SDXL. Chinese text is translated into English via GPT-4o and manually verified.
    - **Design Motivation**: The collection and standardization of interleaved image-text data are highly challenging—data formats vary significantly across domains, and quality is inconsistent, necessitating strict pipeline quality control.

2. **Interleaved Arena Pairwise Evaluation Framework**:

    - **Function**: Achieve more stable open-ended evaluation through pairwise comparisons.
    - **Mechanism**: Draw data instances from the test set to perform pairwise comparisons of outputs from two anonymous models. The evaluation is based on seven dimensions: correctness, text-image consistency, multi-step coherence, content quality, human preference alignment, completeness, and content richness. A roulette matching algorithm is used to sample $E$ different battle pairs for each data instance, covering time $T_k = \lceil \frac{|\mathcal{M}|(|\mathcal{M}|-1)}{2E} \cdot \frac{D_k}{|\mathcal{P}_k|} \rceil$, ensuring all models are evaluated.
    - **Design Motivation**: Pairwise comparisons are more stable than subjective absolute scoring (prior studies have proven that too many ties reduce evaluation efficiency), and Arena-style evaluation has been validated as effective in LLM evaluation.

3. **IntJudge Evaluation Model Training**:

    - **Function**: Provide offline, reproducible, and highly human-aligned automatic evaluation.
    - **Mechanism**: Train on Qwen2-VL-7B. Data originates from two parts: (1) Human-annotated pairwise comparison data on the Dev Set; (2) Reference-Augmented Generation (RAG) augmented data—where the model is provided with gold answers to generate RAG results, which are then paired with standard generation results (with RAG results as the winner). The training loss combines four terms: $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{CE}} + \lambda_2 \mathcal{L}_{\text{CT}} + \lambda_3 \mathcal{L}_{\text{MSE}} + \lambda_4 \mathcal{L}_{\text{PR}}$ (Cross-Entropy + Contrastive + MSE + Pairwise Ranking loss).
    - **Design Motivation**: GPT-based evaluation suffers from self-bias, API privacy, and cost issues. An offline judge model is controllable, reproducible, and introduces no risk of data leakage.

### Loss & Training

IntJudge is trained with a weighted combination of four losses: CE ensures classification accuracy, contrastive loss distinguishes between good and bad outputs, MSE brings predicted scores closer to actual scores, and pairwise ranking loss ensures correct preference ordering.

## Key Experimental Results

### Main Results (Model Win Rate Ranking — IntJudge Evaluation)

| Method | Type | FDT Win Rate | w/ Tie(.5) Win Rate |
|------|------|---------|----------------|
| Human | Annotation | 87.46% | 84.23% |
| GPT-4o+DALL-E3 | Integrated Pipeline | 85.02% | 80.68% |
| Gemini1.5+Flux | Integrated Pipeline | 68.30% | 65.41% |
| SEED-X | Two-stage | 49.86% | 49.72% |
| Anole | End-to-end | 53.42% | 51.33% |
| SEED-LLaMA | End-to-end | 50.13% | 48.48% |
| Show-o | Two-stage | 31.49% | 32.87% |
| NExT-GPT | End-to-end | 30.96% | 32.58% |
| MiniGPT-5 | End-to-end | 24.47% | 27.85% |
| GILL | End-to-end | 24.87% | 30.32% |

### Judge Model Agreement

| Judge | Agreement with Human (FDT) | Agreement with Human (w/o Tie) |
|--------|-------------------|---------------------|
| GPT-4o | 71.08% | 74.58% |
| **IntJudge** | **82.42%** | **87.46%** |
| **Gain** | **+11.34%** | **+12.88%** |

### Key Findings

- Integrated pipelines (GPT-4o+DALL-E3) significantly outperform other approaches across all evaluation methods with an 85%+ win rate, demonstrating that current interleaved image-text generation still relies heavily on the coordination of powerful, independent text and image generation models.
- End-to-end models (such as Anole, SEED-LLaMA) have win rates clustered between 25-53%, lagging far behind human annotations (87%+).
- IntJudge achieves an 82.42% agreement rate with humans, significantly outperforming GPT-4o's 71.08%, thereby achieving superior human alignment as an offline judge model.
- Regarding text, GPT can generate richer information than human annotations; however, regarding images, the natural images in human annotations remain superior to generated ones.
- IntJudge maintains excellent generalization performance on unseen models.

## Highlights & Insights

- **Establishing Standards in an Evaluation Vacuum**: There is almost no reliable evaluation system in the field of interleaved image-text generation. OpenING concurrently provides datasets, judge models, and leaderboards, building a complete evaluation infrastructure. The three-month investment of a 50-person team guarantees data quality.
- **RAG Data Augmentation Strategy**: Utilizing gold answers as references to prompt models to generate RAG results, which are then paired with standard generation results to train the evaluation judge. This bootstrapping-style data augmentation method is elegant, cost-effective, and transferrable to other scenarios requiring judge models.
- **Seven-Dimension Evaluation System**: The seven dimensions from correctness to human preference alignment provide much more fine-grained evaluation signals than a simple single score, aiding in diagnosing specific weaknesses of the models.

## Limitations & Future Work

- Highlighting that 5,400 instances, though an order of magnitude larger than previous works, only averages ~96 instances per task across 56 tasks, which might lead to insufficient coverage for certain tasks.
- Translating Chinese data to English may introduce translation bias, affecting the naturalness of tasks with non-Chinese backgrounds.
- IntJudge is based on Qwen2-VL-7B. Its limited model capacity might lead to imprecise judgments on highly complex interleaved content.
- The evaluation framework focuses on content quality, lacking considerations for generation efficiency (latency and cost).
- Some data is supplemented using GPT-4o+SDXL, which may introduce distribution bias.

## Related Work & Insights

- **vs InterleavedBench**: Only 815 instances and 10 tasks without an offline judge model. OpenING represents a comprehensive upgrade in scale (5,400 instances), coverage (56 tasks), and evaluation tools (IntJudge).
- **vs OpenLEAF**: Only 660 instances across two meta-themes, not open-source, and lacks an offline judge model. OpenING comprehensively surpasses it in all aspects.
- **vs LMSYS Arena**: Arena-style evaluation has matured in the text-only domain; extending it to multimodal interleaved generation in OpenING is a natural progression.
- The finding that integrated pipelines vastly outperform end-to-end models is critical—implying that the current bottleneck of interleaved image-text generation lies not in framework design but in base model capabilities (specifically image generation quality).

## Rating

- Novelty: ⭐⭐⭐⭐ Fills the gap in interleaved image-text generation evaluation, though the core methodologies (benchmark construction + Judge training) lean towards engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive with 10 models, 3 evaluation modes, and multi-dimensional comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed data statistics, and reasonable task design.
- Value: ⭐⭐⭐⭐⭐ Provides much-needed evaluation infrastructure for the rapidly advancing field of interleaved image-text generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[ICML 2026\] ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation](../../ICML2026/multimodal_vlm/eca_efficient_continual_alignment_for_open-ended_image-to-text_generation.md)
- [\[ICLR 2026\] A High Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation](../../ICLR2026/multimodal_vlm/a_high_quality_dataset_and_reliable_evaluation_for_interleaved_image-text_genera.md)
- [\[CVPR 2025\] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)

</div>

<!-- RELATED:END -->
