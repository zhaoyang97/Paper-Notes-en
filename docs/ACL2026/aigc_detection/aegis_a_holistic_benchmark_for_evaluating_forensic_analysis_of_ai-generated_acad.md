---
title: >-
  [Paper Note] AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images
description: >-
  [ACL 2026][AIGC Detection][AI-generated image forensics] AEGIS is the first comprehensive benchmark for academic image forgery forensics, covering 7 major academic image categories and 39 subcategories…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "AI-generated image forensics"
  - "academic image"
  - "benchmark"
  - "MLLM"
  - "manipulation localization"
date: 2026-05-08
content_hash: ed58488e7c8310fa
---

# AEGIS: A Holistic Benchmark for Evaluating Forensic Analysis of AI-Generated Academic Images

**Conference**: ACL 2026  
**arXiv**: [2604.28177](https://arxiv.org/abs/2604.28177)  
**Code**: https://github.com/BUPT-Reasoning-Lab/AEGIS (Available)  
**Area**: Image Generation / AIGC Detection / Academic Integrity  
**Keywords**: AI-generated image forensics, academic image, benchmark, MLLM, manipulation localization

## TL;DR
AEGIS is the first comprehensive benchmark for academic image forgery forensics, covering 7 major academic image categories and 39 subcategories, 4 forgery strategies (entirely fabricated, reference-based rewriting, local inpainting, local editing), and 25 generative models. It proposes four tasks—forgery scope discrimination, text artifact recognition, manipulation type classification, and tampered pixel localization. Evaluating 25 MLLMs and 9 expert models reveals that even GPT-5.1 achieves a composite score of only 48.80%, and expert models reach only 30.09% in pixel IoU, highlighting the structural complementarity of "generation evolving faster than forensics" and "MLLM reasoning vs. expert model sensitivity."

## Background & Motivation

**Background**: The misuse of AI-generated images in academic papers has emerged as a new risk to publishing ethics (cases of retractions and public inquiries are already appearing on Retraction Watch and PubPeer). Existing forensic methods fall into three categories: visual expert models based on the frequency domain, diffusion processes, or patch-level features (DRCT, DIRE, AIDE, etc.), MLLMs used as general discriminators, and hybrid schemes (FakeShield, SIDA, FakeVLM).

**Limitations of Prior Work**: Existing benchmarks (GenImage, Semi-Truths, AIGIBench, DFBench, AIGuard, etc.) are almost entirely oriented toward general scenes such as faces, landscapes, or e-commerce. They adapt poorly to the "fine-grained structures, dense textures, and knowledge-intensive semantics" of academic images. Furthermore, most only evaluate "real/fake" binary classification, ignoring the capabilities needed for real academic review: **forgery scope, text anomalies, manipulation types, and pixel localization**.

**Key Challenge**: Academic image forensics is not a single binary classification but a chained judgment from "global $\rightarrow$ local $\rightarrow$ pixel." Moreover, **expert models excel at low-level visual fingerprints but lack semantic reasoning, while MLLMs excel at semantic reasoning but have poor low-level sensitivity**—no single benchmark has exposed the true structural shortcomings of both model types simultaneously.

**Goal**: Construct an all-encompassing forensic benchmark dedicated to academia that can: (1) systematically cover the diversity of 7 major academic categories and 39 subcategories; (2) simulate 4 typical forgery strategies under 25 SOTA generators; (3) differentiate model capabilities across 4 progressive tasks from overall authenticity to pixel-level localization.

**Key Insight**: Extract 8,000 verified "panels" (the smallest indivisible unit of a figure) from 4,000+ open-source PMC papers. Define 4 forgery strategies based on academic fraud scenarios, generate corresponding forged images using 25 generative models, and overlay expert double-review, automated quality assessment, and quantitative forensic metrics.

**Core Idea**: Treat academic image forensics as a "hierarchical evaluation + cross-modal orthogonal evaluation," and introduce a Normalized Forensic Index to incorporate multi-task consistency into the scoring.

## Method

### Overall Architecture
The construction of AEGIS involves 3 phases, and evaluation spans 4 dimensions: (1) **Paper Parsing**: Fitz is used to extract images and captions from 4,362 PMC papers; dots.ocr verifies caption-figure correspondence; YOLOv7 segments figures into minimal panels; experts manually filter non-academic or low-resolution content to obtain 8,000+ labeled panels. (2) **Forgery Simulation**: Covers 25 generative models (Flux, Midjourney, DALL-E, GPT-Image-1, Janus-Pro, etc.) using four strategies: Text-Constrained Fabrication (TCF), Image-Inference Fabrication (IIF), Targeted Region Replacement (TRR), and Targeted Region Editing (TRE). (3) **Quality Control**: 5 experts perform two layers of local and global review, discarding 29% of forged images. This results in 8,210 high-quality forged samples and 1,795 real images, totaling 20k forensic questions. The evaluation side designs 4 progressive tasks: FSD (forgery scope discrimination: Real/Entire/Partial), TAR (text artifact recognition), MC (multi-classification of insertion/deletion/editing in red-box areas), and TP (localization via bbox or pixel-level mask).

### Key Designs

1.  **Academic-Exclusive Forgery Strategy Layered Simulation**:
    - **Function**: Reproduce real academic fraud scenarios (typical patterns observed in Retraction Watch) in a controlled environment to generate labelable and quantifiable forgery data.
    - **Mechanism**: The 4 strategies cover "global $\rightarrow$ local" and "text-driven $\rightarrow$ reference-driven" scenarios. TCF lets GPT-4o mini rewrite real captions into prompts for text-to-image models to fabricate images from scratch (3,121 samples). IIF uses real images as references for generative models to redraw visually consistent images (2,274 samples). TRR uses SAM to automatically generate masks for inpainting models to redraw local regions (1,650 samples). TRE performs insertion, deletion, or rewriting via masks or text instructions (1,165 samples). SAM-based masks ensure the precision of pixel-level ground truth.
    - **Design Motivation**: Real retraction cases are scarce (only 2-3 examples with traceable ancestry), and they lack structured labels for systematic evaluation. Using synthesis as a proxy for reality necessitates making forgery types align with actual misconduct while controlling granularity—the purpose of the 4 strategies and 25 models.

2.  **Four-Level Progressive Forensic Tasks & NFI**:
    - **Function**: Decouple forensic capability into 4 complementary tasks and measure "balanced capability" via normalized metrics rather than single peaks.
    - **Mechanism**: (a) FSD uses 3-class classification + "Not Sure" to resist forced guessing; (b) TAR focuses on semantic and glyph consistency in text regions; (c) MC provides a red box and requires the model to infer "insertion/deletion/modification" using the caption, forcing joint structural and contextual reasoning; (d) TP uses an adaptive granularity protocol: MLLMs use bbox + CLA/OLR, while experts use mask + IoU/F1. The composite index is defined as $\mathrm{NFI}_i=100\cdot \mathrm{HM}_i\cdot(1-\mathrm{OLR}_i)^\gamma$, where $\mathrm{HM}$ is the harmonic mean of scores from the four tasks, and $\gamma=0.5$ penalizes over-localization (i.e., attempting to game recall by outputting excessive bboxes).
    - **Design Motivation**: Accuracy alone cannot expose structural bias—a model might be extremely strong in TAR but very weak in TP. The HM ensures a high NFI only if a model performs well across all four tasks, while the OLR penalty prevents MLLMs from using "100-box" tricks.

3.  **Large-Scale Cross-Family Baselines & Robustness Perturbations**:
    - **Function**: Expose the generation-forensics gap and measure the vulnerability of forensic paradigms to common post-processing.
    - **Mechanism**: Evaluates 14 closed-source MLLMs (GPT-4.1/5.1/o4-mini, Gemini 2.5/3, Claude Sonnet 4.5, Doubao, Qwen-VL), 11 open-source MLLMs (LLaVA-NeXT, Gemma 3 27B, Qwen2.5-VL-72B, Llama 4 Maverick), 1 unified multimodal model (Janus-Pro-7B), and 9 expert models. Three types of post-processing perturbations are added: Gaussian blur ($r=5$), JPEG compression ($q=50$), and bilinear $0.5\times$ scaling.
    - **Design Motivation**: Revealing the complementarity—expert models are pixel-sharp but fragilly robust, while MLLMs are reasoning-strong but low-level sensitive—suggests that future forensic AIs must coordinate experts as sensors and MLLMs as cognitive agents.

### Loss & Training
AEGIS is an evaluation benchmark rather than a training framework, so there is no loss function. Evaluation is performed on high-resolution PNGs to avoid JPEG interference. Prompts only include task definitions. Closed-source models are accessed via OpenRouter API, while LLaVA and Doubao are called locally or via Volcengine on 8×A40 (48GB) nodes.

## Key Experimental Results

### Main Results
Core results of 34 models on AEGIS (selection):

| Model | FSD ACC | TAR ACC | MC ACC | TP CLA | NFI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Human | 44.20 | 76.14 | 68.01 | – | – |
| GPT-5.1 | 50.99 | 76.43 | 60.07 | 46.87 | **48.80** |
| Gemini 3 Pro | 64.37 | 84.74 | 48.54 | 39.14 | 45.79 |
| GPT-4.1 | 66.34 | 83.57 | 44.55 | 25.93 | 43.31 |
| Claude Sonnet 4.5| 25.36 | 58.76 | 44.80 | 27.41 | 26.83 |
| Qwen3-VL-Plus | 38.77 | 79.25 | 59.28 | 5.76 | 16.53 |
| AIDE (Expert) | 79.54 | – | – | – | – |
| DRCT (Expert) | 55.05 | – | – | – | – |
| FakeShield (Exp+MLLM)| 59.72 | – | – | IoU 30.09 | – |

Key Observations: MLLMs can reach 84.74% in TAR (Gemini 3), but the strongest expert only achieves 30.09% in TP pixel IoU. Expert models have high FSD binary classification ACC (79.54% for AIDE) but show almost no semantic reasoning capability.

### Ablation Study
The impact of Few-Shot / CoT prompts on different tasks (vs. Default):

| Prompt Strategy | FSD | TAR | MC | TP |
| :--- | :--- | :--- | :--- | :--- |
| Few-Shot | +5 pp (pattern matching) | +3 pp | −10 pp (hurts reasoning) | −2 pp |
| CoT (GPT-5.1) | +4.38 pp | +3.33 pp | Significant drop | +4.25 pp (CLA) |
| Default | baseline | baseline | baseline | baseline |

Post-processing robustness: Under Gaussian blur / JPEG compression / $0.5\times$ scaling, expert model scores drop by 10-20 pp, while MLLM scores drop by only 2-5 pp.

### Key Findings
- **Generation-Forensics Asymmetry**: 11 out of 25 generative models pull the average forensic accuracy below 50%, and 4 even below 30% (e.g., Nano Banana Pro). Only a few models like Gemini 3 Pro and GPT-4.1 show resistance across all generators.
- **Visual Density Bias**: Model performance is stable on structured images (Chart/Diagram) but plummets on texture-dense images (Stained Micrograph/Medical Imaging), indicating a heavy reliance on geometric regularity.
- **Orthogonality of Experts vs. MLLMs**: Expert models usually have a Real-F1 lower than Forgery-F1 (strong tendency to over-judge), while MLLMs show the opposite. The robustness of MLLMs vs. the sensitivity of experts provides hard evidence for a "sensor + cognitive agent" hybrid system.

## Highlights & Insights
- Transitioning evaluation from single binary classification to "hierarchical + multi-dimensional" is a methodological upgrade in benchmark design that can be migrated to other tasks requiring "overall authenticity $\rightarrow$ fine-grained localization $\rightarrow$ causal attribution."
- The design of NFI (Harmonic Mean + OLR penalty) is clever: HM forces **multi-task balance**, while the OLR penalty eliminates the "random box" shortcut for gaming metrics.
- The three-step QC—synthetic data + expert audit + IS/FID/CLIP automated evaluation—is highly valuable for simulating academic forgery.

## Limitations & Future Work
- With only 2-3 real retraction cases available, the benchmark remains synthesis-dominated; whether this fully aligns with real-world fraud distributions remains to be verified over time.
- Some SOTA forensic methods (e.g., AIGI-Holmes) do not have open-source weights, limiting baseline coverage.
- The TP task uses different granularities for MLLMs and experts (bbox vs. mask), requiring a more unified protocol for direct cross-paradigm comparison.

## Related Work & Insights
- **vs. DFBench (ACM MM 2025)**: DFBench also covers 4 forgery types but stays at detection ACC; AEGIS adds TAR/MC/TP dimensions for reasoning and localization within the academic domain.
- **vs. FakeShield/SIDA (Pixel-level hybrid models)**: FakeShield’s IoU on AEGIS is only 30.09% (despite being >70% in general domains), showing that pixel-level localization in academia is far from solved.
- **vs. AIGuard (ACL Findings 2025)**: AIGuard focuses on global detection for e-commerce; AEGIS specifically evaluates hybrid forgery strategies and multi-dimensional forensics in the academic domain.

## Rating
- Novelty: ⭐⭐⭐⭐ First forensic benchmark for the academic domain with broad forgery strategy coverage.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 34 models × 4 tasks + robustness + Few-Shot/CoT + error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and high information density; some figures could use more independent captions.
- Value: ⭐⭐⭐⭐⭐ Academic integrity is a high-stake real-world need; validated with real retraction cases, showing direct potential for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts](c-red_a_comprehensive_chinese_benchmark_for_ai-generated_text_detection_derived_.md)
- [\[AAAI 2026\] BAID: A Benchmark for Bias Assessment of AI Detectors](../../AAAI2026/aigc_detection/baid_a_benchmark_for_bias_assessment_of_ai_detectors.md)
- [\[ACL 2026\] MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)

</div>

<!-- RELATED:END -->
