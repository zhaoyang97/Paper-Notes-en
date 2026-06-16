---
title: >-
  [Paper Note] PAI-Bench: A Comprehensive Benchmark for Physical AI
description: >-
  [CVPR 2026][Multimodal VLM][Physical AI] PAI-Bench decomposes the "prediction and perception capabilities required for physical AI" into three tracks: video generation, conditional video generation, and video understanding. Using 2808 real-world samples and task-aligned metrics, it evaluates 15 video generation models and 16 multimodal large models. The concl
tags:
  - CVPR 2026
  - Multimodal VLM
  - Physical AI
  - World Models
date: 2026-05-08
content_hash: cc3b06e3fca63ebb
---
# PAI-Bench: A Comprehensive Benchmark for Physical AI

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_PAI-Bench_A_Comprehensive_Benchmark_For_Physical_AI_CVPR_2026_paper.html)  
**Code**: https://github.com/SHI-Labs/physical-ai-bench  
**Area**: Multimodal VLM  
**Keywords**: Physical AI, Video Generation Evaluation, Video Understanding, World Models, Controllable Generation

## TL;DR
PAI-Bench decomposes the "prediction and perception capabilities required for physical AI" into three tracks: video generation, conditional video generation, and video understanding. Using 2808 real-world samples and task-aligned metrics, it evaluates 15 video generation models and 16 multimodal large models. The conclusion is: current video generation models produce realistic visuals but fail to uphold physical laws, and the physical understanding of MLLMs lags far behind humans.

## Background & Motivation
**Background**: "Physical AI" expects models to perceive the real world and predict its dynamic evolution to support embodied applications like robotics and autonomous driving. This capability can be split into two halves: **Perception** (understanding physical events in videos), mainly relying on Multimodal Large Language Models (MLLM), and **Prediction** (forecasting the next frame/step), mainly relying on Video Generation Models (VGM)—the latter are implicitly trained to "predict future frames" and are thus expected to be world models learning physical laws.

**Limitations of Prior Work**: Evaluations on both sides fail to address the core of "Physical AI." Mainstream MLLM benchmarks test abstract/general abilities like OCR and math, leaving their performance in specialized physical scenarios unmeasured. Mainstream VGM benchmarks (VBench, EvalCrafter, etc.) evaluate aesthetic quality and temporal consistency but rarely check if generated videos conform to physical common sense. Worse, these benchmarks are fragmented—focusing only on prediction or perception—and lack systematic evaluation of "conditional controllable generation" fidelity to control signals.

**Key Challenge**: Physical AI requires integrated prediction and perception grounded in real physical scenarios, whereas existing evaluations are neither unified, physical, nor realistic (often using synthetic/toy scenes). High visual fidelity $\neq$ high physical plausibility; these two are conflated by current metrics.

**Goal**: Establish a unified, realistic, and physically aligned benchmark covering video generation, conditional video generation, and video understanding, equipped with metrics reflecting "physical rationality" rather than just image quality.

**Key Insight**: All evaluations are anchored on "real-world captured videos + physically meaningful tasks" (e.g., dashcams, robot manipulation, first-person view), covering subdomains like autonomous driving, robotics, industry, human activities, and physical common sense. "Visual quality" and "physical rationality" are separated into two independent scores.

**Core Idea**: A systematic "health check" for Physical AI's prediction and perception capabilities using a three-track (Generation / Conditional Generation / Understanding) + dual-score (Quality Score for visuals, Domain Score for physical rationality) design.

## Method
PAI-Bench is not a model but an evaluation protocol + dataset. It decomposes physical AI capability testing into three complementary tracks, following the construction principle of "real video + physical tasks," totaling 2808 high-quality samples.

### Overall Architecture
The three tracks movement correspond to different dimensions of Physical AI:

- **PAI-Bench-G (Video Generation)**: Evaluates "Prediction." Given text/first frame prompts, VGMs generate videos, measured by both Quality Score and Domain Score.
- **PAI-Bench-C (Conditional Video Generation)**: Further evaluates controllability in "Prediction." Given control signals (sketch, edge, depth, segmentation), it assesses if conditional VGMs are faithful to signals, their visual quality, and diversity under the same conditions.
- **PAI-Bench-U (Video Understanding)**: Evaluates "Perception." Given real videos + multiple-choice questions, it assesses MLLMs' physical common sense and embodied reasoning.

Samples for all tracks are collected from the real world (public datasets + web), spanning autonomous driving, robotics, industry, humans, physical common sense, and first-person views. The benchmark provides a "status map" by evaluating 15 VGMs, 4 conditional VGMs (5 configurations), and 16 MLLMs.

### Key Designs

**1. Unified Three-Track Design: Testing Prediction and Perception on One Scale**

Physical AI requires the integrated ability to "understand the present + predict the future." Old benchmarks either only evaluate VGM quality or MLLM Q&A. PAI-Bench unites them: G and C focus on "Prediction" (VGM as a world model forecasting future frames), while U focuses on "Perception" (MLLM understanding physical events). All tracks share real-world data and physically meaningful tasks. PAI-Bench is the first to cover all eight dimensions: video generation, conditional generation, video understanding, and all physical subdomains.

**2. Dual Scores (Quality & Domain): Decoupling "Realistic Drawing" from "Physical Compliance"**

A major trap in VGM evaluation is treating "visual realism" as "physical correctness." PAI-Bench-G separates these. **Quality Score** follows VBench++'s 8 metrics (Subject Consistency SC, Background Consistency BC, Motion Smoothness MS, Aesthetic Quality AQ, Imaging Quality IQ, Overall Consistency OC, I2V Subject IS, I2V Background IB). **Domain Score** addresses physical rationality: high-fidelity captions are generated via Qwen2.5-VL-72B and manually corrected; then QA pairs are generated based on physical ontologies (5636 pairs across 6 subdomains). Qwen3-VL-235B acts as the judge to evaluate generated videos against these QA pairs. The Domain Score is the judge's accuracy, quantifying compliance with physical and semantic constraints. This design achieved a Pearson correlation of $r=0.918$ with human preferences.

**3. Conditional Track PAI-Bench-C: Systematically Evaluating Control Signal Fidelity**

As VGMs increasingly use depth maps, edges, and segmentation for guided generation, "controllability" becomes critical. PAI-Bench-C defines three criteria: **Fidelity** uses projection-comparison metrics—projecting generated videos back to the control modality space (via Canny, Video-Depth-Anything, etc.) and calculating similarity to ground-truth (Blur SSIM↑, Edge F1↑, Depth si-RMSE↓, Mask mIoU↑); **Quality** uses DOVER; **Diversity** uses LPIPS. Data includes 600 videos from AgiBot, OpenDV, and Ego-Exo-4D.

**4. PAI-Bench-U Dual-Ontology + Debiasing: Forcing Real Physical Understanding**

Video understanding benchmarks often suffer from "answering without watching," where models guess based on language priors or single-frame bias. PAI-Bench-U addresses this via:
- **Capability Ontology**: (1) Physical Common Sense (Space, Time, Physical World/Violations); (2) Embodied Reasoning (Predicting Action Effects, Adherence to Physical Constraints).
- **Debiasing**: Diagnostic tests with variable input frames. At 0 frames (text-only), models drop to random guess levels, proving the questions cannot be solved by language priors alone. A significant gap between 1 frame and 32 frames ensures dependence on temporal context.

### Dataset Construction
Data construction involves two stages: "MLLM initial tagging + manual refinement." Track G includes 1044 video-prompt pairs + 5636 QA; Track C uses 600 videos with extracted control signals and novel captions; Track U includes 604 common sense QA and 610 embodied reasoning QA. Total: 2808 samples.

## Key Experimental Results

### Main Results
**PAI-Bench-G (15 VGMs; Quality Ref: 78.0, Domain Ref: 89.8)**:

| Model | Overall | Domain Score | Quality Score |
|------|---------|--------------|---------------|
| Source Videos (Real) | 83.9 | 89.8 | 78.0 |
| Veo3 (Closed-source) | 82.2 | 86.8 | 77.6 |
| Wan2.2-I2V-A14B (SOTA Open) | 82.3 | 87.1 | 77.5 |
| Cosmos-Predict2.5-2B | 81.4 | 84.9 | 78.0 |
| DynamiCrafter (Weak Baseline) | 68.3 | 63.0 | 73.7 |

Key Contrast: Leading VGMs close the gap on Quality Score (~78) relative to real videos, but Domain Scores remain significantly lower than the real video's 89.8—visuals are realistic, but physical rationality lags.

**PAI-Bench-U (16 MLLMs)**:

| Model | Overall | Common Sense Avg. | Embodied Avg. |
|------|---------|-------------------|---------------|
| Human | 93.2 | 93.6 | 95.5 |
| Qwen3-VL-235B-A22B (Best) | 64.7 | 64.9 | 64.4 |
| GPT-5 | 61.8 | 63.9 | 59.7 |
| Qwen2.5-VL-72B | 60.8 | 58.6 | 63.0 |
| Random Guess | 37.0 | 38.9 | 35.2 |

All models (max 64.7) face a ~30-point gap compared to humans (93.2). Closed-source models do not necessarily lead.

### Ablation Study
**PAI-Bench-C: Control Signal Configuration (e.g., Cosmos-Transfer2.5-2B)**

| Control Signal | Edge F1 ↑ | Mask mIoU ↑ | Quality ↑ | Diversity ↑ |
|----------|-----------|-------------|-----------|-------------|
| Blur (Single) | 0.26 | 0.75 | 8.77 | 0.18 |
| Edge (Single) | 0.39 | 0.74 | 8.05 | 0.36 |
| Seg (Single) | 0.13 | 0.71 | 7.87 | 0.44 |
| All (Fused) | 0.45 | 0.77 | 9.24 | 0.13 |

**Track U Debiasing Diagnosis (Accuracy %)**

| Config | Qwen3-VL-8B | GPT-5 | Explanation |
|------|-------------|-------|------|
| #frames=0 (Text-only) | 39.3 | 37.3 | Drops to random level → no language leakage |
| #frames=1 | 43.3 | 52.1 | Single frame insufficient |
| #frames=32 | 47.9 | 68.2 | Dependency on temporal context proven |

### Key Findings
- **Visual Quality $\neq$ Physics**: VGM Quality Scores are near real videos, but Domain Scores lag, indicating "physical compliance" is the current bottleneck for world models.
- **Multiple Signals surpass Single Signals**: In Track C, the "All" condition achieved the highest quality (9.24), suggesting it is better to fuse complementary control signals than use raw noisy video.
- **Segmentation signals are least faithful**: Mask mIoU is lowest for Seg control, likely due to poor temporal consistency in segmentors like SAM2 (missing objects), leading to noisy supervision.
- **MLLM physical understanding is failing**: The strongest model (64.7) is far from human (93.2), suggesting Physical AI is not yet an optimization focus for mainstream MLLMs.

## Highlights & Insights
- **The "Dual Score" design is highly valuable**: Decoupling visual quality from physical rationality prevents "false prosperity" where pretty pictures mask physical failure.
- **Domain Score = Task-aligned QA + MLLM-as-judge**: Translating unquantifiable physical rationality into accuracy against physics-constrained QA pairs makes evaluation quantifiable and explainable.
- **Frame-count ablation as a diagnostic**: Proof that "answers come from vision + time" is an excellent paradigm for validating if a video benchmark actually requires video understanding.
- **Projection-comparison fidelity metrics**: Provides an actionable quantitative protocol for assessing if controllable generation "follows instructions."

## Limitations & Future Work
- Positioned as a "status check," the paper does not propose methods to bridge the gaps it identifies.
- Domain Score depends on an MLLM judge; the judge's own physical understanding limits the evaluation's ceiling ⚠️.
- Fidelity metrics in Track C depend on external extractors (SAM2, Canny); errors in these tools may be conflated with model performance.
- The use of multiple-choice questions in Track U limits the evaluation of open-ended generation or explanation.

## Related Work & Insights
- **vs VBench / EvalCrafter**: PAI-Bench adds the Domain Score for physical rationality and expands to conditional generation and understanding.
- **vs PhyGenBench**: PAI-Bench uses 2808 real samples across all physical subdomains, whereas others use smaller or synthetic scales.
- **Insight**: When a field contains both "generation" and "perception" models, evaluating them on unified real-world data and task-aligned metrics reveals systemic gaps better than isolated benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified Physical AI benchmark covering generation, conditional generation, and understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across many models and detailed debiasing/human alignment analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and definitions.
- Value: ⭐⭐⭐⭐⭐ Provides a crucial metric for world models and embodied AI, highlighting the "visual vs. physical" gap.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] PhyCritic: Multimodal Critic Models for Physical AI](phycritic_multimodal_critic_models_for_physical_ai.md)
- [\[CVPR 2026\] QUANTIPHY: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language Models](quantiphy_a_quantitative_benchmark_evaluating_physical_reasoning_abilities_of_vi.md)
- [\[CVPR 2026\] IPR-1: Interactive Physical Reasoner](ipr-1_interactive_physical_reasoner.md)
- [\[CVPR 2026\] LifeEval: A Multimodal Benchmark for Assistive AI in Egocentric Daily Life Tasks](lifeeval_a_multimodal_benchmark_for_assistive_ai_in_egocentric_daily_life_tasks.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)

</div>

<!-- RELATED:END -->
