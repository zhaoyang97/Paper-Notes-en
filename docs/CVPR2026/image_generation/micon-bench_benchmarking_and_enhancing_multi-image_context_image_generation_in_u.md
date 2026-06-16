---
title: >-
  [Paper Note] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models
description: >-
  [CVPR 2026][Image Generation][benchmark] This paper introduces MICON-Bench, a multi-image context generation benchmark covering 6 tasks (1043 cases) paired with an MLLM-driven Evaluation-by-Checkpoint automated framework. Simultaneously, it proposes DAR (Dynamic Attention Rebalancing), a training-free mechanism that enhances multi-image consistency and genera
tags:
  - CVPR 2026
  - Image Generation
  - benchmark
date: 2026-05-08
content_hash: 10bd17db8c1cc6d7
---
# MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models

**Conference**: CVPR 2026  
**arXiv**: [2602.19497](https://arxiv.org/abs/2602.19497)  
**Code**: [https://github.com/Angusliuuu/MICON-Bench](https://github.com/Angusliuuu/MICON-Bench)  
**Area**: Image Generation / Multimodal Evaluation  
**Keywords**: Multi-image Context Generation, Unified Multimodal Models, benchmark, Dynamic Attention Rebalancing, Checkpoint Evaluation

## TL;DR
This paper introduces MICON-Bench, a multi-image context generation benchmark covering 6 tasks (1043 cases) paired with an MLLM-driven Evaluation-by-Checkpoint automated framework. Simultaneously, it proposes DAR (Dynamic Attention Rebalancing), a training-free mechanism that enhances multi-image consistency and generation quality in UMMs by dynamically adjusting inference-time attention weights.

## Background & Motivation

**Background**: UMMs have demonstrated the capability to process multi-image inputs and generate contextually consistent visual outputs, represented by models such as Nano-Banana, GPT-Image, BAGEL, and OmniGen2. However, systematic evaluation for multi-image context generation remains lacking.

**Limitations of Prior Work**: Existing benchmarks (GenEval, T2ICompBench, ImgEdit-Bench) primarily evaluate text-to-image or single-image editing, failing to address cross-image consistency and complex visual relationship reasoning. While OmniContext includes multiple images, it is restricted to simple object compositions.

**Key Challenge**: UMMs tend to **distribute attention uniformly** across all regions of all reference images—including irrelevant areas—during multi-image input processing, which leads to hallucinations and inconsistencies.

**Core Idea**: (a) A set of 6 standardized tasks with a verifiable checkpoint evaluation system; (b) An attention rebalancing mechanism to adjust focus during inference.

## Method

### Overall Architecture

The study consists of two components: a comprehensive benchmark and a plug-and-play inference mechanism. To address the lack of systematic evaluation for multi-image context generation (generating consistent new images from multiple reference images), the authors built MICON-Bench, featuring 6 task categories and 1043 cases. This is paired with an MLLM-driven "Evaluation-by-Checkpoint" framework that decomposes each case into verifiable fine-grained scoring points. To solve the issue where UMMs distribute attention too broadly across reference regions, DAR (Dynamic Attention Rebalancing) is proposed to dynamically re-weight attention during inference without requiring additional training.

### Key Designs

**1. MICON-Bench: 6 Tasks Ranging from Simple Composition to Causal Reasoning**

Unlike existing benchmarks focused on single-image editing, MICON-Bench decomposes multi-image context generation into 5 composition tasks and 1 complex reasoning task with increasing difficulty:

| Task | Description | Case Count | Ref Images |
|------|-------------|------------|------------|
| Object Composition | Single subject + background combination | 200 | 2-3 |
| Spatial Composition | Spatial relationship constraints for multiple objects | 200 | 2-3 |
| Attribute Disentanglement | Decoupled recombination of subject/style/background | 100 | 3 |
| Component Transfer | Transferring parts/accessories across images | 240 | 2-3 |
| FG/BG Composition | Foreground + background fusion | 200 | 2 |
| Story Generation | Causal reasoning to continue a story | 103 | 2-3 |
| **Total** | | **1043** | **2518 images** |

**2. Evaluation-by-Checkpoint: Decomposing Quality into Pass/Fail Points**

To avoid coarse image-level scoring, this framework defines a set of verifiable checkpoints for each case, covering seven dimensions: instruction following, identity consistency, structure, cross-reference consistency, causality, text anchoring, and overall usability. An MLLM (Qwen3-VL-32B) acts as the verifier to judge each point as pass/fail, with the final score being the mean pass rate. The Story task uses an additional predefined answer set for reasoning evaluation.

**3. Dynamic Attention Rebalancing (DAR): Redirecting Attention to Key Regions**

DAR addresses the diagnostic finding that UMMs pay indiscriminate attention to irrelevant areas. It performs a high-efficiency attention analysis by sampling $m \ll L_q$ query tokens (default $m=64$) and calculating their attention towards reference image key tokens. The total score for each key $r_k = \sum_{i=1}^{m}\sum_{h=1}^{H} \tilde{A}_{i,h,k}$ is normalized via min-max to obtain $\hat{r}_k$. Keys are re-weighted based on dual thresholds: crucial keys ($\hat{r}_k \geq \tau_{high}$) are scaled by $w_k = 1+\gamma$, irrelevant keys ($\hat{r}_k \leq \tau_{low}$) are suppressed to $w_k = 1-\gamma$, and others remain unchanged. Attention is then recalculated as $A = \text{softmax}\left(\frac{Q(w \odot K_{ref})^\top}{\sqrt{d}}\right)$ (default $\gamma=0.15, \tau_{high}=0.7, \tau_{low}=0.3$).

## Key Experimental Results

### Main Results: MICON-Bench Task Scores

| Model | Object | Spatial | Attribute | Component | FG/BG | Story | Avg↑ |
|------|--------|---------|-----------|-----------|-------|-------|------|
| Nano-Banana | 95.60 | 93.79 | 92.13 | 84.23 | 83.13 | 82.84 | 89.25 |
| GPT-Image | 96.45 | 94.41 | 93.39 | 87.69 | 85.99 | 91.51 | 90.15 |
| UNO | 58.40 | 66.68 | 65.28 | 28.84 | 20.96 | 39.08 | 44.76 |
| DreamOmni2 | 88.24 | 84.76 | 85.28 | 59.64 | 76.16 | 59.58 | 75.56 |
| BAGEL | 87.64 | 89.96 | 89.84 | 52.40 | 64.64 | 65.09 | 73.55 |
| **BAGEL + DAR** | **88.04** | **91.88** | **90.76** | **56.06** | **71.24** | **66.34** | **76.31** |
| OmniGen2 | 89.52 | 80.32 | 81.64 | 44.76 | 57.96 | 60.96 | 67.83 |
| **OmniGen2 + DAR** | **89.84** | **81.00** | **82.12** | **48.72** | **59.28** | **60.73** | **69.21** |

### OmniContext Benchmark

| Method | SINGLE Char/Obj | MULTIPLE Char/Obj | SCENE Char/Obj | Avg↑ |
|------|-----------------|-------------------|----------------|------|
| OmniGen2 | 8.18/7.33 | 6.56/7.99 | 6.87/7.90 | 7.53 |
| **OmniGen2+DAR** | **8.30/8.19** | **6.64/8.42** | **7.06/7.97** | **7.77** |
| BAGEL | 5.71/6.22 | 3.03/6.90 | 4.24/5.16 | 5.54 |
| **BAGEL+DAR** | **6.26/6.08** | **4.14/7.18** | **4.78/4.84** | **5.80** |

### XVerseBench Benchmark

| Method | Single-Subject Avg↑ | Multi-Subject Avg↑ | Overall↑ |
|------|--------------------|--------------------|----------|
| OmniGen2 | 52.53 | 49.76 | 51.14 |
| **OmniGen2+DAR** | **53.24** | **50.23** | **51.73** |
| BAGEL | 47.91 | 42.62 | 45.26 |
| **BAGEL+DAR** | **48.54** | **43.91** | **46.23** |

### Key Findings
- MICON-Bench effectively differentiates models: GPT-Image is the strongest (90.15), while the diffusion-based UNO is the weakest (44.76).
- DAR provides the most significant boost to BAGEL: Avg +2.76 (73.55→76.31), with a +6.60 gain in FG/BG.
- DAR shows strong generalization with consistent improvements across three different benchmarks (MICON-Bench, OmniContext, XVerseBench).
- Component Transfer and FG/BG are the most challenging tasks, with even top-tier models scored between 84-88.
- A significant gap remains between open-source and closed-source models (BAGEL 73.55 vs GPT-Image 90.15).

## Highlights & Insights
- **First Systematic Multi-Image Context Generation Benchmark**: 6 tasks cover the full difficulty spectrum from simple composition to causal reasoning.
- **Evaluation-by-Checkpoint Paradigm**: Fine-grained, quantifiable, and scalable, providing a more objective measure than traditional image-level metrics.
- **Concise and Effective DAR Mechanism**: Using only 64 sampled query tokens and dual-threshold re-weighting significantly improves performance without training costs.
- Identifies attention allocation blind spots in UMM multi-image reasoning, providing direction for future model design.

## Limitations & Future Work
- DAR thresholds ($\tau_{high}, \tau_{low}$) and modulation factor ($\gamma$) require manual setting; adaptive schemes have not been explored.
- The sample size for the Story Generation task is relatively small (103 cases).
- Benchmark data generated by Qwen-Image + GPT-4o may introduce generative model bias.
- Higher-order requirements such as 3D consistency and temporal continuity were not evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-image context benchmark + plug-and-play DAR.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7+ models + 3 benchmarks + multiple metrics + comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and refined evaluation workflows.
- Value: ⭐⭐⭐⭐ Benchmark drives evaluation standardization; DAR is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SCIEval: Evaluating and Benchmarking the Faithfulness of Scientific Image Generation and Interpretation with Large Multimodal Models](scieval_evaluating_and_benchmarking_the_faithfulness_of_scientific_image_generat.md)
- [\[CVPR 2026\] Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models](omni_iie_bench_benchmarking_the_practical_capabilities_of_image_editing_models.md)
- [\[ACL 2026\] Multimodal Large Language Models for Multi-Subject In-Context Image Generation](../../ACL2026/image_generation/multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)
- [\[CVPR 2026\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dualconditioned_di.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)

</div>

<!-- RELATED:END -->
