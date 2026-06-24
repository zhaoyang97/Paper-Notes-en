---
title: >-
  [Paper Note] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans
description: >-
  [NeurIPS 2025][Image Generation][Multi-human image generation] This paper introduces MultiHuman-Testbench, the first systematic benchmark for evaluating multi-human image generation. It comprises 1,800 test samples paired with 5,550 face images, a suite of multi-dimensional evaluation metrics including Hungarian-matching-based identity similarity, and proposes Regional Isolation and Implicit Region Assignment techniques to enhance existing methods without additional training.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Multi-human image generation"
  - "identity preservation"
  - "benchmark"
  - "diffusion models"
  - "regional isolation"
date: 2026-05-08
content_hash: db271f513de468ae
---

# MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans

**Conference**: NeurIPS 2025
**arXiv**: [2506.20879](https://arxiv.org/abs/2506.20879)  
**Code**: [GitHub](https://github.com/Qualcomm-AI-research/MultiHuman-Testbench)  
**Area**: Image Segmentation
**Keywords**: Multi-human image generation, identity preservation, benchmark, diffusion models, regional isolation

## TL;DR

This paper introduces MultiHuman-Testbench, the first systematic benchmark for evaluating multi-human image generation. It comprises 1,800 test samples paired with 5,550 face images, a suite of multi-dimensional evaluation metrics including Hungarian-matching-based identity similarity, and proposes Regional Isolation and Implicit Region Assignment techniques to enhance existing methods without additional training.

## Background & Motivation

**Background**: Current text-to-image diffusion models can generate high-quality images, but simultaneously generating multiple humans while preserving individual facial identities, executing specified actions, and maintaining coherent composition remains a significant challenge.

**Limitations of Prior Work**: Existing methods commonly suffer from identity blending, inaccurate person count, and difficulty in scene composition. More critically, no standardized benchmark or well-defined metrics exist specifically for evaluating multi-human generation quality.

**Key Challenge**: Existing benchmarks either focus on single-subject scenarios (e.g., ID preservation), general text-to-image quality, or multi-object composition, but none address the particular complexity of multi-human generation.

**Goal**: To establish a comprehensive benchmark for multi-human image generation and provide standardized evaluation tools.

**Key Insight**: The benchmark is constructed along two dimensions simultaneously—data (diverse face images, carefully designed prompts, and pose conditions) and metrics (count accuracy, identity similarity, prompt alignment, and action detection).

**Core Idea**: Construct a standardized benchmark spanning 4 task dimensions and 5 evaluation metrics, and propose Regional Isolation and Implicit Region Assignment techniques to improve existing methods.

## Method

### Overall Architecture

MultiHuman-Testbench consists of two components: (1) benchmark construction—covering image selection, prompt design, pose estimation, and metric definition; and (2) method improvement—proposing Unified Regional Isolation and Implicit Region Assignment as training-free enhancements to existing multi-human generation models.

### Key Designs

1. **Dataset Construction**:

    - **Face Images**: Approximately 520K images are drawn from FFHQ, SFHQ, and CelebaHQ, then filtered through a multi-stage pipeline (MLLM-based removal of unrecognizable faces → removal of multi-face images → Gemini Flash 2.0 annotation for age/ethnicity/gender), followed by stratified sampling to ensure diversity. The final set contains 5,550 face images with balanced distributions across age (16–35, 35–60, 60+), ethnicity (6 categories), and gender.
    - **Prompts**: Gemini Flash 2.0 generates 100 simple prompts (5 people performing the same action) and 25 complex prompts (different people performing different actions), yielding 125 unique prompts. Each prompt is paired with 3 random face samplings, forming 1,800 test samples in total.
    - **Pose Conditions**: Sourced from top-performing generation results and a Text-to-Pose model, manually curated, and used as regional priors for Task 2.

2. **Evaluation Metric Suite**:

    - **Count Accuracy** ($S_{\text{count}}$): Whether the detected face count matches the reference number, $S_{\text{count}} = \delta_{MN}$.
    - **Hungarian ID Similarity** ($S_{\text{id}}$): ArcFace embeddings compute cosine similarity between reference and generated faces; Hungarian algorithm finds the optimal assignment, then averages:
    $$S_{\text{id}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} X_{ij} s_{ij}$$
    - **HPSv2** ($S_{\text{hps}}$): Human preference score for text–image alignment.
    - **MLLM Action QA**: MLLM-based question answering to assess correctness of simple (Action-S) and complex (Action-C) actions.
    - **Unified Metric**: $S_U = (S_{\text{id}} \times (S_{\text{align}})^2)^{1/3}$, where $S_{\text{align}} = (S_{\text{hps}} + S_{\text{act}} + S_{\text{count}})/3$.

3. **Unified Regional Isolation**: Targeting unified multimodal architectures such as OmniGen, the self-attention mask is modified so that tokens from each reference image $I_k$ attend only to the corresponding latent region $\mathcal{R}_k$, preventing information leakage across different identities. The attention constraint for image tokens is:
    $$A_{\text{iso}, ij} = 1 \quad \text{if } i \in \mathcal{D}_{\text{img}} \text{ and } (j \notin \mathcal{D}_{\text{latent}} \text{ or } j \in \mathcal{R}_k)$$

4. **Implicit Region Assignment**: Eliminates the need for user-specified regional priors. For MH-OmniGen, the self-attention maps of the backbone transformer are probed at intermediate timesteps to obtain region overlap information; combined with segmentation masks from the intermediate latent space, Hungarian matching assigns reference images to corresponding regions. For MH-IR-Diffusion, SAM2 segments proposed face regions in the generated output, and ArcFace similarity with Hungarian matching completes the assignment.

### Loss & Training

The proposed Regional Isolation and Implicit Region Assignment methods are **training-free, plug-and-play** techniques. No additional training is required; only the attention masks and region assignment strategy at inference time are modified.

## Key Experimental Results

### Main Results

**Task 1: Reference-based Multi-Human Generation (In-the-Wild):**

| Model | Count↑ | Multi-ID↑ | HPS↑ | Action-S↑ | Action-C↑ | Unified↑ |
|------|--------|-----------|------|-----------|-----------|----------|
| GPT-Image-1 | **87.9** | 28.8 | **30.3** | **97.0** | **91.1** | 54.3 |
| LoRA (5 views) | 52.6 | 22.0 | 25.9 | 73.0 | 72.9 | 41.0 |
| UniPortrait | 58.5 | 44.2 | 25.9 | 76.2 | 67.2 | 51.7 |
| OmniGen | 60.5 | 49.4 | 26.2 | 87.5 | 71.3 | 59.2 |
| **MH-OmniGen** | 60.3 | **54.5** | 26.3 | 91.6 | 72.9 | **61.6** |

MH-OmniGen improves over OmniGen by 5.1 points on Multi-ID and 4.1 points on Action-S, achieving the best Unified score (61.6). GPT-Image-1 ranks highest on count accuracy and action scores but performs worst on ID preservation (28.8 vs. 54.5).

### Ablation Study

**Task 3: Reference-Free Identity-Consistent Multi-Human Generation:**

| Model | Count↑ | Multi-ID↑ | HPS↑ | Action-S↑ |
|------|--------|-----------|------|-----------|
| ConsiStory | 44.6 | 16.2 | 28.0 | 84.1 |
| DreamStory | 45.0 | 19.7 | 28.2 | 84.8 |
| IR-Diffusion | 62.4 | 27.6 | 29.4 | 86.3 |
| **MH-IR-Diffusion** | **62.6** | **33.3** | 29.2 | 85.9 |

MH-IR-Diffusion achieves the best performance on both Count and Multi-ID, validating the effectiveness of Regional Isolation combined with Hungarian matching.

### Key Findings

- **No open-source method achieves satisfactory quality in in-the-wild multi-human generation**: Even the best-performing method (MH-OmniGen, Unified: 61.6) still leaves substantial room for visual improvement.
- **Backbone architecture is critical**: Methods built on stronger backbones (e.g., Flux, OmniGen-Phi3) substantially outperform those based on SD1.5/SDXL.
- **Count accuracy is a fundamental bottleneck**: Even the strongest T2I model, Flux, achieves only 46.4% count accuracy in 5-person scenes.
- **A fundamental trade-off exists between ID preservation and action accuracy**: GPT-Image-1 achieves the highest action scores but the worst ID preservation, revealing an inherent dilemma.
- **Regional priors significantly improve count accuracy**: Introducing pose or bounding-box conditions leads to large gains in the Count metric.
- **Implicit biases are observed**: Multiple models exhibit implicit biases across age, ethnicity, and gender dimensions.

## Highlights & Insights

- **First systematic benchmark**: Fills the gap in multi-human image generation evaluation; the 4-task, 5-metric framework is comprehensive and well-structured.
- **Elegant Hungarian ID similarity metric**: Optimal assignment with penalties for missing identities is more principled than simple average similarity.
- **Training-free improvement**: Regional Isolation and Implicit Region Assignment are plug-and-play at inference time, requiring no additional training, making them practically accessible.
- **Large-scale model evaluation**: Approximately 30 models are evaluated across 4 tasks, covering commercial (GPT-Image-1), fine-tuned (LoRA), and zero-shot settings, providing a thorough view of the current landscape.

## Limitations & Future Work

- All current methods still have significant room for visual quality improvement; no method consistently passes a "human eye test."
- The weighting scheme of the unified metric (quadratic weighting of the alignment term) is heuristic and may not generalize to all application scenarios.
- The reliability of MLLM Action QA depends on the capability of the underlying MLLM, which may introduce evaluation bias.
- Although face images in the benchmark are sampled with balanced distributions, the race/age categorization itself is based on automatic MLLM annotation, which may contain labeling errors.
- The proposed improvements (MH-OmniGen) are validated only on OmniGen and IR-Diffusion; applicability to other architectures remains to be verified.

## Related Work & Insights

- **Subject-driven generation**: Auxiliary modules such as IP-Adapter and ControlNet enable single-subject ID preservation but have limited effectiveness in multi-person scenarios.
- **Unified multimodal models**: Models such as OmniGen, Show-O, and GPT-Image-1 unify text and visual processing, demonstrating the greatest potential for multi-human generation.
- **Regional isolation**: Methods such as InstantFamily and Regional Prompting leverage explicit regional priors to separate multiple persons, but at the cost of usability.
- **Insights**: Multi-human image generation remains a highly open challenge; the capability of the base model—particularly count accuracy—is the primary bottleneck.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic benchmark focused on multi-human generation; the Hungarian ID metric and regional isolation technique are genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation of approximately 30 models across 4 tasks; exceptional in data volume and coverage.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rigorous metric definitions, though the breadth of content makes the paper somewhat lengthy.
- Value: ⭐⭐⭐⭐ Meaningfully advances research in multi-human generation and clearly identifies the key bottlenecks of current approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](accuquant_simulating_multiple_denoising_steps_for_quantizing.md)
- [\[ICCV 2025\] CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation](../../ICCV2025/image_generation/compslider_compositional_slider_for_disentangled_multiple-attribute_image_genera.md)
- [\[CVPR 2025\] Pippo: High-Resolution Multi-View Humans from a Single Image](../../CVPR2025/image_generation/pippo_high-resolution_multi-view_humans_from_a_single_image.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[ICCV 2025\] CNS-Bench: Benchmarking Image Classifier Robustness Under Continuous Nuisance Shifts](../../ICCV2025/image_generation/cns-bench_benchmarking_image_classifier_robustness_under_continuous_nuisance_shi.md)

</div>

<!-- RELATED:END -->
