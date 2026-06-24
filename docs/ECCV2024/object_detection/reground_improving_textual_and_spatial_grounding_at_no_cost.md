---
title: >-
  [Paper Note] ReGround: Improving Textual and Spatial Grounding at No Cost
description: >-
  [ECCV 2024][Object Detection][textual grounding] By changing the sequential connection of Gated Self-Attention (GSA) and Cross-Attention (CA) in GLIGEN to a parallel connection (network rewiring), the trade-off between textual and spatial grounding is significantly alleviated without introducing new parameters, fine-tuning, or computational overhead.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "textual grounding"
  - "spatial grounding"
  - "network rewiring"
  - "GLIGEN"
  - "diffusion model"
date: 2026-05-08
content_hash: f5ea5a761fd616f6
---

# ReGround: Improving Textual and Spatial Grounding at No Cost

**Conference**: ECCV 2024  
**arXiv**: [2403.13589](https://arxiv.org/abs/2403.13589)  
**Code**: [https://re-ground.github.io](https://re-ground.github.io)  
**Area**: Vision-Language / Layout-Guided Image Generation  
**Keywords**: textual grounding, spatial grounding, network rewiring, GLIGEN, diffusion model

## TL;DR
By changing the sequential connection of Gated Self-Attention (GSA) and Cross-Attention (CA) in GLIGEN to a parallel connection (network rewiring), the trade-off between textual and spatial grounding is significantly alleviated without introducing new parameters, fine-tuning, or computational overhead.

## Background & Motivation

**Background**: Diffusion-model-driven text-to-image (T2I) generation has made significant progress. GLIGEN enables bounding-box-based spatial grounding by introducing a Gated Self-Attention module and has been adopted by numerous downstream tasks.

**Limitations of Prior Work**: While GLIGEN achieves accurate spatial grounding, it often ignores key descriptive information in the text prompt (e.g., "low poly illustration", "draped with a colorful blanket"), a phenomenon the authors refer to as **description omission**.

**Key Challenge**: GSA and CA are sequentially structured in GLIGEN, where the GSA output serves as the input to CA. Consequently, spatial grounding signals dominate the feature representation before reaching CA, suppressing the influence of textual conditions. Reducing the GSA activation duration (scheduled sampling) can improve textual grounding but sacrifices spatial grounding accuracy.

**Goal**: Eliminate the text-spatial grounding trade-off caused by the sequential architecture, ensuring that both grounding capabilities operate without mutual interference.

**Key Insight**: Analysis reveals that CA does not affect spatial grounding (objects remain in correct positions even after CA is removed), whereas the sequential placement of GSA interferes with textual grounding. Therefore, the two modules can be parallelized.

**Core Idea**: Parallelize GSA and CA instead of keeping them sequential—only network rewiring is needed during inference, requiring zero training.

## Method

### Overall Architecture
ReGround is based on the pre-trained GLIGEN model, modifying the connectivity of the attention modules in each U-Net layer during inference. The original pipeline of GLIGEN is: Residual Block → Self-Attention → **GSA → CA** (sequential). ReGround alters this to: Residual Block → Self-Attention → **GSA ∥ CA** (parallel), where the outputs of both modules are summed before flowing into the next layer.

### Key Designs

1. **Analysis of the Description Omission Problem**

    - Function: Systematically analyze the root cause of textual description omission in GLIGEN.
    - Mechanism: Through scheduled sampling experiments (adjusting $\gamma$ from $1.0$ to $0.0$), it is found that longer GSA activation leads to worse textual grounding; however, shortening GSA degrades spatial grounding. This represents an irreconcilable trade-off.
    - Design Motivation: Demonstrate that the issue lies in the architectural design rather than the parameters, providing a theoretical foundation for network rewiring.

2. **Experiment on the Impact of Cross-Attention on Spatial Grounding**

    - Function: Verify whether CA affects spatial grounding.
    - Mechanism: Remove all CA modules in GLIGEN and directly pass the GSA output to the next layer ($F \leftarrow F_{GSA}$). Results show that object silhouettes still accurately land inside the bounding boxes.
    - Design Motivation: Prove that CA does not rely on GSA outputs to perform spatial grounding, meaning the two can operate independently.

3. **Network Rewiring: From Sequential to Parallel**

    - Function: Core innovation—parallelizing GSA and CA during inference.
    - Mechanism: Original sequential formula of GLIGEN:
    $$F_{GSA} \leftarrow \text{GSA}(F_{SA}, \{g_i\}) + F_{SA}$$
    $$F \leftarrow \text{CA}(F_{GSA}, c) + F_{GSA}$$
      ReGround parallel formula:
    $$F \leftarrow \underbrace{\text{GSA}(F_{SA}, \{g_i\})}_{\text{spatial grounding}} + \underbrace{\text{CA}(F_{SA}, c)}_{\text{textual grounding}} + \underbrace{F_{SA}}_{\text{residual}}$$
    - Design Motivation: Under the parallel structure, the input to CA is restored from $F_{GSA}$ to $F_{SA}$, which is exactly the input that CA is supposed to receive in the original LDM. The input to GSA remains unchanged, so spatial grounding is unaffected. The two pathways work independently without mutual interference.

4. **Review of Gated Self-Attention**

    - Function: Explain the construction of grounding tokens in GLIGEN.
    - Mechanism: $g_i = \mathcal{G}(\mathcal{T}(p_i), \mathcal{F}(b_i))$, where $\mathcal{T}$ is the text encoder, $\mathcal{F}$ is the Fourier position encoder, and $\mathcal{G}$ is a shallow MLP. GSA performs joint self-attention between visual features $(f_1, ..., f_{N_l})$ and grounding tokens $(g_1, ..., g_M)$.
    - Design Motivation: Understanding the GSA mechanism is the foundation for analyzing its interaction with CA.

### Loss & Training
- **No Training Required**: The key advantage of ReGround is that it requires absolutely no training or fine-tuning. It only modifies the connection of attention modules during the inference phase of pre-trained GLIGEN.
- **Scheduled Sampling Compatibility**: It can be combined with GLIGEN's scheduled sampling strategy $\beta_t$ to further regulate performance.
- **Zero Extra Overhead**: No new parameters, no additional computation, and no increase in memory footprint.

## Key Experimental Results

### Main Results

| Dataset | Method | CLIP Score ↑ | YOLO Score ↑ |
|--------|------|-------------|-------------|
| MS-COCO-2014 | GLIGEN ($\gamma=1.0$) | 30.44 | 58.13 |
| MS-COCO-2014 | GLIGEN ($\gamma=0.1$) | 31.65 | 22.75 |
| MS-COCO-2014 | **ReGround ($\gamma=1.0$)** | **31.29** | **56.96** |
| MS-COCO-2017 | GLIGEN ($\gamma=1.0$) | 30.47 | 58.30 |
| MS-COCO-2017 | **ReGround ($\gamma=1.0$)** | **31.06** | **57.04** |
| NSR-1K-GPT Counting | GLIGEN ($\gamma=1.0$) | 32.46 | 65.36 |
| NSR-1K-GPT Counting | **ReGround ($\gamma=1.0$)** | **33.20** | **63.92** |

### Human Evaluation and Preference

| Evaluation Method | ReGround Preference Rate | GLIGEN Preference Rate |
|----------|---------------|--------------|
| User Study (92 users) | **70.05%** | 29.95% |
| PickScore (COCO-2017) | **55.66%** | 44.34% |
| PickScore (COCO-Drop) | **57.57%** | 42.43% |

### Key Findings
- At $\gamma=1.0$, ReGround achieves **70.25%** of the CLIP gain that GLIGEN gets when dropping $\gamma$ from $1.0$ to $0.1$, while the YOLO Score only decreases by **3.31%**.
- In the COCO-Drop scenario (removing bounding boxes of 50% of the categories), ReGround's CLIP advantage over GLIGEN expands to **1.57 times** the original.
- ReGround consistently outperforms GLIGEN on the FID metric, indicating an improvement in image quality as well.
- When replacing GLIGEN as the backbone for BoxDiff, it also achieves significant improvements in textual grounding.

## Highlights & Insights
- **Minimalist yet Profound Analysis**: The experiment of removing CA cleverly demonstrates that spatial grounding does not rely on CA, providing solid theoretical support for parallelization. This "subtraction-based" analytical approach is worth learning from.
- **A Paradigm of Zero-Cost Improvement**: Improvements requiring no training, no additional parameters, and no extra computation hold immense value in practical applications. This class of methods matches the philosophy of FreeU—uncovering "free" performance gains by analyzing the network's internal mechanisms.

## Limitations & Future Work
- Only validated on the GLIGEN architecture; applicability to other spatial grounding methods (such as ControlNet) has not been explored.
- Although parallelization does not affect spatial grounding accuracy, the parallel weights are currently fixed to equal weighting (1:1), and has not discussed whether better weighting schemes exist.
- For complex scenes that highly depend on both spatial grounding and texture/textual details, the improvement margin might be limited.

## Related Work & Insights
- **vs GLIGEN [Li et al., CVPR 2023]**: ReGround directly builds on GLIGEN, achieving significant improvements simply by modifying the network connectivity during inference.
- **vs BoxDiff [Kim et al., ICCV 2023]**: BoxDiff utilizes cross-attention maps for extra spatial guidance; ReGround can be combined with it to yield further improvements.
- **vs FreeU [Si et al., ICCV 2023]**: Both represent works that propose "free" improvements after analyzing the internal mechanisms of U-Net, sharing a highly aligned methodology.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea is extremely simple yet the analysis is profound; the sequential $\rightarrow$ parallel network rewiring works surprisingly well.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative evaluation on multiple datasets + user study + PickScore + FID, along with validation of its generalization as a backbone for other methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear analytical logic; flows seamlessly from problem identification to root cause analysis to the solution.
- Value: ⭐⭐⭐⭐ Plug-and-play value for all downstream tasks using GLIGEN, offering high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [\[ECCV 2024\] Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching](towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)
- [\[ECCV 2024\] SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)
- [\[ECCV 2024\] DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)
- [\[ICML 2025\] FG-CLIP: Fine-Grained Visual and Textual Alignment](../../ICML2025/object_detection/fg-clip_fine-grained_visual_and_textual_alignment.md)

</div>

<!-- RELATED:END -->
