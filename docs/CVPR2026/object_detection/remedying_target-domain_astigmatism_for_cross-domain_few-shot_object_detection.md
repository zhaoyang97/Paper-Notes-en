---
title: >-
  [Paper Note] Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection
description: >-
  [CVPR 2026][Object Detection][Paper Note] This work first identifies the "astigmatism" phenomenon in Cross-Domain Few-Shot Object Detection (CD-FSOD), where model attention remains persistently dispersed in the target domain. Inspired by the human foveal vision system, three complementary modules—Positive Pattern Refinement (PPR), Negative Context Modulation (
tags:
  - CVPR 2026
  - Object Detection
date: 2026-05-08
content_hash: 39d5844aed48b4d9
---
# Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection

**Conference**: CVPR 2026  
**arXiv**: [2603.18541](https://arxiv.org/abs/2603.18541)  
**Code**: None  
**Area**: Object Detection  
**Keywords**: Cross-domain few-shot detection, attention astigmatism, biomimetic foveal vision, prototype learning, negative context modeling

## TL;DR

This work first identifies the "astigmatism" phenomenon in Cross-Domain Few-Shot Object Detection (CD-FSOD), where model attention remains persistently dispersed in the target domain. Inspired by the human foveal vision system, three complementary modules—Positive Pattern Refinement (PPR), Negative Context Modulation (NCM), and Textual Semantic Alignment (TSA)—are designed to reshape attention, achieving SOTA performance on six cross-domain benchmarks by a significant margin.

## Background & Motivation

**Background**: Cross-Domain Few-Shot Object Detection (CD-FSOD) aims to adapt a detector pre-trained on a source domain to a target domain with scarce annotations, which is a critical requirement in practical applications such as medical diagnosis and industrial inspection. Existing methods like CD-ViTO have established multi-domain benchmarks, but performance remains unsatisfactory.

**Limitations of Prior Work**: Through an in-depth analysis of attention patterns across Transformer layers, the authors discovered a previously overlooked phenomenon: in the source domain, attention gradually focuses on foreground objects as the network depth increases; however, in the target domain, attention remains dispersed and unfocused, leading to excessively large bounding boxes and numerous redundant predictions. Much like human astigmatism, the model in the target domain fails to focus on key objects.

**Key Challenge**: By measuring the attention distance $\bar{d} = \frac{1}{N}\sum_{i,j} A_{ij} \cdot \|p_i - p_j\|$, it is found that the attention distance in the target domain is consistently higher than in the source domain. Although conventional fine-tuning tends to reduce defocusing (the difference in attention distance is negative), the effect is insufficient—the attention dispersion in the target domain after fine-tuning still far exceeds that of the source domain.

**Goal**: To enhance the model's intrinsic tendency to remedy the astigmatism problem, shifting attention from a dispersed pattern to a focused, object-centric mode, thereby achieving precise detection in the target domain.

**Key Insight**: Inspired by the foveal structure of the human visual system—the central foveal region captures high-detail information (foreground) while the peripheral region captures low-detail information (background), with a center-surround contrast mechanism maintaining focused attention. Accordingly, three modules are designed to enhance the representation of the central region, the peripheral region, and the discriminability between the two.

**Core Idea**: Target region attention is enhanced through class-specific foreground prototypes (upscaling $A_2, A_3$), pseudo-foreground responses in background regions are suppressed through a unified background prototype (downscaling $A_1$), and foreground-background separation is reinforced from a cross-modal perspective using "not [class]" textual cues. This three-pronged approach transforms defocused attention into a focused mode.

## Method

### Overall Architecture

The method is built upon the GLIP detector, using a Swin Transformer backbone to extract multi-scale features. The core idea is to decompose "astigmatism" into three individually addressable attention pathologies and correct them sequentially. During the fine-tuning phase, the model extracts class-specific foreground prototypes and a unified background prototype from support samples and stores them in a prototype library. Simultaneously, the model is jointly trained with detection loss and cross-modal alignment loss. During inference, prototypes are retrieved from the library: PPR refines foreground features, NCM refines background features, and the two complementary features are fused before being fed into the detection head. The three modules correspond to the three roles of foveal vision—central perception (foreground), peripheral perception (background), and center-surround contrast (foreground/background boundary).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    Q["Query Image"] --> BK["Swin Backbone<br/>Multi-scale Visual Features"]
    S["Support Samples<br/>(Few Annotations)"] --> P["Prototype Library<br/>Class-specific Foreground + Unified Background"]
    BK --> PPR["Positive Pattern Refinement (PPR)<br/>Inject foreground prototypes, upscale intra-object attention"]
    BK --> NCM["Negative Context Modulation (NCM)<br/>Inject background prototypes, suppress background pseudo-responses"]
    P -. Foreground Protot. .-> PPR
    P -. Background Protot. .-> NCM
    T["Negative Text: not [class]<br/>(BERT Encoding)"] --> TSA["Textual Semantic Alignment (TSA)<br/>Contrastive loss aligns background prototypes with negative text"]
    P -. Background Protot. .-> TSA
    TSA -. Strengthen FG/BG Boundary .-> NCM
    PPR --> F["Complementary Fusion<br/>Foreground Features + Background Features"]
    NCM --> F
    F --> H["Detection Head → Box + Category"]
```

### Key Designs

**1. Positive Pattern Refinement (PPR): Redirecting dispersed intra-object attention back to the foreground**

The most direct manifestation of astigmatism is that the attention weights $A_2, A_3$ between patches within the same object are low, meaning the model "sees" the target but cannot "solidify" it. PPR first computes class prototypes $\mathbf{p}_{fg}^c$ by mean-pooling the foreground regions of support samples. During inference, it calculates the cosine similarity between each position on the feature map and all class prototypes, uses a threshold $\tau_{fg}$ to generate a foreground mask $\mathbf{M}_{fg}$, and injects temperature-scaled softmax-weighted prototypes into the high-similarity regions within the mask: $\mathbf{f}_v^{pos}(x,y) = \mathbf{f}_v \cdot \mathbf{M}_{fg} + \gamma_{fg} \sum_c w_c \mathbf{p}_{fg}^c \cdot \mathbf{M}_{fg}$. Prototype injection is equivalent to infusing a common class semantic into all patches within the object. Once feature consistency increases, attention between patches of the same object is naturally elevated. This explains why PPR had the largest single-module contribution (+2.06%) in the ablation study—foreground enhancement is the main battlefield for correcting astigmatism.

**2. Negative Context Modulation (NCM): Suppressing pseudo-foreground responses in the background**

The other side of defocusing is excessively high attention weights $A_1$ for background patches, causing the model to mistake large background areas for potential targets, resulting in oversized boxes and redundant predictions. NCM mean-pools regions outside annotated boxes in support samples to obtain a **category-agnostic** unified background prototype $\mathbf{p}_{bg}$. During inference, background regions are similarly located, and the prototype is injected: $\mathbf{f}_v^{neg}(x,y) = \mathbf{f}_v \cdot \mathbf{M}_{bg} + \gamma_{bg} \mathbf{p}_{bg} \cdot \mathbf{M}_{bg}$. Once background features are enhanced, they become more separable from the foreground in the feature space, reducing erroneous foreground attention to the background. A single prototype is used for the background because "non-target" is a universal concept; a unified representation saves parameters while remaining effective—ablation studies show NCM gains are most significant in scenarios with high background ratios (0.7–1.0).

**3. Textual Semantic Alignment (TSA): Reinforcing the foreground-background boundary via language modality**

Pure visual prototypes may be inaccurate when the domain gap is large, and visual streams alone might not sufficiently separate foreground and background. TSA introduces cross-modal supervision by constructing negative text in the form of "not [class]" (e.g., "not sofa, not dog"), encoded by BERT into textual features $\mathbf{F}_t^{bg}$. The background prototype from NCM is treated as the visual feature $\mathbf{F}_v^{bg}$. Both are mapped to a shared semantic space via learnable projection layers and aligned using a contrastive loss: $\mathcal{L}_{ctr} = -\log \frac{\exp(\text{diag}(\mathcal{S})/\tau)}{\sum_{i,j}\exp(\mathcal{S}_{i,j}/\tau)}$. The cleverness of the "not [class]" design lies in its ability to explicitly define background semantics from an opposing perspective, allowing the model to establish a clear foreground/background boundary at the linguistic level, providing center-surround contrastive enhancement for visual prototypes.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \mathcal{L}_{detection} + \lambda_{bg} \cdot \mathcal{L}_{ctr}$, where the detection loss includes classification and localization losses. The TSA loss weight $\lambda_{bg}$ is optimal at $10^3$. During inference, the enhanced features from PPR and NCM are combined via complementary fusion $\mathbf{F}_v^{enhanced} = \mathbf{f}_v^{pos} + \mathbf{f}_v^{neg}$ before entering the detection head.

## Key Experimental Results

### Main Results

| Method | 1-shot Avg mAP | 5-shot Avg mAP | 10-shot Avg mAP |
|------|-------|----------|-------|
| GLIP | 18.98 | 29.36 | 33.93 |
| CD-ViTO* | 15.96 | 28.30 | 33.47 |
| Domain-RAG* | 16.52 | 27.98 | 32.81 |
| VFM-MoE* | 17.59 | 28.45 | 33.94 |
| **Ours** | **23.81** | **33.73** | **39.17** |

Across 6 datasets (ArTaxOr, Clipart1k, DIOR, DeepFish, NEU-DET, UODD) covering diverse domains like insects, cartoons, remote sensing, and underwater, the method achieves SOTA across 1/5/10-shot settings. The average mAP for 5-shot is 5.28 points higher than the strongest baseline (33.73 vs 28.45).

### Ablation Study

| Configuration | ArTaxOr | Clipart1k | DeepFish | NEU-DET | UODD | Average Gain |
|------|---------|------|---------|------|------|------|
| Baseline (GLIP) | 48.11 | 39.28 | 28.40 | 19.55 | 11.40 | - |
| +NCM | 49.95 | 40.52 | 29.58 | 20.48 | 12.21 | +1.06 |
| +NCM+PPR | 52.68 | 42.95 | 31.96 | 22.38 | 14.26 | +2.06 |
| +NCM+PPR+TSA | **54.98** | **44.83** | **33.87** | **23.64** | **15.66** | +1.59 |

The stacking of the three modules is effective, with PPR contributing the most (+2.06%), validating that foreground enhancement is core to correcting astigmatism.

### Key Findings

- Attention distance analysis quantitatively confirms: regular fine-tuning only reduces attention dispersion by 0.25%-1.30%, while the proposed method reduces it by 0.47%-1.72%.
- Background ratio analysis shows that NCM has a more pronounced advantage when the background ratio is high (0.7-1.0), verifying the effectiveness of negative context utilization.
- The number of background text entries reaches the best cost-performance ratio at 200 ( +1.39% AP, with only 84MB extra VRAM).
- The method significantly reduces redundant detection boxes (e.g., from 100 to 9 in maritime scenes); qualitative results are highly convincing.

## Highlights & Insights

- The discovery of the "astigmatism" problem is a significant scientific contribution in itself, revealing a neglected core issue in CD-FSOD through quantitative analysis of the attention distance metric.
- The biomimetic design analogy is very natural: Fovea → PPR, Peripheral Vision → NCM, Center-Surround Contrast → TSA, showing high alignment between biological inspiration and technical solution.
- The "not [class]" negative text prompt design is innovative and practical, defining background semantics from an inverse perspective to strengthen foreground/background separation.
- The method introduces minimal additional parameters and computational overhead, offering good deployment efficiency.

## Limitations & Future Work

- Prototype quality relies heavily on a few support samples; prototypes may be inaccurate in extreme few-shot (e.g., 1-shot) scenarios.
- Only validated on the Swin-Tiny backbone; performance on larger models or different architectures (e.g., ViT-L) remains unknown.
- Treating the background prototype as a unified concept might be oversimplified in complex scenes; hierarchical background modeling may be more effective.
- "not [class]" prompts require prior knowledge of category names in the target domain, which may need adjustment for open-vocabulary scenarios where categories are unknown.

## Related Work & Insights

- **vs CD-ViTO**: CD-ViTO preserves source domain priors through knowledge distillation but does not address attention dispersion. Ours is 5.43 points higher in 5-shot average mAP (33.73 vs 28.30).
- **vs Distill-CDFSOD**: Distillation methods are limited when domain gaps are large; this work directly improves feature quality at the attention mechanism level.
- **vs IPNet**: IPNet designs independent domain alignment paths for foreground and background but does not utilize cross-modal textual information; the TSA module in this work provides additional supervision at the semantic level.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The discovery of the astigmatism problem and the biomimetic foveal vision design are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Very comprehensive across 6 datasets, 3 shot settings, multi-module ablations, and attention visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative logic—from problem discovery and quantitative analysis to biomimetic inspiration and technical solution—is very clear.
- Value: ⭐⭐⭐⭐ The discovery of the astigmatism phenomenon and its rectification method are highly instructive for the CD-FSOD community and can be generalized to other cross-domain tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection](learning_multi-modal_prototypes_for_cross-domain_few-shot_object_detection.md)
- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)
- [\[CVPR 2026\] Anomaly-Related Residual Fields for Cross-domain Anomaly Detection](anomaly-related_residual_fields_for_cross-domain_anomaly_detection.md)
- [\[CVPR 2026\] Black-Box Domain Adaptation for Object Detection with Retention-Driven Knowledge Compression](black-box_domain_adaptation_for_object_detection_with_retention-driven_knowledge.md)

</div>

<!-- RELATED:END -->
