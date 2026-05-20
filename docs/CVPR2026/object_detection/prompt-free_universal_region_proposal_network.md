---
title: >-
  [Paper Note] Prompt-Free Universal Region Proposal Network
description: >-
  [CVPR 2026][Object Detection][Region Proposal] PF-RPN replaces text/image prompts with learnable visual embeddings and introduces three modules—Sparse Image-Aware Adapter (SIA), Cascaded Self-Prompting (CSP)…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Region Proposal"
  - "Prompt-Free Detection"
  - "Zero-Shot Generalization"
  - "Learnable Embedding"
  - "Open World"
date: 2026-05-08
content_hash: 900075d9601d8cbe
---

# Prompt-Free Universal Region Proposal Network

**Conference**: CVPR 2026
**arXiv**: [2603.17554](https://arxiv.org/abs/2603.17554)  
**Code**: [GitHub](https://github.com/tangqh03/PF-RPN)  
**Area**: Object Detection
**Keywords**: Region Proposal, Prompt-Free Detection, Zero-Shot Generalization, Learnable Embedding, Open World

## TL;DR

PF-RPN replaces text/image prompts with learnable visual embeddings and introduces three modules—Sparse Image-Aware Adapter (SIA), Cascaded Self-Prompting (CSP), and Centrality-Guided Query Selection (CG-QS)—to achieve state-of-the-art zero-shot region proposals across 19 cross-domain datasets using only 5% of COCO training data.

## Background & Motivation

**Background**: Region Proposal Networks (RPNs) are a core component of object detection, responsible for generating candidate bounding boxes. Open-vocabulary detection (OVD) models such as Grounding DINO and YOLO-World leverage textual category names or exemplar images as prompts to localize objects, demonstrating cross-domain generalization capability.

**Limitations of Prior Work**: OVD methods rely on external prompts (category names or reference images), which are often unavailable in real-world scenarios—such as industrial defect inspection and underwater object detection, where neither category labels nor reference images can be obtained in advance. Although some prompt-free OVD methods (e.g., GenerateU, DetCLIPv3) utilize large vision-language models (VLMs) to generate textual descriptions and eliminate manual prompts, they introduce substantial memory and latency overhead.

**Key Challenge**: Achieving prompt-free universal object localization requires avoiding both external text/image inputs and computationally expensive generative VLMs. A lightweight and efficient substitute for text embeddings must be identified.

**Goal**: To design a lightweight region proposal network that localizes arbitrary objects in unseen domains using only visual features, without requiring any external prompt.

**Key Insight**: The authors observe that text embeddings in OVD models essentially serve as query signals for matching visual features; therefore, a **learnable visual embedding** can replace text embeddings and be dynamically updated using the visual features of the input image itself. Two further observations are made: (1) intra-object features exhibit stronger localization capability than learnable embeddings, and (2) queries near object centers produce more accurate proposals than those near object boundaries.

**Core Idea**: Replace text prompts with learnable visual embeddings, and achieve prompt-free universal object localization through adaptive aggregation of multi-scale visual features and iterative refinement via cascaded self-prompting.

## Method

### Overall Architecture

PF-RPN is built upon Grounding DINO (Swin-B backbone). Given an input image, the encoder extracts four-scale feature maps $F_i^I \in \mathbb{R}^{H_i \times W_i \times C}$, which are then processed by three core modules: SIA fuses the learnable embedding $F^T$ with multi-scale visual features for initial localization → CSP iteratively refines the embedding via deep-to-shallow cascaded self-prompting to capture hard examples → CG-QS selects high-quality queries using centrality scores → a DETR-like decoder produces the final proposal boxes.

### Key Designs

1. **Sparse Image-Aware Adapter (SIA)**:

    - **Function**: Adaptively fuses the learnable embedding $F^T$ with the most informative visual feature scale.
    - **Mechanism**: Employs a Mixture-of-Experts (MoE) routing mechanism. Global average pooling is applied to each feature scale to obtain a compact representation $\bar{F}_i^I$; a lightweight MLP router predicts per-scale importance weights $w_i = \text{Router}(\bar{F}_i^I)$. The top-$k$ ($k \leq 4$, default $k=2$) scales are selected, and cross-attention updates the embedding: $\tilde{F}^T = \sum_{j=1}^{k} \tilde{w}_{\sigma(j)} \cdot \text{Attn}(F^T, [\bar{F}_{\sigma(j)}^I, F_{\sigma(j)}^I])$.
    - **Design Motivation**: Different feature scales contribute differently to objects of different sizes (shallow scales favor small objects; deep scales capture large objects). Naïvely fusing all scales introduces redundant noise. MoE routing filters out irrelevant scales prior to attention, while combining global and local features to provide coarse-to-fine visual cues.

2. **Cascaded Self-Prompting (CSP)**:

    - **Function**: Iteratively refines the learnable embedding over multiple rounds to capture hard examples (small or occluded objects) missed by SIA.
    - **Mechanism**: Proceeds in a deep-to-shallow cascade. At each scale $i$, a similarity mask $M_i = \mathbb{1}(\cos(\tilde{F}_{i-1}^T, F_i^I) > \delta)$ ($\delta=0.3$) is computed, and the embedding is refined via masked average pooling: $\tilde{F}_i^T = \tilde{F}_{i-1}^T + \text{MAP}(M_i, F_i^I)$.
    - **Design Motivation**: Background regions remain activated after SIA, as single-step adaptation is insufficient. A key observation is exploited: intra-object features have stronger localization capability than learnable embeddings, so activated object-region features are used to reversely guide embedding updates. The deep-to-shallow cascade first aggregates semantics and then integrates structural details. Three iterations are used by default, adding only ~4.6 ms latency.

3. **Centrality-Guided Query Selection (CG-QS)**:

    - **Function**: Selects high-quality query embeddings near object centers for final proposal generation.
    - **Mechanism**: A lightweight MLP predicts a centrality score $g_i$ for each query $f_i$, supervised by $c_i = \sqrt{\frac{\min(l,r)}{\max(l,r)} \times \frac{\min(t,b)}{\max(t,b)}}$ (distances to the four sides of the GT box), with centrality loss $\mathcal{L}_{ctr} = \sum_i \|g_i - c_i\|_1$. At inference, centrality scores are combined with classification scores to determine the final candidate query set.
    - **Design Motivation**: Visualization reveals that queries near object centers produce more accurate bounding boxes than those near boundaries; thus prioritizing central queries reduces false positives and improves proposal quality.

### Loss & Training

The total loss is: $\mathcal{L} = \mathcal{L}_{reg} + \mathcal{L}_{cls} + \mathcal{L}_{rt} + \lambda \mathcal{L}_{ctr}$, where $\mathcal{L}_{reg}$ comprises L1 and GIoU losses, $\mathcal{L}_{cls}$ is a contrastive loss between queries and the learnable embedding, $\mathcal{L}_{rt} = \text{std}(w_i)$ is an auxiliary load-balancing loss for the MoE router (to prevent over-activation of a few experts), and $\lambda=5$.

Training uses only **5% of COCO** (80 categories) and **5% of ImageNet** (1,000 categories with pseudo boxes). ImageNet classification data is incorporated to mitigate encoder bias induced by fine-tuning on detection data. Training is conducted on 4× RTX 4090 GPUs.

## Key Experimental Results

### Main Results

Average Recall is evaluated on CD-FSOD (6 cross-domain datasets) and ODinW13 (13 diverse-domain datasets):

| Method | Prompt-Free | CD-FSOD AR100/300/900 | ODinW13 AR100/300/900 |
|--------|-------------|----------------------|----------------------|
| GDINO (text prompt) | ✗ | 52.9/53.5/54.7 | 72.1/73.4/74.0 |
| GDINO ("object") | ✓ | 54.7/57.8/61.6 | 69.1/70.9/72.4 |
| YOLOE-v8-L | ✗ | 44.4/46.2/47.1 | 66.6/67.8/68.3 |
| GenerateU | ✓ | 47.7/54.1/55.7 | 67.3/71.5/72.2 |
| Cascade RPN | ✓ | 45.8/52.0/56.9 | 60.9/65.5/70.2 |
| **PF-RPN (Ours)** | **✓** | **60.7/65.3/68.2** | **76.5/78.6/79.8** |

PF-RPN outperforms Grounding DINO on CD-FSOD by +7.8/+11.8/+13.5 AR and on ODinW13 by +4.4/+5.2/+5.8 AR. It surpasses GenerateU by +13.0 AR100 while requiring only 0.5 GB VRAM (vs. 12.2 GB for GenerateU) and running approximately 20× faster.

### Ablation Study

| Configuration | CD-FSOD AR100 | Notes |
|--------------|--------------|-------|
| Baseline (GDINO) | 52.9 | No modules |
| + SIA | 57.8 | Visual features more effective than text (+4.9) |
| + SIA + CSP | 60.2 | Cascaded self-prompting reduces missed detections (+2.4) |
| + SIA + CG-QS | 59.6 | Centrality selection improves quality (+1.8) |
| + SIA + CSP + CG-QS (Full) | **60.7** | Three modules are complementary; best performance |

### Key Findings

- **SIA contributes most**: Alone it yields +4.9 AR, demonstrating that visual features are substantially more effective query signals than text embeddings.
- **MoE routing is critical**: Removing MoE drops AR100 from 60.7 to 58.6 on CD-FSOD and from 76.5 to 68.7 on ODinW13. Attention operates within individual scales and cannot select across scales.
- **CSP iteration count**: 3 iterations achieve AR100=60.7, surpassing 1 iteration (59.6) by 1.1 AR with only ~4.6 ms additional latency.
- **Top-k selection**: $k=2$ is optimal (AR100=60.7); $k=1$ provides insufficient information, while $k=3,4$ introduce redundancy.
- **Integration into other detectors**: Embedding into DE-ViT yields +3.7 AP; embedding into CD-ViTO yields +5.5 AP.
- **Backbone-agnostic**: Significant gains are observed for both Swin-B (+7.8 AR100) and ResNet-50 (+5.2 AR100).

## Highlights & Insights

- **Truly prompt-free detection**: The method requires no text, images, or VLMs. Replacing the text channel with a learnable embedding eliminates the prompt-dependency bottleneck in practical OVD deployment—a paradigm transferable to other cross-modal alignment tasks.
- **Exceptional data efficiency**: Strong cross-domain generalization across 19 datasets is achieved with only 5% of COCO. Diminishing returns are observed when scaling from 5% to 10%, suggesting that the model's architectural design is the primary driver of generalization.
- **MoE routing for feature scale selection** is an elegant design: while conventional methods apply FPN or attention for multi-scale fusion, the router here pre-filters the most relevant feature scales before attention, avoiding noise from irrelevant scales.

## Limitations & Future Work

- **Localization only, no classification**: PF-RPN is responsible solely for localization and does not produce category labels; a downstream classifier is required to form a complete detection pipeline.
- **Upper bound under extreme domain gaps**: Performance may still have room for improvement in highly challenging cross-domain scenarios (e.g., ArTaxOr insect imagery).
- **Static threshold $\delta=0.3$**: The cosine similarity threshold in CSP is fixed and may not generalize optimally across all scenarios; an adaptive threshold could yield further improvement.
- **Initialization of learnable embeddings**: Current random initialization may benefit from replacing with pre-trained visual prototypes to improve convergence speed and final performance.

## Related Work & Insights

- **vs. Grounding DINO**: PF-RPN is built on the GDINO framework but removes the text encoder and text prompt dependency, replacing language-guided query selection with SIA+CSP+CG-QS. It consistently outperforms GDINO in the prompt-free setting while being faster (the text encoder's computational overhead is eliminated).
- **vs. GenerateU**: GenerateU employs a generative approach to map visual regions to free-form text names for prompt-free detection, but depends on a large captioner requiring 12.2 GB VRAM. PF-RPN requires only 0.5 GB, runs ~20× faster, and achieves superior performance (+13.0 AR100).
- **vs. YOLOE**: Although YOLOE supports prompt-free detection, its zero-shot generalization is constrained by static text proxies. PF-RPN achieves stronger generalization through dynamic visual embedding updates.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of learnable embeddings replacing text prompts, MoE routing, and cascaded self-prompting is elegant, though the overall framework remains grounded in GDINO.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 19 cross-domain datasets with full ablations, multiple backbones, integration experiments, and efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated module designs and effective visualizations.
- Value: ⭐⭐⭐⭐ Addresses a critical bottleneck in practical OVD deployment with high data efficiency and inference efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_texttoimage_diffusion_v.md)
- [\[CVPR 2026\] UAVGen: Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection](uavgen_visual_prototype_conditioned_focal_region_generation_for_uav_based_object_detection.md)
- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)

</div>

<!-- RELATED:END -->
