---
title: >-
  [Paper Note] Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective
description: >-
  [AAAI 2026][Segmentation][Open-vocabulary semantic segmentation] This paper systematically investigates CLIP's internal mechanisms from an explainability perspective, revealing a "distraction" phenomenon in which CLIP allocates substantial attention resources to target-irrelevant tokens in deeper layers. The proposed training-free method RF-CLIP performs attention redistribution to refocus dispersed resources onto target regions, achieving state-of-the-art performance across 8 benchmarks while maintaining inference efficiency.
tags:
  - AAAI 2026
  - Segmentation
  - Open-vocabulary semantic segmentation
  - CLIP
  - attention redistribution
  - distraction phenomenon
  - training-free
date: 2026-05-08
content_hash: e914d6e76dc97495
---

# Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective

**Conference**: AAAI 2026
**arXiv**: [2511.16170](https://arxiv.org/abs/2511.16170)
**Code**: [github.com/liblacklucy/RF-CLIP](https://github.com/liblacklucy/RF-CLIP)
**Area**: Segmentation
**Keywords**: Open-vocabulary semantic segmentation, CLIP, attention redistribution, distraction phenomenon, training-free

## TL;DR

This paper systematically investigates CLIP's internal mechanisms from an explainability perspective, revealing a "distraction" phenomenon in which CLIP allocates substantial attention resources to target-irrelevant tokens in deeper layers. The proposed training-free method RF-CLIP performs attention redistribution to refocus dispersed resources onto target regions, achieving state-of-the-art performance across 8 benchmarks while maintaining inference efficiency.

## Background & Motivation

Open-vocabulary semantic segmentation (OVSS) associates category prompts with corresponding pixels via pixel-level vision-language alignment. Existing methods fall into three paradigms:

**Joint fine-tuning**: simultaneously fine-tuning CLIP and segmentation components

**Pre-fine-tuning**: retraining CLIP via fine-grained contrastive learning

**Training-free adaptation**: modulating only the last residual attention layer of CLIP, or integrating visual foundation models (VFMs)

However, these approaches **rarely examine CLIP's performance boundaries in dense prediction from an explainability perspective**, nor do they explore the root cause of its inherent inter-layer spatial misalignment.

The authors' systematic analysis reveals a key phenomenon — the **"distraction" phenomenon**:

1. **Shallow layers** (1–2): attention is primarily concentrated on query-relevant tokens, with strong spatial consistency
2. **Deep layers** (7–12): a large number of high-attention tokens unrelated to the target query (distractor tokens) emerge, progressively diminishing the saliency of target regions
3. These distractor tokens **occupy the same spatial positions** across different query points, indicating spuriously high correlation with all queries
4. They manifest as **prominent vertical stripes** in self-attention maps

Further analysis reveals that distractor tokens originate from **over-activation in specific dimensions** — CLIP inherently produces extremely large embedding weights in certain channels (e.g., dimensions 4, 162, 474 for ViT-B/16), a data-independent intrinsic property. Filtering these tokens substantially improves OVSS performance.

## Method

### Overall Architecture

RF-CLIP is a training-free attention modulation method that simulates the human "distraction → refocusing" behavior, correcting CLIP's spatial misalignment layer by layer. Each layer's correction comprises three steps:
1. **Distractor Localization**: identifying distractor tokens that consume disproportionate attention resources
2. **Defocus Localization**: detecting target tokens that receive insufficient attention
3. **Weight Redistribution**: transferring attention from distractor tokens to defocused target tokens

### Key Designs

#### 1. **Discovery and Localization of Distraction Dimensions and Distractor Tokens**

By computing the layer-averaged dense embedding $\bar{f} = \frac{1}{L}\sum_{l=1}^{L}\frac{f^l}{\sum_{j=1}^d f^l[:,j]}$ across all layers, the authors find that three large-scale OVSS benchmark datasets **exhibit consistent weight distribution peaks at the same dimensions** (e.g., dimensions 4, 162, 474 for ViT-B/16), which are defined as distraction dimensions $\mathcal{D}_{dis}$.

Distractor token localization: for the $i$-th token at layer $l$, the maximum embedding weight over distraction dimensions is computed as:

$$\phi_i^l = \max_{j \in \mathcal{D}_{dis}} \frac{f_i^l[j]}{\sum_{k=1}^d f_i^l[k]}$$

Tokens satisfying $\phi_i^l > \tau$ are identified as distractor tokens, with threshold $\tau = 5/d$.

**Design Motivation**: Experiments confirm that tokens with large embedding weights on distraction dimensions inevitably become distractor tokens during self-attention computation. The attention weights of distractor tokens exhibit an **exponential growth** relationship with $\phi_i$.

#### 2. **Defocus Token Localization**

Defocus tokens are treated as foreground instances, and the localization problem is formulated as a bipartite graph cut. The key-key attention $\text{Attn}_{kk}^l$ is used as a similarity matrix for spectral clustering, minimizing the normalized cut energy:

$$\bm{y}_1^l = \arg\min_{\bm{y}^{l\top}\bm{D1}=0} \frac{\bm{y}^{l\top}(\bm{D}^l - \text{Attn}_{kk}^l)\bm{y}^l}{\bm{y}^{l\top}\bm{D}^l\bm{y}^l}$$

where $\bm{y}_1^l$ is the Fiedler vector (the eigenvector corresponding to the second smallest eigenvalue of the generalized eigensystem); tokens satisfying $\bm{y}_1^l[i] > \frac{1}{N}\sum_{j=1}^N \bm{y}_1^l[j]$ are identified as defocus tokens.

**Design Motivation**: Graph cuts naturally partition an image into foreground and background groups, providing robustness across diverse scenes without requiring additional annotations or training.

#### 3. **Weight Redistribution**

Two complementary mechanisms are employed:

**Attention weight redistribution**: the attention weights of distractor tokens are first suppressed, and the reduced amount is retained as a redistribution budget $\Omega$:

$$\text{Attn}_{qk}^{l,h}[i,j] \leftarrow (1-\beta) \cdot \text{Attn}_{qk}^{l,h}[i,j], \quad \forall j \in \mathcal{T}_{dis}$$

$$\Omega[i] = \beta \cdot \sum_{j \in \mathcal{T}_{dis}} \text{Attn}_{qk}^{l,h}[i,j]$$

The budget is then distributed to defocus tokens proportionally to their original attention weights:

$$\text{Attn}_{qk}^{l,h}[i,j] \leftarrow \text{Attn}_{qk}^{l,h}[i,j] + \Omega[i] \cdot \rho[i,j], \quad \forall j \in \mathcal{T}_{def}$$

where $\beta = 0.7$ is the decay factor. This process maintains column normalization, preserving the original attention distribution and effectively preventing model collapse.

**Embedding weight redistribution**: for distractor tokens, embeddings along distraction dimensions are replaced by 3×3 neighborhood averaging:

$$f_i^l[j] = \frac{1}{8} \cdot \sum_{\hat{i} \in \mathcal{O}_i} f_{\hat{i}}^l[j], \quad \forall j \in \mathcal{D}_{dis}, i \in \mathcal{T}_{dis}$$

Only embeddings along distraction dimensions are adjusted, leaving normal-dimension distributions intact.

**Dense prediction**: after correction, the layer-averaged attention $\overline{\text{Attn}}_{kk} = \frac{1}{L}\sum_{l=1}^L \text{Attn}_{kk}^l$ replaces $\text{Attn}_{qk}^L$ at the last layer.

### Loss & Training

RF-CLIP is a **completely training-free** method, requiring no training or fine-tuning. All operations are performed directly during CLIP's inference by applying layer-by-layer modulation to the attention mechanism.

## Key Experimental Results

### Main Results

Based on CLIP ViT-B/16, mIoU (%) on 8 standard benchmarks:

| Method | Extra VFM | VOC21 | Context60 | COCO-Obj | VOC20 | Context59 | COCO-Stuff | Cityscapes | ADE20K | Avg. |
|--------|-----------|-------|-----------|----------|-------|-----------|------------|------------|--------|------|
| ProxyCLIP | DINO | 59.1 | 35.2 | 36.2 | 78.2 | 38.8 | 26.2 | 38.1 | 19.6 | 41.4 |
| CASS | DINO | 65.8 | 36.7 | 37.8 | 87.8 | 40.2 | 26.7 | 39.4 | 20.4 | 44.4 |
| SC-CLIP | ✗ | 64.6 | 36.8 | 37.7 | 84.3 | 40.1 | 26.6 | 41.0 | 20.1 | 43.9 |
| **RF-CLIP** | **✗** | **64.8** | 36.4 | **37.9** | **87.0** | 39.8 | 26.3 | **41.3** | **20.4** | **44.2** |
| RF-CLIP+PAMR | ✗ | **67.2** | **37.9** | **39.1** | 87.0 | **41.4** | **27.5** | **43.0** | **21.0** | **45.5** |

Without any additional VFM, RF-CLIP surpasses ProxyCLIP (+2.8 mIoU) and CASS, which both rely on DINO, achieving a 1.6% average mIoU improvement over methods sharing the same baseline.

### Ablation Study

| Configuration | VOC21 | COCO-Stuff | Cityscapes | ADE20K | Avg. | Notes |
|---------------|-------|------------|------------|--------|------|-------|
| Baseline | 59.1 | 23.6 | 32.1 | 16.9 | 32.9 | Layer-averaged kk attention |
| +Random mean filtering | 58.8 | 21.4 | 31.6 | 14.7 | 31.6 | Random token filtering; performance drops |
| +Distractor localization+mean filtering | 60.3 | 24.4 | 33.6 | 17.5 | 34.0 | Distraction-aware filtering; +1.1% |
| +Attention redistribution | 61.5 | 24.8 | 35.3 | 18.3 | 35.0 | +2.1% |
| +Embedding redistribution | 62.1 | 25.2 | 36.7 | 18.9 | 35.7 | +2.8% |
| +Both redistributions | 63.2 | 25.4 | 38.5 | 19.3 | 36.6 | +3.7% |
| **+Defocus localization** | **64.8** | **26.3** | **41.3** | **20.4** | **38.2** | **+5.3%** |

**Efficiency analysis** (VOC21 benchmark):

| Model | FLOPs (G) | Params (M) | Speed (FPS) | mIoU (%) |
|-------|-----------|------------|-------------|---------|
| Baseline | 16.7 | 149.6 | 12.7 | 58.1 |
| ProxyCLIP | 34.1 | 235.4 | 6.1 | 59.1 |
| **RF-CLIP** | **17.1** | **149.6** | **12.0** | **64.8** |

RF-CLIP runs at twice the inference speed of ProxyCLIP while achieving 5.7% higher mIoU.

**Suppression strategy comparison**:

| Strategy | VOC21 | COCO-Stuff | Cityscapes | ADE20K |
|----------|-------|------------|------------|--------|
| Baseline | 58.1 | 23.0 | 31.1 | 16.3 |
| $-\infty$ masking | 3.5 | 0.1 | 2.0 | 0.1 |
| Low-pass filtering | 7.9 | 1.1 | 6.2 | 1.4 |
| Mean filtering | 59.3 | 24.0 | 35.4 | 18.2 |
| Median filtering | 58.6 | 23.7 | 34.5 | 17.6 |

### Key Findings

1. Directly eliminating distractor tokens ($-\infty$ masking, low-pass filtering) causes catastrophic performance collapse, as it destroys the topological structure of CLIP's high-dimensional space
2. Distractor tokens should maintain spatial consistency with neighboring regions; mean/median filtering is therefore effective
3. Redistributing attention resources to defocus tokens is more effective than distributing to all non-distractor tokens or to the [CLS] token
4. A 3×3 neighborhood is optimal for embedding redistribution; larger neighborhoods degrade performance, indicating that distractor tokens are concentrated in high-frequency regions
5. The threshold $\tau = 5/d$ achieves the best performance across all benchmarks; performance degradation from a low threshold (high false-positive rate) substantially exceeds that from a high threshold

## Highlights & Insights

1. **Explainability-driven method design**: the approach begins with systematic analysis of CLIP's internal mechanisms, identifies the distraction phenomenon, and then devises a targeted solution. This "understand first, then design" paradigm is highly instructive
2. **Training-free method achieves SOTA**: without introducing any additional models or training, RF-CLIP surpasses methods that rely on extra VFMs such as DINO solely by modulating CLIP's own attention mechanism
3. **Data-agnostic nature of distraction dimensions**: the same distraction dimensions appear consistently across different datasets, indicating that this is an intrinsic property of CLIP's pretraining process
4. **Carefully controlled experiments**: the contrast between random token filtering and distractor-aware token filtering is elegantly designed, convincingly demonstrating the importance of distraction-aware processing
5. **"Conservation" design for attention resources**: redistribution maintains column normalization and allocates resources proportionally to original weights, balancing performance improvement with prevention of model collapse

## Limitations & Future Work

1. Thresholds and distraction dimensions must be set separately per CLIP architecture (ViT-B/16 vs. ViT-L/14), limiting generalizability
2. Eigenvalue decomposition in spectral clustering introduces additional computation, though the overall cost remains lower than incorporating a VFM
3. Distractor token identification in ViT-L/14 requires an additional attention-weight condition, making it more complex than for ViT-B/16
4. Bipartite graph cuts may oversimplify highly complex scenes with heavily overlapping multiple objects

## Related Work & Insights

- **Registers (ICLR 2024)**: also identifies high-norm token artifacts in ViT feature maps, but attributes them to low-information background regions
- **CLIPtrase / DeCLIP**: regard distractor tokens as proxies of [CLS], whereas this paper demonstrates experimentally that attention resources are diverted not only from [CLS] but also from foreground tokens
- **ProxyCLIP / CASS**: replace CLIP's attention with DINO's; RF-CLIP demonstrates that directly repairing CLIP itself is more efficient
- **SCLIP / ClearCLIP / NACLIP**: modify only the last-layer attention matrix, neglecting spatial misalignment in intermediate layers
- The discovery of the distraction phenomenon may have implications for applying CLIP to other dense prediction tasks, such as depth estimation and instance segmentation

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Discovering the distraction phenomenon from an explainability perspective and proposing attention redistribution is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 benchmarks, extensive ablations, efficiency analysis, multiple controlled experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, progressive structure from phenomenon discovery to method design, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Training-free SOTA, revealing novel insights into CLIP's internal mechanisms)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[AAAI 2026\] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](infoclip_bridging_vision-language_pretraining_and_open-vocab.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)

</div>

<!-- RELATED:END -->
