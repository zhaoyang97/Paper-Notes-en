---
title: >-
  [Paper Note] Part-level Semantic-guided Contrastive Learning for Fine-grained Visual Classification
description: >-
  [ICLR 2026][Self-Supervised Learning][Fine-grained Classification] PSCL utilizes ClearCLIP to decouple "region selection" and "region representation" into two separate branches. Combined with multi-scale multi-part progressive reasoning and a vision-language contrastive loss incorporating intermediate-granularity categories, it achieves SOTA or highly competitive accuracy across five FGVC datasets.
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "Fine-grained Classification"
  - "ClearCLIP"
  - "Part Localization"
  - "Multi-granularity Text"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 1a15601cd0454c3f
---

# Part-level Semantic-guided Contrastive Learning for Fine-grained Visual Classification

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Bzmb5LeCKx](https://openreview.net/forum?id=Bzmb5LeCKx)  
**Code**: https://github.com/joker-lin9/PSCL  
**Area**: Fine-grained Visual Classification / Vision-Language Contrastive Learning / Part Localization  
**Keywords**: Fine-grained Classification, ClearCLIP, Part Localization, Multi-granularity Text, Contrastive Learning

## TL;DR
PSCL utilizes ClearCLIP to decouple "region selection" and "region representation" into two separate branches. Combined with multi-scale multi-part progressive reasoning and a vision-language contrastive loss incorporating intermediate-granularity categories, it achieves SOTA or highly competitive accuracy across five FGVC datasets.

## Background & Motivation
**Background**: Fine-grained visual classification (FGVC) aims to distinguish sub-categories with extremely similar appearances within the same broad category (e.g., bird species, car models, aircraft types). Mainstream approaches follow two lines: 1) fine-grained feature representation via multi-scale fusion or attention mechanisms (e.g., PMG, TransFG), and 2) discriminative part capture via weakly supervised part localization (cropping and zooming into salient regions).

**Limitations of Prior Work**: The authors observe that models exhibit distinct feature preferences for "rigid objects" (aircraft, cars) and "non-rigid objects" (birds, dogs). FGVC effectively requires two types of features: ① **part-level fine-grained features** to characterize local detail differences, and ② **spatial relationship features** to characterize inter-class structural differences. However, existing methods intertwine these two: spatial relationship features depend on cross-category matching of shared regions, which **conflicts** with the precise representation of part details, especially for non-rigid objects where poses make spatial structures unstable.

**Key Challenge**: ① Part representation and spatial relationship representation compete for the same set of features. When localization and classification share one feature extractor, it leads to "weak grounding" of parts, biasing them toward classification targets. Under occlusion or pose changes, the same branch may fixate on different parts across different images, causing redundancy and training instability. ② Existing methods use a uniform part branch design for all categories, ignoring the homogeneity of part details between similar categories, leading to redundant representations between branches.

**Goal**: The goal is to decouple region selection from feature representation, allowing each part branch to be semantically anchored to a fixed part, while reducing multi-branch redundancy and constraining "inter-class differences" to real-world semantic hierarchies.

**Key Insight**: The authors found that ClearCLIP (a training-free open-vocabulary segmentation variant that removes CLIP residual connections, enables self-attention, and removes FFN) is effective not only for object-level segmentation but also for **part-level semantic concepts**. Thus, part regions can be selected directly and controllably using text prompts (e.g., "engine", "landing gear", "head").

**Core Idea**: Use text-controllable ClearCLIP part masks for "semantic-guided region selection." This is combined with decoupled differential features via Hadamard product to obtain multi-scale multi-part features. These are then converged to true semantic levels through progressive reasoning and a vision-language contrastive loss that introduces "intermediate-granularity categories."

## Method

### Overall Architecture
PSCL (Part-level Semantic-guided Contrastive Learning) is an FGVC framework with dual "vision + text" pathways. In the vision pathway, the input image is simultaneously fed into the backbone and ClearCLIP. The backbone produces multi-scale differential features, while ClearCLIP uses text prompts to calculate matching scores and generate part masks via channel selection. The two are combined via Hadamard product to form multi-scale multi-part features within the Part Localization Module (PLM). These features enter the Multi-scale Multi-part Progressive Reasoning (MMBPR) module, which enhances prediction confidence level-by-level from low to high layers, with part branches capturing local features and the global branch aggregating them based on spatial relationships. In the text pathway, multi-granularity text (coarse, intermediate, and fine) corresponding to each fine-grained label is encoded by the ClearCLIP text encoder and rearranged as prototypes for contrastive learning, constraining visual features to align with real semantic hierarchies (VLCL-MG). During training, three components are jointly optimized using a Focal-Smooth contrastive loss; during inference, only the global branch is retained, discarding ClearCLIP and redundant part branches to significantly accelerate processing.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Image"] --> P["1. Part Localization Module (PLM)<br/>ClearCLIP Text Mask ⊙ Differential Features<br/>Includes ReSAF to suppress scale redundancy"]
    P --> M["2. Multi-part Progressive Reasoning (MMBPR)<br/>Part branches + global branch step-by-step confidence enhancement"]
    M --> V["3. Multi-granularity Textive Contrastive (VLCL-MG)<br/>Intermediate category constraints + multi-granularity text prototypes"]
    V -->|Training| L["Focal-Smooth Contrastive Loss<br/>Stage-wise weighted summation"]
    M -->|Inference (Global Branch Only)| O["Fine-grained Category Prediction"]
```

### Key Designs

**1. Part Localization Module (PLM): Decoupling "Region Selection" and "Region Representation" via Text-controllable ClearCLIP**

This design addresses the pain point where part and spatial representations compete for features and shared extractors lead to weak grounding. PLM splits the input $x \in \mathbb{R}^{C\times H\times W}$ into two independent branches: one for **differential features** (backbone produces multi-scale $f_s \in \mathbb{R}^{C_s\times H_s\times W_s}$; lower features can be skipped if unhelpful for classification, using $s\in\{s_{min},\dots,4\}$), and the other for **part localization** (ClearCLIP). In the localization branch, the image yields patch-level features $F_{img}$ via an image encoder, and $N$ part text prompts yield $F_{text}$ via a text encoder. A similarity tensor $S = F_{img}F_{text}^\top \in \mathbb{R}^{H\times W\times N}$ is computed. An argmax over $N$ channels generates one-hot masks, refined by morphology $M = (S\oplus K)\ominus K$ (dilation then erosion with a $3\times3$ kernel) to denoise connectivity. The final features are $G_{s,n'} = \text{concat}(f'_s\odot M_{s,n},\; f'_s\odot \mathbf{1})$, where $\mathbf{1}$ is a global mask of ones for global features. This anchors each branch to a predefined part; occluded parts can be "deactivated" via masks, stabilizing optimization.

PLM also embeds **ReSAF (Reverse-key Scale-aware Attention Fusion)** to suppress inter-scale redundancy. It reverses the key vector direction to negate similarity scores, guiding high-level queries to "avoid" regions semantically similar to low-level features and instead extract complementary information. Ablations show ReSAF (95.14%) outperforms MLP (94.71%) and standard Cross-Attention (94.99%).

**2. Multi-scale Multi-part Progressive Reasoning (MMBPR): Confidence Enhancement without Lower-level Interference**

To address branch redundancy caused by shared extractors and homogeneous parts, MMBPR extends the multi-scale progressive framework of PMG/PART. Following the ViT architecture, each branch uses 3 learnable class tokens for intermediate representation. Reasoning starts from the lowest level $G_{s_{min},n'}$. Features are flattened into tokens and concatenated with class tokens as $Z$. After passing through Encoder layers (MHSA + MLP + LayerNorm, with **non-shared weights across layers**), the output is split: class tokens $I_{s,n'}$ are sent to VLCL-MG for contrast, while feature tokens are **forwarded to the next stage** to be concatenated with higher-level flattened features. This "class tokens do not pollute higher levels" design ensures high-level branches gain stronger discriminative power.

**3. Vision-Language Contrastive Learning - Multi-Grain (VLCL-MG): Constraining Differences via Intermediate Categories**

To prevent model outputs from unnecessarily amplifying inter-class distances, VLCL-MG introduces **intermediate-granularity categories**. For example, between the coarse category *airplane* and the fine category *Boeing 737-200*, intermediate levels like *narrow-body airliner* and *twinjet* are inserted. Multi-granularity labels $t_{cls}=\{a_{n_A}, b_{n_B}, f_{n_F}\}$ are encoded, and the coarse features are **subtracted before normalization**: $T_{n_F}=\text{norm}(t_{n_F}-f_{text}(coarse))$. This prevents embeddings from clustering, making them more discriminative. Prediction probabilities are given by: $P_{s,n',c}=\sigma(\tau\odot(WI_{s,n'})T_{n_F}^\top + \beta)_c$.

The **Focal-Smooth Contrastive Loss** combines label smoothing and focal loss: $$\text{FSL}_s = -\sum_{n'}\sum_c (1-P_{s,n',c})^\gamma\, \tilde{y}_{s,n',c}\log P_{s,n',c}$$. The smoothing noise $\epsilon_s$ **decreases** as stage $s$ increases (higher stages are more confident). The final loss is $L_{final}=\sum_{s=s_{min}}^{4}\tilde{\epsilon}_s\cdot\text{FSL}_s$.

### Loss & Training
Training uses AdamW, 100 epochs, batch size 16, weight decay 0.01. Learning rates: $1\times10^{-4}$ for RN50, $1\times10^{-5}$ for ViT-B/Swin-B. 10-epoch warmup + cosine annealing. Focal factor $\gamma=4$, smoothing noise $\epsilon_s=[0.4,0.3,0.2,0.1]$, loss weights $\tilde{\epsilon}_s=[0.1,0.2,0.4,1.0]$. Input resolutions: 448 (RN50), 384 (Swin-B), 518 (ViT-B).

## Key Experimental Results

### Main Results
Comparison with SOTA across five datasets (Accuracy %):

| Backbone | Method | AIR | CAR | CUB | NAB | DOG |
|----------|------|-----|-----|-----|-----|-----|
| RN50 | SIA-Net | 94.3 | 95.5 | **90.7** | – | – |
| RN50 | **Ours** | **95.1** | **95.6** | 89.1 | **89.0** | **90.1** |
| ViT-B(448) | ACC-ViT | – | 94.9 | 91.8 | 91.4 | **92.9** |
| ViT-B(448) | **Ours** | 94.3 | 95.1 | **92.2** | **92.5** | 91.0 |
| ViT-B(518) | **Ours** | **96.5** | **96.4** | **92.3** | **93.7** | 92.3 |
| Swin-B | CSQA-Net | 94.7 | **95.6** | 92.6 | 92.3 | – |
| Swin-B | **Ours** | **95.3** | 95.5 | **93.0** | **93.8** | **94.7** |

Improvements are particularly significant with Transformer backbones; ViT-B(518) reaches 96.5% on AIR. Strong performance on non-rigid datasets (CUB/NAB) confirms better modeling of non-rigid characteristics.

### Ablation Study
Module-wise ablation (RN50):

| PLM | MMBPR | VLCL-MG | CUB | AIR | CAR | Note |
|-----|-------|---------|-----|-----|-----|------|
| ✗ | ✗ | ✗ | 85.09 | 91.56 | 91.90 | baseline |
| ✓ | ✗ | ✗ | 88.82 | 94.54 | 95.46 | Localization helps significantly |
| ✗ | ✗ | ✓ | 87.90 | 94.39 | 95.32 | Multi-grain contrast helps |
| ✓ | ✓ | ✗ | 89.09 | 94.54 | 95.54 | Progressive reasoning added |
| ✓ | ✓ | ✓ | **89.13** | **95.14** | **95.59** | Full model |

### Key Findings
- **PLM and VLCL-MG alone provide large gains**, but their combination shows diminishing returns because similar sub-classes often share similar part structures; both modules address similar underlying issues.
- **Smoothing noise $\epsilon_s$ must decay at higher stages**: 0 smoothing yields 89.92%, while $[0.4, 0.3, 0.2, 0.1]$ yields 95.14%.
- **Efficiency**: ClearCLIP is ~17.35 GFLOPs. During inference, ClearCLIP and redundant branches are discarded, making overhead comparable to standard Transformers.

## Highlights & Insights
- **Downscaling open-vocabulary segmentation to the part level**: The authors empirically demonstrate ClearCLIP's effectiveness for part semantic concepts, enabling controllable region selection and decoupling it from representation learning.
- **"Intermediate-granularity categories" as zero-cost prior**: Simple retrieval aligns geometric constraints with real taxonomic hierarchies, and the subtraction trick prevents feature collapse in text embeddings.
- **Training-heavy, Inference-light**: Semantic guidance via ClearCLIP and multi-part branches is used for training, while only the global branch is kept for inference, balancing accuracy and speed.

## Limitations & Future Work
- Hyperparameters (except learning rate) were tuned only on AIR + RN50 and then migrated. Dataset-specific tuning might further improve results.
- PLM and VLCL-MG have functional overlap, showing diminishing marginal returns.
- Part text prompts require manual definition; scalability to new domains and the impact of noise from ChatGPT-4o generated categories need further evaluation.

## Related Work & Insights
- **vs PART / Saliency methods**: Those rely on CAM/saliency, which can be inconsistent across samples. PSCL uses text + ClearCLIP for semantically grounded part selection.
- **vs MP-FGVC**: MP-FGVC uses multimodal prompts at the object level; PSCL moves this to part-level localization and adds intermediate category constraints.
- **vs PMG / TransFG**: PSCL reuses multi-scale progressive reasoning but adds ReSAF to suppress scale redundancy and resolves the "part vs spatial feature" conflict.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Few-way to Many-way: Rethinking Few-shot Fine-grained Image Classification](../../CVPR2026/self_supervised/from_few-way_to_many-way_rethinking_few-shot_fine-grained_image_classification.md)
- [\[CVPR 2026\] Nonparametric Deep Fine-grained Clustering with Low-Rank Guided Vision-Language Model](../../CVPR2026/self_supervised/nonparametric_deep_fine-grained_clustering_with_low-rank_guided_vision-language_.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](../../ICML2026/self_supervised/partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[AAAI 2026\] FineXtrol: Controllable Motion Generation via Fine-Grained Text](../../AAAI2026/self_supervised/finextrol_controllable_motion_generation_via_fine-grained_text.md)
- [\[CVPR 2026\] Learning from Semantic Dictionaries: Discriminative Codebook Contrastive Learning for Unified Visual Representation and Generation](../../CVPR2026/self_supervised/learning_from_semantic_dictionaries_discriminative_codebook_contrastive_learning.md)

</div>

<!-- RELATED:END -->
</div>

## Related Papers

- [\[CVPR 2026\] From Few-way to Many-way: Rethinking Few-shot Fine-grained Image Classification](../../CVPR2026/self_supervised/from_few-way_to_many-way_rethinking_few-shot_fine-grained_image_classification.md)
- [\[CVPR 2026\] Nonparametric Deep Fine-grained Clustering with Low-Rank Guided Vision-Language Model](../../CVPR2026/self_supervised/nonparametric_deep_fine-grained_clustering_with_low-rank_guided_vision-language_.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](../../ICML2026/self_supervised/partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[AAAI 2026\] FineXtrol: Controllable Motion Generation via Fine-Grained Text](../../AAAI2026/self_supervised/finextrol_controllable_motion_generation_via_fine-grained_text.md)
- [\[CVPR 2026\] Learning from Semantic Dictionaries: Discriminative Codebook Contrastive Learning for Unified Visual Representation and Generation](../../CVPR2026/self_supervised/learning_from_semantic_dictionaries_discriminative_codebook_contrastive_learning.md)

</div>

<!-- RELATED:END -->
