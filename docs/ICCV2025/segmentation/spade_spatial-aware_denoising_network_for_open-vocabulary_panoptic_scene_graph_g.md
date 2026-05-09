---
title: >-
  [Paper Note] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation
description: >-
  [ICCV 2025][Segmentation][Panoptic Scene Graph Generation] This paper proposes SPADE — a spatial-aware denoising network for open-vocabulary panoptic scene graph generation (PSG). It adapts a pretrained diffusion model into a PSG-specific spatial prior extractor via DDIM inversion-guided calibration, and designs a relational graph Transformer to capture both long-range and local context. SPADE substantially outperforms prior state-of-the-art methods in both closed-set and open-set settings, with particularly strong performance on spatial relation prediction.
tags:
  - ICCV 2025
  - Segmentation
  - Panoptic Scene Graph Generation
  - Open-Vocabulary
  - Diffusion Models
  - Spatial Relation Reasoning
  - Graph Transformer
date: 2026-05-08
content_hash: dd1baefa2c791fdf
---

# SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation

**Conference**: ICCV 2025
**arXiv**: [2507.05798](https://arxiv.org/abs/2507.05798)
**Code**: None (mentioned on project page)
**Area**: Image Segmentation
**Keywords**: Panoptic Scene Graph Generation, Open-Vocabulary, Diffusion Models, Spatial Relation Reasoning, Graph Transformer

## TL;DR

This paper proposes SPADE — a spatial-aware denoising network for open-vocabulary panoptic scene graph generation (PSG). It adapts a pretrained diffusion model into a PSG-specific spatial prior extractor via DDIM inversion-guided calibration, and designs a relational graph Transformer to capture both long-range and local context. SPADE substantially outperforms prior state-of-the-art methods in both closed-set and open-set settings, with particularly strong performance on spatial relation prediction.

## Background & Motivation

Panoptic scene graph generation (PSG) unifies instance segmentation and relation understanding into subject-predicate-object triplets. While VLM-based open-vocabulary methods have made notable progress, a critical and largely overlooked issue remains:

**Spatial Reasoning Deficiency in VLMs**: Multiple studies have shown that VLMs such as CLIP, BLIP, and GLIP suffer from an inherent weakness in spatial relation understanding — due to the scarcity of spatial descriptions in their training data — making it difficult for these models to determine relations such as "left of / right of / above / below."

**Distance Sensitivity**: Through systematic experiments, the authors find that when two objects are far apart (center distance > 1/3 image width), VLM-based models exhibit a sharp drop in spatial relation prediction performance (e.g., OpenPSG R@50 drops from 43.7 to 37.1).

**Lack of Contextual Reasoning**: Existing methods focus primarily on designing visual prompts to extract VLM knowledge, while neglecting spatial and semantic contextual information among relation pairs.

**Limitations of Directly Using Diffusion Models**: Although diffusion models possess strong spatial compositional capability, their pretrained knowledge is not optimized for the PSG task, making direct application ineffective.

**Core Motivation**: Can the spatial knowledge of diffusion models be injected into VLMs without compromising their inherent open-world recognition capability?

## Method

### Overall Architecture

SPADE is a two-stage approach:
- **Stage 1**: Inversion-guided Calibration — adapts a pretrained diffusion model into a PSG-specific denoising network.
- **Stage 2**: Spatial-aware Context Reasoning — generates high-quality relation queries via a relational graph Transformer.

### Key Designs

1. **Inversion-guided Calibration**:

    - **Inverse Spatial Prior Extraction**: Real images are converted into deterministic noise $z$ via the DDIM inversion process; a teacher diffusion model (BELM) then performs deterministic sampling conditioned on relation prompts "[subject] is [predicate] [object]..." to obtain cross-attention maps $A_i$ as spatial priors.
    - **Implicit Text Encoder**: Since no textual description is available at inference time, the text encoder is replaced by a CLIP image encoder with an MLP adapter: $f_i = \epsilon_\phi(x_i, \mathrm{MLP} \circ \mathrm{CLIP_{image}}(x_i))$.
    - **LoRA Calibration**: Only the low-rank matrices of the UNet cross-attention layers are updated, $\Delta\mathbf{W}_k = \mathbf{B} \times \mathbf{D}$ ($\mathbf{B} \in \mathbb{R}^{m_{in} \times r}$, $\mathbf{D} \in \mathbb{R}^{r \times m_{out}}$), preserving pretrained knowledge.
    - **Calibration Loss**: $\mathcal{L}_{cal} = \frac{1}{N}\sum_{i=1}^{N}(\lambda\|A_i - A_i'\|_1)$, aligning the cross-attention maps computed on real images with those derived from the inversion process.
    - **Design Motivation**: The DDIM inversion process inherently preserves the spatial structure of the input image; LoRA fine-tuning injects PSG-specific spatial knowledge while maximally retaining the diffusion model's world knowledge.

2. **Spatial-aware Relational Graph Transformer (RGT)**:

    - **Spatial-Semantic Graph Construction**: A graph $G \in \mathbb{R}^{N \times N}$ is constructed based on spatial distance between instance masks (adjacent = 1) and feature cosine similarity (above threshold = 1).
    - **Long-range Context Learning** ($\mathrm{RGT_g}$): Self-attention is computed separately over neighbors $\mathcal{P}(r)^+$ and non-neighbors $\mathcal{P}(r)^-$, then fused: $\mathbf{q}_r \leftarrow \mathbf{q}_r + \mathrm{RGT}(\mathbf{q}_r)_{\mathcal{P}^+} + \mathrm{RGT}(\mathbf{q}_r)_{\mathcal{P}^-}$, followed by MLP-based feature fusion.
    - **Local Context Learning** ($\mathrm{RGT_l}$): A GCN aggregates local neighborhood information over graph $G$: $\hat{\mathbf{q}}_r = \mathrm{GCN}(G, \mathbf{q}_r'; \mathbf{W}_l)$.
    - **Relation Query Construction**: Relation queries $\Psi_r$ are constructed by selecting similar object pairs based on cosine distance; an auxiliary loss $\mathcal{L}_{rqc}$ optimizes selection quality.
    - **Design Motivation**: Modeling only connected object pairs is insufficient — non-adjacent objects may also participate in relations (e.g., "a person far away looking at an airplane"). The dual long-range and local reasoning covers relational contexts at different spatial scales.

3. **Open-vocabulary Relation Prediction**:

    - CLIP text encoders encode category/predicate templates, and classification is performed via feature similarity.
    - Dual-path prediction fusion: diffusion feature scores $\mathbf{P}_o$ and CLIP-pooled feature scores $\mathbf{P}_o'$ are combined via geometric mean: $\mathbf{P}^o_{\text{final}} = \mathbf{P}_o^\alpha \cdot \mathbf{P}_o'^{(1-\alpha)}$.
    - Relation prediction follows the same scheme, using joint masks of subject and object for pooling.
    - **Design Motivation**: Diffusion models excel at spatial reasoning but have limited open-vocabulary capability, while CLIP excels at open-world recognition but is weak in spatial reasoning — complementary fusion exploits the advantages of each.

### Loss & Training

- Stage 1 (calibration): Only $\mathcal{L}_{cal}$ (L1 alignment of cross-attention maps); MLP adapter and LoRA parameters are updated.
- Stage 2: $\mathcal{L} = \mathcal{L}_{\text{rel}} + \lambda_{\text{rqc}}\mathcal{L}_{rqc} + \lambda_{\text{mask}}L_{\text{mask}}$
- Hyperparameters: $\alpha=0.34$, $\eta=0.65$, $\lambda_{rqc}=0.6$, $\lambda_{mask}=1$
- Diffusion model and CLIP parameters are frozen during Stage 2.
- Total training: 80 epochs, with learning rate decay at epoch 60.
- Training hardware: 4× A100 GPUs.

## Key Experimental Results

### Main Results

| Dataset / Setting | Metric | SPADE | OpenPSG | OvSGTR | Gain |
|---|---|---|---|---|---|
| PSG Closed-set | R/mR@100 | **54.3/51.7** | 49.3/47.5 | 41.4/28.3 | +5.0/+4.2 |
| PSG Open-set (OvR) | R/mR@50 | **26.7/23.3** | 21.2/19.8 | 19.3/12.4 | +5.5/+3.5 |
| PSG Open-set (OvR) | R/mR@100 | **31.8/25.8** | 25.1/21.4 | 22.8/14.0 | +6.7/+4.4 |
| VG Open-set (OvR) | R/mR@100 | **29.9/13.9** | 25.7/12.1 | 26.7/5.7 | +4.2/+1.8 |
| PSG Open-set (OvD+R) | R@50 | **22.7** | 11.4 | 19.1 | +3.6 |

### Ablation Study

| Configuration | R/mR@50 (PSG Closed-set) | Note |
|---|---|---|
| w/o RGT components | 30.5/26.3 | Baseline |
| + Long-range Neighbor Learning (LCNL) | 35.6/32.2 | Neighbor context yields significant improvement |
| + Long-range Non-neighbor (LCNNL) | 37.1/34.4 | Non-neighbor relations also contribute |
| + Local Context Learning (LCL) | 40.3/35.6 | GCN local reasoning complements long-range |
| + All + $\mathcal{L}_{rqc}$ | **45.1/41.2** | Auxiliary loss further improves performance |

| Calibration Strategy | OvR R@50 | OvD+R R@50 | Note |
|---|---|---|---|
| No calibration (pretrained UNet directly) | 15.3 | 10.1 | Pretrained knowledge mismatched with PSG |
| No LoRA (full fine-tuning) | 18.8 | 12.7 | Destroys pretrained knowledge |
| No inversion (random noise) | 21.0 | 15.9 | Lacks spatial structure prior |
| Full method | **26.7** | **22.7** | Inversion + LoRA is optimal |

### Key Findings

- **Systematic Analysis of Spatial Relations**: SPADE achieves R@50 of 42.3 on distant relations (DR), compared to 37.1 for OpenPSG, with the performance gap narrowing from 6.6 to 4.2 — demonstrating that SPADE effectively improves long-range spatial reasoning.
- **Open-vocabulary Module Ablation**: Fusing diffusion features with pooled CLIP features outperforms either branch alone (26.7 vs. 12.5 vs. 21.4).
- **Inversion Process is Critical**: Random noise sampling (21.0) is substantially inferior to deterministic inversion (26.7), as the inversion process preserves the spatial layout of the original image.

## Highlights & Insights

- **Depth of Problem Identification**: The paper systematically uncovers the spatial reasoning deficiency of VLMs and quantifies the effect of object distance on performance.
- **Creative Use of the Diffusion Inversion Process**: DDIM inversion inherently preserves spatial structure — this property is elegantly repurposed as a spatial prior for PSG.
- **Dual Long-range and Local Context Reasoning**: Separate modeling of neighbor and non-neighbor relations combined with GCN local aggregation comprehensively covers relational contexts at different spatial scales.
- **Complementary Diffusion + Discriminative Dual-path Classification**: Diffusion models handle spatial reasoning while CLIP handles open-world recognition, with complementary fusion leveraging the strengths of each.
- **Necessity of LoRA Calibration**: Full fine-tuning underperforms LoRA, validating the importance of preserving pretrained knowledge.

## Limitations & Future Work

- The two-stage training pipeline is relatively complex, requiring independent UNet calibration before RGT training.
- The computational cost of diffusion model inference is non-trivial; inference speed is not reported.
- Spatial-semantic graph construction relies on fixed thresholds, lacking flexibility.
- Evaluation is limited to two datasets (PSG and VG); broader scene graph generation benchmarks are not explored.
- The relation prompt design ("[subject] is [predicate] [object]") may constrain the types of relations that can be captured.

## Related Work & Insights

- Compared to VLM-based methods such as OpenPSG and OvSGTR, SPADE is the first to introduce diffusion models to enhance spatial reasoning in PSG.
- The LoRA calibration strategy (fine-tuning only low-rank matrices of cross-attention layers) is generalizable to other downstream tasks that require adapting diffusion models.
- The idea of separately reasoning over neighbors and non-neighbors in the relational graph Transformer can be applied to other structured prediction tasks such as HOI detection.
- Unlike DiffusionSG and related work, SPADE leverages cross-attention maps from the inversion process rather than the generative capability of diffusion models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Identifies VLM spatial reasoning deficiencies and innovatively exploits the diffusion inversion process to supply spatial priors.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers closed-set, open-set, spatial relation analysis, and multi-dimensional ablations; efficiency analysis is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, methodology is systematically described, and figures/tables are informative.
- **Value**: ⭐⭐⭐⭐ Reveals a core limitation of VLMs in spatial reasoning and establishes a new paradigm for diffusion model + VLM integration.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_panoptic_scene_graph_realtime.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [\[ICCV 2025\] Auto-Vocabulary Semantic Segmentation](auto-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)

<!-- RELATED:END -->
