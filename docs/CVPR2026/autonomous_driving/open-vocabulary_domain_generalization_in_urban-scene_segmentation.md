---
title: >-
  [Paper Note] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation
description: >-
  [CVPR 2026][Autonomous Driving][Open-vocabulary segmentation] This paper proposes OVDG-SS, a new problem setting that unifies unseen-domain and unseen-category challenges in semantic segmentation, and introduces S2-Corr, a state space model-based module that repairs text-image correlation degradation caused by domain shift, enabling efficient and robust cross-domain open-vocabulary segmentation in autonomous driving scenarios.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Open-vocabulary segmentation
  - domain generalization
  - state space models
  - text-image correlation
  - urban scene segmentation
date: 2026-05-08
content_hash: 9800010924f21dca
---

# Open-Vocabulary Domain Generalization in Urban-Scene Segmentation

**Conference**: CVPR 2026
**arXiv**: [2602.18853](https://arxiv.org/abs/2602.18853)
**Code**: [DZhaoXd/s2_corr](https://github.com/DZhaoXd/s2_corr)
**Area**: Autonomous Driving
**Keywords**: Open-vocabulary segmentation, domain generalization, state space models, text-image correlation, urban scene segmentation

## TL;DR

This paper proposes OVDG-SS, a new problem setting that unifies unseen-domain and unseen-category challenges in semantic segmentation, and introduces S2-Corr, a state space model-based module that repairs text-image correlation degradation caused by domain shift, enabling efficient and robust cross-domain open-vocabulary segmentation in autonomous driving scenarios.

## Background & Motivation

**DG-SS is limited to closed-set recognition**: Although conventional domain generalization semantic segmentation (DG-SS) methods improve cross-domain robustness, they can only recognize a fixed set of categories seen during training and cannot handle novel semantics in open-world environments (e.g., road barriers or traffic cones appearing at night).

**OV-SS is sensitive to domain shift**: Existing open-vocabulary segmentation (OV-SS) models (e.g., CAT-Seg, MaskAdapter) trained on COCO-Stuff can recognize a broad range of concepts, but suffer dramatic performance degradation when transferred to driving scenes—mIoU drops significantly even when category sets overlap (as shown in Table 1, CAT-Seg trained on COCO achieves only 31.6% on Dv-19, compared to 49.3% after training on Cityscapes).

**The two capabilities have not been unified**: DG-SS handles domain shift but cannot recognize novel categories; OV-SS recognizes novel categories but is not robust to domain shift. Autonomous driving requires both—models must simultaneously adapt to unseen domains (adverse weather, different regions) and recognize objects absent from training data.

**Absence of evaluation benchmarks**: No benchmark previously existed that simultaneously covers unseen domains and unseen categories in driving-scene segmentation, making systematic evaluation of OVDG-SS capability infeasible.

**Domain shift corrupts VLM correlations**: Empirical analysis reveals that domain shift causes text-image correlation maps in pretrained VLMs to become noisy and misaligned (as illustrated in Fig. 3, the correlation for the "sky" class spreads to irrelevant regions as domain shift increases), which is the root cause of OV-SS failure under OVDG settings.

**Cross-attention propagates noise**: CAT-Seg uses cross-attention for correlation aggregation; under domain shift, corrupted correlations enter attention computation as noisy keys/values, and errors are progressively amplified along both spatial and category dimensions.

## Method

### Overall Architecture

S2-Corr is built upon CAT-Seg's correlation aggregation pipeline. Given an image-text pair, visual features $\mathbf{F}_v \in \mathbb{R}^{HW \times d}$ and text category embeddings $\mathbf{F}_t \in \mathbb{R}^{N_C \times d}$ are extracted via CLIP (EVA02), and an initial correlation map $\mathbf{C} = \text{Norm}(\mathbf{F}_v \mathbf{F}_t^\top)$ is computed. The correlations are then projected into a $d_f$-dimensional embedding space via learnable projections, followed by two sequential repair stages: **spatial aggregation** and **category aggregation**. The core innovation replaces the original cross-attention aggregation with selective state space models (SSMs), augmented by three additional design components.

### Key Designs

**Design 1: Modulation Before Aggregation**

- **Spatial modulation**: Prior to spatial aggregation, image features $\mathbf{F}_{\pi(t)}$ are linearly projected to produce modulation factors $(\gamma, \beta)$, which apply an affine transformation $\hat{\mathbf{E}} = \mathbf{E} \odot (1 + \gamma) + \beta$ to the correlation embeddings, injecting domain-relevant visual cues.
- **Category modulation**: Prior to category aggregation, multi-domain text prompt templates (e.g., "a photo of {class} at night," "in the rain," across 10 variants) are used to encode domain-aware text features $\mathbf{t}^{(d)}$, generating modulation vectors that perform domain-adaptive adjustment of category embeddings.

**Design 2: Learnable Geometric Decay Prior**

- Dynamic gates $\mathbf{A}_t$ may still propagate long-range noise under domain shift; a geometric decay prior $\boldsymbol{\gamma} \in (0,1)^K$ is introduced.
- Effective decay coefficient: $\mathbf{A}_t^{\text{eff}} = \sigma(\mathbf{w}) \cdot \sigma(\mathbf{W}_a \mathbf{x}_t + \mathbf{b}_a) + (1 - \sigma(\mathbf{w})) \cdot \boldsymbol{\gamma}$
- This preserves the geometric decay pattern $\|\partial \mathbf{h}_t / \partial \mathbf{h}_{t-d}\| \propto (\mathbf{A}_t^{\text{eff}})^d$ while keeping the decay rate learnable, effectively suppressing long-range noise propagation.

**Design 3: Chunk-wise Snake Scanning**

- The flattened sequence is divided into equal-length chunks by row (chunk count set to 16), with sequential updates within each chunk.
- A snake traversal is applied across rows (odd rows processed forward, even rows in reverse), eliminating spatial discontinuities at row boundaries.
- The terminal hidden state is passed between chunks: $\mathbf{h}_{k+1}^{\text{init}} \leftarrow \mathbf{h}_k^{\text{end}}$, maintaining spatial continuity.
- Compared to full-sequence sequential scanning, the chunk-wise design preserves high parallelism and substantially reduces computational overhead.

### Loss & Training

- Implemented with Detectron2 using the AdamW optimizer; learning rate $2 \times 10^{-4}$ for the aggregation module and $2 \times 10^{-6}$ for the EVA-CLIP encoder.
- Correlation embedding dimension 128; 2 spatial blocks + 2 upsampling stages; chunk count 16; decay prior $\gamma = 0.8$.
- Only selected attention projection layers are updated in the visual encoder; only projection weights within residual blocks are trained in the text encoder.
- With ViT-B/16, only 26M parameters are updated; with ViT-L/14, 76.8M parameters are updated.
- Batch size 4; 20k iterations; trained on a single RTX 3090 in 2 hours (ViT-B) / 4 hours (ViT-L).

## Key Experimental Results

### Main Results

**Real-to-Real OVDG-SS (trained on CS-7, Table 2):**

| Method | Backbone | Dv-19 Ave. | Dv-58 Ave. |
|--------|----------|-----------|-----------|
| CAT-Seg | ViT-B/16 | 43.5 | 43.5 |
| MaskAdapter | ViT-B/16 | 45.5 | 43.8 |
| CLIPSelf | ViT-B/16 | 45.7 | 45.0 |
| **S2-Corr** | **ViT-B/16** | **50.3** | **47.9** |
| CAT-Seg | ViT-L/14 | 49.3 | 50.0 |
| CLIPSelf | ViT-L/14 | 53.3 | 51.5 |
| **S2-Corr** | **ViT-L/14** | **55.8** | **53.2** |

**Synthetic-to-Real OVDG-SS (trained on GTA-7, Table 3):**

| Method | Backbone | Dv-19 Ave. | Dv-58 Ave. |
|--------|----------|-----------|-----------|
| CAT-Seg | ViT-B/16 | 43.9 | 45.6 |
| CLIPSelf | ViT-B/16 | 46.2 | 44.4 |
| **S2-Corr** | **ViT-B/16** | **48.2** | **46.7** |
| CAT-Seg | ViT-L/14 | 47.5 | 48.2 |
| **S2-Corr** | **ViT-L/14** | **49.9** | **49.4** |

### Ablation Study

**Incremental component ablation (CS-7 → Dv-19 / Dv-58, Table 4):**

| Design | ViT-B Dv-19 | ViT-B Dv-58 | ViT-L Dv-19 | ViT-L Dv-58 | Avg. |
|--------|------------|------------|------------|------------|------|
| Base (Cross-Attn) | 43.5 | 43.5 | 49.3 | 50.0 | 46.6 |
| +Selective SSM | 45.6 | 44.1 | 50.7 | 50.5 | 47.7 |
| +Modulation | 47.6 | 45.3 | 52.1 | 50.9 | 49.0 |
| +Geometric Decay | 48.3 | 46.4 | 53.2 | 51.8 | 49.9 |
| +Chunk | 49.6 | 47.3 | 55.3 | 52.7 | 51.2 |
| +Snake Scanning | **50.3** | **47.9** | **55.8** | **53.2** | **51.8** |

**Efficiency comparison (ViT-B/16, Table 5):**

| Method | FPS@19cls | FPS@58cls | FPS@150cls | GPU Memory | Training Time |
|--------|---------|---------|----------|---------|---------|
| CAT-Seg | 15.4 | 10.6 | 5.7 | 13.8 GB | 180 min |
| ESC-Net | 15.0 | 9.9 | 5.1 | 15.7 GB | 220 min |
| **S2-Corr** | **26.1** | **22.2** | **18.3** | **9.2 GB** | **140 min** |

### Key Findings

- Replacing cross-attention with SSM alone yields an average gain of +1.1 mIoU, validating that sequential aggregation outperforms windowed attention.
- Noise suppression components (geometric decay + chunk-wise mechanism) provide the largest gains, especially under the large-vocabulary Dv-58 setting.
- As the vocabulary size grows from 19 to 150, CAT-Seg's FPS drops from 15.4 to 5.7 (−63%), whereas S2-Corr's FPS drops only from 26.1 to 18.3 (−30%), demonstrating the scalability afforded by linear complexity.
- S2-Corr consistently outperforms all baselines across all 7 unseen target domains in both synthetic-to-real and real-to-real settings.

## Highlights & Insights

- **New problem formulation**: This work is the first to unify DG-SS and OV-SS into OVDG-SS, establishing a research setting that more faithfully reflects real-world autonomous driving requirements.
- **Systematic benchmark**: The first OVDG-SS driving benchmark is constructed, covering 7 target domains (adverse weather, different regions, construction scenes) and 58 extended categories, with both synthetic-to-real and real-to-real evaluation paradigms.
- **Root-cause-driven design**: The failure of OV-SS under domain shift is first analyzed at its root (correlation map noise + attention-propagated amplification), and solutions are then tailored accordingly, yielding a clear and principled design rationale.
- **Efficiency advantages**: At large vocabulary sizes, S2-Corr achieves 3.2× higher FPS than CAT-Seg, requires only 9.2 GB of GPU memory, and trains in 2 hours, making it highly practical.
- **Novel application of SSMs**: Applying state space models to text-image correlation repair is a novel contribution; the decay gating mechanism is naturally suited to suppressing noise propagation.

## Limitations & Future Work

- Training data uses only a 7-class subset of Cityscapes/GTA, resulting in a small base vocabulary; whether a larger training vocabulary affects method effectiveness remains unexplored.
- The extended categories in ACDC-41 and BDD-41 are generated via Stable Diffusion 2.1 inpainting, which may not faithfully represent the distribution of unseen objects in real scenes.
- Snake scanning is fixed to the row direction; complementary column-direction or multi-directional scanning strategies are not explored.
- The 10 domain-aware text prompt templates are manually designed; learnable prompt tuning is not investigated.
- Validation is conducted solely on EVA02-CLIP; other VLM backbones (e.g., SigLIP, InternVL) are not considered.

## Related Work & Insights

- **DG-SS**: Data augmentation approaches (AdvStyle, DGInStyle) and PEFT-based methods (adapter fine-tuning, parameter selection), all limited to closed-set recognition.
- **Training-free OV-SS**: ClearCLIP, ProxyCLIP, CLIP-DINOiser, etc., which require no training but are not robust to domain shift.
- **Training-based OV-SS**: CAT-Seg (correlation + cross-attention), MaskAdapter, ESC-Net, etc., trained on COCO but degrading significantly when transferred to driving domains.
- **OV-SS + DG combinations**: Naive combinations such as CAT-Seg+AdvStyle and CAT-Seg+DGInStyle are substantially outperformed by S2-Corr.
- **State space models**: Mamba/VMamba applied to vision tasks; this work is the first to employ SSMs for text-image correlation aggregation repair.

## Rating

- Novelty: ⭐⭐⭐⭐ (OVDG-SS is a meaningful new problem setting; the design motivation of S2-Corr is clear and the method is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 target domains, two training settings, two backbones, full ablation, efficiency analysis, and visualizations)
- Writing Quality: ⭐⭐⭐⭐ (The narrative structure—problem analysis → baseline establishment → incremental enhancement—is clear and well-organized)
- Value: ⭐⭐⭐⭐ (Both the benchmark and the method offer practical reference value for open-world perception in autonomous driving)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation](../../NeurIPS2025/autonomous_driving/leveraging_depth_and_language_for_open-vocabulary_domain-generalized_semantic_se.md)
- [\[CVPR 2026\] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](fedbprompt_federated_domain_generalization_person.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)

<!-- RELATED:END -->
