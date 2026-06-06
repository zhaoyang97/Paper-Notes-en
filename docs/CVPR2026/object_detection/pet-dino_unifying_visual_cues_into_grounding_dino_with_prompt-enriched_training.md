---
title: >-
  [Paper Note] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training
description: >-
  [CVPR 2026][Object Detection][Open-set object detection] PET-DINO builds a unified object detector supporting both text and visual prompts on top of Grounding DINO. It introduces an alignment-friendly visual prompt gener…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Open-set object detection"
  - "visual prompts"
  - "Grounding DINO"
  - "training strategy"
  - "prompt learning"
date: 2026-05-08
content_hash: 496d603e9c7a231f
---

# PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training

**Conference**: CVPR 2026  
**arXiv**: [2604.00503](https://arxiv.org/abs/2604.00503)  
**Code**: [https://fuweifuvtoo.github.io/pet-dino](https://fuweifuvtoo.github.io/pet-dino)  
**Area**: Object Detection / Open-Set Detection  
**Keywords**: Open-set object detection, visual prompts, Grounding DINO, training strategy, prompt learning

## TL;DR

PET-DINO builds a unified object detector supporting both text and visual prompts on top of Grounding DINO. It introduces an alignment-friendly visual prompt generation module (AFVPG) and two prompt-enriched training strategies (IBP and DMD), achieving competitive zero-shot detection performance with significantly less training data.

## Background & Motivation

**Background**: Open-set object detection (OSOD) aims to recognize novel categories unseen during training. Text-prompt methods (e.g., Grounding DINO, GLIP) achieve zero-shot generalization by aligning visual features with pretrained text encoders. Visual-prompt methods (e.g., T-Rex2, CP-DETR, YOLOE) use visual representations of target objects as prompts to complement the limitations of text prompts.

**Limitations of Prior Work**: (1) Text features often fail to effectively represent visual concepts in specialized domains or for complex objects, making accurate discrimination difficult; (2) long-tail categories lack sufficient image–text paired samples; (3) existing visual-prompt methods (T-Rex2, CP-DETR) rely on tightly coupled multimodal architectures and multi-stage training pipelines with long development cycles; (4) effective training strategies for data-driven OSOD models remain underexplored.

**Key Challenge**: Visual prompts inherently carry richer information than text descriptions, yet during training they are derived from the input image itself, limiting diversity — it is difficult to model cross-image and category-level global visual prompts, and offline pre-extraction during training is infeasible.

**Goal**: (1) Efficiently extend an advanced text-prompt detector with visual prompt capability, rather than building a multimodal system from scratch; (2) design the first large-scale training strategy for dual-modality prompt detectors, enabling the model to simultaneously simulate diverse real-world usage scenarios during training.

**Key Insight**: An "inheritance-based" strategy — starting from a pretrained Grounding DINO, adding only a visual prompt generation module while sharing existing text-branch parameters, thereby reducing the development cycle.

**Core Idea**: Graft a visual prompt module onto a text-pretrained detector and enhance zero-shot detection capability through intra-batch parallel prompting and a dynamic memory bank prompt-enrichment training strategy.

## Method

### Overall Architecture

PET-DINO supports two detection routes: a text-prompt route (inherited from Grounding DINO) and a visual-prompt route. Text inputs are encoded by a text encoder into embeddings, which interact with image features via a Feature Enhancer to produce text prompts. Input coordinates interact with enhanced image features via the AFVPG module to generate visual prompts. Both types of prompts guide the Query Selection Module to provide positional priors, initializing 900 queries that are iteratively refined through 6 layers of interactive decoders for final detection and classification. During training, only visual-prompt-related network modules are updated; the backbone and other modules are frozen.

### Key Designs

1. **Alignment-Friendly Visual Prompt Generation (AFVPG)**:

    - **Function**: Efficiently extracts visual prompt embeddings from enhanced image features.
    - **Mechanism**: Rather than extracting visual prompts directly from unenhanced backbone features (which performs poorly), AFVPG leverages features $x'_i$ enhanced by deformable self-attention and FFN within the Feature Enhancer. For each category, a learnable content embedding $C \in \mathbb{R}^{K \times D}$ (for $K$ prompt boxes) plus a universal carrier $C'$ are initialized; prompt box coordinates are encoded and used in multi-scale deformable cross-attention with enhanced image features, then aggregated via self-attention and FFN into a global visual prompt vector $V \in \mathbb{R}^{1 \times D}$. AFVPG shares the deformable self-attention and FFN parameters of the Feature Enhancer with the text branch.
    - **Design Motivation**: Using enhanced features better aligns visual prompts with the detector's internal instance representations; parameter sharing allows high-level semantic knowledge from the text branch to assist visual prompt learning. Compared to T-Rex2's encoder, AFVPG improves AP by 4.8 on Visual-I and 2.7 on Visual-G.

2. **Intra-Batch Parallel Prompting (IBP)**:

    - **Function**: Simulates multiple visual prompt usage scenarios in parallel during mini-batch training.
    - **Mechanism**: Exploiting intra-batch parallelism, visual prompts $V_c^j$ from other images in the same batch are used as cross-image prompts for category $c$ in the current image, and same-category prompts are aggregated into a category-level global prompt $V_c^{batch}$. This yields two prompt types per category $c$ on image $i$: $V_c^i$ (from the image itself, corresponding to the Exemplar-Guided Route) and $V_c^{batch}$ (aggregated within the batch, corresponding to the Global-Concept Route). When the current image does not contain category $c$, prompts from other images expand the category discrimination space.
    - **Design Motivation**: Single-image visual prompt diversity is limited and cannot simulate cross-image or category-level global prompt scenarios. IBP aligns training with multiple real-world usage scenarios (interactive, global-concept, cross-image), boosting Visual-G from 12.5 AP to 37.2 AP (+24.7).

3. **Dynamic Memory-Driven Prompting (DMD)**:

    - **Function**: Propagates and enriches visual prompts across training iterations.
    - **Mechanism**: A Visual Cues Bank is maintained — a FIFO queue of length $M=16$ per category — dynamically storing visual prompt embeddings extracted in previous iterations. At each training iteration, $d$ categories are randomly sampled, and their historical prompts are retrieved and aggregated as $V_c^{mem} = \frac{1}{M}\sum_{k=1}^M \tilde{V}_c^k$, serving as a third prompt type. Each category thus has three visual prompts: $\{V_c^i, V_c^{batch}, V_c^{mem}\}$.
    - **Design Motivation**: IBP's prompt diversity is still constrained by the current batch; DMD further expands the prompt distribution through cross-iteration dynamic memory. The bank also enables contrastive learning for rarely co-occurring categories. DMD provides an additional +3.1 AP on Visual-G on top of IBP.

### Loss & Training

The training objective is $\mathcal{L} = \mathcal{L}_{L1} + \mathcal{L}_{GIoU} + \mathcal{L}_{alignment}$, where alignment is Focal loss. A cyclic training strategy is adopted: 8 epochs of visual-prompt training alternated with 1 epoch of text-prompt training to prevent degradation of text capabilities. Only visual-prompt-related modules are trainable (learning rate 1e-4); the backbone is frozen. Training runs for 12 epochs, with the learning rate reduced by 10× at epochs 8 and 11.

## Key Experimental Results

### Main Results

Zero-shot interactive detection (Visual-I), Swin-T:

| Method | VLM sup. | Data | COCO AP | LVIS AP | ODinW35 AP |
|--------|----------|------|---------|---------|------------|
| T-Rex2 | CLIP | 3.1M | 56.6 | 59.3 | 37.7 |
| CP-DETR-T | CLIP | 3.3M | 61.8 | 64.1 | 41.0 |
| **PET-DINO** | **None** | **0.6M** | **64.0** | 61.8 | 38.8 |
| **PET-DINO** | **None** | **2.5M** | **64.3** | **64.5** | **48.3** |

Zero-shot generic detection (Visual-G), Swin-T:

| Method | Data | COCO AP | LVIS AP | ODinW35 AP |
|--------|------|---------|---------|------------|
| T-Rex2 | 3.1M | 38.8 | 37.4 | 23.6 |
| **PET-DINO** | 0.6M | **40.3** | 29.6 | 20.4 |
| **PET-DINO** | 2.5M | 38.4 | 31.5 | **25.5** |

Text prompt retention: PET-DINO Swin-L achieves 54.0 AP on COCO (+1.0 vs. Grounding DINO, +1.0 vs. MM-Grounding-DINO) and 39.3 AP on LVIS (+2.6).

### Ablation Study

| Configuration | COCO Visual-I | COCO Visual-G | COCO Text | Note |
|---------------|---------------|---------------|-----------|------|
| AFVPG only | 67.0 | 12.5 | 49.7 | Visual-G very low |
| AFVPG + DMD | 63.5 | 24.7 | 49.8 | +12.2 Visual-G |
| AFVPG + IBP | 63.2 | 37.2 | 49.6 | IBP contributes most |
| AFVPG + IBP + DMD | 64.0 | **40.3** | 49.8 | Full model |

AFVPG vs. T-Rex2 encoder: Visual-I +4.8 AP, Visual-G +2.7 AP.

Pretrained inheritance vs. training from scratch: Visual-G +7.6 AP (inheritance strategy).

### Key Findings

- **IBP is the key factor for Visual-G improvement**: 12.5 → 37.2 (+24.7 AP), demonstrating that single-image prompts alone cannot learn generalizable category-level representations.
- DMD further contributes +3.1 AP on top of IBP by introducing cross-iteration prompt diversity.
- The Visual-I drop (67.0 → 64.0) in exchange for a large Visual-G gain (12.5 → 40.3) is a reasonable trade-off — the model shifts from copying specific instances toward learning generalizable category patterns.
- Inheriting text pretraining outperforms training from scratch (Visual-G +7.6 AP); global high-level semantic representations aid category concept understanding.
- PET-DINO with only 0.6M data surpasses CP-DETR (3.3M data) on COCO Visual-I.

## Highlights & Insights

- **"Inheritance-based" philosophy**: Grafting visual prompt capability onto a pretrained text detector rather than building a multimodal system from scratch. This not only reduces the development cycle but also leverages semantic knowledge from the text branch to enhance visual prompts — a highly practical engineering approach.
- **IBP cleverly exploits intra-batch parallelism**: At zero cost, it constructs three prompt scenarios (cross-image, category-level, interactive) within a single training pass, covering multiple real-world deployment scenarios simultaneously. This idea generalizes to any model requiring multiple prompt modes.
- **Quantitatively demonstrates the transfer value of text pretraining for visual prompts** (+7.6 AP), challenging the common assumption that visual prompting requires independent training.

## Limitations & Future Work

- Visual-G performance on LVIS still lags behind T-Rex2 (35.7 vs. 47.6, Swin-L); visual prompt generalization for long-tail categories remains challenging.
- No VLM supervision (e.g., CLIP) is used — integrating CLIP features may further improve cross-domain generalization.
- The cyclic training schedule (8:1 visual:text) is manually configured; adaptive scheduling may be more effective.
- The Visual Cues Bank queue length $M=16$ and sampling count $d=40$ require dataset-specific tuning and lack an adaptive mechanism.
- Joint usage of visual and text prompts (e.g., providing both text descriptions and example images simultaneously) is not explored.

## Related Work & Insights

- **vs. T-Rex2**: T-Rex2 is a closed (non-open-vocabulary) dual-modality prompt detector requiring CLIP supervision and large-scale data such as SA-1B. PET-DINO openly supports text prompts, requires no CLIP supervision, uses more than 5× less data, and still achieves superior COCO Visual-I performance. However, T-Rex2 leads substantially on LVIS Visual-G, indicating that large-scale data and CLIP remain advantageous for long-tail categories.
- **vs. CP-DETR**: CP-DETR uses tightly coupled early fusion; PET-DINO's inheritance strategy is more flexible. CP-DETR Swin-L achieves 71.6 AP on LVIS but uses the COCO training set (non-zero-shot).
- **vs. YOLOE**: YOLOE targets real-time scenarios using MobileCLIP; PET-DINO focuses more on accuracy upper bounds.

## Rating

- Novelty: ⭐⭐⭐⭐ The inheritance strategy is a novel approach; IBP/DMD training strategies represent the first systematic exploration in this direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three evaluation protocols, multiple benchmarks, comprehensive ablations, and pretraining analysis; however, fairer comparisons with CLIP-supervised methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, though some descriptions are verbose and could be more concise.
- Value: ⭐⭐⭐⭐ Provides valuable reference for training strategies of dual-modality prompt detectors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[CVPR 2026\] Prompt-Free Universal Region Proposal Network](prompt-free_universal_region_proposal_network.md)

</div>

<!-- RELATED:END -->
