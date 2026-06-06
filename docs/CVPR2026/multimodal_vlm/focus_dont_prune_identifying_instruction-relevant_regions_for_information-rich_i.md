---
title: >-
  [Paper Note] PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding
description: >-
  [CVPR 2026][Multimodal VLM][Large Vision-Language Models] This paper proposes PinPoint, a two-stage framework that first localizes instruction-relevant image regions via Instruction-Region Alignment…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Large Vision-Language Models"
  - "Token Efficiency"
  - "Region Selection"
  - "Contrastive Learning"
  - "Document Understanding"
date: 2026-05-08
content_hash: ff43d87d61f59833
---

# PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.22815](https://arxiv.org/abs/2603.22815)  
**Code**: [GitHub](https://github.com/minckwon/PinPoint)  
**Area**: Multimodal / VLM
**Keywords**: Large Vision-Language Models, Token Efficiency, Region Selection, Contrastive Learning, Document Understanding

## TL;DR

This paper proposes PinPoint, a two-stage framework that first localizes instruction-relevant image regions via Instruction-Region Alignment, then re-encodes the selected regions at fine granularity, achieving higher VQA accuracy with fewer visual tokens.

## Background & Motivation

**Background**: LVLMs (e.g., LLaVA-NeXT, Qwen2-VL) have achieved significant progress on multimodal tasks through high-resolution inputs, but processing information-rich images (e.g., infographics, document layouts) requires a large number of visual tokens, incurring substantial computational overhead.

**Limitations of Prior Work**: Token pruning methods (FastV, PyramidDrop, SparseVLM) prune unimportant tokens based on attention weights from LLM decoding layers, suffering from three key issues:
   - Attention maps are unreliable and may induce hallucinations
   - Semantic fragmentation — visual elements (e.g., text) span multiple tokens, and token-level pruning disrupts semantic integrity
   - Context entanglement — global self-attention entangles tokens from relevant and irrelevant regions

**Key Challenge**: High resolution is needed to capture fine-grained information, yet computational efficiency is required; token-level pruning is too coarse to preserve semantic integrity.

**Goal**: How to substantially reduce the number of visual tokens while maintaining accuracy?

**Key Insight**: The approach simulates human visual strategy — first scanning globally to locate relevant regions, then focusing on details. Region-level selection better respects semantic structure compared to token-level selection.

**Core Idea**: Learnable guidance queries are used to align visual regions and textual instructions in a shared feature space; after selecting instruction-relevant regions, they are re-encoded to remove irrelevant context.

## Method

### Overall Architecture

PinPoint consists of two stages:
1. **Region Selection**: Region-level features are extracted from the full image, and the most relevant regions are localized via Instruction-Region Alignment.
2. **Region Refinement**: Selected regions are re-encoded through the ViT independently, removing irrelevant context introduced by global self-attention to produce more compact and precise visual tokens.

### Key Designs

1. **Region-Level Feature Extraction**:

    - Visual tokens are reorganized into a 2D spatial grid; sliding windows of size $W \times H$ with stride $S$ extract region representations $\mathbf{R}_i \in \mathbb{R}^{W \times H \times d}$.
    - **Design Motivation**: Region-level comparison captures contextual relationships and semantic integrity better than token-level comparison.

2. **Instruction-Region Alignment**:

    - Learnable guidance queries $E \in \mathbb{R}^{K \times d}$ serve as cross-modal bridges.
    - Scaled dot-product attention is applied separately to visual regions and textual instructions:
    $E_i^v = A_i^v \cdot \mathbf{R}_i', \quad E^t = A^t \cdot \mathbf{T}'$
    - Candidate regions are ranked by cosine similarity, and top regions are adaptively selected until coverage reaches a preset ratio $r$.
    - **Design Motivation**: Decoder-only LLMs lack a CLS token for semantic aggregation, and BPE subwords are misaligned with visual features, necessitating an additional module to bridge the modalities.

3. **Dual Contrastive Learning**:

    - **Inter-modal Contrastive Loss** $\mathcal{L}_\text{inter}$: Cross-modal alignment — positive pairs consist of an instruction and its corresponding relevant region; negatives are unpaired samples within the batch.
    - **Intra-image Contrastive Loss** $\mathcal{L}_\text{intra}$: Intra-image region discrimination — pulls the instruction toward answer-relevant regions and pushes it away from irrelevant regions.
    - **Design Motivation**: The dual loss ensures both cross-modal alignment and intra-image region discrimination.

### Loss & Training

- $\mathcal{L}_\text{total} = \mathcal{L}_\text{inter} + \lambda \mathcal{L}_\text{intra}$, with $\lambda = 0.5$
- Only guidance queries and two MLP layers are trained; the LLM, ViT, and Projector are frozen.
- Training: 5 epochs, batch size 32, learning rate 2e-5.
- Window parameters: $W=H=10$, stride=7, coverage ratio $r=0.6$, $K=100$.

## Key Experimental Results

### Main Results

| Model | Method | InfoVQA ANLS↑ | FLOPs(T)↓ | SPDocVQA ANLS↑ | GQA Acc↑ |
|------|------|--------------|-----------|----------------|----------|
| LLaVA-NeXT-7B | Vanilla | 0.2552 | 38.98 (100%) | 0.6628 | 0.7598 |
| LLaVA-NeXT-7B | FastV | 0.2306 | 26.22 (67%) | 0.6099 | 0.7478 |
| LLaVA-NeXT-7B | SparseVLM | 0.2428 | 27.45 (70%) | 0.5726 | 0.7449 |
| LLaVA-NeXT-7B | **PinPoint** | **0.3024** | 25.48 (65%) | **0.6472** | **0.7608** |
| Qwen2-VL-7B | Vanilla | 0.7399 | 51.98 (100%) | 0.9359 | 0.7687 |
| Qwen2-VL-7B | **PinPoint** | **0.7140** | 28.88 (56%) | **0.8977** | **0.7624** |

On InfoVQA, PinPoint outperforms the Vanilla baseline by 18.5% in accuracy while using only 65.3% of the computation.

### Ablation Study

| Configuration | InfoVQA ANLS | Region Accuracy | Note |
|------|-------------|----------|------|
| w/o $\mathcal{L}_\text{intra}$ | 0.3011 | 82% | Missing intra-image contrast reduces region discrimination |
| w/ $\mathcal{L}_\text{intra}$ | 0.3024 | 84% | Full loss enables better region localization |
| ViCrop | 0.2547 | - | Iterative LLM interaction is prohibitively expensive (FLOPs 378%) |
| Ours + Global | 0.3075 | - | Adding global features yields further improvement |

### Key Findings

- A higher proportion of instruction-relevant tokens correlates linearly with higher VQA accuracy.
- Attention-weight-based token pruning may inadvertently discard tokens critical to the answer.
- Region Refinement effectively removes irrelevant context entanglement through isolated re-encoding.

## Highlights & Insights

- The "Focus, Don't Prune" design philosophy — selecting the most relevant regions rather than pruning the least important ones.
- Lightweight design: only guidance queries and two MLPs are trained; all other components remain frozen.
- Cross-model generalization: effective on both LLaVA-NeXT and Qwen2-VL.
- A new annotation dataset for InfoVQA/SPDocVQA/MPDocVQA is provided, containing bounding boxes for multiple supporting evidence regions.

## Limitations & Future Work

- The sliding window granularity is fixed and may not adapt well to all resolutions.
- Gains on natural images (GQA) are less pronounced than on documents and infographics.
- The region selection stage introduces some latency (approximately 381ms vs. 569ms for Vanilla, though subsequent computation is reduced).
- Integration with more recent token pruning methods has not been explored.

## Related Work & Insights

- PinPoint is complementary to token pruning approaches: pruning focuses on efficiency, while PinPoint targets accuracy together with efficiency.
- Instruction-conditioned visual processing is an important direction for LVLMs — letting the model's "what to look at" be determined by "what is being asked."
- The approach is transferable to other tasks requiring selective attention, such as chunk selection in RAG pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of region-level selection and re-encoding is clean and effective, though conceptually intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks, two base models, comprehensive comparisons, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, rich figures, and well-motivated throughout.
- Value: ⭐⭐⭐⭐ Practically valuable for information-dense scenarios with good methodological generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [\[CVPR 2026\] LFPC: Learning to Focus and Precise Cropping for MLLMs](lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)
- [\[CVPR 2026\] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions](seeing_through_touch_tactile_localization.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[ACL 2026\] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends](../../ACL2026/multimodal_vlm/a_survey_on_mllm-based_visually_rich_document_understanding_methods_challenges_a.md)

</div>

<!-- RELATED:END -->
