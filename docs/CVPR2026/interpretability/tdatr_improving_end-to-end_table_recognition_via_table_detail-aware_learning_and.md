---
title: >-
  [Paper Note] TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment
description: >-
  [CVPR2026][Interpretability][Table Recognition] This paper proposes the TDATR framework, which achieves end-to-end table recognition under limited annotation data through a "perceive-then-fuse" strategy and a structure-g…
tags:
  - "CVPR2026"
  - "Interpretability"
  - "Table Recognition"
  - "End-to-End"
  - "Detail-Aware Learning"
  - "Cell Localization"
  - "Visual-Language Alignment"
date: 2026-05-08
content_hash: 9ace63e9db3f8b2a
---

# TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment

**Conference**: CVPR2026  
**arXiv**: [2603.22819](https://arxiv.org/abs/2603.22819)  
**Code**: [github.com/Chunchunwumu/TDATR.git](https://github.com/Chunchunwumu/TDATR.git)  
**Area**: Interpretability  
**Keywords**: Table Recognition, End-to-End, Detail-Aware Learning, Cell Localization, Visual-Language Alignment

## TL;DR
This paper proposes the TDATR framework, which achieves end-to-end table recognition under limited annotation data through a "perceive-then-fuse" strategy and a structure-guided cell localization module, attaining state-of-the-art performance across 7 benchmarks without dataset-specific fine-tuning.

## Background & Motivation
Table Recognition (TR) is a core task in document analysis, requiring the conversion of table images into machine-readable formats such as HTML. Existing approaches fall into two main categories:
- **Modular TR**: Models table structure (TSR) and content (TCR) separately, training them independently and integrating results via post-processing. This approach ignores the intrinsic dependencies between structure and content, leading to suboptimal integration and error accumulation.
- **End-to-End TR**: Generates structured outputs in a unified manner, but heavily relies on large-scale TR-annotated data and exhibits poor generalization in data-limited scenarios. Moreover, most methods do not provide spatial correspondences at the cell level, limiting interpretability.

**Key Challenge**: Although end-to-end methods simplify the pipeline, the annotation cost for TR data is extremely high (requiring simultaneous annotation of both structure and content), causing existing methods to underperform on diverse real-world tables.

**Key Insight**: This paper decouples TR capability learning into two stages—"perceive" and "fuse"—first acquiring fine-grained table detail perception through multi-task pretraining, then learning to fuse with a small amount of TR data, while introducing structure-guided cell localization to enhance interpretability.

## Method

### Overall Architecture
TDATR adopts an architecture consisting of a visual encoder (Swin Transformer), a multimodal language decoder, and a Structure-Guided Cell Localization (SGCL) module, trained in two stages following the "perceive-then-fuse" strategy.

### Key Designs

1. **Table Detail-Aware Learning**:

    - Two categories of pretraining tasks are designed within a unified language modeling paradigm:
    - **Content Recognition Tasks**: Spatially ordered text detection, text detection with bounding box queries, and Markdown parsing — leveraging large-scale multi-source document data (web pages, papers, READMEs, etc.) to enhance OCR and layout understanding.
    - **Structure Understanding Tasks**: Cell detection, spanning cell detection, row/column detection, and structure parsing — capturing table structure at both the cell level and the row/column level.
    - **Design Motivation**: By pretraining on diverse document data, the model acquires robust structural and content perception capabilities without relying on large quantities of TR-specific annotated data.

2. **Structure-Guided Cell Localization (SGCL)**:

    - Cell representations are extracted from hidden states across different layers of the language decoder and aggregated via learnable weights.
    - For each cell, an initial representation $C$ is obtained by average pooling between the `<td` and `</td>` tokens.
    - $C$ is projected into row/column feature spaces, and an adjacency matrix is computed via inner products to generate a structural mask: $M_{xy}^k = \mathbb{1}[\text{Sigmoid}(\langle C_x^k, C_y^k \rangle / \text{dim}(C^k)) > 0]$
    - The structural mask guides bidirectional contextual attention to enhance $C \to C'$.
    - Based on $C'$, an MLP regresses initial bounding boxes, which are then refined via DAB-DETR decoding layers using multi-resolution visual features $P'_3$ and $P'_4$.
    - **Design Motivation**: Anchor points are initialized from TR hidden states, ensuring one-to-one correspondence with TR outputs and eliminating the need for post-processing and unstable bipartite matching.

3. **Fusion Fine-Tuning Stage**:

    - The model is trained on HTML table parsing tasks while simultaneously optimizing the SGCL module to predict precise cell coordinates.
    - The model completes end-to-end TR by implicitly aggregating the table detail perception capabilities acquired in the previous stage.

### Loss & Training
Perception stage: All tasks use cross-entropy loss $L_{ce}$

Fusion stage: $L_f = \lambda_{ce} L_{ce} + \lambda_b L_b + \lambda_{iou} L_{iou} + \lambda_m L_m + \lambda_s L_s$
- $L_b$: Cell regression loss; $L_{iou}$: IoU loss
- $L_m$: Mask alignment loss (Mask-DINO style), enhancing alignment between $C'$ and image features
- $L_s$: Structure-guided loss (BCE), optimizing row/column relation matrices
- Weights: $\lambda_b=0.05, \lambda_{iou}=0.03, \lambda_m=0.03, \lambda_s=0.05, \lambda_{ce}=1.0$

Each stage is trained for 3 epochs using 16×64GB 910B NPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | TDATR | Prev. SOTA | Gain |
|--------|------|-------|----------|------|
| iFLYTAB-full | TEDS | 93.22 | 84.36 (DeepSeek-OCR) | +8.86 |
| TabRecSet | TEDS | 92.70 | 70.70 (EDD) | +22.00 |
| PubTables-1M | TEDS | 97.97 | 95.48 (Dolphin) | +2.49 |
| PubTabNet | TEDS | Further gains reported with -ft variant | - | - |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| w/o Detail-Aware Learning | Significant performance drop | Validates the perceive-then-fuse strategy |
| w/o SGCL | Increased TEDS-D | Structure-content alignment degrades |
| TDATR-ft (additional fine-tuning on PubTabNet) | Gains in both TSR and TR | Confirms benefits of additional data |

### Key Findings
- The TSR performance of end-to-end TR surpasses that of specialist TSR models, demonstrating that content recognition positively promotes structural recognition.
- State-of-the-art results are achieved using far fewer fine-tuning data than baselines, validating the effectiveness of the decoupled learning strategy.
- TEDS-Delta (TEDS − TEDS_S) is substantially better than modular methods, indicating that the end-to-end approach avoids post-processing error accumulation.

## Highlights & Insights
- The "perceive-then-fuse" paradigm is an elegant decoupling strategy: it transforms the hard-to-obtain end-to-end TR annotation requirement into more accessible document data pretraining combined with lightweight TR fine-tuning.
- The SGCL module is carefully designed, using TR decoder hidden states as anchor initializations for DAB-DETR, avoiding the training instability associated with Hungarian matching.
- The newly introduced dataset iFLYTAB-full fills a gap in evaluation benchmarks for Chinese real-world table recognition.

## Limitations & Future Work
- TSR performance on digital tables such as PubTabNet is marginally lower than the best specialist models, as the full TR sequence is approximately twice the length of TSR sequences, increasing generation difficulty.
- The model has 600M parameters; while smaller than large OCR VLMs (e.g., Qwen2.5-VL-72B), it remains relatively large.
- Integration with LLM backbones has not been explored, which may limit understanding of complex document-level context.

## Related Work & Insights
- Dolphin is a comparable end-to-end TR method; TDATR surpasses it by over 2.5% TEDS under the same modeling paradigm.
- Unlike multi-decoder methods such as OmniParser, TDATR employs a single decoder to unify structure and content generation.
- The structure-guided localization idea is generalizable to other document understanding tasks requiring layout-content alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ The perceive-then-fuse strategy and SGCL module design are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven benchmarks, no dataset-specific fine-tuning, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed method descriptions.
- Value: ⭐⭐⭐⭐ Highly practical for table recognition under data-limited conditions.

## Supplementary Notes
- The visual encoder is Swin Transformer (300M) and the language decoder is a Transformer (300M), totaling 600M parameters.
- The SGCL uses $L_d=3$ DAB-DETR decoding layers; the bidirectional enhancement branch consists of 2 self-attention blocks and 1 cross-attention block.
- Maximum decoding length is 4096 tokens; the longer side of input images does not exceed 2048 pixels.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](pixel2phys_distilling_governing_laws_from_visual_dynamics.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](language_models_can_explain_visual_features_via_steering.md)

</div>

<!-- RELATED:END -->
