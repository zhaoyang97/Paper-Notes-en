---
title: >-
  [Paper Note] Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval
description: >-
  [AAAI 2026][Interpretability][Moment Retrieval] This paper proposes DEMR, a framework that introduces Deep Evidential Regression (DER) into video moment retrieval. It mitigates modal imbalance via a Reflective Flipped Fu…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Moment Retrieval"
  - "Evidential Learning"
  - "Uncertainty Estimation"
  - "Cross-Modal Alignment"
  - "Debiasing"
date: 2026-05-08
content_hash: a9a817253b8a0584
---

# Adaptive Evidential Learning for Temporal-Semantic Robustness in Moment Retrieval

**Conference**: AAAI 2026
**arXiv**: [2512.00953v1](https://arxiv.org/abs/2512.00953v1)  
**Code**: [https://github.com/KaijingOfficial/DEMR](https://github.com/KaijingOfficial/DEMR)  
**Area**: Interpretability
**Keywords**: Moment Retrieval, Evidential Learning, Uncertainty Estimation, Cross-Modal Alignment, Debiasing

## TL;DR
This paper proposes DEMR, a framework that introduces Deep Evidential Regression (DER) into video moment retrieval. It mitigates modal imbalance via a Reflective Flipped Fusion (RFF) module and corrects the counter-intuitive uncertainty estimation bias in vanilla DER via a Geom-regularizer, achieving significant improvements on both standard and debiased benchmarks.

## Background & Motivation
Video Moment Retrieval (MR) requires localizing temporal segments in untrimmed videos given natural language queries. Existing methods primarily build on pretrained Transformers (e.g., CLIP-ViT), but suffer from two core limitations:
1. **Limitations of deterministic inference**: Mainstream methods adopt a deterministic paradigm and lack effective strategies for handling hard frames (e.g., when queried objects are absent from the scene). At inference time, they rely on NMS to select the highest-scoring proposals, leading to overconfidence in ambiguous scenarios.
2. **Modality bias in CLIP features**: CLIP is pretrained predominantly on static image-text pairs, biasing it toward object-level visual features and limiting its fine-grained understanding of dynamic actions and textual semantics, which causes over-reliance on visual information during cross-modal fusion.

When the authors attempted to directly apply DER to MR as a baseline, two additional problems emerged: (a) naive concatenation of multimodal features fails to resolve modal imbalance; (b) the gradient of the original DER regularization term depends solely on prediction error and not on the amount of evidence, causing evidence to be over-suppressed for low-error samples while uncertainty remains underestimated for high-error samples — a counter-intuitive behavior.

## Core Problem
How to achieve reliable uncertainty modeling in multimodal moment retrieval? This breaks down into two sub-problems:
1. How to alleviate the visual-textual modal imbalance so that uncertainty estimation is sensitive to both modalities?
2. How to fix the structural defect in the DER regularizer whereby evidence is suppressed for accurate predictions?

## Method

### Overall Architecture
The DEMR pipeline: untrimmed video and natural language query as inputs → frozen CLIP-ViT + SlowFast for video feature extraction, CLIP for text feature extraction → progressive cross-modal alignment via the **RFF module** → **MR Head** for temporal boundary prediction + **DER Head** for uncertainty estimation → two-stage training: first a **QR task** to enhance text sensitivity, then joint training of MR and evidential learning. Uncertainty is used to assist proposal selection at inference.

### Key Designs
1. **Reflective Flipped Fusion (RFF) Module**: A dual-branch structure that alternately flips the roles of video and text features (Query ↔ Key/Value) at each layer, achieving progressive cross-modal alignment via shared cross-attention combined with modality-specific self-attention. This "reflective flipping" design more thoroughly models bidirectional modal interaction than naive concatenation, ensuring both visual and textual branches receive sufficient cross-modal information.
2. **Query Reconstruction (QR) Auxiliary Task**: In the early training stage, one noun is randomly masked from the query (nouns being the semantic units most readily captured by CLIP), and the model is required to reconstruct the masked token using video context and remaining text tokens. This compels the model to extract text-relevant semantic information from the video, thereby enhancing the sensitivity of the textual branch. QR is trained only during the first 30 epochs and subsequently frozen.
3. **Geom-regularizer**: The original DER regularizer $\mathcal{L}^R = \Delta \cdot \Phi$ has a gradient $-\nabla_\Phi \mathcal{L}^R = -\Delta$ that depends only on error and not on the amount of evidence. To address this structural defect, a geometry-constrained regularizer is proposed. The core idea is to constrain the normalized error $\bar{\Delta}$ and evidence $\bar{\Phi}$ to lie on the line $\bar{\Phi} + \bar{\Delta} = 1$, yielding $\mathcal{L}^L = \|\bar{\Phi} + \bar{\Delta} - 1\|_2^2$. Its gradient $-\nabla_{\bar{\Phi}} \mathcal{L}^L = -2(\bar{\Delta} + \bar{\Phi} - 1)$ depends on both error and evidence, realizing adaptive regulation where accurate predictions are assigned high evidence and inaccurate predictions are assigned low evidence.

### Loss & Training
- **Total loss**: $\mathcal{L} = \mathcal{L}_{mr} + \lambda_{der} \cdot \frac{2}{N} \sum_i \mathcal{L}_i^e + \mathcal{L}_{qr}$
- $\mathcal{L}_{mr}$ comprises Smooth L1 + GIoU loss (foreground clips only)
- $\mathcal{L}_i^e = \lambda_{NLL} \mathcal{L}_{NLL} + \lambda_{geom} \mathcal{L}^L$ (NIG negative log-likelihood + Geom regularization)
- **Two-stage training**: Stage 1 trains the QR module (30 epochs, lr=1e-5); Stage 2 trains MR + DER (gradients of the Geom regularizer are detached from the MR branch to focus on optimizing uncertainty)
- Key hyperparameters: $\lambda_{geom}=10^{-2}$, $\lambda_{der}=10^{-3}$

## Key Experimental Results

| Dataset | Metric | DEMR | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| QVHighlights (val) | mAP | 43.0 | 42.9 (CG-DETR) | +0.1 |
| QVHighlights (val) | R1@0.5 | 65.0 | 67.4 (CG-DETR) | -2.4 |
| Charades-STA | R1@0.5 | 60.2 | 58.4 (CG-DETR) | +1.8 |
| Charades-STA | mIoU | 51.6 | 50.1 (CG-DETR/UniVTG) | +1.5 |
| TACoS | R1@0.5 | 37.3 | 39.5 (CG-DETR) | -2.2 |
| QVHighlights (test, MLLM) | mAP@0.75 | 56.82 | 54.40 (LLaVA-MR) | +2.42 |
| Charades-CD | R1@0.3 IID-OOD gap | 3.29% | 12.00% (CM-NAT) | -8.71% |

### Ablation Study
- **RFF module**: mAP improves from 61.1 → 62.4 (+1.3), and the visual-textual uncertainty variance gap ΔVar decreases from 8.32 to 7.03
- **QR task**: Further improvement to 63.8 (+1.4), with ΔVar reduced from 7.03 to 0.98, demonstrating significant modal balancing
- **Geom-regularizer**: Full model achieves 65.0 (+1.2), with correctly calibrated uncertainty (higher error → higher uncertainty)
- Optimal QR settings: mask 1 noun, train for 30 epochs, lr=1e-4
- $\lambda_{geom}$ is optimal at $10^{-2}$; performance degrades noticeably when $\lambda_{der}$ exceeds $10^{-2}$

## Highlights & Insights
1. **First application of evidential regression to moment retrieval**, with systematic analysis of failure modes when directly transferring DER (modal imbalance + counter-intuitive uncertainty)
2. **Elegant Geom-regularizer design**: A simple geometric constraint $\bar{\Phi}+\bar{\Delta}=1$ resolves the structural defect in the gradient field concisely and effectively
3. **Strong interpretability**: Uncertainty intuitively reflects the model's low confidence in OOD regions and high epistemic uncertainty on ambiguous queries, providing a reliability signal for MR models
4. **Debiased generalization**: The IID-OOD gap on Charades-CD/ActivityNet-CD is minimal (3.29%), far outperforming deterministic methods

## Limitations & Future Work
1. **Backbone constraints**: The model uses frozen CLIP-ViT/SlowFast and does not leverage stronger VLM backbones (e.g., InternVideo2, LanguageBind); the paper itself identifies integration with MLLMs as a future direction
2. **Noun dependence in QR**: The QR task only masks nouns, leaving enhancement of verb/adjective semantics insufficient, despite the importance of action semantics in MR
3. **NMS still required**: Although uncertainty provides additional signal, NMS is still needed for final proposal selection at inference, leaving the full potential of uncertainty-guided proposal selection unrealized
4. **Computational overhead**: The multi-layer cross-attention in RFF and NIG distribution learning in DER increase training/inference costs; efficiency comparisons are not reported in the paper

## Related Work & Insights
- **vs UniVTG / QD-DETR / CG-DETR**: These are deterministic MR methods. DEMR's core advantage lies not in absolute performance (which is slightly below CG-DETR on certain metrics) but in its uncertainty estimation capability and superior debiased generalization
- **vs MomentDiff**: A diffusion-based MR method; DEMR outperforms it on all datasets while additionally providing uncertainty quantification
- **vs vanilla DER (Amini 2020)**: DEMR's Geom-regularizer fixes the gradient field defect of the original regularizer, representing a meaningful improvement to the DER framework with potential applicability to other regression tasks

The Geom-regularizer's design principle — constraining error and evidence to a single line — is not specific to MR and can be transferred to any regression task using DER, such as depth estimation or pose estimation. Furthermore, DEMR's property of producing high epistemic uncertainty in OOD regions is naturally suited for active annotation, prioritizing labeling of the temporal segments about which the model is most uncertain.

## Rating
- Novelty: ⭐⭐⭐⭐ (first to introduce DER into MR and systematically address transfer issues)
- Technical Depth: ⭐⭐⭐⭐ (gradient analysis and geometric constraint design of the Geom-regularizer are rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (standard + debiased benchmarks, extensive ablations and visualizations)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rich visualizations)
- Value: ⭐⭐⭐⭐ (open-source code; uncertainty estimation is valuable for downstream applications)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[AAAI 2026\] SCoPe: Intrinsic Semantic Space Control for Mitigating Copyright Infringement in LLMs](scope_intrinsic_semantic_space_control_for_mitigating_copyright_infringement_in_.md)
- [\[ACL 2026\] NOSE: Neural Olfactory-Semantic Embedding with Tri-Modal Orthogonal Contrastive Learning](../../ACL2026/interpretability/nose_neural_olfactory-semantic_embedding_with_tri-modal_orthogonal_contrastive_l.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)

</div>

<!-- RELATED:END -->
