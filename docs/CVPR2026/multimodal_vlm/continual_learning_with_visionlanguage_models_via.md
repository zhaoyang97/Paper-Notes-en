---
title: >-
  [Paper Note] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation
description: >-
  [CVPR 2026][Multimodal VLM][Continual Learning] This paper proposes SeGP-CL, which constructs anchor samples at the semantic boundaries between old and new classes via adversarial PGD, and couples them with Anchor-guided Cross-modal Geometry Distillation (ACGD) and Text Semantic Geometry Regularization (TSGR) to preserve cross-modal semantic-geometric structures during VLM continual learning without requiring replay of old data, achieving state-of-the-art performance on five benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Continual Learning
  - VLM
  - Semantic-Geometry Preservation
  - Adversarial Anchors
  - Cross-Modal Distillation
date: 2026-05-08
content_hash: f7e30c7a02ffc6e9
---

# Continual Learning with Vision-Language Models via Semantic-Geometry Preservation

**Conference**: CVPR 2026
**arXiv**: [2603.12055](https://arxiv.org/abs/2603.12055)
**Code**: None
**Area**: Continual Learning / Vision-Language Models / Catastrophic Forgetting
**Keywords**: Continual Learning, VLM, Semantic-Geometry Preservation, Adversarial Anchors, Cross-Modal Distillation

## TL;DR
This paper proposes SeGP-CL, which constructs anchor samples at the semantic boundaries between old and new classes via adversarial PGD, and couples them with Anchor-guided Cross-modal Geometry Distillation (ACGD) and Text Semantic Geometry Regularization (TSGR) to preserve cross-modal semantic-geometric structures during VLM continual learning without requiring replay of old data, achieving state-of-the-art performance on five benchmarks.

## Background & Motivation

### State of the Field

**Background**: Continual learning with VLMs (e.g., CLIP) is prone to catastrophic forgetting. The authors identify a key insight: the cross-modal geometric drift induced by forgetting is not uniformly distributed, but concentrates in "fragile neighborhoods" at the boundary between old and new semantics—where shared visual patterns are most susceptible to reinterpretation by new task text semantics. Using JSD to measure cross-modal distribution shift before and after incremental updates, the authors find that boundary regions exhibit significantly larger shifts than core regions. Existing methods either conservatively freeze parameters (L2P, DualPrompt) or apply reference-data distillation without sufficient targeting.

### Limitations of Prior Work

**Goal**: How can the most fragile cross-modal semantic-geometric regions in VLM continual learning be precisely identified and preserved without access to old samples, while maintaining the stability of the text semantic reference coordinate system?

## Method

### Overall Architecture
A three-stage pipeline: (1) before training, Dual-objective PGD (DPGD) is applied to new task data to construct adversarial anchor sets pointing toward old-class semantics; (2) during training, ACGD and TSGR are applied on top of the new-task cross-entropy loss to protect old geometric structures; (3) after training, anchors are used to estimate visual prototype drift, enabling dual-path inference. LoRA fine-tunes only the up-projection matrix $B$.

### Key Designs
1. **Dual-objective PGD Anchor Construction (DPGD)**: The $K_{seed}=5$ new-task samples most similar to old-class semantics are selected as seeds; $K_{adv}=10$ steps of PGD simultaneously optimize two objectives: (a) a cross-modal objective that pushes perturbed samples toward old-class text embeddings; and (b) a visual anchoring objective that maintains raw-space consistency with old-class visual prototypes to bridge the modality gap.
2. **Anchor-guided Cross-modal Geometry Distillation (ACGD)**: KL divergence is applied on anchors to distill the old-class distributions of the teacher and student models, with temperature $\tau_A=20$ (high temperature preserves global geometry rather than local relations) and weight $\lambda_{ACGD}=5$.
3. **Text Semantic Geometry Regularization (TSGR)**: A frozen LoRA yields a stable text reference coordinate system; for each new class, a $k=10$ nearest-neighbor subgraph is constructed and KL divergence constrains the student text space subgraph distribution to match the teacher's, with temperature $\tau_T=0.05$ (low temperature preserves compact local relations). Only $|C_t|$ new-class root nodes are constrained, with complexity $O(|C_t|k)$.
4. **Anchor-induced Prototype Transfer and Dual-path Inference**: After training, the feature discrepancy of anchors between teacher and student is used to estimate old-class prototype drift directions via similarity-weighted averaging. At inference, CLIP cross-modal logits and prototype visual logits are fused: $\ell(x,c) = s_{clip}(x,c) + \beta \cdot s_v(x,c)$, with $\beta=0.5$.

### Loss & Training
$\mathcal{L}_{CL} = \mathcal{L}_{cls} + \lambda_{ACGD} \mathcal{L}_{ACGD} + \lambda_{GR} \mathcal{L}_{GR}$, where $\lambda_{ACGD}=5, \lambda_{GR}=1$. LoRA is inserted into the attention projection and FFN linear layers of both the CLIP visual and text encoders. Training uses SGD with cosine decay, batch size 128, learning rate 0.001, 10 epochs per task, on 2× RTX 4090 GPUs.

## Key Experimental Results

| Dataset | Metric | SeGP-CL | Prev. SOTA | Gain |
|---|---|---|---|---|
| CIFAR100 (10 tasks) | Last Acc | 84.6 | 80.6 (MG-CLIP) | +4.0 |
| ImageNet-R (10 tasks) | Last Acc | 84.8 | 82.7 (MG-CLIP) | +2.1 |
| ImageNet-Sub (10 tasks) | Last Acc | 80.5 | 80.2 (RAPF) | +0.3 |
| CUB-200 (10 tasks) | Last Acc | 80.1 | 76.2 (RAPF) | +3.9 |
| UCF101 (10 tasks) | Last Acc | 92.8 | 90.1 (ENGINE) | +2.7 |

SeGP-CL-onlyCLIP, which uses only the CLIP branch (without visual prototypes), still surpasses most prior methods.

### Ablation Study
- ACGD improves Last Acc from 77.0 to 81.7 and reduces Forgetting from 10.9 to 5.8.
- TSGR further improves performance to 82.8/4.7 on top of ACGD.
- Anchor-based distillation significantly outperforms direct distillation on new data (which is even harmful, −0.8 Last Acc) and reference-data distillation.
- $K_{adv}=10$ is optimal; excessive iterations ($K_{adv}=20$) lead to degradation—the goal is to protect fragile neighborhoods rather than to overly approach old prototypes.
- Cross-scenario evaluation: after completing continual learning on CIFAR100, zero-shot accuracy on Food101/Oxford-Pets/ImageNet-1K remains close to the original CLIP, demonstrating preserved generalization ability.

## Highlights & Insights
- The cross-domain inspiration from adversarial robustness to continual learning is elegant: the sensitivity of VLMs to small perturbations is exploited to construct anchors that expose fragile regions.
- No old data or external reference data is required; the approach relies purely on adversarial perturbations of new-task data.
- Dual-path inference effectively bridges the modality gap through complementary cross-modal and visual prototype signals.

## Limitations & Future Work
- Lightweight historical memory of text and visual prototypes is still maintained; although the overhead is small, the method is not entirely history-free.
- Performance depends on the quality of prompt templates and may be limited in severely out-of-distribution scenarios.
- The method addresses only class-incremental learning and has not been extended to cross-domain or cross-task continual learning.

## Related Work & Insights
- vs. ZSCL: the latter relies on CC12M reference data for distillation, whereas SeGP-CL requires no additional data and achieves greater precision (+3.8 Last Acc).
- vs. MG-CLIP: SeGP-CL achieves comprehensive improvements in both forward transfer (FWT 72.3 vs. 70.2) and forgetting (F 0.9 vs. 4.9).
- vs. ENGINE: the latter incorporates text semantics from an external language expert, while SeGP-CL is fully self-contained.
- vs. GIFT: the synthetic old-class image approach is limited by domain gaps, achieving only +2.5 on CIFAR100 compared to SeGP-CL's +7.6.

## Related Work & Insights
- The adversarial anchor construction strategy is generalizable to any incremental learning scenario requiring protection of old knowledge boundaries.
- The subgraph regularization scheme in TSGR is lightweight and efficient, and is worth borrowing for other settings involving semantic space preservation.

## Rating
- Novelty: ⭐⭐⭐⭐ Adversarial anchors for probing fragile boundaries combined with dual-path inference represent a clever cross-domain adaptation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, distillation strategy comparisons, cross-scenario robustness, anchor analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations and well-articulated motivation.
- Value: ⭐⭐⭐⭐ Substantial advancement for VLM continual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [\[CVPR 2026\] GraphVLM: Benchmarking Vision Language Models for Multimodal Graph Learning](graphvlm_benchmarking_vision_language_models_for_multimodal_graph_learning.md)
- [\[CVPR 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semanticsensitive_underwater_image_enha.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Can Vision-Language Models Count? A Synthetic Benchmark and Analysis of Attention-Based Interventions](can_vision-language_models_count_a_synthetic_benchmark_and_analysis_of_attention.md)

</div>

<!-- RELATED:END -->
