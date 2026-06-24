---
title: >-
  [Paper Note] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation
description: >-
  [CVPR 2025][Multimodal VLM][continual learning] This paper proposes the SeGP-CL framework, which precisely detects vulnerable regions at the semantic boundaries of old and new tasks using adversarial anchors via Dual-Targeted PGD (DPGD). By combining Anchor-guided Cross-modal Geometry Distillation (ACGD) and Text Semantic-Geometry Regularization (TSGR) to preserve the cross-modal geometric structure of VLMs, SeGP-CL achieves state-of-the-art (SOTA) performance on five continu…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "continual learning"
  - "VLM"
  - "semantic geometry"
  - "adversarial anchor"
  - "knowledge distillation"
date: 2026-05-08
content_hash: f4c91bc63b08f645
---

# Continual Learning with Vision-Language Models via Semantic-Geometry Preservation

**Conference**: CVPR 2025  
**arXiv**: [2603.12055](https://arxiv.org/abs/2603.12055)  
**Code**: TBD  
**Area**: Multimodal VLM  
**Keywords**: continual learning, VLM, semantic geometry, adversarial anchor, knowledge distillation

## TL;DR
This paper proposes the SeGP-CL framework, which precisely detects vulnerable regions at the semantic boundaries of old and new tasks using adversarial anchors via Dual-Targeted PGD (DPGD). By combining Anchor-guided Cross-modal Geometry Distillation (ACGD) and Text Semantic-Geometry Regularization (TSGR) to preserve the cross-modal geometric structure of VLMs, SeGP-CL achieves state-of-the-art (SOTA) performance on five continual learning benchmarks.

## Background & Motivation
**Background**: VLM-based continual learning methods include prompt-based (L2P, DualPrompt), adapter-based (MoE-Adapter), and text-prior-based methods (ENGINE, DesCLIP). These approaches have achieved progress in mitigating forgetting.

**Limitations of Prior Work**: (1) Existing methods are either overly conservative (freezing a large number of parameters, which hinders learning new knowledge) or fail to target the protection of cross-modal geometric structures during updates. (2) Methods requiring historical data violate the exemplar-free constraint. (3) Distillation methods based on reference data lack targeted constraints on regions sensitive to semantic drift.

**Key Challenge**: Geometric drift does not occur uniformly; rather, it concentrates at the boundary regions between old and new semantics ("boundary vulnerability"), where shared visual patterns are easily re-interpreted by the text semantics of the new task. However, existing methods cannot precisely locate and protect these vulnerable areas.

**Goal**: How to precisely protect the vulnerable regions of VLM cross-modal geometry under exemplar-free conditions?

**Key Insight**: Leverage adversarial attack concepts: if minute perturbations can alter the image-text alignment, the same perturbations can be utilized to actively discover and cover the most vulnerable geometric neighborhoods.

**Core Idea**: Construct adversarial anchors using Dual-Targeted PGD to probe old-new semantic boundaries, and perform cross-modal geometry distillation on these anchors to safeguard the vulnerable regions.

## Method

### Overall Architecture
Three stages: (1) Before training: DPGD constructs the adversarial anchor set $\mathcal{A}_t$; (2) During training: CE loss for learning the new task + ACGD for distilling old knowledge on anchors + TSGR for text semantic regularization; (3) After training: anchor-induced prototype shift + dual-path inference (fusion of the CLIP branch and the visual prototype branch).

### Key Designs

1. **Dual-Targeted PGD (DPGD) Anchor Construction**:

    - **Function**: Construct adversarial samples that fall into the vulnerable regions of the old-new semantic boundaries.
    - **Mechanism**: Select $K_{seed}$ new-task samples with the highest similarity to the old class text prototypes as seeds, and optimize the dual objective $\mathcal{L}_{adv}' = \mathcal{L}_{adv} + \lambda_p \mathcal{L}_{v\text{-}adv}$ using PGD. The textual objective pushes the sample toward the old class text embedding, while the visual objective pulls the sample toward the old class visual prototype. The iteration is formulated as $\delta^{(k+1)} = \Pi(\delta^{(k)} - \gamma \text{sign}(\nabla_\delta \mathcal{L}_{adv}'))$.
    - **Design Motivation**: Attacks utilizing only the textual objective may produce visually implausible anchors due to the modality gap; the dual objective ensures the anchors fall within the old class regions in both the text semantic and visual spaces.

2. **Anchor-guided Cross-modal Geometry Distillation (ACGD)**:

    - **Function**: Distill the cross-modal similarity distribution of the old model on the anchors.
    - **Mechanism**: For each anchor, calculate the cross-modal similarity distributions over all old classes for both the teacher and the student, and apply a KL divergence constraint to prevent student deviation: $\mathcal{L}_{ACGD} = D_{KL}(p^T_{clip} \| p^S_{clip})$.
    - **Design Motivation**: This is more precise than performing distillation on all new data (validated empirically in Fig.2b) because the anchors specifically target zones with the most severe drift.

3. **Text Semantic-Geometry Regularization (TSGR)**:

    - **Function**: Maintain the stability of the relative geometric relationships among text embeddings.
    - **Mechanism**: Construct a key relation subgraph (selecting important edges based on the similarity between text embeddings), and constrain the similarity of each edge in the subgraph to remain consistent before and after updates.
    - **Design Motivation**: Cross-modal alignment depends not only on the image-text relationships but also on the relative positioning among textual concepts. If the textual frame of reference drifts, the semantic coordinates of the old classes will be implicitly reparameterized.

4. **Anchor-Induced Prototype Migration & Dual-Path Inference**:

    - **Function**: Estimate the visual space drift after training using anchors, and transfer/migrate old class prototypes.
    - **Mechanism**: Measure the change in raw visual features of the anchors before and after the update, and correct the old class prototypes using this drift amount. During inference, fuse the CLIP cross-modal logits and the prototype visual logits.
    - **Design Motivation**: The modality gap in CLIP makes pure text matching insufficient; visual prototypes provide complementary information.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{CE} + \lambda_A \mathcal{L}_{ACGD} + \lambda_T \mathcal{L}_{TSGR}$, utilizing LoRA to fine-tune CLIP ViT-L/14. The training overhead increases by less than 20% compared to standard fine-tuning.

## Key Experimental Results

### Main Results

| Dataset | Metric (Last Acc) | SeGP-CL | Prev. SOTA | Gain |
|--------|---------------|---------|----------|------|
| CIFAR100 | Last | **84.6** | 80.6 (MG-CLIP) | +4.0 |
| CUB-200 | Last | **80.1** | 76.2 (RAPF) | +3.9 |
| ImageNet-R | Last | **82.9** | 80.0 (CLAP) | +2.9 |
| Cars-196 | Last | **85.3** | 80.7 (CLAP) | +4.6 |
| OmniBenchmark | Last | **92.8** | 86.6 (ENGINE) | +6.2 |

### Ablation Study

| Configuration | CIFAR100 Last | CUB-200 Last | Description |
|------|-------------|-------------|------|
| Naive LoRA | 70.5 | 66.3 | Baseline, severe forgetting |
| + ACGD | 78.9 | 75.2 | Anchor distillation contributes the most |
| + ACGD + TSGR | 80.7 | 77.1 | Text regularization further improves performance |
| + PT + Dual-Path | 84.6 | 80.1 | Full model |

### Key Findings
- The JSD of geometric drift at the semantic boundary is several times higher than that in the core region, validating the boundary vulnerability hypothesis; specifically, the JSD in boundary regions is approximately 3-5 times higher than that in core regions.
- Anchor-guided distillation (ACGD) outperforms distillation on either all new data or reference data, as it precisely covers the vulnerable regions.
- DPGD with a dual-objective achieves an improvement of approximately 1.5% on CUB-200 compared to single-text-target PGD, indicating that visual constraints are crucial for anchor quality.
- SeGP-CL maintains, and even slightly improves, the original CLIP's zero-shot capability on cross-scenario zero-shot evaluations (Food-101, Oxford-Pets, ImageNet-1K).
- Training overhead increases by only <20%, with near-zero additional computational overhead at inference time (0.00013 GFLOPs).
- In a long-sequence setting (20-step incremental learning), the performance decay curve of SeGP-CL is significantly flatter compared to alternative approaches.

## Highlights & Insights
- **Constructive Use of Adversarial Attacks**: Transformed adversarial attacks from a "threat" to a "diagnostic tool," utilizing PGD to actively discover the most vulnerable representation regions of the model. This strategy can be transferred to any scenario requiring precise localization of model vulnerability.
- **Empirical Finding of Boundary Vulnerability**: JSD measurements demonstrate that drift concentrates at the boundaries between old and new semantics rather than being uniformly distributed, providing clear guidance on "what to protect" in continual learning.
- **Generalizability of Cross-modal Geometry Protection**: The core philosophy of this framework (adversarial probing + anchor distillation) is not restricted to classification; it is also applicable to other downstream VLM tasks, such as retrieval and VQA.
- **Robustness in Cross-modal Transfer**: Protecting cross-modal geometry not only assists cross-modal inference but also indirectly improves the continual adaptation of the pure visual branch, demonstrating that cross-modal constraints can serve as effective regularization mechanisms.

## Limitations & Future Work
- The number of anchors and PGD steps require hyperparameter tuning, and anchor construction incurs extra computational costs (approximately 10 forward passes per step).
- Validated solely on CLIP, and has not been tested on newer VLMs (e.g., SigLIP, InternVL).
- The weight of the textual/visual target $\lambda_p$ in the dual-targeted PGD requires tuning on a per-dataset basis, lacking an adaptive strategy.
- The key relation subgraph in TSGR selects edges based on a fixed threshold, potentially omitting relationships that have lower similarity but are semantically important.
- Future work can explore extending DPGD to continual learning in generative VLMs.

## Related Work & Insights
- **vs MG-CLIP**: MG-CLIP also focuses on the modality gap but adopts a conservative strategy. In contrast, this work actively probes and protects vulnerable regions using adversarial anchors, which is more targeted.
- **vs ZSCL**: ZSCL requires a reference dataset for distillation, whereas this work utilizes synthetic anchors as an alternative, being both exemplar-free and more precise.
- **vs ENGINE**: ENGINE utilizes external LLMs to acquire textual semantics, while this method directly protects existing geometric structures without relying on external resources.
- **vs RAPF**: RAPF employs prompt fusion to mitigate forgetting but lacks targeted protection for vulnerable areas; SeGP-CL demonstrates a particularly pronounced advantage on fine-grained datasets (CUB-200, Cars-196).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Probing vulnerable regions with adversarial anchors is a highly ingenious innovation with clear theoretical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks + detailed ablations + cross-scenario evaluation + computational overhead analysis.
- **Writing Quality**: ⭐⭐⭐⭐ IEEE style with complete mathematical derivations, though somewhat lengthy.
- **Value**: ⭐⭐⭐⭐ A substantial advancement in the field of continual learning, and the discovery of boundary vulnerability is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)
- [\[CVPR 2025\] Can Large Vision-Language Models Correct Semantic Grounding Errors By Themselves?](can_large_vision-language_models_correct_semantic_grounding_errors_by_themselves.md)
- [\[ICCV 2025\] Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models](../../ICCV2025/multimodal_vlm/instruction-grounded_visual_projectors_for_continual_learning_of_generative_visi.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)

</div>

<!-- RELATED:END -->
