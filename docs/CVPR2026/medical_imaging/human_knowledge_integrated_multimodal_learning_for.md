---
title: >-
  [Paper Note] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization
description: >-
  [CVPR 2026][Medical Imaging][Single source domain generalization] This paper proposes the Domain Conformal Bound (DCB) theoretical framework to quantify causal factor discrepancies across domains and derives an optimizable consistency metric SDCD. Expert knowledge is refined accordingly and injected into MedGemma-4B via LoRA, achieving substantial improvements over single source domain generalization SOTA on 8 DR and 2 SOZ datasets.
tags:
  - CVPR 2026
  - Medical Imaging
  - Single source domain generalization
  - domain conformal bound
  - causal factors
  - human knowledge integration
  - diabetic retinopathy
  - MedGemma-4B
  - LoRA
date: 2026-05-08
content_hash: aa5ca73be1092970
---

# Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization

**Conference**: CVPR 2026
**arXiv**: [2603.12369](https://arxiv.org/abs/2603.12369)
**Code**: [GitHub](https://github.com/IMPACTLabASU/GenEval)
**Area**: Medical Imaging / Domain Generalization
**Keywords**: Single source domain generalization, domain conformal bound, causal factors, human knowledge integration, diabetic retinopathy, MedGemma-4B, LoRA

## TL;DR

This paper proposes the Domain Conformal Bound (DCB) theoretical framework to quantify causal factor discrepancies across domains and derives an optimizable consistency metric SDCD. Expert knowledge is refined accordingly and injected into MedGemma-4B via LoRA, achieving substantial improvements over single source domain generalization SOTA on 8 DR and 2 SOZ datasets.

## Background & Motivation

**Cross-domain generalization in medical image classification is a fundamental challenge.** The key bottleneck lies in unknown causal factor discrepancies across domains — for example, neovascularization (a critical indicator of Grade 4 DR) appears only in EyePACS but not in Messidor, forming a causal gap that directly violates the theoretical prerequisite of domain generalization, namely "causal coverage."

**Existing DG methods have not consistently outperformed ERM on DR.** Table 1 shows that improvements from methods such as SPSD-ViT are statistically insignificant (p=0.09). The more practical setting of single source domain generalization (SDG) — training on one domain and deploying across others — poses an even greater challenge, as a single source domain almost inevitably lacks certain causal factors present in target domains.

**Yet human experts possess causal knowledge that generalizes across domains** (e.g., DR grading standards remain consistent across devices and acquisition protocols). The challenge lies in the qualitative and ambiguous nature of expert knowledge (e.g., microaneurysms described as "small red dots" of 15–60 μm are easily confused with venous hemorrhages) — how can such knowledge be quantified, refined, and efficiently integrated into models?

## Method

### Overall Architecture

Step 1: DCB theory quantifies inter-domain causal factor relationship discrepancies → Step 2: SDCD metric evaluates source–target domain consistency → Step 3: Knowledge quantification (YOLOv12 detects lesions → 14-dimensional vector) → Step 4: SDCD-guided greedy ablation refines knowledge subset → Step 5: Refined knowledge and images construct multi-modal prompts; MedGemma-4B fine-tuned via LoRA.

### Key Designs

1. **Domain Conformal Bound (DCB)**:

   - Function: Provides a distribution-free framework to quantify discrepancies in causal factor relationships between two domains.
   - Mechanism: Models causal factors as sparse linear operators $\mathcal{K}$ using SINDy/Koopman theory and constructs confidence intervals $C$ via Mahalanobis distance. Target domain samples whose robustness measures fall within $C$ share source-domain causal patterns with probability $\geq 1-\alpha$.
   - Design Motivation: Addresses a critical gap in DG theory — the inability to quantify causal coverage — making generalization capacity predictable.

2. **Source Domain Consistency Degree (SDCD) and Knowledge Refinement**:

   - Function: Defines an optimizable domain consistency metric and uses it to select the most informative expert knowledge.
   - Mechanism: SDCD is defined as the proportion of target domain samples falling within the source domain DCB. A positive correlation between SDCD and SDG accuracy is established (Pearson r=0.692, p<0.02). Knowledge refinement converts YOLOv12-detected fundus lesions into 14-dimensional IoU vectors; SDCD-guided greedy ablation removes knowledge components that reduce consistency.
   - Design Motivation: Enables prediction of source–target generalization feasibility without requiring target domain labels; knowledge refinement eliminates ambiguous or detrimental components.

3. **GenEval Multi-modal Classification Engine**:

   - Function: Integrates refined expert knowledge into a VLM to enable cross-domain generalization classification.
   - Mechanism: Refined knowledge and images are composed into multi-modal prompts; MedGemma-4B is fine-tuned via LoRA (rank=16, alpha=16, 2.4% trainable parameters) applied to all attention and MLP projection layers.
   - Design Motivation: MedGemma-4B provides medical visual priors; LoRA efficiently injects domain-specific knowledge without compromising general capabilities.

### Loss & Training

Standard CAUSAL_LM loss. Single-domain training requires 1–10 hours; inference takes approximately 424 ms per sample.

## Key Experimental Results

### Main Results

| Source→Target | Method | Accuracy | Gain |
|---|---|---|---|
| EyePACS→Messidor | GenEval | 69.5% | +14.9% vs DRGen |
| EyePACS→Messidor2 | GenEval | 80.5% | +15.1% vs DRGen |
| Messidor→EyePACS | GenEval | 80.0% | +22.6% vs SPSD-ViT |
| MDG Average | GenEval | 79.21% | +5.91% vs SPSD-ViT |
| SOZ Cross-site | GenEval | F1=90.0% | +1.9% vs GPT-4o |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Without knowledge refinement | SDCD 59%, Acc 65% | Raw knowledge contains noise and ambiguous components |
| After refinement | SDCD 83%, Acc 73% | SDCD improvement → accuracy improvement, positive correlation confirmed |
| Zero-shot MedGemma | Avg 71.73% | Large inter-domain discrepancy requires fine-tuning |
| GenEval vs CLIP-DR | F1 75.1% vs 46.8% | Significant effect of knowledge injection |

### Key Findings

- SDCD is positively correlated with SDG accuracy (r=0.692, p<0.02), enabling generalization prediction without target domain labels.
- Knowledge refinement progressively improves SDCD and accuracy from the no-ablation baseline to the optimal subset, validating the effectiveness of greedy ablation.
- Extended SDG (1 source, 6 targets): GenEval achieves 66.2% vs DECO's 50.68% (+15.5%), demonstrating clear advantages in large-scale cross-domain settings.

## Highlights & Insights

- Theory and practice are tightly coupled: DCB/SDCD theoretically explain why existing DG methods fail and provide a concrete path to improvement. SDCD has independent value — it can predict generalization feasibility without any training and can serve as a pre-deployment safety assessment tool.

## Limitations & Future Work

- DCB assumes a continuously differentiable data-generating process, which may introduce errors under sharp threshold effects.
- Human knowledge acquisition depends on domain expert consultation, limiting scalability.
- The greedy ablation for knowledge refinement is not globally optimal.
- Validation is limited to medical imaging scenarios.

## Related Work & Insights

- **vs SPSD-ViT**: DR domain generalization SOTA, but assumes target domain exchangeability and cannot determine whether a new domain lies outside the training support; GenEval provides pre-deployment assessment via DCB.
- **vs BiomedCLIP/CLIP-DR**: Pre-trained VLM transfer; GenEval substantially outperforms via LoRA and knowledge injection (F1 75.1% vs 46.8%).

## Rating

- Novelty: ⭐⭐⭐⭐ DCB/SDCD theoretical framework constitutes an independent contribution; the knowledge refinement paradigm is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale validation across 8 DR and 2 SOZ datasets.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous but dense.
- Value: ⭐⭐⭐⭐ The paradigm of parametric injection of domain expert knowledge into VLMs is transferable to other vertical domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[AAAI 2026\] Experience with Single Domain Generalization in Real World Medical Imaging Deployments](../../AAAI2026/medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[CVPR 2026\] Robust Multi-Source Covid-19 Detection in CT Images](robust_multi-source_covid-19_detection_in_ct_images.md)

</div>

<!-- RELATED:END -->
