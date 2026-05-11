---
title: >-
  [Paper Note] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization
description: >-
  [CVPR 2026][Medical Imaging][Single source domain generalization] This paper proposes GenEval, which quantifies causal coverage gaps via a Domain Conformal Bound (DCB), distills human expert knowledge…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Single source domain generalization"
  - "vision-language model"
  - "causal coverage"
  - "conformal inference"
  - "diabetic retinopathy"
  - "LoRA fine-tuning"
  - "MedGemma"
date: 2026-05-08
content_hash: d5e545157d839e64
---

# Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization

**Conference**: CVPR 2026
**arXiv**: [2603.12369](https://arxiv.org/abs/2603.12369)
**Code**: [IMPACTLabASU/GenEval](https://github.com/IMPACTLabASU/GenEval)
**Area**: Medical Imaging
**Keywords**: Single source domain generalization, vision-language model, causal coverage, conformal inference, diabetic retinopathy, LoRA fine-tuning, MedGemma

## TL;DR

This paper proposes GenEval, which quantifies causal coverage gaps via a Domain Conformal Bound (DCB), distills human expert knowledge, and integrates it with a medical VLM (MedGemma-4B) through LoRA fine-tuning for single source domain generalization (SDG), achieving substantial gains over baselines on DR grading and seizure onset zone (SOZ) detection.

## Background & Motivation

**Domain generalization challenge**: Medical image classifiers suffer severe performance degradation under cross-domain deployment. Existing DG methods fail to consistently and significantly outperform ERM on DR grading (e.g., SPSD-ViT surpasses ERM-ViT by only 1.3%, $p=0.09$, not significant).

**SDG is more challenging**: Clinical settings often provide only a single source of training data, making SDG harder than multi-domain generalization (MDG), with SOTA methods performing even worse.

**Missing causal coverage**: Causal factor gaps exist across domains—e.g., EyePACS contains neovascularization markers absent in Messidor, causing models trained on Messidor to misclassify EyePACS samples.

**Lack of causal coverage quantification**: Domain generalization theoretically requires both causal coverage and source risk minimization, yet no objective method previously existed to quantify the degree of causal coverage.

**Human knowledge is valuable but ambiguous**: Domain experts possess knowledge that can bridge causal gaps, but this knowledge is qualitative and ambiguous (e.g., microaneurysms vs. venous hemorrhage are easily confused), necessitating quantification and refinement.

**General VLMs are insufficiently robust**: Existing medical VLMs (CLIP, CLIP-DR) are fragile on unseen domains and lack uncertainty guarantees.

## Method

### Overall Architecture

GenEval proceeds in two main stages: (1) causal coverage assessment and knowledge refinement; (2) multi-modal VLM classification. The DCB framework is first used to quantify inter-domain causal gaps; SDCD-guided ablation then selects the optimal knowledge subset; finally, the refined knowledge is incorporated into a multi-modal prompt alongside fundus images to fine-tune MedGemma-4B via LoRA.

### Key Designs

**1. Domain Conformal Bound (DCB)**

- A robustness measure $\rho(\mathcal{K}(X_i), D^s)$ is defined based on the Mahalanobis distance between sample $X_i$ and other samples in the source domain.
- Conformal inference is used to construct a prediction interval $C$ such that the robustness measure of in-distribution source samples falls within $C$ with probability $\geq 1-\alpha$.
- If the robustness residual of a target-domain sample falls within $C$, the sample contains no causal factor relationships absent from the source domain.

**2. Source Domain Coverage Degree (SDCD)**

- The percentage of target-domain samples whose robustness residuals fall within the DCB interval is computed as a quantitative indicator of causal coverage.
- SDCD is shown to correlate positively with SDG performance on the target domain (Pearson $r=0.692$, $p<0.02$).

**3. Knowledge Quantification and Refinement**

- YOLOv12 is employed to detect lesions such as hemorrhages, hard exudates, and cotton-wool spots, producing a 14-dimensional real-valued feature vector.
- Expert diagnostic rules (e.g., ICDR grading criteria) are encoded via propositional logic.
- Knowledge dimensions are progressively ablated guided by SDCD, selecting the subset that maximizes mean SDCD (removing neovascularization features yields the best result).

**4. GenEval Multi-modal Classification**

- MedGemma-4B serves as the backbone, fine-tuned via LoRA ($r=16$, $\alpha=16$, dropout=0.05), updating approximately 95M parameters (2.4% of the 4B total).
- Refined expert knowledge is embedded as text in a clinically structured prompt alongside fundus images.
- Inference takes approximately 424 ms per image; end-to-end with YOLO detection, approximately 633 ms.

### Loss & Training

Standard causal language modeling (Causal LM) loss is used for LoRA fine-tuning, minimizing source-domain risk via cross-entropy.

## Key Experimental Results

### Main Results

**SDG — DR Grading (12 source–target transfer pairs)**

| Source → Target | Best Baseline | Baseline Acc. | GenEval | K+D SDCD |
|:---|:---|:---:|:---:|:---:|
| Messidor → Aptos | SPSD-ViT | 48.3% | **56.0%** | 98.0% |
| Messidor → EyePACS | SPSD-ViT | 57.4% | **80.0%** | 94.9% |
| Messidor2 → Aptos | SPSD-ViT | 52.8% | **69.7%** | 76.3% |
| Messidor2 → EyePACS | SPSD-ViT | 72.5% | **77.8%** | 96.3% |
| EyePACS → Messidor2 | DRGen | 65.4% | **80.5%** | 99.8% |
| EyePACS → Messidor | DRGen | 54.6% | **69.5%** | 100.0% |

**Extended SDG (fixed EyePACS training, 6 target domains)**

| Method | APTOS | Messidor | IDRiD | DeepDR | FGADR | RLDL | Avg. |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GDRNet | 52.8 | 65.7 | 70.0 | 40.0 | 7.5 | 44.3 | 46.7 |
| DECO | 59.7 | 70.1 | 74.8 | 40.3 | 9.9 | 49.3 | 50.7 |
| **GenEval** | **73.2** | 69.5 | 70.6 | **59.2** | **56.9** | **67.6** | **66.2** |

### Ablation Study

Knowledge refinement ablation (SDCD-guided):

| Ablation | SDCD (%) | Accuracy (%) |
|:---|:---:|:---:|
| No ablation | 59.0 | 65.0 |
| Remove microaneurysms | 68.0 | 70.0 |
| Remove hemorrhages/exudates | 71.7 | 71.1 |
| Remove venous beading | 82.8 | 73.2 |
| Remove neovascularization | **82.8** | **73.2** |

Removing neovascularization yields the best outcome, as this complex lesion cannot be reliably detected by YOLO, introducing noise that degrades SDCD.

### Key Findings

- **SDCD correlates positively with accuracy** ($r=0.692$, $p<0.02$), validating the monotonicity stated in Lemma 1.
- **Knowledge integration substantially improves SDCD**: K+D SDCD is markedly higher than D-only SDCD, approaching 100% in most cases.
- **MDG also benefits**: GenEval achieves 79.21% average accuracy on four-domain DR vs. 73.3% for SPSD-ViT (+5.9%).
- **VLM comparison**: GenEval achieves macro F1 of 75.1%, surpassing CLIP-DR by +28.3% (46.8% → 75.1%).
- **SOZ cross-center**: GenEval achieves average F1 of 90.0% vs. 88.1% for CuPKL, with more stable cross-center performance.

## Highlights & Insights

- DCB is the first distribution-free theoretical framework for quantifying causal coverage, enabling pre-deployment assessment of generalization feasibility.
- The SDCD-guided knowledge refinement mechanism elegantly uses a measurable indicator to select the optimal knowledge subset, resolving the ambiguity inherent in qualitative expert knowledge.
- Integrating structured expert knowledge as textual prompts into a VLM to bridge inter-domain causal gaps in a multi-modal manner is a novel and compelling idea.
- The evaluation is extensive: 8 DR datasets + 2 SOZ datasets, 12 SDG transfer directions, and comprehensive baselines, ablations, and sensitivity analyses.

## Limitations & Future Work

- The DCB theory assumes a continuously differentiable data-generating mechanism and may not apply to threshold effects or abrupt transitions in cyber-physical hybrid systems.
- YOLO-based knowledge extraction is a performance bottleneck: complex lesions such as neovascularization cannot be reliably detected and must ultimately be excluded.
- The 14-dimensional knowledge vector depends on disease-specific expert rules; adapting to new tasks requires redefining features and logic, incurring high generalization costs.
- SDCD becomes unstable under low signal-to-noise conditions (correlation breaks down when PSNR < 15 dB), potentially failing in poor image quality scenarios.
- Validation is limited to two medical tasks (DR and SOZ); broader medical imaging domains such as pathology and CT remain unexplored.

## Related Work & Insights

- **Medical domain generalization**: Distribution alignment methods such as MMD, CDANN, SD-ViT, and SPSD-ViT consistently fail to outperform ERM; DRGen, DECO, and GDRNet serve as DR-specific baselines.
- **Medical VLMs**: BiomedCLIP and LLaVA-Med enable zero-shot transfer; CLIP-DR introduces ranking-aware prompts; MedGemma-4B is the domain-specific medical foundation model adopted in this work.
- **Conformal inference**: A distribution-free uncertainty quantification framework previously applied to OOD detection and medical AI deployment; this paper innovatively repurposes it to quantify inter-domain causal gaps.

## Rating

- Novelty: ⭐⭐⭐⭐ — DCB theory and SDCD-guided knowledge refinement are original contributions; the multi-modal knowledge integration approach is conceptually inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8+2 datasets, 12 SDG transfer pairs, diverse baselines, and complete ablation and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous, but notation is dense and the writing is lengthy; some proofs require consulting supplementary material.
- Value: ⭐⭐⭐⭐ — Highly valuable for real-world SDG deployment in medical imaging; DCB can serve as a pre-deployment safety check tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[AAAI 2026\] Experience with Single Domain Generalization in Real World Medical Imaging Deployments](../../AAAI2026/medical_imaging/experience_with_single_domain_generalization_in_real_world_medical_imaging_deplo.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)

</div>

<!-- RELATED:END -->
