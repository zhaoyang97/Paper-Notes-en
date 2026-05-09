---
title: >-
  [Paper Note] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection
description: >-
  [ICCV 2025][Object Detection][Mixture of Experts] This work is the first to introduce Mixture of Experts into real-time open-vocabulary object detectors. Through MoE-Tuning, it extends Grounding DINO 1.5 Edge from a dense model into a dynamic inference framework, proposing fine-grained expert decomposition and a pretrained weight allocation strategy. Using only 1.56M open-source data, the resulting model surpasses the original version trained on 20M private data.
tags:
  - ICCV 2025
  - Object Detection
  - Mixture of Experts
  - open-vocabulary detection
  - Grounding DINO
  - fine-grained expert decomposition
  - dynamic inference
date: 2026-05-08
content_hash: d8fa80519b30da29
---

# Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection

**Conference**: ICCV 2025
**arXiv**: [2507.17436](https://arxiv.org/abs/2507.17436)
**Code**: [https://github.com/wengminghe/Dynamic-DINO](https://github.com/wengminghe/Dynamic-DINO)
**Area**: Object Detection / Open-Vocabulary / MoE
**Keywords**: Mixture of Experts, open-vocabulary detection, Grounding DINO, fine-grained expert decomposition, dynamic inference

## TL;DR
This work is the first to introduce Mixture of Experts into real-time open-vocabulary object detectors. Through MoE-Tuning, it extends Grounding DINO 1.5 Edge from a dense model into a dynamic inference framework, proposing fine-grained expert decomposition and a pretrained weight allocation strategy. Using only 1.56M open-source data, the resulting model surpasses the original version trained on 20M private data.

## Background & Motivation
MoE has proven successful in large VLMs (e.g., MoE-LLaVA), yet its application to real-time open-vocabulary detectors remains unexplored. Existing real-time detectors (YOLO-World, Grounding DINO 1.5 Edge) rely on dense models whose single FFN per layer must handle all patterns—different categories, attributes, and spatial relationships—leading to gradient conflicts and long-tail issues. The MoE mechanism of activating different experts for different inputs is naturally suited to this need, but efficiently integrating MoE into existing compact detection models is the core challenge.

## Core Problem
How to bring the advantages of MoE into small real-time open-vocabulary detectors, expanding model capacity and search space without increasing inference cost, while addressing training efficiency under limited data?

## Method

### Overall Architecture
Dynamic-DINO is built upon a reproduced Grounding DINO 1.5 Edge (EfficientViT-L1 + BERT-base + 6-layer decoder). MoE is applied only to the FFN layers of the decoder (since only 900 tokens remain after Language-guided Query Selection, keeping computational overhead low). Training proceeds in two stages: standard pretraining of the base model for 7 epochs, followed by MoE-Tuning expansion and fine-tuning for 10 epochs.

### Key Designs
1. **Fine-Grained Expert Decomposition**: Rather than naively replicating $N$ complete FFNs as experts, each FFN's intermediate dimension is evenly divided into $k$ parts, yielding $k$ smaller experts. Under the constraint of unchanged total parameters, each layer has $k \times N$ experts, expanding the search space from $C_{N}^{k}$ to $C_{kN}^{k}$. At inference, only $k$ experts are activated, equivalent to the activation cost of a single complete FFN. This exponentially enlarges the sub-network search space without increasing parameters.

2. **Pretrained Weight Allocation Strategy**: The core innovation—pretrained FFN parameters are physically distributed to initialize each expert: $W_1$ is split row-wise, $W_2$ column-wise, and $b_2$ is divided by $k$, ensuring that the sum of outputs from $k$ experts equals the original FFN output ($\text{FFN}(x) = \sum_{j=1}^{k} E_j(x)$). Combined with router initialization (duplicating each router weight row $k$ times), this guarantees that MoE-Tuning begins as equivalent to the original dense model, enabling incremental performance gains rather than training from scratch.

3. **Discovery of Expert Collaboration Patterns**: Analysis of expert co-selection frequencies reveals interesting layer-wise differences—in shallow layers, experts tend to collaborate with diverse partners (expanding the search space), while in deeper layers, fixed cooperative groups of 2–3 experts form, with different combinations specializing in distinct visual patterns (e.g., "refrigerator-related tokens" vs. "clothing-related tokens").

### Loss & Training
- Detection losses: L1 + GIoU + Focal Loss (weights 2.0 / 5.0 / 2.0)
- MoE auxiliary loss: load balancing loss ($\alpha = 0.01$) to ensure balanced expert utilization
- MoE-Tuning unfreezes only: Cross-Attention in the Feature Enhancer + MoE layers + Detection Head
- Trained on 8× RTX 3090; MoE-Tuning is 1.87× faster than pretraining (7.5 h vs. 14 h)

## Key Experimental Results

| Method | Training Data | COCO AP | LVIS-mini AP | LVIS-val AP |
|--------|---------------|---------|--------------|-------------|
| GDINO 1.5 Edge (official) | 20M private | 42.9 | 33.5 | 27.3 |
| GDINO 1.5 Edge* (reproduced) | 1.56M open-source | 42.6 | 31.1 | 25.4 |
| **Dynamic-DINO** | **1.56M open-source** | **43.7** | **33.6** | **27.4** |
| Dynamic-DINO (800×1333) | 1.56M open-source | **46.2** | **36.2** | **29.6** |

- Surpasses the official model trained on 20M private data using only 1.56M open-source data
- Significant gains on LVIS rare categories: APr improves from 33.8 → 37.0 (+3.2), effectively alleviating the long-tail problem
- Inference speed: 98 FPS vs. 109 FPS (A100), only a 10% slowdown
- On edge device Jetson Orin NX: +0.24M FLOPs, −0.8 FPS, +1.87 AP
- Gains on RefCOCO: +4.1 / +4.0 / +3.8 on val / testA / testB
- Only +6M parameters yields +0.73 AP, with performance continuing to scale with parameters

### Ablation Study
- Expert decomposition $k=2$ is optimal; excessive splitting ($k=4$) causes overfitting under limited data
- Larger total expert count $N$ is consistently better: $N=16 > N=8 > N=4$
- Initialization strategy is critical: with initialization, +0.6 AP
- MoE can also be applied to the image encoder (+0.5 AP), though this paper focuses on the decoder
- Fine-tuning the detection head contributes the most (LVIS-val +1.1 AP)

## Highlights & Insights
- **First validation of MoE in real-time open-vocabulary detection**: fills a gap in applying MoE to compact multimodal models
- **Fine-grained decomposition is highly creative**: expanding the search space without increasing total parameters represents a "free lunch" perspective
- **Pretrained weight allocation guarantees incremental learning**: mathematically ensures MoE equivalence to the original model at initialization, eliminating training instability
- **Data efficiency of "doing more with less"**: 1.56M open-source vs. 20M private data—MoE-induced pattern specialization makes limited data more effective
- **Expert collaboration pattern analysis is insightful**: the hierarchical structure of exploration in shallow layers and specialization in deep layers aligns with the layered processing observed in human cognition

## Limitations & Future Work
- The current MoE implementation iterates over experts sequentially without parallel optimization, causing an additional ~10% latency
- Validation is limited to GDINO 1.5 Edge; larger models such as GDINO 1.5 Pro or other detection architectures are not tested
- Constrained by 8× RTX 3090, exploration of larger data scales and compute regimes is insufficient
- Excessive expert splitting ($k > 2$) overfits under limited data, indicating that the method has some data-volume requirements

## Related Work & Insights
- **vs. YOLOE**: YOLOE achieves efficiency through re-parameterization and multi-prompt unification, while Dynamic-DINO does so via MoE dynamic routing. The two approaches are complementary—MoE could be applied to the YOLOE architecture
- **vs. Grounding DINO 1.5 Edge**: Dynamic-DINO is a direct extension, surpassing the original model with less data via MoE-Tuning
- **vs. DeepSeekMoE / QwenMoE**: Those methods use random initialization and train from scratch; Dynamic-DINO uses pretrained weight allocation with incremental fine-tuning, making it better suited to low-data regimes

The MoE-Tuning paradigm is generalizable to other vision tasks requiring efficient fine-tuning under limited data. The fine-grained decomposition strategy can be combined with model compression—exploring with MoE during training and consolidating the optimal sub-network at inference. The expert collaboration analysis methodology is worth replicating in other MoE architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of MoE in compact open-vocabulary detection; fine-grained decomposition and weight-allocation initialization are elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on COCO / LVIS / ODinW / RefCOCO; in-depth expert collaboration analysis; edge-device validation
- Writing Quality: ⭐⭐⭐⭐ Analysis figures (Figures 2 / 7 / 8) are highly persuasive; method description is clear
- Value: ⭐⭐⭐⭐ Introduces the MoE paradigm to real-time open-vocabulary detection, though engineering optimization remains to be refined

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](yoloe_realtime_seeing_anything.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](visual-rft_visual_reinforcement_fine-tuning.md)
- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](../../CVPR2026/object_detection/abra_teleporting_finetuned_knowledge_across_domain.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[NeurIPS 2025\] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](../../NeurIPS2025/object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)

</div>

<!-- RELATED:END -->
