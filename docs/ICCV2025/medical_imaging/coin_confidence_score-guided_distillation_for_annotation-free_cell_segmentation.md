---
title: >-
  [Paper Note] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation
description: >-
  [ICCV 2025][Medical Imaging][Cell Instance Segmentation] This paper proposes COIN, a three-stage framework that addresses the critical "error-free instance absence" problem in annotation-free cell instance segmentation.…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Cell Instance Segmentation"
  - "Annotation-Free"
  - "Confidence Score"
  - "Self-Distillation"
  - "Optimal Transport"
date: 2026-05-08
content_hash: 068ef150bc35adab
---

# COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation

**Conference**: ICCV 2025
**arXiv**: [2503.11439](https://arxiv.org/abs/2503.11439)  
**Code**: [https://shjo-april.github.io/COIN/](https://shjo-april.github.io/COIN/)  
**Area**: Medical Imaging
**Keywords**: Cell Instance Segmentation, Annotation-Free, Confidence Score, Self-Distillation, Optimal Transport

## TL;DR

This paper proposes COIN, a three-stage framework that addresses the critical "error-free instance absence" problem in annotation-free cell instance segmentation. The framework combines unsupervised semantic segmentation with optimal transport for pixel-level cell propagation, model–SAM consistency for instance-level confidence scoring, and confidence-guided recursive self-distillation, achieving performance on MoNuSeg and TNBC that surpasses semi-supervised and weakly supervised methods.

## Background & Motivation

Cell instance segmentation (CIS) is essential for understanding cellular morphology in histopathology images. Although unsupervised CIS (UCIS) methods aim to eliminate annotation dependency, existing UCIS models (e.g., SSA, PSM) share a fundamental limitation: **they fail to produce even a single error-free instance with IoU = 1.0**.

The authors identify two root causes: (1) geometry-augmentation-based learning (e.g., rotation augmentation in PSM) biases the model toward geometrically prominent features (e.g., elongated shapes), causing it to overlook circular or subtle cells and yield incomplete instance masks; (2) indiscriminate acceptance of all pseudo-labels propagates errors, since reliable and noisy predictions cannot be distinguished.

These observations motivate two key hypotheses: (a) a more effective cell detection strategy is needed to ensure the existence of error-free instances; (b) an unsupervised accuracy metric is required to select high-confidence instances for training.

## Method

### Overall Architecture

COIN proceeds in three steps. **Step 1** performs pixel-level cell propagation via unsupervised semantic segmentation (USS) combined with optimal transport (OT) to ensure comprehensive cell detection and guarantee the presence of error-free instances. **Step 2** computes instance-level confidence scores by measuring the consistency between model predictions and SAM-refined masks to assess per-instance reliability. **Step 3** applies confidence-guided recursive self-distillation to progressively expand the high-confidence instance set and improve model performance.

### Key Designs

1. **Pixel-Level Cell Propagation (Step 1)**: A unsupervised semantic segmentation model (e.g., DINOv2/MAE) extracts feature maps $F^{us} = U(I_k)$, while an existing UCIS model (e.g., SSA) provides initial masks $M_\theta^{ucis}$. Class-level average pooling (CAP) yields USS centroids $V_\theta^{us}$, which are used together with the feature map to generate a similarity map $S_\theta^{us} = \text{ReLU}(\text{sim}(F^{us}, V_\theta^{us}))$. The key innovation is the introduction of **Optimal Transport (OT)** for refinement: $S_\theta^{OT} = f_{OT}(S_\theta^{us}) \cdot S_\theta^{us}$. OT finds an optimal assignment matrix that prevents pixel overlap, which particularly benefits minority-class (peripheral) cell detection and reduces false negatives. Since USS models are trained on natural images, direct application to pathology images yields high false-positive rates (discrimination by color rather than cellular morphology); OT effectively mitigates this issue.

2. **Instance-Level Confidence Scoring (Step 2)**: Connected component labeling followed by watershed segmentation separates the propagated masks into $N$ instances $\{E_{\theta_t}^i\}$. The centroid of each instance serves as a point prompt for SAM to generate pseudo-GT masks. The core metric is the IoU consistency between model predictions and SAM outputs: $C_{\theta_t}^i = \text{IoU}(E_{\theta_t}^i, \text{SAM}(E_{\theta_t}^i))$. A non-parametric threshold $\delta_k$ (mean + standard deviation) is used for filtering: instances with $C > \delta_k$ are labeled as trusted ($\hat{M}=1$), regions with $C=0$ are labeled as definite background ($\hat{M}=0$), and the remainder are rejected and excluded from training ($\hat{M}=-1$). The elegance of this design lies in **not directly using SAM outputs** (as SAM is prompt-sensitive and error-prone in isolation), but instead employing cross-validation to select instances with high consistency—high agreement between the model and SAM strongly indicates correctly predicted regions.

3. **Confidence-Guided Recursive Self-Distillation (Step 3)**: Trusted instances are used to construct binary pseudo-masks $\hat{M}_{bin}^i(t)$ and edge pseudo-masks $\hat{M}_{edge}^i(t)$ (obtained via the Canny algorithm). The total loss is:
$$\mathcal{L}(t) = \mathcal{L}_{seg}(M_{bin}^{ucis}, \hat{M}_{bin}^i(t)) + \mathcal{L}_{seg}(M_{edge}^{ucis}, \hat{M}_{edge}^i(t))$$
where $\mathcal{L}_{seg} = \mathcal{L}_{ce} + \mathcal{L}_{dice}$. The edge decoder is incorporated to enhance boundary discrimination between adjacent cells. As model parameters $\theta_t$ are updated, the acceptance set $\mathcal{A}_\delta$ evolves dynamically, incrementally admitting more high-confidence instances in each round.

### Loss & Training

The segmentation loss combines cross-entropy and Dice loss, applied separately to the binary mask and the edge mask. Training proceeds as recursive self-distillation: in each round, the current model generates new pseudo-labels → high-confidence instances are selected → the model is retrained on trusted instances → improved predictions promote more instances into the trusted set → the cycle repeats. All experiments are conducted on a single NVIDIA RTX A100 80 GB GPU.

## Key Experimental Results

### Main Results

| Method | Supervision | MoNuSeg AJI↑ | MoNuSeg PQ↑ | MoNuSeg IoU↑ | TNBC AJI↑ |
|--------|-------------|--------------|-------------|--------------|-----------|
| SSA | Annotation-free | 0.259 | 0.185 | 0.618 | 0.273 |
| PSM | Annotation-free | 0.471 | 0.355 | 0.689 | - |
| **SSA+COIN** | **Annotation-free** | **0.580** | **0.536** | **0.776** | **0.568** |
| **PSM+COIN** | **Annotation-free** | **0.579** | **0.539** | **0.777** | - |
| SPPNet | Point annotation | 0.497 | 0.392 | 0.709 | - |
| InstaSAM | Point annotation | 0.574 | - | 0.772 | - |
| TextDiff | Mask + text | 0.510 | 0.410 | 0.726 | 0.464 |

COIN improves SSA's AJI from 0.259 to 0.580 (+124%), surpassing all weakly supervised methods that require point or bounding-box annotations.

### Ablation Study

| Configuration | MoNuSeg AJI↑ | MoNuSeg PQ↑ | Notes |
|---------------|--------------|-------------|-------|
| SSA baseline | 0.259 | 0.185 | Original UCIS model |
| +Step 1 (USS+OT) | ~0.35 | ~0.28 | Propagation recovers error-free instances |
| +Step 1+Step 2 (scoring) | ~0.45 | ~0.40 | High-confidence instance filtering |
| +Step 1+Step 2+Step 3 (distillation) | **0.580** | **0.536** | Recursive expansion of confidence set |

Significant improvements from SSA+COIN are observed across six datasets (MoNuSeg, TNBC, BRCA, CPM-17, CryoNuSeg, PanNuke), with AJI gains of 0.08–0.17, validating the generality and scalability of the proposed framework.

### Key Findings

- OT is critical for cell propagation: compared to alternatives such as K-means and spectral clustering, OT provides the strongest protection for minority-class (peripheral) cells.
- Confidence scoring effectively approximates GT quality: high-confidence instances exhibit AJI scores close to 1.0, whereas randomly sampled instances display a much wider AJI distribution.
- COIN is a model-agnostic framework: it consistently yields substantial improvements when combined with two distinct UCIS baselines, SSA and PSM.
- The edge decoder is essential for discriminating densely packed adjacent cells.

## Highlights & Insights

- The identification of the "error-free instance absence" problem is remarkably precise: the observation that existing UCIS methods cannot produce even a single perfect instance is easy to overlook yet critically important.
- Using model–SAM consistency as an unsupervised proxy for accuracy is an elegant idea: the "consensus" of two independent systems serves as a quality estimator without any human annotation.
- The three-step progressive design is internally coherent: Step 1 improves recall, Step 2 enables precise filtering, and Step 3 generalizes the gains—each step addresses a specific, well-defined problem.

## Limitations & Future Work

- The framework depends on SAM as an external knowledge source; however, SAM's adaptability to histopathology images is limited and may degrade on certain tissue types.
- The convergence and stability of recursive self-distillation lack theoretical guarantees, and the number of iterations requires empirical selection.
- The current work addresses only cell nucleus segmentation; extension to more complex cellular structures (e.g., cytoplasm, cell membrane) is not discussed.
- The computational overhead of OT may become a bottleneck for large-scale images.

## Related Work & Insights

- The paper adapts unsupervised instance segmentation ideas (UIS, e.g., CutLER) to pathology images with key modifications tailored to the dense cell scenario.
- SAM has been applied in weakly supervised segmentation in various ways (SPPNet and InstaSAM use point prompts); COIN is the first to achieve fully annotation-free use of SAM.
- Recursive self-training/self-distillation strategies are common in unsupervised learning, but the combination with confidence-score-based dynamic filtering is a distinctive contribution of this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The definition of the "error-free instance absence" problem and the three-step solution are both novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, multiple baseline comparisons, comprehensive ablation studies, and rich visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the diagrams in Figs. 1–6 are exceptionally well designed.
- **Value**: ⭐⭐⭐⭐⭐ An annotation-free method that outperforms weakly supervised approaches carries substantial practical value for pathology AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](../../ICLR2026/medical_imaging/disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](../../NeurIPS2025/medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)
- [\[AAAI 2026\] Pairing-free Group-level Knowledge Distillation for Robust Gastrointestinal Lesion Classification in White-Light Endoscopy](../../AAAI2026/medical_imaging/pairing-free_group-level_knowledge_distillation_for_robust_gastrointestinal_lesi.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)

</div>

<!-- RELATED:END -->
