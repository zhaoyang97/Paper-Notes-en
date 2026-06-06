---
title: >-
  [Paper Note] PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning
description: >-
  [ICML 2026][Medical Imaging][Whole Slide Images] PathCTM reframes Whole Slide Image (WSI) analysis from "exhaustive high-magnification patching" to "continuous multi-scale reasoning from low-resolution global and high-re…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Whole Slide Images"
  - "MIL Acceleration"
  - "Continuous Thought Machine"
  - "Multi-scale Reasoning"
  - "Confidence-aware Early Stopping"
date: 2026-05-08
content_hash: bbb82f46e4e537b4
---

# PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.19491](https://arxiv.org/abs/2605.19491)  
**Code**: https://github.com/JSGe-AI/PathCTM  
**Area**: Medical Imaging / Pathology / WSI Analysis Efficiency  
**Keywords**: Whole Slide Images, MIL Acceleration, Continuous Thought Machine, Multi-scale Reasoning, Confidence-aware Early Stopping

## TL;DR
PathCTM reframes Whole Slide Image (WSI) analysis from "exhaustive high-magnification patching" to "continuous multi-scale reasoning from low-resolution global and high-resolution local" views. Based on the Continuous Thought Machine, it introduces the thinking-in-scales paradigm, attention-guided regional pruning, and confidence-aware early stopping. This approach reduces the number of patches by 95.95% and inference time by 95.62% while simultaneously improving the AUC.

## Background & Motivation

**Background**: Mainstream WSI analysis (gigapixel pathology images) relies on Multiple Instance Learning (MIL), involving tiling slides into tens of thousands of high-magnification patches, extracting features per patch, and performing slide-level prediction through aggregate modeling (e.g., CLAM, TransMIL, ABMIL). Integrating these with pathology foundation models (Virchow, GigaPath, Prov-GigaPath) yields high performance but remains extremely slow.

**Limitations of Prior Work**: (1) Patch tiling and feature extraction dominate runtime, yet most patches contribute negligibly to final predictions. (2) Existing acceleration methods (ZoomMIL, HAG-MIL, EAGLE, hierarchical distillation) depend on fine-grained annotations or rigid cascade structures; they mimic "coarse-to-fine" formally but lack continuous memory reasoning, resulting in either accuracy degradation or marginal efficiency gains. (3) The recent Continuous Thought Machine (Darlow 2026) supports continuous reasoning but only for single-scale static images—it cannot hallucinate cellular details from low-resolution WSIs and fails to exploit the WSI pyramid structure.

**Key Challenge**: Clinical pathologists perform "multi-scale continuous reasoning"—observing global tissue architecture at low power, identifying suspicious regions, zooming to high power for cellular details, and stopping once sufficient information is gathered. Existing methods are either exhaustive (MIL), rigidly cascaded (ZoomMIL), or utilize single-scale continuous reasoning (CTM), failing to integrate "multi-scale" logic with "continuous reasoning + adaptive early stopping."

**Goal**: Reframe WSI analysis as a dynamic sequence of information seeking—gradually reducing conditional entropy $H(Y | \bm Z_t)$ to maximize information gain within a computational budget. Specifically: (1) maintain memory across scales through continuous reasoning; (2) dynamically select high-resolution regions based on information density; (3) stop once confidence meets the diagnostic requirement.

**Key Insight**: While CTM's "thinking-in-time" is insufficient for WSIs, its concept of "internal time + persistent memory" can be adapted. This paper introduces the "thinking-in-scales" dimension—a joint continuous reasoning of internal time and spatial scale, allowing low-magnification iterations to establish global hypotheses while high-magnification iterations verify local details and enable early stopping.

**Core Idea**: A synergy of three modules: scale-space continuous reasoning, attention-guided hard pruning, and confidence-aware entropy-minimizing early stopping, mimicking the clinical diagnostic workflow.

## Method

### Overall Architecture

WSI input $\to$ low-magnification global features $\to$ CTM-style continuous reasoning for $n$ steps (maintaining FIFO memories $\bm H^t, \bm E^t$) $\to$ if confidence is insufficient, select regions via Top-$K$ attention and switch to the next higher magnification $\to$ cross-scale fusion (concatenating current $\bm S_{out}^{L-1,t}$ with the highest-confidence output from the previous scale $\bm S_{out}^{L,\max}$) $\to$ repeat until confidence reaches the threshold or the budget is exhausted.

Training Goal: Take the lowest loss point $t_l^1$ and the highest confidence point $t_l^2$ at each scale. The loss is $\mathcal{L}_{all} = \frac{1}{z}\sum_l \frac{\mathcal{L}_l^{t_l^1} + \mathcal{L}_l^{t_l^2}}{2}$, optimizing both classification accuracy and uncertainty estimation ("self-awareness").

### Key Designs

1. **Scale-Space Continuous Reasoning (Thinking in Scales)**:
    - **Function**: Performs continuous reasoning across scales on the WSI pyramid, with continuous temporal reasoning within each scale.
    - **Mechanism**: Executes $n$ steps per scale $L$, with state transition $\bm h^t = f_{\theta_{syn}}(\text{concat}(\bm e^t, \bm b^t))$ ($\bm b^t$ is the attention output). FIFO history $\bm H^t \in \mathbb{R}^{D \times M}$ stores the last $M$ pre-activations, and $\bm E^t \in \mathbb{R}^{D \times N}$ stores all post-activations. FIFO updates persist across scale switches to maintain continuity. Cross-scale fusion $\hat y^t = \text{MLP}([\bm S_{out}^{L-1,t} \| \bm S_{out}^{L,\max}])$ prevents global context forgetting.
    - **Design Motivation**: Standard CTM assumes multi-step iteration on a fixed tensor can extract deeper information, but low-magnification WSIs lack fine details. Introducing the scale dimension allows "switching to higher magnification when necessary," mirroring the magnification adjustments of a pathologist.

2. **Attention-Guided Regional Pruning (Conditional Computation)**:
    - **Function**: Converts cross-scale patch selection into an information gain maximization problem under budget constraints.
    - **Mechanism**: Target $\mathcal{S}^* = \arg\max_{|\mathcal{S}| \leq K} I(Y; \mathcal{S} | \bm Z_t)$. Since direct mutual information calculation is infeasible, attention distribution is used as a first-order surrogate (Proposition 1). At the current scale, attention $\bm A^{t^*}$ from the highest-confidence time step $t^*$ is used to select Top-$K$ patches for the next scale. Complexity is reduced from $\mathcal{O}(N)$ to $\mathcal{O}(K)$, where $K \ll N$.
    - **Design Motivation**: Traditional MIL processes all patches, most of which are redundant. Attention-guided pruning concentrates computation on information-dense regions. Using attention from the most confident time step is more accurate than an average over steps.

3. **Confidence-Aware Early Stopping**:
    - **Function**: Dynamically determines when to stop reasoning based on current diagnostic uncertainty.
    - **Mechanism**: Calculates posterior $P(Y | \bm Z_t)$ and entropy $H(Y | \bm Z_t)$ at each step. Reasoning stops if entropy falls below the acceptance margin $\delta$; otherwise, it continues until $n$ steps are completed at the current scale before shifting scales. Confidence $C^t = 1 - \text{normalized entropy}$.
    - **Design Motivation**: Different cases vary in diagnostic difficulty. Adaptive early stopping allocates resources as needed, echoing the clinical practice of "reporting when certain, zooming in when not."

## Key Experimental Results

### Main Results: Four Pathology Diagnostic Tasks

| Task | Method | AUC↑ | Patch Count↓ | Inference (s)↓ | Gain |
|------|------|------|---------|------------|------|
| TCGA-BRCA Subtype | TransMIL | 88.6 | 12,500 | 28.4 | 1× |
| TCGA-BRCA Subtype | EAGLE | 88.2 | 3,200 | 7.8 | 3.6× |
| TCGA-BRCA Subtype | **Ours** | **89.3** | **506** | **1.3** | **21.8×** |
| TCGA-LUAD Grading | TransMIL | 76.5 | 10,800 | 24.7 | 1× |
| TCGA-LUAD Grading | **Ours** | **77.4** | **427** | **1.1** | **22.5×** |
| CAMELYON16 Metastasis| CLAM | 91.2 | 8,500 | 19.3 | 1× |
| CAMELYON16 Metastasis| **Ours** | **91.8** | **352** | **0.84** | **23.0×** |
| TCGA-RCC Subtype | TransMIL | 92.8 | 11,300 | 26.1 | 1× |
| TCGA-RCC Subtype | **Ours** | **93.5** | **474** | **1.2** | **21.7×** |

Average patch reduction of 95.95% and inference time reduction of 95.62%, with a mean AUC increase of +0.7.

### Ablation Study (TCGA-BRCA)

| Configuration | AUC | Patch Count |
|------|------|--------|
| Full PathCTM | 89.3 | 506 |
| − Scale-Space Reasoning (Single-scale CTM)| 85.4 | 8,200 |
| − Attention Pruning (No pruning, all patches)| 89.1 | 12,500 |
| − Early Stopping (Fixed steps)| 89.0 | 950 |

Scale-Space is the most critical component; pruning mainly saves computation with minimal AUC impact; early stopping saves nearly half the patches compared to fixed budget execution.

### Cross-scale Fusion vs. No Fusion

| Configuration | AUC |
|------|------|
| With $\bm S^{L,\max}$ Cross-scale Fusion | 89.3 |
| Only Current Scale $\bm S^{L-1,t}$ | 87.9 |

Cross-scale fusion (maintaining global context) provides a +1.4 AUC gain, proving that "global hypotheses + local verification" must coexist.

### Key Findings
- **WSI analysis is a dynamic reasoning problem**: By treating it as sequential decision-making rather than static aggregation, PathCTM achieves massive efficiency gains and AUC improvements.
- **Fewer patches can be more accurate**: Pruning removes noisy patches, allowing the model to focus its attention more effectively.
- **Module Synergy**: Scale switching, pruning, and early stopping manage different axes of optimization; the absence of any module significantly degrades efficiency.
- **Foundation Model Compatibility**: PathCTM can be applied atop any backbone (Virchow, GigaPath), lowering retraining costs.

## Highlights & Insights
- **"Thinking in Scales" is a logical extension of CTM**: By adding a spatial scale dimension to temporal iterations, "zooming when unclear" becomes a learnable action.
- **Paradigm shift from exhaustive to adaptive**: Unlike previous methods that optimize exhaustive processing (e.g., via distillation), PathCTM minimizes redundant processing entirely.
- **Clinical Alignment**: Matches pathologist behavior ("report when certain"), providing inherent interpretability through reasoning trajectory visualization.
- **Attention as an Information Gain Proxy**: Proposition 1 provides a first-order surrogate ($Attention \approx Influence Gradient$), giving theoretical support to attention-guided selectivity.

## Limitations & Future Work
- Currently validated only on classification; transfer to segmentation, detection, or survival prediction is untested.
- Scale switching involves discrete steps; continuous scale (NeRF-style) reasoning could be explored.
- The early stopping threshold $\delta$ is an empirical hyperparameter; per-case adaptation might be superior.
- Top-$K$ selection is a fixed budget; dynamic $K$ based on uncertainty could further optimize computation.
- Training still requires processing all scales (though inference does not), leading to higher training memory overhead.

## Related Work & Insights
- **vs. CLAM / TransMIL / ABMIL (MIL Baselines)**: These methods use static aggregation on exhaustive patches; PathCTM uses dynamic reasoning on sparse patches.
- **vs. ZoomMIL / HAG-MIL / EAGLE (Multi-scale MIL)**: These rely on rigid cascades; PathCTM uses continuous reasoning and adaptive stopping.
- **vs. CTM (Darlow 2026)**: CTM is designed for single-scale static images; PathCTM introduces the scale dimension for WSIs.
- **Insights**: The "thinking in X" paradigm is applicable to any hierarchical data with dynamic attention and varying sample difficulty (e.g., remote sensing, long video, ultra-long documents).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[ICML 2026\] DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring](discontinuous_galerkin_neural_operator_for_pathology_defocus_deblurring.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](../../CVPR2026/medical_imaging/apex_adaptive_visual_prompting.md)
- [\[ICML 2026\] Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis](learning_multi-scale_hypergraph_for_high-order_brain_connectivity_analysis.md)
- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)

</div>

<!-- RELATED:END -->
