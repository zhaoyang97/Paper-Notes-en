---
title: >-
  [Paper Note] AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection
description: >-
  [ECCV 2024][LLM (Other)][Zero-Shot Anomaly Detection] By concurrently incorporating static (globally shared) and dynamic (instance-specifically generated) learnable prompts into CLIP, and using auxiliary anomaly detection data for optimization, this method establishes a zero-shot SOTA on 14 industrial and medical anomaly detection datasets. The core innovation lies in the hybrid prompt design that achieves dual-tier adaptation at both the "task" and "instance" levels.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Zero-Shot Anomaly Detection"
  - "CLIP"
  - "Learnable Prompts"
  - "Static Prompts"
  - "Dynamic Prompts"
date: 2026-05-08
content_hash: 2db80170b878c24a
---

# AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection

**Conference**: ECCV 2024  
**arXiv**: [2407.15795](https://arxiv.org/abs/2407.15795)  
**Code**: [https://github.com/caoyunkang/AdaCLIP](https://github.com/caoyunkang/AdaCLIP) (open-source, 298⭐)  
**Area**: Medical Images  
**Keywords**: Zero-Shot Anomaly Detection, CLIP, Learnable Prompts, Static Prompts, Dynamic Prompts  

## TL;DR
By concurrently incorporating static (globally shared) and dynamic (instance-specifically generated) learnable prompts into CLIP, and using auxiliary anomaly detection data for optimization, this method establishes a zero-shot SOTA on 14 industrial and medical anomaly detection datasets. The core innovation lies in the hybrid prompt design that achieves dual-tier adaptation at both the "task" and "instance" levels.

## Background & Motivation
Zero-shot anomaly detection (ZSAD) requires a model to directly locate anomalies in unseen classes without requiring any training samples from those target categories. Existing methodology generally splits into two pathways: first, hand-crafted text prompts (e.g., "a photo of a damaged {}"), though these fixed templates fail to adapt to the complex anomaly patterns across different images; second, methods like APRIL-GAN or AnomalyCLIP introduce learning mechanisms, yet they either implement text-only adaptation or lack the capability to dynamically adjust on a per-image basis. The core pain point is that CLIP was never exposed to the downstream task of "anomaly detection" during pre-training, and static, fixed prompts exhibit limited expressivity when resolving highly diverse anomaly patterns across different domains (industrial vs. medical) and categories.

## Core Problem
How to enable high-quality zero-shot anomaly detection capabilities within CLIP without accessing any training data of target categories? This requires the model to apprehend "what anomaly detection is" (task-level adaptation) while dynamically adjusting its detection strategy for each distinct visual instance (instance-level adaptation).

## Method

### Overall Architecture
AdaCLIP is developed on top of a frozen CLIP ViT-L/14@336. Given an input image and a category name, the vision encoder extracts multi-level patch tokens while the text encoder generates normal and abnormal text feature pairs. Patch-level anomaly score maps and image-level anomaly scores are subsequently derived from visual-textual similarity calculations. The key improvement lies in injecting **hybrid learnable prompts** into the first $J$ layers (by default, 4 layers) of CLIP, operant simultaneously on both the visual and textual encoder branches.

### Key Designs

1. **Static Prompts**: A set of learnable parameters (length=5, depth=4) is maintained at each layer and globally shared across all images. Optimized through auxiliary training data, they enable CLIP to establish a broad understanding of "what is an anomaly." In the input sequence of each transformer block, static prompts are inserted—concatenated directly in the first layer, and replacing the previous layer's prompts in subsequent layers. This serves as "anomaly detection spectacles" for CLIP, acting as task-level adaptation.

2. **Dynamic Prompts**: The frozen CLIP vision encoder is first utilized to extract the CLS token feature of the target image, which is then projected through two learnable ProjectLayers to generate visual and text dynamic prompts, respectively. These dynamic prompts are **additively fused** with the static prompts before being injected into the transformer. Dynamic prompts allow the model to extract contextual cues from the input image first to guide the anomaly detection—such as focusing on solder joints for circuit boards, or fabric patterns for textile inspection—realizing instance-level adaptation.

3. **Hybrid Semantic Fusion (HSF)**: From the multi-level patch tokens, the top-$k$ most suspicious anomalies ($k = \text{clusters} \times 5$) are selected according to the initial anomaly score map. K-Means clustering ($k=20$) is then performed to extract cluster centers as semantic summaries. This module mitigates multi-level feature fusion discrepancies, as different blocks capture varied granularities of anomaly clues; HSF successfully abstracts consensus-based anomaly regions via clustering.

4. **Text Prompt Design**: An ensemble strategy over hand-crafted templates is employed. Normal conditions are described by 7 templates (e.g., "flawless {}", "perfect {}"), and abnormal states by 5 templates (e.g., "damaged {}", "{} with defect"). These are combined with 4 perspective templates (e.g., "a bad photo of a {}"), and the final text embeddings for all combinations are averaged to construct two highly representative class-agnostic vectors (normal/abnormal).

### Loss & Training
- **Classification Loss**: Image-level Focal Loss to supervise the overall anomaly score.
- **Segmentation Loss**: Layer-wise Focal Loss + Bidirectional Dice Loss (calculated respectively for normal and abnormal areas).
- **Total Loss**: Bound as $\text{classification\_loss} + \text{seg\_loss}$.
- **Optimizer**: AdamW ($\beta_1=0.5, \beta_2=0.999$), updating only the learnable parameters (static prompts, dynamic prompt generators, and projection layers), while freezing the CLIP backbone.
- **Mixed-precision training (FP16)** is adopted. The authors suggest executing multiple training runs to select the best checkpoint on the validation set.
- **Default Training Set**: MVTec AD + ColonDB; Validation Set: VisA + ClinicDB (demonstrating cross-domain training-to-testing).

## Key Experimental Results

**Industrial Anomaly Detection (7 datasets, Image-level AUROC / Pixel-level AUROC):**

| Dataset | Metric | AdaCLIP | Prev. SOTA | Compared Methods |
|--------|------|---------|----------|----------|
| MVTec AD | Image AUROC | ~91+ | ~88 (AnomalyCLIP) | APRIL-GAN, WinCLIP, SAA |
| VisA | Image AUROC | ~83+ | ~79 (AnomalyCLIP) | Same as above |
| 14-Dataset Avg | Image+Pixel | SOTA | - | Cross-domain Industrial+Medical |

*Note: Specific values vary slightly depending on the exact training subset configuration. The complete result graphs are available in the official GitHub repository.*

**Key Ablation Study:**

| Configuration | Description | Effect |
|------|------|------|
| S only | Static prompts only | Effective but insufficient |
| D only | Dynamic prompts only | Effective but inferior to hybrid |
| SD (hybrid) | Static + dynamic hybrid | Optimal, complementary performance |
| VL branch | Vision + text dual-branch prompts | Outperforms single-branch configurations |
| w/ HSF | Using hybrid semantic fusion | Further improves pixel-level detection accuracy |

### Key Findings
- The hybrid configuration of static + dynamic prompts yields improvements surpassing their individual sums, showing distinct functional complementarity (static prompts handling coarse task-level priors, while dynamic prompts guide fine instance-specific adjustments).
- Injecting prompts into both the vision and text branches concurrently yields significantly better results than single-branch configurations.
- The HSF module mainly enhances pixel-level localization precision, bringing less substantial gains to image-level selection.
- Generalization depends heavily on auxiliary training data diversity—incorporating more diverse data domains enhances testing robustness.

## Highlights & Insights
- **Elegant "static + dynamic" prompt integration**: Static prompts capture task-level priors ("what defines anomaly"), and dynamic prompts acquire instance-specific details ("what constitutes anomaly in this frame"). This clear division of labor optimizes the dual-tier adaptation.
- **Lightweight dynamic prompt generator**: Generated from the frozen CLIP CLS token through standard linear layer projections, avoiding complex, heavy auxiliary neural architectures.
- **Cross-domain generalization layout**: Jointly trained on MVTec (industrial) + ColonDB (medical) and evaluated on VisA (industrial) + ClinicDB (medical), demonstrating strong domain-agnostic properties.
- **Text prompt ensemble methodology**: Averaging text embeddings across multiple templates is significantly more robust than relying on a single phrase, echoing the classifier ensemble philosophy.

## Limitations & Future Work
- **FP16 training instability**: The authors mention having to perform multiple training sessions to optimize checkpoints, hinting at sensitivity to hyperparameters and initialization.
- **Reliance on auxiliary dataset**: Although labeled as a "zero-shot" evaluator on unseen classes, it heavily requires annotated auxiliary anomaly data for prompt tuning, presenting a deploying bottleneck.
- **K-Means bottleneck in HSF**: Computing K-Means per image incurs substantial inference latency, and the algorithm remains sensitive to initializations.
- **Localized patch perspective**: Lacks explicit modeling for structural, global, or long-range topological anomalies.
- **Potential improvements** $\rightarrow$ Swapping K-Means for faster clustering schemes, exploring self-supervised prompts lacking auxiliary labels, or integrating dynamic prompt generations directly with HSF for fully end-to-end adaptive validation.

## Related Work & Insights
- **vs. WinCLIP**: WinCLIP employs sliding windows and multi-scale combinations for zero-shot AD but is wholly reliant on frozen manually designed prompts. AdaCLIP introduces learning to guide CLIP in "understanding" anomaly boundaries.
- **vs. AnomalyCLIP**: AnomalyCLIP utilizes learnable prompting, but remains restricted to static object-agnostic text prompts. It lacks instance-wise adaptation, where AdaCLIP outperforms it by incorporating dynamic image-guided prompts.
- **vs. APRIL-GAN**: APRIL-GAN relies heavily on synthesis data generation for training. AdaCLIP, contrastingly, focuses entirely on optimized prompt designs; their concepts are orthogonal.
- **Core Difference**: AdaCLIP is the pioneer in marrying both static and dynamic prompt adjustments within joint vision-text branches for zero-shot anomaly detection.

## Insights & Connections
- This joint prompt paradigm is highly transferable to other zero-shot visual undertakings like zero-shot semantic segmentation and open-vocabulary detection, offering a standard template for "global task knowledge + localized instance adaptation."
- The swift dynamic prompt creation process (CLS token $\rightarrow$ linear projection $\rightarrow$ Transformer layer injection) remains incredibly lightweight and is worth adopting in diverse prompt learning circumstances.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The hybrid static-dynamic setup is clean and highly functional, though prompting itself is a widely discussed formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Richly verified on 14 distinct datasets (7 industrial and 7 medical), along with granular ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and systematic, though HTML-based rendering templates have occasional visual discrepancies.
- **Value**: ⭐⭐⭐⭐ Delivers definitive progression within the zero-shot anomaly detection domain. The code is well-maintained with high popularity (298 stars), providing solid research inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Zero-Shot Object Counting with Good Exemplars (VA-Count)](zeroshot_object_counting_with_good_exemplars.md)
- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](../../ACL2025/llm_nlp/bilingual_zero-shot_stance_detection.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ECCV 2024\] Reprojection Errors as Prompts for Efficient Scene Coordinate Regression](reprojection_errors_as_prompts_for_efficient_scene_coordinate_regression.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](../../ACL2025/llm_nlp/zero-shot_belief_a_hard_problem_for_llms.md)

</div>

<!-- RELATED:END -->
