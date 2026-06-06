---
title: >-
  [Paper Note] Does YOLO Really Need to See Every Training Image in Every Epoch?
description: >-
  [CVPR 2026][Object Detection][YOLO] This paper proposes the Anti-Forgetting Sampling Strategy (AFSS), which dynamically determines which training images participate in each epoch based on per-image learning sufficiency m…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "YOLO"
  - "training acceleration"
  - "adaptive sampling"
  - "anti-forgetting"
  - "data-efficient learning"
date: 2026-05-08
content_hash: 7f4ba17075a2024f
---

# Does YOLO Really Need to See Every Training Image in Every Epoch?

**Conference**: CVPR 2026
**arXiv**: [2603.17684](https://arxiv.org/abs/2603.17684)  
**Code**: None  
**Area**: Object Detection
**Keywords**: YOLO, training acceleration, adaptive sampling, anti-forgetting, data-efficient learning

## TL;DR

This paper proposes the Anti-Forgetting Sampling Strategy (AFSS), which dynamically determines which training images participate in each epoch based on per-image learning sufficiency measured by $\min(\text{Precision}, \text{Recall})$. AFSS achieves over 1.43× training speedup for YOLO-series detectors while maintaining or even improving detection accuracy.

## Background & Motivation

The YOLO series is renowned for its extremely fast inference speed (YOLO11s reaching 200 FPS), yet training is surprisingly time-consuming:
- YOLO11s requires 43.9 hours to train on COCO (dual RTX 4090), whereas Faster R-CNN takes only 6.5 hours on the same hardware.
- The root cause lies in YOLO's **full-coverage training paradigm**: every epoch traverses the entire training set, with each image processed hundreds of times.
- Once the model has sufficiently learned certain images, continuing to process them at the same frequency yields diminishing returns.

**Core Problem**: Does YOLO truly need to see every training image in every epoch? If not, can training be accelerated by dynamically selecting *what* and *when* to observe?

Limitations of existing alternatives:
- **Curriculum Learning**: Fixed easy-to-hard ordering leads to insufficient learning of hard samples.
- **Dataset Pruning**: Irreversible removal causes forgetting and bias.
- **Dataset Distillation**: Synthetic data lacks real-world diversity.

## Method

### Overall Architecture

Before each epoch begins, AFSS evaluates the learning sufficiency of every training image based on the state dictionary recorded in the previous round, classifying images into three levels — Easy, Moderate, and Hard — each with a distinct sampling strategy:
- **Easy**: Only 2% participate; periodic review prevents forgetting.
- **Moderate**: 40% participate; ensures short-term full coverage.
- **Hard**: 100% participate; ensures sufficient learning.

### Key Designs

#### 1. Learning Sufficiency Metric

Defined as the minimum of per-image detection precision and recall:

$$\text{Learning Sufficiency for } \mathbf{I}_i = \min(P_i, R_i)$$

This design emphasizes the bottleneck dimension — whether classification or localization is unreliable, the image is considered insufficiently learned. The three-tier partition is defined as follows:

| Level | Condition | Interpretation |
|-------|-----------|----------------|
| Easy | $\min(P_i, R_i) > 0.85$ | Model handles confidently |
| Moderate | $0.55 \leq \min(P_i, R_i) \leq 0.85$ | Partially stable; further optimization needed |
| Hard | $\min(P_i, R_i) < 0.55$ | Still challenging (occlusion, small objects, etc.) |

#### 2. Continuous Review — For Easy Images

Only 2% of Easy images participate per epoch, via two-stage sampling:

**Mandatory Review Set $\mathcal{A}_f'$**: $E_1$ images sampled from Easy images unused for more than 10 epochs, preventing forgetting due to long absence.

**Random Diversity Set $\mathcal{A}_r$**: $E_2$ images randomly sampled from the remaining Easy images to provide lightweight variation.

Constraints: $E_1 + E_2 = 0.02 \times |\mathcal{D}_{t-1}^1|$, $E_1 \leq 0.5 \times (E_1 + E_2)$.

#### 3. Short-Term Coverage — For Moderate Images

40% of Moderate images are used per epoch, ensuring each image appears at least once within every 3 epochs:

**Mandatory Coverage Set $\mathcal{B}_f$**: Images unused for 3 consecutive epochs are prioritized:

$$\mathcal{B}_f = \{(\mathbf{I}_i, P_i, R_i, ep_i) \in \mathcal{D}_{t-1}^2 \mid t - 1 - ep_i \geq 3\}$$

**Random Supplement Set $\mathcal{B}_r$**: Randomly sampled from the remaining Moderate images to reach 40% of the total.

#### 4. State Update

After training, the state dictionary $\mathcal{D}_t$ is updated to record each image's precision, recall, and last-used epoch:
- Precision/recall are refreshed every 5 epochs to reduce redundant evaluation.
- Usage records are updated in real time.

The final training set per epoch is: $\Omega = (\mathcal{A}_f' \cup \mathcal{A}_r) \cup (\mathcal{B}_f \cup \mathcal{B}_r) \cup \mathcal{D}_{t-1}^3$, with $|\Omega| < K$ (total number of images).

### Loss & Training

AFSS is an **architecture-agnostic** pure sampling strategy that modifies neither YOLO's loss functions nor model structure:
- Directly applied within the Ultralytics YOLO framework.
- 600 epochs (COCO) / 300 epochs (VOC/DOTA/DIOR-R).
- Default batch size 64, resolution 640×640 (COCO/VOC).

## Key Experimental Results

### Main Results

**Table 1: Training Acceleration on MS COCO 2017 + PASCAL VOC 2007**

| Model | COCO AP | Speedup | VOC mAP | Speedup |
|-------|---------|---------|---------|---------|
| YOLO11s | 47.0 | — | 81.7 | — |
| YOLO11s + AFSS | **47.2** | **1.54×** | **81.8** | **1.64×** |
| YOLO12x | 55.2 | — | 86.2 | — |
| YOLO12x + AFSS | **55.4** | **1.68×** | **86.4** | **1.69×** |

**Table 2: Remote Sensing Detection on DOTA-v1.0 + DIOR-R**

| Model | DOTA mAP | Speedup | DIOR-R mAP | Speedup |
|-------|----------|---------|-----------|---------|
| YOLO11x-OBB | 81.3 | — | 83.6 | — |
| YOLO11x-OBB + AFSS | **81.4** | **1.69×** | **83.7** | **1.70×** |

Consistent effectiveness across YOLOv8/v10/11/12 at all scales (n/s/m/l/x).

### Ablation Study

**Table 3: Comparison with Other Training Strategies (YOLO11s on COCO)**

| Method | AP | Speedup |
|--------|----|---------|
| Curriculum Learning | 43.7 | 1.35× |
| Self-Paced Learning | 44.5 | 1.30× |
| Dataset Pruning | 40.5 | 1.38× |
| Dataset Distillation | 35.6 | 1.50× |
| **AFSS** | **47.2** | **1.54×** |

**Table 4: Module Contribution Ablation**

| LSM | CR | STC | SU | AP | Speedup |
|-----|----|-----|-----|------|---------|
| ✓ | | | | 44.8 | 1.45× |
| ✓ | ✓ | | | 45.5 | 1.34× |
| ✓ | | ✓ | | 46.6 | 1.31× |
| ✓ | ✓ | ✓ | | 47.2 | 1.26× |
| ✓ | ✓ | ✓ | ✓ | **47.2** | **1.54×** |

State Update is the key to acceleration — without SU, accuracy is maintained but speedup is only 1.26×.

**Table 5: Comparison of Learning Sufficiency Metrics**

| Metric | AP | Speedup |
|--------|----|---------|
| Loss-based | 46.0 | 1.52× |
| Gradient-based | 46.9 | 1.45× |
| F1 score | 46.6 | 1.51× |
| **min(Prec, Rec)** | **47.2** | **1.54×** |

### Key Findings

1. **Larger models yield greater speedups**: from n (1.43×) to x (1.68×); larger models learn more efficiently, causing more images to become "Easy" faster.
2. **Optimal continuous review interval is 10 epochs**: shorter intervals (5) waste computation; longer intervals (20) cause forgetting (AP drops to 44.8).
3. **Optimal short-term coverage interval is 3 epochs**: at interval 5, AP drops to 44.2.
4. **Optimal state update interval is 5 epochs**: per-epoch updates impose high computational overhead (speedup only 1.26×); 15-epoch updates result in stale information.
5. The number of Hard images decreases steadily throughout training while Easy and Moderate counts increase, reflecting the model's learning progress.

## Highlights & Insights

1. **Insightful problem framing**: The work reveals the paradox between YOLO's "You Only Look Once" inference philosophy and its "look repeatedly" training paradigm.
2. **Simple and practical design**: No modifications to model architecture or loss functions; purely a plug-and-play data sampling strategy.
3. **Elegantly designed anti-forgetting mechanism**: Three-tier classification combined with mandatory review and short-term coverage achieves an excellent balance between acceleration and forgetting prevention.
4. **Unprecedented experimental scale**: Covers 4 YOLO versions (v8/v10/11/12) × 5 model scales (n/s/m/l/x) × 4 datasets.

## Limitations & Future Work

1. Learning sufficiency evaluation requires additional inference computation (though only once every 5 epochs), which may become a bottleneck for very large datasets.
2. The thresholds for Easy/Moderate/Hard (0.85/0.55) and sampling ratios (2%/40%/100%) are manually set without adaptive adjustment.
3. Validation is limited to the YOLO series; applicability to Transformer-based detectors such as DETR remains unexplored.
4. Inter-image correlations and complementarity (e.g., scene diversity) are not considered; classification is based solely on per-image learning sufficiency.

## Related Work & Insights

- **Curriculum / Self-Paced Learning**: Fixed easy-to-hard ordering leads to insufficient learning of hard samples; AFSS always retains hard samples.
- **Dataset Pruning** (Deep Learning on a Data Diet): Static, irreversible removal causes forgetting; AFSS is dynamic and reversible.
- **Dataset Distillation** (Fetch and Forge): Synthetic data lacks sufficient diversity; AFSS operates on real data subsets.
- The core idea of AFSS is generalizable to other long-training tasks such as segmentation, instance segmentation, and pose estimation.

## Rating

- **Novelty**: ★★★★☆ — Insightful problem framing; method is concise and effective.
- **Technical Depth**: ★★★☆☆ — The approach is primarily engineering-oriented with limited theoretical depth.
- **Experimental Thoroughness**: ★★★★★ — Extremely broad coverage, detailed ablations, and comprehensive comparisons.
- **Writing Quality**: ★★★★★ — Compelling title, fluent narrative, and intuitive figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](../../ICCV2025/object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[CVPR 2026\] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training](pet-dino_unifying_visual_cues_into_grounding_dino_with_prompt-enriched_training.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->
