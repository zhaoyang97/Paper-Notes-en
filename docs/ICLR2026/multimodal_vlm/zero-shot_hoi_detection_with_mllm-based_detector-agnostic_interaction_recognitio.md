---
title: >-
  [Paper Note] Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition
description: >-
  [ICLR 2026][Multimodal VLM][HOI detection] This paper proposes DA-HOI, a zero-shot HOI detection framework that fully decouples object detection from interaction recognition. It replaces conventional CLIP-based features…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "HOI detection"
  - "zero-shot"
  - "MLLM"
  - "interaction recognition"
  - "detector-agnostic"
date: 2026-05-08
content_hash: 4922acfe10c285b2
---

# Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition

**Conference**: ICLR 2026
**arXiv**: [2602.15124](https://arxiv.org/abs/2602.15124)  
**Code**: [https://github.com/SY-Xuan/DA-HOI](https://github.com/SY-Xuan/DA-HOI)  
**Authors**: Shiyu Xuan, Dongkai Wang, Zechao Li, Jinhui Tang
**Area**: Multimodal VLM
**Keywords**: HOI detection, zero-shot, MLLM, interaction recognition, detector-agnostic

## TL;DR

This paper proposes DA-HOI, a zero-shot HOI detection framework that fully decouples object detection from interaction recognition. It replaces conventional CLIP-based features with MLLM VQA capabilities for interaction recognition. The core contributions are deterministic generation (achieving 31.50 mAP training-free), spatial-aware pooling (incorporating spatial priors and cross-attention), and one-pass deterministic matching (reducing $M$ forward passes to one). DA-HOI comprehensively surpasses the state of the art across all four zero-shot settings on HICO-DET and supports plug-and-play detector substitution after training.

## Background & Motivation

**Background**: HOI detection requires simultaneously localizing humans and objects and recognizing their interactions. Recent CLIP-based zero-shot methods (GEN-VLKT, HOICLIP, ADA-CM, LAIN, etc.) have constructed interaction classifiers via text embeddings and achieved preliminary progress, yet performance remains substantially limited.

**Limitations of Prior Work**:

**Insufficient discriminative power of CLIP features**: CLIP excels at category-level alignment but lacks fine-grained discrimination for visually similar interactions such as "holding a cup" vs. "lifting a cup," requiring additional detector features for compensation.

**Tight coupling between detector and interaction recognition**: Two-stage methods including ADA-CM and BCOM couple their interaction recognition modules to specific detector features or inter-object relational modeling (e.g., UPT), necessitating retraining whenever the detector is changed—BCOM's Full mAP drops from 33.74 to 20.31 when switching to Grounding-DINO.

**Low generalization ceiling**: CLIP-based methods inherently align visual and textual features only on training categories, making generalization to unseen verb/object categories difficult.

**Key Challenge**: Open-vocabulary detectors can already localize unseen objects reasonably well; the true bottleneck lies in interaction recognition—which is precisely entangled with a specific detector.

**Key Insight**: MLLMs, trained on large-scale image-text pairs and instruction-following tasks, possess cross-modal generalization and fine-grained understanding capabilities far superior to CLIP. By decomposing HOI detection into two independent pipelines—detection for localization and MLLM for interaction recognition—each component can leverage the strongest available model, and the decoupled architecture enables plug-and-play flexibility.

**Core Idea**: Interaction recognition is formulated as a VQA task posed to an MLLM, with deterministic generation for multi-label confidence scoring, spatial-aware pooling to inject spatial priors, and one-pass matching to eliminate redundant inference overhead.

## Method

### Overall Architecture

DA-HOI decouples HOI detection into two fully independent stages:

1. **Object detection stage**: Any detector (DETR / Grounding-DINO / Yolo-World) produces detection results $\{C^i, B^i\}_{i=1}^{N_{\text{det}}}$.
2. **Interaction recognition stage**: All human-object pairs are enumerated; for each pair $(B_h, B_o, C_o)$, a VQA prompt is constructed and fed to the MLLM (Qwen2.5-VL) to predict interaction confidence scores.

The only interface between the two stages is bounding box coordinates and category labels; no features are shared. Consequently, the detector can be freely replaced after training without retraining the recognition module.

### Key Designs

#### 1. Deterministic Generation

- **Function**: Converts the MLLM's open-ended text generation into deterministic multi-label classification, eliminating format errors and single-output bias.
- **Mechanism**: Rather than allowing the MLLM to freely generate text answers, for each candidate interaction $T_k$ in the candidate set $\Theta(C_o) = \{T_1, T_2, \dots, T_M\}$, the conditional likelihood of the MLLM generating $T_k$ given the prompt is computed as the confidence score:
  $$S_v[k] = p(T_k | I, Q) = \prod_{j=1}^{N} p(t[j] | T_k[<j], I, Q)$$
- **Design Motivation**: Directly querying the MLLM suffers three critical failures: (a) format error rate as high as 36.78%; (b) severe single-output bias, with 80.91% of responses containing only one interaction (despite IR being a multi-label problem); (c) inability to obtain confidence scores required for mAP evaluation. Deterministic generation eliminates all three issues, reducing both format error rate and single-output rate to 0%.
- **Novelty**: Unlike CLIP-based methods such as ADA-CM, which compute visual-text similarity for classification, this work leverages the MLLM's stronger cross-modal understanding via conditional generation probability. Even without any training, the approach achieves 31.50 mAP, surpassing ADA-CM's 25.19.

#### 2. Spatial-Aware Pooling (SAP)

- **Function**: Integrates appearance features with pairwise spatial priors to enhance interaction representations, while filtering non-interactive pairs to reduce computation.
- **Mechanism**: Interaction features are constructed in three steps:
    - (a) Human and object features $f_h, f_o$ extracted via ROIAlign are fused through an MLP to produce an initial interaction feature $f_{\text{inter}}$.
    - (b) A cross-attention layer aggregates contextual information from global image features outside the bounding boxes, mitigating information loss from imprecise detections.
    - (c) A 7-dimensional pairwise spatial vector is encoded:
  $$U = [w_h h_h, w_o h_o, \frac{w_h}{h_h}, \frac{w_o}{h_o}, \text{IoU}(B_h, B_o), \frac{x_h - x_o}{w_h}, \frac{y_h - y_o}{h_h}]$$
  capturing area (distinguishing object sizes), aspect ratio (distinguishing shapes), IoU (measuring human-object overlap), and human-to-object direction (distinguishing left/right/up/down relationships). This vector is projected via MLP and additively fused with the interaction feature.
- **Design Motivation**: ROIAlign features are confined to the bounding box and are sensitive to imprecise detections (partial occlusion, background interference); they also ignore the relative spatial relationship between the human-object pair, which is critical for distinguishing "sit on chair" from "stand next to chair." Ablations show that removing spatial encoding reduces UO Full by 1.62, and removing cross-attention reduces it by 2.23.
- **Additional Function**: A linear classifier $S_{\text{interactiveness}} = \sigma(\text{Linear}(f_{\text{inter}}))$ trained on interaction features filters non-interactive pairs at inference, reducing inference time from 569 ms to 217 ms.

#### 3. One-Pass Deterministic Matching (DM)

- **Function**: Compresses interaction score computation from $M$ forward passes into a single forward pass.
- **Mechanism**: A special token `<|hoi|>` is appended after each candidate in the candidate list, and all candidates are concatenated into a single prompt fed to the LLM. The output feature $\hat{f}_{\text{hoi}}[k]$ at each special token and the interaction feature $\hat{f}_{\text{inter}}$ are extracted; cosine similarity replaces the conditional generation probability:
  $$S_v[k] = \text{cosine}(\hat{f}_{\text{hoi}}[k], \hat{f}_{\text{inter}})$$
- **Design Motivation**: Although deterministic generation is effective, its computation scales linearly with the number of candidates $M$. On HICO-DET, each object category has ~15 candidate interactions on average, requiring 15 LLM forward passes per human-object pair. DM reformulates generation as feature matching, obtaining scores for all candidates in a single forward pass.
- **Efficiency Gain**: SAP + DM jointly reduce inference time from the baseline 569 ms to 91 ms (6.3× speedup).

### Loss & Training

Two-stage training with the visual encoder frozen throughout:

1. **Stage 1**: Train SAP only (30 epochs, lr=1e-4, batch=16) using Binary Focal Loss for interactiveness prediction and spatial encoding.
2. **Stage 2**: Freeze SAP; fine-tune the LLM only with LoRA (16 epochs, lr=1e-4, batch=16) using Focal BCE for deterministic matching.

The final inference confidence score is: $\hat{S}^i_v[k] = S^i_v[k] \cdot S^i_{\text{interactiveness}} \cdot S^i_h \cdot S^i_o$, integrating the interaction score, interactiveness score, and detector confidence. All experiments are conducted on 4 RTX 3090 GPUs.

## Key Experimental Results

### Main Results: Zero-shot Performance on HICO-DET

| Method | RF-UC Full | NF-UC Full | UO Full | UV Full | Avg Full |
|------|-----------|-----------|---------|---------|----------|
| GEN-VLKT | 30.56 | 23.71 | 25.63 | 28.74 | 27.16 |
| HOICLIP | 32.99 | 27.75 | 28.53 | 31.09 | 30.09 |
| CLIP4HOI | 34.08 | 28.90 | 32.58 | 30.42 | 31.50 |
| LAIN | 34.41 | 33.23 | 34.27 | 33.12 | 33.76 |
| EZ-HOI | 36.73 | 34.84 | 36.38 | 36.84 | 36.20 |
| BC-HOI (BLIP2) | 40.99 | 36.40 | 34.18 | 39.89 | 37.87 |
| **DA-HOI (Ours)** | **43.56** | **40.33** | **43.60** | **42.88** | **42.59** |
| Ours + Grounding-DINO | 44.81 | 41.51 | 45.28 | 44.43 | 44.00 |
| Ours + Yolo-World | 44.00 | 42.01 | 44.82 | 43.88 | 43.68 |
| ADA-CM (training-free) | - | - | 25.19 | 25.19 | 25.19 |
| **Ours (training-free)** | - | - | **31.50** | **31.50** | **31.50** |

### Ablation Study: Component Contributions & Inference Efficiency

| Configuration | UO Full | UV Full | Inference Time (ms/img) |
|------|---------|---------|-----------------|
| Baseline (SFT + Det. Gen.) | 39.24 | 37.84 | 569 |
| + SAP only | 42.31 | 41.95 | 217 |
| + DM only | 40.50 | 39.24 | 189 |
| + SAP + DM (Full) | **43.60** | **42.88** | **91** |
| Full − Pairwise Spatial | 41.98 | 40.77 | 86 |
| Full − Cross Attention | 41.37 | 40.74 | 87 |
| Replace SAP with UPT | 41.76 | 40.58 | 122 |

### Key Findings

- **Deterministic generation is the most critical design**: In the training-free setting, it improves performance from 14.23 mAP (naive QA) to 31.50 mAP (+17.27), exceeding the cumulative gain of all fine-tuning components. Even after SFT, omitting deterministic generation yields only 31.61; adding it raises performance to 39.87 (+8.26).
- **SAP is the strongest fine-tuning component**: UO Full +3.07, UV Full +4.11, with simultaneous 2.6× inference speedup (569→217 ms).
- **DM is an efficient accelerator**: SAP+DM jointly reduce inference from 217 ms to 91 ms while further improving performance.
- **MLLM scale effect is pronounced**: LLaVA-0.5B (42.00) → Qwen-3B (43.60) → Qwen-7B (45.99), demonstrating that the method directly benefits from stronger MLLMs.
- **Cross-dataset generalization is outstanding**: HICO-DET→V-COCO achieves 59.91%, surpassing the second-best BCOM (48.87) by 11.04 points and CMMP by 12.26 points.
- **Robustness to candidate ordering**: Full mAP fluctuates by only ±0.02 across 5 different candidate orderings.
- **LoRA outperforms full fine-tuning**: LoRA-only LLM tuning matches or exceeds full tuning, confirming that preserving pretrained MLLM knowledge is beneficial.

## Highlights & Insights

- **Decoupled design is a paradigm-level innovation**: This is the first work to decompose HOI detection into fully independent detection and recognition modules, enabling free detector substitution after training. This allows HOI detection to benefit "for free" from advances in object detection (switching to Grounding-DINO directly improves performance by 1.41) and is transferable to compositional visual understanding tasks such as scene graph generation.
- **Deterministic generation elegantly bridges the gap between generative MLLMs and discriminative tasks**: Using conditional likelihood in place of text generation converts a generative model into a discriminator without any architectural modification. This technique is directly transferable to any scenario requiring multi-label classification or ranking with MLLMs (e.g., attribute recognition, action classification).
- **SAP outperforms the widely adopted UPT**: UPT models relationships across different detections, creating implicit coupling to the detector. SAP focuses solely on the spatial relationship of the current human-object pair and global image features, maintaining decoupling while achieving superior performance.

## Limitations & Future Work

- **Inference efficiency remains improvable**: 91 ms/image ≈ 11 FPS is insufficient for real-time scenarios (autonomous driving, robotics). MLLM knowledge distillation into lightweight models, or batched inference over multiple human-object pairs, warrants investigation.
- **Exhaustive pairing is suboptimal**: The number of human-object pairs scales as $O(N^2)$, leading to substantial redundancy in dense scenes. Learning pairing priors or applying spatial heuristics for pre-filtering could help.
- **Candidate interaction lists must be predefined**: Deterministic generation and matching rely on predefined candidate lists, limiting applicability to fully open-vocabulary interaction discovery.
- **MLLM deployment cost is high**: Even the smallest Qwen2.5-VL 3B model has 3B parameters; mobile deployment requires quantization or distillation.
- **Limited training data diversity**: Training is conducted solely on HICO-DET (600 HOI categories, 80 object categories); validation in more open real-world scenarios remains insufficient.

## Related Work & Insights

- **vs. EZ-HOI**: EZ-HOI also enhances zero-shot capability but remains grounded in CLIP feature alignment; this work replaces CLIP with MLLM for IR, achieving an Avg Full gain of 6.39 (42.59 vs. 36.20), demonstrating the superiority of MLLM cross-modal understanding over CLIP visual-language alignment.
- **vs. BC-HOI**: BC-HOI employs an MLLM (BLIP2) for auxiliary caption supervision but remains coupled to a specific detector; this work uses the MLLM directly for interaction discrimination with full decoupling, achieving a UO Full advantage of 9.42 (43.60 vs. 34.18), confirming that MLLMs should directly participate in discrimination rather than merely providing auxiliary signals.
- **vs. ADA-CM / BCOM**: These methods claim detector-agnosticism, yet their performance collapses upon detector substitution (BCOM: 33.74→17.69) due to implicit reliance on inter-object relationships modeled during training. This work achieves genuine decoupling, with performance improving rather than degrading upon detector substitution.
- **Inspiration**: The deterministic generation approach is transferable to any task requiring structured discriminative output from MLLMs, including scene graph generation, action recognition, and visual grounding.

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupled framework and deterministic generation represent substantive innovations, though individual sub-components (ROIAlign, cross-attention, LoRA) are established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four zero-shot settings, cross-detector and cross-dataset transfer, training-free and fully supervised baselines, multi-MLLM ablations, and training strategy ablations—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, rigorous formulations; some sections are slightly redundant.
- Value: ⭐⭐⭐⭐⭐ Proposes a new paradigm for HOI detection in the MLLM era; the decoupled design carries strong practical engineering value and academic impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[CVPR 2026\] Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](../../CVPR2026/multimodal_vlm/beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](../../CVPR2026/multimodal_vlm/crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](../../CVPR2026/multimodal_vlm/flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)

</div>

<!-- RELATED:END -->
