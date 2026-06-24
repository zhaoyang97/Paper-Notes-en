---
title: >-
  [Paper Note] Can OOD Object Detectors Learn from Foundation Models?
description: >-
  [ECCV 2024][Object Detection][OOD Object Detection] SyncOOD proposes an automated data curation method that leverages LLMs to imagine semantically novel OOD concepts and performs region-level editing on ID images via Stable Diffusion Inpainting to synthesize scene-level OOD samples. After refining bounding boxes with SAM and filtering via feature similarity, a lightweight MLP classifier is trained, substantially outperforming SOTA on multiple OOD detection benchmarks with a m…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "OOD Object Detection"
  - "Synthetic Data"
  - "Foundation Models"
  - "Stable Diffusion"
  - "Scene-Level Editing"
date: 2026-05-08
content_hash: 630e01c5bf1882d9
---

# Can OOD Object Detectors Learn from Foundation Models?

**Conference**: ECCV 2024  
**arXiv**: [2409.05162](https://arxiv.org/abs/2409.05162)  
**Code**: [GitHub](https://github.com/CVMI-Lab/SyncOOD)  
**Area**: Object Detection  
**Keywords**: OOD Object Detection, Synthetic Data, Foundation Models, Stable Diffusion, Scene-Level Editing

## TL;DR

SyncOOD proposes an automated data curation method that leverages LLMs to imagine semantically novel OOD concepts and performs region-level editing on ID images via Stable Diffusion Inpainting to synthesize scene-level OOD samples. After refining bounding boxes with SAM and filtering via feature similarity, a lightweight MLP classifier is trained, substantially outperforming SOTA on multiple OOD detection benchmarks with a minimal amount of synthetic data.

## Background & Motivation

**Background**: Modern object detectors achieve excellent performance on closed-set data, but often misclassify OOD classes as ID classes in open-world applications, threatening deployment reliability. OOD object detection aims to identify and label unknown objects.

**Limitations of Prior Work**:
   - Most methods synthesize OOD data in the detector's latent space (e.g., VOS, SR-VAE, DFDD), which is limited by latent space quality and lacks interpretability.
   - Adversarial sample methods (e.g., SAFE) lack semantic diversity.
   - Video pseudo-supervision methods introduce additional data requirements.
   - All methods are bound to closed-set distributions, potentially biasing towards ID datasets.

**Key Insight**: Can foundation models (LLM + Stable Diffusion + SAM) trained on massive open data be leveraged to synthesize high-quality OOD samples? Two key observations: (1) "Hard" OOD samples close to ID data are more beneficial for learning precise decision boundaries; (2) context acts as a distracting factor in OOD detection.

## Method

### Overall Architecture

SyncOOD consists of two core phases: (1) **OOD Data Synthesis**—utilizing foundation models to automatically generate annotated, scene-level OOD images; (2) **OOD Detector Training**—optimizing ID/OOD decision boundaries via hard-sample mining and a lightweight classifier. The entire pipeline is fully automated and requires almost no human annotation. The key lies in decoupling OOD synthesis into four steps: "concept discovery $\rightarrow$ region-level editing $\rightarrow$ annotation refinement $\rightarrow$ sample filtering".

### Key Designs

**1. LLM-Driven Novel Concept Imagination (Step 1)**

- Based on ID category labels, GPT-4 is utilized via in-context learning to brainstorm $M$ semantically novel, visually similar, and context-compatible OOD concepts for each ID class.
- LLM ensures semantic separability: the imagined novel concepts are semantically separated from the ID classes.
- Experiments show that a single in-context example is sufficient to discover high-quality novel concepts.
- Concepts overlapping with OOD classes in the test set are removed to prevent information leakage.

**2. Region-Level Image Editing (Step 2)**

- Use Stable-Diffusion-Inpainting for box-conditioned editing: $\mathbf{x}^{\text{edit}} = \text{SDI}(\mathbf{x}^{\text{id}}, \mathbf{b}^{\text{id}}, \mathbf{y}^{\text{novel}})$
- The bounding box of the ID object is used as the editing mask, and the novel concept acts as the text prompt condition.
- Key advantage: keeps the original scene context unchanged, replacing only the object inside the target region, which eliminates context bias interference.

**3. SAM Bounding Box Refinement (Step 3)**

- Due to the randomness of diffusion models, the position/size of the edited object may shift.
- SAM is used within the padded region of the edited area to obtain the instance mask with the highest confidence.
- After converting the mask to a bounding box, the IoU with the original box is calculated to filter out samples with excessive scale changes ($\text{IoU} > \gamma$).

**4. Feature Similarity-Based Hard Sample Mining (Step 4)**

- The pre-trained detector is used to extract latent space features of ID/OOD object pairs.
- Visually similar but semantically distinct OOD samples are filtered based on cosine similarity: $\epsilon_{\text{low}} < \text{sim}(\mathbf{z}^{\text{edit}}, \mathbf{z}^{\text{id}}) < \epsilon_{\text{up}}$
- Too high similarity indicates failed editing, while too low indicates image distortion.
- Filter out hard OOD samples that are "just confusing enough".

### Loss & Training

- Train a lightweight 3-layer MLP as a plug-and-play OOD detector.
- Use standard binary classification loss to optimize the ID/OOD decision boundary.
- The detector parameters remain frozen; only the additional MLP is trained, so the ID performance (mAP) is unaffected.
- Following SAFE, multi-scale features are extracted as training samples.
- Learning rate: PASCAL-VOC=1e-4, BDD-100K=5e-5, momentum=0.9, dropout=0.5, batch=32.

## Key Experimental Results

### Main Results

| Method | VOC→COCO FPR95↓ | VOC→OI FPR95↓ | BDD→COCO FPR95↓ | BDD→OI FPR95↓ |
|------|:---:|:---:|:---:|:---:|
| MSP | 70.99 | 73.13 | 80.94 | 79.04 |
| VOS | 47.53 | 51.33 | 44.27 | 35.54 |
| SAFE | 47.40 | 20.06 | 32.56 | 16.04 |
| DFDD | 41.34 | 44.52 | 30.71 | 22.67 |
| **Ours+FRCNN** | **36.44** | **13.34** | **22.67** | **12.96** |
| **Ours+VOS** | **34.97** | **11.25** | **23.09** | **14.12** |

Substantially outperforms SOTA on all benchmarks, utilizing only about 25% (VOC) and 20% (BDD) of the auxiliary data compared to SAFE.

### Ablation Study

| Ablation Item | FPR95(COCO/OI)↓ | Analysis |
|--------|:---:|------|
| Synthetic data size 2k→14k | 37.82~36.70 / 13.87~12.96 | Highly stable performance, high data efficiency |
| Number of concepts 3→8 | 36.96~37.91 / 13.15~13.58 | Insensitive to the number of concepts |
| W/o SAM refinement | 39.55 / 13.72 | SAM provides more accurate bounding boxes |
| W/o similarity filtering | 39.29 / 13.68 | Filter effectively removes noisy samples |
| Using object-centric images | 51.99 / 20.70 | Scene-level editing far outperforms pure object images |
| Scene-level w/o bounding boxes | 48.01 / 18.61 | Precise OOD bounding box annotations are essential |

### Key Findings

- **Extremely high data efficiency**: Only 2k synthetic samples are enough to approach peak performance, significantly superior to SAFE which requires 16k+ samples.
- **Scene-level editing is crucial**: Pure object images and full-image OOD are far inferior to region-level editing.
- **Context consistency is key**: Even slight background modifications lead to significant shifts in detector features.
- **Sweet spot in similarity range**: >0.9 indicates failed editing, whereas too low indicates image distortion.

## Highlights & Insights

- **First to achieve photorealistic scene-level OOD synthesis**: Extends OOD synthesis from the latent space to the pixel space.
- **Seamlessly orchestrating LLM, SD, and SAM**: Effectively cascades concept imagination $\rightarrow$ image editing $\rightarrow$ annotation refinement, leveraging the unique strengths of each.
- **Explicitly decoupling synthesis and selection**: Semantic separability is guaranteed by the LLM, whereas visual similarity is ensured by feature filtering.
- **Plug-and-play design**: Superimposes OOD detection capability onto any detector without modifying the original model.

## Limitations & Future Work

- Synthesis quality is constrained by the capabilities of Stable Diffusion and SAM; certain concepts might fail to edit.
- LLM-imagined concepts must exclude classes overlapping with test-set OOD, posing a risk of prior knowledge leakage.
- Only evaluated on two detectors (Faster R-CNN and VOS); more modern detector architectures have not been explored.
- Computational overhead (GPT-4 API calls, SD generation time) is not discussed.
- Can be extended to multi-modal OOD detection such as 3D and video.

## Related Work & Insights

- **VOS**: A classic method that samples OOD samples in the latent space, acting as the direct baseline for SyncOOD.
- **SAFE**: A similar framework but utilizes adversarial noise, with data efficiency far lower than SyncOOD.
- **Dream-OOD**: Uses diffusion models to synthesize OOD data for image classification, which is inapplicable for detection tasks.
- **Insight**: The open-world knowledge in foundation models can be efficiently injected into downstream detection tasks via automated pipelines.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](../../CVPR2026/object_detection/anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[ECCV 2024\] YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)
- [\[ECCV 2024\] On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines](on_calibration_of_object_detectors_pitfalls_evaluation_and_baselines.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)
- [\[CVPR 2025\] Interpreting Object-level Foundation Models via Visual Precision Search](../../CVPR2025/object_detection/interpreting_object-level_foundation_models_via_visual_precision_search.md)

</div>

<!-- RELATED:END -->
