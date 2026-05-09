---
title: >-
  [Paper Note] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos
description: >-
  [CVPR 2026][Object Detection][novel object detection] This paper proposes the "Show, Don't Tell" paradigm — automatically creating training datasets and training bespoke object detectors by watching human demonstration videos, entirely bypassing language descriptions and prompt engineering. The approach significantly outperforms state-of-the-art open-set/closed-set detectors on novel object recognition in real-world robotic scenarios.
tags:
  - CVPR 2026
  - Object Detection
  - novel object detection
  - self-supervised
  - human demonstration
  - bespoke detector
  - robot manipulation
date: 2026-05-08
content_hash: f5fb1eac578b6d07
---

# Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

**Conference**: CVPR 2026
**arXiv**: [2603.12751](https://arxiv.org/abs/2603.12751)
**Code**: None
**Area**: Object Detection / Robotics
**Keywords**: novel object detection, self-supervised, human demonstration, bespoke detector, robot manipulation

## TL;DR

This paper proposes the "Show, Don't Tell" paradigm — automatically creating training datasets and training bespoke object detectors by watching human demonstration videos, entirely bypassing language descriptions and prompt engineering. The approach significantly outperforms state-of-the-art open-set/closed-set detectors on novel object recognition in real-world robotic scenarios.

## Background & Motivation

**Background**: Accurate recognition and localization of target objects is a prerequisite for robotic manipulation tasks such as grasping and assembly. Current object detection methods fall into two main categories: closed-set detectors (YOLO, Faster R-CNN, etc.) perform well on predefined categories but cannot handle objects unseen during training; open-set detectors (e.g., VLM-based GroundingDINO, OWL-ViT) perform zero-shot detection via language descriptions, theoretically capable of handling arbitrary objects.

**Limitations of Prior Work**: Closed-set detectors fail outright on out-of-distribution (OOD) novel objects. While open-set detectors are theoretically viable, they suffer from severe practical issues in deployment — requiring humans to carefully craft text prompts for each new object (prompt engineering), a process that is both costly and unreliable. This is especially problematic when distinguishing visually similar but functionally distinct object instances (e.g., same-category products of different brands, tools of different colors), where natural language cannot provide sufficient discriminability.

**Key Challenge**: Language, as a medium for object description, has fundamental limitations — it excels at category-level semantics ("a cup") but is highly inefficient for instance-level precise identification. What is truly needed is a language-free, adaptive object recognition approach that can quickly learn to recognize specific objects from a single human demonstration.

**Goal**: (1) How to automatically extract object information from human demonstration videos and construct training datasets? (2) How to rapidly train a high-accuracy detector specialized for specific objects? (3) How to integrate the entire pipeline into a real robotic system for end-to-end deployment?

**Key Insight**: The authors observe that humans naturally display and manipulate target objects from multiple viewpoints when demonstrating manipulation tasks, and this process itself provides rich multi-view training data. Exploiting this "implicit supervision" can entirely bypass the bottleneck of language description.

**Core Idea**: Replace language descriptions with visual information from human demonstration videos to automatically create training datasets for bespoke object detectors, realizing the "Show, Don't Tell" paradigm for novel object recognition.

## Method

### Overall Architecture

The complete system pipeline consists of four stages: (1) **Demonstration Recording**: a human performs task operations within the robot's sensor field of view; (2) **Automated Dataset Creation**: a visual processing pipeline automatically extracts, segments, and annotates image regions of manipulated objects from the demonstration video; (3) **Bespoke Detector Training**: a lightweight object detector is rapidly trained on the automatically generated annotations; (4) **Robot Deployment**: the trained detector is integrated into the robot's perception-planning-execution loop for autonomous object recognition and manipulation. The entire process is fully automatic, requiring no manual annotation or language input from demonstration to deployment.

### Key Designs

1. **Self-supervised Dataset Creation Pipeline (Auto-Dataset Pipeline)**:

    - **Function**: Automatically extract training samples and annotations for target objects from human demonstration videos.
    - **Mechanism**: The natural motion and appearance variation of objects in demonstration videos are exploited as supervision signals. The pipeline first applies a class-agnostic segmentation network to detect candidate object regions in each frame, then uses motion analysis to identify which objects are being actively manipulated by the human (objects whose motion trajectories are consistent with hand motion). Confirmed manipulation targets are automatically cropped, augmented, and annotated with bounding boxes. The temporal sequence of video frames naturally provides diverse samples of the target object under varying viewpoints, illumination conditions, and occlusions. Annotation accuracy is ensured through multi-frame consistency verification — only objects consistently tracked and confirmed across multiple frames are included in the final dataset.
    - **Design Motivation**: Avoids the high cost of manual annotation while ensuring high consistency between training data and actual deployment scenarios. Traditional methods require pre-collecting and annotating large datasets, whereas this approach merges data collection with task demonstration.

2. **Bespoke Detector**:

    - **Function**: Rapidly train a specialized detector for only the objects required in the current task.
    - **Mechanism**: Rather than pursuing a large general-purpose model, this approach adopts a "small but precise" strategy. A lightweight detection network (e.g., a compact model based on YOLO or SSD architecture) is trained for each specific deployment scenario, requiring the model to distinguish only a small number of target object categories (typically 3–10), enabling training to complete within minutes. The small number of model parameters also reduces the amount of training data required, which matches the small-scale but high-quality output of the automated dataset pipeline. The detector's inference speed is sufficient to meet real-time robot control requirements.
    - **Design Motivation**: Specialized models achieve significantly higher accuracy than general-purpose models on specific tasks. Furthermore, rapid training supports online adaptation — the robot can quickly learn to recognize new objects for each new task.

3. **End-to-End Robot Integration System**:

    - **Function**: Integrate automated object recognition with robot manipulation planning into a complete closed-loop system.
    - **Mechanism**: The full "Show, Don't Tell" pipeline is deployed on the robot, including a multi-camera perception module, an automated dataset creation service, an online model training module, and a grasp planner driven by detection results. When a human demonstrates a new task, the system automatically triggers the data creation and model training workflow, then immediately switches to autonomous operation mode upon training completion. Real-time detection results are passed to the motion planner in the form of 6-DoF object poses, guiding the robot end-effector to perform precise grasping.
    - **Design Motivation**: Validates the complete feasibility from concept to real-world deployment, addressing engineering challenges that pure vision-based approaches often fail to overcome.

### Loss & Training

The bespoke detector follows a standard object detection training paradigm: the classification branch uses cross-entropy loss $\mathcal{L}_{cls} = -\sum_i y_i \log(\hat{y}_i)$, and the bounding box regression branch uses a combination of $\ell_1$ loss and GIoU loss $\mathcal{L}_{box} = \lambda_1 \|\mathbf{b} - \hat{\mathbf{b}}\|_1 + \lambda_2 \mathcal{L}_{GIoU}$. A rapid fine-tuning strategy (on the order of minutes) ensures low latency from demonstration to deployment. Training data are entirely generated by the automated pipeline with no manual annotation. Data augmentation strategies (color jitter, random cropping, affine transformations) further diversify training samples.

## Key Experimental Results

### Main Results: Object Detection and Task Completion Rate

| Method | Type | Novel Object Detection Accuracy | Instance Discrimination | Manual Prompting Required | End-to-End Task Completion Rate |
|--------|------|---------------------------------|------------------------|--------------------------|----------------------------------|
| Pretrained YOLO | Closed-set | Very low (OOD failure) | None | None | Low |
| GroundingDINO | Open-set | Moderate | Weak (text-quality dependent) | High (per-object prompts) | Moderate |
| OWL-ViT + CLIP | Open-set | Moderate–low | Weak | High (fine-grained prompts) | Moderate–low |
| Few-shot Detector | Few-shot | Moderate | Moderate | Medium (manual support set annotation) | Moderate |
| **Show, Don't Tell** | **Bespoke** | **Significantly best** | **Strong (instance-level)** | **Zero (fully automatic)** | **Highest** |

### Ablation Study: Contribution of Key Components

| Configuration | Change in Detection Performance | Notes |
|---------------|---------------------------------|-------|
| Full system | Baseline (best) | Auto-dataset + bespoke detector + multi-frame verification |
| Remove multi-frame consistency verification | Notable drop | Increased annotation noise, lower training data quality |
| Replace bespoke detector with large general model | Significant drop | General models insufficient for instance-level discrimination |
| Reduce demonstration video length (50%) | Slight drop | System exhibits some robustness to data volume |
| Single-frame object extraction only | Notable drop | Multi-view coverage is critical for detector generalization |
| Remove data augmentation | Moderate drop | Affine transformations and color jitter are important for small-dataset training |

### Key Findings

- **"Show" significantly outperforms "Tell"**: The bespoke detector substantially surpasses all language-based open-set methods on novel object detection, with a particularly pronounced advantage in instance-level discrimination (distinguishing different instances of the same category).
- **Automatically generated dataset quality is sufficient for training**: After multi-frame consistency verification, the quality of automatically extracted annotations is sufficient to train high-performance detectors.
- **Rapid adaptation capability**: A single human demonstration (a few minutes of video) per novel object enables the system to complete the full workflow from data creation to detector deployment within minutes.
- **Real-robot validation**: The integrated system is validated on real-world robotic manipulation tasks, and high detection accuracy directly translates to higher task success rates.
- **Multi-view coverage is critical**: Ablation experiments show that multi-frame extraction and multi-view data are essential for the generalization capability of the final detector.

## Highlights & Insights

- **Paradigm innovation is highly inspiring**: The paradigm shift from "Tell" (language description) to "Show" (visual demonstration) addresses a neglected issue in the VLM era — language is not the optimal interface for all visual recognition tasks. In scenarios requiring precise instance-level identification, direct visual alignment may be a more natural path. This insight is transferable to domains such as industrial quality inspection and personalized recommendation.
- **End-to-end engineering closure**: The work covers the complete pipeline from data collection, automatic annotation, and model training to robot deployment. This system-level engineering integration has high practical value and reproducibility.
- **"Small and specialized" beats "large and general"**: In specific application scenarios, a rapidly trained bespoke detector may be more effective than a large general-purpose open-set detector, offering a valuable counterpoint to the current trend of pursuing universal visual models.

## Limitations & Future Work

- **Lack of cross-scenario knowledge transfer**: Each new task/object combination requires training a detector from scratch, with no reuse of features learned in prior scenarios. Introducing a meta-learning mechanism could enable faster convergence with fewer demonstrations.
- **Dependence on demonstration video quality**: System performance is positively correlated with the quality of human demonstrations (lighting, occlusion, completeness of object display); more robust data extraction strategies may be needed in unstructured environments.
- **Bottleneck for near-identical objects**: When multiple objects are nearly visually indistinguishable, the discriminability of purely visual approaches may be limited; auxiliary cues (spatial position, grasping order) could be introduced.
- **Scalability**: The number of objects validated in the paper is small; scalability to large-scale scenarios involving dozens of objects (e.g., warehouse sorting) remains to be investigated.

## Related Work & Insights

- **vs. GroundingDINO / OWL-ViT**: These open-vocabulary detection methods rely on text prompts, whereas this paper entirely bypasses language. Open-set methods are more general at the category level, but "Show, Don't Tell" is more precise at the instance level.
- **vs. Few-shot Object Detection (FSOD)**: FSOD typically follows a meta-learning paradigm, requiring manually annotated support sets and relatively heavy models. This paper automatically constructs training sets from video and uses a lightweight bespoke model.
- **vs. Learning from Demonstration (LfD)**: LfD traditionally focuses on learning action policies from demonstrations. This paper innovatively extends "learning from demonstration" to the perception level, forming a complete perception-execution closed loop.

## Rating

- Novelty: ⭐⭐⭐⭐ A paradigm-level innovation — using visual demonstration instead of language description to teach a detector to recognize novel objects; the idea is concise and compelling.
- Experimental Thoroughness: ⭐⭐⭐ Includes real-robot validation and ablation analysis, though the full paper was not completely accessible and quantitative comparison details remain to be confirmed.
- Writing Quality: ⭐⭐⭐⭐ The "Show, Don't Tell" naming is precise and evocative, with a clear and coherent narrative.
- Value: ⭐⭐⭐⭐ Provides a practical and engineering-feasible solution for object recognition in robotic scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](detecting_unknown_objects_via_energy-based_separation.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[CVPR 2026\] PHAC: Promptable Human Amodal Completion](phac_promptable_human_amodal_completion.md)
- [\[CVPR 2026\] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)

</div>

<!-- RELATED:END -->
