---
title: >-
  [Paper Note] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos
description: >-
  [CVPR 2025][Object Detection][Novel Object Detection] This paper proposes the "Show, Don't Tell" paradigm, which automatically creates training datasets by watching human manipulation demonstration videos to train specialized object detectors for identifying novel objects. It completely bypasses the reliance on language descriptions or prompt engineering in traditional methods, significantly improving the performance of manipulating object detection and recognition on real-wo…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Novel Object Detection"
  - "Human Demonstration Videos"
  - "Self-Supervised Learning"
  - "Robotic Manipulation"
  - "Open-Set Detection"
date: 2026-05-08
content_hash: c6446591eb896146
---

# Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

**Conference**: CVPR 2025  
**arXiv**: [2603.12751](https://arxiv.org/abs/2603.12751)  
**Code**: None  
**Area**: Object Detection / Robotics  
**Keywords**: Novel Object Detection, Human Demonstration Videos, Self-Supervised Learning, Robotic Manipulation, Open-Set Detection

## TL;DR
This paper proposes the "Show, Don't Tell" paradigm, which automatically creates training datasets by watching human manipulation demonstration videos to train specialized object detectors for identifying novel objects. It completely bypasses the reliance on language descriptions or prompt engineering in traditional methods, significantly improving the performance of manipulating object detection and recognition on real-world robotic systems.

## Background & Motivation

**Background**: Robots need to quickly identify and detect novel objects during task execution. Current mainstream methods rely on closed-set object detectors (such as pre-trained object detection models), which often fail when facing out-of-distribution (OOD) objects. Although open-set detectors (such as VLMs) occasionally succeed, they require expensive and tedious manual prompt engineering to uniquely identify novel object instances.

**Limitations of Prior Work**: Closed-set detectors cannot generalize to unseen objects. While open-set detectors offer some flexibility, they face two key issues: (1) they require humans to provide precise language descriptions to distinguish object instances, a process that is time-consuming and error-prone; (2) language descriptions themselves are inherently ambiguous, making it difficult to precisely refer to specific objects.

**Key Challenge**: Utilizing language as an intermediate medium for object recognition presents fundamental limitations—for objects that have complex appearances or are difficult to describe precisely in words, textual descriptions struggle to capture all visual details to uniquely identify them.

**Goal**: (1) Eliminate the reliance on language descriptions and prompt engineering; (2) automatically generate training data using human demonstration videos; (3) rapidly train specialized detectors for task-specific objects.

**Key Insight**: The authors observe that human demonstration videos inherently contain rich visual information about the objects to be manipulated. Instead of using language to "tell" the detector what to look for, it is better to directly "show" it using videos.

**Core Idea**: Automatically create labeled training datasets from human demonstration videos to bypass language descriptions and directly train specialized detectors for task-related objects, realizing the new paradigm of "Show, Don't Tell."

## Method

### Overall Architecture
The overall pipeline of the system is: input a human task demonstration video → automatically extract and annotate regions of the manipulated objects from the video → train a specialized object detector using these automatically generated annotation data → deploy the detector on a real-world robot for object recognition and task execution.

### Key Designs

1. **Automatic Dataset Creation Module**:

    - **Function**: Automatically extract and annotate object instances from human demonstration videos.
    - **Mechanism**: Utilize motion information and visual cues in the video to automatically identify object regions manipulated by human hands. By tracking hand movements and object interactions, diverse visual samples of the objects under different viewpoints and lighting conditions are extracted, automatically generating bounding box annotations. This approach covers multiple viewpoints and pose variations of the object, providing rich positive samples.
    - **Design Motivation**: Human demonstration videos naturally contain the complete process of objects being grasped, moved, and placed. These frames themselves constitute a high-quality dataset of object appearances, requiring no additional manual annotation effort.

2. **Specialized Object Detector Training**:

    - **Function**: Rapidly train a task-specific object detection model using the automatically created dataset.
    - **Mechanism**: Train or fine-tune a lightweight detection network with the automatically generated annotations as supervision signals. The training objective is to enable the detector to distinguish specific object instances appearing in the demonstrations. By limiting the detection scope to task-relevant objects, the detector can converge quickly with high accuracy.
    - **Design Motivation**: Unlike general open-set detectors, the specialized detector only needs to recognize a limited number of specific objects, allowing it to achieve high accuracy with minimal data and computational overhead.

3. **Robot End-to-End Deployment System**:

    - **Function**: Integrate the trained detector into robot systems for closed-loop task execution.
    - **Mechanism**: Incorporate the specialized detector as the core of the robot's perception module, detecting task-relevant objects in the workspace in real-time to provide accurate object localization info for subsequent grasp planning and manipulation.
    - **Design Motivation**: End-to-end system integration ensures highly efficient transition from demonstration to deployment, minimizing human intervention.

### Loss & Training
Standard object detection losses are employed for training, including classification loss and bounding box regression loss. Since the training data is automatically generated from the demonstration videos, the entire training process is entirely self-supervised, requiring no human annotations.

## Key Experimental Results

### Main Results

| Method | Object Detection mAP | Object Recognition Accuracy | Task Success Rate |
|------|-------------|---------------|-----------|
| Closed-set Detector | Low | Low | Low |
| VLM + Prompt | Medium | Medium | Medium |
| Show Don't Tell (Ours) | **Significantly Highest** | **Significantly Highest** | **Significantly Highest** |

### Ablation Study

| Configuration | Detection Performance | Description |
|------|---------|------|
| Full pipeline | Optimal | Complete self-supervised training pipeline |
| w/o Automatic Data Augmentation | Drop significantly | Data diversity is crucial for generalization |
| w/o Multi-view Sampling | Decrease | Multi-view coverage improves robustness |
| Only using a few frames | Slight decrease | More frames provide better coverage |

### Key Findings
- The method that completely bypasses language descriptions significantly outperforms prompt-dependent approaches in detecting and identifying novel objects, validating the core hypothesis of "showing is better than telling."
- The quality of the automatically generated dataset is sufficient to train highly accurate specialized detectors.
- The system leads to a substantial improvement in manipulation success rates during real-world robot tasks.

## Highlights & Insights
- **Paradigm Innovation by Bypassing Language**: It completely avoids the bottleneck of language descriptions and directly establishes object representations using visual information, which opens up new avenues in human-robot interaction and robot learning.
- **Self-Supervised Data Flywheel**: The demonstration video itself serves as the annotation source. This "using data to generate data" strategy is highly efficient and can be extended to broader few-shot learning scenarios.
- **Practical Engineering Value**: The end-to-end integration into real robot systems validates the practicality of the method, offering not only academic contributions but also industrial application value.

## Limitations & Future Work
- It relies on the quality of human demonstration videos—if objects are severely occluded or contain motion blur in the demonstration, the quality of automatic annotations might degrade.
- Currently, it primarily targets manipulation objects; its ability to detect static objects in the background remains to be verified.
- The scenario of dynamic object class growth is not explored—when new tasks introduce more objects, retraining might be required.
- It would be worth considering combining a small amount of language prompts as a supplement to form a hybrid "Show + Tell" mode.
- Extending the method to more complex multi-object interaction scenarios is also an important direction.

## Related Work & Insights
- **vs OWL-ViT/Grounding DINO**: These open-set detectors rely on text prompts to specify targets, while this work completely bypasses the language loop and is more precise in specific object instance recognition.
- **vs Few-shot Object Detection**: Traditional few-shot detection requires humans to provide reference images, whereas this work automatically extracts them from demonstration videos, resulting in a more natural workflow.
- **vs Visual Prompting**: Visual prompting methods (e.g., SAM) require user-annotated points or boxes, while this work automatically extracts this information from videos.
- This work inspires a more general idea: using human behavioral data (not only videos but also tactile/force feedback) to automatically annotate training data.

## Rating
- Novelty: ⭐⭐⭐⭐ Paradigm innovation—"Show, Don't Tell" is a clear and convincing new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on real-world robot systems, though comparisons on open datasets could be richer.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the theme of "Show, Don't Tell" is consistently maintained throughout.
- Value: ⭐⭐⭐⭐ Directly valuable for the robotics manipulation and object detection communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios](anomalyncd_towards_novel_anomaly_class_discovery_in_industrial_scenarios.md)
- [\[ECCV 2024\] GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](../../ECCV2024/object_detection/gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)
- [\[NeurIPS 2025\] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](../../NeurIPS2025/object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[CVPR 2026\] ElasticFormer: Detecting Objects in HRW Shots via Elastic Computing Vision Transformer](../../CVPR2026/object_detection/elasticformer_detecting_objects_in_hrw_shots_via_elastic_computing_vision_transf.md)

</div>

<!-- RELATED:END -->
