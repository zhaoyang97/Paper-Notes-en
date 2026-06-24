---
title: >-
  [Paper Note] Functionality Understanding and Segmentation in 3D Scenes
description: >-
  [CVPR 2025][3D Vision][Functional Understanding] Fun3DU introduces the first approach for functional understanding in 3D scenes. By leveraging LLM chain-of-thought to parse task descriptions, utilizing VLMs to localize and segment functional objects across multi-view images, and applying 2D-3D voting aggregation, it substantially outperforms open-vocabulary 3D segmentation baselines on SceneFun3D (mIoU +13.2).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Functional Understanding"
  - "3D Scene Segmentation"
  - "Vision-Language Models"
  - "Zero-shot Reasoning"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: 6069b26b78da45e2
---

# Functionality Understanding and Segmentation in 3D Scenes

**Conference**: CVPR 2025  
**arXiv**: [2411.16310](https://arxiv.org/abs/2411.16310)  
**Code**: [https://tev-fbk.github.io/fun3du/](https://tev-fbk.github.io/fun3du/)  
**Area**: 3D Vision  
**Keywords**: Functional Understanding, 3D Scene Segmentation, Vision-Language Models, Zero-shot Reasoning, Chain-of-Thought

## TL;DR
Fun3DU introduces the first approach for functional understanding in 3D scenes. By leveraging LLM chain-of-thought to parse task descriptions, utilizing VLMs to localize and segment functional objects across multi-view images, and applying 2D-3D voting aggregation, it substantially outperforms open-vocabulary 3D segmentation baselines on SceneFun3D (mIoU +13.2).

## Background & Motivation

**Background**: 3D scene understanding has primarily centered around semantic or instance segmentation—identifying common furniture objects such as tables, chairs, and cabinets. Recent open-vocabulary 3D segmentation methods (OpenMask3D, LERF, OpenIns3D) can localize objects in 3D scenes using natural language descriptions.

**Limitations of Prior Work**: Functional understanding fundamentally differs from traditional object segmentation. Given a task description like "turn on the ceiling light", the system must understand that the action requires operating a light switch (which is not explicitly mentioned in the description) and then localize the switch in the 3D scene. This requires both world knowledge (to reason about functional objects) and fine-grained spatial awareness (to localize small interactive components like handles, buttons, and knobs). Existing open-vocabulary methods are heavily biased towards segmenting large furniture objects, rendering them completely ineffective for small functional objects.

**Key Challenge**: Open-vocabulary 3D segmentation methods rely on 3D proposal modules pre-trained on 3D datasets (such as ScanNet) that bias towards large objects, leaving the models with near-zero capability to identify small interactive components (handles, knobs, buttons). Furthermore, task descriptions typically do not mention the functional objects directly, demanding reasoning.

**Goal**: How can we localize and segment functional interactive elements in real 3D scenes based on natural language task descriptions without any task-specific training?

**Key Insight**: Although 3D data is too scarce to train models capable of understanding functionality, 2D pre-trained vision-language models (VLMs) possess rich world knowledge and fine-grained visual perception. By combining multiple pre-trained 2D models (LLM for task understanding, VLM for object localization, and SAM for segmentation), recognition can be performed on 2D views and subsequently projected back into 3D point clouds.

**Core Idea**: Utilize LLM chain-of-thought reasoning to infer functional object names, point to and segment these objects in selected views using a VLM, and aggregate the predictions into 3D point clouds via multi-view voting under a zero-training pipeline.

## Method

### Overall Architecture
The input consists of scene point clouds, multi-view RGB-D images, and task descriptions. The workflow proceeds through four modules: (1) LLM parses the task description to extract the functional object $F$ and context object $O$; (2) an open-vocabulary segmenter localizes the context object $O$ in all views and selects an optimal subset of views; (3) VLM localizes and segments the functional object $F$ in the selected views; (4) the 2D segmentations are projected back to the point cloud using 2D-3D correspondences and aggregated through multi-view voting.

### Key Designs

1. **Task Description Understanding (Chain-of-Thought Reasoning)**:

    - **Function**: Infer the functional objects $F$ to be segmented and the contextual objects $O$ containing them from natural language task descriptions.
    - **Mechanism**: LLM (Llama3.1-9B, 4-bit quantized) is employed via Chain-of-Thought reasoning. The system prompt configures the LLM's role as an assistant for an embodied robotic manipulator and provides a list of executable actions as a "stopping criterion". The LLM is first prompted to list the sequence of actions needed to complete the task (preventing incorrect levels of abstraction) and then extracts the hierarchical relationship between $F$ (e.g., "door handle") and $O$ (e.g., "cabinet").
    - **Design Motivation**: Querying the LLM directly leads to two issues: (a) ambiguity in the level of abstraction (e.g., "open the door" may output "door" instead of "handle"); (b) contextual object hallucination (e.g., correctly identifying "handle" but pairing it with "cabinet" instead of "door"). The action sequence provides a clear stopping criterion, and the hierarchical relationship prevents contextual mismatch.

2. **Score-based View Selection**:

    - **Function**: Filter out a small subset of views (~50) with the best visibility of the context object from thousands of view frames, boosting both accuracy and efficiency of functional object segmentation.
    - **Mechanism**: First, OWLv2 + RobustSAM segment the context object $O$ in all views. For each mask, three scores are computed: mask confidence $S_m$, distance distribution uniformity $S_d$ (higher when the mask is closer to the image center), and angular distribution uniformity $S_\alpha$ (higher when mask pixels are evenly distributed around the center). The distance and angular distributions are evaluated using KL divergence against a reference uniform distribution: $S_d = 1 - D_{KL}(P_d || U_d)$. The final selection score is $S_O = \lambda_m S_m + \lambda_d S_d + \lambda_\alpha S_\alpha$, selecting the top-50 views.
    - **Design Motivation**: Most of the thousands of views do not contain the target object or are of poor view quality; processing all of them is computationally inefficient and introduces noise. Polar-distribution-based scoring identifies the best views where the object is centered and fully visible.

3. **VLM-guided Functional Object Segmentation**:

    - **Function**: Accurately localize and segment functional objects within the selected views.
    - **Mechanism**: Query the Molmo VLM with "Point to all the F in order to D" (e.g., "Point to all the handles in order to open the bottom drawer"). The VLM returns coordinate points in the image, which are then used as prompts for SAM to generate precise segmentations. Incorporating both $F$ and the complete task description $D$ in the query enables the VLM to resolve ambiguities between semantically identical but task-irrelevant objects.
    - **Design Motivation**: Rather than providing the functional object's name alone, including the complete task description allows the VLM to perform contextual disambiguation—for instance, a prompt for "handle" might segment handles on neighboring furniture, whereas including "cabinet under the TV" isolates the target search region.

### Loss & Training
Fun3DU is an entirely training-free approach. All utilized models (Llama3.1, OWLv2, RobustSAM, Molmo, SAM) are frozen, pre-trained models that require no fine-tuning. Multi-view voting aggregation: The score for each 3D point is calculated as $s_i = \sum_{k=1}^K |{p^k \text{ s.t. } \Gamma^k(p^k)=c_i}|$, counting the number of view pixels that map to the point and belong to the functional object mask. After normalization, a threshold of $\tau=0.7$ is applied to generate the final 3D mask.

## Key Experimental Results

### Main Results

**SceneFun3D split0 (30 scenes)**:

| Method | mAP | AP50 | AP25 | mAR | AR50 | AR25 | mIoU |
|------|-----|------|------|-----|------|------|------|
| **Fun3DU** | **7.6** | **16.9** | **33.3** | 27.4 | **38.2** | 46.7 | **15.2** |
| OpenMask3D | 0.2 | 0.2 | 0.4 | 20.3 | 24.5 | 27.0 | 0.2 |
| OpenIns3D | 0.0 | 0.0 | 0.0 | 40.5 | 46.7 | **51.5** | 0.1 |
| LERF | 0.0 | 0.0 | 0.0 | 34.2 | 35.1 | 36.0 | 0.0 |

**SceneFun3D split1 (200 scenes)**:

| Method | mAP | AP50 | AP25 | mIoU |
|------|-----|------|------|------|
| **Fun3DU** | **6.1** | **12.6** | **23.1** | **11.5** |
| OpenMask3D | 0.0 | 0.0 | 0.0 | 0.1 |
| OpenIns3D | 0.0 | 0.0 | 0.0 | 0.1 |
| LERF | 0.0 | 0.0 | 0.0 | 0.0 |

Fun3DU outperforms the closest competitor OpenMask3D on AP25 by 32.9 points (split0) and 23.1 points (split1).

### Ablation Study

| Configuration | Description |
|------|------|
| All baseline methods | AP is near zero but AR is relatively high $\rightarrow$ severe under-segmentation, showing a tendency to segment the whole furniture rather than functional small components |
| Fun3DU split0 vs split1 | Performance drops on split1 $\rightarrow$ scenes are more complex (point clouds contain up to 13M points vs 8M points) |
| OpenMask3D split1 drops significantly | OpenMask3D relies heavily on 3D encoders, making it more sensitive to scene complexity |
| Fun3DU/LERF/OpenIns3D relatively stable | Segmenting directly on 2D views is more robust to scene complexity |

### Key Findings
- **All open-vocabulary 3D segmentation baselines fail almost completely**—AP near zero indicates they cannot segment small functional elements, only large furniture. This confirms that functional segmentation requires specialized design.
- OpenIns3D/LERF achieve relatively high AR but zero AP—they can "recall" the vicinity of interest but lack precise segmentation boundaries (segmenting the whole cabinet rather than the handle).
- The point-pointing capability of the VLM (Molmo) is key—it successfully bridges task semantics and fine-grained object localization.
- Chain-of-Thought reasoning successfully avoids two typical failure modes of LLMs: abstract-level mismatch and context hallucination.

## Highlights & Insights
- **Importance of Task Definition**: Functional understanding is an overlooked but crucial task—for embodied AI, localizing a "switch" is more operationally meaningful than localizing a "cabinet". The design of SceneFun3D benchmarks (not explicitly naming target objects) increases realism and challenge.
- **Bypassing 3D Limits with 2D Capabilities**: The bias of 3D data and models against small objects is systemic. The strategy of segmenting in 2D and then projecting to 3D elegantly bypasses this bottleneck.
- **Polar-based Scoring for View Selection**: Evaluating view quality through the uniformity of distance and angular distributions—objects centered and uniformly distributed imply high visibility. This scoring method is novel and intuitive.

## Limitations & Future Work
- The maximum mIoU is only 15.2%—although significantly higher than baselines, the absolute performance remains low, and functional segmentation has substantial room for improvement.
- The cascade pipeline relies entirely on pre-trained models; errors at each step accumulate (LLM reasoning error $\rightarrow$ view selection error $\rightarrow$ segmentation failure).
- Processing VLM queries for 50 views remains computationally demanding.
- Evaluations are limited to indoor scenes; applicability to outdoor/industrial scenarios is unverified.
- Handling spatial descriptions (e.g., "top drawer" vs. "bottom drawer") depends on the VLM's spatial reasoning capability, which may fail in complex layouts.

## Related Work & Insights
- **vs OpenMask3D/OpenIns3D**: Open-vocabulary 3D segmentation methods rely on 3D proposal modules, are biased towards large objects, and lack reasoning capabilities. Fun3DU resolves these two fundamental limitations through LLM reasoning + VLM fine-grained localization in 2D.
- **vs LERF**: LERF, based on NeRF's language fields, requires scene-by-scene training and has weak localization capability for tiny objects. Fun3DU is zero-training and achieves precise localization at the 2D level.
- **vs 2D VLMs (LLaVA/Molmo)**: Molmo's point-pointing capability makes it exceptionally suited for functional object routing, showing superiority over text-only VLMs like LLaVA.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ For defining and tackling 3D functional understanding for the first time with a highly targeted method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on dedicated benchmarks, though more ablation analyses could be included.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly stated, and the four-module pipeline flows logically.
- Value: ⭐⭐⭐⭐⭐ Direct significance for embodied AI and human-robot interaction, filling the gap in 3D functional understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](../../ICCV2025/3d_vision/articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[CVPR 2025\] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](spectromotion_dynamic_3d_reconstruction_of_specular_scenes.md)
- [\[CVPR 2025\] Wonderland: Navigating 3D Scenes from a Single Image](wonderland_navigating_3d_scenes_from_a_single_image.md)

</div>

<!-- RELATED:END -->
