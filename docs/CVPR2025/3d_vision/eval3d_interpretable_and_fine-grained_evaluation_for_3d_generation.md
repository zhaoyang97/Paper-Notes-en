---
title: >-
  [Paper Note] Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation
description: >-
  [CVPR 2025][3D Vision][3D Generation Evaluation] This paper proposes Eval3D, a fine-grained and interpretable evaluation tool for 3D generation quality. The core idea is to utilize various foundation models and tools as probes to detect inconsistencies in the semantic, geometric, structural, and text-alignment aspects of generated 3D assets. This achieves pixel-precise measurements and 3D spatial feedback, aligning more closely with human judgment than existing metrics.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Generation Evaluation"
  - "Interpretable Metrics"
  - "Semantic Consistency"
  - "Geometric Consistency"
  - "Foundation Model Probes"
date: 2026-05-08
content_hash: fda7ce2a08d2904c
---

# Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2504.18509](https://arxiv.org/abs/2504.18509)  
**Code**: [https://eval3d.github.io/](https://eval3d.github.io/)  
**Area**: 3D Vision  
**Keywords**: 3D Generation Evaluation, Interpretable Metrics, Semantic Consistency, Geometric Consistency, Foundation Model Probes

## TL;DR

This paper proposes Eval3D, a fine-grained and interpretable evaluation tool for 3D generation quality. The core idea is to utilize various foundation models and tools as probes to detect inconsistencies in the semantic, geometric, structural, and text-alignment aspects of generated 3D assets. This achieves pixel-precise measurements and 3D spatial feedback, aligning more closely with human judgment than existing metrics.

## Background & Motivation

**Background**: The 3D generation field is moving rapidly (e.g., diffusion-model-based text-to-3D methods), but generated 3D assets often suffer from multi-view inconsistency issues, including the Janus problem, texture-geometry misalignment, and semantic drift. Evaluating these issues requires reliable quantitative tools.

**Limitations of Prior Work**: Existing 3D evaluation metrics have significant drawbacks—distribution-level metrics like FID/KID cannot capture instance-level geometric quality; CLIP Score only provides coarse-grained semantic alignment evaluation; evaluations based on multimodal large language models (such as GPT-4V) yield coarse, uninterpretable outputs, and lack pixel-level localization capabilities.

**Key Challenge**: The quality of 3D generation is multi-dimensional (semantic, geometry, structure, aesthetics, text alignment), and existing single metrics fail to comprehensively cover these dimensions, let alone provide fine-grained spatial feedback on "where things went wrong".

**Goal**: To design an interpretable evaluation toolkit covering multiple complementary dimensions, which can (1) provide pixel-level/point-level quality measurements; (2) localize specific defects in 3D space; and (3) align highly with human judgment.

**Key Insight**: The authors' key observation is that many desired properties of 3D generation (such as semantic consistency and geometric consistency) can be effectively captured by measuring the consistency across various foundation models and tools. For instance, if a 3D asset is semantically consistent, images rendered from different viewpoints should produce consistent representations in the DINO feature space.

**Core Idea**: Utilizing various foundation models (DINO, Zero-123, normal estimators, etc.) as probes to evaluate 3D asset quality across five dimensions: semantic consistency, structural consistency, geometric consistency, text-3D alignment, and aesthetics. The generation quality is quantified by measuring the inconsistencies in the probe outputs.

## Method

### Overall Architecture

Eval3D takes a generated 3D asset as input, renders RGB images and normal maps from multiple predefined viewpoints, and then computes quality scores for each dimension using five independent evaluation modules. The output consists of global scores for each dimension and pixel-level/point-level quality heatmaps, allowing precise localization of defects in 3D space.

### Key Designs

1. **Semantic Consistency**:

    - **Function**: Measures whether the generated 3D asset maintains semantic consistency in multi-view scenarios (e.g., detecting Janus problems).
    - **Mechanism**: Projects the surface points of the 3D asset onto DINO feature maps of multiple viewpoints using differentiable rendering, collects DINO feature vectors for each 3D point across all visible viewpoints, and computes the standard deviation of these features. High standard deviation indicates that the 3D point presents different semantic appearances from different viewpoints (e.g., a dog's face seen from both the front and back views), indicating semantic inconsistency.
    - **Design Motivation**: DINO features possess a degree of viewpoint invariance but are sensitive to changes in semantic content, making them highly suitable for detecting Janus problems. Aggregating information in 3D space allows for the precise localization of problematic regions (e.g., extra noses/faces).

2. **Geometric Consistency**:

    - **Function**: Evaluates whether texture and underlying geometry are aligned.
    - **Mechanism**: Compares two types of normal information—(1) geometry normals rendered directly from the 3D mesh, and (2) predicted normals from rendered RGB images using a pre-trained image normal estimation model (like Omnidata). A larger discrepancy between the two indicates a worse match between texture and geometry. Discrepancies are calculated via pixel-level angular errors, with bright yellow areas representing large deviations.
    - **Design Motivation**: 3D generation methods (especially those based on NeRF/3DGS) often suffer from issues where "the texture looks correct but the geometry is wrong". Comparing two independent normal sources can effectively detect such errors.

3. **Structural Consistency**:

    - **Function**: Evaluates the global geometric coherence of the generated 3D asset.
    - **Mechanism**: Renders views of the generated asset from multiple rotation angles and compares them with novel-view synthesis predictions from Zero-123. DreamSim is used to measure image similarity between both. If the geometry of the generated asset is coherent, the novel view synthesized from a known viewpoint and camera path should be consistent with the actual render.
    - **Design Motivation**: Zero-123 has learned 3D priors of natural objects and can predict reasonable novel views. Comparing its predictions with actual renders measures whether the generated asset complies with the 3D structural rules of natural objects.

### Loss & Training

Eval3D is an evaluation tool rather than a training method, and thus does not involve a loss function. Its metrics are computed based on pre-trained foundation models without additional training. Additionally, it includes text-3D alignment (computing alignment scores between text and multi-view images using CLIP) and aesthetic scoring (evaluating the visual appeal of rendered images using a LAION aesthetic scorer).

## Key Experimental Results

### Main Results — Alignment with Human Judgment

| Evaluation Metric | Kendall's $\tau$ Correlation with Human Ranking |
|----------|-------------------------------|
| FID | 0.14 |
| CLIP Score | 0.31 |
| GPT-4V Evaluation | 0.38 |
| **Eval3D (Combined)** | **0.52** |

### Multi-Model Benchmarking

| 3D Generation Model | Semantic Consistency ↓ | Geometric Consistency ↓ | Structural Consistency ↑ | Text Alignment ↑ | Aesthetics ↑ |
|------------|-------------|-------------|-------------|-----------|--------|
| DreamFusion | High Inconsistency | Medium | Low | Medium | Low |
| Magic3D | Medium | Medium | Medium | Medium | Medium |
| Instant3D | Low Inconsistency | Low Deviation | Relatively High | Relatively High | Relatively High |
| LGM | Medium | Relatively High | Medium | Relatively High | Medium |

### Key Findings

- The combined score of Eval3D aligns significantly better with human judgment than FID, CLIP Score, and GPT-4V evaluations.
- Specific locations of Janus problems, geometric errors, and texture artifacts can be accurately localized via 3D inconsistency heatmaps.
- Real physical objects (Objaverse GT) score extremely high on semantic consistency, validating the effectiveness of the metric.
- Performance across dimensions varies significantly among different generative models—some are semantically consistent but geometrically coarse, while others exhibit the opposite.

## Highlights & Insights

- **The evaluation philosophy of "foundation models as probes" is very clever**: Instead of training a new evaluation model, it leverages the consistency of existing foundation models as quality signals. This approach has high scalability—as foundation model capabilities improve, the evaluation accuracy will naturally advance as well.
- **3D spatial feedback** is a major highlight: It not only provides a global score of "how good it is" but also highlights "where it goes wrong" on the 3D mesh, which is extremely valuable for algorithm developers to locate and fix issues.
- Decoupling the evaluation into five independent dimensions makes the evaluation results interpretable, enabling developers to targetedly improve specific aspects.

## Limitations & Future Work

- How to combine the weights of each dimension into a final score lacks theoretical guidance and currently relies on empirical settings.
- Sensitive to the biases of the foundation models themselves—if DINO or Zero-123 performs poorly on certain object categories, the evaluation results may be inaccurate.
- The semantic consistency metric assumes that the object should remain consistent across all views, but might misjudge intentionally symmetric or stylized artistic 3D assets.
- Future work can extend this to evaluate the temporal consistency of animated 3D assets, or assess 3D scenes instead of single objects.

## Related Work & Insights

- **vs CLIP Score**: CLIP Score only evaluates text-image alignment at the image level, whereas Eval3D provides multi-dimensional, pixel-level evaluation including geometry.
- **vs GPT-4V Evaluation**: GPT-4V provides coarse-grained evaluation described in natural language, while Eval3D outputs quantifiable numerical metrics and spatial heatmaps.
- **vs FID/KID**: Traditional distribution-level metrics cannot capture quality differences on an instance-by-instance basis, whereas Eval3D performs instance-level evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of "foundation models as probes" is novel, though it is based entirely on existing tool combinations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model benchmarking and human alignment experiments are thorough, although ablation studies are limited.
- Writing Quality: ⭐⭐⭐⭐ Great visualization effects and clearly presented concepts.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed standardized evaluation tool for the 3D generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[CVPR 2025\] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects](gen3deval_using_vllms_for_automatic_evaluation_of_generated_3d_objects.md)
- [\[CVPR 2026\] Animator-Centric Skeleton Generation on Objects with Fine-Grained Details](../../CVPR2026/3d_vision/animator-centric_skeleton_generation_on_objects_with_fine-grained_details.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](../../NeurIPS2025/3d_vision/micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[CVPR 2025\] MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing](micas_multi-grained_in-context_adaptive_sampling_for_3d_point_cloud_processing.md)

</div>

<!-- RELATED:END -->
