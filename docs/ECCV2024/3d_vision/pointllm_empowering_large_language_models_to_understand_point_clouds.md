---
title: >-
  [Paper Note] PointLLM: Empowering Large Language Models to Understand Point Clouds
description: >-
  [ECCV2024][3D Vision][Point Cloud Understanding] PointLLM connects a point cloud encoder (Point-BERT) to the LLaMA large language model via an MLP projection layer. Utilizing 730K instruction-following data (660K brief descriptions + 70K complex instructions) for two-stage training, it achieves a generative accuracy of 53.4% on 3D object classification (surpassing LLaVA-13B's 44.2%) and a human evaluation win rate of 55% over human annotations in object description tasks.
tags:
  - "ECCV2024"
  - "3D Vision"
  - "Point Cloud Understanding"
  - "Multimodal LLM"
  - "3D Object Description"
  - "Instruction Tuning"
  - "LLaMA"
date: 2026-05-08
content_hash: 3ea2aa8527c1573c
---

# PointLLM: Empowering Large Language Models to Understand Point Clouds

**Conference**: ECCV2024  
**arXiv**: [2308.16911](https://arxiv.org/abs/2308.16911)  
**Code**: [https://github.com/OpenRobotLab/PointLLM](https://github.com/OpenRobotLab/PointLLM)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Understanding, Multimodal LLM, 3D Object Description, Instruction Tuning, LLaMA

## TL;DR
PointLLM connects a point cloud encoder (Point-BERT) to the LLaMA large language model via an MLP projection layer. Utilizing 730K instruction-following data (660K brief descriptions + 70K complex instructions) for two-stage training, it achieves a generative accuracy of 53.4% on 3D object classification (surpassing LLaVA-13B's 44.2%) and a human evaluation win rate of 55% over human annotations in object description tasks.

## Background & Motivation

**Background**: Multimodal LLMs (such as LLaVA, InstructBLIP) have demonstrated powerful capabilities in vision-language tasks, but they primarily rely on 2D image inputs. 3D understanding is still dominated by specialized models (PointNet++, Point-BERT) and lacks deep integration with LLMs.

**Limitations of Prior Work**: 2D images suffer from perspective dependence and depth ambiguity. Although some 3D-LLMs attempt 3D inputs, they rely on multi-view image feature extraction (inherently still 2D) and achieve limited performance. Point clouds, as a native 3D representation, are more direct, but systematic work connecting them with LLMs is lacking.

**Key Challenge**: LLMs understand discrete token sequences, whereas point clouds are continuous 3D spatial data. An effective "translation layer" is required to map 3D geometric and color information into the token space that LLMs can comprehend.

**Goal**: This paper aims to address: (1) How to effectively embed point cloud features into LLMs? (2) How to collect large-scale 3D instruction-tuning data? (3) Whether point cloud LLMs can outperform 2D multimodal LLMs on 3D understanding tasks?

**Key Insight**: Leveraging the architecture design of LLaVA (encoder + projection layer + LLM), but replacing the image encoder with a point cloud encoder, and constructing point-cloud-centric instruction data.

**Core Idea**: Point-BERT encoder extracts point cloud features $\rightarrow$ MLP projects features into LLM token space $\rightarrow$ combined with text tokens, fed into LLaMA $\rightarrow$ two-stage training (feature alignment then instruction tuning).

## Method

### Overall Architecture
Input 8192 3D points with RGB $\rightarrow$ Point-BERT encodes them into 513 feature tokens of dimension 384 $\rightarrow$ 3-layer MLP projects them to the LLM token dimension (5120) $\rightarrow$ wrapped with special tokens `<p_start>` and `<p_end>` $\rightarrow$ concatenated with text instruction tokens and fed into LLaMA-7B/13B (Vicuna) $\rightarrow$ autoregressive generation of answers.

### Key Designs

1. **Point Cloud Encoder (Point-BERT)**:

    - **Function**: Encodes 8192×6 (XYZ+RGB) point clouds into 513 feature tokens.
    - **Mechanism**: Pre-trained Point-BERT provides rich 3D geometric and semantic features; its weights are frozen and not updated during both training stages.
    - **Design Motivation**: The pre-trained point cloud encoder already learns excellent 3D representations, avoiding the need to retrain it for LLM alignment.

2. **3-layer MLP Projector**:

    - **Function**: Maps 384-dimensional point cloud features to the 5120-dimensional LLM token space (hidden dimensions 1024 $\rightarrow$ 2048 $\rightarrow$ 4096).
    - **Mechanism**: Multi-layer MLP provides stronger cross-modal feature transformation capability compared to a single linear layer.
    - **Ablation Validation**: 3-layer MLP (53.4%) > 2-layer (52.8%) > 1-layer (51.1%) > direct mapping (50.6%).

3. **Two-Stage Training**:

    - Stage 1 (Feature Alignment): Freeze the encoder and LLM, train only the projector using 660K brief description instructions.
    - Stage 2 (Instruction Tuning): Freeze the encoder, jointly train the projector and LLM (using 15K detailed descriptions + 40K single-turn QA + 15K multi-turn conversations).
    - **Design Motivation**: Align modalities first, then fine-tune understanding capabilities, consistent with the LLaVA training paradigm.

### Loss & Training
Standard autoregressive cross-entropy loss, computed only on text output tokens. Data sources: Cap3D (Objaverse 3D object descriptions) + QA pairs generated by GPT-4. 30 brief description prompt templates + 30 complex prompt templates are used to increase diversity.

## Key Experimental Results

### Main Results

**Generative 3D Object Classification (ModelNet40 + Objaverse)**:

| Model | Input | Avg. Accuracy |
|------|------|-----------|
| InstructBLIP-13B | Single-view Image | 31.5% |
| LLaVA-13B | Single-view Image | 44.2% |
| 3D-LLM | 3D+Multi-view | 45.3% |
| Point-Bind LLM | Point Cloud | 25.5% |
| **PointLLM-7B** | **Point Cloud** | **52.8%** |
| **PointLLM-13B** | **Point Cloud** | **53.4%** |

**3D Object Captioning (Objaverse 200 Objects)**:

| Model | Correctness↑ | Hallucination↓ | Precision | GPT-4 Score↑ |
|------|--------|-------|------|---------|
| LLaVA-13B | 2.43 | 0.86 | 74.0% | 38.3 |
| 3D-LLM | 1.77 | 1.16 | 60.4% | 33.4 |
| **PointLLM-13B** | **3.10** | 0.84 | **78.8%** | **48.2** |
| Human Annotation | 2.67 | 0.22 | 92.5% | 100.0 |

Human evaluation win rate: PointLLM vs. Human Annotation = **55% win rate**.

### Ablation Study

| Configuration | Accuracy |
|------|-------|
| Full Model (3-layer MLP + All Data) | 52.82% |
| Single-turn QA Data Only | 40.14% |
| + Multi-turn Conversation | 45.79% |
| + Detailed Description (Full) | **52.82%** |
| Max pooling token reduction | 48.72% (-4.1%, 75% faster training) |
| Alignment data 200K $\rightarrow$ 400K $\rightarrow$ 600K | 49% $\rightarrow$ 50.5% $\rightarrow$ plateau |

### Key Findings
- **Direct point cloud input outperforms 2D images**: PointLLM-13B (53.4%) significantly outperforms LLaVA-13B (44.2%) which uses single/multi-view images, indicating that native 3D inputs indeed provide richer geometric information.
- **Correctness surpasses human annotations**: PointLLM generates more detailed and accurate descriptions than human annotations (correctness 3.10 vs 2.67), though it also exhibits higher hallucination rates (0.84 vs 0.22).
- **Data diversity is key**: The three instruction types (single-turn, multi-turn, detailed descriptions) are all indispensable; using single-turn QA alone drops accuracy to 40.1%.
- **Accuracy-speed trade-off of Max pooling**: Compressing 75% of tokens saves 75% of training time but sacrifices 4% in accuracy.

## Highlights & Insights
- **First systematic point cloud-LLM framework**: Completely defines the problem (data construction + model architecture + evaluation metrics), providing a solid baseline and dataset for future 3D-LLM research. The architecture is elegant and simple (Point-BERT + MLP + LLaMA), making it easy to replicate and extend.
- **GPT-4 assisted data construction pipeline**: Uses GPT-4 to generate diverse instruction data (QA, multi-turn dialogue, detailed descriptions) from Cap3D captions, addressing the scarcity of 3D instruction data. This data pipeline can generalize to any 3D asset dataset.
- **Deep and thoughtful evaluation design**: Points out that traditional NLP metrics (BLEU/ROUGE) are unsuitable for open-ended 3D descriptions (e.g., "private jet" and "airplane" score 0), introducing GPT-4 and human evaluations instead.

## Limitations & Future Work
- **Confined to object-level point clouds**: Trained and tested on Objaverse single objects, failing to comprehend scene-level point clouds (poor performance when tested on ScanNet).
- **Dependence on RGB point clouds**: Requires colored point cloud inputs. For geometry-only datasets (e.g., ModelNet40), colors must be artificially assigned (e.g., set to black), limiting practical application scenarios.
- **Hallucination issues**: The 13B model exhibits a higher hallucination rate (0.84) than the 7B model (0.66), suggesting that larger LLMs are harder to fine-tune precisely.
- **High training cost**: Requires 126 GPU-hours for the 7B model and 213 GPU-hours for the 13B model on A100.
- **Potential improvements**: (1) Extend to scene-level data (ScanNet/S3DIS); (2) Introduce LoRA to reduce fine-tuning costs; (3) Incorporate 3D detection/segmentation heads to support a wider range of 3D tasks.

## Related Work & Insights
- **vs LLaVA**: Highly symmetric architecture, but replaces the image encoder with a point cloud encoder. The key finding is that native 3D inputs are better suited for 3D understanding tasks than 2D images.
- **vs 3D-LLM**: 3D-LLM relies on multi-view images to extract 3D features (which is fundamentally still 2D), whereas PointLLM directly utilizes point clouds, yielding superior performance in both classification and description tasks.
- **vs Point-Bind LLM**: Point-Bind uses contrastive learning to align point clouds and language, but has weak generation capabilities (achieving only 4.5% on Objaverse classification). PointLLM's two-stage fine-tuning approach is much more effective.

## Rating
- **Novelty**: ⭐⭐⭐ Architecture is similar to LLaVA (encoder + projection + LLM); the innovation mainly lies in problem definition and data construction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across classification and captioning tasks with various evaluation metrics; includes comprehensive ablations on projection layers, data size, and pooling.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly described methodology and in-depth discussion on evaluation metrics.
- **Value**: ⭐⭐⭐⭐ Pioneering integration of point clouds and LLMs; open-sourced dataset and code establish a strong foundation for future 3D multimodal LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Empowering Large Language Models with 3D Situation Awareness](../../CVPR2025/3d_vision/empowering_large_language_models_with_3d_situation_awareness.md)
- [\[ICLR 2026\] Do 3D Large Language Models Really Understand 3D Spatial Relationships?](../../ICLR2026/3d_vision/do_3d_large_language_models_really_understand_3d_spatial_relationships.md)
- [\[ECCV 2024\] SegPoint: Segment Any Point Cloud via Large Language Model](segpoint_segment_any_point_cloud_via_large_language_model.md)
- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)
- [\[ECCV 2024\] Progressive Classifier and Feature Extractor Adaptation for Unsupervised Domain Adaptation on Point Clouds](progressive_classifier_and_feature_extractor_adaptation_for_unsupervised_domain_.md)

</div>

<!-- RELATED:END -->
