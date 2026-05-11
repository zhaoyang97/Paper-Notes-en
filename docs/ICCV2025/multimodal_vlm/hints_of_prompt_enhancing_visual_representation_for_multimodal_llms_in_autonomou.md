---
title: >-
  [Paper Note] Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving
description: >-
  [Multimodal VLM] This paper proposes the Hints of Prompt (HoP) framework, which enhances CLIP visual representations through three hierarchical hints (Affinity/Semantic/Question hint) to capture instance-level structure…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: d4c918e532ed9556
---

# Hints of Prompt: Enhancing Visual Representation for Multimodal LLMs in Autonomous Driving

## Basic Information
- **Conference**: ICCV 2025
- **arXiv**: 2411.13076
- **Code**: Not released
- **Area**: Multimodal VLM / Autonomous Driving
- **Keywords**: Multimodal large language models, autonomous driving VQA, visual representation enhancement, prompt fusion, domain adaptation

## TL;DR

This paper proposes the Hints of Prompt (HoP) framework, which enhances CLIP visual representations through three hierarchical hints (Affinity/Semantic/Question hint) to capture instance-level structure, domain-specific semantics, and question relevance. HoP surpasses the fully trained baseline on autonomous driving VQA tasks using only 25% of the training data.

## Background & Motivation

- **Problem Definition**: Visual question answering (VQA) in autonomous driving scenarios requires fine-grained hierarchical visual features to handle complex interactions and long-tail cases. However, general-purpose MLLMs (e.g., LLaVA) combined with CLIP visual encoders struggle to accurately represent driving-specific scenes.
- **Limitations of Prior Work**:
    - CLIP visual tokens lack inter-token affinity relationships (as shown in Fig. 3), losing instance-level structural information.
    - General-purpose visual encoders underrepresent domain-specific semantics (e.g., distant vehicles, pedestrians, traffic signs).
    - Visual and text tokens are processed separately, making it difficult for the model to focus on image regions relevant to a specific question.
    - Existing multi-encoder fusion methods (e.g., A-MoF) employ complex fusion strategies and lack rapid adaptation capability for the driving domain.
- **Key Observations**: DINOv2 tokens maintain strong intra-instance affinity and can compensate for CLIP's deficiency; sparse queries from DETR-style models encode driving-relevant semantics; text embeddings from the LLM can guide visual attention toward question-relevant regions.

## Method

### Overall Architecture

HoP builds upon LLaVA-v1.5 and introduces three types of hint tokens to augment CLIP visual representations. These hints are fused via a Hint Fusion module and passed through an adapter into the LLM.

### Three Types of Hint Tokens

**1. Affinity Hint**
- Source: 576 feature tokens extracted by DINOv2-large.
- Function: Provides instance-level structural information and enhances inter-token affinity relationships.
- Validation: Using only the inter-token similarity matrix from DINOv2 as the hint already yields significant improvement (Tab. 4), confirming that the gain stems from token-wise affinity rather than DINOv2-specific features.

**2. Semantic Hint**
- Source: Top-K sparse queries from Mask2Former or GroundingDINO (default: 16 tokens).
- Function: Introduces high-level, driving-specific semantic information (vehicles, traffic signs, pedestrians, etc.).
- Each token is augmented with the embedding of its corresponding category label.
- Mask2Former outperforms GroundingDINO (Tab. 5), as it is trained on Cityscapes and better suited to driving scenes.

**3. Question Hint**
- Source: The text embedding layer of the LLM (superior to the CLIP text encoder, Tab. 6).
- Function: Aligns visual features with the question context, focusing attention on question-relevant image regions.

### Hint Fusion Module

Five fusion strategies were explored; the final design adopts **Joint Cross-Attention**:

$$\mathbf{P}_f = \mathbf{P} + \text{CA}(\mathbf{P}, [\mathbf{P}, \mathbf{H}_A, \mathbf{H}_S, \mathbf{H}_Q])$$

- CLIP visual tokens $\mathbf{P}$ serve as queries, attending to the concatenation of themselves and all hint tokens as keys and values.
- Only 8.7M parameters and 3.8 GFLOPs, achieving the best efficiency.
- Outperforms alternative strategies including Concatenation (which degrades performance), and Sequential/Parallel Cross-Attention variants.

### Efficient HoP

- Two lightweight heads are distilled from the CLIP backbone to replace DINOv2 and Mask2Former.
- Affinity head: A 4-layer ViT-like decoder trained via cosine similarity distillation from DINOv2 features.
- Semantic head: A ViTDet-like neck combined with a Mask2Former head, trained on Cityscapes.
- Inference latency is only 661 ms (vs. 956 ms for HoP), reduced to 281 ms after quantization.

## Key Experimental Results

### Main Results on LingoQA

| Method | LLM | Lingo-Judge↑ | BLEU-4↑ | METEOR↑ | CIDEr↑ |
|--------|-----|-------------|---------|---------|--------|
| GPT-4V (zero-shot) | - | 59.6 | 6.30 | 12.4 | 42.8 |
| LLaVA-v1.5 | Vicuna-7B | 63.2 | 14.1 | 19.3 | 63.7 |
| LLaVA-v1.5 (+A-MoF) | Vicuna-7B | 64.2 | 14.5 | 19.1 | 64.7 |
| VTS | InternLM2-7B | 64.2 | 14.5 | 20.5 | 56.9 |
| Efficient HoP | Vicuna-7B | 66.8 | 15.2 | 20.0 | 66.2 |
| **HoP** | **Vicuna-7B** | **67.8** | **15.8** | **20.3** | **70.9** |

HoP achieves state-of-the-art performance across all metrics.

### Ablation Study (Hint Combination Effects)

| Configuration | Lingo-Judge↑ |
|---------------|-------------|
| Baseline (no hint) | 63.2 |
| +Affinity hint only | ~64.6 |
| +Semantic hint only | ~64.0 |
| +Question hint only | ~63.8 |
| +AH+SH | ~66.0 |
| +AH+SH+QH (ALL) | 67.4 |
| +AH+SH(+cls)+QH | **67.8** |

Each hint individually improves performance; combining all three yields additive gains, with category label embeddings providing further improvement.

### Fusion Strategy Comparison

| Fusion Strategy | Lingo-Judge↑ | Params (M) | GFLOPs |
|----------------|-------------|------------|--------|
| Concatenation | 58.8 | - | - |
| Self-Cross-Attention | 65.2 | 12.9 | 5.0 |
| Parallel Cross-Attention | 66.6 | 17.1 | 8.63 |
| Sequential Cross-Attention | 67.0 | 17.1 | 8.63 |
| **Joint Cross-Attention** | **67.8** | **8.7** | **3.8** |

Joint Cross-Attention achieves the best performance with the fewest parameters and lowest computational cost.

### Data Efficiency

HoP trained on only 25% of the data achieves a Lingo-Judge score of 64.0, surpassing the LLaVA-v1.5 baseline trained on the full dataset (63.2), demonstrating strong domain adaptation efficiency.

## Highlights & Insights

1. **Elegant three-level hint design**: The framework progressively enhances visual representations from instance-level (Affinity) to semantic-level (Semantic) to query-level (Question).
2. **Outstanding data efficiency**: Surpassing the full-data baseline with only 25% of training data demonstrates that the hints effectively bridge the domain gap.
3. **Key deficiency identified in CLIP tokens**: CLIP loses inter-token affinity relationships—an observation with important implications for understanding CLIP's limitations in downstream tasks.
4. **Concatenation is harmful**: Directly concatenating multi-level information confuses the adapter and LLM, underscoring the importance of fusion strategy design.
5. **Practical Efficient HoP**: With a latency of 281 ms after quantization, the efficient variant is suitable for real-world deployment.

## Limitations & Future Work

- The full HoP variant requires running DINOv2 and Mask2Former separately, increasing inference latency by approximately 50%.
- Evaluation is limited to autonomous driving VQA; generalization to other domains (e.g., medical imaging, remote sensing) remains unexplored.
- The Semantic hint depends on the quality of the pretrained detection/segmentation models.
- Experiments are conducted solely on 7B-scale LLMs; performance on larger models is unknown.

## Related Work & Insights

- **QA-ViT**: Similarly incorporates textual information into visual features, but the fusion strategy in this work is more flexible and general.
- **A-MoF**: A multi-encoder fusion approach with complex strategies that are not specifically optimized for the driving domain.
- **Hint-AD / TOKEN**: Other visual enhancement frameworks for autonomous driving, but HoP's dynamic hint mechanism and lightweight fusion offer greater scalability.
- **Insight**: The multi-source hint augmentation paradigm proposed here can be extended to other domain-specific MLLM applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The three-hint design and Joint Cross-Attention fusion are original contributions)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive validation across three datasets with thorough ablations)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear logic with rich figures and tables)
- **Value**: ⭐⭐⭐⭐ (Practical applicability to driving VQA; Efficient HoP is deployment-ready)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)

</div>

<!-- RELATED:END -->
