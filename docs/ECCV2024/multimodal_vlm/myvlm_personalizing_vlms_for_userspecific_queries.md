---
title: >-
  [Paper Note] MyVLM: Personalizing VLMs for User-Specific Queries
description: >-
  [ECCV 2024][Multimodal VLM][VLM personalization] MyVLM is the first to explore the VLM personalization problem. It detects user-specific concepts (e.g., "your dog") using an external concept recognition head, and learns concept embeddings in the VLM's intermediate feature space to guide the language model to naturally incorporate the concept in its responses. It achieves personalized captioning and VQA with only 3-5 images.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "VLM personalization"
  - "concept embedding"
  - "BLIP-2"
  - "LLaVA"
  - "few-shot learning"
date: 2026-05-08
content_hash: 5d82121da019638d
---

# MyVLM: Personalizing VLMs for User-Specific Queries

**Conference**: ECCV 2024  
**arXiv**: [2403.14599](https://arxiv.org/abs/2403.14599)  
**Code**: [https://snap-research.github.io/MyVLM/](https://snap-research.github.io/MyVLM/)  
**Area**: Multimodal VLM  
**Keywords**: VLM personalization, concept embedding, BLIP-2, LLaVA, few-shot learning

## TL;DR

MyVLM is the first to explore the VLM personalization problem. It detects user-specific concepts (e.g., "your dog") using an external concept recognition head, and learns concept embeddings in the VLM's intermediate feature space to guide the language model to naturally incorporate the concept in its responses. It achieves personalized captioning and VQA with only 3-5 images.

## Background & Motivation

1. **Background**: VLMs (such as BLIP-2 and LLaVA) possess powerful visual understanding and text generation capabilities, enabling them to describe image content. However, they only contain general knowledge and cannot recognize a specific user's pets, friends, or personal belongings.

2. **Limitations of Prior Work**: (1) Fine-tuning a VLM for each user is computationally expensive and may lead to catastrophic forgetting; (2) model editing techniques can only modify responses to specific queries and fail to generalize to new images; (3) the feature space of the VLM's visual encoder is not sufficiently expressive, making it difficult to effectively distinguish semantically similar concepts (e.g., distinguishing "your dog" from "other dogs").

3. **Key Challenge**: Personalization requires the model to recognize new concepts and generalize them across different scenarios, without modifying the original weights of the VLM (which would otherwise damage its general capabilities). Furthermore, concept recognition and concept communication represent two distinct challenges.

4. **Goal**: (1) How to teach the VLM to recognize user-specific concepts without altering its original weights? (2) How to enable the language model to naturally integrate concept identifiers into the context when generating responses?

5. **Key Insight**: Decompose the problem into two distinct steps: "recognition" and "communication" — first, use an external head to detect whether the concept is present in the image, and then use a learnable embedding to guide the LLM to refer to the concept in its output.

6. **Core Idea**: Combining an external concept recognition head, a learnable concept embedding, and attention regularization to achieve recognition and integration of user-specific concepts without modifying the VLM weights.

## Method

### Overall Architecture

MyVLM operates in two stages: (1) Recognition — an external concept recognition head (a CLIP-based classifier or a face recognition network) detects whether the target concept is present in the image; (2) Communication — the learned concept embedding vector is appended to the visual encoder output, processed through the Q-Former/linear projection layer, and fed into the LLM to guide it in using the concept identifier in its responses. All original weights of the VLM are frozen throughout the process.

### Key Designs

1. **External Concept Recognition Head**:
    - **Function**: Determines whether a user-specific concept appears in the current image.
    - **Mechanism**: Employs a linear classifier over the CLIP embedding space for object recognition, and a pre-trained face recognition network for identity recognition. An independent head is used for each concept.
    - **Design Motivation**: The frozen visual encoder features of the VLM are insufficient to distinguish semantically similar objects (e.g., different dogs). Using external dedicated heads avoids modifying the visual encoder while offering flexible scalability to new concepts.

2. **Concept Embedding Learning**:
    - **Function**: Learns an embedding vector that enables the LLM to incorporate the concept identifier during generation.
    - **Mechanism**: The concept embedding $e_*$ is appended to the visual features and input into the Q-Former. Using 3-5 images containing the concept along with their captions (which contain the concept identifier S*), the embedding is optimized via the standard cross-entropy loss: $e_* = \arg\min_e \sum_{i=1}^{N} \mathcal{L}_{CE}(t_i, o(I_i, e))$.
    - **Design Motivation**: Learning the embedding in the intermediate feature space of the VLM leverages the existing vision-language bridge (Q-Former/linear layer) to communicate concept information to the LLM without modifying any original parameters.

3. **Generalization Improvement Mechanism**:
    - **Function**: Prevents the concept embedding from disrupting the original behavior of the VLM.
    - **Mechanism**: (1) Key/Value Normalization — Key/Value projections of the concept embedding are often significantly larger than those of the original image features, and thus need to be normalized to the average norm: $\hat{k}_* = \frac{k_*}{\|k_*\|} \cdot n_k$; (2) Attention Regularization — prevents the Q-Former query tokens from over-focusing on the concept embedding while ignoring other image features: $\mathcal{L}_{reg} = \|softmax(Q \cdot \hat{k}_*)\|_2^2$.
    - **Design Motivation**: Appending the embedding directly can cause attention to be dominated by the concept token, leading the query tokens to ignore original image information and generate unnatural captions.

### Loss & Training

Total Loss = Cross-Entropy Loss + Attention Regularization. Only the concept embedding vector (a d-dimensional vector) is optimized, which converges with just 3-5 images. The concept identifiers adopt the DreamBooth strategy: rare words for objects and short names for people.

## Key Experimental Results

### Main Results

| Method | Recall↑ | CLIPScore↑ | Sentence Sim↑ |
|------|---------|------------|---------------|
| PALAVRA (baseline) | 68.2 | 25.1 | 0.42 |
| Textual Inv. (baseline) | 72.5 | 25.8 | 0.44 |
| MyVLM (BLIP-2) | **89.3** | **27.2** | **0.51** |
| MyVLM (LLaVA) | **91.7** | **27.6** | **0.53** |

### Ablation Study

| Configuration | Recall | CLIPScore | Description |
|------|--------|-----------|------|
| w/o K/V Normalization | 82.1 | 25.9 | Unnatural captions |
| w/o Attention Regularization | 85.4 | 26.3 | Concept token dominates attention |
| w/o Concept Recognition Head | 71.8 | 27.0 | Outputs concepts even for non-concept images |
| Full MyVLM | **91.7** | **27.6** | Full model |

### Key Findings

- **The concept recognition head is indispensable**: Without the recognition head, the model erroneously generates concept identifiers even for images that do not contain the target concept.
- Attention visualization demonstrates that the concept embedding indeed "attends to" the regions where the concept is located, learning meaningful spatial associations.
- **MyVLM can achieve zero-shot transfer to VQA and REC tasks**: Embeddings trained on captioning can undergo direct zero-shot transfer to personalized visual question answering (VQA) and referring expression comprehension (REC), validating that the embeddings indeed capture the semantics of the concepts.

## Highlights & Insights

- **Pioneering Problem Definition**: Formally defines the task of VLM personalization for the first time, separating the concept recognition and communication sub-problems, and establishing a framework for future research.
- **K/V Normalization + Attention Regularization**: Resolves the common issue where excessively large embedding norms (similar to Textual Inversion) cause attention imbalance, which can be transferred to other scenarios requiring token insertion in the attention space.
- **Cross-Task Transferability**: Embeddings trained on captioning can perform zero-shot transfer to VQA and REC, demonstrating that the learned embeddings capture general concept representations rather than task-specific signals.

## Limitations & Future Work

- Each concept requires its own independent recognition head and embedding vector, which hinders efficiency when scaling to a large number of concepts.
- Identity recognition heavily relies on face recognition networks, which may fail in cases of profile views or occlusions.
- Currently, only single-concept queries are supported; the interaction when multiple concepts appear simultaneously remains unexplored.
- Experiments were only conducted on BLIP-2 and LLaVA; validation on more VLM architectures is required.

## Related Work & Insights

- **vs. Textual Inversion / DreamBooth**: These approaches learn concepts in image generation tasks, whereas MyVLM transfers the personalization paradigm to VLM understanding tasks.
- **vs. PALAVRA**: PALAVRA optimizes token embeddings in the CLIP text space for retrieval, while MyVLM operates in the intermediate feature space of the VLM, allowing it to generate contextualized descriptions.
- **vs. Model Editing**: Model editing modifies responses to specific queries, while MyVLM generalizes effectively to novel images and questions.
- VLM personalization can be combined with techniques like RAG to support larger-scale user knowledge bases.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formally defines the VLM personalization task for the first time with an elegant method design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on a custom dataset across multiple architectures and tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivated clearly with an easy-to-understand explanation of the methodology.
- Value: ⭐⭐⭐⭐⭐ Opens up a promising new research direction for VLM personalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Interactive Episodic Memory with User Feedback](../../CVPR2026/multimodal_vlm/interactive_episodic_memory_with_user_feedback.md)
- [\[ICML 2026\] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](../../ICML2026/multimodal_vlm/slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)
- [\[ACL 2025\] MIRe: Enhancing Multimodal Queries Representation via Fusion-Free Modality Interaction](../../ACL2025/multimodal_vlm/mire_enhancing_multimodal_queries_representation_via_fusion-free_modality_intera.md)
- [\[CVPR 2026\] DeepAlign: Mitigating Modality Conflict through Modality-Specific Alignment](../../CVPR2026/multimodal_vlm/deepalign_mitigating_modality_conflict_through_modality-specific_alignment.md)
- [\[CVPR 2026\] LVLM-Aided Alignment of Task-Specific Vision Models](../../CVPR2026/multimodal_vlm/lvlm-aided_alignment_of_task-specific_vision_models.md)

</div>

<!-- RELATED:END -->
