---
title: >-
  [Paper Note] RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Personalized Multimodal Large Language Models] The RAP (Retrieval-Augmented Personalization) framework is proposed to achieve personalization in MLLMs via a three-step "Remember-Retrieve-Generate" pipeline. It stores user concepts in an external database, dynamically retrieves relevant concept information using a multimodal retriever, and injects it into the MLLM to generate personalized responses. Each concept requires only 1 image and its descrip…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Personalized Multimodal Large Language Models"
  - "Retrieval-Augmented Generation"
  - "User Concept Memory"
  - "Personalized Dialogue"
  - "Data Construction"
date: 2026-05-08
content_hash: afd70d2dcf338072
---

# RAP: Retrieval-Augmented Personalization for Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2410.13360](https://arxiv.org/abs/2410.13360)  
**Code**: [https://hoar012.github.io/RAP-Project/](https://hoar012.github.io/RAP-Project/)  
**Area**: Multimodal VLM  
**Keywords**: Personalized Multimodal Large Language Models, Retrieval-Augmented Generation, User Concept Memory, Personalized Dialogue, Data Construction

## TL;DR

The RAP (Retrieval-Augmented Personalization) framework is proposed to achieve personalization in MLLMs via a three-step "Remember-Retrieve-Generate" pipeline. It stores user concepts in an external database, dynamically retrieves relevant concept information using a multimodal retriever, and injects it into the MLLM to generate personalized responses. Each concept requires only 1 image and its description, supporting real-time updates.

## Background & Motivation

Although existing MLLMs excel in general visual understanding, they lack user-specific knowledge (e.g., pet names, friends' identities), preventing them from acting as personalized assistants. Existing methods like MyVLM and Yo'LLaVA memorize concepts by learning external classification heads or special tokens, but they suffer from three major issues: (1) requiring a large number of annotated images (5 positive samples + 150-200 negative samples); (2) necessitating model retraining whenever a new concept is added; (3) being unable to edit concepts in real-time. The core motivation of RAP is to borrow the idea of RAG to decouple concept knowledge from model parameters into an external database, achieving "train once, adapt to infinite users."

## Method

### Overall Architecture

RAP consists of three phases: **(a) Remember**: constructing a key-value database to store the image and description of each concept; **(b) Retrieve**: when a user initiates a conversation, an open-world detector detects Regions of Interest (ROIs) in the image, and a multimodal retriever retrieves relevant concepts from the database; **(c) Generate**: the retrieved concept information (image + text description) along with the original input is fed into the MLLM to generate personalized responses.

### Key Designs

1. **External Concept Database (Remember)**:
    - **Function**: Stores personalized user concepts as key-value pairs, where each concept contains an avatar image $\mathbf{I}_j$, a name, and a short description $\mathbf{T}_j$.
    - **Mechanism**: The key $k_j$ of a concept is its visual feature extracted by a pre-trained image encoder $\mathcal{E}(\cdot)$. The database supports real-time addition and deletion of concepts without retraining.
    - **Design Motivation**: Decouple user knowledge from model parameters to external storage, requiring only 1 image and its description per concept, significantly lowering the barrier to personalization.

2. **Multimodal Retriever (Retrieve)**:
    - **Function**: Automatically identifies potential concepts in the input image and retrieves matching entries from the database when a user starts a conversation.
    - **Mechanism**: Uses YOLO-World as a general detector $\mathcal{R}(\cdot)$ to detect ROIs, extracts the CLIP visual feature $v_i = \mathcal{E}(\mathbf{X}_u^i)$ for each ROI, calculates the Euclidean distance $Dist(v_i, k_j) = \|v_i - k_j\|$ to all database keys, and selects the Top-K nearest neighbors. Text-based retrieval based on concept names is also supported.
    - **Design Motivation**: Replaces training dedicated classification heads for each concept with a general detector, enabling generalization to infinite new concepts without retraining.

3. **Personalized Training Dataset Construction Pipeline**:
    - **Function**: Provides large-scale training data for the personalization generation capability of the MLLM.
    - **Mechanism**: Comprises three types of data: (a) visual grounding data (RefCOCO + ILSVRC-VID + TAO + CustomConcept101) to train the model to locate concept positions in images; (b) instruction following data (image description, QA) annotated using Gemini-1.5; (c) negative samples data (adding noisy concepts in the input while keeping the answer unchanged) to train the model's noise-filtering capability. Additionally, diffusion models are used for data augmentation to generate novel perspectives.
    - **Design Motivation**: Existing work lacks large-scale training data for personalization, and the model needs to learn to "use relevant information + ignore irrelevant information."

### Loss & Training

- Uses the standard autoregressive language modeling loss $\prod_{i=1}^{L} p_\theta(\mathbf{X}_{a,i} | \mathbf{X}_v, \mathbf{X}_q, \mathbf{M}_1, \cdots \mathbf{M}_K, \mathbf{X}_{a,<i})$.
- During training, the detector and retriever parameters are frozen, and only the MLLM is trained (using LoRA to reduce trainable parameters).
- Trained based on LLaVA-1.5-13B and Phi3-V-3.8B, using 8x A100 GPUs, with a batch size of 64, learning rate of 1e-4, for 1 epoch.
- Part of the LLaVA-Instruct-665k data is retained to maintain general knowledge capability.

## Key Experimental Results

### Main Results

| Task | Metric | RAP-LLaVA | MyVLM | Yo'LLaVA | LLaVA-LoRA |
|------|------|-----------|-------|----------|------------|
| Personalized Captioning | F1-score | **94.97** | 85.50 | - | 87.82 |
| Visual Question Answering | Weighted Acc | **0.936** | - | 0.906 | 0.741 |
| Visual Recognition | Weighted Acc | **0.980** | 0.919 | 0.924 | 0.825 |
| Text Question Answering | Acc | **0.938** | - | 0.883 | 0.583 |

| Method | Positive Samples Needed | Negative Samples Needed | Supports Real-time Editing | Supports Text-only QA |
|---------|-----------|-----------|-----------|------------|
| RAP (Ours) | **1** | **0** | **✓** | **✓** |
| MyVLM | n | 150 | ✗ | ✗ |
| Yo'LLaVA | n | 200 | ✗ | ✓ |
| Fine-tuning | n | 0 | ✗ | ✓ |

### Ablation Study

| Configuration | Recall | Precision | F1-score | Note |
|------|--------|-----------|----------|------|
| RAP-LLaVA (Full) | 93.51 | 96.47 | 94.97 | - |
| Skip Retrieval (Oracle Information) | 96.16 | 100.0 | 98.04 | Retrieval bottleneck accounts for ~3% performance loss |
| Remove Text Information | 94.91 | 88.66 | 91.68 | Text descriptions aid accurate matching |
| Remove Data Augmentation | 89.25 | 98.01 | 93.42 | Augmentation improves recall |
| Remove Negative Sample Training | 95.74 | 58.21 | 72.40 | Negative samples are critical for precision |

### Key Findings

- Negative sample training is key to precision: removing it drops Precision from 96.47% to 58.21%, causing the model to incorrectly output irrelevant concepts.
- RAP is competitive with GPT-4V: RAP-LLaVA's performance with 1 image (0.936) is close to GPT-4V's performance with 5 images (0.937).
- As the number of concepts in the database increases, RAP's performance degrades the slowest (benefiting from its retrieval-over-memorization architecture).
- In text-only QA, RAP-LLaVA (0.938) significantly outperforms LLaVA-LoRA (0.583) because RAP can retrieve relevant information based on text names.

## Highlights & Insights

- **Minimalist Data Requirement**: Requires only 1 image and its description per concept, with no negative samples or retraining needed, making it the most data-efficient personalization scheme to date.
- **Real-Time Concept Editing**: Concepts can be added or deleted by simply modifying the external database without updating the model, suitable for real-time dynamic scenarios.
- **First Integration of RAG and Personalization**: Introducing the RAG paradigm into MLLM personalization represents a pioneering work at the intersection of retrieval augmentation and user personalization.
- **Data Construction Pipeline**: Systematically designs a construction workflow involving visual grounding, instruction-following, and negative samples, which is highly reusable.

## Limitations & Future Work

- The retriever's accuracy serves as the system bottleneck: under oracle retrieval conditions, the F1 score can reach 98.04%, indicating a performance gap of ~3% in retrieval.
- Constrained by the LLM context length, RAP-LLaVA can only retrieve 2 concepts (3 for RAP-Phi3-V), which limits multi-concept scenarios.
- Users are required to manually provide description information for each concept; automated description generation could further alleviate user burden.
- Has not been validated in temporal scenarios like video understanding.

## Related Work & Insights

- Core difference from MyVLM (external classification heads) and Yo'LLaVA (learning special tokens): RAP externalizes concept knowledge to a database, avoiding the computational overhead of continuous learning.
- Though RAG has been thoroughly validated in NLP (e.g., DPR, Self-RAG), this is its first systematic application in the MLLM personalization domain.
- Negative sample training strategy is critical for the robustness of retrieval-augmented systems and can be extended to other RAG scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ RAG + personalization is a novel combination, though individual components (retrieval, database, LoRA tuning) are mature techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers description, question answering, and recognition tasks with thorough ablation studies, though the evaluation dataset size is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, motivation is well-articulated, and the data construction pipeline details are clearly described.
- **Value**: ⭐⭐⭐⭐ Extremely low data requirements and real-time editing capabilities make it highly valuable in real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](../../NeurIPS2025/multimodal_vlm/windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[ACL 2026\] UniversalRAG: Retrieval-Augmented Generation for Multimodal Corpora](../../ACL2026/multimodal_vlm/universalrag_retrieval-augmented_generation_over_corpora_of_diverse_modalities_a.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
