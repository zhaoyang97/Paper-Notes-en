---
title: >-
  [Paper Note] Training-Free Personalization via Retrieval and Reasoning on Fingerprints
description: >-
  [VLM Reasoning] This work proposes R2P, the first training-free VLM personalization method. It leverages the inherent world knowledge of VLMs to extract concept "fingerprint" attributes, achieving personal concept recognition through a retrieval-reasoning paradigm and cross-modal attribute verification, without requiring any fine-tuning or large-scale pre-training.
tags:
  - "VLM Reasoning"
date: 2026-05-08
content_hash: 9fd1436a362f1e60
---

# Training-Free Personalization via Retrieval and Reasoning on Fingerprints

## Meta Info
- **Conference**: ICCV 2025
- **arXiv**: [2503.18623](https://arxiv.org/abs/2503.18623)
- **Code**: [Project Page](https://deepayan137.github.io/R2P/)
- **Area**: Multimodal VLM
- **Keywords**: VLM Personalization, Training-Free, Retrieval-Reasoning, Concept Fingerprints, Cross-Modal Verification

## TL;DR

This work proposes R2P, the first training-free VLM personalization method. It leverages the inherent world knowledge of VLMs to extract concept "fingerprint" attributes, achieving personal concept recognition through a retrieval-reasoning paradigm and cross-modal attribute verification, without requiring any fine-tuning or large-scale pre-training.

## Background & Motivation

VLM personalization aims to enable models to understand user-specific concepts (e.g., "my keys", "what is Fluffy doing"). However, existing methods suffer from severe limitations:

**Training Dependency**: Methods like MyVLM and Yo'LLaVA require fine-tuning for each new concept, necessitating the collection of multiple reference samples and numerous negative samples.

**Large-Scale Pre-training**: Methods like RAP require expensive large-scale instruction tuning on synthesized personalization data.

**Poor Scalability**: Adding new concepts requires retraining, leading to high time and computational costs.

**Key Insight**: VLMs have already been exposed to almost all semantic concepts through training on web-scale data. Can we leverage the existing internal knowledge of VLMs to achieve personalization without any training?

## Method

### Overall Architecture

R2P consists of two phases:
- **Phase I: Personal Database Creation**: Extracts "fingerprint" attributes for each concept.
- **Phase II: Inference Phase**: Recognizes personal concepts through a retrieval-reasoning pipeline.

### Phase I: Concept Fingerprint Extraction

Given a reference image $I_i$, a concept name $c_i$, and a semantic category $g_i$, a VLM is used to extract unique fingerprint attributes $A_i$ and a description $d_i$:

$$\{A_i, d_i\} = \Phi_{\text{VLM}}(P_D^V, P_D^T)$$

Fingerprint attributes are used to uniquely identify concepts (e.g., "orange and white fur pattern" or "bell on collar" to distinguish different cats). Meanwhile, CLIP is employed to encode the embeddings of the reference image and description.

### Phase II: Retrieval-Reasoning Inference

**1. Multimodal Concept Retrieval**: Calculate the visual and textual similarity between the query image and all concepts in the database:

$$s_{q,i} = \frac{1}{2}(s_{q,i}^{V,V} + s_{q,i}^{V,T})$$

Select the top-$K$ candidate concepts $\mathcal{C}^K$.

**2. Attribute-Focused CoT Reasoning**: Prompt the VLM to focus on fingerprint attributes to infer the best-matching candidate concept $\tilde{c}$ in the query image, while extracting the shared attribute set $A_{q,i}$ between each candidate concept and the query image.

**3. Cross-Modal Attribute Verification**: Since VLMs may hallucinate, the reasoning result is verified. The attribute-based cross-modal similarity is calculated:

$$s_{q,i}^{V,A} = \frac{1}{|A_{q,i}|} \sum_{a_j \in A_{q,i}} \langle f_q^V, f_{a,j}^T \rangle$$

If $\tilde{c} = \tilde{c}_a$ (the CoT reasoning result is consistent with the attribute verification), the verification passes.

**4. Pairwise Reasoning (Triggered upon verification failure)**: Compare each candidate concept with the query image one by one, calculating the matching probability:

$$p_i = \frac{\lambda_i^{\text{Yes}}}{\lambda_i^{\text{Yes}} + \lambda_i^{\text{No}}}$$

Select the concept with the highest probability as the final prediction.

## Experiments

### Identification and Description Task Results

| Method | Training? | MyVLM Wtd↑ | Yo'LLaVA Wtd↑ | Yo'LLaVA Recall↑ | PerVA Wtd↑ | PerVA Recall↑ |
|------|------|------------|---------------|-------------------|------------|---------------|
| MyVLM | ✓ | 93.8 | - | 0.1 | 62.2 | 12.3 |
| Yo'LLaVA | ✓ | 96.4 | 89.7 | 64.8 | 72.0 | 38.2 |
| RAP | ✓ | 96.6 | 91.8 | 81.6 | 89.0 | 64.1 |
| MiniCPM-o + prompt | ✗ | 96.1 | 89.4 | 78.5 | 88.5 | 65.7 |
| **R2P (Ours)** | **✗** | **97.4** | **94.0** | **87.1** | **91.8** | **72.5** |

R2P achieves state-of-the-art (SOTA) results across all datasets while being training-free. On PerVA, it outperforms the training-based method RAP by 2.8% Wtd and 8.4% Recall.

### PerVA Dataset

The authors introduce a new personalization benchmark:
- 329 personal concepts spanning 21 everyday object categories.
- Emphasizes **visual ambiguity**: multiple instances from the same category possess high visual similarity.
- Contains challenges such as deformation, illumination variations, and different states.

### Ablation Study

| Component | Yo'LLaVA Pos.Acc | Yo'LLaVA Neg.Acc | Yo'LLaVA Wtd |
|------|------------------|------------------|-------------|
| Visual Retrieval Only | 89.6 | 85.2 | 87.4 |
| + Text Retrieval | 91.8 | 88.7 | 90.3 |
| + CoT Reasoning | 95.2 | 90.4 | 92.8 |
| + Attribute Verification | 95.8 | 91.6 | 93.7 |
| + Pairwise Reasoning | **96.1** | **91.9** | **94.0** |

Each component progressively improves the performance, with cross-modal attribute verification and pairwise reasoning effectively reducing VLM hallucinations.

### Key Findings

1. Training-free methods outperform training-based ones, demonstrating that the world knowledge embedded in VLMs is sufficient to support personalization.
2. Purely textual reasoning (CoT) contributes significantly to positive accuracy, while cross-modal verification enhances negative accuracy.
3. Multimodal retrieval combining visual and textual modalities is more robust than single-modality retrieval.
4. When concepts are highly visually similar, the fine-grained discriminative capability of fingerprint attributes is critical.

## Highlights & Insights

1. **Pioneering Training-Free VLM Personalization**: Challenges the assumption that training is necessary for personalization.
2. **Concept Fingerprints**: Regulates individual recognition as an attribute-matching problem, providing great interpretability.
3. **Hierarchical Reasoning Pipeline**: The pipeline of retrieval $\rightarrow$ CoT $\rightarrow$ verification $\rightarrow$ pairwise reasoning elegantly handles queries of varying difficulties.
4. The PerVA dataset fills the gap in personalization evaluation under visually ambiguous scenarios.

## Limitations & Future Work

1. Relies on the quality of fingerprint attributes generated by the VLM; VLM hallucinations can lead to incorrect fingerprints.
2. The inference pipeline involves multiple VLM calls, resulting in higher latency.
3. It assumes that users provide accurate semantic category information.
4. Currently, it only supports a single reference image.

## Related Work

- **VLM Personalization**: MyVLM (concept vectors), Yo'LLaVA (concept tokens), RAP (retrieval + instruction tuning)
- **Attribute-Based Recognition**: Zero-shot learning, compositional recognition
- **Retrieval-Augmented Reasoning**: RAG, ICL

## Rating

- **Novelty**: ★★★★★ — Pioneering work in the training-free personalization paradigm.
- **Value**: ★★★★★ — New concepts can be added without training, making it highly practical and deployment-friendly.
- **Experimental Thoroughness**: ★★★★☆ — Three datasets + exhaustive ablations + new dataset.
- **Writing Quality**: ★★★★★ — Extremely clear description of methodology, with well-motivated steps for every part of the pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[CVPR 2025\] Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents](../../CVPR2025/vlm_reasoning/document_haystacks_vision-language_reasoning_over_piles_of_1000_documents.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](../../ACL2025/vlm_reasoning/mammoth_vl_multimodal_reasoning.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/vlm_reasoning/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)

</div>

<!-- RELATED:END -->
