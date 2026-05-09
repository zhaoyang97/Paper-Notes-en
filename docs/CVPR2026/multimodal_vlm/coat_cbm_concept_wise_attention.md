---
title: >-
  [Paper Note] Concept-wise Attention for Fine-grained Concept Bottleneck Models
description: >-
  [CVPR 2026][Multimodal VLM][Concept Bottleneck Models] CoAt-CBM achieves adaptive fine-grained image–concept alignment via learnable concept-wise visual queries and Concept Contrastive Optimization (CCO), surpassing both existing concept bottleneck models and black-box models while maintaining high interpretability.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Concept Bottleneck Models
  - Interpretability
  - CLIP
  - Contrastive Learning
  - Fine-grained Alignment
date: 2026-05-08
content_hash: ceeee7243432bb76
---

# Concept-wise Attention for Fine-grained Concept Bottleneck Models

**Conference**: CVPR 2026
**arXiv**: [2604.15748](https://arxiv.org/abs/2604.15748)
**Code**: Unavailable (to be released upon acceptance)
**Area**: Multimodal VLM
**Keywords**: Concept Bottleneck Models, Interpretability, CLIP, Contrastive Learning, Fine-grained Alignment

## TL;DR

CoAt-CBM achieves adaptive fine-grained image–concept alignment via learnable concept-wise visual queries and Concept Contrastive Optimization (CCO), surpassing both existing concept bottleneck models and black-box models while maintaining high interpretability.

## Background & Motivation

**Background**: Concept Bottleneck Models (CBMs) provide transparent decision paths by first predicting a set of human-understandable concepts and then performing final classification based on those concepts. Recent work has leveraged pretrained vision-language models such as CLIP to enhance CBM performance.

**Limitations of Prior Work**: Existing VLM-based CBMs face two critical limitations. First, when computing concept scores, they either rely on frozen coarse-grained global features (ResCBM, HybridCBM), resulting in a coarse-to-fine granularity mismatch, or employ optimal transport (DOT-CBM) to assign patch tokens, which depends on pretrained structural priors and incurs high computational cost. Second, the commonly used BCE loss treats each concept independently, ignoring mutual exclusivity among concepts and failing to exploit negative concepts as references to improve discrimination of positive concepts.

**Key Challenge**: Pretraining bias leads to inaccurate fine-grained alignment between visual features and textual concepts, while independently optimized loss functions prevent the model from learning the relative importance among concepts.

**Goal**: To achieve adaptive fine-grained image–concept alignment while simultaneously improving classification performance and interpretability.

**Key Insight**: Introducing learnable concept-wise visual queries to adaptively disentangle visual features, and replacing BCE with contrastive constraints to model inter-concept relationships.

**Core Idea**: Each concept is assigned a learnable query that extracts concept-specific representations from visual features via an attention mechanism; a multi-positive contrastive loss then optimizes the relative ranking of concept scores.

## Method

### Overall Architecture

The CoAt-CBM pipeline proceeds as follows: (1) construct a domain knowledge base and concept library; (2) a CLIP visual encoder extracts global and patch features; (3) a concept-wise attention module uses learnable queries to extract concept-specific visual embeddings; (4) cosine similarities between visual and text embeddings yield a concept score vector; (5) a linear classifier produces predictions based on the concept scores.

### Key Designs

1. **Concept-wise Attention Module**:

    - **Function**: Adaptively extracts visual features relevant to each concept.
    - **Mechanism**: A learnable query $\mathbf{q}_i \in \mathbb{R}^{d_k}$ is defined for each of the $n$ concepts. The global and patch features $\mathbf{Z}$ extracted by CLIP are projected into keys and values. Each query computes attention weights via scaled dot-product attention $\bm{\alpha}_i = \text{Softmax}(\mathbf{K}\mathbf{q}_i / \sqrt{d_k})$, and the concept-level visual embedding is obtained by weighted aggregation $\mathbf{e}_i = \mathbf{V}^\top \bm{\alpha}_i$. Different queries automatically learn to attend to different visual regions.
    - **Design Motivation**: To overcome the granularity mismatch of frozen global features and the dependence of OT-based methods on structural priors, enabling the model to dynamically disentangle visual features into concept-specific representations.

2. **Concept Contrastive Optimization (CCO)**:

    - **Function**: Enhances the discriminability of concept scores through contrastive constraints.
    - **Mechanism**: Concept scores are partitioned into a positive set $\mathbf{s}^+$ (concepts associated with the image category) and a negative set $\mathbf{s}^-$ (irrelevant concepts). A multi-positive contrastive loss $\mathcal{L}_{CCO} = -\log \frac{\sum \exp(s_i^+/\tau)}{\sum \exp(s_i^+/\tau) + \sum \exp(s_i^-/\tau)}$ forces the model to assign higher scores to positive concepts. Rather than optimizing each concept in isolation, negative concepts serve as references to enhance positive concept discrimination.
    - **Design Motivation**: The independence assumption of BCE limits the model's ability to exploit inter-concept relationships; contrastive optimization explicitly models the relative ordering between positive and negative concepts.

3. **Domain Knowledge Concept Library Construction**:

    - **Function**: Establishes a reliable concept set, reducing hallucination and incompleteness.
    - **Mechanism**: Domain-specific knowledge descriptions for each category are collected from specialized websites and used as the input basis for concept generation by GPT-3.5-Turbo. Concept generation is grounded entirely in the domain knowledge base rather than in the model's limited parametric knowledge, reducing hallucinations and omissions.
    - **Design Motivation**: Directly prompting LLMs to generate concepts leads to hallucinations or omissions, while learnable concepts lack semantic clarity.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{CCO}$, where $\lambda = 0.5$ by default. Training uses CLIP-ViT-L/14 with the AdamW optimizer on a single RTX 3090.

## Key Experimental Results

### Main Results

| Method | Interpretable | CIFAR-10 | CIFAR-100 | CUB-200 |
|--------|---------------|----------|-----------|---------|
| Linear Probe | ✗ | 97.93 | 87.26 | 85.48 |
| HybridCBM | ✓ | 97.91 | 86.22 | 84.25 |
| DOT-CBM | ✓ | 97.75 | 84.75 | 83.76 |
| **CoAt-CBM** | **✓** | **98.51** | **89.19** | **89.13** |

### Ablation Study

| Configuration | CIFAR-10 CDR | CIFAR-10 CC |
|---------------|-------------|------------|
| CoAt-CBM w/o CCO | 9.88 | 25.48 |
| CoAt-CBM_BCE | 82.16 | 85.42 |
| **CoAt-CBM** | **89.64** | **94.76** |

### Key Findings

- CoAt-CBM surpasses the black-box Linear Probe while maintaining full interpretability, challenging the assumption that interpretability necessarily sacrifices performance.
- A gain of 4.88% on CUB-200 (89.13 vs. 84.25) demonstrates particularly significant improvements on fine-grained classification.
- CCO is critical for interpretability metrics: CDR improves from 9.88% to 89.64%, revealing that under BCE training, models may classify accurately yet produce concept scores inconsistent with image content.
- The concept-wise attention module consistently outperforms Adapter and LoRA alternatives.

## Highlights & Insights

- **CCO reveals a fundamental flaw of BCE**: Even when classification is accurate, models trained with BCE nearly fail at concept-level interpretability (CDR of only 9.88%). By introducing inter-concept contrast, CCO aligns score rankings closely with actual image content.
- **Clear few-shot advantage**: CoAt-CBM outperforms Linear Probe and LoRA-LP across all settings from 1-shot to 16-shot, indicating that concept priors provide effective inductive bias.
- **Interpretable class–concept associations**: CCO transforms the class–concept association matrix from a noisy state into a clear diagonal structure.

## Limitations & Future Work

- The quality of the concept library depends on the quality of domain knowledge collected, which may be insufficient for obscure domains.
- The one-query-per-concept design may encounter memory bottlenecks when the number of concepts is very large.
- Validation is primarily conducted on classification tasks; extension to more complex tasks such as detection and segmentation remains to be explored.

## Related Work & Insights

- **vs. HybridCBM**: HybridCBM uses learnable concept vectors to capture missing concepts but still relies on frozen global features; CoAt-CBM achieves finer-grained alignment through the attention mechanism.
- **vs. DOT-CBM**: DOT-CBM aligns patches and concepts via optimal transport, incurring high computational overhead and dependence on structural priors; CoAt-CBM is more flexible and efficient.
- **vs. PCBM**: PCBM constructs the concept bottleneck using projection distances, with accuracy limited by global feature quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of concept-wise attention and CCO elegantly addresses two key problems.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ten datasets, comprehensive interpretability evaluation, and coverage from few-shot to full-data settings.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is clear; the design of interpretability metrics is convincing.
- **Value**: ⭐⭐⭐⭐⭐ The first work to enable interpretable CBMs to comprehensively surpass black-box models, with significant practical implications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models](clip-free_label_free_unsupervised_concept_bottleneck_models.md)
- [\[CVPR 2026\] Vision-Language Models Encode Clinical Guidelines for Concept-Based Medical Reasoning](vision-language_models_encode_clinical_guidelines_for_concept-based_medical_reas.md)
- [\[CVPR 2026\] DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)
- [\[CVPR 2026\] No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](no_hard_negatives_required_concept_centric_learning_leads_to_compositionality_wi.md)
- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)

<!-- RELATED:END -->
