---
title: >-
  [Paper Note] HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval
description: >-
  [ACL 2025 (Long Paper)][Multimodal Efficiency][Multimodal Dense Retrieval] This paper proposes HotelMatch-LLM, an asymmetric architecture that employs an SLM to encode queries and an LLM to encode hotel documents. Combined with a tri-objective multi-task optimization (retrieval alignment + MLM geographic prediction + visual facility recognition) and patch-level mean pooling for multi-image processing, it significantly outperforms SOTA methods like MARVEL and VISTA on travel-d…
tags:
  - "ACL 2025 (Long Paper)"
  - "Multimodal Efficiency"
  - "Multimodal Dense Retrieval"
  - "Asymmetric Encoder"
  - "Multi-Task Optimization"
  - "Hotel Search"
  - "SLM-LLM Joint Training"
date: 2026-05-08
content_hash: 1ad3e0d2bc16de91
---

# HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2506.07296](https://arxiv.org/abs/2506.07296)  
**Code**: None  
**Institution**: Leiden University, Booking.com  
**Area**: Multimodal Retrieval / Information Retrieval / Travel Search  
**Keywords**: Multimodal Dense Retrieval, Asymmetric Encoder, Multi-Task Optimization, Hotel Search, SLM-LLM Joint Training  

## TL;DR
This paper proposes HotelMatch-LLM, an asymmetric architecture that employs an SLM to encode queries and an LLM to encode hotel documents. Combined with a tri-objective multi-task optimization (retrieval alignment + MLM geographic prediction + visual facility recognition) and patch-level mean pooling for multi-image processing, it significantly outperforms SOTA methods like MARVEL and VISTA on travel-domain multimodal retrieval tasks.

## Background & Motivation
Current online travel search platforms (e.g., Booking.com) rely on predefined filters: users must first select a country/city, and then set parameters such as price and star rating. This approach fails to handle complex user needs described in natural language, such as "a boutique hotel with a pool and sea view" or "hotels rated above 3 stars close to a subway station." Existing multimodal retrieval models (MARVEL, VISTA) perform well in web search, but only support single-image inputs, which cannot handle hotel galleries that often contain dozens or hundreds of images. Meanwhile, online inference costs for LLM query embedding are prohibitively high, making them unsuitable for production environments with millions of daily queries.

## Core Problem
1. **How to support natural language multimodal retrieval in the travel domain?** The traditional keyword + filter paradigm does not support free-form queries.
2. **How to efficiently process a large number of attribute images?** A hotel can have up to 306 images, whereas existing methods can only handle a single image.
3. **How to balance retrieval performance and online inference efficiency?** Using an LLM for query embeddings is effective but too slow.

## Method

### Overall Architecture
HotelMatch-LLM utilizes an asymmetric dual-encoder retrieval architecture. The query side is encoded by a small model (GTR-Base-110M), while the document side is encoded by a large model (GTR-Large-335M) to represent the textual and visual information of hotels. Relevance is calculated via cosine similarity. During training, three objectives are optimized jointly: retrieval alignment, MLM geographic information prediction, and visual facility recognition.

**Input**: query text $\rightarrow$ SLM encoding to obtain query embedding; hotel images (multiple) $\rightarrow$ CLIP encoding + mean pooling + linear projection to obtain visual tokens, concatenated with hotel text descriptions $\rightarrow$ LLM encoding to obtain document embedding.

### Key Designs

1. **Multi-Image Patch Mean Pooling**:
   Each image is passed through a CLIP vision encoder to extract 49 patch embeddings (224×224 image + 32×32 window). Then, mean pooling is performed across the corresponding patch positions of all images, resulting in a fixed-size $(49 \times \text{dim})$ representation. Consequently, regardless of how many images a hotel has, they are ultimately compressed into 49 visual tokens. These are then projected into the token embedding space of the LLM via a linear layer and concatenated with text tokens before being fed into the LLM. This design theoretically can process an **infinite number** of images.

2. **SLM-LLM Asymmetric Architecture**:
   The core insight is that hotel data is far more complex than queries, thus requiring a larger model for representation. Therefore, an SLM (GTR-Base, 110M) is used for online query embedding, and an LLM (GTR-Large, 335M) is used for offline document embedding. The SLM output is projected into the same dimensional space as the LLM via a linear layer. A key detail is that the SLM and LLM require **different learning rates** (SLM: 5e-4, LLM: 5e-6); using a shared learning rate leads to a severe drop in performance.

3. **Domain-Specific Multi-Task Optimization**:

    - **Retrieval Loss $\mathcal{L}_{Ret}$** (weight 0.7): Standard contrastive learning using softmax + cross-entropy to pull closer the cosine similarity between queries and positive documents.
    - **MLM Loss $\mathcal{L}_{MLM}$** (weight 0.2): Masks city and country tokens in the hotel description to predict geographic info, enhancing the model's understanding of geographic features.
    - **Visual Facility Recognition Loss $\mathcal{L}_{VisF}$** (weight 0.1): Predicts the presence of 120 types of facilities (swimming pool, gym, balcony, etc.) based on document embeddings, with labels automatically identified via the MUMIC method. Binary cross-entropy loss is used.
    - Weights are determined on the validation set via grid search (with a step size of 0.1).

### Loss & Training
- Total Loss: $$\mathcal{L}_{final} = 0.7 \cdot \mathcal{L}_{Ret} + 0.2 \cdot \mathcal{L}_{MLM} + 0.1 \cdot \mathcal{L}_{VisF}$$
- Training labels are generated by GPT-4o (binary relevance annotations of query-hotel pairs), which exhibit a strong correlation with human annotations.
- Training runs for 10 epochs with early stopping (no improvement on the validation set for 5 steps).
- Evaluation follows a process of first retrieving top-100 candidates with CLIP, followed by re-ranking.

## Key Experimental Results

| Test Set (Metric) | Real-world MRR/nDCG | Vision-driven MRR/nDCG | Text-driven MRR/nDCG | OOD MRR/nDCG |
|---|---|---|---|---|
| BM25 | .506/.401 | .138/.195 | .798/.825 | .588/.489 |
| MARVEL (Fine-tuned) | .603/.503 | .219/.326 | .810/.833 | .660/.515 |
| VISTA (Fine-tuned) | .582/.465 | .216/.321 | .802/.839 | .662/.513 |
| **HotelMatch-LLM** | **.681/.600** | **.247/.362** | **.863/.884** | **.704/.558** |

- Main test set MRR@10: 0.681 vs MARVEL 0.603 $\rightarrow$ **Gain 12.9%**
- Statistically significantly outperforms all baselines across all four test sets (paired t-test, p<0.05, Bonferroni correction).
- Full-ranking (3.1M documents) MRR: 0.675 vs MARVEL 0.589.

**Efficiency Comparison**:

| Model | Inference Latency (ms) | MRR | nDCG |
|---|---|---|---|
| VISTA | 16.17 | .572 | .465 |
| MARVEL | 31.07 | .603 | .503 |
| HotelMatch-LLM | 18.69 | .681 | .600 |

Inference latency is only 2.5ms higher than VISTA but achieves substantially better results; it is 1.7x faster than MARVEL.

### Ablation Study
- **Multi-task Ablation**: Removing MLM lead to the largest drop (MRR .681 $\rightarrow$ .650, -4.6%), indicating that geographic information understanding is crucial for hotel retrieval. Removing VisF also led to a noticeable drop ($\rightarrow$ .664). Removing both $\rightarrow$ .632.
- **Multi-image Method Comparison**: mean pooling over patches (unlimited images) > 1TPI-Patch (.672, max 50 images) > 1TPI-CLS (.652, max 50 images).
- **LLM Backbone Generalization**: When using Zeta-Alpha-E5-Mistral-7B as the document encoder, MRR reaches .719, and Stella-en-1.5B reaches .694, indicating that larger LLMs provide better hotel representations.
- **Learning Rate**: Separating learning rates for SLM and LLM (5e-4 / 5e-6) yields MRR=.681, whereas a shared learning rate yields only .315, showing an extreme performance gap.
- **Without Vision**: HotelMatch-LLM w/o vision MRR .595, showing that visual info contributes significantly.

## Highlights & Insights
- **Asymmetric architecture concept is highly practical**: Using an SLM for online queries and an LLM for offline documents is highly rational and easy to deploy in industrial-grade systems. The latency is only 18.69ms, which is close to the lightweight VISTA.
- **Patch-level mean pooling for multi-image processing is simple and elegant**: It is free from image quantity limitations and avoids token explosion. It handled up to 306 images in the experiments.
- **Tri-objective multi-task design is highly domain-targeted**: MLM for geographic prediction and VisF for facility recognition represent domain-specific signals in hotel retrieval scenarios, rather than arbitrary auxiliary tasks.
- **The separate learning rates detail is crucial**: Utilizing differing LRs when jointly training the SLM and LLM is key, as a shared LR immediately causes performance to collapse.

## Limitations & Future Work
- **Dependency on GPT-4o for generating training labels**: If the synthetic annotations contain bias, the model quality could be affected, and it hinders exact reproducibility.
- **No support for multimodal queries**: Currently, the model only supports text-only queries, failing to handle queries with images like "find hotels similar to this image".
- **Lack of personalization**: User historical preferences and interaction behaviors are not considered.
- **Dataset is not publicly available**: The HotelMatch dataset is from Booking.com's internal data, preventing reproducibility verification.
- **Relatively small SLM/LLM backbones**: The GTR model used in the main experiments maximums at 335M, which is small compared to today's mainstream 7B+ models. Although a 7B model was evaluated in the generalization experiments, it was not fully explored.

## Related Work & Insights
- **vs MARVEL** (ACL 2024): MARVEL is a SOTA multimodal retrieval method, but it only handles a single image and employs a symmetric encoder architecture. HotelMatch-LLM significantly outperforms MARVEL on all test sets and is 1.7x faster during inference.
- **vs VISTA** (ACL 2024): VISTA also supports only a single image and underperforms MARVEL in the domain-specific scenario of this paper. HotelMatch-LLM achieves all-round superior results while increasing latency by only 2.5ms.
- **vs CLIP**: Zero-shot CLIP performs average on domain-specific data, and its multimodal version does not bring significant improvement, demonstrating that domain-specific fine-tuning is crucial for travel search.

## Inspirations & Connections
- The patch-level mean pooling idea for multi-image aggregation can be transferred to other multi-image scenarios (product retrieval, real estate search, medical multi-view imaging, etc.).
- The split-learning-rate joint training trick for SLM+LLM serves as a valuable reference for all asymmetric architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of an asymmetric architecture, domain-specific multi-task learning, and multi-image handling is meaningful, even if individual components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four test sets, extensive ablations, generalization analyses, and efficiency evaluations, but the lack of a public dataset remains a major drawback.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, standard formulation, and complete experimental logic.
- Value: ⭐⭐⭐⭐ Strong industrial utility; the experience with asymmetric architectures and multi-image processing is highly valuable to the IR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval](omgm_orchestrate_multiple_granularities_and_modalities_for_efficient_multimodal_.md)
- [\[ICLR 2026\] Task-Related Token Compression in Multimodal Large Language Models from an Explainability Perspective](../../ICLR2026/vlm_efficiency/task-related_token_compression_in_multimodal_large_language_models_from_an_expla.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](../../ICCV2025/vlm_efficiency/folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/vlm_efficiency/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)

</div>

<!-- RELATED:END -->
