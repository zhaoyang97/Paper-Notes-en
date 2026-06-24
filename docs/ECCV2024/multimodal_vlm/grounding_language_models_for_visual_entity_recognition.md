---
title: >-
  [Paper Note] Grounding Language Models for Visual Entity Recognition
description: >-
  [ECCV 2024][Multimodal VLM][Visual Entity Recognition] AutoVER is proposed, which is the first method to apply Multimodal Large Language Models (MLLMs) to large-scale visual entity recognition. By integrating retrieval capability directly inside the MLLM, and combining contrastive training with trie-constrained decoding, it substantially outperforms prior methods like PaLI-17B on the Oven-Wiki benchmark.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Visual Entity Recognition"
  - "Multimodal Large Language Models"
  - "Retrieval-Augmented Generation"
  - "Constrained Decoding"
  - "Knowledge Grounding"
date: 2026-05-08
content_hash: 6e53a4ae5511d4bf
---

# Grounding Language Models for Visual Entity Recognition

**Conference**: ECCV 2024  
**arXiv**: [2402.18695](https://arxiv.org/abs/2402.18695)  
**Code**: [https://github.com/MrZilinXiao/AutoVER](https://github.com/MrZilinXiao/AutoVER)  
**Area**: Information Retrieval  
**Keywords**: Visual Entity Recognition, Multimodal Large Language Models, Retrieval-Augmented Generation, Constrained Decoding, Knowledge Grounding

## TL;DR

AutoVER is proposed, which is the first method to apply Multimodal Large Language Models (MLLMs) to large-scale visual entity recognition. By integrating retrieval capability directly inside the MLLM, and combining contrastive training with trie-constrained decoding, it substantially outperforms prior methods like PaLI-17B on the Oven-Wiki benchmark.

## Background & Motivation

- **Challenges of Visual Entity Recognition (VER)**:
  1. The answer space exceeds 6 million Wikipedia entities, rendering classifier-based approaches infeasible.
  2. Generative VQA models are prone to hallucinating—the generated text may not map to any legitimate entity.
  3. Existing methods overlook the visual information of candidate entities.
  4. Generalization to out-of-domain unseen entities is difficult.
- **Oven-Wiki Benchmark**: Given an image + a question, find the exact entity answer from Wikipedia.
- **Core Idea**: Reframing entity recognition as a constrained sequence-to-sequence generation problem.

## Method

### Overall Architecture

AutoVER = MLLM (based on LLaVA/Vicuna) + Multimodal Entity Encoder + Retrieval-Augmented Constrained Decoding

### Key Designs

**1. Joint Contrastive-Generative Training**
- MLLM side: A special token `<ret>` is added, and its last-layer hidden state serves as the query representation $Q$.
- Entity encoder side: A 2-layer Transformer fuses the entity image and textual description, outputting the entity representation $E$.
- Contrastive loss (InfoNCE): $\mathcal{L}_{query2ent} = -\frac{1}{N}\sum \log \frac{\exp(sim(Q_i, E_i)/\tau)}{\sum_j \exp(sim(Q_i, E_j)/\tau)}$
- Language modeling loss (next token prediction): $\mathcal{L}_{LM}$
- Total loss: $\mathcal{L} = \mathcal{L}_{LM} + \lambda_r \cdot \mathcal{L}_{query2ent}$

**2. Hard Negative Mining**
- **vision-hard**: A pretrained ViT classifier is used to identify visually similar entities (sharing predicted categories).
- **kb-hard**: Utilizing the Wikidata category hierarchy, entities that share a parent node are treated as knowledge-similar entities.
- Avoid duplicate entities in the same batch via rejection sampling.

**3. Retrieval-Augmented Constrained Decoding (Inference Phase)**
- Pre-cache all entity vectors using the trained entity encoder $\to$ Construct an entity vector database.
- During inference, perform top-$k$ similarity search using the representation of the `<ret>` token to retrieve $k=300$ candidates.
- Dynamically construct a **prefix tree (trie)** covering candidate entity identifiers.
- During autoregressive generation, the prefix tree constrains the valid tokens at each step, eliminating invalid decoding paths.
- **Guarantees that the generated content always matches an entity in the knowledge base.**

### Loss & Training

- Initialization: LLaVA architecture, Vicuna-7B/13B, CLIP-ViTL/14-336px.
- Training data: Oven-Wiki ~5 million query-entity pairs.
- 32× V100 GPUs, batch size 256.
- $\lambda_r = 1$, entity descriptions truncated to 77 tokens.

## Key Experimental Results

### Main Results (Oven-Wiki Validation Accuracy)

| Method | Entity seen | Entity unseen | Entity hm | Query seen | Query unseen | Overall hm |
|------|-------------|---------------|-----------|------------|--------------|------------|
| CLIP Fusion | 32.7 | 4.3 | 7.7 | 33.4 | 2.2 | 5.4 |
| PaLI-17B | 30.6 | 12.4 | 17.6 | 44.2 | 22.4 | 22.1 |
| GPT-4V (zero-shot) | 29.8 | 19.3 | 23.4 | 56.5 | 52.7 | 32.9 |
| **AutoVER-7B** | **61.5** | **21.7** | **32.1** | **69.0** | **31.4** | **36.8** |
| **AutoVER-13B** | **63.6** | **24.5** | **35.6** | **68.6** | **32.3** | **39.2** |

### Ablation Study

- Removing contrastive training $\to$ entity seen accuracy drops by approximately 10%.
- Removing constrained decoding $\to$ entity unseen accuracy drops significantly.
- Removing hard negative mining $\to$ fine-grained recognition capability decreases.

### Key Findings

- Entity seen accuracy doubles from PaLI-17B's 30.6% to 61.5% (with fewer parameters).
- Constrained decoding **eliminates hallucinations**: guaranteeing that the generated content corresponds to a real entity.
- Demonstrates strong performance in zero-shot transfer on A-OKVQA-Ent, proving generalization capability.
- AutoVER-13B achieves 53.7% on the human evaluation set (compared to Human+Search at 77.7%, there is still a gap).

## Highlights & Insights

1. **Unified retrieval-and-generation framework**: Retrieval capability is built directly into the MLLM, eliminating the need for external retrievers.
2. **Prefix tree constrained decoding** ensures grounded generation results, thoroughly eradicating hallucinations.
3. **Joint contrastive-generative training** balances retrieval precision and generation quality.
4. **Hard negative mining strategy** (dual-path: visual + knowledge base) effectively enhances fine-grained discrimination capabilities.

## Limitations & Future Work

- The accuracy on the entity unseen subset remains low (21.7%), indicating that out-of-domain generalization is still a primary bottleneck.
- Pre-caching the entity database requires substantial storage and top-$k$ search overhead.
- The prefix tree may face efficiency issues when the candidate pool is extremely large (e.g., > 10,000).
- There is still room for improvement regarding the reasoning demands on the query split (such as spatial relationships and common sense).

## Related Work & Insights

- FROMAGe/GILL inspired the design of adding retrieval tokens in MLLMs.
- Generative Entity Linking (GENRE) offers insights for knowledge grounding in the textual domain.
- Inspirational: Extrapolating the constrained decoding framework to other tasks requiring structured output (e.g., knowledge graph completion, structured information extraction).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Unified RAG framework + constrained decoding eliminating hallucinations)
- Technical Depth: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple splits + zero-shot transfer + ablations)
- Writing Quality: ⭐⭐⭐⭐
- Overall Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WikiCLIP: An Efficient Contrastive Baseline for Open-domain Visual Entity Recognition](../../CVPR2026/multimodal_vlm/wikiclip_an_efficient_contrastive_baseline_for_open-domain_visual_entity_recogni.md)
- [\[ECCV 2024\] LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers](loa-trans_enhancing_visual_grounding_by_location-aware_transformers.md)
- [\[ECCV 2024\] ArtVLM: Attribute Recognition Through Vision-Based Prefix Language Modeling](artvlm_attribute_recognition_through_vision-based_prefix_language_modeling.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)
- [\[ECCV 2024\] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)

</div>

<!-- RELATED:END -->
