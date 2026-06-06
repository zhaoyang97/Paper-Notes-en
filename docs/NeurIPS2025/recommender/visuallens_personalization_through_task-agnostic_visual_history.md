---
title: >-
  [Paper Note] VisualLens: Personalization through Task-Agnostic Visual History
description: >-
  [NeurIPS 2025][Recommender Systems][Multimodal Recommendation] This paper proposes VisualLens, a framework that leverages users' task-agnostic visual history (everyday photos) to enable cross-domain personalized recommen…
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "Multimodal Recommendation"
  - "Visual History"
  - "Personalization"
  - "MLLM"
  - "User Profiling"
date: 2026-05-08
content_hash: d8e960290040419f
---

# VisualLens: Personalization through Task-Agnostic Visual History

**Conference**: NeurIPS 2025
**arXiv**: [2411.16034](https://arxiv.org/abs/2411.16034)  
**Code**: None  
**Area**: Recommender Systems
**Keywords**: Multimodal Recommendation, Visual History, Personalization, MLLM, User Profiling

## TL;DR
This paper proposes VisualLens, a framework that leverages users' task-agnostic visual history (everyday photos) to enable cross-domain personalized recommendation via spectrum user profiles and multimodal large language models. On the newly constructed Google Review-V and Yelp-V datasets, VisualLens surpasses GPT-4o by 2–5% in Hit@3.

## Background & Motivation
**Background**: Existing recommender systems primarily rely on users' item interaction histories (e.g., purchase records, click logs) or textual signals. While multimodal recommendation has advanced (e.g., UniMP), it remains confined to domain-specific item-level histories and cannot support cross-domain general-purpose personalized recommendation.

**Limitations of Prior Work**: Item-level interaction histories are not always available (cold-start problem) and do not generalize across tasks — an e-commerce platform may know a user's shopping preferences while remaining uninformed about their restaurant preferences. Existing multimodal recommendation methods rely heavily on structured item features and cannot handle unstructured visual signals.

**Key Challenge**: Everyday photos taken by users contain rich preference signals (e.g., preferences for architectural styles or food types), yet a large semantic gap exists between these photos and the recommendation task — photos may be entirely irrelevant to the current query and tend to be noisy with low information density.

**Goal**: (1) How to extract user preferences from task-agnostic everyday photos? (2) How to process large volumes of visual history within limited model context windows? (3) The lack of datasets for evaluating such systems.

**Key Insight**: Inspired by Vannevar Bush's Memex concept, the authors hypothesize that users' visual history (everyday photos) encodes cross-domain preference signals usable for general-purpose personalized recommendation. The key insight is to represent each image as a "spectrum triplet" (raw image → caption → aspect words), balancing information richness with semantic clarity.

**Core Idea**: Compress everyday photos into (image, caption, aspect words) triplets via spectrum user profiles, and achieve cross-domain recommendation from visual history through image gridification and joint training.

## Method

### Overall Architecture
VisualLens comprises two main stages: offline user profile generation and online recommendation. In the offline stage, CLIP encodings, captions, and aspect words are generated for each image in the user's visual history to construct the spectrum user profile. In the online stage, given a recommendation query $q$: (1) images relevant to $q$ are retrieved via CLIP similarity; (2) the retrieved images are arranged into a $d \times d$ grid and a query-conditioned user profile is generated; (3) the profile is matched against candidates for scoring and ranking. The overall framework is modular, with each component replaceable by stronger models independently.

### Key Designs
1. **Spectrum User Profile**:

    - Function: Represents each image as a triplet (raw image, caption, aspect words), forming a spectrum from information richness to semantic clarity.
    - Mechanism: Image vectors are encoded with CLIP ViT-L/14@336px; concise captions of ≤30 words are generated with LLaVA-v1.6-8B; key aspect words (e.g., *dome*, *balcony*) are then extracted.
    - Design Motivation: Raw images are information-rich but noisy, while pure text descriptions incur significant information loss. The triplet strikes a balance between the two; ablation studies confirm that all three representations contribute.
    - All modules are plug-and-play and can be replaced with stronger encoders or generative models.

2. **Image Gridification**:

    - Function: Arranges $w$ retrieved images into a $d \times d$ grid ($d^2 = w$) and feeds it as a single image into the multimodal model.
    - Mechanism: $w = 64$ images are arranged into an $8 \times 8$ grid, with each sub-image labeled (1–64) to associate it with corresponding captions and aspect words; slots are filled with black backgrounds when fewer than 64 images are available.
    - Design Motivation: Directly inputting 64 images would far exceed the MLLM context window limit (e.g., PaliGemma generates 4,096 tokens for a 896×896 image). Gridification converts the multi-image problem into a single-image understanding task.
    - Multi-image caption pre-training on the DOCCI dataset ensures the model accurately identifies each sub-image within the grid.

3. **History Retrieval**:

    - Function: Retrieves images from the user's visual history most relevant to the current query.
    - Mechanism: For each recommendation category $c$, $n = 10\text{K}$ candidate items are randomly sampled and their CLIP visual encodings are averaged to obtain a category vector $v_c$; the top-$w$ images from the user's history are selected by cosine similarity to $v_c$.
    - Design Motivation: Users' visual histories are diverse and noisy, with many images irrelevant to the current query. Retrieval focuses the model on relevant visual signals; ablation studies show this is the highest-contributing component.

4. **Iterative Aspect Word Refinement**:

    - Function: Improves the alignment between aspect words and user preferences through multiple rounds of iteration.
    - Mechanism: Initial aspect words are generated by LLaVA-v1.6 ($\mathcal{W}^0$); in each round, Llama-3.1-70B filters useful aspect words ($\mathcal{W}^j$) using ground truth; convergence is reached in approximately 4 rounds, yielding the training target $\mathcal{W}$.
    - Design Motivation: Zero-shot generated aspect words are of inconsistent quality and may include irrelevant terms (e.g., "blue sky"). Iterative refinement aligns aspect words more closely with the requirements of the recommendation task.

### Loss & Training
- **Multi-Image Caption Pre-training**: LoRA fine-tuning on the DOCCI dataset (15,000+ images) teaches the model to describe each sub-image in the grid separately, laying the foundation for subsequent gridified recommendation.
- **Joint Training**: Aspect word generation loss (CE) and candidate matching loss (BCE) are optimized jointly: $\mathcal{L}_\text{joint} = \mathcal{L}_\text{asp} + \lambda \cdot \mathcal{L}_\text{pred}$, where $\lambda = 2$.
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning is applied on pre-trained backbones (MiniCPM-V2.5 / PaliGemma).

## Key Experimental Results

### Main Results (Google Review-V / Yelp-V)

| Method | Modality | Size | GR-V Hit@1 | GR-V Hit@3 | GR-V Hit@10 | GR-V MRR | Yelp Hit@1 | Yelp Hit@3 | Yelp Hit@10 | Yelp MRR |
|--------|----------|------|------------|------------|-------------|----------|------------|------------|-------------|----------|
| Random | - | - | 7.6 | 21.0 | 55.0 | 21.2 | 13.0 | 33.6 | 72.7 | 30.0 |
| UniMP | T+I | 3B | 13.8 | 34.1 | 73.0 | 30.5 | 22.4 | 48.5 | 85.0 | 38.3 |
| GPT-4o | T+I | - | 17.1 | 37.3 | 80.1 | 34.3 | 26.1 | 54.5 | 90.5 | 41.7 |
| **VisualLens (8B)** | T+I | 8B | **18.5** | **38.9** | **82.3** | **35.4** | **28.3** | **59.1** | **91.0** | **44.9** |
| Human | - | - | 22.0 | 45.0 | - | - | 36.0 | 66.0 | - | - |

### Ablation Study (PaliGemma 3B backbone)

| Configuration | GR-V Hit@3 | GR-V MRR | Yelp Hit@3 | Yelp MRR | Notes |
|---------------|------------|----------|------------|----------|-------|
| Full VisualLens | 36.3 | 33.5 | 58.8 | 44.3 | All components |
| w/o Joint Training | 35.8 | 33.0 | 57.9 | 43.3 | Hit@3 −0.5/−0.9 |
| w/o Iterative Refinement + Joint Training | 35.2 | 32.5 | 57.5 | 42.9 | Hit@3 −1.1/−1.3 |
| w/o Captions | 34.7 | 31.9 | 55.3 | 41.2 | Captions contribute significantly |
| w/o Aspect Words | 33.9 | 31.2 | 53.9 | 40.4 | Aspect words contribute most |
| Images Only | 32.5 | 29.6 | 48.2 | 38.8 | Text augmentation is critical |
| w/o Retrieval (all images) | 27.9 | 25.9 | 45.7 | 36.8 | Retrieval contributes most (−7%/−12%) |

### Key Findings
- VisualLens (8B) surpasses GPT-4o by 1.6% (GR-V) and 4.6% (Yelp-V) in Hit@3, closing approximately 75% of the gap with human annotation performance.
- History retrieval is the most important component; removing it causes a 7–12% drop in Hit@3. Aspect words contribute more than captions.
- MRR saturates when the number of user history images reaches ~100 and stabilizes when the number of candidates exceeds 50.
- Cross-history-length and cross-category transferability are strong; long-history test MRR is highest (GR-V: 38.0 vs. 35.4).
- Ambiguous categories (e.g., "area", "station") yield the worst recommendation performance; generic categories (e.g., "museum", "hotel") perform best.
- Transfer learning effects exist between adjacent categories, e.g., "deli" and "takeout" benefit from their similarity to the "restaurant" category.

## Highlights & Insights
- **Novel Problem Formulation**: This is the first work to systematically study the problem of "task-agnostic visual history → cross-domain personalization," introducing two dedicated benchmarks (GR-V: 15.69M training samples; Yelp-V: 4.12M) and filling an evaluation gap in this direction.
- **Elegant Spectrum Representation Design**: The (image, caption, aspect words) triplet forms a spectrum between information quantity and semantic clarity. Image gridification elegantly addresses the MLLM context window constraint, enabling an 8B model to process 64 images simultaneously.

## Limitations & Future Work
- Individual modules (encoders, caption generation, etc.) do not use the strongest available models; the framework prioritizes modularity over peak performance.
- Only static images are processed; richer modalities such as video and audio are not covered.
- Evaluation is limited to QA-format recommendation tasks and does not cover ranked list generation or conversational recommendation scenarios.
- Privacy concerns: inferring user preferences from everyday photos requires rigorous privacy protection mechanisms for real-world deployment.

## Related Work & Insights
- **vs. UniMP (Wei et al., 2024a)**: UniMP is the current state-of-the-art multimodal recommender but relies on item-level interaction histories. VisualLens uses task-agnostic visual history and surpasses UniMP by ~5–10% Hit@3 at the same parameter scale.
- **vs. GPT-4o**: Direct prompting of GPT-4o yields strong performance on visual recommendation, yet VisualLens (8B) still surpasses it by 1.6–4.6% Hit@3 through specialized spectrum profiling and joint training.
- **vs. ReLLa (Lin et al., 2024)**: ReLLa enhances text-based recommendation via retrieval augmentation. VisualLens extends the retrieval paradigm to the visual modality, where the retrieval module proves to be the largest contributor.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First framework for recommendation driven by task-agnostic visual history; both the problem formulation and datasets are entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two large-scale datasets, multiple baselines, detailed ablation studies, and transferability analysis.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear with rich figures and tables, though some notation could be simplified.
- Value: ⭐⭐⭐⭐ Opens a new direction for visual history-based personalization, though privacy challenges in real-world deployment require further attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](mmpb_its_time_for_multi-modal_personalization.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](../../ACL2026/recommender/learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[ACL 2026\] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](../../ACL2026/recommender/from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)
- [\[NeurIPS 2025\] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)

</div>

<!-- RELATED:END -->
