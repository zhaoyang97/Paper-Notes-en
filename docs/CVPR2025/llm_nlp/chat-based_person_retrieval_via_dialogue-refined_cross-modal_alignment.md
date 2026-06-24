---
title: >-
  [Paper Note] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment
description: >-
  [CVPR 2025][LLM (Other)][Person Retrieval] This paper proposes a new paradigm of Chat-based Person Retrieval (ChatPR), builds the first dialogue-image paired dataset ChatPedes, and designs the DiaNA framework to achieve fine-grained cross-modal alignment between dialogues and images via an adaptive attribute refiner, significantly outperforming traditional single-sentence text retrieval methods.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Person Retrieval"
  - "Cross-Modal Alignment"
  - "Dialogue Interaction"
  - "Attribute Refining"
  - "Data Augmentation"
date: 2026-05-08
content_hash: a3f029975f082e6a
---

# Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Person Retrieval, Cross-Modal Alignment, Dialogue Interaction, Attribute Refining, Data Augmentation

## TL;DR

This paper proposes a new paradigm of Chat-based Person Retrieval (ChatPR), builds the first dialogue-image paired dataset ChatPedes, and designs the DiaNA framework to achieve fine-grained cross-modal alignment between dialogues and images via an adaptive attribute refiner, significantly outperforming traditional single-sentence text retrieval methods.

## Background & Motivation

**Background**: Traditional Text-based Person Retrieval (TPR) relies on the user providing a descriptive text query, based on which the system retrieves the target person from an image gallery. Such methods assume that users can fully express their search intent in a single input.

**Limitations of Prior Work**: In real-world scenarios, it is difficult for users to provide a complete and accurate description at once. For instance, users might not remember the exact color of the target person's clothing or might describe certain attributes vaguely. The incompleteness of a single query severely limits retrieval accuracy. Furthermore, the semantic granularity of user descriptions varies, with some being too vague ("a person in dark clothes") to effectively distinguish different candidate persons.

**Key Challenge**: The information capacity of a single text query is limited and cannot fully capture the user's search intent, yet existing TPR methods rely on the unrealistic assumption of a "one-off perfect description."

**Goal**: (1) How to construct a chat-based person retrieval system that allows users to progressively refine queries through multi-turn interactions? (2) Given the lack of dialogue-image paired datasets, how to efficiently build one? (3) How to establish effective cross-modal alignment between dialogue and image modalities?

**Key Insight**: The authors observe that when humans search for someone in real life, they typically narrow down the scope through multi-turn dialogues ("What color are they wearing? Are they wearing a hat? What does their backpack look like?"). Therefore, person retrieval is upgraded from a single query to a multi-turn dialogue mode, and Large Language Models are leveraged to automatically generate dialogue data to address the data scarcity.

**Core Idea**: To replace single-sentence queries with multi-turn dialogues for person retrieval, and perform fine-grained cross-modal alignment by bottlenecking dialogue and visual information through adaptive attribute refiners.

## Method

### Overall Architecture

The DiaNA (Dialogue-refined Cross-modal Alignment) framework takes multi-turn dialogues as query inputs to retrieve matching person images from an image gallery. The overall pipeline is: input multi-turn dialogue text $\rightarrow$ extract multi-turn semantic features via dialogue encoder $\rightarrow$ extract key attributes via adaptive dialogue attribute refiner $\rightarrow$ align with attributes extracted from the visual side through adaptive visual attribute refiner $\rightarrow$ perform fine-grained cross-modal matching $\rightarrow$ output retrieval results.

### Key Designs

1. **ChatPedes Dataset Construction**:

    - **Function**: To construct the first large-scale dataset for chat-based person retrieval.
    - **Mechanism**: Utilizing Large Language Models (LLMs) to automatically generate QA dialogues. Given the attribute annotations of person images, the LLM generates a sequence of questions regarding the person's appearance and simulates a user answering based on ground truth attributes. Each dialogue consists of multiple rounds of questions and answers, progressively revealing detailed features (e.g., clothing colors, accessories, physical characteristics) of the person.
    - **Design Motivation**: Manually labeling dialogue-image pairs is extremely expensive and difficult to scale. Utilizing LLMs can efficiently automate the question generation and answer simulation process.

2. **Adaptive Attribute Refiner**:

    - **Function**: To bottleneck dialogue and image information into key attribute representations, facilitating fine-grained alignment.
    - **Mechanism**: Two parallel adaptive attribute refiners are designed—the dialogue-side refiner and the visual-side refiner. The dialogue side extracts and fuses the attribute information mentioned in each round (e.g., "red top," "black pants," "carrying a backpack") from multi-turn dialogues, compressing long dialogues into structured attribute bottleneck representations. The visual side performs a similar operation on image features, extracting attribute-related visual features. The bottleneck representations from both sides are matched at a fine-grained level in a unified attribute space.
    - **Design Motivation**: Raw feature dimensions and semantic spaces of dialogues and images differ vastly, leading to poor direct alignment. Mapping both modalities to a shared attribute space via attribute bottlenecks reduces alignment difficulty and captures fine-grained attribute matching relationships.

3. **Random Round Retaining Data Augmentation**:

    - **Function**: To improve model generalization across dialogues of various lengths.
    - **Mechanism**: During training, a random number of rounds within a dialogue are retained (rather than always using the full dialogue) to simulate real-world scenarios where users might initiate retrieval after only a few turns of interaction. Consequently, the model can not only handle complete dialogues but also output valid retrieval results in the early stages of a dialogue.
    - **Design Motivation**: In practice, users may request retrieval results after any arbitrary round, requiring the model to remain robust across different dialogue lengths.

### Loss & Training

A contrastive learning framework is adopted for training, pulling matching dialogue-image pairs closer in the feature space and pushing non-matching pairs apart. Combining global contrastive losses with fine-grained attribute alignment losses ensures that the model captures both global semantic similarity and local attribute consistency. The Random Round Retaining strategy serves as a training-time data augmentation technique, dynamically changing the number of retained dialogue turns in each epoch.

## Key Experimental Results

### Main Results

| Method | Data Type | Rank-1 | Rank-5 | Rank-10 | mAP |
|------|----------|--------|--------|---------|-----|
| Traditional TPR (Single) | Single-sentence text | Baseline | Baseline | Baseline | Baseline |
| DiaNA (1 Round) | 1-round dialogue | Outperforms TPR | - | - | - |
| DiaNA (3 Rounds) | 3-round dialogue | Significantly outperforms TPR | - | - | - |
| DiaNA (Full) | Full dialogue | Best | Best | Best | Best |

The paper demonstrates that DiaNA significantly outperforms existing TPR methods across all dialogue rounds, with retrieval accuracy steadily increasing as dialogue rounds accumulate.

### Ablation Study

| Component | Change in Rank-1 |
|------|------------|
| w/o Dialogue Attribute Refiner | Significant drop |
| w/o Visual Attribute Refiner | Drop |
| w/o Random Round Retaining | Obvious drop in short-dialogue scenarios |
| Full DiaNA | Best |

### Key Findings

- The multi-turn dialogue retrieval paradigm significantly outperforms traditional single-sentence queries, achieving clear improvements even with only 1-2 dialogue rounds.
- The adaptive attribute refiner is a crucial component for cross-modal alignment; bottlenecking information at the attribute level significantly improves alignment efficiency.
- The Random Round Retaining strategy is vital for model robustness, particularly in scenarios with incomplete dialogues.

## Highlights & Insights

- **Paradigm Innovation**: Upgrading person retrieval from a static, single-sentence query to a dynamic dialogue interaction mode, which better aligns with actual human search behavior.
- **Clever Data Construction**: Leveraging LLMs to automatically generate dialogue data, addressing the cold-start problem of labeled data scarcity for the new task.
- **Attribute Bottleneck Design**: Mapping both dialogue and image representations to an attribute space to achieve cross-modal alignment is more efficient and interpretable than directly aligning them in the raw feature space.
- **Progressive Refinement**: As dialogue turns increase, retrieval results become progressively accurate, aligned with the intuition of stepwise information accumulation.

## Limitations & Future Work

- The ChatPedes dataset is automatically generated by LLMs; the naturalness and diversity of the dialogues may not match real human conversations.
- The current framework assumes that user answers are accurate, ignoring potential user memory errors or vague descriptions.
- Active questioning strategies remain unexplored—the system should learn to propose the most discriminative questions at different stages to accelerate retrieval.
- Reinforcement learning could be introduced to optimize the dialogue strategy, enabling the system to achieve maximum retrieval accuracy with the minimum number of rounds.
- The approach is only verified on the person retrieval task and could be extended to more general chat-based image retrieval scenarios.

## Related Work & Insights

- **TPR Methods such as IRRA and LGUR**: The strongest existing single-sentence text-based person retrieval methods; experiments in this paper demonstrate that the dialogue paradigm significantly outperforms them.
- **Visual Dialog**: Although it involves interaction between images and dialogues, the goal is to answer questions about an image rather than retrieving images using dialogues.
- **Insights**: The chat-based retrieval paradigm can be generalized to other fine-grained visual retrieval tasks, such as product search, vehicle retrieval, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ (Proposes a brand-new ChatPR paradigm with an innovative problem definition)
- Experimental Thoroughness: ⭐⭐⭐ (Reasonable experimental setup, but the dataset is relatively single)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (The new paradigm has broad potential for generalization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](../../ACL2025/llm_nlp/cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](../../ACL2025/llm_nlp/towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](../../ACL2025/llm_nlp/beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](../../ACL2025/llm_nlp/from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)
- [\[CVPR 2025\] Imagine and Seek: Improving Composed Image Retrieval with an Imagined Proxy](imagine_and_seek_improving_composed_image_retrieval_with_an_imagined_proxy.md)

</div>

<!-- RELATED:END -->
