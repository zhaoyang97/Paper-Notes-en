---
title: >-
  [Paper Note] On Entity Identification in Language Models
description: >-
  [ACL 2025][LLM (Other)][Entity Identification] This paper proposes a clustering-based evaluation framework (Purity/Inverse Purity) to analyze the entity identification capabilities in LLM internal representations. It finds that entity information becomes linearly separable ($F_1 \approx 0.9$) within a 20-dimensional subspace in early layers (~ normalized position 0.2), and different LLMs converge to structurally isomorphic entity encodings. This provides systematic evidence f…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Entity Identification"
  - "Internal Representations"
  - "Linear Separability"
  - "Clustering Evaluation"
  - "Cross-Model Isomorphism"
date: 2026-05-08
content_hash: 6aac1174b4bde625
---

# On Entity Identification in Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.02701](https://arxiv.org/abs/2506.02701)  
**Code**: [https://github.com/masaki-sakata/entity-identification](https://github.com/masaki-sakata/entity-identification)  
**Area**: LLM/NLP  
**Keywords**: Entity Identification, Internal Representations, Linear Separability, Clustering Evaluation, Cross-Model Isomorphism

## TL;DR
This paper proposes a clustering-based evaluation framework (Purity/Inverse Purity) to analyze the entity identification capabilities in LLM internal representations. It finds that entity information becomes linearly separable ($F_1 \approx 0.9$) within a 20-dimensional subspace in early layers (~ normalized position 0.2), and different LLMs converge to structurally isomorphic entity encodings. This provides systematic evidence for the "emergence of discrete knowledge structures in LLMs from raw text training."

## Background & Motivation

**Background**: Transformer-based language models have been shown to recall factual knowledge (e.g., "Barack Obama was born in __" → "Hawaii"), and previous studies (Meng et al. 2022, Geva et al. 2023) analyzed the information flow of factual recall. However, these studies assumed that the entity mentions in the input are explicit and unambiguous.

**Limitations of Prior Work**:
   - **Mention ambiguity**: The same surface form can refer to different entities (e.g., "Obama" might refer to Barack Obama or Michelle Obama).
   - **Mention variability**: The same entity can have multiple surface forms (e.g., "Māori All Blacks" and "New Zealand Maori Rugby" refer to the same team).
   - Prior work did not analyze whether LLM internal representations truly distinguish different entities when facing these two challenges.

**Key Challenge**: While LLMs can perform entity-related tasks (such as NER and relation extraction), it remains unclear whether they truly construct internal distinctions between entity identities—do they "know," or do they just "get it right by chance"?

**Goal**: (1) To what extent do LLM internal representations distinguish different mentions of the same entity? (2) In which layers and in what dimensional subspaces is entity information encoded? (3) Are the entity representation structures across different LLMs similar?

**Key Insight**: The entity identification problem is formulated as a clustering quality evaluation. If an LLM "knows" that two mentions refer to the same entity, their internal representations should cluster together in the embedding space.

**Core Idea**: Using Purity/Inverse Purity (IP) clustering metrics to quantify the entity differentiation of LLM internal representations, revealing highly linearly separable entity encodings in low-dimensional subspaces.

## Method

### Overall Architecture
The input consists of sentences containing entity mentions. Hidden states from each layer of the LLMs are extracted as entity representations, and clustering metrics are then used to measure whether representations of the same entity cluster together and those of different entities are separated. A layer-wise analysis is conducted on five autoregressive models (GPT-2, LLaMA-2 7B/13B, LLaMA-3 8B, and Mistral 7B).

### Key Designs

1. **Clustering Evaluation Framework (Purity + Inverse Purity)**:

    - **Function**: Computes F1 scores using entity classes as ground-truth labels and nearest-neighbor clustering in the representation space as predicted partitions.
    - **Mechanism**: Purity measures whether a cluster contains only the same entity (similar to precision), while IP measures whether all mentions of the same entity are grouped in the same cluster (similar to recall). An F1 score of 1.0 indicates perfect distinction.
    - **Design Motivation**: In contrast to linear probes (which require training a classifier), clustering is unsupervised and directly evaluates the geometric properties of the representations, eliminating any bias from optimizers or initialization.

2. **Dimensionality Impact Analysis and LDA Reduction**:

    - **Function**: Evaluates entity identification rates across different dimensions to determine the optimal dimensionality for analysis.
    - **Mechanism**: After reducing the LLM representation dimensionality to 20 using Linear Discriminant Analysis (LDA), the F1 score decreases by only about 3% (from 0.93 to 0.90 for LLaMA-2 7B), whereas random embeddings show scores near zero in low dimensions but falsely high scores in high dimensions. This demonstrates that LLM entity information is indeed encoded in a low-dimensional subspace rather than being an artifact of the curse of dimensionality.
    - **Design Motivation**: Distance-based metrics can be affected by the curse of dimensionality, so it is necessary to prove that the findings are not statistical artifacts of high-dimensional spaces.

3. **Defining Difficulty (Ambiguity & Variability)**:

    - **Function**: Quantifies mention ambiguity using entropy $H = -\sum p_i \log p_i$ and mention variability using normalized Levenshtein distance.
    - **Mechanism**: Mentions with high ambiguity (e.g., "Georgia" referring to either the country or the US state) are harder to distinguish, as are entities with high variability (where different surface forms for the same entity differ significantly).
    - **Design Motivation**: This provides a fine-grained difficulty gradient to evaluate model performance.

### Experimental Data
- The ZELDA-TRAIN dataset (for entity disambiguation tasks) is utilized, filtered to retain entities occurring at least 5 times.
- For autoregressive models, a "duplicate input" strategy is used: inputting the sentence twice and using the embeddings of the second mention occurrence as the entity representation (allowing the model to leverage the full context).

## Key Experimental Results

### Main Results: Entity Identification Performance

| Dimension of Analysis | Metric | Result |
|---------|------|------|
| Mention Ambiguity | AUC (20D) | 0.8–0.9 (LLMs) vs ~0 (Random) |
| Mention Variability | AUC (20D) | 0.66–0.8 (LLMs) vs ~0 (Random/FastText) |
| Linear Probe | F1 (20D) | ~0.9 |
| Entity Information Peak | Normalized Layer Position | ~0.2 (e.g., Layers 6-8 in LLaMA-2 7B) |

### Layer-wise Analysis (LLaMA-2 7B)

| Layer | Ambiguity AUC | Variability AUC | Description |
|------|---------|---------|------|
| Layer 0 | 0.38 | 0.30 | Token embeddings cannot distinguish |
| Layer 8 | **0.87** | **0.81** | Peak—reaches optimal distinction after contextualization |
| Layer 16 | Decreased | Decreased | Later layers start serving next-token prediction |
| Layer 32 | Decreased further | Decreased further | Entity information is "consumed" |

### Cross-Model Isomorphism Analysis (RSA)

| Model Pair | Spearman Correlation |
|--------|-----------------|
| LLaMA-2 vs LLaMA-3 | High |
| LLaMA-2 vs Mistral | High |
| GPT-2 vs Others | Relatively low |

### Impact of Entity Identification on Downstream Tasks

| Task | Low Variability Entities | High Variability Entities |
|------|----------|----------|
| Word Prediction Consistency | **71%** | 39% |
| Entity Disambiguation Accuracy | Increases with F1 (Pearson $r \approx 0.18$, $\beta = 0.31$) | Decreases with F1 |

### Key Findings
- **Entity information peaks in early layers and then decays**: Peak identification quality is reached around normalized position 0.2 (e.g., layers 6–8 in a 32-layer model), after which differentiation decreases. Later layers likely cater to other functionalities (e.g., syntax, next-token prediction).
- **Linearly separable in a 20-dimensional subspace**: Entity information is compactly compressed within extremely low-dimensional spaces, supporting the linear representation hypothesis.
- **Convergence among LLMs**: LLaMA-2/3 and Mistral form structurally isomorphic entity representation spaces, bolstering the Platonic Representation Hypothesis (where different models converge toward similar encodings of world knowledge).
- **Entity identification quality directly impacts downstream performance**: High-distinctiveness entities (low variability) show a word prediction consistency of 71%, compared to only 39% for low-distinctiveness ones.

## Highlights & Insights
- **Discovery of "entity encoding in early layers"**: Provides a fresh perspective on the LLM information processing pipeline—entity identities are differentiated in lower layers, and higher layers reuse this information for more complex tasks. This aligns with findings from Meng et al. (2022) using activation patching that "entity information is encoded in early layers," but here it is verified via a completely different approach (geometric analysis instead of causal intervention).
- **Low-dimensional linear separability combined with cross-model isomorphism**: The convergence of these two findings yields a powerful conclusion—entity coding is not a random feature unique to each model, but a convergent, structured organization of knowledge.
- **Advantages of unsupervised clustering methods**: Compared to linear probes, Purity/IP does not require training classifiers, evading optimizer artifacts and reflecting the geometric properties of the representation space more purely.
- **Practical implications**: Pinpointing where entity information resides (early layers, 20D subspaces) can guide applications such as knowledge editing, model pruning, and retrieval-augmented generation.

## Limitations & Future Work
- **Limited to English Wikipedia entities**: Is entity encoding language-independent? Cross-lingual analysis is a vital direction for future research.
- **Descriptive rather than causal**: The study uncovers characteristics of entity encoding but does not explain how or why these encodings form (e.g., which training dynamics lead to early-layer entity differentiation).
- **Is perfect distinction always desirable?**: The paper acknowledges that "neural collapse" (where intra-class representations collapse to a single point) can be detrimental in some scenarios—hierarchical relationships (e.g., encoding "painter" as a hypernym) may require some degree of representation overlap.
- **Sufficiency of data scale**: Although approximately 160k sentences were used, this cannot guarantee coverage of all possible entity ambiguities.

## Related Work & Insights
- **vs. Gurnee & Tegmark (2024)**: They found that geographical/temporal features achieve the highest probe scores in early layers; this work extends that observation to a broader set of entity types.
- **vs. Abdou et al. (2021)**: They utilized RSA to analyze the alignment between color word representations and human perception; this work applies isomorphic analysis to the discrete structures of entity identities.
- **vs. Huh et al. (2024) Platonic Representation Hypothesis**: The structural isomorphism in entity representations across models provides fresh evidence for this hypothesis from the dimension of entity identity.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a clustering perspective, layer-wise analysis, and cross-model isomorphism is highly systematic.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 models + multi-dimensional analysis + mitigation of the curse of dimensionality.
- Writing Quality: ⭐⭐⭐⭐ Systematic analysis and clearly presented findings.
- Value: ⭐⭐⭐⭐ Significant contribution to understanding the mechanism of entity encoding in LLMs, providing support for linear representation and representation convergence hypotheses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)
- [\[ACL 2025\] Entity Framing and Role Portrayal in the News](entity_framing_and_role_portrayal_in_the_news.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)
- [\[ACL 2025\] COSMIC: Generalized Refusal Direction Identification in LLM Activations](cosmic_generalized_refusal_direction_identification_in_llm_activations.md)
- [\[ACL 2025\] EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States](enigmatom_improve_llms_theory-of-mind_reasoning_capabilities_with_neural_knowled.md)

</div>

<!-- RELATED:END -->
