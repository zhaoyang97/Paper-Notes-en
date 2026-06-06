---
title: >-
  [Paper Note] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings
description: >-
  [ACL 2026][Dialogue Systems][Dialogue Sentence Embeddings] The TaDSE framework is proposed to leverage existing template information in dialogues as auxiliary anchors. Through three stages—template-aware data augmentatio…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Dialogue Sentence Embeddings"
  - "Contrastive Learning"
  - "Template Enhancement"
  - "Intent Classification"
  - "Unsupervised Representation Learning"
date: 2026-05-08
content_hash: 9ddcc3eb9473b9f7
---

# Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings

**Conference**: ACL 2026  
**arXiv**: [2305.14299](https://arxiv.org/abs/2305.14299)  
**Code**: [GitHub](https://github.com/minsik-ai/Template-Contrastive-Embedding)  
**Area**: Dialogue Systems  
**Keywords**: Dialogue Sentence Embeddings, Contrastive Learning, Template Enhancement, Intent Classification, Unsupervised Representation Learning

## TL;DR
The TaDSE framework is proposed to leverage existing template information in dialogues as auxiliary anchors. Through three stages—template-aware data augmentation, paired contrastive training, and semantic compression inference—it significantly improves the quality of task-oriented dialogue sentence embeddings in unsupervised settings, outperforming Prev. SOTA and even commercial supervised embedding models across five benchmarks.

## Background & Motivation

**Background**: Learning high-quality dialogue sentence embeddings is crucial for downstream tasks such as intent classification and slot filling in low-annotation scenarios. Existing unsupervised sentence embedding methods (e.g., SimCSE, PromptBERT) perform well on general text but show significant degradation when transferred to the dialogue domain due to the specific semantic relational structures between dialogue utterances.

**Limitations of Prior Work**: Obtaining utterance-level semantic relationship annotations in the dialogue domain is difficult, whereas token-level annotations (e.g., entities, slots, templates) are relatively easier to acquire. However, existing sentence embedding frameworks are self-supervised at the sentence level and cannot utilize this rich token-level auxiliary knowledge. General data augmentation methods (e.g., back-translation, rule-based substitution) tend to introduce semantic shift or require additional model support.

**Key Challenge**: Dialogues contain a large amount of structured template information (where one template corresponds to multiple utterances with different expressions), but this utterance-template pairing has never been utilized for embedding learning. Existing methods only perform contrastive learning within the utterance space, ignoring templates as semantic anchors to constrain the structure of the embedding space.

**Goal**: Design an unsupervised framework that leverages template information to enhance dialogue sentence embeddings, resulting in more compact clusters for semantically similar utterances and clearer decision boundaries.

**Key Insight**: The authors observe that templates serve as the "semantic skeleton" of utterances—utterances under the same template share a core semantic structure and differ only in slot values. Introducing templates as auxiliary representations into contrastive learning helps the model learn to distinguish correct utterance-template pairs, thereby improving the embedding space.

**Core Idea**: Expand the diversity of utterance-template pairs through template-aware data augmentation, followed by joint training using a triple contrastive loss (template loss + utterance loss + pairing loss), and finally optimize embeddings by integrating template representations during inference via semantic compression.

## Method

### Overall Architecture
TaDSE consists of three stages: (1) Template Data Augmentation—generating a large number of natural synthetic utterances through permutations of existing slots and templates; (2) Paired Contrastive Training—simultaneously learning template, utterance, and utterance-template pair representations optimized by three joint contrastive losses; (3) Semantic Compression Inference—integrating template representations into utterance representations proportionally during inference to enhance semantic discriminability. The input consists of dialogue utterances and their corresponding templates, and the output is the optimized sentence embedding vector.

### Key Designs

1.  **Template Data Augmentation**:
    - **Function**: Expand the diversity of utterance-template pairs in the training data.
    - **Mechanism**: Extracts slots (e.g., CITY, DEVICE) and their high-frequency values from the dataset to construct a Slot Book, then performs permutations of slot values (top-k frequency values) for each template to generate natural synthetic utterances. For example, "Book a flight to {CITY}" can generate "Book a flight to Paris/Tokyo/London". A total of 834,000 augmented utterances were generated across 5 datasets, averaging 16 utterances per template.
    - **Design Motivation**: The effectiveness of paired contrastive learning depends on having diverse utterance samples for each template. The original datasets have low utterance/template ratios; augmentation drastically increases pairing diversity, enabling the model to better learn discriminative capabilities.

2.  **Triple Contrastive Loss**:
    - **Function**: Jointly learn representations for the template space, utterance space, and utterance-template pair space.
    - **Mechanism**: (a) The template loss $L^t$ uses dropout noise to generate positive pairs, pulling two encodings of the same template together; (b) The utterance loss $L^u$ learns utterance representations similar to the SimCSE framework; (c) The paired loss $L^{pair}$ treats correct utterance-template pairs as positive samples and other utterances as negative samples, teaching the model to distinguish semantically matching pairs. The final loss is $L^{train} = L^t + \lambda^u L^u + \lambda^{pair} L^{pair}$.
    - **Design Motivation**: Utterance-only contrastive learning cannot leverage the structural information of templates; the introduction of the paired loss allows the model to use templates as semantic anchors, pulling utterances of the same template closer and pushing those of different templates further apart, forming clearer semantic clusters.

3.  **Semantic Compression Inference**:
    - **Function**: Integrate template information during the inference stage to further optimize embeddings.
    - **Mechanism**: The final representation is $repr_i = \lambda^{comp} t_i + (1 - \lambda^{comp}) u_i$, where $\lambda^{comp}$ is tuned on the validation set. Adding the template component enhances specific semantic dimensions, allowing utterances that are similar in appearance but different in semantics to be distinguished.
    - **Design Motivation**: Templates are the semantic essence of utterances; moderate integration can enhance discriminative power near decision boundaries. Furthermore, the optimal value of $\lambda^{comp}$ serves as an analytical tool to measure the quality of template-utterance semantic alignment.

### Loss & Training
The triple contrastive losses are all based on the InfoNCE framework, using in-batch negative sampling. Each loss has independent temperature hyperparameters $\tau_t$, $\tau_u$, and $\tau_{pair}$. Transfer learning is performed on a BERT-base model based on SimCSE, and intent classification is evaluated using kNN on the training set. Optionally, a trainable MLP layer $W_A$ is added to the template branch to adjust template representation dimensions.

## Key Experimental Results

### Main Results

| Model | SNIPS | ATIS | MASSIVE | HWU64 | CLINC150 | Average |
|-------|-------|------|---------|-------|----------|------|
| BERT | 80.00 | 78.05 | 41.86 | 50.84 | 33.35 | 56.82 |
| SimCSE | 91.71 | 85.67 | 76.77 | 81.08 | 71.00 | 81.25 |
| DSE | 95.86 | 87.01 | 76.77 | 79.28 | 70.16 | 81.82 |
| **TaDSE** | **97.00** | **89.70** | **78.18** | **82.77** | 70.56 | **83.64** |
| TaDSE w/ MLP | 96.29 | 89.14 | **79.15** | 82.29 | **72.49** | **83.87** |

Comparison with commercial supervised embeddings (TaDSE is unsupervised with only 110M parameters):

| Model | SNIPS | ATIS | Average |
|-------|-------|------|------|
| OpenAI-large | 98.57 | 84.77 | 91.67 |
| Gemini-001 | 98.29 | 86.00 | 92.15 |
| **TaDSE** | 97.00 | **89.70** | **93.35** |

### Ablation Study

| Configuration | SNIPS | ATIS | MASSIVE | CLINC150 |
|------|-------|------|---------|----------|
| w/o Augmentation (SimCSE) | 91.71 | 85.67 | 77.00 | 71.05 |
| + Augmentation | 93.29 | 86.00 | 77.37 | 70.98 |
| + $L^t$ | 95.29 | 88.47 | 78.58 | 71.53 |
| + $L^t$ + $L^{pair}$ | 96.14 | 89.59 | 79.39 | 72.98 |
| + $L^{t'}$ (MLP) + $L^{pair}$ | **97.00** | 88.69 | **79.83** | **73.45** |

### Key Findings
- The paired loss $L^{pair}$ contributes the most; its individual introduction on SNIPS improves results from 93.29 to 96.14 (+2.85%), proving the effectiveness of utterance-template pairing.
- The template loss $L^t$ itself significantly improves performance (+2.0% to +2.5%), indicating that the salient semantic information in templates makes an independent contribution to embedding learning.
- Augmentation stability varies by dataset: SNIPS/ATIS improve continuously with more augmentation (augmentation-stable), while MASSIVE/CLINC150 may decline under high-order augmentation.
- Semantic compression consistently yields positive gains on augmentation-stable datasets (SNIPS +0.29%, ATIS +0.44%), validating the quality of template-utterance semantic alignment.

## Highlights & Insights
- The idea of using templates as semantic anchors is ingenious—transforming existing token-level annotations in dialogues into auxiliary signals for sentence-level contrastive learning, achieving "free" supervised signal injection. This approach can be generalized to any domain with structured templates/schemas.
- Semantic compression is not only an inference enhancement method but also an analytical tool—the optimal value of $\lambda^{comp}$ reflects the quality of template-utterance semantic alignment in the embedding space, providing an interpretable window into the representation space.
- The unsupervised 110M small model outperforms supervised commercial embeddings from OpenAI and Google in average accuracy, demonstrating the significant potential of domain-specific methods.

## Limitations & Future Work
- Dependency on template and slot annotations; for dialogue datasets without template labels, additional automatic template extraction steps are required (the NER scheme for CLINC150 in the paper showed limited effectiveness).
- Evaluation was only conducted on intent classification tasks; effectiveness on other downstream tasks (e.g., Dialogue State Tracking, Response Selection) has not been verified.
- The effect of semantic compression is uncertain on non-augmentation-stable datasets, indicating some sensitivity to data quality.
- Future work could consider integrating LLMs to automatically generate high-quality templates to remove dependency on manual annotations.

## Related Work & Insights
- **vs SimCSE**: SimCSE uses only dropout noise for positive samples, whereas TaDSE introduces templates as semantic anchors for paired contrastive learning, better utilizing structural information in the dialogue domain.
- **vs DSE**: DSE uses consecutive utterances as positive pairs for contrastive learning, but it remains at the utterance-utterance level; TaDSE introduces cross-granularity utterance-template pairs, providing more precise semantic correlation signals.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of templates as contrastive learning anchors and the semantic compression test are innovative, though the base framework is still built on SimCSE.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across five datasets with thorough ablation and comparison with commercial models, though verification on more downstream tasks is missing.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured, the method derivation is complete, and tables/figures are informative.
- Value: ⭐⭐⭐⭐ Provides an effective paradigm for dialogue embedding learning by utilizing template information, which can be extended to other domains with structured annotations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ACL 2026\] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[ACL 2026\] Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)

</div>

<!-- RELATED:END -->
