---
title: >-
  [Paper Note] Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames
description: >-
  [ACL 2026][LLM/NLP][metaphor detection] This paper proposes ConceptFrameMet, the first computational framework that integrates FrameNet semantic frames with source domains from Conceptual Metaphor Theory (CMT). A RoBERTa-based multi-task model is trained to jointly detect metaphors and predict their semantic frames and source domains. Combined with a log-likelihood ratio (LLR) statistical method for identifying salient metaphorical patterns in discourse, the framework reveals that liberal and conservative outlets employ the same source domains in immigration discourse yet select systematically different semantic frames to convey opposing associations.
tags:
  - ACL 2026
  - LLM/NLP
  - metaphor detection
  - Conceptual Metaphor Theory
  - semantic frames
  - discourse analysis
  - media framing
date: 2026-05-08
content_hash: 80455192857d913e
---

# Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames

**Conference**: ACL 2026
**arXiv**: [2604.20454](https://arxiv.org/abs/2604.20454)
**Code**: [https://github.com/julia-nixie/ConceptFrameMet](https://github.com/julia-nixie/ConceptFrameMet)
**Area**: LLM/NLP
**Keywords**: metaphor detection, Conceptual Metaphor Theory, semantic frames, discourse analysis, media framing

## TL;DR

This paper proposes ConceptFrameMet, the first computational framework that integrates FrameNet semantic frames with source domains from Conceptual Metaphor Theory (CMT). A RoBERTa-based multi-task model is trained to jointly detect metaphors and predict their semantic frames and source domains. Combined with a log-likelihood ratio (LLR) statistical method for identifying salient metaphorical patterns in discourse, the framework reveals that liberal and conservative outlets employ the same source domains in immigration discourse yet select systematically different semantic frames to convey opposing associations.

## Background & Motivation

**Background**: Conceptual Metaphor Theory (CMT) is the dominant framework for metaphor analysis, understanding abstract target concepts through source domains such as WATER, ANIMAL, and WAR. NLP research on metaphor has focused primarily on metaphor detection and source domain mapping.

**Limitations of Prior Work**: Source domains alone do not fully account for the specific associations conveyed by a metaphor. For example, "illegal aliens flood into our country" and "waves of immigrants have always enriched us" both draw on the WATER source domain yet convey diametrically opposed attitudes—the former foregrounds uncontrollable inundation, the latter naturalizes immigration as a landscape feature. Existing work cannot explain why metaphors from the same source domain are simultaneously exploited by ideologically opposed groups.

**Key Challenge**: A source domain denotes a cluster of associations, but which associations are activated depends on the semantic frame evoked by the specific lexical item. The frame of "flood" is Filling (foregrounding movement and the negative consequences of overflow), whereas "wave" and "tide" evoke Quantified_mass or Natural_features (more neutral associations). This source domain × semantic frame interaction has been consistently overlooked in NLP.

**Goal**: (1) Develop a computational model that automatically detects metaphors and predicts their source domains and semantic frames; (2) design a statistical method to identify salient metaphorical patterns in discourse; (3) analyze ideological differences in metaphorical framing.

**Key Insight**: The paper introduces constructionist linguistic theory (Sullivan 2013, 2025) into NLP—semantic frames serve as the mechanism that "selects" specific associations from a source domain, and the source domain × frame interaction uniquely determines the associative meaning of a metaphor.

**Core Idea**: Source domains specify clusters of associations, while semantic frames pinpoint precise associations within those clusters. Their interaction—rather than either dimension in isolation—is the key to analyzing the framing effects of metaphor.

## Method

### Overall Architecture

ConceptFrameMet comprises two components: (1) a RoBERTa-based multi-task model that jointly detects metaphors, predicts semantic frames (797 classes from FrameNet 1.7), and predicts source domains (99 classes from the LCC dataset); (2) a log-likelihood ratio (LLR) statistical module that computes salience scores for source domain and semantic frame combinations within a target discourse to identify discourse metaphors.

### Key Designs

1. **Semantic Frame Classifier**:

    - Function: Predicts one of 797 FrameNet semantic frames for a target word.
    - Mechanism: Fine-tunes RoBERTa-base with SEP-delimited input (separating the target word from its sentential context). Achieves 86.1% accuracy and 64.8% macro-F1 on the FrameNet 1.7 test set, approaching the performance of SOTA methods that rely on extensive data augmentation. Zero-shot LLMs (Gemini 2.5, Claude Sonnet 4.0) fall substantially short of the fine-tuned RoBERTa.
    - Design Motivation: Serves as the foundational module for downstream source domain prediction and discourse analysis. The SEP input format outperforms the MASK format by preserving information from the target word itself.

2. **Source Domain Classifier (with Semantic Frame Augmentation)**:

    - Function: Predicts one of 99 source domains for a given metaphor.
    - Mechanism: Builds on the RoBERTa SEP backbone by incorporating the frozen probability distribution output by the semantic frame classifier as an additional feature vector. A Frames_ATTN variant is proposed: two sets of semantic frame vectors are maintained—one trainable and one frozen—with source domain embeddings serving as queries to attend over the trainable matrix, highlighting frames that are discriminative for source domain prediction, while the frozen vectors serve as a residual. This yields a macro-F1 improvement of 20 percentage points on underrepresented classes.
    - Design Motivation: Validates the core hypothesis that semantic frames genuinely help distinguish semantically adjacent source domains. The attention mechanism allows the model to learn which frames are discriminative for which source domains.

3. **Log-Likelihood Ratio Salience Analysis**:

    - Function: Identifies source domain and semantic frame combinations that are significantly overrepresented in a target discourse.
    - Mechanism: Applies the LLR method of Rayson & Garside (2000) to compare the frequency distributions of source domains and frames in a target corpus (e.g., climate change news metaphors) against a reference corpus (a general metaphor dataset). High LLR values indicate overuse within the target discourse, reflecting salience as a discourse metaphor.
    - Design Motivation: Raw frequency counts cannot distinguish discourse-specific metaphor usage from common metaphors in general language. LLR surfaces patterns that are anomalously prominent within a particular discourse.

### Loss & Training

The three classifiers are fine-tuned independently, all using RoBERTa-base. The metaphor detector is fine-tuned on the VUA dataset. The semantic frame classifier is fine-tuned on FrameNet 1.7 (19,391 / 2,272 / 6,714 train/dev/test). The source domain classifier is fine-tuned on the large-scale LCC dataset (11,704 / 2,509 / 2,509), with the frozen semantic frame probability distribution introduced as an auxiliary feature during source domain prediction.

## Key Experimental Results

### Main Results

**Semantic Frame Prediction (FrameNet 1.7 Test Set)**

| Method | Accuracy | micro-F1 | macro-F1 |
|--------|----------|----------|----------|
| RoBERTa MASK | 0.806 | 0.806 | 0.053 |
| RoBERTa SEP | 0.861 | 0.866 | 0.648 |
| Gemini 2.5 | 0.508 | 0.508 | 0.430 |
| Claude Sonnet 4.0 | 0.736 | 0.736 | 0.600 |

**Source Domain Prediction (LCC Test Set)**

| Method | Accuracy | F1 |
|--------|----------|----|
| RoBERTa SEP | 0.833 | 0.740 |
| Frames_CONCAT | 0.837 | 0.754 |
| **Frames_ATTN** | **0.838** | **0.756** |
| Gemini 2.5 | 0.528 | 0.345 |

### Ablation Study

| Configuration | Description | Performance |
|---------------|-------------|-------------|
| No frame information | RoBERTa only | F1 0.740 |
| CONCAT fusion | Simple concatenation of frame vectors | F1 0.754 (+1.4) |
| ATTN fusion | Attention-based frame integration | F1 0.756 (+1.6) |
| Low-frequency class improvement | Classes with <10 samples | macro-F1 +20 pp |

### Key Findings

- The most salient source domains in climate change discourse are BODY (climate as a "sick body"), WAR ("fight against"), and MACHINE ("levers of change").
- Conservative and liberal outlets exhibit similar source domain distributions in immigration discourse, but their semantic frame selections differ significantly.
- Conservative sources favor frames emphasizing uncontrollability (e.g., Motion_directional within the WATER domain), while liberal sources prefer neutral or "victim-oriented" frames (e.g., Quantified_mass).
- Within the ANIMAL source domain, conservative outlets tend toward Biological_urge (animal instinct/aggression), while liberal outlets use Self_motion (autonomous movement, more neutral).
- Zero-shot LLMs underperform fine-tuned small models substantially on fine-grained classification tasks (797 frame classes, 99 source domain classes), demonstrating that such tasks still require dedicated training.

## Highlights & Insights

- The theoretical contribution is substantial: this is the first work to introduce the source domain × semantic frame interaction from constructionist linguistics into NLP, providing a new analytical dimension for understanding why metaphors are simultaneously adopted by opposing ideological groups.
- The empirical finding that conservative and liberal sources select different semantic frames under the same source domain has direct relevance to political communication research.
- The design of using target-task embeddings as queries to select auxiliary features in Frames_ATTN is transferable to other NLP tasks requiring multi-granularity feature fusion.

## Limitations & Future Work

- The macro-F1 of the semantic frame classifier remains relatively low (0.648), largely due to the presence of many semantically adjacent fine-grained classes among the 797 categories.
- The analysis is limited to English corpora; cross-lingual variation in metaphorical framing warrants future investigation.
- The LLR method is statistical in nature and cannot capture the dynamic evolution of metaphors within context.
- Future work could extend the framework to additional discourse types, including social media and political speeches.

## Related Work & Insights

- **vs. Mendelsohn & Budak (2025)**: Their work finds that opposing ideologies use the same source domains but cannot explain why; this paper provides an explanation through the semantic frame dimension.
- **vs. Gordon et al. (2015)**: They encode semantic roles without analyzing frame–source domain interactions; this paper achieves a systematic integration of both for the first time.
- **vs. Stowe et al. (2021)**: They use frames to assist metaphor generation; this paper uses frames to support metaphor analysis and source domain prediction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to introduce the source domain × frame interaction from constructionist metaphor theory into NLP; theoretically innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Analyses across two discourse domains with multiple baselines; quantitative evaluation relies primarily on classification metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous interdisciplinary argumentation, vivid examples, and tight integration of theory and empirics.
- Value: ⭐⭐⭐⭐ Opens a new direction for metaphor analysis and framing effect research with cross-disciplinary impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](../../AAAI2026/llm_nlp/do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](../../AAAI2026/llm_nlp/vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](../../ICLR2026/llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)
- [\[ICLR 2026\] KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](../../ICLR2026/llm_nlp/kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)

</div>

<!-- RELATED:END -->
