---
title: >-
  [Paper Note] Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames
description: >-
  [ACL 2026][LLM/NLP][Metaphor Detection] This paper proposes ConceptFrameMet, the first computational framework integrating FrameNet semantic frames and Conceptual Metaphor Theory (CMT) source domains. Using a multi-task…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Metaphor Detection"
  - "Conceptual Metaphor Theory"
  - "Semantic Frames"
  - "Discourse Analysis"
  - "Media Framing"
date: 2026-05-08
content_hash: 5407646fc21d1f00
---

# Not All Animals Are Equal: Metaphorical Framing through Source Domains and Semantic Frames

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.20454](https://arxiv.org/abs/2604.20454)  
**Code**: [https://github.com/julia-nixie/ConceptFrameMet](https://github.com/julia-nixie/ConceptFrameMet)  
**Area**: LLM/NLP  
**Keywords**: Metaphor Detection, Conceptual Metaphor Theory, Semantic Frames, Discourse Analysis, Media Framing

## TL;DR

This paper proposes ConceptFrameMet, the first computational framework integrating FrameNet semantic frames and Conceptual Metaphor Theory (CMT) source domains. Using a multi-task RoBERTa model to detect metaphors and predict their semantic frames and source domains, combined with Log-Likelihood Ratio (LLR) statistics, the study identifies significant metaphorical patterns in discourse. It reveals that while liberals and conservatives use the same source domains in immigration discourse, they select different semantic frames to convey distinct associations.

## Background & Motivation

**Background**: Conceptual Metaphor Theory (CMT) is the dominant framework for analyzing metaphors—understanding abstract target concepts through source domains (e.g., WATER, ANIMAL, WAR). Metaphor research in NLP has focused primarily on metaphor detection and source domain mapping.

**Limitations of Prior Work**: Source domains alone cannot fully explain the specific associations conveyed by a metaphor. For example, "illegal aliens flood into our country" and "waves of immigrants have always enriched us" both originate from the WATER source domain but convey opposing attitudes—the former emphasizes loss of control (flooding), while the latter suggests a natural landscape. Existing work fails to explain why opposing ideological camps simultaneously use the same source domains.

**Key Challenge**: A source domain points to a cluster of associations, but which specific associations are activated depends on the semantic frame corresponding to the vocabulary used. The semantic frame of "flood" is Filling (emphasizing motion and negative outcomes of overflow), whereas "wave"/"tide" suggests Quantified_mass or Natural_features (more neutral associations). This interaction between source domains and semantic frames has been ignored by NLP.

**Goal**: (1) Build a computational model to automatically detect metaphors and predict source domains and semantic frames; (2) design statistical methods to discover significant metaphorical patterns in discourse; (3) analyze ideological differences in the use of metaphorical frames.

**Key Insight**: This work introduces constructivist linguistic theories (Sullivan 2013, 2025) to NLP—semantic frames are the mechanism for "picking" specific associations from a source domain. The interaction of source domain $\times$ semantic frame uniquely defines the metaphorical association.

**Core Idea**: Use source domains to specify clusters of associations and semantic frames to pinpoint specific associations within those clusters—it is the interaction between them, rather than either dimension alone, that is key to analyzing metaphorical framing effects.

## Method

### Overall Architecture

ConceptFrameMet consists of two parts: (1) A multi-task model based on RoBERTa that jointly detects metaphors and predicts semantic frames (797 classes from FrameNet 1.7) and source domains (99 classes from the LCC dataset); (2) a Log-Likelihood Ratio (LLR) statistical module that calculates significance scores for source domain and semantic frame combinations in specific discourses to identify discourse metaphors.

### Key Designs

1.  **Semantic Frame Classifier**:
    *   **Function**: Predicts one of 797 semantic frames from FrameNet for a target word.
    *   **Mechanism**: Fine-tunes RoBERTa-base using SEP-delimited input (separating the target word from the context sentence). It achieves 86.1% accuracy and 64.8% macro-F1 on the FrameNet 1.7 test set, performing close to SOTA methods that require heavy data augmentation. Comparative analysis shows zero-shot LLMs (Gemini 2.5, Claude Sonnet 4.0) are significantly inferior to fine-tuned RoBERTa.
    *   **Design Motivation**: Serves as the foundational module for downstream source domain prediction and discourse analysis. The SEP input format outperforms the MASK format by preserving information about the target word itself.

2.  **Source Domain Classifier (with Semantic Frame Enhancement)**:
    *   **Function**: Predicts one of 99 source domains for a metaphor.
    *   **Mechanism**: Based on RoBERTa SEP, the probability distribution output from the semantic frame classifier is introduced as a frozen feature vector. A Frames_ATTN variant is proposed: it maintains both trainable and frozen semantic frame vectors. Source domain embeddings are used as queries to apply attention to the trainable matrix, highlighting semantic frames important for source domain prediction, with the frozen vector acting as a residual. Macro-F1 improves by 20 percentage points on underrepresented classes.
    *   **Design Motivation**: Validates the core hypothesis that semantic frames help distinguish between semantically similar source domains. The attention mechanism allows the model to learn which frames are discriminative for specific source domains.

3.  **Log-Likelihood Ratio Significance Analysis**:
    *   **Function**: Discovers combinations of source domains and semantic frames that are significantly overused in specific discourses.
    *   **Mechanism**: Uses the LLR method from Rayson & Garside (2000) to compare the frequency distribution of source domains/frames in a specific corpus (e.g., metaphors in climate change news) against a reference corpus (a general metaphor dataset). High LLR values indicate overuse in that discourse, reflecting its significance as a discourse metaphor.
    *   **Design Motivation**: Frequency counts alone cannot distinguish discourse-specific metaphor usage from common language metaphors. LLR identifies patterns that are "unusually prominent in this specific discourse."

### Loss & Training

Three classifiers are fine-tuned independently using RoBERTa-base. The metaphor detector is fine-tuned on the VUA dataset. The semantic frame classifier is fine-tuned on FrameNet 1.7 (19391/2272/6714 train/val/test). The source domain classifier is fine-tuned on the large-scale LCC dataset (11704/2509/2509), using the frozen semantic frame probability distribution as an additional feature during source domain prediction.

## Key Experimental Results

### Main Results

**Semantic Frame Prediction Performance (FrameNet 1.7 Test Set)**

| Method | Accuracy | micro-F1 | macro-F1 |
| :--- | :--- | :--- | :--- |
| RoBERTa MASK | 0.806 | 0.806 | 0.053 |
| RoBERTa SEP | 0.861 | 0.866 | 0.648 |
| Gemini 2.5 | 0.508 | 0.508 | 0.430 |
| Claude Sonnet 4.0 | 0.736 | 0.736 | 0.600 |

**Source Domain Prediction Performance (LCC Test Set)**

| Method | Accuracy | F1 |
| :--- | :--- | :--- |
| RoBERTa SEP | 0.833 | 0.740 |
| Frames_CONCAT | 0.837 | 0.754 |
| **Frames_ATTN** | **0.838** | **0.756** |
| Gemini 2.5 | 0.528 | 0.345 |

### Ablation Study

| Configuration | Description | Effect |
| :--- | :--- | :--- |
| No Frame Info | RoBERTa only | F1 0.740 |
| CONCAT Fusion | Simple concatenation of frame vectors | F1 0.754 (+1.4) |
| ATTN Fusion | Attention-based frame fusion | F1 0.756 (+1.6) |
| Low-freq Class Boost | Classes with <10 samples | macro-F1 +20 pts |

### Key Findings

- The most significant source domains in climate change discourse are BODY (climate as a "sick body"), WAR ("fight against"), and MACHINE ("levers of change").
- In immigration discourse, conservatives and liberals use similar source domain distributions, but their choices of semantic frames differ significantly.
- Conservatives prefer frames emphasizing uncontrollability (e.g., Motion_directional for the WATER domain), while liberals prefer neutral or "victimizing" frames (e.g., Quantified_mass).
- Within the ANIMAL source domain, conservatives lean toward Biological_urge (animal instinct/aggression), while liberals use Self_motion (autonomous movement, more neutral).
- Zero-shot LLMs perform much worse than fine-tuned small models on fine-grained classification (797 frames, 99 source domains), indicating these tasks still require specialized training.

## Highlights & Insights

- Significant theoretical contribution: This is the first work to introduce the "source domain $\times$ semantic frame" interaction theory from constructivist linguistics into NLP, providing a new analytical dimension for understanding why opposing camps use the same metaphors.
- The empirical finding that conservatives and liberals select different semantic frames under the same source domains offers direct value to political communication studies.
- The design of using target task embeddings as queries to select auxiliary features in Frames_ATTN can be transferred to other NLP tasks requiring multi-granularity feature fusion.

## Limitations & Future Work

- The macro-F1 of the semantic frame classifier remains relatively low (0.648), primarily due to the large number of semantically similar sub-classes among the 797 categories.
- The analysis is limited to English corpora; cross-linguistic differences in metaphorical framing merit further exploration.
- The Log-Likelihood Ratio method is statistical and cannot capture the dynamic evolution of metaphors within a context.
- Future work could extend to more discourse types such as social media and political speeches.

## Related Work & Insights

- **vs Mendelsohn & Budak (2025)**: They observed that opposing ideologies use the same source domains but could not explain why; this paper provides an explanation via the semantic frame dimension.
- **vs Gordon et al. (2015)**: They encoded semantic roles but did not analyze the interaction between frames and source domains; this paper is the first to systematically combine the two.
- **vs Stowe et al. (2021)**: They used frames to assist metaphor generation; this paper uses frames for metaphor analysis and source domain prediction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Significant theoretical innovation by introducing source domain $\times$ frame interaction into NLP.
- Experimental Thoroughness: ⭐⭐⭐⭐ Analysis of two discourse domains and multiple baseline comparisons, though quantitative evaluation relies mainly on classification metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous interdisciplinary argumentation with vivid examples and tight integration of theory and empirical evidence.
- Value: ⭐⭐⭐⭐ Opens new directions for metaphor analysis and framing effect research with interdisciplinary impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Differential Syntactic and Semantic Encoding in LLMs](../../ICML2026/llm_nlp/differential_syntactic_and_semantic_encoding_in_llms.md)
- [\[ACL 2026\] Unlocking the Potential of Diffusion Language Models through Template Infilling](unlocking_the_potential_of_diffusion_language_models_through_template_infilling.md)
- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](../../AAAI2026/llm_nlp/vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)
- [\[ICML 2026\] SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling](../../ICML2026/llm_nlp/sac-opt_semantic_anchors_for_iterative_correction_in_optimization_modeling.md)
- [\[ICML 2026\] Position: Adversarial ML for LLMs Is Not Making Any Progress](../../ICML2026/llm_nlp/position_adversarial_ml_for_llms_is_not_making_any_progress.md)

</div>

<!-- RELATED:END -->
