---
title: >-
  [Paper Note] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations
description: >-
  [ICLR 2026][Interpretability][Sparse Autoencoders] Using the Serbian digraphic system (Latin/Cyrillic) as a natural controlled experiment…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "script-invariance"
  - "Serbian digraphia"
  - "semantic representations"
date: 2026-05-08
content_hash: 61c825212201c9d3
---

# One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations

**Conference**: ICLR 2026
**arXiv**: [2603.08869](https://arxiv.org/abs/2603.08869)
**Code**: None
**Area**: LLM Interpretability / Multilingual
**Keywords**: Sparse Autoencoders, script-invariance, Serbian digraphia, semantic representations, interpretability

## TL;DR
Using the Serbian digraphic system (Latin/Cyrillic) as a natural controlled experiment, this paper investigates whether features learned by Sparse Autoencoders (SAEs) capture abstract semantics beyond surface-level tokenization. The study finds that identical sentences across scripts activate highly overlapping SAE features (Jaccard ≈ 0.58), that script switching induces smaller representational differences than same-script paraphrasing, and that this invariance strengthens with model scale — demonstrating that SAE features genuinely capture semantic structure beyond orthography.

## Background & Motivation

**Background**: SAEs have become a key tool in mechanistic interpretability, decomposing neural network activations into sparse, interpretable features. However, a fundamental question remains unanswered: do SAE-learned features represent abstract semantics, or are they bound to the specific written form of the text?

**Limitations of Prior Work**: Cross-lingual representation studies (multilingual BERT / XLM-R) demonstrate cross-lingual transfer, but different languages introduce confounds such as lexical, grammatical, and cultural differences that are difficult to control. Cross-script studies on Hindi–Urdu introduce noise due to imperfect script mappings.

**Key Challenge**: An ideal controlled experiment is needed — one that holds semantics completely constant while varying only the writing system, while also ensuring entirely disjoint tokenizations. Only then can one cleanly test whether SAE features truly capture semantics.

**Key Insight**: Serbian is one of the very few languages with an active digraphic system — Latin and Cyrillic scripts are used interchangeably in everyday life, with a deterministic lossless character-level mapping. Critically, the two scripts are tokenized entirely differently, sharing zero tokens. This constitutes a perfect controlled experiment.

**Core Idea**: Serbian digraphia provides a natural controlled experiment demonstrating that SAE features capture abstract semantic representations beyond surface-level tokenization.

## Method

### Overall Architecture
Input: 30 sentence triplets (original / paraphrase / random) × 3 language variants (English / Serbian Latin / Serbian Cyrillic) = 270 sentences. Models used: the Gemma model family (270M–27B) with Gemma Scope 2 SAEs (65,536 features). Output: SAE feature overlap analysis across 14 comparison types.

### Key Designs

1. **Serbian Digraphia as a Controlled Experiment**:

    - Function: Creates comparison conditions that hold semantics constant while varying all surface features (tokenization).
    - Mechanism: The Latin and Cyrillic versions of the same sentence are semantically identical, yet produce entirely different token sequences (zero shared tokens). LaBSE confirms cross-script semantic similarity > 0.95.
    - Design Motivation: Eliminates confounds present in cross-lingual research (lexical differences / grammatical differences / cultural differences). The deterministic mapping guarantees zero semantic change.

2. **SAE Feature Extraction Pipeline**:

    - Function: Extracts the set of SAE features activated by each sentence.
    - Mechanism: Sentence → tokenizer → Gemma forward pass → hidden state of the last token at the target layer → SAE encoder yielding 65,536-dimensional activations → JumpReLU threshold ($\tau = 0.1$) → active feature set $F(s) = \{i : a_i > \tau\}$.
    - Design Motivation: Last-token pooling is more robust than mean pooling (verified experimentally). The fixed threshold $\tau = 0.1$ corresponds to the standard JumpReLU setting.

3. **Systematic Design of 14 Comparison Types**:

    - Function: Systematically tests semantic similarity vs. script invariance vs. random baselines.
    - Mechanism: Jaccard similarity $J(s_1, s_2) = |F(s_1) \cap F(s_2)| / |F(s_1) \cup F(s_2)|$. Comparison dimensions include:
        - Baselines: same-script original vs. paraphrase (semantically similar); original vs. random (semantically unrelated).
        - Core tests: cross-script original (script change only); cross-script paraphrase (script + wording change).
        - Random baselines: cross-script random; cross-language random.
    - Design Motivation: Multi-level comparisons enable distinguishing whether script invariance is truly driven by semantics rather than other confounds.

### Evaluation Metrics
- Jaccard similarity: 0 (no overlap) to 1 (identical).
- Mean computed over 30 sentence pairs per comparison type.
- Averages reported across all models and layers.

## Key Experimental Results

### Main Results: Cross-Script Representational Invariance (Average Across All Models)

| Comparison Type | Mean Jaccard Similarity |
|----------------|------------------------|
| Cross-script original (same sentence, different script) | 0.58 |
| Cross-script paraphrase (different paraphrase, different script) | 0.59 |
| Cross-script cross-paraphrase | 0.47 |
| Cross-script random | 0.28 |
| Cross-language random | 0.19 |

### Ablation Study: Effect of Model Scale

| Model | Cross-Script Original | Cross-Script Random | Gap (Signal − Noise) |
|-------|-----------------------|---------------------|----------------------|
| Gemma-270M | 0.501 | 0.421 | 0.080 |
| Gemma-1B | 0.537 | 0.324 | 0.213 |
| Gemma-4B | 0.571 | 0.253 | 0.318 |
| Gemma-12B | 0.624 | 0.233 | 0.391 |
| Gemma-27B | 0.649 | 0.211 | 0.438 |

### Key Findings
- **Script change < paraphrase change**: Cross-script original (0.58) exceeds same-script paraphrase (0.54), indicating that switching scripts induces smaller representational differences than rewording — SAE features prioritize semantic encoding over orthographic form.
- **Clear semantic hierarchy**: Cross-script original (0.58) >> cross-script cross-paraphrase (0.47) >> cross-script random (0.28) >> cross-language random (0.19), perfectly consistent with expected semantic similarity ordering.
- **Significant scale effect**: Cross-script original Jaccard increases from 0.50 (270M) to 0.65 (27B), while the random baseline drops from 0.42 to 0.21 — larger models develop more robust script-agnostic representations.
- **Refuting the memorization hypothesis**: Cross-script cross-paraphrase pairs (Latin original vs. Cyrillic paraphrase) are virtually absent from training data co-occurrences, yet still yield an overlap of 0.47, indicating genuine semantic alignment rather than memorization.

## Highlights & Insights
- **Serbian digraphia as a general-purpose evaluation paradigm**: The experimental design is exceptionally elegant — it exploits a unique property of natural language to eliminate all confounding variables. This approach could serve as a standard test for evaluating whether any representation learning method captures abstract semantics.
- **"Script change < paraphrase change"** is a highly counterintuitive and compelling finding: entirely different token sequences yield more similar representations than same-script paraphrases, providing strong evidence that SAE features operate beyond the token level.
- **Bidirectional scale effect**: Larger models not only increase cross-script similarity (genuine semantic alignment) but also reduce the random baseline (improved feature sparsity) — improvement in both directions simultaneously.
- **Methodological minimalism, profound insights**: Without complex models or training procedures, carefully designed contrastive experiments yield important conclusions about the nature of LLM representations.

## Limitations & Future Work
- Only the Gemma model family is tested; other architectures (LLaMA / GPT) and SAEs trained with different methods may behave differently.
- Only 30 sentence triplets are used, limiting scale and domain coverage.
- Only feature overlap is measured; causal relationships are not established — activation patching and feature ablation are needed to verify whether shared features actually drive cross-script comprehension.
- The deterministic mapping of Serbian represents an ideal case; other multi-script languages (Japanese kanji / kana) involve more complex mappings.
- Identifying which specific SAE features exhibit the greatest script invariance could reveal interpretable semantic anchors.

## Related Work & Insights
- **vs. Multilingual BERT (Pires et al.)**: Cross-lingual transfer in multilingual BERT may be influenced by vocabulary overlap. Serbian digraphia completely eliminates this confound.
- **vs. Hindi–Urdu studies**: Hindi–Urdu script mappings are imperfect due to lexical and cultural differences. The deterministic Serbian mapping provides a cleaner control.
- **vs. SAE interpretability research (Bricken / Cunningham)**: Prior SAE research has been conducted primarily on monolingual English. This work is the first to evaluate the semantic abstraction of SAE features from a cross-script perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The use of Serbian digraphia as a controlled experiment is exceptionally elegant and represents an ideal testbed for this line of research.
- Experimental Thoroughness: ⭐⭐⭐ Coverage across five model scales is adequate, but the dataset is small (30 sentences) and only one model family is evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ Experimental design is clear and conclusions are rigorously derived.
- Value: ⭐⭐⭐⭐ Makes an important contribution to understanding the nature of LLM representations and proposes a reusable evaluation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning](dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)

</div>

<!-- RELATED:END -->
