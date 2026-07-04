---
title: >-
  [Paper Note] Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues
description: >-
  [ACL 2025][LLM Pretraining][Chinese Grammatical Error Correction] This paper proposes a Chinese grammatical error correction method that integrates pre-trained language models with multi-level linguistic clues (pinyin, glyphs, and dependency syntax). By explicitly injecting linguistic prior knowledge, it enhances the correction model's ability to identify and amend Chinese-specific error types.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Chinese Grammatical Error Correction"
  - "Pre-trained Models"
  - "Linguistic Clues"
  - "GEC"
  - "Multi-granularity Features"
date: 2026-05-08
content_hash: 66680409d8297d28
---

# Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues

**Conference**: ACL 2025  
**Area**: LLM Pre-training  
**Keywords**: Chinese Grammatical Error Correction, Pre-trained Models, Linguistic Clues, GEC, Multi-granularity Features

## TL;DR
This paper proposes a Chinese grammatical error correction method that integrates pre-trained language models with multi-level linguistic clues (pinyin, glyphs, and dependency syntax). By explicitly injecting linguistic prior knowledge, it enhances the correction model's ability to identify and amend Chinese-specific error types.

## Background & Motivation

**Background**: Grammatical Error Correction (GEC) is a classic NLP task. In recent years, methods based on pre-trained models (such as BERT, BART, T5) have become mainstream. While mature solutions exist for English GEC, Chinese GEC faces unique challenges: Chinese lacks explicit word boundaries, grammatical error types differ significantly from English (e.g., missing articles or preposition errors do not exist in Chinese), and Chinese-specific pinyin and glyph information are crucial correction clues.

**Limitations of Prior Work**: Current Chinese GEC methods mainly fall into two categories: (1) Seq2Seq-based generation methods, which directly translate erroneous sentences into correct ones but often ignore Chinese-specific linguistic information (such as spelling errors caused by phonetic or visual similarity); (2) Edit-based methods, which predict edit operations (replace, insert, delete) at each position but struggle with errors requiring global semantic understanding (such as word order errors, missing/redundant components). Neither category fully utilizes multi-level Chinese linguistic clues.

**Key Challenge**: Chinese error types are diverse (spelling errors, grammatical errors, semantic collocation errors) and require different levels of linguistic knowledge to address. However, the implicit representations of pre-trained models struggle to cover all these levels.

**Goal**: To design a GEC framework that simultaneously leverages the semantic understanding capability of pre-trained models and Chinese-specific linguistic clues, achieving unified handling of various Chinese error types.

**Key Insight**: A large portion of Chinese spelling errors stem from phonetic similarity (homophones) or visual similarity (cognates/homoglyphs). This information is crucial for correction but is not present on the text surface. Explicitly modeling these clues can compensate for the limitations of pre-trained models.

**Core Idea**: On top of the pre-trained Seq2Seq model, integrate three auxiliary modules—a pinyin encoder, a glyph encoder, and a dependency syntax parser—to provide multi-level linguistic clues for enhanced correction.

## Method

### Overall Architecture
The model uses a pre-trained Chinese BART/T5 as its backbone network, taking an erroneous sentence as input and outputting the corrected sentence. On the encoder side, in addition to standard token embeddings, three types of auxiliary information (pinyin embeddings, glyph embeddings, and syntactic features) are integrated. On the decoder side, a constrained decoding strategy is employed to ensure output fluency.

### Key Designs

1. **Pinyin-Aware Encoder**:

    - Function: Captures homophonic relationships in Chinese, helping the model identify spelling errors based on pronunciation similarity.
    - Mechanism: Converts each Chinese character in the input text into a pinyin sequence (including onset, rime, and tone), and encodes the pinyin into vectors using a specialized pinyin embedding layer. The pinyin vector and token embedding are combined via a gated fusion mechanism: $h_{fused} = g \odot h_{token} + (1-g) \odot h_{pinyin}$, where the gating parameter $g$ is adaptively determined by the input. Thus, when the model encounters characters with identical pronunciation but different characters, it can identify potential phonetic errors through pinyin clues.
    - Design Motivation: Approximately 40% of Chinese spelling errors are homophonic errors, and traditional text-only models cannot utilize this crucial information.

2. **Glyph Feature Encoder**:

    - Function: Captures visual similarity relationships in Chinese, helping the model identify spelling errors based on visual similarity.
    - Mechanism: Renders each Chinese character into an image and extracts glyph feature vectors using a lightweight CNN (such as the first few layers of ResNet-18). Glyph features are similarly fused with token embeddings through a gating mechanism. For visually similar characters (e.g., "ji" and "yi", "wu" and "xu"), the glyph features exhibit high similarity, allowing the model to identify potential visual errors based on this.
    - Design Motivation: Approximately 30% of Chinese spelling errors are visual similarity errors, and glyph information provides visual similarity cues that cannot be obtained from the text surface.

3. **Syntax-Enhanced Decoder**:

    - Function: Utilizes syntactic dependency relationships to help the model process structural grammatical errors (e.g., word order and missing components).
    - Mechanism: Runs a dependency syntactic parser on the input sentence to obtain the dependency tree structure. It then encodes dependency relations as an additional attention bias, guiding the model to focus on syntactically related positions within the decoder's cross-attention. Specifically, if word $i$ and word $j$ share a dependency relation, a positive bias is added to the attention score.
    - Design Motivation: Grammatical errors (such as errors in "ba" constructions or "bei" constructions) require global syntactic understanding; pure sequence modeling struggles to capture these long-range syntactic constraints.

### Loss & Training
The primary loss is the standard sequence-to-sequence cross-entropy loss. Additionally, a pinyin prediction auxiliary loss (predicting the pinyin of each character in the output sentence) and an error detection auxiliary loss (binary classification to predict whether each input position contains an error) are introduced. These two auxiliary losses encourage the encoder to learn better representations.

## Key Experimental Results

### Main Results

| Method | SIGHAN15 P↑ | SIGHAN15 R↑ | SIGHAN15 F1↑ | MuCGEC F1↑ | NLPCC F1↑ |
|------|-------------|-------------|--------------|-----------|-----------|
| BART-Chinese | 73.2 | 67.8 | 70.4 | 42.1 | 38.5 |
| GECToR-Chinese | 71.5 | 70.2 | 70.8 | 40.3 | 37.2 |
| SCOPE | 75.8 | 69.3 | 72.4 | 43.8 | 40.1 |
| MaskGEC | 74.6 | 71.5 | 73.0 | 44.2 | 41.3 |
| **Ours**| **78.3** | **73.6** | **75.9** | **47.5** | **44.2** |

### Ablation Study

| Configuration | SIGHAN15 F1↑ | MuCGEC F1↑ | Description |
|------|--------------|-----------|------|
| Full Model | 75.9 | 47.5 | All components |
| w/o Pinyin Encoder | 73.1 | 44.8 | Pinyin contribution +2.8 |
| w/o Glyph Encoder | 74.2 | 45.6 | Glyph contribution +1.7 |
| w/o Syntactic Enhancement | 74.8 | 46.1 | Syntactic contribution +1.1 |
| w/o Auxiliary Loss | 75.0 | 46.3 | Auxiliary loss contribution +0.9 |
| Pinyin + Glyph (w/o Syntax) | 75.2 | 46.8 | Mutual complementarity of the three |

### Key Findings
- The pinyin encoder makes the largest contribution (+2.8 F1), validating the high frequency of homophonic errors in Chinese GEC and the necessity of introducing pinyin.
- The glyph encoder contributes the second most (+1.7 F1), with the improvement in correcting visually similar errors stemming from the visual features extracted by the CNN.
- Syntactic enhancement contributes relatively more to more complex datasets (MuCGEC, NLPCC) that contain more grammatical errors.
- The three linguistic clues are highly complementary; using all three simultaneously yields better performance than any combination of two.
- On the spelling-only error subset, the improvement from pinyin + glyph is particularly significant (+4.5 F1), but its advantage diminishes on the grammar-only error subset.

## Highlights & Insights
- The design of the fusion strategy for multi-level linguistic clues is reasonable, and the gating mechanism allows the model to adaptively decide when to rely on which clues.
- The approach of rendering Chinese characters as images to extract features is highly creative, ingeniously transforming glyph similarity into vector similarity within the embedding space.
- The design of auxiliary losses (pinyin prediction and error detection) acts as an effective regularization, improving the representation quality of the encoder.

## Limitations & Future Work
- Training the glyph CNN requires additional parameters and computational resources, which may not be suitable in resource-constrained scenarios.
- The syntactic parser itself may perform poorly on sentences containing errors, potentially leading to error propagation.
- The current approach assumes that input sentences have distinct types of errors, exhibiting limited performance on fluent but semantically inappropriate sentences.
- Future work could consider incorporating LLMs for post-processing or re-ranking, combining the advantages of edit-based and generative models.

## Related Work & Insights
- **vs SCOPE (Li et al., 2022)**: SCOPE utilizes pinyin information for spelling correction, whereas ours extends this to glyphs and syntax, covering a wider range of error types.
- **vs Linguistic Rules-Based (Wang et al., 2022)**: While prior work generates training corpora based on linguistic rules, ours directly encodes linguistic clues into the model.
- **vs GECToR (Omelianchuk et al., 2020)**: GECToR is a classic edit-based method, whereas our Seq2Seq method achieves superior performance on Chinese GEC.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of multi-level clues is innovative; the combination of pinyin + glyph + syntax is relatively novel in GEC.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple datasets with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The motivation for the proposed method is clear, and the analysis of Chinese linguistic characteristics is comprehensive.
- Value: ⭐⭐⭐⭐ Directly practical and valuable for Chinese NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](between_circuits_chomsky.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](../../ICCV2025/llm_pretraining/dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)

</div>

<!-- RELATED:END -->
