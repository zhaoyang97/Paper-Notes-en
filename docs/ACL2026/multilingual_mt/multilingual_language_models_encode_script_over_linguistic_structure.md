---
title: >-
  [Paper Note] Multilingual Language Models Encode Script Over Linguistic Structure
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual representations] This paper systematically analyzes language-associated units in multilingual LMs using the LAPE metric and sparse autoencoders. It discovers th…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual representations"
  - "scripts"
  - "orthography"
  - "language-associated neurons"
  - "sparse autoencoders"
date: 2026-05-08
content_hash: 1ace67e9b33df201
---

# Multilingual Language Models Encode Script Over Linguistic Structure

**Conference**: ACL 2026  
**arXiv**: [2604.05090](https://arxiv.org/abs/2604.05090)  
**Code**: [GitHub](https://github.com/loadthecode0/multilingual-interpretability)  
**Area**: Human Understanding / Multilingual Interpretability  
**Keywords**: Multilingual representations, scripts, orthography, language-associated neurons, sparse autoencoders

## TL;DR

This paper systematically analyzes language-associated units in multilingual LMs using the LAPE metric and sparse autoencoders. It discovers that these units are primarily driven by orthography (script) rather than abstract linguistic structure: romanized transliterations activate almost entirely non-overlapping sets of neurons, word order scrambling has minimal impact, typological information only becomes accessible in deeper layers, and causal interventions demonstrate that functional importance is tied to surface-form invariance.

## Background & Motivation

**Background**: Multilingual language models (e.g., Llama, Gemma) compress representations of numerous languages into a shared parameter space. However, the nature of this internal organization—whether based on abstract linguistic identity or surface-form cues—remains unclear.

**Limitations of Prior Work**: (1) Previous studies (Tang et al., 2024) identified language-associated neurons using the LAPE metric and demonstrated causal control but did not answer what linguistic properties these neurons actually encode; (2) The "interlingua" hypothesis suggests that multilingual models form a unified, language-agnostic representation space, but direct evidence is insufficient; (3) Bilingual cognition research shows that comprehension and production can share semantic representations while separating surface processing, yet whether a similar phenomenon exists in LMs is unknown.

**Key Challenge**: While the existence of language-associated units is confirmed, do they encode abstract linguistic identity or surface cues such as orthography?

**Goal**: To systematically answer four research questions: (i) Language vs. Script—what do language-associated units encode? (ii) Structural Perturbation Robustness—how does word order scrambling affect them? (iii) Typological Alignment—what is their relationship with genetic, phonological, and syntactic features? (iv) Hierarchical Organization—how do these properties change with depth?

**Key Insight**: Designed controlled experiments—romanization (changing script while keeping content constant) and word order scrambling (changing structure while keeping surface form constant)—to disentangle the contributions of orthography and linguistic structure.

**Core Idea**: Multilingual LMs organize representations around surface forms (scripts); linguistic abstraction emerges layer by layer but never collapses into a unified interlingua.

## Method

### Overall Architecture

The analysis was conducted on four models: Llama-3.2-1B, Llama-3-8B, Gemma-2-2B, and Gemma-2-9B, spanning languages across Latin, Cyrillic, Devanagari, Arabo-Persian, and ideographic scripts. LAPE (Language Activation Probability Entropy) was used to locate language-associated units at the neuron level, and SAE-LAPE was used to locate language-associated features in the latent space of sparse autoencoders. Four types of experiments—romanization, scrambling, typological probing, and causal intervention—were performed to answer the research questions.

### Key Designs

1.  **Romanization Control Experiment**:
    - **Function**: To disentangle the contributions of script and language identity to language-associated units.
    - **Mechanism**: For non-Latin languages in the FLORES+ dataset, romanized versions (with/without diacritics) were generated using the ICU Transliterator. Language-associated units were identified for each version using LAPE, and overlap was measured via Jaccard similarity. Results showed that native script, romanized (with diacritics), and romanized (without diacritics) versions of Hindi activated almost entirely disjoint sets of neurons.
    - **Design Motivation**: If units encode abstract identity, they should remain stable across script changes; if they primarily encode orthography, they should reorganize.

2.  **Word Order Scrambling Experiment**:
    - **Function**: To test the dependence of language-associated units on syntactic structure.
    - **Mechanism**: Random word-level scrambling was applied to the evaluation corpora. SAE-LAPE was re-run to identify units, and stability was measured via Jaccard similarity. Results showed that most languages retained over 70% of their units, with unique scripts (Chinese, Japanese, Thai) showing the highest stability.
    - **Design Motivation**: This serves as a contrast to romanization—if script changes cause massive shifts but word order changes have little effect, surface form is prioritized over structure.

3.  **Typological Probing and Causal Intervention**:
    - **Function**: To examine whether deeper layers encode linguistic structures beyond surface forms.
    - **Mechanism**: Linear probes were used to decode lang2vec typological features (genetic, phonological, syntactic). Findings indicated that a subset of "overlap" neurons across scripts carried the strongest typological signals. Genetic features were decodable from shallow layers, while phonologic features emerged only in the deepest layers. Causal interventions showed that ablating script-invariant neurons caused only modest perplexity changes, whereas ablating script-specific neurons led to catastrophic degradation (PPL increased by 7.74x), confirming the latter anchors linguistic identity and surface realization.
    - **Design Motivation**: Combining probing and intervention distinguishes "representation accessibility" from "functional necessity"—accessible information is not necessarily required for generation.

### Loss & Training

This is an analytical study with no training involved. Pre-trained Top-K SAEs (for Llama) and JumpReLU SAEs (for Gemma) were used, focusing on MLP sub-layer activations.

## Key Experimental Results

### Main Results

**Overlap of Language-Associated Units after Romanization (Jaccard Similarity, Llama-3.2-1B)**

| Language | Native vs. Romanized (Neurons) | Native vs. Romanized (SAE Features) | Romanized vs. English |
| :--- | :--- | :--- | :--- |
| Hindi | ~0.05 | ~0.02 | ~0.00 |
| Chinese | ~0.05 | ~0.03 | ~0.00 |
| Russian | ~0.08 | ~0.04 | ~0.00 |
| Spanish | ~0.40 | ~0.30 | ~0.05 |

**Causal Intervention: Cross-lingual Mean Substitution (Llama-3.2-1B)**

| Language | Neuron Set | PPL ratio (target) | PPL ratio (random) |
| :--- | :--- | :--- | :--- |
| English | overlap | 0.95 | 0.99 |
| English | only-native | 1.50 | 0.96 |
| Hindi | overlap | 1.05 | 0.98 |
| Hindi | only-native | 0.31 | 0.97 |

### Ablation Study

**Stability of Units after Word Order Scrambling (Jaccard Similarity)**

| Language Type | Neuron Overlap | SAE Feature Overlap |
| :--- | :--- | :--- |
| Unique Scripts (ZH, JP, TH, KO) | >0.70 | >0.70 |
| Latin Script Languages | ~0.60 | ~0.40-0.60 |
| Cyrillic Script Languages | ~0.65 | ~0.65 |

### Key Findings

-   Romanization leads to almost complete reorganization of language-associated units (Jaccard < 0.1), confirming orthography as the primary driver.
-   Romanized representations align with neither the native script nor English, forming isolated "third subspaces."
-   Word order scrambling results in only minor changes, indicating that units depend on lexical statistics rather than syntactic structure.
-   Script-invariant neurons encode the strongest typological signals; genetic features are accessible early, while phonology emerges late.
-   In causal interventions, ablating script-specific neurons causes catastrophic degradation (language switching), while script-invariant ablation has a mild effect.
-   These patterns consistently replicate across Llama and Gemma models ranging from 1B to 9B parameters.

## Highlights & Insights

-   The experimental design is highly elegant: romanization perturbs surface while keeping content, and scrambling perturbs structure while keeping surface, cleanly disentangling script and structure.
-   The concept of "capacity fragmentation" is significant—models allocate independent internal features for different script variants of the same language, wasting representational capacity. This has direct implications for efficiency optimization in multilingual models.
-   Distinguishing "detectability" from "functional necessity" is a critical methodological contribution—many interpretability works stop at probing, while this study validates findings through causal intervention.

## Limitations & Future Work

-   Analysis focuses on MLP layers and does not cover language-associated patterns in attention heads.
-   Romanization depends on the ICU Transliterator; the quality of transliteration for certain languages may impact conclusions.
-   Only 4 model families were analyzed; applicability to other architectures (e.g., Mistral, Qwen) is unknown.
-   The study does not explore how to utilize these findings to improve multilingual models—for instance, through explicit alignment to reduce capacity fragmentation.

## Related Work & Insights

-   **vs. Tang et al. (2024)**: Tang located language-associated neurons but did not analyze their content; this paper extends localization to interpretation, revealing the dominance of orthography.
-   **vs. Wendler et al. (2024)**: Works supporting the interlingua hypothesis emphasize semantic alignment feasibility; this paper points out that even if semantic alignment is achievable, the representation space remains deeply fragmented by script.
-   **vs. Andrylie et al. (2025)**: Extended LAPE analysis to the SAE level but lacked controlled experiments; this paper provides causal-level evidence via romanization and scrambling tests.

## Rating

-   Novelty: ⭐⭐⭐⭐⭐ First systematic answer to "what language-associated units encode" with elegant design.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × multiple languages × probing + intervention + controls.
-   Writing Quality: ⭐⭐⭐⭐⭐ Clear research questions, tight logical chain, and strong conclusions.
-   Value: ⭐⭐⭐⭐ Important insights for multilingual model design and interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts](structure-guided_entity_resolution_fine-tuning_llms_for_robust_name_matching_in_.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)

</div>

<!-- RELATED:END -->
