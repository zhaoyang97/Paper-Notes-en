---
title: >-
  [Paper Note] Multilingual Language Models Encode Script Over Linguistic Structure
description: >-
  [ACL 2026][multilingual representations] This paper systematically analyzes language-associated units in multilingual LMs using the LAPE metric and sparse autoencoders, finding that these units are primarily driven by orthography (writing system) rather than abstract linguistic structure. Romanization activates nearly entirely disjoint sets of neurons; word-order shuffling has minimal effect; typological information becomes accessible only gradually in deeper layers; and causal interventions reveal that functional importance correlates with surface-form invariance.
tags:
  - ACL 2026
  - multilingual representations
  - writing systems
  - orthography
  - language-associated neurons
  - sparse autoencoders
date: 2026-05-08
content_hash: ec5ab14f6fb085df
---

# Multilingual Language Models Encode Script Over Linguistic Structure

**Conference**: ACL 2026
**arXiv**: [2604.05090](https://arxiv.org/abs/2604.05090)
**Code**: [GitHub](https://github.com/loadthecode0/multilingual-interpretability)
**Area**: Human Understanding / Multilingual Interpretability
**Keywords**: multilingual representations, writing systems, orthography, language-associated neurons, sparse autoencoders

## TL;DR

This paper systematically analyzes language-associated units in multilingual LMs using the LAPE metric and sparse autoencoders, finding that these units are primarily driven by orthography (writing system) rather than abstract linguistic structure. Romanization activates nearly entirely disjoint sets of neurons; word-order shuffling has minimal effect; typological information becomes accessible only gradually in deeper layers; and causal interventions reveal that functional importance correlates with surface-form invariance.

## Background & Motivation

**Background**: Multilingual language models (e.g., Llama, Gemma) compress representations of multiple languages into a shared parameter space, yet the nature of this internal organization remains unclear—specifically, whether it is governed by abstract language identity or surface-form cues.

**Limitations of Prior Work**: (1) Prior work (Tang et al., 2024) identified language-associated neurons via the LAPE metric and demonstrated causal manipulability, but did not address what linguistic properties these neurons actually encode. (2) The *interlingua* hypothesis posits that multilingual models form a unified language-agnostic representation space, yet direct evidence remains insufficient. (3) Bilingual cognition research suggests that comprehension and production may share semantic representations while separating surface-level processing; whether analogous phenomena exist in LMs is unknown.

**Key Challenge**: Although the existence of language-associated units has been established, it remains unclear whether they encode abstract language identity or surface cues such as orthography.

**Goal**: To systematically address four research questions: (i) language vs. writing system—what do language-associated units encode? (ii) robustness to structural perturbation—how does word-order shuffling affect them? (iii) typological alignment—what is their relationship to genealogical, phonological, and syntactic features? (iv) layer-wise organization—how do these properties evolve with depth?

**Key Insight**: Controlled experiments are designed using Romanization (changing the writing system while preserving content) and word-order shuffling (changing structure while preserving surface form) to disentangle the contributions of orthography and linguistic structure.

**Core Idea**: Multilingual LMs organize representations around surface form (writing system); linguistic abstraction emerges layer by layer but never collapses into a unified interlingua.

## Method

### Overall Architecture

Analyses are conducted on four models—Llama-3.2-1B, Llama-3-8B, Gemma-2-2B, and Gemma-2-9B—covering languages across Latin, Cyrillic, Devanagari, Arabo-Persian, and logographic writing systems. LAPE (Language Activation Probability Entropy) is used to identify language-associated units at the raw neuron level, while SAE-LAPE operates in the latent space of sparse autoencoders (SAEs) to identify language-associated features. Four experimental paradigms are employed to address the research questions: Romanization experiments, word-order shuffling experiments, typological probing, and causal interventions.

### Key Designs

1. **Romanization Controlled Experiment**:

    - **Function**: Disentangle the contributions of writing system and language identity to language-associated units.
    - **Mechanism**: For non-Latin languages in the FLORES+ dataset, romanized versions (with and without diacritics) are generated using the ICU Transliterator. LAPE is applied separately to identify language-associated units for each script variant, and Jaccard similarity is used to measure the overlap between units activated by the original script and those activated by the romanized versions. Results show that Hindi in its native script, romanized with diacritics, and romanized without diacritics activate nearly entirely disjoint sets of neurons.
    - **Design Motivation**: If language-associated units encode abstract language identity, they should remain stable under changes in writing system. If they primarily encode orthography, they should reorganize when the script changes.

2. **Word-Order Shuffling Experiment**:

    - **Function**: Test the degree to which language-associated units depend on syntactic structure.
    - **Mechanism**: Evaluation corpora are word-level randomly shuffled, SAE-LAPE is re-applied to identify language-associated units, and Jaccard similarity measures overlap before and after shuffling. Results show that most languages retain a large proportion of language-associated units (overlap >0.7), with languages using distinctive writing systems (Chinese, Japanese, Thai) exhibiting the highest stability.
    - **Design Motivation**: This serves as a contrastive condition to Romanization—if script changes cause dramatic reorganization while word-order changes have negligible effect, this confirms the primacy of surface form over structure.

3. **Typological Probing and Causal Intervention**:

    - **Function**: Examine whether deeper layers encode linguistic structure beyond surface form.
    - **Mechanism**: Linear probes are trained to decode lang2vec typological features (genealogical, phonological, syntactic) from model representations. The subset of neurons invariant across writing systems ("overlap" neurons) is found to carry the strongest typological signal; genealogical features are decodable from shallow layers, while phonological features emerge only at the deepest layers. Causal interventions show that ablating script-invariant neurons causes only mild perplexity increases, whereas ablating script-specific neurons leads to catastrophic degradation (PPL increases by 7.74×), confirming that the latter anchor language identity and surface-form realization.
    - **Design Motivation**: Combining probing and intervention distinguishes *representational accessibility* from *functional necessity*—the fact that information can be probed does not imply it is necessary for generation.

### Loss & Training

This is an analytical study; no training is performed. Pre-trained Top-K SAEs (for Llama models) and JumpReLU SAEs (for Gemma models) are used, with analyses focused on MLP sublayer activations.

## Key Experimental Results

### Main Results

**Overlap of Language-Associated Units After Romanization (Jaccard Similarity, Llama-3.2-1B)**

| Language | Native vs. Romanized (neurons) | Native vs. Romanized (SAE features) | Romanized vs. English |
|----------|-------------------------------|-------------------------------------|----------------------|
| Hindi | ~0.05 | ~0.02 | ~0.00 |
| Chinese | ~0.05 | ~0.03 | ~0.00 |
| Russian | ~0.08 | ~0.04 | ~0.00 |
| Spanish | ~0.40 | ~0.30 | ~0.05 |

**Causal Intervention: Cross-lingual Mean Activation Substitution (Llama-3.2-1B)**

| Language | Neuron Set | PPL ratio (target) | PPL ratio (random) |
|----------|-----------|-------------------|-------------------|
| English | overlap | 0.95 | 0.99 |
| English | only-native | 1.50 | 0.96 |
| Hindi | overlap | 1.05 | 0.98 |
| Hindi | only-native | 0.31 | 0.97 |

### Ablation Study

**Unit Stability After Word-Order Shuffling (Jaccard Similarity)**

| Language Type | Neuron Overlap | SAE Feature Overlap |
|--------------|---------------|-------------------|
| Distinctive scripts (Chinese, Japanese, Thai, Korean) | >0.70 | >0.70 |
| Latin-script languages | ~0.60 | ~0.40–0.60 |
| Cyrillic-script languages | ~0.65 | ~0.65 |

### Key Findings

- Romanization causes nearly complete reorganization of language-associated units (Jaccard < 0.1), confirming orthography as the primary driver.
- After Romanization, representations align neither with the original script nor with English, forming an isolated third subspace.
- Word-order shuffling induces only minor unit-level changes, indicating that language-associated units depend on lexical statistics rather than syntactic structure.
- Script-invariant neurons carry the strongest typological signal; genealogical features are decodable in shallow layers, while phonological features emerge in deeper layers.
- In causal interventions, ablating script-specific neurons causes catastrophic degradation (language switching), whereas ablating invariant neurons has a mild effect.
- The above patterns replicate consistently across Llama and Gemma models at the 1B–9B scale.

## Highlights & Insights

- The experimental design is particularly elegant: Romanization changes surface form while preserving content; word-order shuffling changes structure while preserving surface form. These two orthogonal conditions cleanly disentangle the contributions of orthography and linguistic structure.
- The concept of "capacity fragmentation" has far-reaching implications—models allocate independent internal features to different script variants of the same language, wasting representational capacity. This has direct implications for efficiency optimization in multilingual models.
- Distinguishing *probeability* from *functional necessity* is an important methodological contribution. Much interpretability work stops at probing; this paper goes further by validating findings through causal intervention.

## Limitations & Future Work

- The analysis focuses on MLP sublayers and does not cover language-associated patterns in attention heads.
- Romanization relies on the ICU Transliterator; transcription quality for certain languages may affect conclusions.
- Only four model families are analyzed; generalizability to other architectures (e.g., Mistral, Qwen) is unknown.
- The paper does not explore how findings could be leveraged to improve multilingual models—for example, by explicitly aligning representations to reduce capacity fragmentation.

## Related Work & Insights

- **vs. Tang et al. (2024)**: Tang et al. identified language-associated neurons but did not analyze their encoded content. This paper extends from identification to interpretation, revealing the dominant role of orthography.
- **vs. Wendler et al. (2024)**: Work supporting the interlingua hypothesis emphasizes the achievability of semantic alignment. This paper shows that even if semantic alignment is achievable, the representation space remains deeply fragmented along script boundaries.
- **vs. Andrylie et al. (2025)**: That work extends LAPE analysis to the SAE level but without controlled experiments. This paper provides causal-level evidence through Romanization and shuffling experiments.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic investigation into what language-associated units encode, with elegant experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × multiple languages × probing + intervention + controlled comparisons; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions are clearly stated, reasoning is tightly structured, and conclusions are well-supported.
- Value: ⭐⭐⭐⭐ Important implications for multilingual model design and interpretability research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)

<!-- RELATED:END -->
