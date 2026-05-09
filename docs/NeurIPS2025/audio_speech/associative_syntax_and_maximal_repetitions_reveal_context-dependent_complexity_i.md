---
title: >-
  [Paper Note] Associative Syntax and Maximal Repetitions Reveal Context-Dependent Complexity in Fruit Bat Communication
description: >-
  [NeurIPS 2025 (Workshop on AI for Non-Human Animal Communication)][Audio & Speech][animal communication] This paper proposes an unsupervised approach for inferring discrete units, grammar types, and temporal structure from fruit bat vocalizations, and introduces Maximal Repetitions (MRs) to animal communication research for the first time, finding that communicative complexity is significantly higher in conflict contexts than in affiliative ones.
tags:
  - NeurIPS 2025 (Workshop on AI for Non-Human Animal Communication)
  - "Audio & Speech"
  - animal communication
  - fruit bat vocalizations
  - unsupervised clustering
  - associative syntax
  - maximal repetitions
date: 2026-05-08
content_hash: 9746bc17858f4057
---

# Associative Syntax and Maximal Repetitions Reveal Context-Dependent Complexity in Fruit Bat Communication

**Conference**: NeurIPS 2025 (Workshop on AI for Non-Human Animal Communication)  
**arXiv**: [2512.01033](https://arxiv.org/abs/2512.01033)  
**Code**: [https://github.com/gg4u/decodingNonHumanCommunication](https://github.com/gg4u/decodingNonHumanCommunication)  
**Area**: Audio & Speech / Animal Communication  
**Keywords**: animal communication, fruit bat vocalizations, unsupervised clustering, associative syntax, maximal repetitions

## TL;DR

This paper proposes an unsupervised approach for inferring discrete units, grammar types, and temporal structure from fruit bat vocalizations, and introduces Maximal Repetitions (MRs) to animal communication research for the first time, finding that communicative complexity is significantly higher in conflict contexts than in affiliative ones.

## Background & Motivation

**State of the Field**: Quantifying communicative complexity in graded vocal systems is a central challenge in animal communication research. Existing unsupervised annotation methods, such as the manifold-learning clustering approach of Sainburg et al., assume discrete vocal boundaries and are suited to discrete vocalization systems. The bat grammar analysis of Zhang et al. relies on expert-annotated syllable ground-truth labels, limiting scalability.

**Limitations of Prior Work**: (1) In graded vocalization systems (e.g., fruit bats, mice, and even human phonemes), time-frequency features overlap across syllable types, degrading the performance of existing clustering methods. (2) The Social Complexity Hypothesis for Communicative Complexity (SCHCC) recommends information-theoretic metrics, but measures such as Shannon entropy fail to capture long-range dependencies and combinatorial capacity. (3) Existing methods cannot effectively distinguish associative from combinatorial grammar across multiple behavioral contexts.

**Root Cause**: How can a limited repertoire of vocal units encode complex information? This problem is analogous to nucleotide encoding of protein information in genetics. Existing metrics are insufficient to capture such combinatorial complexity.

**Paper Goals**: (1) How does dimensionality reduction affect unsupervised clustering quality in graded vocal systems? (2) How do grammar and temporal structure encode behavioral context information?

**Starting Point**: The paper borrows the concept of Maximal Repetitions (MRs) from computational linguistics — MR length scaling is mathematically related to block entropy and follows a power-law distribution in natural language, reflecting strong long-range dependencies. This tool is introduced to animal communication research for the first time.

**Core Idea**: By improving unsupervised manifold-learning clustering to infer the fruit bat vocal repertoire, and employing MRs as a novel metric, the paper reveals that communication in conflict scenarios is more complex than in affiliative ones.

## Method

### Overall Architecture

The method comprises two experimental modules. The first focuses on repertoire inference: mel spectrograms of fruit bat vocalizations are processed through dimensionality reduction and manifold learning, followed by clustering, yielding unsupervised labels for each vocal unit. The second focuses on grammar type and temporal structure analysis: vocalizations are encoded as syllable sequences, and behavioral classifiers, statistical tests, and MR extraction are applied to analyze communicative patterns across behavioral contexts. Data are drawn from the fruit bat vocalization dataset annotated by Prat et al., comprising 41 individuals across 8 behavioral contexts (mating protests, fighting, threats, biting, feeding, grooming, kissing, and isolation/mother–pup interactions).

### Key Designs

1. **Improved Unsupervised Syllable Clustering Pipeline**:

    - Function: Automatically infer discrete syllable types and repertoire size from continuous graded vocalizations.
    - Mechanism: The pipeline follows Sainburg et al.'s UMAP + HDBSCAN framework but systematically varies the input spectrogram dimensionality. A key finding is that coarse-graining along the time dimension substantially improves clustering. Three axes of exploration are considered: spectrogram settings (exploring extreme time–frequency trade-offs), PCA dimensionality reduction (applied to latent representations from various autoencoder architectures), and dynamic threshold segmentation (dynamically estimating the noise floor to isolate shorter sub-units).
    - Design Motivation: The original method distinguishes only 2 vocalization classes on fruit bat data (isolation vs. non-isolation); the improved pipeline identifies 7 syllable types. Time-axis compression is effective because information in graded systems is encoded in continuous acoustic modulation.

2. **Associative vs. Combinatorial Grammar Classification**:

    - Function: Determine the grammar type of fruit bat vocalizations — whether information is determined by syllable composition or by syllable order.
    - Mechanism: A random forest classifier is trained on 18 syllable sequence features (including syllable richness, sequence length, transition counts, and sequence entropy) to classify behavioral context. Key test: if classification $F_1 > 0.9$ is preserved after random permutation of sequences, the grammar is associative. Results show $F_1 > 0.9$ both before and after permutation.
    - Design Motivation: Grammar type determines the complexity ceiling of a communicative system. Associative grammar implies that information resides in *which* syllables are used rather than their *order*.

3. **Maximal Repetitions (MRs) Analysis**:

    - Function: Quantify combinatorial complexity of vocalization sequences across behavioral contexts.
    - Mechanism: A prefix–suffix tree algorithm extracts the longest recurrently occurring sub-sequences from syllable sequences. Likelihood-ratio tests determine the distribution type of MR lengths: an exponential distribution implies a simple memoryless decay process, while a heavy-tailed distribution (power law) implies long-range dependencies. Results indicate a truncated power-law distribution ($\alpha = 1.79$). Syllable transition networks are further constructed, and the small-world coefficient $\omega$ and mean clustering coefficient are computed.
    - Design Motivation: MRs are used in computational linguistics to analyze text information compression properties; their length distribution relates to the Hilberg conjecture (sublinear block entropy growth → strong long-range dependence and high compressibility in language). Introducing this tool to animal communication provides a complexity measure that goes beyond Shannon entropy.

### Evaluation Strategy

Clustering evaluation employs a two-tier strategy: internal validation uses the silhouette coefficient; external validation uses agglomerative clustering with a pairwise DTW + MFCC distance matrix (quantile distance threshold $q = 0.05$) as a proxy ground truth, with ARI and NMI measuring agreement. Behavioral difference testing uses the Wilcoxon rank-sum test. MR distribution type testing uses likelihood-ratio tests.

## Key Experimental Results

### Main Results

| Metric | Value |
|--------|-------|
| Syllable types identified (improved) | 7 (baseline: 2) |
| Clustering silhouette coefficient | > 0.5 |
| Assignment accuracy | 95% |
| DTW proxy label syllable types | 27 ± 2 / vocalizer |
| ARI (vs. proxy labels) | 0.12 ± 0.01 |
| NMI | 0.30 ± 0.01 |
| Total inferred syllable types (HDBSCAN) | 14 |

### Ablation Study (Behavioral Complexity Comparison)

| Behavioral Context | MR Length Trend | Small-World Coeff. ω | Mean Clustering Coeff. | Network Density |
|-------------------|----------------|----------------------|------------------------|----------------|
| Mating protests | Longest (heavy-tailed) | ≈0.00 | 0.62 | 0.81 |
| Fighting | Longer | 0.03 | 0.44 | 0.26 |
| Threats | Longer | 0.10 | 0.35 | 0.18 |
| Biting | Moderate | 0.05 | 0.46 | 0.40 |
| Feeding | Shorter | 0.53 | 0.13 | 0.15 |
| Grooming | Shorter | 0.65 | 0.09 | 0.11 |
| Kissing | Shorter | 0.63 | 0.12 | 0.13 |
| Isolation | Short / simple repetition | — | 0.00 | 0.10 |

### Key Findings

- **Associative Grammar**: Permutation tests confirm that fruit bat vocalizations exhibit associative grammar (order does not affect context classification), consistent with domain expert priors.
- **Context-Dependent Syllable Use**: Syllable distributions in isolation (mother–pup interaction) contexts differ significantly from all other contexts (Wilcoxon, $p < 0.05$).
- **Heavy-Tailed MR Distribution**: The exponential distribution hypothesis is rejected ($p < 0.05$); the truncated power law with $\alpha = 1.79$ indicates long-range temporal structure encoding combinatorial complexity.
- **Conflict > Affiliative Complexity**: Conflict behaviors (mating protests, fighting, threats) yield longer MRs and small-world network topology ($\omega \approx 0$); affiliative behaviors (feeding, grooming, kissing) yield more random networks ($\omega > 0.5$).
- **Distinctiveness of Isolation Context**: Mother–pup interactions are dominated by simple repetitions of specific syllables, reflecting immature vocalization patterns.

## Highlights & Insights

- **First Application of MRs to Animal Communication**: Provides a new means of measuring communicative combinatorial complexity without relying on traditional information-theoretic metrics, generalizable to other species.
- **"Disagreement Requires More Complex Signals"**: The finding that conflict communication is more complex can be interpreted as low compressibility of information when expressing disagreement — an explanation with cross-species generality.
- **Counterintuitive Finding on Temporal Coarse-Graining**: Reducing the temporal resolution of spectrograms actually improves clustering quality in graded systems, revealing the nature of information encoding in temporal modulation.
- **Small-World Networks Align with Behavioral Type**: The topological structure of syllable transition networks (small-world vs. random) corresponds closely to conflict vs. affiliative behavior types.

## Limitations & Future Work

- Validation on a single species (fruit bats); cross-species generalizability of the MR method requires testing.
- The "conflict" and "affiliative" labels represent the authors' subjective interpretation of behavioral annotations in the original dataset.
- Agreement between clustering and proxy labels is low (ARI = 0.12); different clustering methods may affect downstream conclusions.
- As a workshop paper, scope and experimental scale are limited.
- The relationship between MR content and specific semantics is not explored in depth.

## Related Work & Insights

- The manifold-learning clustering of Sainburg et al. serves as the foundational method; this paper improves its performance on graded systems through dimensionality reduction strategies.
- The bat behavioral classifier of Zhang et al. is adapted for unsupervised labels and multi-behavior classification.
- Dębowski's MR theory links MR scaling to the Hilberg conjecture, motivating its application in animal communication.
- Sainburg's cross-species information decay study demonstrates that birdsong and human speech exhibit similar exponential/power-law decay patterns for short- and long-range sequences.

## Rating

⭐⭐⭐ (3/5)

As a workshop paper, the contribution lies in the methodological cross-disciplinary transfer from computational linguistics to animal communication. The core finding — that conflict communication is more complex — is plausible but not surprising. The introduction of MRs as a complexity metric is the primary highlight. The experimental scale is limited, and cross-species validation is needed to confirm the generality of the metric.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[NeurIPS 2025\] Merlin L48 Spectrogram Dataset](merlin_l48_spectrogram_dataset.md)

<!-- RELATED:END -->
