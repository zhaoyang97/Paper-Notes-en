---
title: >-
  [Paper Note] Why Isn't Relational Learning Taking Over the World?
description: >-
  [AAAI 2026][Relational Learning] This position paper systematically analyzes why relational learning has failed to dominate the AI landscape, identifying core issues including unrealistic datasets…
tags:
  - "AAAI 2026"
  - "Relational Learning"
  - "Knowledge Graphs"
  - "Statistical Relational AI"
  - "Evaluation Methodology"
  - "Entity Prediction"
date: 2026-05-08
content_hash: 2909a9418764b53f
---

# Why Isn't Relational Learning Taking Over the World?

**Conference**: AAAI 2026
**arXiv**: [2507.13558](https://arxiv.org/abs/2507.13558)  
**Code**: None  
**Area**: Other (Relational Learning / Knowledge Graphs)
**Keywords**: Relational Learning, Knowledge Graphs, Statistical Relational AI, Evaluation Methodology, Entity Prediction

## TL;DR

This position paper systematically analyzes why relational learning has failed to dominate the AI landscape, identifying core issues including unrealistic datasets, fundamentally flawed evaluation methodologies, the absence of negative examples, and theoretical difficulties with aggregation operations. It further delineates the key improvements necessary for relational learning to realize its potential.

## Background & Motivation

1. **Background**: Contemporary AI is dominated by text and image models (e.g., GPT, Stable Diffusion), which model pixels, tokens, and phonemes. Yet the real world consists of entities (objects, events), their attributes, and their relations—not the surface forms of these media.

2. **Limitations of Prior Work**: Relational learning (also known as Statistical Relational AI) studies how to learn predictive models from entities, attributes, and relations. Nearly all of the most valuable corporate data resides in spreadsheets and relational databases—filled with product IDs, student IDs, transaction numbers, and similar identifiers—rather than in text or image form. Nevertheless, relational learning has not received commensurate attention in industry or academia.

3. **Key Challenge**: Relational data constitutes the most pervasive and valuable form of data, yet relational learning research and application remain far below their potential. The reasons extend beyond technical difficulty to systemic problems in dataset construction, evaluation methodology, and fundamental modeling assumptions within the research community.

4. **Goal**: To diagnose the underlying reasons why relational learning has not entered the mainstream and to chart directions for future research.

5. **Key Insight**: A systematic critical analysis along four dimensions: datasets, training methodology, evaluation metrics, and forward-looking requirements.

6. **Core Idea**: For relational learning to fulfill its potential, the field requires more realistic datasets, more principled evaluation methods, correct treatment of missing data and negative examples, and probabilistic predictions oriented toward downstream decision-making.

## Method

### Overall Architecture

Rather than proposing a specific algorithm, this paper conducts a systematic critical examination of the relational learning field. The analysis proceeds through the following core themes: the intrinsic characteristics of relational data, knowledge graph representations, problems with standard datasets, strategies for handling absent negative examples during training, deficiencies in existing evaluation methods, and future research directions.

### Key Designs (Core Arguments)

1. **Fundamental Problems with Datasets**:
    - Argument: Standard datasets (e.g., FB15k, WN18) deviate substantially from realistic settings.
    - Core Analysis: FB15k retains only entities and relations appearing more than 100 times, filtering out all reified entities. Yet in Wikidata, over 98% of entities appear as subjects in fewer than 10 triples. Large-scale data typically implies more entities with fewer triples per entity, yielding a long-tail distribution. Methods trained and evaluated on such heavily filtered datasets cannot generalize to more realistic scenarios.
    - Insight: FB15k-237 removes inverse relations to test whether methods can exploit patterns beyond reciprocal relations; however, it and FB15k assess different capabilities and should not be treated as simply "better."

2. **The Challenge of Missing Negative Examples During Training**:
    - Argument: The absence of negative examples under the open-world assumption is a fundamental challenge for relational learning.
    - Core Analysis: On a dataset containing only positive examples, the optimal prediction under log loss is "everything is true"—yielding zero training loss but infinite test loss. This is typically mitigated via contrastive learning (adding random triples as negatives), but the choice of the negative-to-positive ratio amounts to "fabricated inputs" that, unlike prior probabilities, are not overridden by observed data. Two key properties distinguish relational models from tabular models: parameter sharing/weight tying and aggregation operations.
    - Insight: Estimating probabilities without negative examples is a fundamental problem that cannot be resolved without external meta-information.

3. **Systematic Deficiencies in Evaluation Methods**:
    - Argument: The prevailing ranking-based evaluation metrics (MRR, hit@k) exhibit multiple fundamental flaws.
    - Core Analysis: (1) They cannot handle queries with no valid answer (e.g., "Who is the Pope's spouse?"); (2) the query itself leaks information about the test set; (3) some queries are trivially easy (e.g., predicting a football team's location achieves near-perfect hit@10); (4) some queries are nearly impossible to answer correctly (e.g., predicting which team has a particular forward—even an omniscient oracle could not reliably guess); (5) rankings discard actual probability information—an overconfident predictor and an appropriately uncertain one may produce identical rankings; (6) evaluation is divorced from downstream decision-making tasks.
    - Insight: State-of-the-art methods achieve only approximately 55.8% hit@10 on FB15k-237, which is of limited utility for real-world tasks where correctness cannot be easily verified.

### The Aggregation Problem

Aggregation operations (e.g., predicting gender from movies a user has watched) represent the Achilles' heel of relational learning. Existing methods either assume that related entities provide independent evidence (e.g., noisy-or, sum, logistic regression) or effectively reduce to treating only a single related entity (e.g., max, average, attention). Determining whether evidence is independent is extremely difficult, and model behavior is often unreasonable when the number of related entities ranges from zero to infinity.

### Forward-Looking Directions

- Realistic public datasets (e.g., government environmental data) are needed, rather than data released simply because "no one cares about it."
- Probabilistic predictions paired with utility functions are needed to support downstream decision-making.
- Explicit modeling of the reasons for data missingness (non-random missingness) is required.
- Entity prediction must account for three answer types: known entities, unrepresented entities, and no entity.
- A distinction must be drawn between learning general knowledge and learning the properties of specific entities.
- Embedding/latent feature dimensionality should match entity complexity rather than be fixed.
- The ultimate goal is joint modeling across multiple heterogeneous datasets—which is, in essence, science.

## Key Experimental Results

This paper is a position paper and includes no new experiments. The following key figures are cited:

| Aspect | Data | Note |
|--------|------|------|
| Wikidata scale | ~1.65B triples, 117M items | Large-scale knowledge graph |
| Entity sparsity | 98%+ of entities have <10 triples | Standard datasets filter these out |
| FB15k-237 SOTA | hit@10 ≈ 55.8% | Far from reliable for practical applications |
| Random triple accuracy | >99.9999999995% | Nearly all random triples are false — accuracy is not a meaningful metric |

### Key Findings

- The evaluation framework for relational learning is severely misaligned with the requirements of practical applications.
- The construction methodology of standard benchmark datasets (filtering low-frequency entities) systematically simplifies the problem.
- Knowledge graphs are considerably more heterogeneous than graph learning benchmarks; the only universal structure is that introduced by reification.
- Missing data is not missing at random—the majority of facts are absent from knowledge bases.

## Highlights & Insights

- **Incisive Critical Analysis**: Concise, well-chosen examples (the Pope's spouse, predicting a football team's location) expose the absurdity of prevailing evaluation methods.
- **Wikidata Illustrations**: The use of concrete instances (e.g., Christine Sinclair) effectively grounds abstract concepts.
- **Theoretical Analysis of Aggregation**: The independence assumption and model behavior in the limit of infinite related entities constitute a profound and central difficulty in relational learning.
- **Interdisciplinary Perspective**: The paper connects the ultimate goals of relational learning to scientific methodology—jointly modeling heterogeneous data and constructing revisable hypotheses.
- The necessity of probabilistic prediction combined with utility theory for downstream decision-making is made explicit.

## Limitations & Future Work

- As a position paper, no concrete methodological validation is provided for the proposed directions.
- The potential of LLMs in relational reasoning receives insufficient discussion—can LLMs partially address some of the difficulties in relational learning?
- Coverage of recent knowledge graph embedding methods that incorporate text (e.g., KG-BERT, StAR) is limited.
- The proposed direction of "jointly modeling multiple heterogeneous datasets," while ambitious, lacks concrete and actionable pathways.

## Related Work & Insights

- **vs. Knowledge Graph Embeddings (TransE, etc.)**: Fixed-size embeddings are theoretically unjustified—the United States and a player–team relation should not share the same embedding dimensionality.
- **vs. Graph Neural Networks**: Graph learning and relational learning address isomorphic problems, yet the structure of typical relational databases differs substantially from graph benchmarks.
- **vs. LLMs/Generative AI**: Current AI models perception or description (words, pixels) rather than entities and relations themselves.

## Rating

- **Novelty**: ⭐⭐⭐⭐ While some arguments have precedents in the community, their systematic integration and the constructive directions proposed are valuable.
- **Experimental Thoroughness**: ⭐⭐ Position paper; no experimental validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Distinctive style—humorous yet incisive, with carefully chosen examples and exceptionally strong readability.
- **Value**: ⭐⭐⭐⭐ Offers important reflective value for the relational learning community, though its impact on the broader AI community may be limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICML 2026\] Knowing Isn't Understanding: Re-Grounding Generative Proactivity with Epistemic and Behavioral Insight](../../ICML2026/others/knowing_isnt_understanding_re-grounding_generative_proactivity_with_epistemic_an.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](../../NeurIPS2025/others/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](../../CVPR2026/others/crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[AAAI 2026\] Bandit Learning in Housing Markets](bandit_learning_in_housing_markets.md)

</div>

<!-- RELATED:END -->
