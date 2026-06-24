---
title: >-
  [Paper Note] SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists
description: >-
  Proposes the SPOT system, which fine-tunes LLaMA 3 to translate natural language scene descriptions into YAML queries, combining this with a semantic tag bundling mechanism to enable reliable natural language access to OpenStreetMap data, serving geolocation verification in investigative journalism.
tags:

date: 2026-05-08
content_hash: 0a254a88d114586f
---

# SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists

- **Conference**: ACL 2025
- **arXiv**: [2506.13188](https://arxiv.org/abs/2506.13188)
- **Code**: [GitHub](https://github.com/dw-innovation/kid2-spot) | [Demo](https://www.findthatspot.io/)
- **Area**: Other
- **Keywords**: Natural Language Interface, OpenStreetMap, Geolocation Verification, Investigative Journalism, LLM Fine-Tuning, YAML Query

## TL;DR

Proposes the SPOT system, which fine-tunes LLaMA 3 to translate natural language scene descriptions into YAML queries, combining this with a semantic tag bundling mechanism to enable reliable natural language access to OpenStreetMap data, serving geolocation verification in investigative journalism.

## Background & Motivation

- **Core Problem**: OpenStreetMap (OSM) is a crucial resource for investigative journalists conducting geolocation verification, but its query language, OverpassQL, poses a high barrier to entry for non-technical users.
- **Limitations of Prior Work**:
    - **Overpass Turbo**: Requires mastery of OverpassQL syntax, making it difficult for non-technical users.
    - **GeoGuessr GPT**: Based on ChatGPT but is not open-source and does not connect to the OSM database.
    - **GeoSpy**: Only accepts image input and does not support natural language.
    - **EarthKit**: Requires users to manually select OSM tags, still presenting a technical barrier.
    - **OverpassT5 (Staniek et al.)**: Directly generates OverpassQL but requires users to understand the OSM tag schema.
- **Design Motivation**: To build a fully open-source, reliable, and accurate OSM geospatial search tool for investigative journalists that supports unstructured natural language input.

## Method

### Overall Architecture

SPOT consists of four core components: (1) OSM tag bundling construction and indexing → (2) synthetic training data generation → (3) LLaMA 3 model fine-tuning → (4) inference and post-processing. User inputs a natural language description → the model outputs a YAML query → semantic search replaces concepts with OSM tags → PostGIS database query → interactive map displays the results.

### Key Designs

1. **Multi-layer Intermediate Representation (YAML)**: Instead of directly generating OverpassQL, the model first generates a YAML structured query without OSM tags (containing search area, entities, properties, and spatial relationships), and then maps the entity names to OSM tag bundles via a semantic search engine. This decoupled design avoids the need to retrain the model when OSM tags are updated.
2. **Semantic Tag Bundling System**: Groups visually similar OSM tags (e.g., light rail / subway / tram into the same bundle) and combines BM25 + SBERT hybrid retrieval to handle typos and synonyms in user queries.
3. **Synthetic Training Data Pipeline**: Generates 43,976 training samples by randomly combining YAML field values + 7 personas + 5 writing styles + GPT-4o, covering real-world scenarios such as typos, grammatical errors, non-Latin alphabets, and ambiguous spatial terms.

### Loss & Training

Fine-tunes LLaMA 3 using LoRA (rank=32, alpha=64) with a learning rate of 1e-5, weight decay of 0.01, and early stopping patience of 10.

## Key Experimental Results

### Main Results (195 Real-World User Query Benchmark)

| Model | Adaptation | Area | Entity | Entity* | Property | Relation |
|------|---------|------|--------|---------|----------|----------|
| GPT-4o | Zero-shot | 88.14 | 2.28 | 90.21 | 3.03 | 9.8 |
| GPT-4o | One-shot | 89.18 | 1.13 | 92.03 | 10.96 | 11.11 |
| **Mistral** | Adapter | **93.33** | **82.54** | 95.01 | **56.58** | 45.45 |
| LLaMA 3 | Adapter | 92.31 | 81.41 | **96.15** | 50.00 | 48.05 |
| Qwen2.5 | Adapter | 92.31 | 82.31 | 95.69 | 51.95 | **52.60** |
| Phi | Adapter | 92.82 | 79.59 | 94.10 | 53.33 | 53.90 |
| mT5 | Adapter | 88.21 | 72.34 | 90.02 | 48.89 | 37.01 |

### Ablation Study (Hallucination Comparison: GPT-4o vs. Fine-tuned Models)

| Model | Entity Omission | Entity Hallucination | Property Omission | Property Hallucination |
|------|---------|---------|---------|---------|
| GPT-4o (0-shot) | 48 | 37 | 53 | — |
| Fine-tuned LLMs | Significantly Reduced | Significantly Reduced | Significantly Reduced | — |

### Key Findings

1. **Fine-tuned Small Models Far Outperform GPT-4o Zero/Few-shot**: GPT-4o achieves only 2.28% in entity recognition (zero-shot), whereas the fine-tuned Mistral reaches 82.54%, indicating that the OSM tag schema requires domain adaptation.
2. **Synthetic Data Pipeline is Effective**: 43K synthetic samples cover various real-world user input patterns (typos, non-Latin alphabets, ambiguous spatial terms), making the fine-tuned model robust.
3. **YAML Intermediate Representation Outperforms Direct OverpassQL Generation**: The decoupled design allows the tag system to be updated independently, and YAML syntax is more fault-tolerant than JSON.
4. **Properties and Relations Remain Challenging**: Even for the best fine-tuned model, property accuracy (~56%) and relation accuracy (~53%) still have significant room for improvement.

## Highlights & Insights

- The first full-stack open-source natural language geospatial search system oriented towards investigative journalism, which has been deployed in production.
- Innovative multi-layer intermediate representation design (YAML → semantic search → OSM tags), decoupling language understanding and tag mapping.
- The synthetic data pipeline design is derived from user research within the professional OSINT community, covering real-world scenarios such as typos, multilingual queries, and ambiguous spatial terms.
- Fine-tuned open-source small models (LLaMA 3 8B) significantly outperform GPT-4o zero/few-shot across core metrics.
- Model weights and the training pipeline are fully open-sourced, enabling deployment on private infrastructure to meet the security requirements of news organizations.

## Limitations & Future Work

- The recognition accuracy of properties and spatial relations remains low (~50-56%), and queries with complex multi-entities and multi-relations may fail.
- Relies on the completeness and coverage of OSM data, which can be sparse in certain developing regions.
- The tag bundling list is statically hand-crafted, requiring manual maintenance and updates for newly emerging geospatial feature types.
- Evaluated on only 195 test queries, which is a relatively small scale and may not fully cover all real-world scenarios.
- Training data is fully generated by a synthetic pipeline, which may lead to distribution shift relative to real user queries.
- The end-to-end geolocation success rate (i.e., whether users can actually find the target location) was not evaluated.

## Related Work & Insights

- **Text-to-SQL**: DAIL-SQL (Gao et al. 2024), MCS-SQL (Lee et al. 2025), Jang et al. 2023 (T5 adapter tuning), Zhang et al. 2024 (LLaMA adapter)
- **OSM Query**: OverpassT5 (Staniek et al. 2024, direct OverpassQL generation), Lawrence & Riezler 2016 (semantic parsing intermediate representation), Will 2021
- **Geospatial Tools**: Overpass Turbo (native OQL), GeoSpy (image input, closed-source), EarthKit (semi-structured, manually selected tags), GeoGuessr GPT (ChatGPT wrapper, closed-source)
- **LLM Fine-Tuning**: LoRA (Hu et al.), Unsloth accelerated training, LLaMA 3 (Touvron et al.)
- **Semantic Retrieval**: SBERT (Reimers & Gurevych 2019), Elasticsearch hybrid retrieval (BM25 + vector)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel system design combining multi-layer intermediate representation and semantic tag bundling
- **Value**: ⭐⭐⭐⭐⭐ — Already deployed in production and fully open-sourced, directly serving investigative journalists
- **Experimental Thoroughness**: ⭐⭐⭐ — Small test set, with room for improvement in some metrics
- **Overall**: ⭐⭐⭐⭐

<!-- SPOT: open-source NL-to-OSM geospatial search tool for investigative journalism -->
<!-- Fine-tuned LLaMA 3 with synthetic data pipeline and semantic tag bundling -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)
- [\[ACL 2025\] Developmentally-plausible Working Memory Shapes a Critical Period for Language Acquisition](developmentally-plausible_working_memory_shapes_a_critical_period_for_language_a.md)

</div>

<!-- RELATED:END -->
