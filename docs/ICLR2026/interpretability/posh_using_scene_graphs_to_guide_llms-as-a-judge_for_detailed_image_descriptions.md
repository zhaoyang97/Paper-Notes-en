---
title: >-
  [Paper Note] PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions
description: >-
  [ICLR 2026][Interpretability][detailed image description] This paper proposes PoSh, an evaluation metric that extracts scene graphs $G(d) = \langle O(d), E(d)…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "detailed image description"
  - "scene graph"
  - "LLM-as-Judge"
  - "fine-grained evaluation"
  - "assistive text"
date: 2026-05-08
content_hash: c28bab0e1cd27873
---

# PoSh: Using Scene Graphs to Guide LLMs-as-a-Judge for Detailed Image Descriptions

**Conference**: ICLR 2026
**arXiv**: [2510.19060](https://arxiv.org/abs/2510.19060)
**Code**: [GitHub](https://github.com/amith-ananthram/posh)
**Area**: Interpretability
**Keywords**: detailed image description, scene graph, LLM-as-Judge, fine-grained evaluation, assistive text

## TL;DR
This paper proposes PoSh, an evaluation metric that extracts scene graphs $G(d) = \langle O(d), E(d), K(d) \rangle$ from both generated and reference descriptions as structured rubrics, guiding an open-source 14B LLM (Qwen3-14B) to perform QA-based fine-grained error localization. PoSh surpasses GPT-4o-as-Judge by +0.05 Spearman ρ on the DOCENT artwork benchmark and CapArena, while remaining fully reproducible.

## Background & Motivation

**Background**: VLMs are capable of generating detailed image descriptions (100–300 words), yet evaluation methods lag significantly behind. CIDEr/SPICE were designed for short texts; LLM-as-Judge approaches are non-reproducible and produce coarse-grained, uninterpretable scores.

**Limitations of Prior Work**:
- Attribute/relation misattachment in long descriptions is a core error type (e.g., "a man pouring water" described as "a man in the center"), yet existing metrics are insensitive to this
- SPICE/CAPTURE use scene graphs but ignore object attachment, leading to inflated scores
- Closed-source LLM evaluation (GPT-4o) is costly and non-reproducible; open-source LLM-as-Judge (LLaVA-Critic) does not provide interpretable fine-grained scores
- Benchmarks with fine-grained human judgments are lacking — most detailed description benchmarks have no human annotations

**Key Challenge**: There is a need for evaluation methods that are simultaneously cheap, reliable, and interpretable, yet these properties are typically in tension.

**Goal**: To jointly achieve interpretability (fine-grained error localization to text spans), high correlation with human judgments, and full open-source reproducibility.

**Key Insight**: Scene graphs reduce the surface-form diversity of descriptions to visual components (entities + attributes + relations), serving as a structured checklist for an LLM-Judge, where each component is independently verified for presence and scores are aggregated into coarse-grained metrics.

**Core Idea**: Scene graphs structure *what* to evaluate (entities, attributes, relations), while LLM-QA flexibly handles *how* to compare (surface-form variation).

## Method

### Overall Architecture

PoSh operates in three steps:
1. **Scene Graph Extraction**: Sentence-level scene graphs are extracted from both generated and reference descriptions using dependency parsing (spaCy) and coreference resolution (Maverick), then merged into a complete scene graph.
2. **Fine-Grained Scoring**: Each component in the scene graph is converted into a templated question, and Qwen3-14B performs QA to verify its presence in the counterpart text.
3. **Coarse-Grained Aggregation**: Mistakes scores (generated → reference) and omissions scores (reference → generated) are each averaged separately.

### Key Designs

1. **Attachment-Preserving Scene Graph Extraction**:
    - Function: Extracts a structured representation $G(d) = \langle O(d), E(d), K(d) \rangle$ from descriptive text, where $O$ is the entity set, $E \subseteq O \times A$ is the attribute edge set, and $K \subseteq O \times R \times O$ is the relation edge set.
    - Mechanism: Sentence-level dependency parsing → cross-sentence coreference resolution for entity merging → retention of attachment links from each attribute/relation to its host entity → localization of each component to the source text span.
    - Design Motivation: SPICE's disregard for attachment links allows misattributed properties (e.g., assigning A's attribute to B) to go unpunished. PoSh preserves attachment chains to ensure the correct entity identifier is used when verifying attributes and relations.

2. **Three-Round QA Verification with Unique Identifiers**:
    - Function: Generates templated questions for each scene graph component and uses an LLM to judge its presence in the counterpart text (scored 1–5).
    - Mechanism: Collisions among entities of the same type (e.g., multiple "man" instances) require unique identifiers. Verification proceeds in three rounds: (1) top-level entity ("man" itself) → (2) part-of/subordinate entity ("face of the man") → (3) attributes/relations (using the simplest confirmed identifier). Identifier candidates include class names, surface forms, attribute modifiers, and relational modifiers, rewritten into natural expressions by the LLM.
    - Design Motivation: Avoids forcing alignment between scene graph components from two descriptions — the counterpart text may refer to the same object using entirely different words (e.g., the reference uses "trio" while the generation mentions three individuals separately).

3. **Interpretable Coarse-Grained Aggregation**:
    - Function: Aggregates per-component fine-grained scores into three dimensions: mistakes, omissions, and overall.
    - Mechanism: $\text{Mistakes} = \text{mean}_{c \in O(\text{gen})}(\pi(c))$, $\text{Omissions} = \text{mean}_{c \in O(\text{ref})}(\rho(c))$, where $\pi(c) = \Psi(c_{\text{gen}}, \text{ref})$ and $\rho(c) = \Psi(c_{\text{ref}}, \text{gen})$.
    - Design Motivation: Coarse-grained scores are derived directly from the mean of fine-grained scores — given a total score, one can trace back to which attributes of which entities caused errors, providing diagnostic capability.

### Loss & Training

PoSh is an inference-time metric with no training process. The QA scorer Ψ uses Qwen3-14B; presence scores are extracted as weighted averages over token logits (1–5), with an entity presence threshold of 2 (tuned on a small validation set). Runtime efficiency: 400 samples in 15 minutes on a single H100 GPU (~2 seconds each), compared to over 2 hours for DCScore due to its GPT-4 dependency.

## Key Experimental Results

### DOCENT Benchmark — Coarse-Grained Metric Comparison

| Metric | Parameters | Mistakes ρ | Omissions ρ | Overall ρ | Reproducible |
|--------|-----------|-----------|------------|----------|-------------|
| SPICE | - | 0.308 | 0.464 | 0.458 | ✓ |
| CAPTURE | - | 0.259 | 0.447 | 0.453 | ✓ |
| LLaVA-Critic | 72B | 0.412 | 0.509 | 0.546 | ✓ |
| DCScore | GPT-4o | **0.541** | 0.395 | 0.471 | ✗ |
| GPT-4o (ref+img) | - | 0.484 | 0.380 | 0.510 | ✗ |
| **PoSh** | **14B** | 0.519 | **0.581** | **0.599** | **✓** |

### Fine-Grained Metric Comparison (DOCENT)

| Method | Mistakes F1 | Omissions F1 |
|--------|------------|-------------|
| Random | 0.503 | 0.499 |
| 4GramEmbed | 0.483 | 0.641 |
| SGEmbed | 0.514 | 0.658 |
| **PoSh** | **0.580** | **0.680** |

### Key Findings
- PoSh achieves an overall accuracy of 70.7% on DOCENT, surpassing GPT-4o (67.3%) and GPT-5 text-only (68.0%), while being fully open-source and reproducible.
- On CapArena, PoSh's model ranking correlation with human judgments on complex scenes (≥3 persons) outperforms the 72B LLaVA-Critic (ρ=0.727 vs. 0.686).
- Scene graph subcomponent validation: element extraction F1=0.892, element verification F1=0.852 — high-quality structured extraction is the foundation of PoSh's success.
- Using PoSh as an RL reward function (DAPO) outperforms SFT: omission improvement +0.432, overall improvement +0.135.
- The DOCENT leaderboard reveals that open-source models are competitive on mistakes but lag substantially behind closed-source models on omissions — coverage is the key gap.

## Highlights & Insights
- **Scene Graphs as Structured Rubrics**: The design leverages the structured dimensionality reduction of scene graphs (reducing surface-form diversity of evaluation targets) while maintaining flexibility through LLM-QA (avoiding forced alignment) — the two approaches are complementary.
- **Interpretability from Fine- to Coarse-Grained**: Every coarse-grained score is grounded in corresponding fine-grained span-level error evidence, a capability absent from existing metrics including GPT-4o-as-Judge.
- **Social Value of the DOCENT Benchmark**: Assistive text generation is critical for web accessibility for visually impaired users; the complex visual scenes in artworks (averaging 161 visual components) represent a genuine challenge for current VLMs.

## Limitations & Future Work
- Quality depends on the dependency parsing and coreference resolution pipeline — tool maturity for non-English languages may be insufficient.
- All components (entities/attributes/relations) are currently weighted equally; task-specific weighting could be introduced in future work.
- DOCENT covers only 100 images with generated judgments, limited in scale by the annotation cost of fine-grained labeling (~18 minutes per sample).
- The reference-based design is sensitive to the quality and coverage of reference descriptions.

## Related Work & Insights
- **vs. SPICE**: SPICE also uses scene graphs but ignores object attachment, inflating scores for misattributed details; PoSh preserves attachment chains to ensure correct verification.
- **vs. DCScore**: DCScore uses GPT-4 to extract and verify factoids, achieving the strongest mistakes correlation (ρ=0.541), but insufficient extraction coverage weakens its omissions performance (ρ=0.395); PoSh uses syntactic analysis to ensure full coverage.
- **vs. LLaVA-Critic**: This 72B VLM-as-Judge performs best on CapArena overall, but provides no interpretable fine-grained scores; PoSh achieves comparable performance at 14B while remaining fully interpretable.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of scene graphs and LLM-QA is elegantly designed; the DOCENT benchmark fills an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers DOCENT fine- and coarse-grained evaluation, cross-domain CapArena, RL reward function experiments, and subcomponent validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, societal impact is compelling, and experiments are systematic and comprehensive.
- Value: ⭐⭐⭐⭐ — Provides a deployable open-source tool for detailed image description evaluation, advancing progress in assistive text generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] On the Possible Detectability of Image-in-Image Steganography](../../CVPR2026/interpretability/on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[ICLR 2026\] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)
- [\[CVPR 2026\] Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](../../CVPR2026/interpretability/edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](../../NeurIPS2025/interpretability/urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](../../ACL2026/interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)

</div>

<!-- RELATED:END -->
