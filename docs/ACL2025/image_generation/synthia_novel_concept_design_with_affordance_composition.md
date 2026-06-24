---
title: >-
  [Paper Note] Synthia: Novel Concept Design with Affordance Composition
description: >-
  [ACL 2025][Image Generation][affordance] Synthia proposes a novel concept design framework based on affordance composition. By leveraging a hierarchical concept ontology, an affordance sampling strategy, and curriculum learning to fine-tune a T2I model, it generates innovative designs that are both visually novel and functionally coherent.
tags:
  - "ACL 2025"
  - "Image Generation"
  - "affordance"
  - "concept design"
  - "curriculum learning"
  - "contrastive learning"
  - "T2I models"
date: 2026-05-08
content_hash: 3f4e012f619e2626
---

# Synthia: Novel Concept Design with Affordance Composition

**Conference**: ACL 2025  
**arXiv**: [2502.17793](https://arxiv.org/abs/2502.17793)  
**Code**: Yes ([https://github.com/HyeonjeongHa/SYNTHIA](https://github.com/HyeonjeongHa/SYNTHIA))  
**Area**: Other  
**Keywords**: affordance, concept design, curriculum learning, contrastive learning, T2I models

## TL;DR

Synthia proposes a novel concept design framework based on affordance composition. By leveraging a hierarchical concept ontology, an affordance sampling strategy, and curriculum learning to fine-tune a T2I model, it generates innovative designs that are both visually novel and functionally coherent.

## Background & Motivation

Text-to-image (T2I) models are widely applied in AI-driven design, but existing methods suffer from two core limitations:

**Neglecting functional coherence**: Existing approaches rely on complex text descriptions to generate visual variations without considering whether multiple functions can be harmoniously integrated into a single coherent concept. For instance, when requested to exhibit both "driving" and "vacuuming" functions, Stable Diffusion merely generates a car, lacking the vacuuming functionality altogether.

**Lacking a structured functional foundation**: Directly feeding LLM-generated prompts into T2I models suffers from a lack of understanding regarding the hierarchical "concept-part-affordance" structure.

The core idea of Synthia is to use **affordances** as control signals for concept composition, rather than relying on complex textual descriptions. For example, given the affordances "brew + deliver", the model should automatically synthesize a novel design that possesses the functions of both a coffee maker and a cart.

## Method

### Overall Architecture

Synthia comprises three phases:
1. **Affordance Composition Curriculum Construction** — Creating training data from easy to hard based on ontology.
2. **Affordance-based Curriculum Learning** — Fine-tuning the T2I model combined with contrastive objectives.
3. **Evaluation** — Automatic and human evaluation of faithfulness, novelty, practicality, and coherence.

### Key Designs

1. **Hierarchical Concept Ontology $\mathcal{O} = (\mathcal{S}, \mathcal{C}, \mathcal{P}, \mathcal{A})$**

    - Four-level structure: Super-category $\rightarrow$ Concept $\rightarrow$ Part $\rightarrow$ Affordance.
    - Example: furniture $\rightarrow$ sofa $\rightarrow$ {leg, cushion} $\rightarrow$ {support, rest}.
    - Scale: 30 super-categories, 590 concepts, 1,172 parts, 686 affordances.
    - Design Motivation: Provides a structured functional foundation for the T2I model to avoid compositions based solely on superficial visual features.

2. **Affordance Sampling Strategy**

    - Defines concept distance $D_\mathcal{C}(c_i, c_j)$ by fusing affordance-level Jaccard similarity and BERT semantic similarity ($\alpha=0.7, \beta=0.3$).
    - Derives affordance distance $D_\mathcal{A}(a_i, a_j)$ by averaging the pairwise distances of associated concepts.
    - Design Motivation: Avoids redundant combinations caused by random sampling (e.g., cook + heat), ensuring the selection of sufficiently distinct affordance pairs.

3. **Three-Stage Curriculum Construction**

    - Phase 1: Near-distance affordance pairs $\rightarrow$ learning basic concept-affordance associations.
    - Phase 2: Medium-distance $\rightarrow$ learning fine-grained compositional structures.
    - Phase 3: Far-distance $\rightarrow$ challenging the model to synthesize genuinely novel and functionally coherent concepts.
    - A total of 600 affordance pairs are sampled, with 10 images generated per pair (via DALL-E) and the Top-3 retained after CLIP filtering.

4. **Contrastive Learning Fine-Tuning**

    - Positive Constraint: Pseudo-novel concept images corresponding to the target affordances.
    - Negative Constraint: Existing concept images in the ontology that already contain the target affordances.
    - Total Loss: $\mathcal{L} = \mathcal{L}_{pos} - \gamma \cdot \mathcal{L}_{neg}$
    - Includes noise prediction loss to prevent catastrophic forgetting.

### Inference

At inference time, only the affordances are provided as positive constraints, requiring no negative constraints or complex descriptions. Prompt format: "a new design that has functions of {desired affordances}."

## Key Experimental Results

### Main Results — Automatic and Human Evaluation (Table 1)

| Model | Faithfulness (Auto/Human) | Novelty (Auto/Human) | Practicality (Auto/Human) | Coherence (Auto/Human) |
|------|-------------------|-------------------|-------------------|-------------------|
| Stable Diffusion | 3.77/2.96 | 3.74/2.44 | 3.34/3.02 | 3.29/2.75 |
| Kandinsky3 | 3.38/2.95 | 4.02/2.98 | 2.92/3.01 | 3.89/3.41 |
| ConceptLab | 3.39/2.73 | 4.08/3.11 | 2.93/2.68 | 3.96/3.54 |
| **Synthia** | **3.99/3.81** | **4.55/3.89** | **3.35/3.38** | **4.81/4.06** |

Synthia achieves an absolute gain of **25.1% and 14.7%** in human evaluation for novelty and coherence, respectively.

### Ablation Study

| Ablation Item | Key Findings |
|--------|----------|
| Training Data Size | Gradual improvement from 200 $\rightarrow$ 400 $\rightarrow$ 600 pairs, with 600 pairs being optimal. |
| Affordance Distance | Synthia's novelty consistently outperforms baselines on far-distance pairs. |
| Curriculum Learning vs. Random Training | Curriculum learning significantly outperforms random training in the early stages of training. |
| 3/4 Affordance Inputs | Trained only on pairs (2), yet maintains high performance on 3/4 affordances. |

### Key Findings

1. Existing T2I models tend to generate existing concepts rather than novel designs on near-distance affordances.
2. Curriculum learning significantly accelerates training and guides the model to generate high-quality novel concepts.
3. Synthia even outperforms DALL-E in relative evaluations, suggesting that the fine-tuned concept composition capability surpasses that of the original base model.
4. The human evaluation IAA is 67.5%, and the alignment rate between automatic and human evaluation reaches 91.25%.

## Highlights & Insights

1. **Unique Affordance Perspective**: Synthetically treats "functionality" as a first-class citizen in concept design, rather than relying solely on visual features.
2. **Exquisite Ontology Design**: The hierarchical concept ontology provides a solid foundation for structured functional compositions.
3. **Effective Curriculum Learning**: The training strategy of affordance composition from easy to hard significantly outperforms random training.
4. **Simple Inference**: Inference requires no complex prompts or negative constraints, but only the affordance keywords.

## Limitations & Future Work

- Training data relies heavily on pseudo-novel images generated by DALL-E, and is thus limited by the quality of DALL-E itself.
- Ontology construction requires manual design, which is costly to scale to more domains.
- Only the composition of 2 affordances is evaluated; the performance on larger numbers of affordances remains to be validated.
- Physics feasibility and manufacturing constraints are not considered.

## Related Work & Insights

- ConceptLab leverages Diffusion Prior to optimize generation but neglects functional coherence.
- Concept Weaver refines based on template images and similarly disregards affordance.
- The theory of combinatorial creativity (Han et al., 2018) provides a psychological foundation for far-distance affordance composition.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Affordance-driven concept design is a novel perspective; the ontology + curriculum learning framework is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive automatic and human evaluations with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Practical application value for AI-assisted design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](../../CVPR2025/image_generation/from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)
- [\[ICML 2025\] Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency](../../ICML2025/image_generation/shielded_diffusion_generating_novel_and_diverse_images_using_sparse_repellency.md)
- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_affordance_aware_hierarchical_model_robotic_manipulation.md)
- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](../../ECCV2024/image_generation/text2place_affordance-aware_text_guided_human_placement.md)

</div>

<!-- RELATED:END -->
