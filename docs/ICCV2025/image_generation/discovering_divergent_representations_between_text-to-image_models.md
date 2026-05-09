---
title: >-
  [Paper Note] Discovering Divergent Representations between Text-to-Image Models
description: >-
  [ICCV 2025][Image Generation][text-to-image model comparison] This paper proposes CompCon (Comparing Concepts), an evolutionary search algorithm that automatically discovers "divergent representations" between two text-to-image models — identifying which visual attributes differ between models and which prompt types trigger these differences — and introduces the ID² benchmark dataset for systematic evaluation.
tags:
  - ICCV 2025
  - Image Generation
  - text-to-image model comparison
  - divergent representation discovery
  - evolutionary search
  - visual attributes
  - model bias
date: 2026-05-08
content_hash: 5c61317cede1a752
---

# Discovering Divergent Representations between Text-to-Image Models

**Conference**: ICCV 2025
**arXiv**: [2509.08940](https://arxiv.org/abs/2509.08940)
**Code**: [https://github.com/adobe-research/CompCon](https://github.com/adobe-research/CompCon)
**Area**: Diffusion Models / Model Analysis
**Keywords**: text-to-image model comparison, divergent representation discovery, evolutionary search, visual attributes, model bias

## TL;DR

This paper proposes CompCon (Comparing Concepts), an evolutionary search algorithm that automatically discovers "divergent representations" between two text-to-image models — identifying which visual attributes differ between models and which prompt types trigger these differences — and introduces the ID² benchmark dataset for systematic evaluation.

## Background & Motivation

**Background**: Text-to-image (T2I) models such as Stable Diffusion, DALL-E, PixArt, and Playground have become mainstream generative tools. Each model is shaped by distinct training data, architecture, and optimization strategies, and thus may generate visually distinct images from the same text prompt. However, systematic tools for understanding *where* these models differ remain lacking.

**Limitations of Prior Work**: Analysis of model differences has largely relied on manual subjective evaluation or global statistics such as FID. The former does not scale, while the latter lacks interpretability. Although it is broadly known that different models "look different," it is difficult to precisely characterize *under what conditions* and *along which visual dimensions* these differences manifest.

**Key Challenge**: Inter-model differences are input-dependent — the same pair of models may produce consistent outputs for certain prompt types while diverging substantially for others. Simple comparisons of aggregate output distributions cannot capture such fine-grained, input-conditioned differences.

**Goal**: To automatically discover "divergent representations" between two T2I models — i.e., to identify (visual attribute, prompt type) pairs such that, given a certain class of prompts, one model tends to exhibit a particular visual attribute while the other does not.

**Key Insight**: The authors draw inspiration from evolutionary search — starting from an initial set of hypotheses and iteratively refining them through a propose → verify → filter → evolve cycle to progressively discover more accurate and informative divergent representations.

**Core Idea**: An LLM/VLM-driven evolutionary search algorithm automatically generates hypotheses about which visual attributes diverge under which prompt types. A VLM classifier then validates each hypothesis on generated images; hypotheses that pass validation are retained and iteratively evolved toward more precise descriptions.

## Method

### Overall Architecture

CompCon proceeds in three stages: (1) given a set of prompts, images are generated from both models; (2) a VLM analyzes the two image sets to identify visual differences and propose "divergence hypotheses" (e.g., "Model A more frequently generates flames"); (3) an evolutionary search iteratively refines these hypotheses while simultaneously identifying the prompt characteristics that trigger each divergence (e.g., "prompts involving intense emotional expression"). The final output is a set of validated (visual attribute, trigger condition) pairs.

### Key Designs

1. **Evolutionary Attribute Search**:

    - **Function**: Automatically discovers visual attributes present in one model's outputs but absent in the other's.
    - **Mechanism**: Maintains a population of attribute hypotheses. In each iteration: (a) a VLM (e.g., GPT-4V) observes generated images from both models and proposes new visual divergence hypotheses; (b) a VLM classifier performs binary classification for each hypothesis across all images (presence or absence of the attribute); (c) a "divergence score" is computed as the difference in attribute occurrence frequency between the two models; (d) high-scoring hypotheses are retained, low-scoring ones are discarded, and an LLM mutates or recombines high-scoring hypotheses to generate new candidates.
    - **Design Motivation**: Exhaustive enumeration of all possible visual attributes is infeasible. The evolutionary search leverages the creative capacity of LLMs and the visual judgment of VLMs to efficiently explore a vast hypothesis space.

2. **Prompt Concept Linking**:

    - **Function**: Identifies the prompt features or concepts that trigger a given visual divergence.
    - **Mechanism**: For each high-divergence visual attribute discovered, prompts whose corresponding images exhibit the attribute are separated from those that do not. An LLM then analyzes the semantic differences between these two groups to extract "trigger concepts." For example, the "flames" attribute may be primarily triggered by "prompts involving emotional expression." This step employs iterative refinement: an initial description is validated on additional data, then adjusted based on misclassified examples.
    - **Design Motivation**: Knowing that "Model A more frequently generates flames" is insufficient; understanding *under what conditions* this divergence arises is essential for users to interpret and leverage model characteristics.

3. **ID² Benchmark Dataset (Input-Dependent Differences Dataset)**:

    - **Function**: Provides a standardized evaluation framework for measuring the performance of divergence discovery algorithms.
    - **Mechanism**: An automated data generation pipeline constructs 60 known input-dependent differences. For each difference, a divergent visual attribute and its associated divergent prompt description are provided; an LLM then generates matched prompt pairs with and without the attribute, and a T2I model generates the corresponding image pairs. Evaluation metrics assess whether an algorithm can accurately recall these known differences.
    - **Design Motivation**: Prior work lacks an objective benchmark for evaluating divergence discovery capability; ID² fills this gap.

### Loss & Training

CompCon does not involve neural network training. The core metric is the divergence score $\Delta(a) = |P(\text{attr}=a | \text{model}_1) - P(\text{attr}=a | \text{model}_2)|$, which measures the difference in attribute occurrence frequency between the two models. The evolutionary search applies a threshold filter ($\Delta > \tau$, $\tau = 0.2$) to retain meaningful divergences, and employs a deduplication mechanism based on semantic similarity to merge approximately equivalent hypotheses.

## Key Experimental Results

### Main Results

| Method | Recall@10 (ID²) | Precision@10 | Avg. Divergence Score |
|--------|-----------------|-------------|----------------------|
| CompCon | **0.58** | **0.72** | **0.41** |
| LLM-only baseline | 0.32 | 0.45 | 0.28 |
| VLM-only baseline | 0.38 | 0.51 | 0.33 |
| TF-IDF baseline | 0.21 | 0.30 | 0.19 |
| Random sampling | 0.08 | 0.12 | 0.11 |

### Model Comparison Findings

| Model Pair | Discovered Divergence | Trigger Condition |
|------------|----------------------|------------------|
| PixArt vs. SD 3.5 | PixArt generates "rain-slicked streets" | Prompts associated with solitude |
| SD 3.5 vs. Playground | SD 3.5 more frequently depicts African Americans | Prompts involving media professions |
| DALL-E 3 vs. SD 3.5 | DALL-E 3 exhibits more cartoon-like style | Prompts involving animals |
| PixArt vs. Playground | Playground produces higher-contrast lighting | Prompts involving architecture |

### Key Findings

- CompCon outperforms the LLM-only baseline by 81% in Recall, demonstrating that visual signals (VLM judgments) are critical for divergence discovery.
- Evolutionary search improves over single-round search by approximately 30%; iterative refinement uncovers more non-intuitive divergences.
- Surprising bias patterns are identified, such as specific models associating certain racial groups with particular occupations.
- The ID² dataset contains 60 known differences spanning multiple dimensions including style, content, and composition.

## Highlights & Insights

- **Synergy of Evolutionary Search with LLMs and VLMs**: LLMs' language reasoning generates hypotheses; VLMs' visual judgment validates them; both are coordinated through an evolutionary framework. This "AI-driven scientific discovery" paradigm is generalizable to other comparative analysis tasks.
- **Precise Localization of Input-Dependent Differences**: Rather than comparing aggregate output distributions, the method precisely identifies *which inputs* trigger *which visual attribute* differences — a granularity of analysis valuable for model auditing and safety assessment.
- **Social Bias Discovery**: Beyond stylistic differences, the method can surface latent social biases related to race, gender, and other attributes embedded in models, carrying significant implications for AI fairness research.

## Limitations & Future Work

- CompCon's effectiveness is strongly dependent on the accuracy of the VLM classifier, and subtle visual differences may be missed.
- The evolutionary search requires a large number of API calls (LLM + VLM), resulting in relatively high computational cost.
- The current framework is limited to pairwise model comparison; scaling to simultaneous comparison of multiple models remains an open challenge.
- The ID² dataset is generated via an automated pipeline and may not cover all types of divergences.
- Future directions include: extending the method to video generation model comparison; incorporating human feedback to improve divergence discovery; and using discovered divergences to guide model improvement.

## Related Work & Insights

- **vs. VisDiff**: VisDiff compares visual differences between datasets, whereas CompCon compares differences between models while accounting for input dependence.
- **vs. Traditional Metrics (FID/CLIPScore)**: FID and CLIPScore yield global statistics that cannot explain where specific differences lie. CompCon provides interpretable, concrete divergence descriptions.
- **vs. Model Auditing Tools**: Existing auditing tools (e.g., DALL-E probes) primarily detect known bias categories, whereas CompCon can automatically discover previously unknown divergence patterns.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic approach to discovering input-dependent divergent representations between T2I models; both the problem formulation and methodology are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Constructs the ID² benchmark dataset and conducts systematic analysis across multiple model pairs.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; case studies are vivid and informative.
- Value: ⭐⭐⭐⭐ Practically valuable across multiple scenarios including model understanding, model selection, and bias auditing.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models](adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)
- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model](adaptive_routing_of_text_to_image_generation_requests_between_large_cloud_model_and_light_weight_edge_model.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)

<!-- RELATED:END -->
