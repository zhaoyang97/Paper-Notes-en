---
title: >-
  [Paper Note] LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces
description: >-
  [ICML2025][Image Generation][Pluralistic Alignment] Through a two-year community-based participatory research effort, this work constructs the LIVS dataset containing 37,710 pairs of multi-criteria preference annotations for the pluralistic alignment of text-to-image models in inclusive urban public space design, and validates its effectiveness by fine-tuning SDXL with DPO.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Pluralistic Alignment"
  - "Text-to-Image"
  - "DPO"
  - "Intersectionality"
  - "Urban Planning"
  - "Inclusive Design"
  - "Community Engagement"
date: 2026-05-08
content_hash: 4959924cbc5aba07
---

# LIVS: A Pluralistic Alignment Dataset for Inclusive Public Spaces

**Conference**: ICML2025  
**arXiv**: [2503.01894](https://arxiv.org/abs/2503.01894)  
**Code**: [mid-space.one](https://mid-space.one)  
**Area**: Dataset / Fairness / Pluralistic Alignment  
**Keywords**: Pluralistic Alignment, Text-to-Image, DPO, Intersectionality, Urban Planning, Inclusive Design, Community Engagement

## TL;DR

Through a two-year community-based participatory research effort, this work constructs the LIVS dataset containing 37,710 pairs of multi-criteria preference annotations for the pluralistic alignment of text-to-image models in inclusive urban public space design, and validates its effectiveness by fine-tuning SDXL with DPO.

## Background & Motivation

- **Limitations of Existing T2I Alignment**: Current alignment frameworks (e.g., Simulacra, Pic-a-Pic, ImageReward) typically assume a single global "good/bad" standard, focusing primarily on aesthetics or content moderation while ignoring the distinct demands of diverse communities regarding accessibility, safety, and inclusion.
- **Neglected Intersectionality**: Generative models are typically calibrated for the "average user," which may systematically exclude or misrepresent historically marginalized groups. Users with different intersecting identities (e.g., disability + ethnicity) have vastly different criteria for evaluating public spaces.
- **Lack of Local Knowledge**: Global crowdsourced data struggles to capture the historical, spatial, and cultural contexts of specific communities, a limitation that is particularly prominent in fields like urban planning.
- **Core Goal**: To propose a pluralistic alignment paradigm that, leveraging a community-driven, multi-criteria preference dataset, enables T2I models to generate outputs reflecting localized, intersectional, and diverse values.

## Method

### 1. Community-Participatory Dataset Construction

Spanning two years, in collaboration with 30 community organizations in Montreal, this work conducted 11 workshops, 34 interviews, and 5 batches of annotation:

- **Concept Collection** → 634 initial concepts (covering lighting, accessibility facilities, multilingual signage, etc.)
- **Criteria Refinement** → Distilled into **6 core criteria** through semantic aggregation and voting discussions:
    - **Accessibility**: Ramps, elevators, tactile paving
    - **Safety**: Lighting, visibility, protection
    - **Comfort**: Seating, shade, noise control
    - **Invitingness**: Greenery, open layout
    - **Inclusivity**: Multilingual signage, cultural needs
    - **Diversity**: Diverse crowds, multi-use spaces

### 2. Prompting and Image Generation

- Workshops collected 440 human-written prompts, and GPT-4o was used with three strategies to expand these to approximately 2,910 synthetic prompts.
- For each prompt, up to 20 images were generated using SDXL, and the 4 most distinct images were greedily selected based on CLIP similarity.
- Ultimately, **13,462 images** were retained for annotation.

### 3. Multi-Criteria Preference Annotation

- Each time, a pair of images was shown, and 3 out of the 6 criteria were randomly assigned.
- Annotators used a slider $[-1, +1]$ to indicate the direction and intensity of preference, with 0 indicating neutrality.
- The 5 batches of annotation yielded a total of **37,710 image-level comparisons**, corresponding to approximately 113,130 criteria-level annotations.

### 4. DPO Fine-tuning

Multi-criteria annotations were aggregated into binary preference signals, using majority voting to resolve conflicts between criteria:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x_w, x_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(x_w)}{\pi_{\text{ref}}(x_w)} - \beta \log \frac{\pi_\theta(x_l)}{\pi_{\text{ref}}(x_l)} \right) \right]$$

where $x_w, x_l$ denote the preferred and dispreferred images respectively, $\pi_\theta$ is the fine-tuned model, and $\pi_{\text{ref}}$ is the baseline SDXL.

## Key Experimental Results

### Case Study I: Whether Multi-Criteria DPO Improves Alignment

| Evaluation Result | Quantity | Proportion |
|:--|:--|:--|
| Preference for DPO model | 700 | 32% |
| Preference for baseline SDXL | 300 | 14% |
| Neutral/No preference | 1,100 | 50% |

- Criteria with a larger volume of annotations (Comfort, Invitingness) showed more significant DPO improvements.
- Inclusivity and Diversity exhibited higher rates of neutrality, indicating stronger subjectivity.

### Case Study II: Influence of Identity on Preferences

- Most participants slightly preferred the DPO outputs; participants who joined in later stages (and did not participate in the criteria-building workshops) showed no distinct preference between the two models.
- Annotators with mobility impairments preferred the DPO outputs (indicating that some accessibility features were captured).

### Case Study III: Impact of Prompt Sources

- Human-written prompts (Method 0) resulted in **fewer neutral annotations**, meaning visual differences were more pronounced.
- Prompts generated by GPT-4o had higher rates of neutrality, likely due to a lack of local context and detail.

### Case Study IV: Rating Differences Across Intersecting Identities

- Different groups with intersecting identities (e.g., disability $\times$ race) gave systematically different ratings on criteria such as Accessibility and Safety.
- This confirms that a single global alignment objective cannot capture pluralistic preferences.

## Highlights & Insights

1. **Real Community Engagement**: The process spanning two years, involving 30 organizations, and refining 634 concepts into 6 criteria is highly compelling, far surpassing common crowdsourced annotations.
2. **Multi-Criteria Preference Learning**: For the first time, an intersectional perspective is introduced to T2I alignment, with 6 criteria covering the core dimensions of urban public spaces.
3. **Signal Value of Neutral Annotations**: Approximately 50% neutrality does not indicate failure; instead, it reflects the heterogeneity and ambiguity of community values, providing an essential basis for future alignment method design.
4. **Human vs. LLM Prompts Comparison**: It reveals the unique role of human creativity in generating distinctive visual outputs.
5. **Reusable Methodology**: The pipeline of community engagement + multi-criteria annotation + DPO can be extended to areas such as cultural heritage preservation, healthcare, and education.

## Limitations & Future Work

- **Geographical Limitations**: The study only covers Montreal, a mid-sized multicultural city. Generalization to other cultural contexts remains to be validated.
- **Small Scale of Participants**: There are only about 18 core annotators and an evaluation set of 2,200 entries, resulting in limited statistical significance.
- **Simplification of DPO**: Multi-criteria feedback was collapsed into binary labels via majority voting, discarding conflicts and fine-grained information between criteria.
- **Unused Neutral Annotations**: About 50% of neutral annotations were ignored during training, wasting potential alignment signals.
- **Visual Limitations**: DPO improved sidewalk and seating layouts, but the generation of detailed features like ramps, tactile paving, and multilingual signs remains unstable.
- **Future Directions**: Pareto multi-objective optimization, user-personalized alignment layers, and partial retrieval of reward signals from neutral annotations.

## Related Work & Insights

- **Preference Datasets**: Simulacra (238K), Pic-a-Pic (500K+), ImageReward (137K), and HPS/HPS v2 all employ single-objective global standards.
- **Multi-Dimensional Preferences**: MPS (Zhang et al., 2024b) is trained on 4 dimensions (aesthetics/semantics/detail/overall) but lacks community engagement.
- **Pluralistic Alignment Theory**: Sorensen et al. (2024) proposed Overton pluralism, steerable alignment, and distributional pluralism. LIVS connects with these theories at a practical level.
- **DPO**: The DPO framework by Rafailov et al. (2024) is systematically applied to T2I alignment in the urban planning domain for the first time here.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduces intersectionality and community engagement to T2I alignment with a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐ — The 4 case studies are clearly structured, but the participant scale and data volume are relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Methodological descriptions are detailed, and the community engagement process is highly transparent.
- Value: ⭐⭐⭐⭐ — Provides a reusable dataset and methodological paradigm for fairness and pluralistic alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)](taming_diffusion_for_dataset_distillation_with_high_representativeness.md)
- [\[CVPR 2026\] BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment](../../CVPR2026/image_generation/biovita_biological_dataset_model_and_benchmark_for_visual-textual-acoustic_align.md)
- [\[ICML 2025\] Theoretical Guarantees on the Best-of-n Alignment Policy](theoretical_guarantees_on_the_best-of-n_alignment_policy.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](../../NeurIPS2025/image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[CVPR 2026\] Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective](../../CVPR2026/image_generation/toward_diffusible_high-dimensional_latent_spaces_a_frequency_perspective.md)

</div>

<!-- RELATED:END -->
