---
title: >-
  [Paper Note] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Models] From a cognitive semiotics perspective, this paper uncovers the "literal superiority bias" in VLMs—the tendency of models to favor literal interpretations over metaphori…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Semiotic Gap"
  - "Literal Bias"
  - "Iconographic Abstraction"
  - "Noun Compounds"
date: 2026-05-08
content_hash: f25440b15a69f9a7
---

# More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage

**Conference**: ACL 2026  
**arXiv**: [2604.17354](https://arxiv.org/abs/2604.17354)  
**Code**: [GitHub](https://github.com/risehnhew/More-than-meets-the-eye)  
**Area**: Multimodal VLM / Semiotic Understanding  
**Keywords**: Vision-Language Models, Semiotic Gap, Literal Bias, Iconographic Abstraction, Noun Compounds

## TL;DR

From a cognitive semiotics perspective, this paper uncovers the "literal superiority bias" in VLMs—the tendency of models to favor literal interpretations over metaphorical/idiomatical ones when presented with high-fidelity images. By introducing the DIVA benchmark (iconographic simplified images) and the Semantic Alignment Gap metric, it demonstrates that reducing visual fidelity can significantly narrow the gap between literal and idiomatic interpretations.

## Background & Motivation

**Background**: Text-to-image models can generate highly realistic images, and VLMs excel at decoding the literal content of images. However, a fundamental cognitive gap remains in understanding abstract meanings such as idioms and metaphors.

**Limitations of Prior Work**: (1) Existing VL benchmarks primarily focus on literal vision-text alignment (object detection, attribute binding, etc.) and under-evaluate figurative meanings; (2) The visual representation of noun compounds (e.g., "Eye Candy") requires a shift from literal iconicity to idiomatic symbolism, yet models are often misled by high-fidelity visual details; (3) There is a lack of consistent evaluation metrics across architectures—discriminative models use cosine similarity, generative models use token probabilities, and closed-source models rely on behavioral probing.

**Key Challenge**: The pre-training objectives of VLMs over-optimize for physical reconstruction and visual iconicity. When faced with tasks requiring abstract/symbolic understanding, high-fidelity visual details become "cognitive interference"—the model sees an "Eye" and focuses only on the physical organ rather than the metaphorical meaning of "Eye Candy."

**Goal**: (1) Quantify the degree of literal bias in VLMs; (2) Validate the hypothesis that "reducing visual fidelity improves symbolic understanding"; (3) Provide a unified evaluation framework across diverse architectures.

**Key Insight**: Derived from semiotic theory—icons convey meaning via similarity, while symbols convey meaning via convention. Text is naturally symbolic, whereas images are typically iconic. When the iconicity (high-fidelity detail) of an image is too high, the model is locked into literal decoding.

**Core Idea**: Through "Iconographic Abstraction"—systematically reducing the visual fidelity of images (removing textures, lighting, and simplifying composition)—images are transformed from "realistic simulations" to "semiotic symbols," thereby releasing the model's potential for symbolic understanding.

## Method

### Overall Architecture

The construction of the DIVA benchmark follows these steps: (1) 100 English noun compounds (NCs) along with their literal and idiomatic high-fidelity images are obtained from the SemEval-2025 AdMIRe task; (2) Gemini is used to generate corresponding iconographic (low-fidelity, schematic) images, creating 5 comparative images for each NC (high-idiomatic, high-literal, weak-idiomatic, weak-literal, and distractor); (3) Human verification is performed by 3 annotators. During evaluation, the semantic matching score difference between literal and idiomatic images is calculated for each NC.

### Key Designs

1.  **Semantic Alignment Gap ($\Delta$) and Signed Literal Bias ($b$):**
    - **Function**: To uniformly quantify the magnitude and direction of the literal-idiomatic gap across different VLM architectures.
    - **Mechanism**: For each noun compound $t$, the model's semantic matching score difference between the literal image $v_{lit}$ and the idiomatic image $v_{id}$ is calculated. $b(t) = \mathcal{S}(v_{lit}, t) - \mathcal{S}(v_{id}, t)$ measures the direction (positive values indicate literal preference), while $\Delta(t) = |b(t)|$ measures intensity. $\mathcal{S}$ is implemented in three ways based on the model architecture.
    - **Design Motivation**: Existing evaluations are either limited to specific architectures or fail to distinguish between direction and intensity. As a relative internal measure, $\Delta$ allows for meaningful trend analysis within architecture families.

2.  **Tri-fold Scoring:**
    - **Function**: To make the $\Delta$ metric applicable to discriminative, open-source generative, and closed-source models.
    - **Mechanism**: (i) Discriminative models (CLIP/SigLIP) use cosine similarity in embedding space; (ii) Open-source generative models (LLaVA/InternVL) use token probabilities (LID) from forced Yes/No responses; (iii) Closed-source models (GPT-5/Claude) use self-reported confidence scores $\gamma \in [0,100]$, validated by behavioral frequencies from 10-shot forced-choice trials.
    - **Design Motivation**: Since "confidence signals" are acquired differently across architectures, a unified metric must accommodate this heterogeneity.

3.  **Iconographic Abstraction Pipeline:**
    - **Function**: To convert high-fidelity images into low-fidelity iconographic images, preserving the semantic core while removing visual noise.
    - **Mechanism**: A two-stage process using Gemini—semantic distillation (retaining intended meaning while stripping incidental scene details) and geometric reconstruction (constraining to a flat, low-detail icon style). After generation, human annotators verify semantic retention and style constraints.
    - **Design Motivation**: Based on "semantic anchorage" theory—when visual signals become more "digital" (non-analog), models are less likely to default to literal interpretations and more likely to adopt a symbolic stance.

### Loss & Training

This work is purely evaluative and does not involve model training. DIVA contains 1,000 iconographic images (100 NCs × 5 comparisons × 2 semantic directions).

## Key Experimental Results

### Main Results

| Model | $\Delta$ (AdMIRe/Hi-Fi) | $\Delta$ (DIVA/Iconic) | $\Delta$ Reduction |
| :--- | :--- | :--- | :--- |
| SigLIP 2 | 0.245 | 0.178 | -27% |
| EVA-CLIP-18B | 0.262 | 0.191 | -27% |
| InternVL3-78B | 0.138 | 0.089 | -36% |
| Qwen2.5-VL-32B | 0.145 | 0.095 | -34% |
| LLaVA-OV-7B | 0.176 | 0.122 | -31% |
| GPT-5 | 0.065 | 0.021 | -68% |
| Claude 4.5 Sonnet | 0.072 | 0.028 | -61% |

### Ablation Study

| Analysis Dimension | Result |
| :--- | :--- |
| Discriminative vs. Generative | Discriminative models have the largest $\Delta$ (~0.25), while generative ones are significantly smaller (~0.14), and closed-source are the smallest (~0.07). |
| Model Scale Effects | Within the same architecture, larger models do not necessarily have a smaller $\Delta$—scaling alone does not automatically resolve literal bias. |
| 5-way Selection Accuracy | Iconographic images improved accuracy across all model families (Discriminative: 42.3→58.7%, Closed-source: 78.5→91.3%). |

### Key Findings

- All models exhibit a positive $b(t)$ (literal preference) under all conditions, and this is more pronounced with high-fidelity images.
- Iconographic abstraction consistently reduces $\Delta$ across all architecture families—GPT-5 dropped from 0.065 to 0.021, approaching nearly zero bias.
- Discriminative models suffer the most from "cognitive interference"—CLIP-like models rely excessively on texture and surface features.
- Spearman correlation analysis indicates that human evaluations are highly consistent with the $\Delta$ metric ($\rho=0.64-0.73$).

## Highlights & Insights

- Approaching VLM evaluation through semiotic theory is a highly novel perspective—transforming the question of "why models fail to understand metaphors" into a quantifiable measurement along the "icon-symbol continuum."
- The counter-intuitive finding that "high fidelity serves as cognitive interference" is very enlightening—more realistic images do not necessarily facilitate understanding, challenging the implicit assumption that "clearer is better."
- The tri-modal design of the $\Delta$ metric effectively addresses the challenge of comparable cross-architecture evaluation.

## Limitations & Future Work

- The scope is limited to English noun compounds and does not address cross-cultural metaphors (e.g., the Chinese "Iron Rice Bowl").
- The specific style of iconographic images (flat design) might introduce style bias—models might perform better due to familiarity with specific visual styles.
- Self-reported confidence in closed-source models might reflect instruction-following tendencies rather than true semantic judgment.
- Ours serves as a diagnostic tool but does not propose specific methods for model improvement.

## Related Work & Insights

- **vs T2I-CompBench/GenEval**: These benchmarks focus on physical compositionality (e.g., a blue ball next to a red square), whereas Ours focuses on semantic compositionality—where noun combinations produce abstract meanings that transcend literal parts.
- **vs AdMIRe (SemEval-2025)**: While AdMIRe evaluates model alignment with idiomatic images, its use of high-fidelity images can introduce confounding factors; DIVA controls for visual complexity through iconification.
- **vs IconQA**: IconQA uses icon-style charts for reasoning but does not involve figurative understanding.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Semiotic perspective + Iconographic Abstraction hypothesis + Unified cross-architecture metric; highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 8 models across three architectural paradigms with human validation, though limited to English noun compounds.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Elegant theoretical framework, rigorous methodology, and clear exposition.
- **Value**: ⭐⭐⭐⭐ Deeply reveals the literal bias problem in VLMs, though it lacks a solution for mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] More than the Sum: Panorama-Language Models for Adverse Omni-Scenes](../../CVPR2026/multimodal_vlm/more_than_the_sum_panorama-language_models_for_adverse_omni-scenes.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](../../AAAI2026/multimodal_vlm/patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
