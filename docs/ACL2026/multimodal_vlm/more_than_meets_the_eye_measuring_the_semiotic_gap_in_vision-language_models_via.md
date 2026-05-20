---
title: >-
  [Paper Note] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Models] This paper exposes the "literal superiority bias" in VLMs from a cognitive-semiotic perspective—models tend toward literal rather than metaphorical/idiomatic interpretat…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Semiotic Gap"
  - "Literal Bias"
  - "Iconographic Abstraction"
  - "Noun Compounds"
date: 2026-05-08
content_hash: 7438ac8a607c87b9
---

# More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage

**Conference**: ACL 2026
**arXiv**: [2604.17354](https://arxiv.org/abs/2604.17354)  
**Code**: [GitHub](https://github.com/risehnhew/More-than-meets-the-eye)  
**Area**: Multimodal VLM / Semiotic Understanding
**Keywords**: Vision-Language Models, Semiotic Gap, Literal Bias, Iconographic Abstraction, Noun Compounds

## TL;DR

This paper exposes the "literal superiority bias" in VLMs from a cognitive-semiotic perspective—models tend toward literal rather than metaphorical/idiomatic interpretations of high-fidelity images. By introducing the DIVA benchmark (iconographically abstracted images) and the Semantic Alignment Gap metric, the paper demonstrates that reducing visual fidelity significantly narrows the gap between literal and idiomatic interpretations.

## Background & Motivation

**Background**: Text-to-image models can generate highly photorealistic images, and VLMs excel at decoding their literal content. However, a fundamental cognitive gap persists in understanding abstract meanings such as idioms and figurative language.

**Limitations of Prior Work**: (1) Existing VL benchmarks focus primarily on literal visual-text alignment (object detection, attribute binding, etc.), with insufficient evaluation of figurative meaning; (2) Visual representations of noun compounds (e.g., "Eye Candy") require a shift from literal iconicity to idiomatic symbolism, yet models are frequently misled by high-fidelity visual detail; (3) A consistent cross-architecture evaluation metric is lacking—discriminative models use cosine similarity, generative models use token probabilities, and closed-source models permit only behavioral probing.

**Key Challenge**: VLM pre-training objectives over-optimize for physical reconstruction and visual simulation (iconicity), causing high-fidelity visual detail to act as "cognitive interference" in tasks requiring abstract or symbolic understanding—a model seeing "Eye" fixates on the eye itself rather than the figurative meaning of "Eye Candy."

**Goal**: (1) Quantify the degree of literal bias in VLMs; (2) Test the hypothesis that reducing visual fidelity improves symbolic understanding; (3) Provide a unified evaluation framework applicable across architectures.

**Key Insight**: Drawing from semiotic theory—icons convey meaning through resemblance, symbols through convention. Text is inherently symbolic, whereas images are typically iconic. When an image's iconicity (high-fidelity detail) is too strong, the model becomes anchored to a literal interpretation.

**Core Idea**: Through *Iconographic Abstraction*—systematically reducing visual fidelity (removing texture, lighting, and compositional complexity)—images are transformed from realistic simulations into meaningful signs, thereby unlocking the model's capacity for symbolic understanding.

## Method

### Overall Architecture

The DIVA benchmark construction pipeline: (1) Obtain literal and idiomatic high-fidelity images for 100 English noun compounds from the SemEval-2025 AdMIRe task; (2) Use Gemini to generate corresponding iconographic (low-fidelity, schematic) images, producing 5 contrastive images per compound (high-idiomatic, high-literal, weak-idiomatic, weak-literal, distractor); (3) Three annotators perform manual verification. During evaluation, the difference in semantic matching scores between literal and idiomatic images is computed per noun compound.

### Key Designs

1. **Semantic Alignment Gap ($\Delta$) and Signed Literal Bias ($b$)**:

    - **Function**: Unified quantification of the direction and magnitude of the literal–idiomatic gap across VLM architectures.
    - **Mechanism**: For each noun compound $t$, the difference in semantic matching scores between the literal image $v_{lit}$ and the idiomatic image $v_{id}$ is computed. $b(t) = \mathcal{S}(v_{lit}, t) - \mathcal{S}(v_{id}, t)$ captures direction (positive indicates literal preference), and $\Delta(t) = |b(t)|$ captures magnitude. $\mathcal{S}$ has three implementations depending on model architecture.
    - **Design Motivation**: Existing evaluations are either architecture-specific or conflate direction and magnitude. As a model-internal relative measure, $\Delta$ enables meaningful trend analysis within the same architectural family.

2. **Tri-fold Scoring Function**:

    - **Function**: Extends the $\Delta$ metric to discriminative, open-source generative, and closed-source models.
    - **Mechanism**: (i) Discriminative models (CLIP/SigLIP) use cosine similarity in embedding space; (ii) Open-source generative models (LLaVA/InternVL) use token probabilities from forced Yes/No responses (LID); (iii) Closed-source models (GPT-5/Claude) use self-reported confidence scores $\gamma \in [0, 100]$, validated by behavioral frequency from 10 repeated forced-choice trials.
    - **Design Motivation**: Confidence signals differ fundamentally across architectures; a unified metric must accommodate this heterogeneity. Each implementation enables consistent trend analysis within its respective paradigm.

3. **Iconographic Abstraction Pipeline**:

    - **Function**: Converts high-fidelity images into low-fidelity iconographic images, preserving the semantic core while removing visual noise.
    - **Mechanism**: Gemini is applied in two stages—semantic distillation (retaining intentional meaning while discarding incidental scene details) and geometric reconstruction (constraining output to a flat, low-detail icon style). Generated images are verified by human annotators for semantic preservation and style compliance.
    - **Design Motivation**: Grounded in semantic anchorage theory—when visual signals become more "digital" (less analog), models are less prone to defaulting to literal interpretations and more inclined to adopt a symbolic stance.

### Loss & Training

This paper involves no model training; it is a purely evaluative work. DIVA comprises 1,000 iconographic images (100 noun compounds × 5 contrastive types × 2 semantic directions).

## Key Experimental Results

### Main Results

| Model | $\Delta$ (AdMIRe / High-Fidelity) | $\Delta$ (DIVA / Iconographic) | Reduction |
|---|---|---|---|
| SigLIP 2 | 0.245 | 0.178 | −27% |
| EVA-CLIP-18B | 0.262 | 0.191 | −27% |
| InternVL3-78B | 0.138 | 0.089 | −36% |
| Qwen2.5-VL-32B | 0.145 | 0.095 | −34% |
| LLaVA-OV-7B | 0.176 | 0.122 | −31% |
| GPT-5 | 0.065 | 0.021 | −68% |
| Claude 4.5 Sonnet | 0.072 | 0.028 | −61% |

### Ablation Study

| Analysis Dimension | Result |
|---|---|
| Discriminative vs. Generative | Discriminative models exhibit the largest $\Delta$ (~0.25); generative models are substantially lower (~0.14); closed-source models are lowest (~0.07) |
| Model Scale Effect | Within the same architecture family, larger models do not necessarily yield smaller $\Delta$—scale alone does not resolve literal bias |
| 5-way Selection Accuracy | Iconographic images improve accuracy across all model families (discriminative: 42.3→58.7%; closed-source: 78.5→91.3%) |

### Key Findings

- All models exhibit positive $b(t)$ (literal preference) under all conditions, with the effect more pronounced on high-fidelity images.
- Iconographic abstraction consistently reduces $\Delta$ across all architectural families—GPT-5 drops from 0.065 to 0.021, approaching zero bias.
- Discriminative models are most severely affected by "cognitive interference"—CLIP-family models over-rely on texture and surface features.
- Spearman correlation analysis reveals high agreement between human evaluation and the $\Delta$ metric ($\rho = 0.64$–$0.73$).

## Highlights & Insights

- Applying semiotic theory to VLM evaluation is a highly novel angle—it reframes "why models fail to understand figurative language" as a measurable position along the icon–symbol continuum.
- The counterintuitive finding that "high fidelity is cognitive interference" is particularly thought-provoking—more photorealistic images do not necessarily facilitate understanding, challenging the implicit assumption that sharper images are always better.
- The tri-fold design of the $\Delta$ metric elegantly resolves cross-architecture comparability.

## Limitations & Future Work

- The study is restricted to English noun compounds; cross-cultural metaphors (e.g., the Chinese idiom "iron rice bowl") are not addressed.
- The specific style of iconographic images (flat design) may introduce style bias—models familiar with this aesthetic may perform better for reasons unrelated to symbolic understanding.
- Self-reported confidence scores from closed-source models may reflect instruction-following tendencies rather than genuine semantic judgments.
- The work serves only as a diagnostic tool and does not propose methods for improving model performance.

## Related Work & Insights

- **vs. T2I-CompBench/GenEval**: These benchmarks focus on physical compositionality (e.g., a blue ball next to a red cube), whereas this paper addresses semantic compositionality—noun combinations that generate abstract meanings beyond their literal constituents.
- **vs. AdMIRe (SemEval-2025)**: AdMIRe evaluates whether models can align with idiomatic images, but the use of high-fidelity images introduces confounds; DIVA controls for visual complexity through iconographic abstraction.
- **vs. IconQA**: IconQA employs icon-style diagrams for reasoning tasks but does not address figurative meaning comprehension.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Semiotic perspective + iconographic abstraction hypothesis + unified cross-architecture metric; highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Eight models, three architectural paradigms, human evaluation validation; however, limited to English noun compounds.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretically elegant framework, rigorous methodology, clear argumentation.
- **Value**: ⭐⭐⭐⭐ Offers profound insight into literal bias in VLMs, but lacks prescriptive remedies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] More than the Sum: Panorama-Language Models for Adverse Omni-Scenes](../../CVPR2026/multimodal_vlm/more_than_the_sum_panorama-language_models_for_adverse_omni-scenes.md)
- [\[ACL 2026\] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models](medlaybench-v_a_large-scale_benchmark_for_expert-lay_semantic_alignment_in_medic.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](../../AAAI2026/multimodal_vlm/patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
