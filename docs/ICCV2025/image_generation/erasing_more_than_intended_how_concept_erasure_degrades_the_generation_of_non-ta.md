---
title: >-
  [Paper Note] Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts
description: >-
  [ICCV 2025][Image Generation][Concept Erasure] This paper systematically analyzes the unintended negative effects of concept erasure techniques on non-target concepts (spillover degradation) in text-to-image models. It proposes EraseBench, a comprehensive evaluation framework covering multiple dimensions including visual similarity, binomial association, and semantic relatedness. The findings reveal that current state-of-the-art erasure methods remain unreliable in preserving the generation quality of non-target concepts.
tags:
  - ICCV 2025
  - Image Generation
  - Concept Erasure
  - EraseBench
  - Concept Entanglement
  - Spillover Degradation
  - Text-to-Image Safety
date: 2026-05-08
content_hash: b7c736cc6511fddf
---

# Erasing More Than Intended? How Concept Erasure Degrades the Generation of Non-Target Concepts

**Conference**: ICCV 2025
**arXiv**: [2501.09833](https://arxiv.org/abs/2501.09833)
**Code**: None
**Area**: Diffusion Models / Text-to-Image Generation Safety
**Keywords**: Concept Erasure, EraseBench, Concept Entanglement, Spillover Degradation, Text-to-Image Safety

## TL;DR

This paper systematically analyzes the unintended negative effects of concept erasure techniques on non-target concepts (spillover degradation) in text-to-image models. It proposes EraseBench, a comprehensive evaluation framework covering multiple dimensions including visual similarity, binomial association, and semantic relatedness. The findings reveal that current state-of-the-art erasure methods remain unreliable in preserving the generation quality of non-target concepts.

## Background & Motivation

Concept erasure techniques aim to remove undesirable concepts (e.g., NSFW content, copyrighted material) from text-to-image (T2I) models and are regarded as an important means of safe model deployment. However, this paper identifies a severely overlooked problem — **Concept Entanglement**.

Specifically, when a concept is erased, other concepts that are visually similar, semantically related, or binomially paired with it are also unintentionally affected, manifesting as:

**Over-Erasure**: Degraded T2I alignment for non-target concepts — e.g., after erasing "cat," the model also fails to generate "tiger" correctly.

**Artifacts**: Generated images of non-target concepts exhibit misaligned body parts, cropped subjects, or distorted text.

**Style Leakage**: After erasing one artist's style, closely related artists cannot be rendered correctly either.

**Concept Leakage**: When too many related concepts are introduced into the retain set, previously erased concepts are partially recovered.

The paper's central argument is that **existing concept erasure evaluation frameworks are overly simplistic — they focus solely on whether target concepts are successfully removed, while neglecting the cascading effects on related concepts.** In real-world deployment, such uncontrolled spillover degradation undermines the reliability of "sanitized" models.

## Method

### Overall Architecture

Rather than proposing a new erasure method, this paper constructs a comprehensive evaluation benchmark, **EraseBench**, and uses it to systematically evaluate five state-of-the-art erasure techniques (ESD, UCE, Receler, MACE, AdvUnlearn). The EraseBench pipeline consists of: concept collection → human verification → diverse prompt construction → multi-dimensional evaluation.

### Key Designs

1. **EraseBench Multi-Dimensional Evaluation Framework**: Four relational dimensions between concepts are defined to test the side effects of erasure:

    - **Visual Similarity**: After erasing "cat," visually similar concepts such as "tiger" and "cheetah" are tested.
    - **Artistic Similarity**: After erasing Van Gogh, stylistically related artists such as Cézanne and Bernard are tested.
    - **Subset–Superset Relations**: After erasing "goldfish," related concepts such as "guppy" and "koi" are tested.
    - **Binomial Relations**: After erasing "sun," the closely associated concept "moon" is tested.

   Each dimension includes multiple primary concepts (used for erasure) and related non-target concepts (used for side-effect evaluation). Concept collection leverages LLMs and ImageNet hierarchical taxonomy, with human verification to ensure successful generation by the T2I model.

2. **Three-Dimensional Evaluation Metric System**:

    - **Efficacy (Eff.)** ↓: Effectiveness of target concept erasure (CLIP zero-shot classification accuracy).
    - **Generality (Gen.)** ↓: Erasure generalization to paraphrased or synonymous concepts.
    - **Sensitivity (Sens.)** ↑: Preservation of non-target but related concepts (a new metric and the core contribution of this paper).
    - RAHF (aesthetics + artifact scoring) and Gecko (VQA-based alignment scoring) are additionally employed for multi-dimensional quality evaluation.

3. **Gecko VQA Evaluation Pipeline**: Gemini 1.5 is used for a two-step VQA evaluation — first generating relevant questions from the text prompt, then answering them based on the generated image. The final score is the mean proportion of correctly answered questions, supporting retrospective analysis of which textual aspects are misaligned with the generated image.

### Loss & Training

As this is an evaluation study, no new model training is involved. The five baseline methods cover the major technical paradigms of concept erasure:
- **ESD**: Fine-tuning model weights.
- **UCE**: Introducing targeted weight perturbations.
- **Receler**: Adversarial training with parameter-efficient fine-tuning.
- **MACE**: Parameter-efficient fine-tuning (LoRA).
- **AdvUnlearn**: Adversarial training with textual embedding optimization.

## Key Experimental Results

### Main Results

CLIP zero-shot classification results across four dimensions (averaged over 10+ concepts):

| Dimension | Method | Eff. ↓ | Gen. ↓ | Sens. ↑ | HM ↑ |
|-----------|--------|--------|--------|---------|------|
| Visual Sim. | Original | 86.5 | 90.2 | 85.0 | 15.97 |
| Visual Sim. | ESD | 24.5 | 50.5 | 65.9 | 61.70 |
| Visual Sim. | UCE | 41.8 | 68.3 | **82.7** | 49.32 |
| Visual Sim. | Receler | **8.1** | **20.0** | 65.4 | 77.58 |
| Visual Sim. | MACE | 15.6 | 37.7 | 66.4 | 69.83 |
| Visual Sim. | AdvUnlearn | 8.7 | 39.1 | 64.3 | 69.88 |
| Binomial | UCE | 18.9 | 31.4 | **86.1** | **77.88** |
| Binomial | Receler | 10.3 | **12.6** | 57.5 | 75.04 |

UCE achieves the highest Sensitivity (best preservation of non-target concepts) but underperforms other methods in Efficacy.

### Ablation Study

Gecko VQA evaluation results (6,246 text–image pairs):

| Method | Erased Concepts ↓ | Non-Erased Concepts ↑ | Drop |
|--------|-------------------|----------------------|------|
| Original | 84.1 | 77.6 | — |
| UCE | 57.6 (−26.4) | 74.3 (−3.4) | Smallest |
| MACE | 38.2 (−45.9) | 67.9 (−9.8) | Largest |
| AdvUnlearn | 43.1 (−41.0) | 68.6 (−9.0) | Moderate |

The score drops on non-erased concepts are small but statistically significant (Wilcoxon rank-sum test, $\alpha = 0.01$).

### Key Findings

- **Core Finding**: All five SOTA erasure methods cause Sensitivity degradation in non-target concepts, indicating that concept erasure inevitably produces spillover effects.
- **Retain Set Dilemma**: Including visually similar concepts in the retain set can partially mitigate over-erasure, but at the cost of increased concept leakage — previously erased concepts are partially recovered.
- **Anchor Concept Limitations**: Using anchor concepts (e.g., Post-Impressionism as an anchor for Van Gogh) does not consistently improve the quality of non-target concepts.
- **Intra-type Multi-Concept Erasure Outperforms Inter-type**: Simultaneously erasing multiple related concepts reduces artifacts more effectively than erasing unrelated concepts (78.3% vs. 71.4%).
- Human preference experiments (1,650+ responses) validate the conclusions drawn from automated metrics.

## Highlights & Insights

- This is the first systematic study to reveal the spillover degradation problem in concept erasure, filling a critical gap in evaluation methodology.
- The multi-dimensional design of EraseBench (visual / artistic / subset–superset / binomial) provides a standardized testing framework for future research.
- The introduction of the Sensitivity metric is a key innovation, extending evaluation from "how thoroughly is the concept erased" to "how much collateral damage is caused."
- The retain set dilemma (mitigating over-erasure vs. concept leakage) exposes a fundamental tension in current methods.

## Limitations & Future Work

- Experiments are conducted solely on Stable Diffusion v1.4, without covering newer architectures such as SD3 or Flux.1.
- Concept selection still relies on manual curation; automated concept space search could improve scalability.
- No concrete solution is proposed to address concept entanglement, as this is purely an evaluation study.
- Theoretical analysis of why concept entanglement occurs is absent.

## Related Work & Insights

- Unlike Pham et al.'s concept recovery attacks, this paper focuses on the degradation of legitimate non-target concepts rather than adversarial recovery.
- EraseBench can serve as a standard evaluation benchmark for future concept erasure methods.
- Implication: Future erasure methods may need to perform finer-grained disentanglement in the concept representation space, rather than relying on simple weight modification.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel and important problem formulation; comprehensive EraseBench design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five methods, four dimensions, three metrics, and human evaluation — exceptionally thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-organized findings, rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Important cautionary contribution to AI safety; the benchmark has lasting value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICLR 2026\] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution](../../ICLR2026/image_generation/concept-trak_understanding_how_diffusion_models_learn_concepts_through_concept-l.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)

<!-- RELATED:END -->
