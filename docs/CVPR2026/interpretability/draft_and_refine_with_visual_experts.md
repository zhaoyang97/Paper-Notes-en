---
title: >-
  [Paper Note] Draft and Refine with Visual Experts
description: >-
  [CVPR 2026][Interpretability][Visual Utilization Quantification] This paper proposes DnR (Draft and Refine), an agent framework built upon a question-conditioned Visual Utilization metric that quantifies the degree to wh…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Visual Utilization Quantification"
  - "Agent Framework"
  - "Hallucination Mitigation"
  - "Visual Expert Collaboration"
  - "Training-Free"
date: 2026-05-08
content_hash: 9058019db07d0e9b
---

# Draft and Refine with Visual Experts

**Conference**: CVPR 2026
**arXiv**: [2511.11005](https://arxiv.org/abs/2511.11005)
**Code**: [GitHub](https://github.com/SungheonJeong/DnR)
**Area**: Interpretability
**Keywords**: Visual Utilization Quantification, Agent Framework, Hallucination Mitigation, Visual Expert Collaboration, Training-Free

## TL;DR

This paper proposes DnR (Draft and Refine), an agent framework built upon a question-conditioned Visual Utilization metric that quantifies the degree to which LVLMs actually rely on visual evidence. Through iterative rendering feedback from external visual experts (detection, segmentation, OCR, etc.), DnR improves visual grounding and reduces hallucinations.

## Background & Motivation

**Hallucination in LVLMs**: Current large vision-language models over-rely on language priors rather than visual evidence, producing ungrounded hallucinated responses.

**Lack of Visual Utilization Quantification**: Existing methods cannot measure the extent to which LVLMs actually depend on visual input during inference.

**Limitations of Existing Tool-Calling Approaches**: Current agent systems invoke experts via language-driven CoT or text-based confidence scores, inheriting the inherent biases of language models.

**High Cost of Learning-Based Coordination Frameworks**: Jointly optimizing multiple experts requires expensive and inflexible joint training.

**Not All Visual Information Is Equal**: Different questions require attention to different image regions; globally enhancing visual dependency may instead introduce noise.

**Core Problem**: Can a VLM autonomously determine when and which visual expert to invoke based on its own perceptual needs, rather than language bias?

## Method

### Overall Architecture

DnR consists of four steps: (1) the LVLM generates an initial draft answer; (2) a question-conditioned relevance map $r(x|q)$ is constructed; (3) visual utilization $U_q(x)$ is computed via relevance-mask perturbation; (4) each expert's output is rendered onto the image, and the expert yielding the greatest utilization gain is selected for refinement.

### Key Designs

#### Query-Conditioned Relevance Map

An LLM decomposes the question $q$ into a set of visually addressable sub-queries $Q = \{q_1, ..., q_m\}$, and a CLIP-based localization model generates a spatial relevance map: $r(x|q) = \frac{1}{m} \sum_{q_i \in Q} R(x|q_i)$.

#### Question-Conditioned Visual Utilization

Gumbel-k sampling is applied over the relevance distribution to generate Top-k masks (occluding salient regions) and Bottom-k masks (occluding irrelevant regions). A semantic encoder $g(\cdot)$ measures the semantic deviation between original and masked predictions:

$$U_q(x) = \alpha \cdot \mathbb{E}_{\tau \in \mathcal{M}_q^{\text{top}}}[d_\tau] + (1-\alpha) \cdot \mathbb{E}_{\tau \in \mathcal{M}_q^{\text{bottom}}}[d_\tau]$$

The weight $\alpha$ is adaptively determined by the entropy and contrast of the relevance map.

#### Expert Selection and Rendering Integration

Structured outputs from each expert (CLIP, SAM, OCR, etc.) are rendered onto the original image (via graying, blurring, highlighting, etc.), and the LVLM is re-queried. The expert is selected as $j^* = \arg\max_j (U_q^{(j)} - U_q^{\text{base}})_+$. If no expert yields improvement, the refinement step is skipped. A lightweight trainable selector $S_\theta$ can replace exhaustive evaluation.

### Loss & Training

The selector is trained using cross-entropy loss $\mathcal{L} = -\mathbb{E}[\log S_\theta(j^*|s)]$. The main framework is training-free.

## Key Experimental Results

### Main Results: IDEFICS on Multiple Benchmarks — Draft vs. DnR

| Benchmark | Draft | DnR | Gain |
|-----------|-------|-----|------|
| VQAv2 | 37.8 | 47.85 | +10.05 |
| GQA | 24.1 | 25.5 | +1.4 |
| VCR | 15.58 | 21.11 | +5.53 |
| VSR | 52.76 | 54.27 | +1.51 |
| MME | 1392 | 1432 | +40 |

### Ablation Study

| Dimension | Key Findings |
|-----------|------|
| Revision Rate | Varies substantially across tasks (VQAv2: 29.8%, GQA: 1.5%) |
| Correction/Degradation | VQAv2: 46.2% correction vs. 14.3% degradation |
| Pearson/Spearman Correlation | GQA: 0.449/0.364; VCR: 0.38/0.421 |

### Key Findings

- Visual utilization exhibits significant positive correlation with task accuracy.
- Revision rates are highest on tasks requiring fine-grained visual understanding.
- Rendering strategies (graying, blurring, highlighting) vary in effectiveness across experts and tasks.
- The framework integrates new experts without retraining.

## Highlights & Insights

- The paper is the first to introduce a **quantifiable visual utilization metric**, providing a measurable standard for evaluating visual grounding in VLMs.
- The rendering mechanism is elegantly designed — it converts structured expert outputs into visual cues directly consumable by the LVLM without architectural modification.
- Utilization-driven expert selection is more reliable than language-driven CoT, as it is grounded in the model's actual perceptual behavior.
- The framework is highly modular; new experts can be integrated in a plug-and-play manner.

## Limitations & Future Work

- Rendering strategies and hyperparameters require tuning per dataset and model.
- Repeated masking and LVLM re-querying incurs substantial inference overhead.
- Exhaustive expert evaluation scales linearly with the number of experts (mitigable via the lightweight selector).
- The visual utilization metric depends on the quality of the relevance map.

## Related Work & Insights

- Compared to programmatic reasoning agents such as VisProg, DnR requires no code execution.
- Compared to hallucination mitigation methods (e.g., VCD, OPERA), DnR takes a more principled approach by addressing the problem from the perspective of visual utilization.
- The rendering integration paradigm may inspire tool-calling frameworks in other domains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)
- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](pixel2phys_distilling_governing_laws_from_visual_dynamics.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](language_models_can_explain_visual_features_via_steering.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)

</div>

<!-- RELATED:END -->
