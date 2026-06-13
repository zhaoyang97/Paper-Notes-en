---
title: >-
  [Paper Note] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
description: >-
  [ICLR 2026][Multimodal VLM][multimodal benchmark evaluation] Through a large-scale empirical study, this work quantifies intra-modal and inter-modal dependencies across 23 VQA benchmarks…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "multimodal benchmark evaluation"
  - "modality dependency"
  - "VQA"
  - "dataset bias"
  - "MLLM evaluation"
date: 2026-05-08
content_hash: 42cff9ffedabde2a
---

# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional

**Conference**: ICLR 2026
**arXiv**: [2509.23499](https://arxiv.org/abs/2509.23499)  
**Code**: [GitHub](https://github.com/divyam3897/multimodal-spectrum)  
**Area**: Signal Communication
**Keywords**: multimodal benchmark evaluation, modality dependency, VQA, dataset bias, MLLM evaluation

## TL;DR

Through a large-scale empirical study, this work quantifies intra-modal and inter-modal dependencies across 23 VQA benchmarks, revealing that most benchmarks contain severe unimodal shortcuts and that eliminating text bias tends to introduce image bias. A quantitative evaluation framework for multimodal benchmark design is proposed.

## Background & Motivation

1. **Background**: The rapid development of multimodal large language models (MLLMs) has been accompanied by a proliferation of evaluation benchmarks—over 200 multimodal benchmarks have been proposed—yet systematic understanding of what these datasets actually measure remains lacking.

2. **Limitations of Prior Work**: Relationships, redundancies, and unique contributions among benchmarks remain unclear. The addition or removal of benchmarks across evaluation cycles lacks justification (e.g., Gemini 2.5 omits several benchmarks used by Gemini 1.5), making it difficult to determine whether performance gains reflect genuine capability improvements or adaptation to different biases.

3. **Key Challenge**: Benchmark design has fallen into a "cat-and-mouse game"—new datasets are created to eliminate text bias, yet inadvertently introduce image bias; models achieve high scores via unimodal shortcuts that do not reflect genuine multimodal understanding.

4. **Goal**: To systematically quantify the strength of intra-modal dependencies (answerable from a single modality) and inter-modal dependencies (requiring interaction between both modalities) in existing multimodal benchmarks.

5. **Key Insight**: A modality shuffling method is employed that breaks inter-modal associations while preserving the marginal distribution of each modality, measuring performance degradation to quantify each modality's contribution.

6. **Core Idea**: Multimodal datasets are inherently multi-dimensional—each benchmark differs in its degree of visual dependency, textual dependency, and cross-modal interaction dependency, necessitating multi-dimensional characterization rather than a single aggregated score.

## Method

### Overall Architecture

Four evaluation conditions are proposed: (1) normally paired input $\mathcal{M}(f_\theta(\mathbf{x_1}, \mathbf{x_2}), \mathbf{y})$; (2) image-only (text randomly replaced); (3) text-only (image randomly replaced); (4) fully random (both modalities replaced). Intra-modal and inter-modal dependencies are quantified by comparing performance across these four conditions. A multi-model majority-vote ensemble is used to reduce the influence of individual model biases.

### Key Designs

**1. Modality Shuffling Evaluation**

- **Function**: Quantifies the independent and interactive contributions of each modality within a dataset.
- **Mechanism**: One modality's samples are randomly shuffled within the dataset, preserving alignment between the other modality and the labels. The degree of performance degradation reflects that modality's contribution. Compared to zeroing out (blank images) or perturbation-based methods, shuffling preserves each modality's marginal distribution and avoids out-of-distribution inputs.
- **Design Motivation**: A simple method is needed to disentangle modality contributions without introducing confounding out-of-distribution factors.

**2. Multi-granularity Analysis Framework**

- **Function**: Prevents aggregate metrics from masking unimodal dependencies present in subcategories.
- **Mechanism**: Analysis is conducted not only at the dataset level but also at the subcategory level (e.g., question type, knowledge domain), revealing strong unimodal biases in subgroups of datasets that appear balanced globally.
- **Design Motivation**: For instance, COCO as a whole appears to be a cross-modal interaction dataset, yet its "relative position" subcategory exhibits pronounced textual dependency.

**3. Robustness Validation Across Scales and Architectures**

- **Function**: Verifies that observed modality dependencies are intrinsic dataset properties rather than artifacts of specific models.
- **Mechanism**: Cambrian-1 models at 8B/13B/34B scales and their ensemble, along with architecturally distinct models including LLaVA-Next, Qwen2.5-VL, and Qwen3-VL, are used to confirm consistency of findings.
- **Design Motivation**: Multimodal dependency is a function of both data and model; marginalizing over model influence is necessary to obtain intrinsic data properties.

### Loss & Training

This is an analytical study with no training involved. Standard accuracy is used as the evaluation metric, and modality dependency is quantified by comparing accuracy differences across the four input conditions.

## Key Experimental Results

### Main Results

The 23 benchmarks are categorized by modality dependency type:

| Category | Datasets | Typical Behavior |
|----------|----------|-----------------|
| Cross-modal interaction only | MME, POPE, COCO, V*Bench | Only 4 datasets; performance degrades to chance after modality shuffling |
| Text dependency | GQA (+26%), ScienceQA (+17.5%), MMMU (+11.35%) | Text alone far exceeds chance performance |
| Image dependency | MMBench (+41%), SEED, TextVQA, ChartQA | Image alone far exceeds chance performance |
| Dual dependency | MMMU-Pro, MathVista | Both modalities independently contribute |

### Ablation Study

Effect of model scale on modality dependency:

| Dataset | 8B→34B Trend | Notes |
|---------|-------------|-------|
| MMMU | Both image and text dependency increase | Larger models are better at exploiting unimodal shortcuts |
| MMBench | Image dependency increases | Scale exacerbates rather than alleviates bias |
| POPE | No change | Pure interaction dataset unaffected by scale |
| AI2D | Text dependency increases | Larger models rely more on textual priors |

### Key Findings

- **Truly multimodal benchmarks are exceedingly rare**: Only 4 of 23 benchmarks exhibit purely cross-modal interaction dependency.
- **Eliminating text bias ≠ achieving multimodality**: Many efforts merely replace text dependency with image dependency.
- **Model scale amplifies bias**: Larger models do not automatically learn better multimodal reasoning; they become more adept at exploiting unimodal shortcuts.
- **Aggregate metrics are misleading**: Globally balanced datasets may harbor severe unimodal dependencies at the subcategory level.

## Highlights & Insights

- **Exposes a fundamental problem in multimodal evaluation**: Most benchmarks do not genuinely test multimodal capabilities.
- **Quantitative framework directly actionable**: Provides an operational tool for future benchmark design.
- **Reflection on the field's direction**: Chasing leaderboard scores may not reflect genuine progress in multimodal capability.
- **Recommends reporting modality-specific baselines**: In addition to overall scores, image-only, text-only, and random baselines should be reported.

## Limitations & Future Work

- Analysis is limited to the MC-VQA format and does not cover open-ended generation tasks.
- The multiple-choice format itself may reduce the demand for cross-modal interaction.
- Automatic remediation of unimodal biases in existing datasets is not explored.
- Extension to evaluation of additional modalities such as video and audio remains for future work.

## Related Work & Insights

- Perceptual Score (Gat et al., 2021) provides the methodological foundation for quantifying modality contributions.
- Debiasing efforts such as VQA-CP demonstrate the difficulty of designing unbiased benchmarks.
- Insight: Advances in evaluation methodology may be more valuable than the proliferation of evaluation benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale systematic quantification of modality dependencies across multimodal benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 benchmarks, multiple models, multiple scales, and subcategory analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear analysis and excellent visualizations.
- Value: ⭐⭐⭐⭐⭐ Profound methodological impact on multimodal evaluation.

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [\[AAAI 2026\] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](../../AAAI2026/multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [\[ICLR 2026\] Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](../../NeurIPS2025/multimodal_vlm/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)

</div>

<!-- RELATED:END -->
