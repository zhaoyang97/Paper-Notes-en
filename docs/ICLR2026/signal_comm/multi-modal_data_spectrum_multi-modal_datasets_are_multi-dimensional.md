---
title: >-
  [Paper Note] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
description: >-
  [ICLR 2026][Signal & Communication][multimodal learning] Through a large-scale empirical study, intra-modal and inter-modal dependencies are quantified across 23 VQA benchmarks, revealing that many benchmarks designed to eliminate text bias inadvertently introduce image bias. A multi-dimensional characterization framework for multimodal datasets is proposed.
tags:
  - ICLR 2026
  - "Signal & Communication"
  - multimodal learning
  - benchmark evaluation
  - modality dependency
  - VQA
  - MLLM
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

1. **State of the Field**: The rapid development of multimodal large language models (MLLMs) has been accompanied by a proliferation of evaluation benchmarks—over 200 multimodal benchmarks have been proposed—yet systematic understanding of what these datasets actually measure remains lacking.

2. **Limitations of Prior Work**: Relationships, redundancies, and unique contributions among benchmarks remain unclear. The addition or removal of benchmarks across evaluation cycles lacks justification (e.g., Gemini 2.5 omits several benchmarks used by Gemini 1.5), making it difficult to determine whether performance gains reflect genuine capability improvements or adaptation to different biases.

3. **Root Cause**: Benchmark design has fallen into a "cat-and-mouse game"—new datasets are created to eliminate text bias, yet inadvertently introduce image bias; models achieve high scores via unimodal shortcuts that do not reflect genuine multimodal understanding.

4. **Paper Goals**: To systematically quantify the strength of intra-modal dependencies (answerable from a single modality) and inter-modal dependencies (requiring interaction between both modalities) in existing multimodal benchmarks.

5. **Starting Point**: A modality shuffling method is employed that breaks inter-modal associations while preserving the marginal distribution of each modality, measuring performance degradation to quantify each modality's contribution.

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

# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional

**Conference**: ICLR 2026
**arXiv**: [2509.23499](https://arxiv.org/abs/2509.23499)
**Code**: [GitHub](https://github.com/divyam3897/multimodal-spectrum)
**Area**: signal_comm
**Keywords**: multimodal learning, benchmark evaluation, modality dependency, VQA, MLLM

## TL;DR

Through a large-scale empirical study, intra-modal and inter-modal dependencies are quantified across 23 VQA benchmarks, revealing that many benchmarks designed to eliminate text bias inadvertently introduce image bias. A multi-dimensional characterization framework for multimodal datasets is proposed.

## Background & Motivation

1. **State of the Field**: MLLMs are developing rapidly, accompanied by the emergence of over 200 evaluation benchmarks, yet systematic understanding of what these benchmarks actually measure remains absent.

2. **Limitations of Prior Work**: Benchmark selection lacks scientific grounding—the datasets used in Gemini 1.5 and Gemini 2.5 evaluations differ without explanation. It is difficult to determine whether model performance improvements represent genuine multimodal capability gains or exploitation of unimodal shortcuts.

3. **Root Cause**: Benchmark development is caught in a "cat-and-mouse game"—new datasets are designed to eliminate specific unimodal biases but are subsequently found to introduce new ones (e.g., VQA → VQAv2 → VQA-CP → MMMU → MMMU-Pro).

4. **Paper Goals**: To conduct a systematic modality dependency analysis of existing multimodal benchmarks and provide a quantitative characterization framework.

5. **Starting Point**: A modality shuffling method is employed that disrupts inter-modal dependencies while preserving unimodal marginal distributions, measuring performance changes across four input conditions.

6. **Core Idea**: Multimodal datasets are inherently multi-dimensional; the strength of intra-modal dependencies (answerable from a single modality) and inter-modal dependencies (requiring joint reasoning) varies substantially both within and across benchmarks.

## Method

### Overall Architecture

A diagnostic framework with four evaluation conditions is proposed: (1) **Paired modalities (normal)**: standard performance on original paired data; (2) **Image-only**: text is replaced by text from a random sample, isolating image contribution; (3) **Text-only**: image is replaced by an image from a random sample, isolating text contribution; (4) **Random**: both modalities are randomly replaced, establishing a baseline. A multi-model majority-vote ensemble is used to eliminate individual model biases.

### Key Designs

**1. Modality Shuffling Rather Than Zeroing**

- **Function**: Disrupts inter-modal dependencies while preserving the marginal distribution of each modality.
- **Mechanism**: One modality's input is replaced by the corresponding modal input from another sample in the same dataset, rather than using blank images or empty strings. The model still receives valid inputs, but the cross-modal alignment is broken.
- **Design Motivation**: Zeroing or adding perturbations creates unnatural out-of-distribution inputs, inducing unpredictable model behavior that confounds the measurement of modality dependency.

**2. Subcategory-level Granular Analysis**

- **Function**: Reveals subgroup-level biases that may be masked by aggregate metrics.
- **Mechanism**: Datasets are partitioned by question type, object category, and other features; the modality shuffling diagnostic is applied independently to each subset.
- **Design Motivation**: Datasets that appear globally balanced may exhibit strong unimodal dependencies within specific subcategories.

**3. Validation Across Model Scales and Architectures**

- **Function**: Ensures findings reflect intrinsic data properties rather than model-specific biases.
- **Mechanism**: A majority-vote ensemble of Cambrian-1 models at 8B/13B/34B scales is used, with additional validation on architecturally distinct models including LLaVA-Next, Qwen2.5-VL, and Qwen3-VL.
- **Design Motivation**: Modality dependency is a joint function of data and model; marginalizing over individual model influence is necessary to obtain a robust estimate of the data's intrinsic dependency characteristics.

### Loss & Training

This is an analytical study with no model training. Accuracy is used as the core evaluation metric; modality dependency strength is quantified by the difference in accuracy across the four conditions.

## Key Experimental Results

### Main Results

Modality dependency classification of 23 benchmarks:

| Dependency Type | Representative Datasets | Characteristics |
|----------------|------------------------|-----------------|
| Inter-modal dependency only | MME, POPE, COCO, V*Bench | Very few—only 4/23 datasets |
| Includes text intra-modal dependency | GQA (+26%), ScienceQA (+17.5%), MMMU (+11.35%), AI2D (+34.94%) | Text alone substantially exceeds chance |
| Includes image intra-modal dependency | MMBench (+41%), SEED, TextVQA, MMMU-Pro, MMVP | Eliminating text bias introduces image bias |

### Ablation Study

| Configuration | Key Finding | Notes |
|--------------|------------|-------|
| Increasing model scale (8B→34B) | Unimodal bias does not decrease but increases | Larger models exhibit greater image and text dependency on MMMU |
| Different model types | Bias patterns are consistent across models | Cambrian, LLaVA-Next, and Qwen models behave similarly |
| Subcategory analysis | Aggregate metrics mask subgroup biases | Higher-grade ScienceQA questions rely almost entirely on text |

### Key Findings

- **Only 4/23 benchmarks** exhibit purely cross-modal interaction dependency—far fewer than expected.
- New benchmarks designed to eliminate text bias (e.g., MMBench, SEED) instead introduce image bias, replacing one unimodal shortcut with another.
- Increasing model scale does not mitigate unimodal bias and may exacerbate it.
- Subcategory analysis reveals that even globally balanced datasets exhibit strong biases in specific subsets.

## Highlights & Insights

- **Exposes the fundamental problem of multimodal evaluation**: Assessing models with a single aggregated score is insufficient; unimodal baseline performance must also be reported.
- **Provides a tool for judging whether model progress is genuine**: Performance improvements may merely reflect greater exploitation of unimodal dependencies.
- **Practical guidelines for new benchmark design**: The core objective should be to require both modalities jointly for correct answers, rather than merely eliminating dependency on one modality.
- **Profound insight into the "cat-and-mouse game"**: Only systematic quantification of modality dependency can break this cycle.

## Limitations & Future Work

- Analysis is limited to multiple-choice VQA format and does not cover open-ended generation tasks.
- The modality shuffling method may have limitations when answer options themselves contain modal information.
- Extension to evaluating models' ability to abstain is needed.
- Future work should advance benchmark design for open-ended answer generation and evaluation.

## Related Work & Insights

- Perceptual Score (Gat et al., 2021) provides the foundational methodology, which this work extends to large-scale analysis across 23 benchmarks.
- Analogous to the insight in NAS that "search space design matters more than the search algorithm"—benchmark design is more fundamental than model improvement.
- Insight: Reporting modality-specific baselines when evaluating multimodal models should become a community norm.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic analysis reveals an important and overlooked problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 benchmarks, multiple model scales and types, subcategory analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic and outstanding visualizations.
- Value: ⭐⭐⭐⭐ Significant methodological guidance for the multimodal evaluation community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](../../NeurIPS2025/signal_comm/multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/signal_comm/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[AAAI 2026\] Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](../../AAAI2026/signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)

<!-- RELATED:END -->
