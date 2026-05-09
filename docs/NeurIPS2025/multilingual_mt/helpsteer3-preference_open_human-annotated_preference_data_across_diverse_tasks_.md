---
title: >-
  [Paper Note] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages
description: >-
  [NeurIPS 2025][Preference Dataset] NVIDIA releases a 40K+ open-source human-annotated preference dataset covering general, STEM, code, and multilingual (13 languages) tasks. The reward model trained on this dataset achieves 82.4% (+10%) on RM-Bench, with a commercially friendly CC-BY-4.0 license.
tags:
  - NeurIPS 2025
  - Preference Dataset
  - Human Annotation
  - STEM
  - Code
  - Multilingual
  - Reward Models
  - CC-BY-4.0
date: 2026-05-08
content_hash: d1c7feba3668dcac
---

# HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages

**Conference**: NeurIPS 2025
**arXiv**: [2505.11475](https://arxiv.org/abs/2505.11475)
**Code**: [https://huggingface.co/datasets/nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)
**Area**: Multilingual Translation
**Keywords**: Preference Dataset, Human Annotation, STEM, Code, Multilingual, Reward Models, CC-BY-4.0

## TL;DR
NVIDIA releases a 40K+ open-source human-annotated preference dataset covering general, STEM, code, and multilingual (13 languages) tasks. The reward model trained on this dataset achieves 82.4% (+10%) on RM-Bench, with a commercially friendly CC-BY-4.0 license.

## Background & Motivation

**State of the Field**: Preference data has evolved from low-quality annotations (HH-RLHF) to GPT-4-labeled data (UltraFeedback) to synthetically filtered datasets (HelpSteer2), yet diversity remains insufficient — nearly all mainstream datasets are English-only.

**Limitations of Prior Work**: As LLM applications expand into programming, scientific reasoning, and multilingual interaction, RLHF data must cover these emerging domains. GPT-4-annotated data is restricted from commercial use under terms of service.

**Root Cause**: High quality, diversity, and permissive licensing are difficult to achieve simultaneously — prior datasets satisfy at most two of these three requirements.

**Paper Goals**: Construct a large-scale, high-quality, permissively licensed preference dataset spanning STEM, code, and multilingual domains.

**Starting Point**: Engage annotators with domain-specific expertise (scientists, engineers, multilingual speakers) under a stratified quality control framework.

**Core Idea**: Expert-stratified annotation + multi-domain and multilingual coverage + CC-BY-4.0 = the most comprehensive open-source preference dataset to date.

## Method

### Overall Architecture
Stratified data collection: different subsets are sourced differently (ShareGPT/WildChat), with 17 models generating paired responses, 3–5 independent expert annotators providing 7-point ratings with rationales, and rigorous post-processing applied throughout.

### Key Designs

1. **Stratified Annotator Expertise**:

    - General: everyday scenarios; STEM: relevant degree and work experience; Code: software engineering background for code quality evaluation; Multilingual: language fluency to verify correct target-language usage.
    - Weighted Cohen's $\kappa$ reaches 0.890 (strong agreement); positional bias is negligible (mean preference shift $-0.003$).

2. **Broad Coverage**:

    - Code (8,857 samples): 14 programming languages, Python at 38.2%.
    - Multilingual (8,063 samples): 13 natural languages, Chinese at 30.2%.
    - STEM (4,918 samples): high-difficulty scientific problems.
    - General (18,638 samples): general-purpose scenarios.

3. **Multi-Level Quality Control**:

    - 3–5 independent annotators → retain the 3 most consistent → remove invalid entries and filter outliers.
    - Bias detection: positional bias $< 0.003$, standard deviation $1.950$.

## Key Experimental Results

### Main Results

| Dataset / RM | RM-Bench Overall | RM-Bench Hard | JudgeBench |
|---|---|---|---|
| Trained on HelpSteer2 | ~72% | ~61% | ~63% |
| English RM (Gen+STEM+Code) | 79.9% | 71.1% | **73.7%** |
| **Multilingual RM** | **82.4%** | **80.0%** | 69.4% |

### Dataset Statistics Comparison

| Metric | HelpSteer3 | HelpSteer2 |
|---|---|---|
| Total Samples | 40,476 | 9,125 |
| Avg. Context Length | 2,638 | 711 |
| Cohen's $\kappa$ | 0.890 | 0.878 |

### Key Findings
- RM-Bench Hard improves from ~56% to 80% (+24%); high-quality data provides the greatest benefit on difficult samples.
- A 4.4× scale-up with substantially expanded diversity is achieved while maintaining high inter-annotator agreement ($\kappa = 0.890$).
- The CC-BY-4.0 license enables unrestricted commercial use.

## Highlights & Insights
- **First systematic treatment of diversity**: STEM + Code + Multilingual breaks the precedent of English-only preference data.
- **Quality and scale achieved simultaneously**: a 4× expansion while maintaining $\kappa > 0.89$.
- **+10% on RM-Bench** demonstrates that data quality is the critical bottleneck for reward model performance.

## Limitations & Future Work
- Multilingual distribution is uneven (Chinese at 30%; some languages are underrepresented).
- Coverage is text-only; multimodal preferences are not addressed.
- Annotation of certain subjective tasks may remain contentious.

## Related Work & Insights
- **vs. HelpSteer2**: 4.4× larger with multi-domain and multilingual expansion.
- **vs. UltraFeedback**: human annotation vs. GPT-4 annotation, with no licensing restrictions.
- **vs. Skywork-Preference**: higher annotation quality ($\kappa = 0.890$) with broader dimensional coverage.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematically constructed multi-dimensional preference dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple RM training runs, multi-benchmark evaluation, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Data construction pipeline described in thorough detail.
- Value: ⭐⭐⭐⭐⭐ CC-BY-4.0 open release with a +10% RM improvement; directly applicable to industrial settings.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection](dcad-2000_a_multilingual_dataset_across_2000_languages_with_data_cleaning_as_ano.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)
- [\[NeurIPS 2025\] How Data Mixing Shapes In-Context Learning: Asymptotic Equivalence for Transformers with MLPs](how_data_mixing_shapes_in-context_learning_asymptotic_equivalence_for_transforme.md)
- [\[NeurIPS 2025\] Quantifying Climate Policy Action and Its Links to Development Outcomes: A Cross-National Data-Driven Analysis](quantifying_climate_policy_action_and_its_links_to_development_outcomes_a_cross-.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/multilingual_mt/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)

<!-- RELATED:END -->
