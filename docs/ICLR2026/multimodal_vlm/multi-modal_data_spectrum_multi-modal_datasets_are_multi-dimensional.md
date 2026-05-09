---
title: >-
  [Paper Note] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
description: >-
  [ICLR2026][Multimodal VLM][Multimodal benchmark evaluation] A large-scale empirical study reveals severe unimodal dependency issues across 23 VQA benchmarks — many benchmarks designed to eliminate text bias have instead introduced image bias, with models exploiting unimodal shortcuts rather than performing genuine cross-modal reasoning.
tags:
  - ICLR2026
  - Multimodal VLM
  - Multimodal benchmark evaluation
  - VQA
  - modality bias
  - unimodal shortcuts
  - MLLM
date: 2026-05-08
content_hash: 996a76fe08b8e334
---

# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional

**Conference**: ICLR2026  
**arXiv**: [2509.23499](https://arxiv.org/abs/2509.23499)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: Multimodal benchmark evaluation, VQA, modality bias, unimodal shortcuts, MLLM  

## TL;DR
A large-scale empirical study reveals severe unimodal dependency issues across 23 VQA benchmarks — many benchmarks designed to eliminate text bias have instead introduced image bias, with models exploiting unimodal shortcuts rather than performing genuine cross-modal reasoning.

## Background & Motivation
- Multimodal large language models (MLLMs) have achieved rapidly rising scores on various VQA benchmarks, but do these high scores truly reflect cross-modal understanding?
- Early VQA research identified the text-only bias problem (models answering correctly from the question alone), prompting the community to design a set of "debiased" benchmarks.
- However, whether these debiased benchmarks genuinely resolve the issue — or introduce new biases — has not been systematically quantified.
- A unified framework is needed to measure **intra-modality dependency** (predictability within a single modality) and **inter-modality dependency** (necessity of cross-modal interaction) in datasets.

## Method

### Overall Architecture
Given a multimodal dataset $\mathcal{D} = \{(\mathbf{x}_1, \mathbf{x}_2, \mathbf{y})\}$ (image, text, label), MLLM performance is measured under four input conditions to quantify modality dependency: (1) normal paired input; (2) image only (text randomly replaced); (3) text only (image randomly replaced); (4) fully random (both modalities replaced). Performance differences reveal the independent contribution of each modality and the necessity of cross-modal interaction.

### Key Designs
1. **Modality Shuffling**:

    - Zero-masking (e.g., blank images) or perturbation-based approaches are avoided, as they create unnatural out-of-distribution inputs.
    - Shuffling preserves the marginal distribution of each modality while disrupting only cross-modal correlations.
    - This more accurately quantifies inter-modality dependency without inducing unpredictable model behavior.

2. **Multi-model Ensemble Voting**:

    - Analysis based on a single model may be confounded by model-specific inductive biases.
    - Majority voting is performed across a diverse set of models: Cambrian-1 at three scales (8B, 13B, 34B), LLaVA-Next, Qwen2.5-VL, and Qwen3-VL.
    - This marginalizes single-model bias and yields more robust estimates of dataset-intrinsic dependency.

3. **Sub-category Granularity Analysis**:

    - Globally aggregated metrics may obscure unimodal dependency within sub-categories.
    - Modality shuffling is applied separately to different sub-categories of the same benchmark (e.g., question types, knowledge domains).
    - This reveals that benchmarks appearing "cross-modal" at the global level may still exhibit strong unimodal dependency in specific sub-categories.

4. **Systematic Evaluation Across 23 VQA Benchmarks**: Coverage includes general VQA (VizWiz, GQA, MME, etc.), expert VQA (ScienceQA, MathVista, MMMU, etc.), spatial understanding (POPE, RealWorldQA, etc.), and OCR/document understanding (TextVQA, ChartQA, etc.).

## Key Experimental Results

### Main Results (Modality Dependency Analysis Across 23 VQA Benchmarks)

| Dependency Type | Representative Benchmarks | Description |
|----------------|--------------------------|-------------|
| Cross-modal only | MME, POPE, COCO, V*Bench | Only 4/23 benchmarks genuinely require cross-modal reasoning |
| Text bias present | GQA (+26%), ScienceQA (+17.5%), MMMU (+11.4%) | Models answer correctly without viewing the image |
| Image bias present | MMBench (+41%), SEED, TextVQA, ChartQA | Models answer correctly without viewing the question |
| Dual bias | MMMU-Pro, Q-Bench, MM-Star | Both modalities independently provide shortcuts |

### Effect of Model Scale

| Model | MMBench Image Bias | MMMU Text Bias | POPE Cross-modal |
|-------|--------------------|----------------|-----------------|
| 8B | Moderate | Moderate | Random level |
| 13B | Increased | Increased | Random level |
| 34B | **Highest** | **Highest** | Random level |
| Ensemble | Highest | Highest | Random level |

### Cross-model Type Analysis

| Model | Release Date | MMBench Image Bias | GQA Text Bias |
|-------|-------------|-------------------|--------------|
| Cambrian-8B | 2024.06 | Medium | Medium |
| LLaVA-Next | 2024.05 | Medium | Medium |
| Qwen2.5-VL | 2025.04 | High | Medium |
| Qwen3-VL | 2025.11 | **Highest** | Medium |

### Key Findings
- Only 4 out of 23 benchmarks genuinely require cross-modal reasoning — the vast majority can be largely solved by unimodal signals alone.
- Many benchmarks designed to eliminate text bias have instead introduced stronger image bias (MMBench image bias: +41%).
- **Larger models are more adept at exploiting unimodal shortcuts** — the 34B model exhibits the highest image bias on MMBench.
- Modality dependency varies substantially across sub-categories within the same benchmark — global metrics mask unimodal dependency at the sub-category level.
- Consistent bias patterns are observed across different model architectures and generations — the problem lies in dataset design rather than in the models.

## Highlights & Insights
- The first systematic quantification of modality bias distributions across 23 mainstream VQA benchmarks, with findings that directly address a critical community pain point.
- The finding that "eliminating text bias → introducing image bias" is highly counter-intuitive and serves as an important warning for benchmark design.
- The conclusion that larger models are better at exploiting shortcuts challenges the "scale solves everything" assumption.
- The data spectrum framework is concise and general, and can be extended to other multimodal tasks (audio-visual VQA, multimodal retrieval, etc.).

## Limitations & Future Work
- The analysis is primarily based on black-box probing (text-only/image-only testing) and lacks examination of internal model mechanisms (e.g., attention, gradients).
- Coverage is limited to the VQA paradigm; multimodal generation, retrieval, and other task formats are not addressed.
- The quantitative metrics of the data spectrum are contingent on the capability ceiling of the specific MLLMs used; different models may yield different spectral distributions.
- No concrete recommendations are provided for designing benchmarks that genuinely require cross-modal reasoning.

## Related Work & Insights
- **VQA bias analysis**: VQA-CP, AdVQA — this paper extends point-wise bias analysis to systematic spectral analysis.
- **MLLM evaluation**: MMBench, MM-Vet — this paper reveals that these evaluations may overestimate cross-modal reasoning ability.
- **Dataset design**: Balanced VQA, CounterFactual VQA — eliminating one type of bias while inadvertently introducing another.
- **Insight**: Future multimodal benchmark design should first calibrate the bias distribution on the data spectrum to ensure that cross-modal reasoning is the genuine bottleneck rather than a shortcut.
- **Extended reflection**: The data spectrum methodology can be generalized to benchmark quality auditing for more modality combinations, such as audio-language and video-language.
- **Recommendation**: All multimodal benchmark papers should report image-only, text-only, and random baseline scores, rather than reporting only full-input performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The data spectrum framework is novel, with counter-intuitive and valuable findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 benchmarks × 36 models with multi-scale and multi-type validation — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; radar charts and sub-category visualizations are intuitive.
- Value: ⭐⭐⭐⭐⭐ Directly informative for community benchmark design and model evaluation standards.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Contamination Detection for VLMs using Multi-Modal Semantic Perturbation](contamination_detection_for_vlms_using_multi-modal_semantic_perturbation.md)
- [\[AAAI 2026\] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](../../AAAI2026/multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICLR 2026\] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [\[CVPR 2026\] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](../../CVPR2026/multimodal_vlm/crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)

<!-- RELATED:END -->
