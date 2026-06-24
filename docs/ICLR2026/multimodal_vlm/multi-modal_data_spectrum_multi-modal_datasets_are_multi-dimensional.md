---
title: >-
  [Paper Note] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
description: >-
  [ICLR 2026][Multimodal VLM][Multi-modal learning] This work quantifies intra-modality and inter-modality dependencies across 23 VQA benchmarks through a large-scale empirical study. It reveals that many benchmarks designed to eliminate text bias have inadvertently introduced image bias and proposes a multi-dimensional characterization framework for multi-modal datasets.
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Multi-modal learning"
  - "benchmark evaluation"
  - "modality dependence"
  - "VQA"
  - "MLLM"
date: 2026-05-08
content_hash: cce54bd4e0065e8b
---

# Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional

**Conference**: ICLR 2026  
**arXiv**: [2509.23499](https://arxiv.org/abs/2509.23499)  
**Code**: [GitHub](https://github.com/divyam3897/multimodal-spectrum)  
**Area**: Signal Communication  
**Keywords**: Multi-modal learning, benchmark evaluation, modality dependence, VQA, MLLM

## TL;DR

This work quantifies intra-modality and inter-modality dependencies across 23 VQA benchmarks through a large-scale empirical study. It reveals that many benchmarks designed to eliminate text bias have inadvertently introduced image bias and proposes a multi-dimensional characterization framework for multi-modal datasets.

## Background & Motivation

1. **Background**: Multi-modal Large Language Models (MLLMs) are developing rapidly, accompanied by the emergence of over 200 evaluation benchmarks. However, there is a lack of systematic understanding regarding what these benchmarks actually measure.

2. **Limitations of Prior Work**: Benchmark selection lacks a scientific basis—for instance, Gemini 1.5 and 2.5 were evaluated using different sets of datasets without explanation for the change. It is difficult to determine whether model performance improvements represent true progress in multi-modal capabilities or the exploitation of single-modality shortcuts.

3. **Key Challenge**: Benchmark development has fallen into a "cat-and-mouse" cycle—new datasets are designed to eliminate specific single-modality biases, only to be found later to introduce new ones (e.g., VQA → VQAv2 → VQA-CP → MMMU → MMMU-Pro).

4. **Goal**: To conduct a systematic modality dependence analysis of existing multi-modal benchmarks and provide a quantitative characterization framework.

5. **Key Insight**: By employing modality shuffling, the inter-modality dependencies can be disrupted while maintaining the marginal distributions of single modalities, allowing for the measurement of performance changes under four distinct input conditions.

6. **Core Idea**: Multi-modal datasets are inherently multi-dimensional, where the strength of intra-modality dependencies (solvable via a single modality) and inter-modality dependencies (requiring joint reasoning) varies significantly both within and across benchmarks.

## Method

### Overall Architecture

The core of the diagnosis involves splitting a multi-modal sample into image and text input streams and feeding them into the same model under four "input configurations" to observe accuracy degradation. The configurations are: normal pairing $\mathcal{M}(f_\theta(\mathbf{x_1}, \mathbf{x_2}), \mathbf{y})$, image only (text replaced with the text of a random sample from the dataset), text only (image replaced with a random sample's image), and a fully random baseline where both streams are shuffled. A smaller drop in accuracy after a specific stream is shuffled indicates that the model is more capable of bypassing the other modality and relying on that stream alone to answer. The independent contributions of each modality and their interaction contributions are quantified accordingly.

### Key Designs

**1. Using modality shuffling instead of zeroing to sever inter-modality dependencies**

To isolate the contribution of a specific modality, the most direct approach is to remove it—using a blank image or an empty string. However, this pushes the input out-of-distribution (OOD), as the model may exhibit unpredictable behavior when faced with "empty inputs," contaminating the measurement results. This work adopts and adapts the shuffling approach from Perceptual Score (Gat et al. 2021): the target modality is replaced with the corresponding modality from another sample in the same dataset. The image remains a real image and the question remains a valid question, but the alignment with the label $\mathbf{y}$ is broken. In this way, the marginal distribution of each modality remains unchanged, and only the cross-modal correlation is destroyed. The resulting performance gap then cleanly corresponds to "inter-modality dependence," whereas zeroing or perturbation would introduce noise from "input anomalies."

**2. Descending to sub-category granularity to prevent aggregated scores from masking local biases**

Calculating an average drop across the entire dataset can easily create the illusion that a dataset is "balanced," as subgroups with strong single-modality dependencies may be diluted by subgroups with weak dependencies. This work slices datasets into subsets based on attributes such as question type or object category, running the four-configuration diagnosis independently on each subset. Results indicate that datasets which globally appear to require dual-modal interaction may rely almost entirely on a single modality in specific sub-categories—for example, higher-grade questions in ScienceQA actually depend strongly on text priors. This structural bias is only visible when examined at the subgroup level.

**3. Marginalizing across model scales and architectures to confirm dependencies originate from the data itself**

Observing the performance gap of a single model makes it difficult to distinguish whether the dataset inherently possesses shortcuts or if the model itself is biased toward one modality. This work treats modality dependence as a joint function of both data and model. To capture the inherent characteristics of the data, the model dimension is marginalized: a majority-vote ensemble of Cambrian-1 at three scales (8B/13B/34B) is used, supplemented by cross-verification across different architectures such as LLaVA-Next and the Qwen series. When bias patterns remain consistent across these diverse models, it can be confidently attributed as a property of the benchmark rather than a quirk of a specific model.

### Mechanism

This is a diagnostic study that does not train any models; all conclusions are derived from evaluating existing models, with Multiple Choice VQA (MCVQA) accuracy used as the uniform metric. The four configurations yield four accuracy scores: the fully random configuration serves as the baseline floor. The improvement of the normal pairing relative to the baseline measures the overall solvability of the dataset, while the improvements of the image-only and text-only configurations relative to the floor quantify the strength of intra-image and intra-text dependencies, respectively. Only the remaining improvement, where the two single-modality configurations show little gain over the baseline while the normal pairing is significantly higher, is attributed to true inter-modality dependence requiring joint reasoning.

## Key Experimental Results

### Main Results

Modality dependence classification of 23 benchmarks:

| Dependency Type | Representative Datasets | Characteristics |
|---------|-----------|------|
| Inter-modality only | MME, POPE, COCO, V*Bench | Extremely rare, only 4/23 datasets |
| With Intra-text dependency | GQA(+26%), ScienceQA(+17.5%), MMMU(+11.35%), AI2D(+34.94%) | Can significantly exceed random baseline using text only |
| With Intra-image dependency | MMBench(+41%), SEED, TextVQA, MMMU-Pro, MMVP | Eliminating text bias inadvertently introduced image bias |

### Ablation Study

| Configuration | Key Findings | Description |
|------|---------|------|
| Increasing Model Scale (8B→34B) | Single-modality bias increases rather than decreases | Larger models on MMMU showed increased image and text dependence |
| Different Model Types | Bias patterns are consistent across models | Cambrian, LLaVA-Next, and Qwen models exhibit similar behavior |
| Sub-category Analysis | Aggregated metrics mask subgroup biases | Higher-grade questions in ScienceQA rely almost entirely on text |

### Key Findings

- **Only 4 out of 23 benchmarks** demonstrate pure inter-modality dependence, far fewer than expected.
- New benchmarks designed to eliminate text bias (e.g., MMBench, SEED) have introduced image bias instead—replacing one single-modality shortcut with another.
- Increasing model scale does not alleviate single-modality bias and may even exacerbate it.
- Sub-category analysis shows that even globally balanced datasets still exhibit strong biases in specific subsets.

## Highlights & Insights

- **Reveals the fundamental problem of multi-modal evaluation**: Relying on a single aggregated score to evaluate models is insufficient; single-modality baseline performance must be reported simultaneously.
- **Provides a tool to judge if model progress is genuine**: Performance gains may simply reflect models becoming more adept at exploiting single-modality dependencies.
- **Practical guide for designing new benchmarks**: The core objective should be requiring both modalities to answer, rather than merely eliminating dependence on one specific modality.
- **Profound insight into the "cat-and-mouse game"**: Only through systematic quantification of modality dependence can this cycle be broken.

## Limitations & Future Work

- The analysis is limited to the Multiple Choice VQA format and does not cover open-ended generation tasks.
- The modality shuffling method may have limitations when the options themselves contain modality-specific information.
- There is a need to extend the evaluation to the model's ability to actively abstain from answering.
- Future work should focus on benchmark designs for open-ended answer generation and evaluation.

## Related Work & Insights

- Perceptual Score (Gat et al., 2021) provided the foundational methodology, which this work extends to a large-scale analysis of 23 benchmarks.
- Similar to the insight in NAS that "search space design is more important than the search algorithm"—benchmark design is more fundamental than model improvement.
- Insight: When evaluating multi-modal models, modality-specific baselines should be reported to establish a community norm.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic analysis reveals important overlooked issues.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 benchmarks, multiple model scales and types, and sub-category analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic and excellent visualization.
- Value: ⭐⭐⭐⭐ Provides important methodological guidance for the multi-modal evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Patch-as-Decodable-Token: Towards Unified Multi-Modal Vision Tasks in MLLMs](patch-as-decodable-token_towards_unified_multi-modal_vision_tasks_in_mllms.md)
- [\[ICLR 2026\] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [\[AAAI 2026\] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](../../AAAI2026/multimodal_vlm/imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [\[CVPR 2026\] Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning](../../CVPR2026/multimodal_vlm/hierarchical_attacks_for_multi-modal_multi-agent_reasoning.md)
- [\[ICLR 2026\] Calibrated Information Bottleneck for Trusted Multi-modal Clustering](calibrated_information_bottleneck_for_trusted_multi-modal_clustering.md)

</div>

<!-- RELATED:END -->
