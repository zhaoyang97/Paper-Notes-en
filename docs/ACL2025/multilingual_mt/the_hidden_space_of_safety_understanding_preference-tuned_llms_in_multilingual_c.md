---
title: >-
  [Paper Note] The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts
description: >-
  [ACL2025][Multilingual & Machine Translation][multilingual alignment] This paper systematically analyzes the impact of preference tuning (such as RLHF/DPO) on the internal representation space of LLMs in multilingual scenarios. It finds that while the alignment mechanism effectively separates the latent space representations of harmful and harmless content in English, this effect is significantly degraded in non-English languages such as Hindi, Chinese, and German…
tags:
  - "ACL2025"
  - "Multilingual & Machine Translation"
  - "multilingual alignment"
  - "LLM safety"
  - "preference tuning"
  - "representation analysis"
  - "cross-lingual"
date: 2026-05-08
content_hash: 187a7ecebe782f76
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts

**Conference**: ACL2025  
**arXiv**: [2504.02708](https://arxiv.org/abs/2504.02708)  
**Code**: TBD  
**Area**: Multilingual Translation  
**Keywords**: multilingual alignment, LLM safety, preference tuning, representation analysis, cross-lingual

## TL;DR

This paper systematically analyzes the impact of preference tuning (such as RLHF/DPO) on the internal representation space of LLMs in multilingual scenarios. It finds that while the alignment mechanism effectively separates the latent space representations of harmful and harmless content in English, this effect is significantly degraded in non-English languages such as Hindi, Chinese, and German, revealing a severe monolingual bias in current alignment methods.

## Background & Motivation

1. **Alignment tuning is core to LLM safety**: Preference optimization methods like RLHF and DPO have become standard post-training stages to ensure LLMs are safe, reliable, and aligned with human values.
2. **Preference data is dominated by English**: The vast majority of current alignment datasets (e.g., HH-RLHF, Anthropic data) are in English, with an extreme scarcity of preference data for low-resource languages.
3. **Inconsistent multilingual safety performance**: Although LLMs can correctly refuse harmful requests in English (e.g., "how to build a bomb"), the model might directly output detailed steps when the same request is asked in Hindi.
4. **Unclear latent space mechanisms**: How alignment reshapes the internal representation space of the model and whether safety constraints generalize cross-lingually lack systematic quantitative research.
5. **Jailbreak attacks are more successful in multilingual contexts**: Prior research has shown that non-English languages can bypass safety filters, fundamentally reflecting insufficient multilingual alignment.
6. **Impact on fairness and global deployment**: Global deployment faces ethical barriers when linguistic differences in alignment quality mean non-English users face a higher risk of exposure to harmful content.

## Method

### Core Analysis Framework

This paper proposes a multilingual alignment evaluation method based on latent space analysis. The core idea is: **alignment should separate harmful and harmless content in the representation space of the model, and the degree of separation reflects the alignment strength**.

### Data Preparation

- **Balanced Toxicity Dataset**: 5,000 samples per language (2,500 toxic + 2,500 non-toxic), covering 9 languages, focusing on English (en), Hindi (hi), Chinese (zh), and German (de).
- **Multilingual Parallel Text Detoxification Dataset**: Parallel sentence pairs containing toxic and detoxified versions, identical in semantics but differing in toxic expressions, used for more rigorous controlled experiments.

### Representation Extraction and Visualization

- Run forward passes using the reference model $\pi_{ref}$ (after SFT, before alignment) and the aligned model $\pi_\theta$ (after alignment).
- Extract the final-layer embedding representations, **probing Only during the input processing stage** to avoid contamination from memorization and bias in the decoding stage.
- Use PCA to reduce dimensionality to 2D for visualization and observe the segregation of harmful/harmless clusters.

### Metrics for Quantitative Separation

1. **Between-class variance ratio**: Measures the variance ratio between the two classes (harmful and harmless). A larger value indicates better separation.
2. **Bhattacharyya Distance**: Measures the degree of overlap between two distributions on a logarithmic scale (ranging from $10^{-3}$ to $10^{+1}$). A larger value indicates more distinct separation.
3. **Silhouette Score**: Evaluates clustering quality by measuring the tightness and separation of samples belonging to correct clusters.

### Evaluated Models

Covering 7 open-source models across 4 model families: Llama-2 (7B), Llama-3.1 (8B), Llama-Guard-3 (8B), Qwen-2.5 (7B), Gemma-2 (9B), Gemma-3 (12B), Phi-4 (14B). Each model is evaluated by comparing its reference version and aligned version.

## Key Experimental Results

### Table 1: Llama-2 Between-class Variance Improvement Before/After Alignment

| Language | Before Alignment Between-class Var | After Alignment Between-class Var | Improvement Size |
|:---|:---:|:---:|:---:|
| English | 0.81% | 61.20% | **+60.39%** |
| Hindi | - | - | +19.98% |
| Chinese | - | - | +10.09% |
| German | - | - | +26.85% |

- The improvement of between-class variance for English is 6 times that of Chinese and 3 times that of Hindi.
- The PCA explained variance ratio is 49.61%, and the first 10 principal components are used for more comprehensive measurement.

### Table 2: Cross-Model and Cross-Lingual Comparison of Bhattacharyya Distance (Log Scale)

| Model | en (Δ Direction) | hi (Δ Direction) | zh (Δ Direction) | de (Δ Direction) |
|:---|:---:|:---:|:---:|:---:|
| Llama-2 | ↑ Significantly increased | ↑/↓ Unstable | ↑ Slightly increased | ↑ Moderately increased |
| Llama-3.1 | ↑ Significantly increased | ↓ Reversed degradation | ↑ Slightly increased | ↑ Moderately increased |
| Gemma-2 | ↑ Significantly increased | ↑ Moderately increased | ↑ Moderately increased | ↑ Moderately increased |
| Qwen-2.5 | ↑ Significantly increased | ↑ Moderately increased | ↑ Moderately increased | ↑ Moderately increased |

- English shows a consistent enhancement of separation across all models.
- Hindi shows a "reverse effect" on some models: the aligned model has poorer cluster separation than the unaligned model.
- Silhouette Score shows a consistent pattern: the clustering quality improvement for English is much higher than for other languages.

### Parallel Detoxification Experiment (Harder Setup)

- English: Even when the harmful and harmless sentences differ by only 1-2 words, the model still maintains meaningful representation separation after alignment.
- Hindi: The model fails to capture a clear distribution shift in the low-dimensional space both before and after alignment.

## Highlights & Insights

- **Unique analytical perspective**: Instead of evaluating safety through generated content, it directly probes changes in the representation distribution of the latent space, providing a new dimension for understanding alignment mechanisms.
- **Complete quantitative metric system**: Three complementary metrics—PCA visualization, Bhattacharyya Distance, and Silhouette Score—are combined to validate the conclusions.
- **Revealing critical safety hazards**: The "reverse effect" in Hindi (where separation decreases after alignment) provides a representation-level explanation for multilingual jailbreak attacks.
- **Comprehensive model coverage**: Spanning 4 model families and 7 models, the conclusions have good generalizability.
- **Rigorous experimental design**: Using parallel detoxification corpora for controlled experiments tests the sensitivity of alignment given highly similar semantics.

## Limitations & Future Work

- **Focus only on safety**: Alignment methods also affect multiple capabilities such as reasoning, instruction following, and planning, while this paper only analyzes the toxicity/safety dimension.
- **Limited language coverage**: Only 3 non-English languages are analyzed, all of which are medium-resource languages, without involving actual low-resource languages (e.g., Swahili, Thai).
- **Diagnostic rather than solution-oriented**: This paper primarily diagnoses problems and does not propose specific improved alignment methods or multilingual fine-tuning strategies.
- **Limited dataset scale**: Although 5,000 samples per language exceeds previous work, it may not be sufficient to represent the full complexity of multilingual scenarios.
- **Lack of reference model for Phi-4**: Phi-4 has no publicly available reference (unaligned) model, making before-and-after comparison impossible.

## Related Work & Insights

### vs Dang et al. (2024) "RLHF Can Speak Many Languages"

Dang et al. study how to adapt RLHF to multilingual settings from a training perspective, proposing cross-lingual preference transfer methods. This paper is complementary: instead of focusing on how to improve training, it diagnoses where and to what extent existing aligned models fail in multilingual scenarios from the perspective of the representation space. Combining the two can form a complete "diagnosis -> treatment" pipeline.

### vs Lin et al. (2024) Multilingual Alignment Analysis

Lin et al. also analyze the effect of alignment in multilingual scenarios, but only with a sample size of around 200. This paper uses 5,000 samples and introduces more rigorous quantitative metrics such as Bhattacharyya Distance and Silhouette Score, achieving a significant improvement in methodology and statistical reliability.

### vs Son et al. (2024) Multilingual Jailbreak Research

Son et al. discover from an adversarial attack perspective that non-English languages are more prone to jailbreaking LLMs. This paper provides a mechanistic explanation for this phenomenon from the representation space perspective: the separation of safety constraints is insufficient in the latent space of non-English languages, making the model unable to effectively distinguish harmful and harmless inputs internally.

## Rating

- Novelty: ⭐⭐⭐⭐ — Probing multilingual alignment from the perspective of latent space representation is both novel and insightful.
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive model coverage but limited language coverage, lacking low-resource language validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich visualizations, and thorough ethical discussion.
- Value: ⭐⭐⭐⭐ — Provides a quantitative diagnostic tool and important empirical findings for multilingual LLM safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](implicit_cross-lingual_rewarding_for_efficient_multilingual_preference_alignment.md)

</div>

<!-- RELATED:END -->
