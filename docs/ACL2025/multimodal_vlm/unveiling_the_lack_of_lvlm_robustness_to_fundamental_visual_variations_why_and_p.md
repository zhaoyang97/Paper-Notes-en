---
title: >-
  [Paper Note] Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward
description: >-
  [ACL 2025][Multimodal VLM][LVLM robustness] This work proposes V2R-Bench, a benchmark framework to systematically evaluate the robustness of 21 LVLMs against four fundamental visual variations (position, scale, orientation, and context). It reveals significant vulnerabilities in even advanced models on simple visual tasks and demonstrates through component-level analysis that these loopholes stem from insufficient multimodal alignment and error accumulation in pipelined archi…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "LVLM robustness"
  - "visual variations"
  - "V2R-Bench"
  - "multimodal alignment"
  - "positional bias"
  - "visual acuity threshold"
date: 2026-05-08
content_hash: d510d20a6638afff
---

# Unveiling the Lack of LVLM Robustness to Fundamental Visual Variations: Why and Path Forward

**Conference**: ACL 2025  
**arXiv**: [2504.16727](https://arxiv.org/abs/2504.16727)  
**Code**: [GitHub](https://github.com/toward-agi/Visual-Variations-Robustness)  
**Area**: Multimodal VLM  
**Keywords**: LVLM robustness, visual variations, V2R-Bench, multimodal alignment, positional bias, visual acuity threshold  
**Authors**: Zhiyuan Fan, Yumeng Wang, Sandeep Polisetty, Yi R. (May) Fung (Hong Kong University of Science and Technology)  

## TL;DR

This work proposes V2R-Bench, a benchmark framework to systematically evaluate the robustness of 21 LVLMs against four fundamental visual variations (position, scale, orientation, and context). It reveals significant vulnerabilities in even advanced models on simple visual tasks and demonstrates through component-level analysis that these loopholes stem from insufficient multimodal alignment and error accumulation in pipelined architectures rather than data scarcity.

## Background & Motivation

Existing LVLMs (such as GPT-4o, LLaVA, etc.) perform exceptionally well on complex vision-language tasks. However, a fundamental question remains neglected: **Are models robust to inevitable visual variations in natural scenes?** In the real world, objects naturally exhibit position shifts, scale variations, orientation rotations, and contextual changes due to camera angles, distances, and environmental transitions.

Prior robustness studies primarily focus on socioeconomic/cultural biases, adversarial attacks, or image corruptions, while the impact of **natural visual variations** remains severely under-studied. This gap implies that the reliability of LVLMs in real-world deployment lacks guarantees.

## Method

### 1. V2R-Bench Evaluation Framework

The framework incorporates an automated data generation pipeline and uniquely designed evaluation metrics:

**Four Dimensions of Visual Variations**:
- **Position Variation (Position)**: Spatial position shifts of objects within the image, detecting whether models have visual blind spots.
- **Scale Variation (Scale)**: Size of objects transitioning from large to small, testing the perception boundary of models.
- **Orientation Variation (Orientation)**: Changes in object rotation angles, evaluating the capability to handle non-standard poses.
- **Context Variation (Context)**: Changing the surrounding environment of objects, verifying whether models perform genuine visual perception instead of contextual inference.

Given an image $I$, generate the set of transformed images: $\mathcal{D} = \{T(I,v) | v \in P \times S \times R \times C\}$

**Evaluation Metrics**:
- **Performance Consistency** $C_m$: Measures whether the model maintains stable task performance under visual variations.
- **Semantic Stability** $S_s$: Output embedding consistency based on cosine similarity.
- **Token Stability** $S_t$: Output token consistency based on Jaccard similarity.
- **LLM-as-Judge**: LLM acts as a judge for qualitative evaluation.

**Data Scale**: A total of 428K images, covering two categories of tasks: basic visual tasks (90 object categories for recognition + 8 orientation detections) and extended multimodal benchmark tasks.

### 2. Cross-Modal Diagnosis Framework

Systematic analysis is conducted on various components of LVLMs:

- **Visual Encoder**: Evaluates feature quality, using text encoders for zero-shot analysis on contrastive learning encoders, and linear probing on self-supervised encoders.
- **Multimodal Projector**: Compares the visual task performance of features before and after projection to quantify information loss; measures the alignment quality between projected features and language embeddings.
- **Language Model**: Bypasses upstream modules and directly provides textualized visual information to test the robustness of the LLM itself.

### 3. Vision-Language Feature Visualization

Decodes aligned visual features into language tokens: $t = \text{topk}(\text{softmax}(hE^\top))$, providing an intuitive understanding of how models handle cross-modal information propagation.

## Key Experimental Results

### Table 1: Accuracy on Basic Visual Tasks (A Selection of Representative Models)

| Model | Position-Object | Position-Orientation | Scale-Object | Scale-Orientation | Context-Object | Context-Orientation |
|------|----------|----------|----------|----------|-----------|-----------|
| GPT-4o | 31.5 | 78.2 | 31.9 | 69.5 | 26.7 | 79.9 |
| Qwen2-VL-7B | 5.9 | 40.2 | 17.5 | 50.9 | 6.7 | 42.7 |
| LLaVA-Onevision | 16.0 | 44.6 | 24.5 | 47.4 | 17.3 | 45.9 |
| Molmo-7B-D | 26.0 | 62.6 | 27.8 | 63.2 | 26.2 | 64.6 |
| LLaMA3.2-90B-V | 6.3 | 41.2 | 18.2 | 51.7 | 7.1 | 43.7 |
| Phi3.5-Vision | 6.2 | 40.8 | 7.9 | 43.0 | 6.7 | 41.7 |

**Key Findings**: Even GPT-4o achieves only ~31% accuracy in object recognition, falling far below its performance on complex VQA benchmarks.

### Table 2: Information Loss Caused by Multimodal Projectors

| Model | ViT Object Recognition | Post-Projection Object Recognition | Drop | ViT Orientation Detection | Post-Projection Orientation Detection | Drop |
|------|------------|-------------|---------|------------|-------------|---------|
| LLaVA1.5 | 44.2% | 3.2% | -41.0 | 82.6% | 11.7% | -70.9 |
| LLaVA1.6 | 40.6% | 14.1% | -26.5 | 91.2% | 20.7% | -70.5 |
| Qwen2-VL | 44.6% | 65.9% | +21.3 | 100% | 100% | 0 |
| InternVL | 22.1% | 13.2% | -8.9 | 96.4% | 33.9% | -62.5 |

**Key Findings**: Qwen2-VL, which adopts a native visual encoder (directly co-trained with the language model), shows improved performance after projection, demonstrating the potential of unified architectures.

### Table 3: Quantitative Analysis of Positional Bias

| Model | Center-Object | Peripheral-Object | Difference | Center-Orientation | Peripheral-Orientation | Difference |
|------|----------|----------|------|----------|----------|------|
| LLaVA1.6 | 1.94 | 1.95 | +0.01 | 17.94 | 19.99 | +2.05 |
| InternVL | 2.01 | 2.79 | +0.78 | 21.58 | 30.26 | +8.68 |
| Qwen2-VL | 5.78 | 7.02 | +1.24 | 39.36 | 44.08 | +4.72 |
| LLaMA3.2-11B-V | 3.72 | 6.58 | +2.86 | 27.19 | 35.88 | +8.69 |

**Key Findings**: All models consistently perform better in peripheral regions of the image than in central regions, contradicting the theory of effective receptive fields.

## Key Findings

1. **Counter-Intuitive Positional Bias**: Models achieve higher accuracy at the image margins than in the center, which contradicts the effective receptive field theory (wherein central areas are expected to be superior).
2. **Human-like Visual Acuity Threshold**: When object scale shrinks to $1/100$ of the image area (i.e., $1/10$ of width and height), performance plateaus at its minimum, identifying a clear perceptual lower bound.
3. **Orientation-Selective Vulnerability**: Models are robust to certain orientations but fail significantly on others, demonstrating orientation bias.
4. **Context-Dependent Inference**: Models lean towards contextual deduction rather than direct visual perception, a phenomenon further substantiated by OCR experiments.
5. **Scaling Laws Remain Valid for Robustness**: Under the same architecture, larger models demonstrate superior robustness.
6. **Vulnerability Roots in Architecture Rather Than Data**: Synthetic data augmentation has limited efficacy in improving robustness. Directly training on spatial tasks boosts accuracy but fails to enhance robustness.

## Highlights & Insights

- **First Systematic Study** of LVLM robustness regarding fundamental visual variations, filling a critical research gap.
- **Large-Scale Evaluation**: A comprehensive assessment involving 428K images across 21 models.
- **Innovative Visualization Method**: Decodes aligned visual features into language tokens, visually demonstrating the cross-modal alignment process for the first time.
- **Component-Level Attribution**: Pinpoints vulnerabilities as stemming from information loss in multimodal projectors and error accumulation within pipelined architectures.
- **Definitive Conclusion on Architecture vs. Data**: Confirms via synthetic data experiments that the issue is an architectural defect rather than a data-level deficiency.
- **Empirical Support for Unified Architectures**: Highlights the prominent superiority of Qwen2-VL's unified architectural design, offering empirical backing for future design pathways.

## Limitations & Future Work

- Only explored data augmentation as a mitigation strategy, leaving architectural modifications, pre-training strategy improvements, or test-time self-correction methods unattempted.
- The four visual variations do not exhaust all possible visual transformations, leaving long-tail scenarios uncovered.
- The scale of mitigation experiments is limited; the constrained effectiveness of data augmentation could be tied to insufficient data volume.
- Security Risks: Visual variations can serve as a novel attack vector, enabling natural visual adversarial generation without the need for model training.
- Fails to deeply analyze the robustness performance of newer, unified multimodal architectures (such as Janus-Pro).

## Related Work & Insights

- **LVLM Evaluation Benchmarks**: Benchmarks like MMBench, SEED-Bench, and MME prioritize capability evaluation, whereas this work centers on robustness assessment.
- **Robustness Research**: Existing literature emphasizes socioeconomic/cultural biases (Ananthram et al., 2024) and adversarial attacks (Liu et al., 2024), while this study focuses on natural visual variations for the first time.
- **Multimodal Alignment**: Alignment module designs in LLaVA, Flamingo, Qwen-VL, etc. are scrutinized, with this work revealing their core bottlenecks.
- **Unified Architectures**: Systems like Janus-Pro seek to unify vision and language modalities; this study provides empirical evidence corroborating the validity of this research direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study on LVLM robustness under basic visual variations; the problem definition is highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Exceptional completeness with 21 models, 428K images, component-level analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, deeply analyzed, and intuitively visualized.
- **Value**: ⭐⭐⭐⭐ — Uncovers foundational limitations of LVLMs, offering crucial guidance for future architectural refinements in the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VLSBench: Unveiling Visual Leakage in Multimodal Safety](vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] VLMInferSlow: Evaluating the Efficiency Robustness of Large Vision-Language Models as a Service](vlminferslow_evaluating_the_efficiency_robustness_of.md)
- [\[ICML 2026\] Unveiling Visual Counting Bottlenecks in Vision-Language Models](../../ICML2026/multimodal_vlm/unveiling_the_visual_counting_bottleneck_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
