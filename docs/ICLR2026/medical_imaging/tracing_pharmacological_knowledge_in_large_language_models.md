---
title: >-
  [Paper Note] Tracing Pharmacological Knowledge in Large Language Models
description: >-
  [Medical Imaging] The first systematic causal analysis of the encoding mechanisms for drug-group semantics in biomedical LLMs, revealing that drug-group knowledge is stored in early layers, distributed across multiple tokens (not the last token alone), and that linearly separable semantic information is already present at the embedding layer.
tags:
  - Medical Imaging
date: 2026-05-08
content_hash: f6afa10950c437a0
---

# Tracing Pharmacological Knowledge in Large Language Models

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2603.03407](https://arxiv.org/abs/2603.03407)
- **Code**: Not released
- **Area**: Medical Imaging
- **Keywords**: Mechanistic interpretability, pharmacological knowledge, activation patching, linear probing, LLM internal representations

## TL;DR

The first systematic causal analysis of the encoding mechanisms for drug-group semantics in biomedical LLMs, revealing that drug-group knowledge is stored in early layers, distributed across multiple tokens (not the last token alone), and that linearly separable semantic information is already present at the embedding layer.

## Background & Motivation

Large language models have demonstrated strong capabilities in pharmacology and drug discovery tasks, including target identification, drug interaction prediction, and automated hypothesis generation. However, **the mechanisms by which LLMs internally encode pharmacological concepts—such as drug classes, functional groups, and therapeutic effects—remain largely unexplored**.

Key questions:
1. In **which layers** is drug-group information stored?
2. Is drug-group semantics **concentrated in a single token or distributed across multiple tokens**?
3. **How do biomedical fine-tuned models differ from general-purpose models** in pharmacological knowledge representations?

Understanding these mechanisms is critical for improving model reliability and generalizability, particularly in high-stakes biomedical settings.

## Method

### Overall Architecture

The study adopts complementary perspectives via a **causal approach (activation patching)** and a **correlational approach (linear probing)**:
- Activation patching tests necessity and sufficiency.
- Linear probing assesses linear separability.

### 1. Dataset Construction

Drug–drug-group relationships are parsed from the U.S. National Library of Medicine to construct a **binary-choice QA dataset**:

```
Question: Which compound is categorized as vasoconstrictor agents?
A) ergotamine      B) araldite
Answer:
```

The binary-choice format is adopted for the following reasons:
- Drug name tokenization varies considerably, making single-token evaluation unreliable.
- Correct answers are not unique (multiple drugs may belong to the same pharmacological class).
- Predictions are determined by selecting the option with the highest logit value.

### 2. Activation Patching

Objective: **Causally localize** which model components (layer × token position) are responsible for storing drug-group knowledge.

Three forward passes are performed:
1. **Clean pass**: correct prompt, with activations cached.
2. **Counterfactual pass**: the drug group is altered so that the correct answer flips.
3. **Patched pass**: selected components in the counterfactual run are replaced with activations from the clean run.

**Symmetric token substitution** is used (to avoid out-of-distribution inputs introduced by Gaussian noise), with the following normalized logit-difference metric:

$$\text{metric}(r, r') = \frac{\text{LD}_{\text{pt}}(r, r') - \text{LD}_*(r, r')}{\text{LD}_{\text{cl}}(r, r') - \text{LD}_*(r, r')}$$

Patching is applied separately to the **residual stream** and **MLP outputs**:
- Residual stream: per-layer, per-token position.
- MLP outputs: using a 10-layer window (since single-layer interventions produce negligible effects).

### 3. Linear Probing

Semantically opposing drug-group pairs are constructed (e.g., α-agonists vs. antagonists, CNS stimulants vs. depressants), with 300 prompts generated per pair.

Two probe types are employed:
- **Single-token probe**: trained on activations at each token position within the drug-group span.
- **Sum-pooled probe**: trained on the sum-pooled activations across all tokens in the span.

L2-regularized logistic regression ($C = 10^{-3}$) is used, with stratified cross-validation to prevent data leakage.

## Experiments

### Baseline Performance

| Model | Accuracy |
|-------|----------|
| BioGPT | 0.600 |
| OpenBioLLM-8B | 0.920 |
| BioMistral-7B | 0.860 |
| Llama-3.1-8B-Instruct | 0.900 |
| Gemma3-4B | 0.860 |

With the exception of BioGPT, all models demonstrate strong drug-class knowledge. The Llama series achieves the best overall performance.

### Activation Patching Results

#### Residual Stream Patching

- **Causal effects concentrate in early layers** (layers 0–10) at drug-group token positions.
- The **strongest causal effects** arise at **middle tokens** within the drug-group span, not the last token.
- This **contrasts with** the findings of Meng et al. (2023) on factual knowledge, where the final subject token was identified as most critical.

#### MLP Patching

- Early MLP layers (0–10) consistently produce positive causal effects at drug-group span positions.
- Causal influence at middle tokens similarly exceeds that at the last token.
- The maximum mean effect at drug-group tokens is 0.76, approaching 0.80 observed at the final prompt token.

### Linear Probing Results

| Probe Type | Accuracy |
|-----------|----------|
| Single-token activations | ~0.52–0.63 (near chance) |
| Sum-pooled activations | **1.000** (perfect separation) |

Key findings:
- **Semantic information is not aligned to a single token** but is distributed across the entire span.
- **Linear separability after aggregation**: sum-pooled probes achieve perfect accuracy across all layers.
- **Encoding at the embedding layer**: sum-pooled probes reach 1.0 accuracy before the first Transformer block (layer 0).
- **Cross-model consistency**: instruction-tuned (Llama-Instruct) and biomedical fine-tuned (OpenBioLLM) models exhibit consistent behavior.

## Highlights & Insights

- The first systematic study of pharmacological knowledge mechanisms in LLMs.
- Reveals that drug-group semantics are **distributed representations**, challenging the prevailing assumption that the last token is most critical.
- Complementary causal intervention (patching) and correlational analysis (probing) provide a complete chain of evidence.
- Semantic information is present at the embedding layer, suggesting that token embeddings themselves carry rich pharmacological semantics.
- The experimental design is elegant: semantically opposing drug-group pairs (agonists vs. antagonists) ensure probe validity.

## Limitations & Future Work

- Only drug-group-level knowledge is studied; individual drugs and other biomedical concepts are not examined.
- Specific attention heads or circuits are not analyzed; the study remains at the layer and token level.
- The model scope is limited, focusing primarily on Llama-3.1-8B and OpenBioLLM-8B.
- Dataset construction relies on the NLM database, and drug-group classifications may not provide complete coverage.
- The effect of interventions on downstream model behavior (e.g., editing pharmacological knowledge) is not explored.
- The perfect accuracy of sum-pooled probes may be partly attributable to information leakage arising from differences in token counts.

## Related Work & Insights

### Mechanistic Interpretability
- **Activation patching**: Meng et al. (ROME) — localizing factual knowledge; Conmy et al. — induction circuits.
- **Linear probing**: Hewitt & Manning (syntactic structure); Tenney et al. (layer-wise information in BERT).
- **Lexical semantic probing**: Vulic et al.; Geva et al. (FFN layers constructing predictions).

### Biomedical LLM Interpretability
- Ahsan et al.: representations of gender and racial attributes in clinical LLMs — gender is concentrated in middle MLP layers, while race is more diffuse.
- The present work extends this line of inquiry to pharmacological concepts, uncovering the importance of early layers and distributed token representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of mechanistic interpretability to pharmacological knowledge.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous dual methodology combining causal and correlational approaches.
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited in the number of models and drug groups examined.
- **Value**: ⭐⭐⭐ — Informative for understanding biomedical LLMs, though direct practical applications remain limited.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](../../AAAI2026/medical_imaging/small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)

<!-- RELATED:END -->
