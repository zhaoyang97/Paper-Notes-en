---
title: >-
  [Paper Note] Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search
description: >-
  [ICLR 2026][Image Generation][text-to-image bias] This paper proposes Bias-Guided Prompt Search (BGPS), which combines LLM decoding guidance with a diffusion model intermediate-layer attribute classifier to automatically…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "text-to-image bias"
  - "automated prompt search"
  - "fairness"
  - "bias auditing"
  - "diffusion models"
date: 2026-05-08
content_hash: ee64f2781b24e284
---

# Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search

**Conference**: ICLR 2026
**arXiv**: [2512.08724](https://arxiv.org/abs/2512.08724)
**Code**: None
**Area**: Image Generation
**Keywords**: text-to-image bias, automated prompt search, fairness, bias auditing, diffusion models

## TL;DR

This paper proposes Bias-Guided Prompt Search (BGPS), which combines LLM decoding guidance with a diffusion model intermediate-layer attribute classifier to automatically discover interpretable text prompts that maximally expose hidden social biases in T2I models—revealing residual biases even in debiased models.

## Background & Motivation

Text-to-image (T2I) diffusion models have been repeatedly shown to exhibit social biases along sensitive attributes such as gender, race, and age. For example, Stable Diffusion generates 100% male faces for the prompt "engineer." Existing bias evaluation and mitigation methods face a **fundamental trade-off between coverage and interpretability**:

**Manual or LLM-assisted curation**: produces interpretable prompts but explores only a limited portion of the prompt space.

**Gradient-based optimization methods** (e.g., PEZ): can identify high-bias regions but produce unreadable text (e.g., "nurse keras matplotlib tbody"), making them unsuitable for practical auditing.

A more critical issue is that **debiased models** appear balanced on standard benchmarks (49% male), yet generate 79% male images when evaluated on BGPS-discovered prompts—indicating that existing debiasing methods address only surface-level symptoms.

## Method

### Overall Architecture

BGPS maximizes the joint probability $\mathbb{P}(A=a, \boldsymbol{s})$, where $A$ denotes a sensitive attribute (e.g., gender) and $\boldsymbol{s}$ denotes the prompt text:

$$\max_{\boldsymbol{s}} J(a, \boldsymbol{s}) = \max_{\boldsymbol{s}} \log \mathbb{P}(\boldsymbol{s}) + \lambda \log\left(\frac{1}{K}\sum_{i=1}^K \mathbb{P}(A=a \mid \boldsymbol{x_0}^i, \boldsymbol{\epsilon}_1^i, \ldots, \boldsymbol{\epsilon}_T^i, \boldsymbol{s})\right)$$

Two core components:
1. **LLM language prior** $\mathbb{P}(\boldsymbol{s})$: ensures generated prompts are natural, interpretable, and attribute-neutral.
2. **Attribute classifier** $\mathbb{P}(A=a|\cdot)$: a lightweight linear classifier trained on diffusion model intermediate-layer activations, used to guide LLM decoding direction.

The hyperparameter $\lambda$ controls the relative weight of the two terms—larger $\lambda$ yields stronger bias exposure but may reduce prompt naturalness.

### Key Designs

#### 1. Attribute Classifier

A lightweight linear classification head trained on UNet intermediate-layer activations from Stable Diffusion 1.5:
- Input: UNet intermediate-layer features at a given diffusion step
- Output: probabilities over sensitive attributes (2-class gender / 4-class race)
- Averaged over $K$ generation samples to reduce single-sample variance

#### 2. Beam Search Decoding

Leverages the autoregressive factorization of the LLM, $\mathbb{P}(\boldsymbol{s}) = \prod_{i=1}^N p(s_i | s_{<i})$, for token-level scoring and search:

| Parameter | Description |
|-----------|-------------|
| Beam width $B$ | Number of high-scoring candidates retained |
| Expansion factor $E$ | Number of candidates expanded per step |
| Additional expansion $E'$ | Sampling factor for increasing diversity |

**Prompt diversity guarantee**:
- The first token is sampled directly from the full LLM logits distribution
- Subsequent tokens are selected via $B \times E$ sampling followed by top-$B$ scoring
- Beams that reach EOS are saved and removed from the pool

#### 3. LLM Instruction Design

The LLM is explicitly instructed to:
- Generate attribute-neutral prompts (without mentioning gender, race, etc.)
- Generate natural prompts that typical users might input
- Default LLM: Mistral-7B-v0.2

### Loss & Training

BGPS does not involve model training. The attribute classifier is a pretrained linear head. The optimization procedure is:
- Search algorithm: guided beam search
- Evaluation: 100 discovered prompts × 10 images each → classification statistics
- Perplexity evaluation: uses GPT-2 (distinct from the search LLM) to ensure independent assessment

## Key Experimental Results

### Main Results

**Table 1: Male-biased prompt discovery (Mistral-7B-0.2)**

| Method | Base Male%↑ | Debiased-FT Male% | Debiased-DL Male% | PPL↓ | Explicit Gender%↓ |
|--------|------------|-------------------|-------------------|------|-------------------|
| Manual curation | 0.53 | 0.49 | 0.31 | 96 | 0 |
| PEZ (gradient-based) | 0.80 | 0.78 | **0.84** | 1387 | 94 |
| LLM only | 0.69 | 0.59 | 0.44 | 71 | 1 |
| **BGPS (λ=10)** | 0.76 | 0.66 | 0.46 | **52** | 2 |
| **BGPS (λ=100)** | **0.92** | **0.79** | 0.70 | 122 | 17 |

Key finding: BGPS (λ=100) achieves 92% male ratio on the base model with a perplexity of only 122 (far below PEZ's 1387), with only 17% of prompts explicitly mentioning gender.

**Table 2: Gender bias amplification for specific occupations**

| Occupation | LLM Male% | BGPS Male% | LLM Female% | BGPS Female% |
|------------|-----------|-----------|------------|-------------|
| Engineer | 0.73 | **0.84** | 0.21 | **0.68** |
| Doctor | 0.67 | **0.82** | 0.33 | **0.78** |
| Nurse | 0.40 | **0.61** | 0.52 | **0.87** |
| Scientist | 0.69 | **0.83** | 0.29 | **0.64** |

### Ablation Study

**Comparison across LLMs (3 LLMs × 3 models)**:
- Mistral-7B-0.2, Qwen3-8B, and Llama-3.2-1B can all be effectively guided by BGPS
- The smaller Llama-1B model yields comparable PPL but more prompts explicitly mention gender (weaker instruction-following)
- BGPS is robust to LLM choice

**Beyond occupational bias (Table 3)**:

| Context | Condition | Male% | Female% |
|---------|-----------|-------|---------|
| Objects | LLM only | 0.10 | 0.00 |
| Objects | BGPS (male-biased) | **0.54** | 0.26 |
| Activities | BGPS (male-biased) | **0.73** | 0.07 |
| Scenes | BGPS (male-biased) | **0.80** | 0.10 |

### Key Findings

1. **Fragility of debiased models**: LoRA fine-tuned debiased models achieve 49% male ratio on standard prompts (balanced), but BGPS prompts push this to 79%.
2. **Dramatic effect of linguistic modifiers**: adding "with intense focus" to "scientist" raises the male ratio from 65% to 95%.
3. **Systematic linguistic associations**: cognitive words ("serious", "concerned") → male; emotional words ("compassionate", "joyful") → female.
4. **BGPS introduces new words rather than amplifying existing biased ones**: approximately half of the substituted words are novel bias-associated terms.

## Highlights & Insights

1. **Breakthrough on the coverage–interpretability trade-off**: BGPS-generated prompts have perplexity 17–26× lower than PEZ while maintaining comparable bias discovery capability.
2. **Auditing value for debiasing methods**: reveals the deeper problem that debiased models "pass the exam without truly learning."
3. **Semantic-level bias analysis**: word frequency analysis and the $\delta_w$ metric provide quantitative tools for understanding how bias is encoded.
4. **Practical auditing tool**: requires only gray-box access (intermediate-layer activations), applicable to commercial T2I system auditing.
5. **Complementary design**: BGPS-discovered bias prompts can be directly incorporated into debiasing training sets, forming a closed loop.

## Limitations & Future Work

1. In-depth validation is limited to Stable Diffusion 1.5 and its debiased variants; evaluation on newer models (SDXL, FLUX) is limited.
2. The attribute classifier covers only gender (2 classes) and race (4 classes); other sensitive attributes such as age and disability are not thoroughly explored.
3. The attribute classifier itself is trained on manually curated data, which may introduce additional bias.
4. Beam search has limited search efficiency; evolutionary algorithms or reinforcement learning could be explored.
5. The practical effectiveness of using BGPS-discovered bias prompts for debiasing training has not been evaluated.

## Related Work & Insights

- **VGD (Visually-Guided Decoding)**: the direct inspiration for BGPS—repurposing image inversion for bias discovery.
- **Difflens (SAE-based debiasing)**: one of the test targets for BGPS, which exposes its residual biases.
- **OpenBias**: uses LLM-proposed biases + VQA evaluation, but does not perform guided search.
- **GELDA**: a "semi-automated" framework in which LLMs propose biased modifiers.
- Insight: the guided decoding paradigm can be generalized to discover other types of model vulnerability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First method to automatically discover interpretable bias-eliciting prompts, filling a critical gap.
- Technical Contribution: ⭐⭐⭐⭐ — The combination of LLM guidance and attribute classifiers is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations across multiple LLMs, models, and attribute dimensions.
- Writing Quality: ⭐⭐⭐⭐ — Research motivation and positioning are very clearly articulated.
- Overall Recommendation: ⭐⭐⭐⭐⭐ — A work of significant societal impact with substantial contributions to T2I fairness research.

## Background & Motivation

## Core Problem

## Method

## Key Experimental Results

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Inspiration & Connections

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](../../CVPR2026/image_generation/autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [\[ICLR 2026\] The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICLR 2026\] Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions](temporal_concept_dynamics_in_diffusion_models_via_prompt-conditioned_interventio.md)

</div>

<!-- RELATED:END -->
