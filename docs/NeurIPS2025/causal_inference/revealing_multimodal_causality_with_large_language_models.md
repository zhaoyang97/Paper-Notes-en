---
title: >-
  [Paper Note] Revealing Multimodal Causality with Large Language Models
description: >-
  [NeurIPS 2025][Causal Inference][Multimodal causal discovery] This paper proposes MLLM-CD, the first framework for causal discovery from multimodal unstructured data (text + images). It identifies causal variables via co…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "Multimodal causal discovery"
  - "large language models"
  - "contrastive factor discovery"
  - "counterfactual reasoning"
  - "unstructured data"
date: 2026-05-08
content_hash: d58dc448219ce33e
---

# Revealing Multimodal Causality with Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.17784](https://arxiv.org/abs/2509.17784)
**Code**: [GitHub](https://github.com/JinLi-i/MLLM-CD)
**Area**: Causal Inference
**Keywords**: Multimodal causal discovery, large language models, contrastive factor discovery, counterfactual reasoning, unstructured data

## TL;DR

This paper proposes MLLM-CD, the first framework for causal discovery from multimodal unstructured data (text + images). It identifies causal variables via contrastive factor discovery, infers causal structure through statistical methods, and resolves structural ambiguity via iterative multimodal counterfactual reasoning.

## Background & Motivation

- **Background**: Causal discovery aims to infer causal structures from data and is fundamental to scientific progress. Traditional methods rely on predefined structured variables and cannot directly handle unstructured data (text, images, etc.). With the proliferation of multimodal data (e.g., clinical notes + medical images + lab results in medical diagnosis), causal discovery from multimodal unstructured data has become increasingly pressing.
- **Limitations of Prior Work**: Although LLMs have advanced text-based causal discovery (e.g., COAT), extending them to multimodal settings faces two key challenges: (1) **Difficulty in cross-modal factor discovery**: causal variables may be embedded across modalities and only identifiable through cross-modal interaction (e.g., "smaller apples score lower" requires joint understanding of images and text); (2) **Insufficient handling of structural ambiguity**: multiple causal structures can produce identical statistical dependencies from purely observational data, and the greater number of variables in multimodal settings exacerbates this ambiguity.
- **Key Challenge**: Naively extending COAT to multimodal settings discovers only a limited number of causal factors and leaves inferred causal edges undirected, falling far short of solving the multimodal causal discovery problem.

## Method

### Overall Architecture

MLLM-CD consists of three core modules operating iteratively: (1) a Contrastive Factor Discovery (CFD) module that leverages MLLMs to identify multimodal causal variables from contrastive sample pairs; (2) a statistical causal structure discovery module (e.g., the FCI algorithm) that infers causal relationships; and (3) an Iterative Multimodal Counterfactual Reasoning (MCR) module that generates counterfactual samples to resolve structural ambiguity and iteratively refine the causal graph.

### Key Designs

**1. Contrastive Factor Discovery (CFD) Module**

- **Function**: Identifies a complete set of causal variables from multimodal unstructured data.
- **Mechanism**: Comprises intra-modal and inter-modal contrastive exploration. **Intra-modal contrast**: selects top-$K$ sample pairs $\mathcal{P}_i$ with maximum semantic distance within each modality and prompts the MLLM to analyze variables implied by their differences. **Inter-modal contrast**: constructs sample pairs with maximum cross-modal mismatch, scored as $s(a,b) = (1 - \text{sim}(\mathbf{e}_{ai}, \mathbf{e}_{bj})) + |y_i - y_j|$, and prompts the MLLM to identify variables hidden in cross-modal dependencies. The MLLM then merges and deduplicates the discovered variables and annotates each sample with variable values.
- **Design Motivation**: Relying solely on the MLLM's general knowledge tends to surface only the most salient factors (e.g., taste, aroma), whereas contrastive signals can reveal latent but important factors (e.g., nutritional content).

**2. Iterative Multimodal Counterfactual Reasoning (MCR) Module**

- **Function**: Resolves ambiguities (e.g., undirected edges) in the causal structure through counterfactual sample generation.
- **Mechanism**: Performs counterfactual interventions on a variable $V_a$ involved in an uncertain relationship — the MLLM predicts how other variables would change under hypothetical values of $V_a$ and generates corresponding multimodal counterfactual samples. Generated samples undergo dual validation: (1) **Semantic validity**: ensures embedding similarity between counterfactual and original samples $\geq \tau_{\text{sem}}$; (2) **Causal consistency**: verifies that the proportion of changed non-descendant nodes $R_{\text{indep}} \leq \tau_{\text{causal}}$. Validated samples are added to the dataset for the next round of causal discovery.
- **Design Motivation**: The Markov equivalence class problem arising from purely observational data can only be resolved by introducing interventional or counterfactual data. The world knowledge of MLLMs provides counterfactual evidence beyond what is available in observational data.

**3. Statistical Causal Structure Discovery**

- **Function**: Infers a causal DAG from structured data.
- **Mechanism**: Employs the FCI algorithm to handle potential latent confounders, taking the structured data $\mathcal{D}_S^{(t)}$ and variable set $\mathbf{V}^{(t)} \cup \{Y\}$ output by CFD as input to produce a causal graph $\mathcal{G}^{(t)}$.
- **Design Motivation**: Statistical methods provide theoretical rigor for causal inference, while MLLM reasoning serves as a complement rather than a replacement.

### Loss & Training

No model training is involved. The framework uses four MLLMs: GPT-4o, Gemini 2.0, LLaMA 4 Maverick, and Grok-2v. CLIP is used to extract semantic representations for contrastive exploration, and counterfactual images are generated using Stable Diffusion 3.5 or Gemini 2.0.

## Key Experimental Results

### Main Results: MAG Dataset (Gemini 2.0)

| Method | NF ↑ | AF ↑ | ESHD ↓ |
|--------|------|------|--------|
| META | 0.67 | 0.51 | 18.67 |
| COAT | 0.51 | 0.37 | 16.00 |
| Pairwise | - | 0.51 | 30.00 |
| **MLLM-CD** | **0.87** | **0.60** | **14.00** |

### Ablation Study (Gemini 2.0)

| Variant | MAG NF | MAG AF | MAG ESHD | Lung NF | Lung AF | Lung ESHD |
|---------|--------|--------|----------|---------|---------|-----------|
| w/o Both | 0.54 | 0.41 | 16.33 | 0.55 | 0.13 | 9.67 |
| w/o CFD | 0.73 | 0.47 | 15.00 | 0.62 | 0.36 | 8.00 |
| w/o CR | 0.81 | 0.52 | 15.67 | 0.94 | 0.38 | 5.33 |
| **MLLM-CD** | **0.87** | **0.60** | **14.00** | **0.97** | **0.87** | **4.67** |

### Key Findings

1. **Substantial gains in factor discovery**: MLLM-CD achieves an average NF of 0.89 (across 4 MLLMs), far surpassing COAT (0.53) and META (0.52).
2. **Significant improvement in structure discovery**: Average ESHD decreases from 16.42 (COAT) to 13.42.
3. **Complementary roles of CFD and MCR**: CFD primarily improves factor identification completeness, while MCR primarily enhances causal structure precision.
4. **MCR is more impactful on small datasets**: On the Lung Cancer dataset, MCR improves AF from 0.38 to 0.87.
5. **Consistent effectiveness across MLLMs**: MLLM-CD achieves top performance on GPT-4o, Gemini 2.0, LLaMA 4, and Grok-2v.

## Highlights & Insights

- The first causal discovery framework targeting multimodal unstructured data, significantly broadening the applicability of causal discovery.
- The dual intra-modal/inter-modal contrastive exploration strategy in CFD is elegantly designed and effectively addresses the identification of latent causal variables.
- The dual validation mechanism (semantic + causal consistency) in the MCR module skillfully balances MLLM knowledge injection with statistical rigor.
- Establishes the first benchmark datasets for multimodal unstructured causal discovery (MAG + Lung Cancer).

## Limitations & Future Work

- Benchmark datasets are relatively small (MAG: 200 samples; Lung Cancer: 60 samples), and scalability remains to be validated.
- The range of modalities that MLLMs can process is limited by their inherent capabilities; sensor data, genomic data, etc., cannot be handled directly.
- Ground-truth causal graphs depend on domain expert knowledge.
- MLLMs may suffer from hallucinations and training data biases, affecting counterfactual reasoning quality.
- Future work plans to develop larger-scale benchmarks, expand modality coverage, and investigate uncertainty quantification.

## Related Work & Insights

- **LLM-based causal discovery**: COAT pioneered LLM-driven causal discovery from unstructured data but was limited to the text modality.
- **Causal representation learning**: Extracts high-level representations and causal dependencies from low-level observations, though practical application remains challenging.
- **Insights**: The world knowledge of MLLMs can serve as a source of counterfactual evidence beyond observational data, offering a fundamentally new methodological perspective for causal discovery. The integration of statistical methods with LLM reasoning represents an important direction for the field.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First framework for multimodal unstructured causal discovery; both the contrastive factor discovery and counterfactual reasoning modules are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic + real datasets, 4 MLLMs, comprehensive ablations, and sampling strategy analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous problem formulation and detailed method description.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for multimodal causal discovery with important application prospects in medical diagnosis and beyond.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[NeurIPS 2025\] From Black-box to Causal-box: Towards Building More Interpretable Models](from_black-box_to_causal-box_towards_building_more_interpretable_models.md)
- [\[NeurIPS 2025\] Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)

</div>

<!-- RELATED:END -->
