---
title: >-
  [Paper Note] Data-Constrained Synthesis of Training Data for De-Identification
description: >-
  [ACL 2025][LLM Pretraining][Synthetic Data] This work systematically investigates how to generate synthetic clinical text using domain-adapted LLMs under data-constrained conditions and how to train NER models for Personal Identifiable Information (PII) detection via machine labeling. The study reveals that the quality of the machine labeler, rather than the scale of the generative model, is the key factor determining the utility of synthetic data.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Synthetic Data"
  - "De-identification"
  - "Data Constraints"
  - "Domain Adaptation"
  - "PII Detection"
date: 2026-05-08
content_hash: 4fb3cac1b9636ef2
---

# Data-Constrained Synthesis of Training Data for De-Identification

**Conference**: ACL 2025  
**arXiv**: [2502.14677](https://arxiv.org/abs/2502.14677)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Synthetic Data, De-identification, Data Constraints, Domain Adaptation, PII Detection

## TL;DR

This work systematically investigates how to generate synthetic clinical text using domain-adapted LLMs under data-constrained conditions and how to train NER models for Personal Identifiable Information (PII) detection via machine labeling. The study reveals that the quality of the machine labeler, rather than the scale of the generative model, is the key factor determining the utility of synthetic data.

## Background & Motivation

In sensitive domains such as clinical settings, widely available datasets are extremely scarce due to privacy risks. Traditional de-identification methods rely on human annotation of PII (Personally Identifiable Information), which is costly and time-consuming. Automated de-identification usually relies on NER models to detect PII that needs to be removed, but the PII datasets required to train these NER models are themselves sensitive, creating a circular dilemma.

With the improvement of LLM generation capabilities, leveraging synthetic datasets to address data scarcity has become a feasible solution. However, prior work has mainly focused on privacy evaluations of synthetic data or how to build the strongest models, with little systematic study on how to effectively generate synthetic data under resource-constrained conditions. This paper fills this gap by exploring the following core questions:

1. How much data is required for domain adaptation?
2. What is the impact of the machine labeler's quality?
3. How does the volume of synthetic data affect downstream tasks?
4. Is the scale of the generative model critical?

## Method

### Overall Architecture

The overall workflow consists of four steps: (1) performing domain adaptation on a general LLM using sensitive gold-standard corpora; (2) generating synthetic clinical text using the adapted LLM; (3) machine-labeling the synthetic text using an NER model trained on gold-standard data; (4) training a new NER model with the machine-labeled synthetic corpus.

### Key Designs

1. **Domain Adaptation of Generative Models**: QLoRA ($r=8$, $\alpha=32$) is used to fine-tune GPT-SW3 (Swedish, 6.7B parameters) and FLOR (Spanish, 6.3B parameters) for domain adaptation. Autoregressive language modeling training is conducted using clinical data without instruction tuning to maintain simplicity. The first three words of the validation set are used as generation prompts.

2. **Synthetic Text Generation**: Inference is performed using the vLLM library with nucleus sampling ($p=0.95$) and a temperature of 1.0. Eighty samples are generated per prompt, making the synthetic corpus four times the size of the original data. The advantage of synthetic data is that its scale is limited only by computational resources.

3. **Machine Labeling**: SweDeClin-BERT (Swedish) and roberta-base-bne (Spanish) are used as encoder models, fine-tuned on gold-standard data, and applied to PII labeling. Documents are chunked into 128-word segments to fit the context window.

4. **Cross-lingual Validation**: Validation is conducted on Swedish (SEPR PHI, 21,553 sentences) and Spanish (MEDDOCAN, 1,000 medical texts), enhancing the generalizability of the findings.

### Loss & Training

- Generative models are fine-tuned via QLoRA using autoregressive language modeling loss.
- NER models are trained for 6 epochs with a batch size of 16.
- All experiments are conducted using 5-fold cross-validation.
- Token-level F1-score is used as the evaluation metric.

## Key Experimental Results

### Main Results — Constraining Total Data Volume (Table 1)

| Data Ratio | SEPR PHI Gold | SEPR PHI Synthetic | Δ | MEDDOCAN Gold | MEDDOCAN Synthetic | Δ |
|---------|--------------|-------------------|------|--------------|-------------------|------|
| 5% | 0.707 | 0.724 | -0.017 | 0.931 | 0.309 | 0.622 |
| 25% | 0.871 | 0.847 | 0.024 | 0.967 | 0.964 | 0.003 |
| 50% | 0.908 | 0.885 | 0.023 | 0.973 | 0.970 | 0.003 |
| 95% | 0.926 | 0.896 | 0.029 | 0.978 | 0.973 | 0.005 |

### Ablation Study — Impact of Various Factors

**Domain Adaptation Data Volume (Table 2):**

| Domain Adaptation Data | SEPR PHI | MEDDOCAN |
|------------|---------|---------|
| 0% | 0.547 | 0.295 |
| 5% | 0.873 | 0.313 |
| 25% | 0.877 | 0.970 |
| 50% | 0.896 | 0.970 |
| 95% | 0.896 | 0.973 |
| Gold | 0.926 | 0.978 |

**Impact of Model Scale (Table 4):**

| Model Scale | SEPR PHI | MEDDOCAN |
|---------|---------|---------|
| Small (~1.3B) | 0.883 | 0.973 |
| Large (~6.5B) | 0.896 | 0.973 |
| Gold | 0.926 | 0.978 |

**Impact of Synthetic Data Volume (Table 5):**

| Synthetic Volume | SEPR PHI | MEDDOCAN |
|-------|---------|---------|
| 5% | 0.814 | 0.938 |
| 100% | 0.889 | 0.968 |
| 400% | 0.896 | 0.973 |

### Key Findings

1. **Diminishing Marginal Returns in Domain Adaptation**: Using 25%–50% of the data yields near-optimal domain adaptation, with almost no further improvement when increasing to 95%.
2. **Machine Labeler Quality is the Core Bottleneck**: Comparing Table 2 and Table 3, when a high-quality labeler is fixed, the synthetic model performance closely tracks the gold standard. Conversely, when high-quality domain adaptation is fixed, variations in labeler quality directly dictate downstream performance.
3. **Small Models are Sufficient**: Generative models with 1.3B parameters yield almost identical performance to 6.5B models, showing completely identical results on MEDDOCAN.
4. **Limited Marginal Gain from Synthetic Data Volume**: The difference between 100% and 400% synthetic volume is within one standard deviation.
5. **Privacy Aspects**: More domain adaptation data actually reduces n-gram recall (with 5-gram recall dropping from 0.328 to 0.122), indicating that more thorough training mitigates memorization.

## Highlights & Insights

- **Counter-intuitive Finding**: In the synthetic data pipeline, the scale and data volume of the generative model are not the bottlenecks; rather, the labeling model is. This is highly significant for resource-constrained institutions, suggesting they should prioritize investing in high-quality labeling models.
- **A New Perspective on the Privacy-Utility Trade-off**: Utilizing more data actually decreases memorization risk. As more unique n-grams are learned, the probability of memorizing any single n-gram drops.
- **Practicality of the Approach**: This provides a viable solution for cross-institutional collaboration, where institutions can share synthetic data instead of raw sensitive data.

## Limitations & Future Work

- PII detection may be a task with lower domain-specificity requirements. Whether the conclusions generalize to highly domain-specific tasks (e.g., ICD-diagnostic coding) remains to be verified.
- Instruction tuning was not utilized; tasks requiring document-level semantics may need different strategies.
- Privacy evaluation is solely based on n-gram metrics, which has limitations in quantifying real privacy risks.
- Only two languages were evaluated; validation across more languages would strengthen the generalizability.

## Related Work & Insights

- Libbi et al. (2021) utilized GPT-2 and 1 million documents for a similar pipeline. This work achieves comparable performance with significantly less data and more modern techniques.
- Xu et al. (2023) conducted data-constraint experiments on relation extraction tasks. This work shifts the focus to NER/PII detection.
- Differentially private learning (Yue et al., 2023; Igamberdiev et al., 2024) represents an alternative route, but it remains difficult to implement and highly inefficient.

## Rating

- **Novelty**: 7/10 — Although the method itself is not entirely new, the systematic ablation study design and the "labeler is king" finding are novel.
- **Experimental Thoroughness**: 9/10 — Highly comprehensive evaluation featuring bilingual validation, 5-fold cross-validation, and multi-dimensional ablations (data volume, model scale, synthetic volume, privacy).
- **Writing Quality**: 8/10 — Clear structure, highly logical experimental design, and intuitive tables.
- **Value**: 8/10 — Direct practical value for clinical NLP and privacy-preservation communities, serving as a guideline for resource-constrained organizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models](optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)
- [\[ACL 2025\] Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)

</div>

<!-- RELATED:END -->
