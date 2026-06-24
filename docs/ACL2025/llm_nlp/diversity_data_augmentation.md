---
title: >-
  [Paper Note] Diversity-oriented Data Augmentation with Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Data Augmentation] This paper proposes the DoAug framework, which fine-tunes an LLM paraphraser using SFT+DPO, combined with coreset selection and diversity sampling. While maintaining semantic coherence, it significantly enhances the diversity of the augmented dataset, leading to an average performance improvement of 10.52% across 12 datasets, outperforming the sub-optimal baseline by 3.76 percentage points.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Data Augmentation"
  - "Diversity"
  - "Paraphrase Generation"
  - "DPO"
  - "Coreset Selection"
  - "Text Classification"
date: 2026-05-08
content_hash: d338502fef76b369
---

# Diversity-oriented Data Augmentation with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.11671](https://arxiv.org/abs/2502.11671)  
**Code**: [https://github.com/CNICDS/DoAug](https://github.com/CNICDS/DoAug)  
**Area**: LLM/NLP  
**Keywords**: Data Augmentation, Diversity, Paraphrase Generation, DPO, Coreset Selection, Text Classification  

## TL;DR

This paper proposes the DoAug framework, which fine-tunes an LLM paraphraser using SFT+DPO, combined with coreset selection and diversity sampling. While maintaining semantic coherence, it significantly enhances the diversity of the augmented dataset, leading to an average performance improvement of 10.52% across 12 datasets, outperforming the sub-optimal baseline by 3.76 percentage points.

## Background & Motivation

**Three Elements of High-Quality Datasets**: The authors point out that a high-quality dataset for training NLP models should possess three characteristics: large scale (Large), coherent labels (Coherent), and diverse distributions (Diverse). However, existing data augmentation methods focus almost exclusively on expanding data volume, neglecting diversity.

**Limitations of Prior Work**: Early random perturbation methods (e.g., EDA, AEDA) are prone to introducing noise that disrupts label coherence (such as deleting "not") or generating redundant samples that fail to improve diversity. Methods based on back-translation and BERT Unmask offer limited rewriting capability.

**Potential and Deficiencies of LLM Paraphrasing**: While methods like AugGPT directly leverage LLMs for paraphrasing to preserve semantics, they do not explicitly encourage diversified outputs, leading to highly repetitive generated texts.

**Relationship between Diversity and Performance**: Prior research (Gontijo-Lopes et al., 2020) indicates that model performance gains are maximized when both data diversity and affinity are jointly improved. However, no prior work has integrated diversity optimization with LLM-based paraphrase augmentation.

**Computational Efficiency Issues**: Performing LLM-based augmentation on every sample of a large dataset is computationally expensive, requiring a sample selection strategy to reduce costs.

**Goal**: To design a framework that can both maintain semantic consistency between augmented and original data (high affinity) and maximize dataset diversity, thereby significantly boosting downstream task performance.

## Method

### Overall Architecture

DoAug consists of four stages: (1) SFT to train the LLM paraphraser; (2) DPO diversity fine-tuning; (3) Coreset selection of samples to be augmented; (4) Diversity sampling to generate the final augmented dataset. The base LLM used is LLaMA-3.2-1B-Instruct (BF16).

### Module 1: LLM Paraphraser Training (SFT + LoRA)

- Sample 100k sentence pairs from the ChatGPT Paraphrases dataset as $\mathcal{D}_{\text{SFT}}$
- Utilize LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning, decomposing the weight update $\Delta W$ into $BA$ ($r \ll \min(d,k)$) while freezing the original weights $W_0$
- Training Objective: Enable the LLM to rewrite sentence expressions while keeping the semantics unchanged

### Module 2: DPO Diversity Enhancement

- **Preference Dataset Construction**: Sample 50k groups from the original paraphrase dataset, each containing one original sentence $x$ and five paraphrases $[y_1,...,y_5]$. Compute the Euclidean distance between each paraphrase and the original sentence in the embedding space. The paraphrase with the largest distance is designated as chosen ($y_w$), and the one with the smallest distance as rejected ($y_l$).
- **DPO Loss**: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$
- Encourage the LLM to construct paraphrases that are further from the original sentence (more diverse) using contrastive learning, while avoiding drifting too far from the SFT model.

### Module 3: Coreset Selection and Diversity Sampling

- **Coreset Selection**: First, train the downstream task model to collect training dynamics metrics (EL2N, entropy, variance, AUM). Based on importance, classify the samples into three groups: high importance (for augmentation), medium importance (to be kept), and low importance (to be pruned), in a 1:1:1 ratio.
- **Diversity Sampling**: For each seed sentence, use beam search to generate $K=5$ candidate paraphrases, sort them by embedding distance to the original sentence, and keep only the paraphrase with the largest distance (most diverse).
- **Final Dataset**: Original sentences + paraphrases of high-importance samples $\cup$ original sentences of medium-importance samples

### Training and Inference

- Both SFT and DPO employ LoRA fine-tuning, resulting in a small computational overhead (using a 1B model).
- During inference, beam search paraphrasing is performed sequentially on the selected coreset, keeping the overall pipeline costs manageable.

## Key Experimental Results

### Table 1: Downstream Classification Accuracy on 12 Datasets

| Method | ANLI | ChemProt | CoLA | MNLI | MPQA | MRPC | RCT | RTE | SST-2 | SUBJ | Symptoms | Yelp | Avg.Gain |
|------|------|----------|------|------|------|------|-----|-----|-------|------|----------|------|----------|
| Original | 35.75 | 58.33 | 74.56 | 42.81 | 89.17 | 76.50 | 71.62 | 53.61 | 86.97 | 95.75 | 74.06 | 51.48 | - |
| AugGPT | 36.43 | 65.73 | 75.17 | 53.77 | 89.67 | 75.25 | 78.90 | 54.87 | 87.63 | 95.44 | 79.25 | 55.47 | 5.64% |
| Taboo | 35.83 | 69.66 | 72.90 | 57.26 | 89.34 | 76.74 | 78.48 | **58.01** | 86.74 | 95.12 | 89.40 | 56.30 | 6.76% |
| **DoAug** | **38.46** | **70.22** | **75.62** | **59.76** | **89.78** | **80.97** | **80.10** | 56.05 | **88.64** | **95.80** | **90.74** | **56.57** | **10.52%** |

### Table 2: Six Diversity Metrics (Normalized to [0,1], Average Across 12 Datasets)

| Method | Distance | Dispersion | Radius | Homogeneity | Vocabulary | 3-grams | Average |
|------|----------|------------|--------|-------------|------------|---------|---------|
| Original | 0.00 | 0.00 | 0.78 | 0.74 | 0.00 | 0.00 | 0.25 |
| Hint | 0.56 | 0.51 | 0.98 | 0.86 | 0.45 | 0.68 | 0.67 |
| **DoAug** | **1.00** | **1.00** | 0.87 | 0.98 | **1.00** | **1.00** | **0.98** |

### Key Findings

- DoAug achieves optimal performance on 11 out of 12 datasets (with only RTE slightly lower than Taboo), achieving an average gain of 10.52%, which significantly outperforms the sub-optimal method's 6.76%.
- Among the 6 diversity metrics, 4 are optimal and 2 are close to optimal, yielding a comprehensive score of 0.98 (out of 1.0).
- The affinity (semantic preservation) is second only to Unmask (the latter is naturally biased to be high as it operates within the BERT embedding space).
- Human evaluation shows that 95% of the paraphrases preserve correct semantics, while DeepSeek-V3 evaluation shows 97%.
- Ablation study shows that coreset selection contributes the most; DPO primarily enhances lexical diversity, while diversity sampling mainly boosts sample-level diversity in the latent space.
- Substitution experiments demonstrate that DPO cannot be replaced by high-temperature sampling or prompt-based diversity incentives.
- Replacing the LLM augmenter with Qwen2.5-1.5B or switching the downstream model to GPT-2/T5-large preserves the performance advantages, proving the architecture-agnostic nature of the framework.

## Highlights & Insights

- Elevates data augmentation from mere "volume expansion" to "diversity enhancement", providing a clear and highly practical problem definition.
- The SFT $\rightarrow$ DPO two-stage training pipeline is elegant: SFT guarantees paraphrasing capabilities, while DPO explicitly guides diversification.
- The construction method of preference data is ingenious—requiring no human annotation, it automatically selects chosen/rejected pairs using embedding distances.
- Coreset selection makes the method naturally suitable for low-resource scenarios, reducing LLM inference overhead while focusing on high-value samples.
- Solid empirical scale: 12 datasets $\times$ 12 baselines $\times$ 10 random seeds, providing highly reliable statistical results.

## Limitations & Future Work

- There is no unified standard for evaluating diversity, and the six metrics adopted in ours may not fully capture the complete concept of diversity.
- It is validated only on sentence-level classification tasks and English corpora, without extension to generative tasks such as mathematical reasoning, instruction following, and creative writing.
- Multimodal scenarios and cross-lingual generalizability are not considered.
- Coreset selection relies on prior training of a downstream model to collect training dynamics, which increases the complexity of the overall workflow.
- Utilizing LLMs for augmentation carries potential risks of amplifying demographic bias and producing factual hallucinations.
- The construction of DPO preferences depends heavily on the quality of the embedding model, and different embeddings may yield varying results.

## Related Work & Insights

- **Text Data Augmentation**: Character/word-level perturbations (EDA, AEDA), back-translation, BERT Unmask, LLM paraphrasing (AugGPT, Self-LLMDA), and diversity-incentivized prompting (Chain/Hint/Taboo, Cegin et al. 2024).
- **Dataset Diversity Evaluation**: Token-level and embedding-level metrics (Tevet & Berant 2021, Lai et al. 2020, Yu et al. 2022), and joint diversity-affinity analysis (Gontijo-Lopes et al. 2020).
- **Preference Alignment**: RLHF/PPO (Ouyang et al. 2022), DPO (Rafailov et al. 2024).
- **Coreset Selection**: Training dynamics metrics such as EL2N / entropy / AUM (Paul et al. 2021, Coleman et al. 2020, Pleiss et al. 2020), and CCS (Zheng et al. 2023).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to apply DPO preference alignment to optimize diversity targets in data augmentation. The automatic construction method for preference data is novel.
- **Effectiveness**: ⭐⭐⭐⭐⭐ — Comprehensive experiments on 12 datasets $\times$ 12 baselines $\times$ 10 random seeds render the average gain of 10.52% highly convincing.
- **Practicality**: ⭐⭐⭐⭐ — Based on a compact 1B model + LoRA fine-tuning, demonstrating practical applicability in low-resource text classification scenarios.
- **Clarity**: ⭐⭐⭐⭐ — The framework is clearly described, and the ablation and substitution experiments sufficiently explain the role of each component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)

</div>

<!-- RELATED:END -->
