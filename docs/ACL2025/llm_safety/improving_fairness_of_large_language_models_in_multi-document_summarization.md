---
title: >-
  [Paper Note] Improving Fairness of Large Language Models in Multi-document Summarization
description: >-
  [ACL 2025][LLM Safety][Fairness] Proposes FairPO (Fair Preference Optimization), which optimizes both summary-level and corpus-level fairness in multi-document summarization through perturbation-based preference pair generation and fairness-aware preference tuning.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Fairness"
  - "Multi-document Summarization"
  - "Preference Optimization"
  - "DPO"
  - "Equal Coverage"
date: 2026-05-08
content_hash: 964e2beddeba4761
---

# Improving Fairness of Large Language Models in Multi-document Summarization

**Conference**: ACL 2025  
**arXiv**: [2506.07479](https://arxiv.org/abs/2506.07479)  
**Code**: [GitHub](https://github.com/leehaoyuan/coverage_fairness)  
**Area**: AI Safety  
**Keywords**: Fairness, Multi-document Summarization, Preference Optimization, DPO, Equal Coverage  

## TL;DR

Proposes FairPO (Fair Preference Optimization), which optimizes both summary-level and corpus-level fairness in multi-document summarization through perturbation-based preference pair generation and fairness-aware preference tuning.

## Background & Motivation

### 1. Background
Multi-document summarization (MDS) aims to extract key information from multiple documents, such as summarizing multiple reviews of a product. Each document usually carries social attributes (e.g., sentiment: positive/neutral/negative), and documents with different attributes may contain conflicting viewpoints.

### 2. Limitations of Prior Work
- Existing methods mainly focus on **summary-level fairness** (whether a single summary balancedly covers documents of different attributes), neglecting **corpus-level fairness** (whether certain attributes are systematically over- or under-represented in the entire corpus).
- The prompting method of Zhang et al. (2023) relies on user prior knowledge of fairness issues, offering limited practicality.
- The policy gradient method of Huang et al. (2024) is designed for T5 and is difficult to generalize to modern LLMs.
- All of these methods only focus on summary-level fairness.

### 3. Key Challenge
LLMs exhibit both summary-level bias (a single summary biasing toward certain attributes) and corpus-level bias (systematically over-representing certain attributes) in multi-document summarization, but existing methods cannot optimize both simultaneously.

### 4. Goal
Propose a preference tuning method that can simultaneously improve summary-level and corpus-level fairness, without depending on user prior knowledge, and without compromising other aspects of summary quality (relevance, factuality, fluency).

### 5. Key Insight
Combine preference tuning (DPO) with fairness metrics to generate preference pairs reflecting differences in fairness by perturbing input document sets, and then optimize corpus-level fairness via dynamic weighting.

### 6. Core Idea

**Generate preference pairs by "perturbing" input document sets (removing some documents with specific attributes), and utilize a DPO variant with dynamic weight allocation to simultaneously optimize both levels of fairness.**

## Method

### Overall Architecture

FairPO consists of two core modules:
1. **Perturbation-based Preference Pair Generation**
2. **Fairness-aware Preference Tuning**

### Key Designs

#### Module 1: Perturbation-based Preference Pair Generation

For each document set $D$, FairPO first generates an initial summary $S$, identifying the most over-represented attribute value $k^+$ and the most under-represented attribute value $k^-$ (based on differences in coverage probability). Then, it removes $\alpha\%$ of documents with $k^+$ and $k^-$ attributes from the document set respectively, generating perturbed summaries $S^+$ and $S^-$.

Among $S, S^+, S^-$:
- The one with the **lowest Equal Coverage** serves as the chosen summary $S_c$ (most fair).
- The one with the **highest Equal Coverage** serves as the rejected summary $S_r$ (least fair).

The coverage probability is estimated via an entailment model:

$$p(d_i, s_j) = \max\{p(d_{i,l}, s_j) | d_{i,l} \in d_i\}$$

$$EC(D,S) = \frac{1}{K}\sum_{k=1}^{K}|\mathbb{E}(\{c(d_i, S) | a_i = k\})|$$

#### Module 2: Fairness-aware Preference Tuning

FairPO modifies the DPO objective function, introducing separated weights $w_c$ and $w_r$:

$$\sigma(-m) \cdot \beta\left(w_r \log\frac{\pi_\theta(S_r|D)}{\pi_{ref}(S_r|D)} - w_c \log\frac{\pi_\theta(S_c|D)}{\pi_{ref}(S_c|D)}\right)$$

The weights are calculated based on the dynamic estimation of corpus-level fairness. For each attribute value $k$, FairPO estimates the over-representation $O(k)$ and under-representation $U(k)$:

$$O(k) = \frac{\sum_{(D,S)\in T_k^+}|C_k(D,S)| \cdot \pi_\theta(S|D)/|S|}{\sum_{(D,S)\in T_k^+}\pi_\theta(S|D)/|S|}$$

The intermediate weights are calculated using a sigmoid-like function:

$$w_{c,k} = \frac{2}{1 + (O(k)/U(k))^{C_k(D,S_c)/\tau}}$$

**Design Intuition**: If an attribute value is systematically under-represented in the corpus ($U(k)>O(k)$), and the chosen summary happens to over-represent it ($C_k>0$), then this chosen summary helps balance the corpus-level fairness and should receive a higher weight.

### Loss & Training
- Fine-tuned using LoRA, with learning rate $5e-5$, batch size 16, and trained for 2 epochs.
- Perturbation ratio $\alpha = 10\%$.
- Temperature parameter $\tau$ is tuned on the validation set according to the dataset and model (range 1-3).

## Key Experimental Results

### Main Results

| Method | Amazon EC↓ | Amazon CP↓ | MITweet EC | MITweet CP↓ | SemEval EC↓ | SemEval CP↓ | Avg EC↓ | Avg CP↓ |
|------|-----------|-----------|-----------|------------|-----------|------------|---------|---------|
| Llama3.1 | 7.95 | 1.89 | 4.50 | 0.59 | 2.98 | 1.41 | 5.14 | 1.30 |
| +DPO | 7.23 | 1.27 | 4.25 | 0.47 | 2.66 | 1.09 | 4.72 | 0.94 |
| +OPTune | 6.70 | 0.62 | 4.33 | 0.51 | 2.60 | 0.95 | 4.54 | 0.69 |
| **+FairPO** | **6.87** | **0.42** | **4.24** | **0.42** | **2.49** | **0.66** | **4.53** | **0.50** |
| Gemma2 | 8.32 | 2.48 | 4.20 | 0.60 | 2.81 | 0.96 | 5.11 | 1.35 |
| **+FairPO** | **6.18** | **0.44** | **3.76** | **0.48** | **2.50** | **0.45** | **4.15** | **0.46** |

FairPO achieves the best overall performance across all models, especially with significant improvements in CP (corpus-level fairness).

### Ablation Study

| Variant | Overall EC↓ | Overall CP↓ |
|------|-----------|-----------|
| FairPO | **4.39** | **0.39** |
| w/o pert. (no perturbation) | 4.54 | 0.54 |
| w/o fair. (no fairness-aware weights) | 4.42 | 0.64 |
| w/o rew. (no reward margin) | 4.42 | 0.64 |

Both modules contribute to the performance; removing either of them leads to a performance drop.

### Key Findings
1. **Human Evaluation**: Out of 30 summary pairs, FairPO is fairer in 18 pairs (vs 9 pairs for DPO), showing a significant difference ($p<0.05$).
2. **Summary Quality Preservation**: FairPO's fluency, relevance, and factuality are comparable to those of the original LLM, while the Prompt method significantly degrades quality.
3. **Three Datasets**: Covering different social attributes (sentiment, political ideology, stance), with document set sizes ranging from 8 to 30.

## Highlights & Insights

1. **Simultaneous optimization of two-level fairness** is the core contribution of this paper, whereas prior work only focused on the summary level.
2. **Perturbation-based preference pair generation** is highly ingenious—by targetedly removing documents to "amplify" the model's bias tendencies, it generates preference pairs with significant fairness differences.
3. **Dynamic weight allocation** is exquisitely designed, allowing real-time tracking of corpus-level fairness status during training and adjustment of the optimization direction.
4. The method is highly generalizable and applicable to various LLMs (Llama, Mistral, Gemma) without modifying the model architecture.

## Limitations & Future Work

1. Currently, fairness is optimized only within a single domain; simultaneously optimizing multiple social attributes across domains is a more challenging scenario.
2. From the three candidate summaries, only two are selected to construct preference pairs; how to exploit the information of all three summaries is worth exploring.
3. Fairness metrics rely on entailment models, whose inherent biases may affect evaluation.
4. The scale of the experiments is limited (1000 training / 300 test per dataset), and performance on a larger scale remains to be validated.

## Related Work & Insights

- **DPO Preference Tuning** (Rafailov et al., 2024): The foundation of FairPO, optimizing policies directly through preference pairs.
- **OPTune** (Chen et al., 2024): Online preference tuning, weighted by EC differences—FairPO goes a step further by considering corpus-level fairness.
- **Insight**: The weighting strategy of preference tuning can flexibly inject different optimization objectives (fairness, safety, diversity, etc.), not limited to quality improvement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to apply preference tuning to MDS fairness, with an innovative perturbation + dynamic weighting design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 3 datasets, 3 models, ablation studies, human evaluation, and quality evaluation, which are quite complete.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, complete methodology derivation, and well-organized experiments.
- **Value**: ⭐⭐⭐⭐ — Fairness is an important problem in MDS, and the method is both general and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)
- [\[ACL 2025\] ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)
- [\[ACL 2025\] A Statistical and Multi-Perspective Revisiting of the Membership Inference Attack in Large Language Models](a_statistical_and_multi-perspective_revisiting_of_the_membership_inference_attac.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](../../NeurIPS2025/llm_safety/distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)

</div>

<!-- RELATED:END -->
