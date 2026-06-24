---
title: >-
  [Paper Note] CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization
description: >-
  [ACL 2025][Medical LLM][radiology report summarization] Proposes CSTRL, a context-driven sequential transfer learning approach for abstractive radiology report summarization. By optimizing Gap Sentence Generation (GSG) pre-training, utilizing Fisher matrix regularization to prevent catastrophic forgetting, and combining knowledge distillation for model compression, it significantly outperforms existing methods on the MIMIC-CXR and Open-I datasets.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "radiology report summarization"
  - "sequential transfer learning"
  - "knowledge distillation"
  - "Fisher matrix regularization"
  - "gap sentence generation"
date: 2026-05-08
content_hash: b8c531d862f151a1
---

# CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization

**Conference**: ACL 2025  
**arXiv**: [2503.05750](https://arxiv.org/abs/2503.05750)  
**Code**: [GitHub](https://github.com/fahmidahossain/Report_Summarization)  
**Area**: Medical NLP  
**Keywords**: radiology report summarization, sequential transfer learning, knowledge distillation, Fisher matrix regularization, gap sentence generation

## TL;DR

Proposes CSTRL, a context-driven sequential transfer learning approach for abstractive radiology report summarization. By optimizing Gap Sentence Generation (GSG) pre-training, utilizing Fisher matrix regularization to prevent catastrophic forgetting, and combining knowledge distillation for model compression, it significantly outperforms existing methods on the MIMIC-CXR and Open-I datasets.

## Background & Motivation

**Background**: Radiology reports consist of two core sections: Findings and Impression. Automatically generating the Impression from the Findings is an abstractive summarization task. Since 86% of radiologists receive less than an hour of formal training on drafting Impressions annually, automated generation is highly demanded.

**Limitations of Prior Work**: While pre-trained models perform well on general summarization tasks, they face severe challenges in the medical domain—specifically with complex medical terminology and the critical need for clinical contextual accuracy. For instance, losing context in phrases like "diverticulosis without diverticulitis" can lead to severe clinical misjudgment.

**Key Challenge**: Existing methods suffer from several limitations: (a) lacking methodologies that focus specifically on core findings; (b) contextual drift caused by subtle nuances in medical terminology; (c) generally low BLEU scores of generated Impressions; and (d) difficulties in reducing dimensionality and computational complexity for real-time production deployment.

**Goal**: To automatically generate high-quality Impressions from radiology report Findings while maintaining clinical contextual accuracy.

**Key Insight**: Adopting a sequential transfer learning strategy—first pre-training the model via an optimized GSG task to understand key sentences, then transferring it to the summarization task, using Fisher matrix regularization to prevent catastrophic forgetting, and finally compressing the model using knowledge distillation.

**Core Idea**: A two-step sequential transfer learning pipeline (GSG $\rightarrow$ summarization) secured by Fisher matrix regularization, combined with contextual tagging and knowledge distillation, to achieve highly accurate and low-complexity radiology report summarization.

## Method

### Overall Architecture

CSTRL comprises four core components: (1) optimized GSG pre-training; (2) sequential transfer learning based on Fisher matrix regularization; (3) Contextual Tagging; and (4) knowledge distillation (Teacher-Student). The base model selected is T5 (Text-to-Text Transfer Transformer).

### Key Designs

1. **Optimized GSG Technique**: Improving upon PEGASUS's GSG, this method assesses sentence importance using a combined metric of ROUGE and BLEU instead of solely ROUGE. For each sentence $x_i$, its importance score is $W_i = F1(\text{ROUGE}(x_i, D \setminus \{x_i\}) + \text{BLEU}(x_i, D \setminus \{x_i\}))$. After identifying key sentences, selective masking is applied based on sentence length (masking 3 words for $\ge 5$ words, 2 words for 4 words, and 1 word for $\le 3$ words), and T5 is trained to predict the masked sentences. The rationale for introducing BLEU is the highly consistent terminology in the medical domain, where n-gram exact matching is vital for clinical accuracy.

2. **Sequential Transfer Learning with Fisher Matrix Regularization**: The weights after GSG pre-training serve as the initial parameters for the summarization task. However, direct fine-tuning leads to catastrophic forgetting. CSTRL computes the Fisher information matrix $F_{ij} = \mathbb{E}[(\frac{\partial \log p(y|x;\theta)}{\partial \theta_i})(\frac{\partial \log p(y|x;\theta)}{\partial \theta_j})]$ to identify critical parameters. During fine-tuning, a regularization penalty $R(\theta) = \frac{1}{2}\sum_i F_{ii}(\theta_i - \theta_i^*)^2$ is introduced to restrict the shift of these key parameters, with the penalty weight dynamically adjusted during training.

3. **Contextual Tagging**: TF-IDF is used to extract key terms from the Impression, which are then mapped to corresponding Concept Unique Identifiers (CUIs) in the UMLS MRCONSO database to construct a tag set. T5 is trained to generate these tags from the Findings, thereby preserving the clinical semantic consistency of the generated summaries.

### Loss & Training

The knowledge distillation stage employs a combined loss function: $\mathcal{L} = (1-\alpha)\mathcal{L}_{CE} + \alpha \mathcal{L}_{KL}$, where the cross-entropy loss $\mathcal{L}_{CE}$ is computed on hard labels, and the KL divergence loss $\mathcal{L}_{KL} = T^2 \cdot \text{KLDiv}(\text{softmax}(S/T), \text{softmax}(T_t/T))$ is computed on soft labels from the teacher model. The temperature is set to $T=20$ and weight to $\alpha=0.7$. The teacher model consists of 6 layers, 512 dimensions, and 8 heads, while the student model consists of 3 layers, 128 dimensions, and 4 heads.

## Key Experimental Results

### Main Results

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 |
|------|-----|-----|-----|-----|-----|-----|
| ChestXRayBERT | 41.3 | 28.6 | 41.5 | 28.5 | 14.4 | 6.1 |
| Content Selector | 53.6 | 40.8 | 51.8 | – | – | – |
| Meta-Llama-3-8B | – | – | 29.0 | – | – | 9.4 |
| **CSTRL (Ours)** | **58.1** | **48.5** | **56.5** | **65.0** | **47.9** | **38.9** |

Compared to ChestXRayBERT, CSTRL improves BLEU-1/2/3 by **56.2%, 40.5%, and 84.3%** respectively, and ROUGE-1/2/L by **28.9%, 41.0%, and 26.5%**.

### Ablation Study

| Setting (GSG/Fisher/Layer Unfreezing) | R-1 | R-2 | R-L | B-1 | B-2 | B-3 |
|--------------------------------------|-----|-----|-----|-----|-----|-----|
| ✗ / ✗ / ✗ (Baseline) | 55.9 | 45.2 | 54.2 | 63.2 | 45.4 | 35.4 |
| ✓ / ✗ / ✗ (GSG only) | 55.9 | 45.2 | 54.2 | 63.2 | 45.4 | 35.4 |
| ✓ / ✓ / ✗ (**Full CSTRL**) | **58.2** | **48.5** | **56.5** | **65.0** | **47.9** | **38.9** |
| ✓ / ✗ / ✓ (Layer Unfreezing Alternative) | 53.4 | 43.1 | 51.9 | 61.5 | 42.3 | 32.3 |

### Knowledge Distillation Results

| Model | R-1 | R-2 | R-L | B-1 | B-2 | B-3 |
|------|-----|-----|-----|-----|-----|-----|
| CSTRL-Teacher | 58.1 | 48.5 | 56.5 | 65.0 | 47.9 | 38.9 |
| CSTRL-Student (×8) | 49.8 | 37.9 | 48.8 | 61.0 | 37.1 | 26.3 |
| CSTRL-Student (×16) | 47.8 | 36.3 | 46.7 | 58.5 | 35.9 | 25.2 |
| CSTRL-Student (×32) | 46.0 | 34.9 | 44.9 | 56.4 | 34.6 | 24.3 |

### Key Findings

- Fisher matrix regularization is the key driver of performance gains—without it, the knowledge learned during GSG pre-training is almost entirely overwritten during fine-tuning (catastrophic forgetting).
- The progressive layer unfreezing strategy actually degrades performance, demonstrating that parameter-level fine-grained control via the Fisher matrix is superior to layer-level coarse-grained control.
- Using only 40,000 samples (32.8% of the data) yields performance close to that of using the full dataset.
- The student model compressed by knowledge distillation ($\times 8$ compression) still retains respectable performance (R-1: 49.8 vs 58.1).

## Highlights & Insights

- **GSG using a ROUGE+BLEU composite score** is a simple yet effective improvement that exploits the domain-specific characteristic of terminology consistency in medical texts.
- **Fisher matrix regularization** addresses the core challenge of catastrophic forgetting in sequential transfer learning, significantly outperforming layer unfreezing strategies.
- **MRCONSO contextual tagging** elegantly leverages the UMLS knowledge base to guarantee the medical semantic accuracy of the generated texts.
- Using T5-small as the base model substantially outperforms Llama-3-8B, demonstrating that task-specific training strategies can be more crucial than model scale.

## Limitations & Future Work

- The evaluation is limited to chest X-ray reports (MIMIC-CXR and Open-I), and generalizability to other modalities of radiology reports (e.g., CT, MRI) has not been validated.
- The computational cost of compiling the Fisher matrix is high, raising scalability concerns for larger models.
- Performance degrades significantly after knowledge distillation (R-1 drops from 58.1 to 49.8 with $\times 8$ compression), presenting trade-offs for production deployment.
- UMLS contextual tagging relies on domain-specific knowledge bases, which requires extra effort when transferring to other medical subdomains.
- Comparisons with newer LLMs such as GPT-4 are currently lacking.

## Related Work & Insights

- The GSG method from **PEGASUS** (Zhang et al., 2020) serves as the foundation for this work, with CSTRL's combined ROUGE+BLEU scoring being a critical enhancement.
- The Fisher matrix approach of **Elastic Weight Consolidation** (Kirkpatrick et al., 2017) is successfully adapted to sequential transfer learning in NLP.
- **ChestXRayBERT** (Cai et al., 2023) serves as the most direct baseline, representing the prior SOTA in this domain.
- Insight: In low-resource medical NLP scenarios, a carefully designed transfer learning pipeline can outperform straightforward fine-tuning of large language models.

## Rating

- **Novelty**: ⭐⭐⭐ — The combination of optimized GSG and Fisher matrix regularization offers some degree of novelty, though the individual components leverage existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The ablation studies are comprehensive, encompassing knowledge distillation, low-resource scenarios, and factual consistency assessments.
- **Writing Quality**: ⭐⭐⭐ — Well-structured, although some formulas are presented in a somewhat verbose manner.
- **Value**: ⭐⭐⭐⭐ — Demonstrates significant performance gains in the practical domain of radiology report summarization, and the code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Automated Structured Radiology Report Generation](automated_structured_radiology_report_generation.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ACL 2026\] PCoA: A New Benchmark for Medical Aspect-Based Summarization With Phrase-Level Context Attribution](../../ACL2026/medical_nlp/pcoa_a_new_benchmark_for_medical_aspect-based_summarization_with_phrase-level_co.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](../../ACL2026/medical_nlp/rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)

</div>

<!-- RELATED:END -->
