---
title: >-
  [Paper Note] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations
description: >-
  [ACL 2025][LLM (Other)][knowledge boundaries] By probing the internal representations of LLMs, this study reveals that knowledge boundary cognition is linearly structured across multiple languages. A training-free alignment method is proposed to achieve cross-lingual transfer of knowledge boundary perception, and a "weak-to-strong generalization" phenomenon is discovered.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "knowledge boundaries"
  - "cross-lingual transfer"
  - "internal representation probing"
  - "hallucination mitigation"
  - "low-resource languages"
  - "subspace alignment"
date: 2026-05-08
content_hash: 7c6c886f653dd999
---

# Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations

**Conference**: ACL 2025  
**arXiv**: [2504.13816](https://arxiv.org/abs/2504.13816)  
**Code**: [https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries](https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries)  
**Area**: LLM/NLP  
**Keywords**: knowledge boundaries, cross-lingual transfer, internal representation probing, hallucination mitigation, low-resource languages, subspace alignment  

## TL;DR

By probing the internal representations of LLMs, this study reveals that knowledge boundary cognition is linearly structured across multiple languages. A training-free alignment method is proposed to achieve cross-lingual transfer of knowledge boundary perception, and a "weak-to-strong generalization" phenomenon is discovered.

## Background & Motivation

**Knowledge Boundaries and Hallucinations**: LLMs are prone to hallucinating when answering questions beyond their knowledge scope; understanding knowledge boundaries is a critical prerequisite for mitigating hallucinations.

**English-Centric Bias**: Existing research on knowledge boundaries focuses almost exclusively on English, lacking a systematic analysis of multilingual scenarios.

**Cross-Lingual Inconsistency**: The perception of knowledge boundaries across different languages may not be aligned, leading to inconsistent and unsafe outputs in multilingual applications.

**The Dilemma of Low-Resource Languages**: The boundary perception capability of low-resource languages is significantly weaker than that of high-resource languages; however, their representation spaces are more compact, harboring potential for transfer.

**Evidence at the Representation Level**: Prior studies show that a linearly separable true/false structure is encoded within the internal representations of LLMs, but this has not yet been extended to multilingual and cross-lingual settings.

**Lack of Evaluation Benchmarks**: Currently, there are no standard multilingual benchmarks for knowledge boundaries, hindering systematic research.

## Method

### Overall Architecture

This paper progressively analyzes the cross-lingual knowledge boundary cognition of LLMs from three levels: (1) **Probing Analysis**—layer-wise probing of knowledge boundary encoding patterns in multilingual representations; (2) **Training-free Alignment**—leveraging the discovered linear structure to achieve cross-lingual zero-shot/training-free transfer; (3) **Fine-tuning-based Enhancement**—further improving cross-lingual cognition through SFT with bilingual translation pairs. Concurrently, a multilingual knowledge boundary evaluation suite comprising three types of tasks is constructed.

### Module 1: Multilingual Knowledge Boundary Probing (§4)

For each language and layer, a linear classifier $f: \mathbb{R}^d \to \mathcal{C}$ is trained using the representation of the last token of the question $\mathbf{E} \in \mathbb{R}^{n \times d}$ as input. A total of $k \times m$ classifiers are trained ($k$ = number of layers, $m$ = number of languages), and each in-distribution probe is evaluated in a zero-shot manner on all other languages.

**Key Findings**:
- Knowledge boundary cognition is encoded in the middle to upper-middle layers (e.g., layer 19 is optimal for Qwen2.5-7B).
- A significant ID/OOD performance gap exists at the lower layers (due to language-specific static embeddings), while middle layers converge into a unified knowledge space.
- Low-resource languages (e.g., Khmer) exhibit the best relative transferability—their discriminative features are present in the representations of high-resource languages, but not vice versa.

### Module 2: Training-free Subspace Alignment (§5)

**LDA Geometric Analysis**: LDA classifiers are trained using three sets of labels (language, domain $\times$ truth/falsity, and binary truth/falsity). After projection, it is discovered that: (i) languages are encoded in a parallel structure; (ii) truth/falsity is encoded in a language-neutral manner; (iii) true/false is separable by a near-horizontal hyperplane.

**Mean Shifting**: Computes the difference in means between the source and target language training sets:

$$\Delta\boldsymbol{\mu} = \boldsymbol{\mu}_{\text{in}} - \boldsymbol{\mu}_{\text{ood}}, \quad \mathbf{X}_{\text{shifted}}^{\text{test}} = \mathbf{X}_{\text{ood}}^{\text{test}} + \Delta\boldsymbol{\mu}$$

**Linear Projection**: Solves the least-squares problem $\mathbf{W} = \arg\min_{\mathbf{W}} \|\mathbf{X}_{\text{in}} - \mathbf{X}_{\text{ood}}^{\text{train}}\mathbf{W}\|_F^2$. The SVD pseudoinverse yields $\mathbf{W} = \mathbf{X}_{\text{ood}}^{\text{train}+}\mathbf{X}_{\text{in}}$, projecting target language representations onto the source language subspace: $\mathbf{X}_{\text{shifted}}^{\text{test}} = \mathbf{X}_{\text{ood}}^{\text{test}}\mathbf{W}$.

**Weak-to-Strong Generalization**: Probes trained on a low-resource language (Khmer) yield better performance on other languages after post-projection than on Khmer itself. The underlying reason is that projection into the low-resource subspace plays a denoising/regularization role—when English is projected into the Khmer subspace, its effective dimension decreases from 116 to 87, and the participation ratio drops from 26.26 to 19.29.

### Module 3: Fine-tuning-based Enhancement (§6)

Fine-tuning the LLM with SFT data containing only **question translation pairs** (without answers), such as Khmer $\to$ English pairs, consistently improves the knowledge boundary probing performance across all languages. The average accuracy at the optimal layer of Qwen2.5-7B reaches 88% (+2.3%).

**Self-Defense Mechanism**: When fine-tuning Qwen2.5 on low-resource $\to$ English translation pairs, the model's Chinese representations unexpectedly show significant improvement (Chinese is one of Qwen's dominant languages). The hypothesis is that addressing unanswerable questions in low-resource languages activates latent safety mechanisms associated with dominant languages.

### Evaluation Dataset

| Dataset | No. Languages | Type | Size |
|--------|--------|------|------|
| FreshQAParallel | 8 (en/zh/vi/th/id/ms/km/lo) | True/false premise question pairs | 9,600 test |
| SeaRefuse | 5 (en/zh/id/th/vi) | Entity answerable/unanswerable questions | 64k train + 6k test |
| TrueFalseMultiLang | 8 (en/es/de/it/pt/fr/id/th) | True/false statements | 48,680 test |

## Experiments

### Table 1: Impact of False-Premise Hints on Generation Performance

| Setting | en | zh | vi | th | km | id | ms | lo |
|------|-----|-----|-----|-----|-----|-----|-----|-----|
| **Qwen2.5-7B-Inst Baseline** | 30.61 | 36.05 | 19.73 | 19.73 | 8.16 | 22.45 | 19.05 | 0.68 |
| **+ FP-Hinted** | 41.50 | 45.58 | 44.22 | 32.65 | 11.56 | 38.10 | 37.41 | 2.04 |
| **Qwen2.5-72B-Inst Baseline** | 58.50 | 60.54 | 61.90 | 55.10 | 33.33 | 59.18 | 55.78 | 31.29 |
| **+ FP-Hinted** | 72.11 | 70.75 | 68.03 | 67.35 | 44.90 | 72.79 | 73.47 | 38.10 |

→ Prompting false premises results in the largest improvement in Vietnamese (+24.49%), indicating that the model has encoded the knowledge boundaries internally but has not fully utilized them during generation.

### Table 2: Subspace Dimensional Analysis (English vs. Khmer Projection)

| Metric | Original (km) | Projected (km) | Original (en) | Projected (en) |
|------|----------|-----------|----------|-----------|
| Effective Dim | 103 | 97 | 116 | 87 |
| Participation Ratio | 15.93 | 18.07 | 26.26 | 19.29 |

→ Projecting English into the Khmer subspace significantly reduces the dimension and makes the PR more compact, confirming the denoising effect of the low-resource subspace.

### Key Findings

1. **Linear Projection almost eliminates the ID-OOD gap**: On all models (7B-72B), OOD performance after linear projection is close to ID performance, far outperforming Mean Shifting.
2. **Cross-lingual generalization of SFT**: Fine-tuning solely on Khmer-English translation pairs is sufficient to improve the knowledge boundary probing accuracy across all 8 languages.
3. **Directionality of transfer**: The high $\to$ medium and medium $\to$ low transfer chains perform best (e.g., on Qwen2.5-14B, the best performance for Thai comes from the Chinese probe at 88.31%, and the best for Khmer comes from the Malay probe at 88.15%).
4. **Non-parallel corpus approximation**: Mean Shifting can even calculate language means using non-parallel corpora, achieving performance comparable to using parallel corpora.

## Highlights & Insights

- **First systematic study on multilingual knowledge boundaries**: Fills a crucial gap in this field from English to multilingual scenarios.
- **Discovery and exploitation of linear structures**: Knowledge boundaries exhibit a linearly separable geometric structure across languages, enabling training-free transfer.
- **Weak-to-strong generalization phenomenon**: Low-resource language subspaces act as inductive biases to filter noise, offering a novel theoretical explanation with practical value.
- **Self-defense mechanism**: Fine-tuning on non-dominant language pairs unexpectedly enhances the safety representations of dominant languages, revealing the interconnectedness of multilingual safety mechanisms within LLMs.
- **Comprehensive evaluation suite**: Three complementary types of datasets cover different definitions of knowledge boundaries, constructed with rigorous quality control (human annotation + verification by linguists).

## Limitations & Future Work

1. This work only analyzes knowledge boundary perception at the representation level, leaving how representations evolve during generation (particularly in CoT reasoning) unexplored.
2. The expressiveness of linear probes is limited; non-linear probes might capture more complex boundary patterns.
3. The experimental languages are predominantly Southeast Asian; generalization to languages with vast morphological differences (e.g., Arabic, Finnish) remains unverified.
4. Linear projection requires parallel corpora; although the quantity requirement is low (a few hundred pairs), it may still pose a bottleneck for extremely low-resource languages.

## Related Work & Insights

- **Knowledge Boundaries**: Azaria & Mitchell (2023) propose that LLMs' internal states "know when they lie"; Marks & Tegmark (2024) discover the linear geometry of true/false representations; Bürger et al. (2024) briefly demonstrate English-to-German transfer.
- **Multilingual LLMs**: Zhao et al. (2024) propose a three-stage hypothesis (source language $\to$ anchor language thinking $\to$ source language generation); Tang et al. (2024) identify language-specific neurons; Zhang et al. (2024b) find that fine-tuning on question translations improves multilingual performance.
- **Cross-Lingual Alignment**: Chang et al. (2022) discover that language differences in multilingual encoders are primarily encoded in subspace means; Xu et al. (2023) explore cross-lingual projection of knowledge.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic study on multilingual knowledge boundaries, with novel findings of weak-to-strong generalization and self-defense mechanisms.
- **Technical Depth**: ⭐⭐⭐⭐ — A progressive three-layer approach from probing to training-free alignment to SFT, with rigorous mathematical derivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple model families and scales (7B–72B), 8 languages, three dataset types, and extensive ablations.
- **Value**: ⭐⭐⭐⭐ — The training-free method can be directly deployed as a hallucination detection signal; datasets and code are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [\[ACL 2025\] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)
- [\[ACL 2025\] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction](genknowsub_improving_modularity_and_reusability_of_llms_through_general_knowledg.md)
- [\[ACL 2025\] Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes](improving_preference_extraction_in_llms_by_identifying_latent_knowledge_through_.md)
- [\[ACL 2025\] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching](self-tuning_instructing_llms_to_effectively_acquire_new_knowledge_through_self-t.md)

</div>

<!-- RELATED:END -->
