---
title: >-
  [Paper Note] Do Sparse Autoencoders Generalize? A Case Study of Answerability
description: >-
  [ICML 2025][Interpretability][Sparse Autoencoder] This paper systematically evaluates the out-of-domain (OOD) generalization capabilities of features extracted by Sparse Autoencoders (SAEs) on the task of "answerability." The study reveals highly inconsistent OOD transfer performance of SAE features—outperforming residual stream linear probes on some datasets while performing near-randomly on others, highlighting the fundamental limitations of current SAE interpretability met…
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Sparse Autoencoder"
  - "Feature Generalization"
  - "Answerability Detection"
  - "Linear Probe"
date: 2026-05-08
content_hash: f5bfca7eda512651
---

# Do Sparse Autoencoders Generalize? A Case Study of Answerability

**Conference**: ICML 2025  
**arXiv**: [2502.19964](https://arxiv.org/abs/2502.19964)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Sparse Autoencoder, Interpretability, Feature Generalization, Answerability Detection, Linear Probe

## TL;DR

This paper systematically evaluates the out-of-domain (OOD) generalization capabilities of features extracted by Sparse Autoencoders (SAEs) on the task of "answerability." The study reveals highly inconsistent OOD transfer performance of SAE features—outperforming residual stream linear probes on some datasets while performing near-randomly on others, highlighting the fundamental limitations of current SAE interpretability methods in capturing abstract concepts.

## Background & Motivation

### Problem Origin

The black-box nature of language models remains a fundamental obstacle to their safe deployment. Sparse Autoencoders (SAEs), as an unsupervised interpretability method, learn disentangled and interpretable features by reconstructing neural activations through a sparse bottleneck. They have shown promise in downstream tasks such as bug detection, bias identification, and sentiment analysis. However, a core question has been overlooked: **Can SAEs truly capture abstract concepts that generalize across domains?**

### Why Answerability?

- **High-Level Semantic Concept**: Answerability (the model's judgment of "whether I can answer this question") is an abstract capability universally present across different tasks and domains.
- **High Heterogeneity**: The unanswerability of a math problem versus that of a reading comprehension question may be represented completely differently, making it highly suitable for testing generalization.
- **Practical Significance**: Answerability detection is directly related to hallucination control and selective answering capabilities.

### Limitations of Prior Work

| Direction | Limitations of Prior Work | Ours |
|------|-------------|---------|
| SAE Training & Optimization | Evaluated only by reconstruction quality, disconnected from downstream tasks | Directly evaluate downstream classification generalization performance |
| SAE Downstream Evaluation | Focus on simple syntactic features, without testing generalization | OOD evaluation across 5 heterogeneous datasets |
| Multilingual/Syntactic Generalization | Only consider linguistic variants or syntactic transformations | Consider the concept of answerability across different semantic domains |
| Bioweapon Classification | Mainly lexical-level tasks, with limited generalization scenarios | More complex high-level concepts, more diverse distribution shifts |

## Method

### Overall Architecture

The evaluation methodology of this work can be decomposed into three core steps:

**Step 1: SAE Feature Discovery**  
Use the Gemma Scope pretrained SAE (Lieberum et al., 2024) to decompose the activations of the Gemma 2 instruction-tuned model. The largest width of 131k SAE is selected, trained at Layer 20 and Layer 31.

**Step 2: Probe Training**  
On the in-domain dataset SQUAD, two types of probes are trained:
- **SAE Probe** (1-sparse): Select a single most predictive SAE feature, plus scale + bias.
- **Residual Stream Linear Probe**: Train a linear classifier directly on residual stream activations, acting as an upper-bound baseline.

**Step 3: OOD Generalization Evaluation**  
Evaluate the in-domain trained probes on 4 OOD datasets to measure differences in generalization ability.

### Key Designs

#### SAE Feature Selection Pipeline

1. Sample 2000 balanced instances from SQUAD (1000 answerable, 1000 unanswerable).
2. Collect the SAE feature activations (131k dimensions) at the last token position.
3. Use 5-fold cross-validation to evaluate the predictive capacity of each SAE dimension for answerability feature-by-feature.
4. Select the top-K best-performing features.
5. Train the final probe (learning scale and bias parameters) for each top feature, forming a **1-sparse SAE probe**.

The core philosophy of this approach is: if the SAE has truly learned the abstract representation of "answerability", there should exist one (or a few) SAE feature dimensions that encode this concept across domains.

#### Mathematical Formulation

$$\mathbf{f} = \text{ReLU}(W_e(\mathbf{x} - \mathbf{b}_d) + \mathbf{b}_e)$$

$$\hat{\mathbf{x}} = W_d \mathbf{f} + \mathbf{b}_d$$

Where $W_e \in \mathbb{R}^{d_{sae} \times d_{model}}$, $d_{sae} \gg d_{model}$, achieving sparse coding through overcomplete representation.

Training loss:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \lambda \|\mathbf{f}\|_1$$

The first term is the reconstruction error, and the second term is L1 regularization ensuring sparsity. The Gemma Scope SAE used in this work employs an updated sparsification mechanism (Lieberum et al., 2024; Gao et al., 2024), but the basic philosophy remains consistent.

#### Residual Stream Linear Probe (Baseline)

Directly train a linear classifier $y = \sigma(\mathbf{w}^T \mathbf{x} + b)$ on the model's residual stream activations $\mathbf{x}$, and use bootstrap analysis to ensure robustness. The in-domain accuracy reaches 85-90%, serving as a strong baseline for the SAE probes.

#### Evaluation Dataset Design

An evaluation matrix covering various types of distribution shifts was constructed:

| Dataset | Size | Type | Source of Distribution Shift |
|--------|------|------|-------------|
| SQUAD (test) | 1800 | In-domain test | None (In-distribution) |
| SQUAD (variations) | 1800 | Near-domain | Prompt template variations |
| IDK | 484 | OOD | Different question styles |
| BoolQ | 2000 | OOD | Yes/No questions vs. open-ended questions |
| Equation | 2000 | OOD (Synthetic) | Mathematical equations, completely different semantics |
| Celebrity | 600 | OOD (Synthetic) | Celebrity facts, open-world knowledge |

Among these, **Equation** and **Celebrity** are synthetic datasets constructed in this work:
- Equation: Given simple equations (e.g., n=53, v=90), ask whether an expression can be computed using the defined variables, independent of any textual understanding.
- Celebrity: Given short paragraphs about real or fictional celebrities, ask factual questions such as their age, testing the boundaries of the model's world knowledge.

### Loss & Training

- **SAE Pre-training**: Pre-trained weights provided by Gemma Scope are used; this work does not retrain the SAEs.
- **Probe Training**: Trained on 2000 in-domain SQUAD samples using cross-entropy loss.
- **Feature Selection**: Ranked based on 5-fold cross-validation AUC/accuracy.
- **Evaluation**: All OOD tests use the in-domain trained probes without any fine-tuning, strictly evaluating generalization performance.

## Key Experimental Results

### Main Results

| Evaluation Dataset | SAE Top-1 Feature | SAE Top-5 Features | Residual Stream Linear Probe | Observations |
|-----------|---------------|---------------|--------------|------|
| SQUAD (In-domain) | ~75% | ~80% | **85-90%** | Linear probe leads by a large margin |
| Equation (Synthetic) | **High** | **High** | Medium | SAE generalizes well |
| IDK | **High** | **High** | Medium | SAE generalizes well |
| Celebrity | **High** | **High** | Medium | SAE generalizes well |
| BoolQ | ~Random | ~Random | Medium-low | SAE fails almost completely |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Layer 20 vs Layer 31 | Layer 31 slightly better | High-level concepts are more clearly represented in deeper layers |
| 131k-width SAE | Only available maximum width | Larger dictionary may offer better disentanglement |
| Last token position | Standard practice | The last token aggregates information from the entire sequence |
| Bootstrap Analysis | Residual stream probe has high variance | Linear probe OOD performance is also unstable |
| Feature Ranking Top-1/5/10 | High rank fluctuation | The optimal in-domain feature is not necessarily optimal for OOD |

### Key Findings

1. **In-Domain vs OOD Reversal**: Although the residual stream linear probe consistently outperforms SAE features in-domain, SAE features perform better on certain OOD datasets. This suggests that the sparse disentanglement of SAEs may provide a regularization effect in specific contexts.

2. **Extreme Inconsistency in SAE Generalization**: The same set of top SAE features generalizes well to Equation, IDK, and Celebrity, but drops to near-random performance on BoolQ. This indicates that "answerability" in different datasets may be encoded as completely distinct features by the model.

3. **Answerability is Not a Single Concept**: The computational pathways within the model for answerability in BoolQ (yes/no questions) versus SQUAD (extractive QA) may be entirely different, rendering single-feature SAEs unable to cover both uniformly.

4. **Inelastic Reliability of Linear Probes**: Even though the residual stream contains richer information, the OOD performance of linear probes exhibits high variance, demonstrating that generalization issues are not unique to SAEs.

## Highlights & Insights

- **Elegant Experimental Design**: By constructing synthetic datasets (Equation, Celebrity), the evaluation of answerability is pushed to completely different semantic domains, exposing the true generalization boundaries of SAEs.
- **Counter-Intuitive Finding**: Stronger in-domain performance does not equate to stronger OOD performance; indeed, a "loose" single-feature SAE probe can sometimes generalize better than a "tight" full-dimensional linear probe in specific settings.
- **Concept Granularity Issue**: The paper poses an important open question—"answerability" within models may be a combination of multiple fine-grained mechanisms rather than a single, easily disentangled feature.
- **Warning for AI Safety**: If relatively straightforward concepts like answerability fail to generalize stably, the reliability of using SAEs to detect more complex behaviors, such as deception or bias, remains highly questionable.

## Limitations & Future Work

1. **Evaluation Restricted to Gemma 2**: It remains unknown whether these conclusions generalize to other architectures like LLaMA or GPT.
2. **Fixed SAE Width**: Only the 131k width variant was evaluated; how dictionary scale affects generalization is left unexplored.
3. **1-sparse Probe Constraint**: Only single-feature probes were constructed; it is unclear whether sparse multi-feature combinations can improve generalization.
4. **Lack of Feature Visualizations**: There is no in-depth analysis of why specific features generalize well on some domains but fail entirely on others.
5. **Promising Future Directions**:
    - Leverage few-shot OOD samples for feature selection.
    - Train multi-feature combination probes ($k$-sparse, $k>1$).
    - Conduct cross-model comparative evaluations on SAE generalization.
    - Investigate the specific mechanistic reasons behind the failure on BoolQ.

## Related Work & Insights

- **Cunningham et al. (2023)** and **Bricken et al. (2023)**: Foundational works on SAE interpretability, upon which this paper questions their generalization capacity.
- **Bricken et al. (2024)**: Compare SAEs and linear probes on bioweapon classification, finding that format mismatches can cause performance degradation; this paper extends this study to more complex tasks.
- **Kantamneni et al. (2024; 2025)**: SAEs may show advantages in small data and corrupted data regimes, but generally do not outperform standard probes.
- **Barez et al. (2025)**: Even advanced interpretability methods may have fundamental limitations in guaranteeing AI safety.
- **Inspiration for Idea Generation**: The inconsistency of SAE feature generalization suggests opportunities to design **adaptive sparse coding** methods—either by introducing cross-domain validation signals during feature selection or by training **hierarchical SAEs** to identify domain-general and domain-specific features separately.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first systematic evaluation of SAE cross-domain generalization on complex high-level concepts with a novel experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covered various distribution shifts across 6 datasets, though restricted to a single model and a single SAE width.
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear, analysis is well-structured, and visualizations are intuitive.
- Value: ⭐⭐⭐⭐ — Serves as an important warning to the SAE interpretability community, driving more rigorous evaluation paradigms for generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ACL 2026\] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](../../ACL2026/interpretability/understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](../../ACL2026/interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)
- [\[ICML 2026\] Ensembling Sparse Autoencoders](../../ICML2026/interpretability/ensembling_sparse_autoencoders.md)
- [\[ACL 2026\] Interpretable Semantic Gradients in SSD: A PCA Sweep Approach and a Case Study on AI Discourse](../../ACL2026/interpretability/interpretable_semantic_gradients_in_ssd_a_pca_sweep_approach_and_a_case_study_on.md)

</div>

<!-- RELATED:END -->
