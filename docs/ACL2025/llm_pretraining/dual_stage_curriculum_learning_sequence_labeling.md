---
title: >-
  [Paper Note] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling
description: >-
  [ACL 2025][LLM Pretraining][Curriculum Learning] This paper proposes a Dual-stage Curriculum Learning (DCL) framework for sequence labeling. By employing a data-level and model-level two-stage training strategy from easy to difficult, combined with a Bayesian uncertainty-based token-level dynamic difficulty metric and a Root function training scheduler, the framework achieves the dual benefits of performance improvements and over 27% training acceleration across three tasks:…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Curriculum Learning"
  - "Sequence Labeling"
  - "Bayesian Uncertainty"
  - "Chinese Word Segmentation"
  - "Named Entity Recognition"
date: 2026-05-08
content_hash: 1ad1c75456b48593
---

# An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling

**Conference**: ACL 2025  
**arXiv**: [2402.13534](https://arxiv.org/abs/2402.13534)  
**Code**: [GitHub](https://github.com/tangxuemei1995/DCL4SeqLabeling)  
**Area**: Sequence Labeling / Curriculum Learning  
**Keywords**: Curriculum Learning, Sequence Labeling, Bayesian Uncertainty, Chinese Word Segmentation, Named Entity Recognition

## TL;DR

This paper proposes a Dual-stage Curriculum Learning (DCL) framework for sequence labeling. By employing a data-level and model-level two-stage training strategy from easy to difficult, combined with a Bayesian uncertainty-based token-level dynamic difficulty metric and a Root function training scheduler, the framework achieves the dual benefits of performance improvements and over 27% training acceleration across three tasks: CWS, POS, and NER.

## Background & Motivation

**Core Problem**: With the introduction of external knowledge (lexicons, n-grams, syntax) into sequence labeling models, data heterogeneity increases and model complexity skyrockets, making training highly expensive. How can complex models be trained faster while maintaining or even improving performance?

**Double-Edged Sword of External Knowledge**: In recent years, performance improvements in sequence labeling have heavily relied on integrating external knowledge such as lexicons, n-grams, and syntactic trees. However, this introduces input data heterogeneity and parameter expansion in auxiliary encoding modules (attention mechanisms, GNNs), dramatically increasing the computational cost of training high-performance models.

**Potential and Limitations of Curriculum Learning**: Curriculum Learning (CL), by simulating the "easy-to-difficult" learning process of humans, has succeeded in tasks such as machine translation, dialogue generation, and text classification. However, existing CL difficulty metrics primarily focus on the sentence level (e.g., sentence length, sentence-level confidence), lacking token-level and word-level difficulty metrics tailored for sequence labeling. Consequently, they cannot precisely capture the signal of "which tokens are the most difficult to label" in sequence labeling tasks.

**Limitations of Static Curriculums**: Traditional CL determines the difficulty ranking of samples before training and keeps it fixed. However, model capacity continuously evolves during training—samples initially deemed "difficult" may become "easy" after the model acquires certain capabilities. Static ranking fails to adapt to the learning dynamic of models.

**Cold Start Dilemma**: Directly applying curriculum learning to a student model suffers from a cold start problem—models with weak initial capability cannot reliably evaluate sample difficulty, leading to inaccurate curriculum scheduling.

## Method

### Overall Architecture

The DCL framework consists of three core components: a teacher sequence labeling model (RoBERTa + Softmax), a student sequence labeling model (complex models such as McASP or SynSemGCN), and a dual-stage curriculum learning training strategy. DCL is independent of specific sequence labeling model architectures, allowing plug-and-play integration with any encoder-decoder combination.

| Stage | Goal | Input | Output |
|------|------|------|------|
| Data-level CL | Initialize sample difficulty ranking, alleviate cold start | All training data $\mathcal{D}$ | $\mathcal{D}_r$ sorted in ascending order of difficulty |
| Model-level CL | Dynamically expand training set, adapt to learning trajectory | Sorted subset $\mathcal{D}_s$ + remaining $\mathcal{D}_o$ | Student model $\theta$ trained to convergence |

**Data-level Curriculum Learning (Stage 1)**: A basic teacher model $\theta_0$ is trained on all training data $\mathcal{D}$ for $E_0$ epochs (fewer than needed for convergence). Then, $\theta_0$ is used to compute difficulty scores $S(\theta_0)$ for all samples, sorting them in ascending order of difficulty to form $\mathcal{D}_r$. The role of the teacher model is to provide a reasonable initial difficulty ranking, preventing the student model from facing randomly ordered data during cold start.

**Model-level Curriculum Learning (Stage 2)**: The student training set $\mathcal{D}_s$ is initialized with the simplest $\lambda_0$ proportion of samples in $\mathcal{D}_r$. After each epoch, the current student model $\theta_*$ is used to re-evaluate the difficulty of the remaining data $\mathcal{D}_o$ and re-rank them. The training set is expanded by increasing $\lambda$ according to the Root function scheduler. When $\lambda = 1$, the model is trained to convergence using the full dataset.

### Key Designs

**1. Bayesian Uncertainty (BU) Token-level Difficulty Metric**

Based on Monte Carlo dropout to approximate Bayesian inference, the difficulty metric is refined from the sentence level to the token level. For each token $x_i$, $K$ random forward passes are executed (randomly deactivating different dropout neurons each time) to obtain $K$ predicted distributions $P(y_i|x_i)_1, \ldots, P(y_i|x_i)_K$. The probability variance of each token over the tag set $T$ is computed as:

$$var(x_i, \theta) = \sum_{y_i \in T} \left( \frac{1}{K} \sum_{k=1}^K P(y_i|x_i)_k^2 - \mathbb{E}[P(y_i|x_i)]^2 \right)$$

The final sentence-level difficulty score considers both the most uncertain position in the sequence and the overall level of uncertainty: $S(\theta)^{BU} = var(\theta)_{max} + var(\theta)_{aver.}$, where $var(\theta)_{max}$ captures the most difficult-to-label token (e.g., rare entity boundaries) and $var(\theta)_{aver.}$ reflects the average difficulty of labeling the entire sentence. Ablation studies show that both make comparable contributions to performance, and neither can be omitted.

**2. Root Function Training Scheduler**

Controls the growth pace of the proportion of new samples $\lambda$ added at each epoch:

$$\lambda = \min\left(1, \sqrt{\frac{1-\lambda_0^2}{E_{grow}} \cdot t + \lambda_0^2}\right)$$

where $\lambda_0$ is the initial training set proportion, $E_{grow}$ is the number of epochs required for $\lambda$ to grow to 1, and $t$ is the current epoch. The characteristic of the Root function is that it grows faster in the early stages (quickly exposing the model to medium-difficulty samples) and slows down in the later stages (giving the model sufficient time to digest the most difficult samples), which better aligns with the law of learning from fast to slow compared to linear schedulers.

**3. Model-level Dynamic Difficulty Re-evaluation Mechanism**

The key difference from static curriculum learning is that DCL re-computes the difficulty and re-ranks the remaining data $\mathcal{D}_o$ using the current student model $\theta_*$ at each epoch during the model-level stage. As the student model's capability gradually strengthens, the relative difficulty of samples changes—samples previously judged as "difficult" may become "easy" as the model masters relevant patterns. Dynamic re-evaluation enables newly added samples in each round to precisely match the model's current learning frontier, achieving true adaptive curriculum learning.

## Key Experimental Results

### Main Results: Joint CWS and POS Tagging

On three Chinese datasets CTB5, CTB6, and PKU, taking SynSemGCN as the student model, the paper compares the baseline without CL against DCL settings with different difficulty metrics:

| Model / CL Setting | CTB5 CWS | CTB5 POS | CTB6 CWS | CTB6 POS | PKU CWS | PKU POS |
|----------------|----------|----------|----------|----------|---------|---------|
| SynSemGCN (W/o CL) | 98.83 | 96.77 | 97.86 | 94.98 | 98.05 | 95.50 |
| + DCL (Random) | 98.84 | 97.86 | 97.99 | 95.05 | 98.48 | 96.40 |
| + DCL (Length) | 98.80 | 96.84 | 97.40 | 94.94 | 98.53 | 96.48 |
| + DCL (TLC) | 98.83 | 97.81 | 97.98 | 95.02 | 98.61 | 96.55 |
| + DCL (MNLP) | 98.78 | 97.72 | 98.04 | 95.13 | 98.56 | 96.48 |
| **+ DCL (BU)** | **98.90** | **97.95** | **98.05** | **95.14** | **98.59** | **96.54** |

### Ablation Study and Training Efficiency

| Setting | CTB5 CWS | CTB5 POS | Training Time |
|------|----------|----------|----------|
| SynSemGCN + DCL (BU) Full | 98.90 | 97.95 | 287 min |
| W/o Data-level CL | 98.90 | 97.88 | — |
| W/o Model-level CL | 98.85 | 97.51 | — |
| W/o DCL (Original SynSemGCN) | 98.75 | 96.73 | 393 min |

Ablation of BU difficulty metric components (McASP model, CTB5):

| Setting | CWS F1 | POS F1 |
|------|--------|--------|
| McASP + DCL (BU) Full | 98.91 | 96.87 |
| W/o $var(\theta)_{max}$ | 98.78 | 96.78 |
| W/o $var(\theta)_{aver.}$ | 98.86 | 96.74 |
| McASP (W/o CL) | 98.73 | 96.60 |

### NER Generalization Experiments

| Model | Weibo (Chinese) | OntoNotes4 (Chinese) | CoNLL-2003 (English) |
|------|-------------|-------------------|-------------------|
| BERT (W/o CL) | 66.22 | 79.15 | 90.94 |
| BERT + CL (Length) | 66.81 | 79.63 | 90.79 |
| BERT + DCL (BU) | **66.74** | **80.02** | **91.77** |

### Key Findings

1. **BU metric is consistently optimal**: On the vast majority of datasets, BU outperforms TLC, MNLP, sentence length, and random sorting, indicating that uncertainty-based metrics capture the true difficulty of sequence labeling better than heuristic metrics.
2. **Training acceleration of 27%**: DCL reduces the training time of SynSemGCN from 393 minutes to 287 minutes while actually delivering superior performance.
3. **Model-level CL contributes more**: Removing model-level CL leads to a 0.44 drop in POS F1, whereas removing data-level CL only results in a 0.07 drop. This is because model-level CL affects the entire training process while data-level CL only affects the initial ranking.
4. **Dual BU components are complementary**: Removing $var(\theta)_{max}$ reduces CWS F1 from 98.91 to 98.78, and removing $var(\theta)_{aver.}$ reduces it to 98.86, indicating comparable contributions from both.
5. **Cross-task generalization**: DCL is effective across six datasets in CWS, POS, and NER (both Chinese and English) tasks, validating the generalizability of the framework.

## Highlights & Insights

1. **Clear division of labor in the two stages**: Data-level CL uses a lightweight teacher model to provide a coarse-grained initial ranking to resolve the cold start, while model-level CL uses the student model itself to perform fine-grained dynamic re-ranking to realize adaptation. The two stages are complementary rather than redundant.
2. **Token-level metric fills the gap**: Previous CL difficulty metrics remained at the sentence level, whereas the core challenge of sequence labeling lies in a minority of hard-to-label tokens. The BU metric precisely captures this characteristic through the combination of max and average.
3. **Model-agnostic plug-and-play design**: DCL does not modify the model architecture; it merely alters the presentation order of the training data. It can be paired with any sequence labeling model.
4. **Win-win in performance and efficiency**: Training on small subsets in the early stage of curriculum learning reduces computation, while progressive learning from easy to difficult brings better convergence quality.

## Limitations & Future Work

- The difficulty metric involves multiple hyperparameters (number of Monte Carlo dropout passes $K$, initial ratio $\lambda_0$, growth epochs $E_{grow}$), leading to high tuning costs.
- The work only explores the easy-to-difficult curriculum strategy and does not investigate the comparative efficacy of an anti-curriculum (difficult-to-easy) strategy.
- The training scheduler only compares the Root function with linear functions, without exploring other scheduling policies such as stair-step or cosine schedules.
- The experimental dataset scale is relatively small (CTB5 has around 19K sentences), so the scalability in massive sequence labeling scenarios has yet to be verified.
- The NER generalization experiments only utilize BERT as the student model and have not verified the effectiveness of DCL on complex NER models.

## Related Work & Insights

- **Bengio et al. (2009)**: Pioneering work in curriculum learning, proposing the easy-to-difficult training paradigm.
- **Tian et al. (2020b) McASP**: A sequence labeling model that fuses lexicons and n-grams based on multi-attention mechanisms, serving as one of the baselines in this paper.
- **Tang et al. (2024) SynSemGCN**: A sequence labeling model that fuses syntactic and semantic knowledge via GCN, serving as another baseline in this paper.
- **Gal & Ghahramani (2016)**: Monte Carlo dropout method, which forms the theoretical foundation of the BU metric in this work.
- **Wang et al. (2021)**: Survey of curriculum learning in NLP, organizing the classification systems of data selection strategies and training schedulers.

## Rating

- **Novelty**: ⭐⭐⭐ — The dual-stage CL framework and the BU token-level metric possess some novelty, but overall representing a combination and refinement of existing techniques.
- **Practicality**: ⭐⭐⭐⭐ — The dual benefits of performance gains and a 27% training acceleration hold substantial engineering value, and the model-agnostic design makes it easy to deploy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple datasets, multiple models, comprehensive ablation experiments, and cross-task generalization are analyzed, although the dataset scales are somewhat small.
- **Writing Quality**: ⭐⭐⭐ — Clear in structure but with somewhat cumbersome notation, and some equation formatting could be more concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [\[ACL 2025\] Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)

</div>

<!-- RELATED:END -->
