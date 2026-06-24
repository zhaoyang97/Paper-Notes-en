---
title: >-
  [Paper Note] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence
description: >-
  [ECCV 2024][LLM Evaluation][Source-Free Domain Adaptation] The LFTL (Learn from the Learnt) framework is proposed. Consisting of two core modules—Contrastive Active Sampling (CAS) and Visual Persistence-guided Adaptation (VPA)—it achieves highly efficient domain adaptation under source-free and extremely low target annotation budgets ($\le 5\%$), reaching 87.4% accuracy on VisDA-C with only 1% annotation.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Source-Free Domain Adaptation"
  - "Active Learning"
  - "Contrastive Sampling"
  - "Visual Persistence"
  - "Domain Adaptation"
date: 2026-05-08
content_hash: af6408bca85fcd6e
---

# Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence

**Conference**: ECCV 2024  
**arXiv**: [2407.18899](https://arxiv.org/abs/2407.18899)  
**Code**: Yes ([https://github.com/lyumengyao/lftl](https://github.com/lyumengyao/lftl))  
**Area**: LLM Evaluation  
**Keywords**: Source-Free Domain Adaptation, Active Learning, Contrastive Sampling, Visual Persistence, Domain Adaptation

## TL;DR

The LFTL (Learn from the Learnt) framework is proposed. Consisting of two core modules—Contrastive Active Sampling (CAS) and Visual Persistence-guided Adaptation (VPA)—it achieves highly efficient domain adaptation under source-free and extremely low target annotation budgets ($\le 5\%$), reaching 87.4% accuracy on VisDA-C with only 1% annotation.

## Background & Motivation

### Practical Dilemmas of Domain Adaptation

Deep neural networks perform exceptionally well when the distributions of training and testing data are consistent, but their performance drops sharply under domain shift. Domain Adaptation (DA) aims to transfer source domain knowledge to the target domain. However, existing DA methods face two severe practical constraints:

**Inaccessibility of Source Data**: Due to strict data protection regulations and storage/computation resource limits, access to source domain data is often unavailable during adaptation. Although Source-Free Unsupervised Domain Adaptation (SFUDA) methods do not require source data, the lack of deterministic supervisory signals from any domain exacerbates the **ill-posedness of adaptation**, leading to a performance ceiling.

**Low Annotation Budget is Feasible but Underutilized**: Since target domain samples are fully accessible during the adaptation phase, allocating a minimal annotation budget can yield significant performance gains. Therefore, Source-Free Active DA (SFADA) presents a more practical setting—relying on no source data while actively querying target domain labels under a minimal annotation budget.

### Three New Challenges of SFADA

Combining "source-free data" with "active learning" introduces unique challenges:

**Difficulty in Query Selection**: Standard active learning criteria (such as entropy, margin, etc.) fail under domain shift; ADA methods typically rely on source data to identify diverse target samples, which is unavailable in the source-free setting.

**Difficulty in Cross-Domain Alignment**: Without source data, manifold alignment through minimizing distribution divergence is impossible, yet the annotation cost raises performance expectations.

**Difficulty in Continuous Improvement**: The iterative query-adaptation process must ensure performance gains in every round.

### Limitations of Prior Work

- **MHPL**: Ensembles three active learning strategies but relies on source dissimilarity metrics, **allowing only single-round sampling**, which fails to guarantee continuous improvement.
- **SALAD**: Introduces an extra Guided Attention Transfer Network, suffering from high computational overhead and low annotation efficiency.
- **SFUDA Methods (e.g., SHOT++)**: Use no annotations but require 21.6K iterations (5.8h) on VisDA-C to reach 87.3% accuracy, whereas LFTL requires only 780 iterations (0.3h computation + 1.83h annotation time) to achieve 87.4%.

### Key Insight: "Learn from the Learnt"

Source domain knowledge is encapsulated in the pre-trained model $\mathcal{M}_s$; after each active learning round, the model from the previous round $\mathcal{M}_t^{(r-1)}$ also encapsulates acquired target domain knowledge. **Leveraging intermediate results—such as hypotheses and feature representations from previous rounds—serves as a zero-overhead source of information.**

## Method

### Overall Architecture

LFTL alternately executes two phases:
1. **Contrastive Active Sampling (CAS)**: Utilizes hypotheses from the previous round's model to query the target samples that are currently the most informative.
2. **Visual Persistence-guided Adaptation (VPA)**: Uses feature representations provided by models from previous rounds to guide distribution alignment in the target domain.

Given a source pre-trained model $\mathcal{M}_s$ and a total annotation budget $B$, adaptation is performed in $R$ iterations, where each round queries $b = B/R$ samples and adapts the model.

### Key Designs

#### 1. **Contrastive Active Sampling (CAS)**

**Function**: Selects the $b$ most informative samples from the unlabeled target sample pool to request annotations in each active learning round.

**Mechanism**: Inspired by contrastive decoding, it identifies "still difficult" samples by comparing the differences in predictive distribution between the current model $\mathcal{M}_t^{(r)}$ and the model from the previous round $\mathcal{M}_t^{(r-1)}$:

$$\tilde{\mathbf{p}}^{(r)}(\cdot|x_{tu}^i) = \begin{cases} \log \mathbf{p}^{(r)} & \text{if } r=0 \\ \log \mathbf{p}^{(r)} + \alpha(\log \mathbf{p}^{(r)} - \log \mathbf{p}^{(r-1)}) & \text{if } r>0 \end{cases}$$

- When $r=0$ (the first round): Prediction goes directly to the source model.
- When $r>0$: Compares predictions of the current model with those from the previous round, **magnifying the difference between the two rounds**.

On the contrastive-decoded prediction $\tilde{\mathbf{p}}$, the **Best-versus-Second-Best (BvSB)** is applied as the uncertainty metric:

$$u_{cm}(x_{tu}^i) = \tilde{p}(y_a^i | x_{tu}^i) - \tilde{p}(y_b^i | x_{tu}^i)$$

Smaller $u_{cm}$ $\rightarrow$ higher uncertainty and higher novelty of the sample $\rightarrow$ prioritized for annotation query.

**Class-Balancing Factor**: To prevent over-sampling of certain "easy-to-transfer" categories, a class transferability estimate $u_{ct}$ is introduced. It calculates the frequency of each category among high-confidence samples, where higher frequencies imply easier transfer, and their weights are thus reduced. The final hybrid criterion is:

$$u^i = u_{cm}^i + \lambda \cdot u_{ct}(y_a^i)$$

**Design Motivation**: (1) Utilizing the model from the previous round as a "control" incurs zero cost and requires no extra computation; (2) Contrastive decoding simultaneously considers individual uncertainty and progress stagnation across training time; (3) The class-balancing factor avoids class sampling bias under domain shift.

#### 2. **Visual Persistence-guided Adaptation (VPA)**

**Function**: Uses active annotated samples as anchors to guide the alignment of unlabeled features.

**Mechanism**: Treating active labeled samples as "representative anchors", it encourages unlabeled samples to cluster around their nearest anchors through minimizing soft similarity:

$$\mathcal{L}_{ac} = -\mathbb{E}_{x_{tu} \sim \mathcal{T}_u} \mathbf{d}_{tu}^T \log(\mathbf{d}_{tu})$$

where $\mathbf{d}_{tu} = \delta[\mathcal{D}(f_e(x_{tu}), \mathbf{f}(\mathcal{T}_l))]$ is the normalized cosine distance vector between the unlabeled sample and all anchors.

**Persistent Memory Bank**: The key novelty is that the feature representations of anchors are not computed solely using the current model, but are fused with historical predictions via Exponential Moving Average (EMA):

$$\tilde{\mathbf{f}}(x_{tl}) \leftarrow \gamma \mathbf{f}(x_{tl}) + (1-\gamma) \tilde{\mathbf{f}}(x_{tl}), \quad \gamma = 0.9$$

This ensures that source domain knowledge and domain-invariant knowledge learned during intermediate rounds are not forgotten during iterations.

**Design Motivation**: (1) Distribution alignment cannot be executed directly; thus, labeled anchors are leveraged to approximate the source distribution structure; (2) The EMA persistent memory bank tackles "catastrophic forgetting" during iterative adaptation by effectively retaining the domain-invariant information at each round; (3) Avoiding pseudo-labels avoids error accumulation.

#### 3. **Entropy Minimization Regularization**

A standard entropy minimization loss is introduced to promote discriminative features:

$$\mathcal{L}_{ent} = -\mathbb{E}_{x_{tu} \sim \mathcal{T}_u} \mathbf{p}(\cdot|x_{tu})^T \log[\mathbf{p}(\cdot|x_{tu})]$$

### Loss & Training

The total loss consists of three components:

$$\mathcal{L} = \mathcal{L}_{ce} + \beta_1 \mathcal{L}_{vpa} + \beta_2 \mathcal{L}_{ent}$$

- $\mathcal{L}_{ce}$: Standard cross-entropy loss on labeled samples
- $\mathcal{L}_{vpa}$: Visual persistence-guided clustering loss on unlabeled samples
- $\mathcal{L}_{ent}$: Entropy minimization on unlabeled samples

**Highly Efficient Training**: The entire query-adaptation process on VisDA-C requires only 780 training iterations (0.3h). Factoring in the estimated annotation time (1.83h), the total time is far less than the 5.8h required by the SFUDA method SHOT++.

## Key Experimental Results

### Main Results: VisDA-C (ResNet101)

| Method | Setting | Annotation % | Mean Accuracy (%) |
|------|------|-------|-------------|
| SHOT++ | SFUDA | 0% | 87.3 |
| MHPL | SFADA | 5% | 85.9 |
| SALAD | SFADA | 5% | 86.1 |
| LADA | ADA (Requires Source Data) | 5% | 87.5 |
| **LFTL (Ours)** | **SFADA** | **1%** | **87.4** |
| **LFTL (Ours)** | **SFADA** | **5%** | **89.0** |

LFTL outperforms **SHOT++ using only 1% annotation** (which uses 0% annotation but requires $27\times$ the training iterations), and surpasses LADA (which requires source data) at 5% annotation.

### Ablation Study: Component Contributions (VisDA-C, 5% Annotation)

| Configuration | Accuracy (%) | Description |
|------|----------|------|
| Source Model (No adaptation) | 52.3 | Baseline |
| + Standard Active Learning (Entropy) | 82.1 | Limited efficacy of standard active learning criteria under domain shift |
| + CAS (W/O Class Balancing) | 85.7 | Contrastive sampling yields significant gains |
| + CAS (W/ Class Balancing) | 86.3 | Class-balancing factor brings further improvement |
| + CAS + VPA (W/O EMA) | 87.5 | Anchor clustering guidance is effective |
| + CAS + VPA (W/ EMA Persistent Memory) | **89.0** | EMA prevents forgetting, full components reach the optimum |

### Key Findings

1. **Striking Efficiency**: On VisDA-C, the entire query-adaptation workflow of LFTL is **$17\times$ faster** than SHOT++ (780 vs. 21.6K iterations) without using source data.
2. **Continuous Improvement**: Performance steadily improves (87.4% $\rightarrow$ 89.0%) as the annotation budget increases from 1% to 5%, a guarantee that methods like MHPL lack.
3. **Active Sampling Time Saved by 25%**: Compared to the ADA SOTA (LADA), the sampling phase is 25% faster, and adaptation is approximately $17\times$ faster.
4. **Strong Generalizability**: Achieves SOTA SFADA performance across three benchmarks of different scales: Office-31, DomainNet, and VisDA-C.

## Highlights & Insights

1. **Transfer of the "Contrastive Decoding" Concept**: Adapting contrastive decoding from NLP to active learning, using the predictive difference between successive model states to identify novel samples is simple and elegant.
2. **Key Concept of Zero Extra Overhead**: Both CAS and VPA utilize intermediate results naturally generated during iteration (the model from the previous round, features from previous rounds) without introducing extra computation.
3. **Elegant EMA Persistent Memory Design**: Using momentum updates to fuse historical and current feature evaluations solves the forgetting issue in multi-round iterations with a single equation.
4. **No Pseudo-labeling**: Prevents error accumulation caused by pseudo-labels under extreme source-free and low-annotation conditions.

## Limitations & Future Work

1. **Limited to Closed-Set DA**: More complex semantic shift scenarios such as open-set, partial-set, and universal DA remain unexplored.
2. **Hyperparameter Sensitivity in Contrastive Sampling**: Hyperparameters such as $\alpha$, $\lambda$, and $\kappa$ require tuning, and different datasets might require different configurations.
3. **Omission of Detection/Segmentation Tasks**: The experiments focus on classification tasks, leaving the applicability of the domain adaptation method to object detection and semantic segmentation unverified.
4. **Annotation Budget Allocation Strategy**: Allocating a fixed $b = B/R$ samples per round is used; adaptive budget allocation strategies were not investigated.

## Related Work & Insights

- **SHOT++ [NeurIPS 2021]**: A strong SFUDA baseline. This work proves that a minor annotation budget can substantially outperform purely unlabeled methods.
- **LADA [CVPR 2022]**: ADA SOTA, which relies on source data. LFTL achieves comparable or superior performance without requiring source data.
- **Contrastive Decoding [ACL 2023]**: An NLP technique that inspired the design of CAS.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Although the SFADA setting is not a pioneer, the designs of CAS and VPA are novel and natural.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive validation across three benchmarks, exhaustive ablation, and thorough efficiency comparison.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-motivated progression of problems and a clear mathematical notation system.
- **Value**: ⭐⭐⭐⭐ — High reference value for practical deployment scenarios with standout efficiency advantages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ECCV 2024\] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)
- [\[ECCV 2024\] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[NeurIPS 2025\] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](../../NeurIPS2025/llm_evaluation/adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)

</div>

<!-- RELATED:END -->
