---
title: >-
  [Paper Note] Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations
description: >-
  [NeurIPS 2025][Causal Inference][knowledge distillation] This paper proposes CoD (Counterfactual-explanation-infused Distillation), which injects counterfactual explanations into few-shot training sets to precisely map t…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "knowledge distillation"
  - "Counterfactual Explanation"
  - "few-shot learning"
  - "LLM Compression"
  - "Decision Boundary"
date: 2026-05-08
content_hash: 76e6c7581d1c52d4
---

# Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations

**Conference**: NeurIPS 2025
**arXiv**: [2510.21631](https://arxiv.org/abs/2510.21631)  
**Code**: [FaisalHamman/CoD](https://github.com/FaisalHamman/CoD)  
**Area**: Causal Inference
**Keywords**: knowledge distillation, Counterfactual Explanation, few-shot learning, LLM Compression, Decision Boundary

## TL;DR

This paper proposes CoD (Counterfactual-explanation-infused Distillation), which injects counterfactual explanations into few-shot training sets to precisely map the teacher's decision boundary, achieving significant improvements over standard distillation methods across 6 datasets using only 8–512 samples.

## Background & Motivation

**Background**: Knowledge distillation (KD) is the predominant approach for compressing large teacher LLMs into smaller student models. Task-aware KD further enables selective knowledge transfer targeting specific downstream tasks.

**Limitations of Prior Work**:
- Existing task-aware distillation methods (KD/LWD/TED) all assume access to sufficient labeled data;
- In few-shot settings, sparse data points cannot uniquely determine the teacher's decision boundary—multiple distinct student decision surfaces can fit the same sparse set of points (unfaithful distillation);
- Data selection strategies for distillation remain severely understudied, especially under few-shot settings.

**Key Challenge**: Under extremely limited data budgets, how can the student faithfully replicate the teacher's decision boundary?

**Key Insight**: Counterfactual explanations (CFEs) naturally reside near decision boundaries—they represent the minimal perturbation inputs that flip model predictions. Such samples can precisely supplement information in high-uncertainty regions.

**Mechanism**: Using half the budget on original samples and the other half on their corresponding CFEs (with total budget unchanged) characterizes the teacher's decision surface more accurately than using all original samples.

## Method

### Overall Architecture (Algorithm 1: CoD)

1. Given a budget of $k$ samples, select $k/2$ original labeled samples.
2. Generate a counterfactual explanation (CFE) for each original sample, yielding $k/2$ CFEs.
3. Merge into $k$ training samples (original + CFE pairs) and train the student via standard distillation.

### CFE Generation Pipeline

A hybrid strategy is adopted, combining LLM generation with teacher model verification:

- **Generation phase**: Given an input text and its label, GPT-4o is prompted to generate semantically similar variants with flipped labels (minimal modification principle).
- **Verification phase**: Candidate CFEs are fed into the teacher model to confirm that predictions do indeed flip; only valid CFEs are retained.
- **Manifold constraint**: LLM-based generation ensures that CFEs lie on the natural language data manifold (semantically plausible and grammatically correct), avoiding out-of-distribution samples produced by optimization-based methods.

**Example**: Original sentence "I loved the movie" (positive) → CFE "I hated the movie" (negative).

### Theoretical Guarantees

#### Theorem 1: Statistical Perspective (Fisher Information)

Under a logistic regression setting:

- Fisher information matrix $\mathcal{I}(\mathbf{w}_t; \mathcal{D}) = \sum_i p_t(1|\mathbf{x}_i)(1 - p_t(1|\mathbf{x}_i)) \mathbf{x}_i \mathbf{x}_i^\top$
- The weight factor $p(1-p)$ is maximized at $p = 0.5$ (i.e., on the decision boundary).
- CFEs naturally lie near the decision boundary → $\mathbf{w}_t^\top \mathbf{x}_c \approx 0$ → they contribute the maximum Fisher information.
- **Conclusion**: The FIM of the CFE dataset strictly dominates that of the standard dataset in the Loewner order, i.e., $\mathcal{I}(\mathbf{w}_t; \mathcal{D}_{cf}) \succ \mathcal{I}(\mathbf{w}_t; \mathcal{D})$, leading to smaller parameter estimation error for the student.

#### Theorem 2: Geometric Perspective (Hausdorff Distance)

Extended to nonlinear model settings:

- The teacher/student decision boundaries $\mathcal{M}_t, \mathcal{M}_s$ are defined as the level sets where $f(\mathbf{x}) = 0.5$.
- The line segment connecting an original sample and its CFE must cross the teacher boundary (since predictions differ at both endpoints).
- If the student matches the teacher's predictions at both endpoints, the student boundary also intersects this segment.
- **Conclusion**: $H(\mathcal{M}_s, \mathcal{M}_t) \leq \alpha + \varepsilon$
    - $\alpha$: maximum CFE perturbation distance (smaller is better);
    - $\varepsilon$: boundary coverage density (denser is better).
- **Intuition**: CFE pairs act like "pins" that anchor the student boundary close to the teacher boundary.

### Training Loss

$$\mathcal{L} = \mathcal{L}_{\text{hard}} + \alpha \cdot \mathcal{L}_{\text{KD}} + \beta \cdot \mathcal{L}_{\text{LWD}}$$

- $\mathcal{L}_{\text{hard}}$: cross-entropy between student predictions and ground-truth labels;
- $\mathcal{L}_{\text{KD}}$: KL divergence between teacher and student outputs;
- $\mathcal{L}_{\text{LWD}}$: MSE alignment of intermediate hidden representations (optional);
- Each mini-batch contains input–CFE pairs during training.

## Key Experimental Results

### Experimental Setup

- **Teacher/Student models**: DeBERTa-v3-base (100M) → small (44M) / xsmall (22M); Qwen2.5-1.5B → 0.5B
- **Baselines**: Standard KD, LWD (layer-wise alignment), TED (task-aware layer-wise distillation)
- **Datasets**: SST2, Sentiment140, IMDB, CoLA, Amazon Polarity, Yelp (all binary classification)
- **Few-shot setting**: $k \in \{8, 16, 32, 64, 128, 512\}$
- **Fair comparison**: CoD uses $k/2$ original + $k/2$ CFE; baselines use $k$ original samples

### Main Results (DeBERTa-v3 base→small)

| Dataset | Method | k=8 | k=16 | k=32 | k=64 |
|---------|--------|-----|------|------|------|
| IMDB | LWD | 76.0% | 83.6% | 87.5% | 88.9% |
| IMDB | LWD+CoD | **86.1%** | **88.6%** | **89.3%** | **89.8%** |
| Amazon | KD | 67.1% | 71.2% | 75.8% | 78.9% |
| Amazon | KD+CoD | **75.8%** | **79.5%** | **81.9%** | **81.2%** |
| SST2 | LWD | 62.7% | 72.1% | 77.6% | 81.7% |
| SST2 | LWD+CoD | **69.4%** | **78.5%** | **83.2%** | **83.0%** |

**Key Findings**:

- CoD yields the most substantial gains under extremely low data regimes ($k \leq 64$), with improvements exceeding 10 points on IMDB at $k=8$.
- As $k$ increases to 512, the advantage of CoD diminishes but remains competitive—while using only half the real labeled data.
- CoD is compatible with all three baseline methods (KD/LWD/TED), consistently improving each.

### Qwen2.5 Experiments (1.5B→0.5B)

Experiments on CoLA and Yelp confirm that CoD is equally effective for generative LLMs, with notable improvements in the $k=64$ to $k=512$ range.

### Ablation Study

| Ablation Configuration | Key Finding |
|------------------------|-------------|
| Remove soft labels ($\alpha=0$) | Significant performance drop; teacher soft-label calibration is critical to CFE effectiveness |
| Replace soft labels with random values | Dramatic degradation due to conflicting signals with hard labels |
| Different prompt templates for CFE generation | CoD is robust to prompt choice, with low variance |
| TED under few-shot setting | Does not outperform simple KD/LWD, yet TED+CoD still yields consistent gains |

## Highlights & Insights

- **Explainability → Training Signal**: The paper repurposes counterfactual explanations from XAI—originally designed to explain model decisions—as training signals to guide model compression, elegantly bridging explainability and model distillation.
- **Fisher Information Perspective**: The intuition that samples near the decision boundary carry the most parameter estimation information aligns perfectly with the theoretical formulation.
- **Hausdorff Distance Geometric Analysis**: Complements the statistical theory for nonlinear models and provides quantitative guarantees on student–teacher boundary alignment.
- **Fair and Convincing Experimental Design**: CoD redistributes the total data budget—half original, half CFE—and still outperforms baselines that use all original samples.
- **Simple Method + CoD Outperforms Complex Methods**: TED underperforms KD/LWD in few-shot settings, whereas simple KD+CoD achieves the best results, suggesting that data quality matters more than algorithmic complexity.

## Limitations & Future Work

- **Restricted to Binary Classification**: Current theory and experiments are limited to binary classification tasks; multi-class extension requires redefining the minimal-flip strategy for CFEs.
- **CFE Generation Cost**: Reliance on GPT-4o for CFE generation introduces additional API overhead; open-source alternatives warrant exploration.
- **Strong Theoretical Assumptions**: Theorem 1 assumes logistic regression and equal student–teacher capacity; the exact distillation assumption in Theorem 2 holds only approximately in practice.
- **Future Directions**: (1) Extension to multi-class, sequence labeling, and generation tasks; (2) iterative CFE generation using the student model itself to reduce dependence on external models; (3) active learning-style sampling to select the most informative samples for CFE generation.

## Related Work & Insights

- **vs. Standard KD/LWD/TED**: CoD is an orthogonal data augmentation strategy that can be directly combined with any distillation method.
- **vs. Data Augmentation Methods**: Conventional augmentations (synonym replacement, back-translation) do not guarantee proximity to the decision boundary; CFEs satisfy this by definition.
- **vs. Counterfactual Robustness/Fairness**: Prior work uses CFEs for debiasing or out-of-distribution generalization; this paper is the first to apply CFEs to knowledge distillation.
- **vs. Active Learning**: Both address data selection, but CoD requires no iterative querying—CFEs are generated in a single pass.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic application of counterfactual explanations to knowledge distillation; entirely novel perspective.
- Theoretical Depth: ⭐⭐⭐⭐ — Dual statistical and geometric guarantees, though linearity assumptions limit generality.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 6 datasets × 2 model families × 6 values of $k$ × 3 baselines × ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Intuition, theory, and experiments are well integrated, with excellent figures.
- Value: ⭐⭐⭐⭐ — Directly applicable to few-shot LLM deployment, though CFE generation cost must be considered.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)
- [\[NeurIPS 2025\] Root Cause Analysis of Outliers with Missing Structural Knowledge](root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)
- [\[NeurIPS 2025\] Performative Validity of Recourse Explanations](performative_validity_of_recourse_explanations.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)

</div>

<!-- RELATED:END -->
