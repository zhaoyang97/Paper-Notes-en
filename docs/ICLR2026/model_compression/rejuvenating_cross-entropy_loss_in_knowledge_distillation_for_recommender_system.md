---
title: >-
  [Paper Note] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems
description: >-
  [ICLR 2026][Model Compression][knowledge distillation] This paper theoretically demonstrates that CE loss maximizes a lower bound of NDCG in recommender system KD only when a *closure assumption* is satisfied—the candida…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "knowledge distillation"
  - "cross-entropy"
  - "NDCG"
  - "recommender system"
  - "ranking"
  - "partial NDCG"
date: 2026-05-08
content_hash: 956fbccb0607391b
---

# Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems

**Conference**: ICLR 2026
**arXiv**: [2509.20989](https://arxiv.org/abs/2509.20989)  
**Code**: [GitHub](https://github.com/BDML-lab/RCE-KD)  
**Area**: Recommender Systems / Knowledge Distillation / Model Compression
**Keywords**: knowledge distillation, cross-entropy, NDCG, recommender system, ranking, partial NDCG

## TL;DR
This paper theoretically demonstrates that CE loss maximizes a lower bound of NDCG in recommender system KD only when a *closure assumption* is satisfied—the candidate subset must contain the student's top-ranked items. However, the actual KD objective is to distill the ranking of the teacher's top items, and these two requirements conflict, explaining why vanilla CE performs poorly. Accordingly, the paper proposes RCE-KD: the teacher's top-K items are split into two groups based on whether they appear in the student's top-K, handled respectively by exact CE and sampling-approximated closure CE, with an adaptive fusion weight that evolves dynamically throughout training.

## Background & Motivation
**Background**: Knowledge distillation in recommender systems compresses large teacher models into compact student models. Response-based KD (CE loss, RRD, CD, etc.) is the dominant paradigm, and CE loss has proven highly successful in KD for CV and NLP.

**Limitations of Prior Work**:
   - CE loss performs **surprisingly poorly** in recommender KD — vanilla CE consistently underperforms all baselines (CD, RRD, HetComp, etc.) across three settings: MF→MF, LightGCN→LightGCN, and HSTU→HSTU.
   - Recommender KD has two unique characteristics: (1) it focuses on ranking rather than exact scores, particularly the ranking of teacher top items; and (2) since item sets are extremely large (millions), CE can only be computed over small subsets.
   - Existing theoretical connections between CE and NDCG apply only to binary labels and full-item settings, covering neither the practical setup of recommender KD.

**Key Challenge**: CE constrains partial NDCG only under the *closure assumption*—every item in the subset must have all student-higher-ranked items also present in the subset. Yet KD targets the ranking of teacher top items, and the student's and teacher's top items have almost no overlap early in training.

**Core Idea**: Split the teacher's top-K into two groups (intersection / difference with student top-K); for the first group, compute CE directly over the student's top-K (exactly satisfying closure); for the second group, use an adaptive sampling strategy to approximately satisfy closure.

## Method

### Overall Architecture
Pre-train large teacher + small student → obtain teacher prediction scores $\mathbf{r}_u^T$ → split teacher top-K $\mathcal{Q}_u^T$ into $(\mathcal{Q}_u^T)_1 = \mathcal{Q}_u^T \cap \mathcal{Q}_u^S$ and $(\mathcal{Q}_u^T)_2 = \mathcal{Q}_u^T \setminus (\mathcal{Q}_u^T)_1$ → compute $\mathcal{L}_1$ (exact) and $\mathcal{L}_2$ (sampling approximation) separately → adaptive fusion → jointly train the student with the base loss.

### Key Designs

1. **Theoretical Foundation: Extending CE → NDCG to Recommender KD**

    - Function: Formally establish the connection between CE loss and NDCG in the recommender KD setting.
    - **Theorem 4.1 (Full-item KD)**: Minimizing CE over the full item set is equivalent to maximizing a lower bound of NDCG, with relevance scores $y_i = \log_2(\sigma(r_{ui}^T) + 1)$. This provides a theoretical motivation for using CE in recommender KD.
    - **Theorem 4.4 (Partial-item KD)**: Minimizing CE over a subset $\mathcal{J}^u$ constrains partial NDCG, but only if $\mathcal{J}^u$ satisfies the **closure assumption (Assumption 4.3)**: for every item in the subset, all items ranked higher by the student must also be in the subset.
    - Design Motivation: Full-item KD is impractical at scale, but the theoretical guarantee for partial-item KD requires closure — revealing why vanilla CE fails.

2. **Splitting Strategy**

    - Function: Partition the teacher's top-K into two groups based on overlap with the student's top-K.
    - $(\mathcal{Q}_u^T)_1 = \mathcal{Q}_u^T \cap \mathcal{Q}_u^S$: items deemed important by both teacher and student → compute $\mathcal{L}_1$ directly over $\mathcal{Q}_u^S$ (which naturally satisfies closure).
    - $(\mathcal{Q}_u^T)_2 = \mathcal{Q}_u^T \setminus (\mathcal{Q}_u^T)_1$: items important to the teacher but ranked low by the student → require a sampling strategy to approximately satisfy closure.
    - Design Motivation: Computing CE directly over $\mathcal{Q}_u^T$ violates closure; naively appending student top items yields an excessively large subset. Splitting and handling each group separately is more efficient.

3. **Adaptive Sampling Strategy (for $\mathcal{L}_2$)**

    - Function: For each item $i \in (\mathcal{Q}_u^T)_2$, identify all items ranked higher by the student and increase their sampling probability; sample $L$ items and merge with $(\mathcal{Q}_u^T)_2$.
    - Mechanism: Sampling probability $p_j \propto e^{z_j/\tau}$, where $z_j$ is the cumulative count of items in $(\mathcal{Q}_u^T)_2$ that the student ranks below $j$. When the student ranks teacher top items poorly, the distribution approximates uniform sampling to cover more items; as training progresses and student rankings improve, sampling concentrates on critical items.
    - Design Motivation: Exactly satisfying closure would require including too many items; adaptive sampling balances efficiency and approximation quality.

4. **Adaptive Loss Fusion**

    - Fusion weight $\gamma = \exp(-\beta \cdot |(\mathcal{Q}_u^T)_1| / |\mathcal{Q}_u^T|)$, updated each epoch.
    - Small intersection (student has not yet learned well) → large $\gamma$ → emphasize pushing up the ranking of items in $(\mathcal{Q}_u^T)_2$.
    - Large intersection (student has largely caught up) → small $\gamma$ → emphasize refining the ordering within $(\mathcal{Q}_u^T)_1$.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{Base} + \lambda \cdot \mathcal{L}_{RCE-KD}$, where $\mathcal{L}_{RCE-KD} = (1-\gamma)\mathcal{L}_1 + \gamma \mathcal{L}_2$
- Teacher predictions are pre-cached; the teacher is not re-inferred during training.
- Sampling temperature $\tau=10$ is fixed; re-sampling is performed each epoch.

## Key Experimental Results

### Main Results (3 datasets × 3 backbones × 2 metrics)

| Dataset | Backbone | Method | Recall@20 | NDCG@20 |
|---------|----------|--------|-----------|---------|
| CiteULike | MF→MF | CD | baseline | baseline |
| | | RRD | improved | improved |
| | | HetComp | 2nd best | 2nd best |
| | | **RCE-KD** | **best** | **best** |
| Gowalla | same | **RCE-KD** | **best** | **best** |
| Yelp | same | **RCE-KD** | **best** | **best** |

RCE-KD achieves state-of-the-art results across all 9 settings (3 datasets × 3 backbones: MF/LightGCN/HSTU) with statistical significance ($p \leq 0.05$). Student performance approaches or matches the teacher.

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| $\mathcal{L}_1$ only (CE over student top items) | better than vanilla CE | benefit of satisfying closure |
| $\mathcal{L}_2$ only (sampling-approximated closure) | better than vanilla CE | approximate closure is effective |
| $\mathcal{L}_1 + \mathcal{L}_2$ with fixed weight | worse than adaptive | necessity of adaptive $\gamma$ |
| **Full RCE-KD** | **best** | splitting + sampling + adaptive fusion all contribute |

### Training Efficiency

| Method | Relative Training Time vs. Student | GPU Memory |
|--------|------------------------------------|------------|
| RCE-KD | ~1.1–1.3× | comparable to CE |
| RRD | ~2–3× | significantly higher |
| HetComp | ~2–4× | significantly higher |

RCE-KD adds only random sampling overhead on top of CE, achieving the highest training efficiency.

### Key Findings
- The root cause of vanilla CE's poor performance in recommender KD is the violation of the closure assumption — the overlap between student and teacher top items is extremely low (~10–20%) early in training.
- NDCG evolution curves during training empirically validate that RCE-KD successfully constrains NDCG, consistent with the theoretical prediction.
- RCE-KD generalizes to sequential recommendation and multimodal recommendation settings.

## Highlights & Insights
- **The discovery of the closure assumption** is the most valuable contribution — it precisely explains a puzzling empirical phenomenon (why CE underperforms in recommender KD) and directly guides algorithm design.
- The **theory-driven → algorithm design** paradigm is elegant: analyze the theoretical applicability conditions of CE → identify violations in practice → design an algorithm to remedy those violations.
- **Adaptive $\gamma$ scheduling** cleverly uses the overlap ratio between student and teacher top items as a proxy for training progress.

## Limitations & Future Work
- The degree of approximation to the closure assumption is not theoretically quantified — how well does sampling $L$ items approximate closure?
- The sampling temperature $\tau=10$ is fixed and may not be optimal across different datasets.
- Validation is limited to implicit-feedback ranking tasks; applicability to other recommendation paradigms such as rating prediction remains unknown.
- Partial NDCG only ensures ordering quality within the subset, without guaranteeing the ranking quality of items outside it.

## Related Work & Insights
- **vs. RRD (Kang 2020)**: RRD distills with ListMLE loss without theoretical analysis of CE; RCE-KD is grounded in theoretical analysis of CE and is more training-efficient.
- **vs. CD (Lee 2019)**: CD aligns predictions with point-wise loss with moderate effectiveness; this paper shows that list-wise CE, when correctly applied, is superior.
- **vs. CE-NDCG theory (Bruch 2019)**: Prior theory applies only to binary labels and full-item settings; this paper extends it to continuous labels (teacher scores) and partial-item KD.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical discovery of the closure assumption is original; the splitting + sampling + adaptive fusion design is theory-driven.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets × 3 backbones × multiple KD settings + comprehensive ablations + efficiency comparison + generalization validation.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; the logical chain from problem → theory → method → experiments is complete.
- Value: ⭐⭐⭐⭐ Provides both theoretical guidance and a practical method for using CE in recommender KD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](../../ICCV2025/model_compression/ea-kd_entropy-based_adaptive_knowledge_distillation.md)
- [\[ICLR 2026\] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](../../AAAI2026/model_compression/ctpd_cross_tokenizer_preference_distillation.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](../../ICML2026/model_compression/entropy-aware_on-policy_distillation_of_language_models.md)

</div>

<!-- RELATED:END -->
