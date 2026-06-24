---
title: >-
  [Paper Note] System-Aware Unlearning Algorithms: Use Lesser, Forget Faster
description: >-
  [ICML2025][LLM Safety][machine unlearning] Proposes a new definition of system-aware unlearning that restricts the adversary's capability to access only what is actually stored in the system rather than all remaining data. Based on core set and selective sampling, an exact unlearning algorithm for linear classification is designed, achieving sublinear memory and extremely low deletion time.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "machine unlearning"
  - "system-aware"
  - "selective sampling"
  - "core set"
  - "linear classification"
  - "deletion capacity"
date: 2026-05-08
content_hash: 76238d4d1a4a6c05
---

# System-Aware Unlearning Algorithms: Use Lesser, Forget Faster

**Conference**: ICML2025  
**arXiv**: [2506.06073](https://arxiv.org/abs/2506.06073)  
**Code**: Not open-source  
**Area**: Others/Machine Unlearning  
**Keywords**: machine unlearning, system-aware, selective sampling, core set, linear classification, deletion capacity

## TL;DR

Proposes a new definition of system-aware unlearning that restricts the adversary's capability to access only what is actually stored in the system rather than all remaining data. Based on core set and selective sampling, an exact unlearning algorithm for linear classification is designed, achieving sublinear memory and extremely low deletion time.

## Background & Motivation

**Background**: Machine unlearning aims to make trained ML models efficiently "forget" requested data points to comply with the "right to be forgotten" in privacy regulations like GDPR and CCPA. The current standard definitions (exact/approximate unlearning) require the unlearned model to be statistically indistinguishable from a model retrained from scratch (without the deleted data).

**Limitations of Prior Work**:
1. The standard definition assumes the strongest adversary who can access all remaining data $S \setminus U$, which is overly pessimistic and unrealistic.
2. Cherapanamjeri et al. (2025) proved that under the traditional definition, even exact unlearning for linear classifiers must store the **entire dataset**, making unlearning infeasible on large-scale data.
3. Existing efficient unlearning algorithms are nearly limited to convex loss functions and cannot generalize to more general model classes.

**Core Idea**: If a learning system only stores and uses a small fraction of the training data (a core set), even a completely compromised system will only reveal the core set to the adversary. Thus, most data points naturally possess privacy protection—"the less used, the faster forgotten."

## Method

### 1. System-Aware Unlearning Definition

The State-of-System $\textsf{I}_A(S, U)$ is defined as all information **actually stored** in the system (models, retained samples, auxiliary statistics, etc.) after the unlearning algorithm processes the deletion requests.

**Definition 2.3 (System-Aware-$(\varepsilon, \delta)$-Unlearning)**: An algorithm $A$ is a system-aware-$(\varepsilon, \delta)$-unlearning algorithm if for any training set $S$, there exists $S' \subseteq S$ such that for all deletion requests $U \subseteq S$ and measurable set $F$:

$$\Pr(\textsf{I}_A(S, U) \in F) \leq e^\varepsilon \cdot \Pr(\textsf{I}_A(S' \setminus U, \emptyset) \in F) + \delta$$

and its reverse inequality holds. When $S' = S$, it degenerates to the traditional definition, so system-aware unlearning strictly generalizes the traditional definition.

**Key Theoretical Guarantee (Theorem 2.4)**: Through a mutual information argument, the information about $U$ leaked by $S' \setminus U$ is no more than that by $S \setminus U$:

$$\sup_\mu (\mathsf{MI}(U; S' \setminus U) - \mathsf{MI}(U; S \setminus U)) \leq 0$$

### 2. Core-Set-Based Unlearning Framework

**Core Set Algorithm (Definition 2.5)**: If for a learning algorithm $A$ there exists a mapping $\mathfrak{C}$ such that $\mathfrak{C}(S) \subseteq S$ and $A(S) = A(\mathfrak{C}(S))$, then $\mathfrak{C}(S)$ is a core set.

**General Framework (Theorem 3.1)**: By combining any traditional $(\varepsilon, \delta)$-unlearning algorithm $A_{un}$ with a core set mapping $\mathfrak{C}$, a system-aware unlearning algorithm $A_{CS}$ is constructed: during the learning phase, the core set is extracted and stored exclusively; during unlearning, updates are only performed on deleted points within the core set. No computation is required for deletion requests outside the core set.

### 3. Exact Unlearning Algorithm for Linear Classification (Algorithm 1)

**Selective Sampling**: BBQSampler (Cesa-Bianchi et al., 2009) is adopted as the core set construction method. For each arriving sample $x_t$, the query condition is checked:

$$x_t^\top A_{t-1}^{-1} x_t > T^{-\kappa}$$

If satisfied, the label is queried, the sample is added to the core set $\mathcal{Q}$, and the values are updated: $A_t \leftarrow A_{t-1} + x_t x_t^\top$, $b_t \leftarrow b_{t-1} + y_t x_t$, $w_t \leftarrow A_t^{-1} b_t$.

**DeletionUpdate**: For a deleted point $(x, y) \in U \cap \mathcal{Q}$ in the core set, an inverse update is executed: $A \leftarrow A - xx^\top$, $b \leftarrow b - yx$, $w \leftarrow A^{-1}b$ (using the Sherman-Morrison formula, taking $O(d^2)$ time).

**Monotonicity (Theorem 4.2)**: The query condition of BBQSampler depends only on $x$ (not on the label $y$), thus possessing monotonicity: $\mathfrak{C}(\mathfrak{C}(S) \setminus U) = \mathfrak{C}(S) \setminus U$. This implies that there is no need to rerun selective sampling during unlearning to determine a new core set.

**Theoretical Bounds (Theorem 4.3)**:
- Memory: $N_T = O(d \cdot T^\kappa \log T)$, sublinear in $T$
- Excess risk after $K$ core set deletions: $\mathcal{E}(w) = O\left(\frac{N_T \log T + \log(1/\delta)}{T - T_{\bar\varepsilon} - N_T}\right)$
- Core set deletion capacity: $K = O\left(\frac{\bar\varepsilon^2 \cdot T^\kappa}{d \log T \cdot \log(1/\delta)}\right)$
- Expected deletion time: $\mathbb{E}[\text{time per deletion}] \leq \frac{d^3 T^\kappa \log T}{T}$ (under a uniform deletion distribution)

### 4. Generalization to General Function Classes (Algorithm 2)

Based on GeneralBBQSampler (Gentile et al., 2022), the method is generalized to a general function class $\mathcal{F}$, where the core set size is controlled by an eluder-dimension-like quantity $\mathfrak{D}(\mathcal{F}, S)$.

## Key Experimental Results

### Datasets
- **Purchase Dataset**: Binary classification purchase data, 249,215 samples, $d=600$
- **Margin Dataset**: Synthetic data, 200,000 samples, $d=100$, hard margin $\gamma=0.1$

### Baselines

| Method | Description |
|------|------|
| Algorithm 1 (Ours) | Core-set-based unlearning via selective sampling |
| SISA (Bourtoule et al., 2021) | Sharding and slicing + storage of intermediate models |
| Exact Retraining | Retraining from scratch after each deletion |
| Sekhari et al. (2021) | Linear regression degenerated into exact retraining |

### Main Results (80,000 deletions, approx. 40% of the data)

**Table 1: Purchase Dataset**

| Metric | Algorithm 1 | SISA | Exact Retraining |
|------|------------|------|------------------|
| Initial Training Time (s) | 186.3 | 30.2 | 1828.7 |
| Cumulative Deletion Time (s) | **58.3** | 1174.3 | 579.3 |
| Data Storage Ratio | **13.1%** | 100% | 100% |

**Table 2: Margin Dataset**

| Metric | Algorithm 1 | SISA | Exact Retraining |
|------|------------|------|------------------|
| Initial Training Time (s) | **1.3** | 20.6 | 67.8 |
| Cumulative Deletion Time (s) | **0.5** | 697.1 | 27.1 |
| Data Storage Ratio | **0.6%** | 100% | 100% |

### Key Findings
1. Algorithm 1 maintains comparable accuracy to exact retraining with significantly lower memory (13.1% / 0.6%) on both datasets.
2. Under label-dependent deletion sequences (distribution shifts), the accuracy of Algorithm 1 remains stable, whereas SISA degrades significantly.
3. The better the data compressibility (Margin Dataset), the greater the advantage of Algorithm 1—completely outperforming in training/deletion time and memory.

## Highlights & Insights

1. **Breakthrough in Definition**: Relaxing the adversary model from "knowing all remaining data" to "only observing what is actually stored in the system" is both reasonable and powerful—bypassing the theoretical lower bound that "exact unlearning for linear classification must store all data".
2. **Elegant Reduction**: The reduction from selective sampling to unlearning is highly natural—"only query when uncertain, only unlearn what was queried".
3. **Monotonicity is Key**: The query condition of BBQSampler relies solely on the input $x$ (not on the labels), ensuring that deletions do not cause new points to be queried, thus avoiding rerunning the sampler during unlearning.
4. **Theoretical Completeness**: Provides a comprehensive tradeoff analysis of memory, deletion capacity, excess risk, and expected deletion time.

## Limitations & Future Work

1. **Limited to Classification Tasks**: The current algorithm and theory only target binary classification (linear and general function classes), leaving regression, generation, and other tasks unaddressed.
2. **Realizability Assumption**: Requires the existence of a Bayes optimal predictor within the hypothesis class (realizability), which limits practical applicability.
3. **Limited Scale of Experiments**: Only validated on medium-scale linear classification, without testing in deep networks or large-model fine-tuning scenarios.
4. **Approximate Variant Unexplored**: The paper mainly focuses on exact system-aware unlearning ($\varepsilon=\delta=0$); designing algorithms for the approximate version is left as future work.
5. **Lack of Verifiable Guarantees for Out-of-Core-Set Deletions**: Although points outside the core set are "deleted for free", there lacks auditability for this claim.

## Related Work & Insights

- **Sekhari et al. (2021)**: Proposed the standard definition of traditional certified unlearning and the concept of deletion capacity.
- **Cherapanamjeri et al. (2025)**: Proved that the memory lower bound for exact linear classification unlearning under the traditional definition is $\Omega(T)$, which this paper bypasses.
- **Bourtoule et al. (2021) SISA**: A data sharding method, serving as the main baseline in the experiments.
- **Cesa-Bianchi et al. (2009) BBQSampler**: The source of the core sampling algorithm used in this paper.
- **Ghazi et al. (2023)**: Ticketed learning-unlearning, establishing the connection between unlearning memory complexity and active learning query complexity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Bypasses the theoretical barriers of traditional unlearning from a definitional level, offering a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments are limited to linear classification, with restricted scale and scenarios.
- Writing Quality: ⭐⭐⭐⭐ — Highly clear theoretical arguments; builds logically from definitions to frameworks, algorithms, theory, and experiments.
- Value: ⭐⭐⭐⭐ — Provides a new theoretical paradigm for machine unlearning, with the potential to advance more practical unlearning algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](../../ACL2025/llm_safety/answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](../../ACL2025/llm_safety/manu_modality_aware_unlearning.md)

</div>

<!-- RELATED:END -->
