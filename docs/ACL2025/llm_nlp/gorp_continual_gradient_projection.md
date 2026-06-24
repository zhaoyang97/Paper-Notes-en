---
title: >-
  [Paper Note] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs
description: >-
  [ACL 2025][LLM (Other)][Continual Learning] GORP proposes to unify the gradients of full-rank parameters and LoRA low-rank parameters by projecting them into a low-rank gradient subspace for joint updates. By utilizing the first moment of Adam to implicitly construct a shared gradient space across tasks, it alleviates catastrophic forgetting. In continual学习 settings on T5 and LLaMA2, its performance is close to the multi-task joint training upper bound.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Continual Learning"
  - "Gradient Projection"
  - "LoRA"
  - "Catastrophic Forgetting"
  - "Low-Rank Optimization"
date: 2026-05-08
content_hash: b9c3e372bfa4d53d
---

# GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs

**Conference**: ACL 2025  
**arXiv**: [2507.02503](https://arxiv.org/abs/2507.02503)  
**Code**: [https://github.com/Wcxwcxw/GORP](https://github.com/Wcxwcxw/GORP)  
**Area**: LLM/NLP  
**Keywords**: Continual Learning, Gradient Projection, LoRA, Catastrophic Forgetting, Low-Rank Optimization

## TL;DR
GORP proposes to unify the gradients of full-rank parameters and LoRA low-rank parameters by projecting them into a low-rank gradient subspace for joint updates. By utilizing the first moment of Adam to implicitly construct a shared gradient space across tasks, it alleviates catastrophic forgetting. In continual学习 settings on T5 and LLaMA2, its performance is close to the multi-task joint training upper bound.

## Background & Motivation
**Background**: Continual fine-tuning of LLMs requires learning on sequential tasks and faces catastrophic forgetting. LoRA is widely used in continual learning due to its parameter efficiency.

**Limitations of Prior Work**:
   - The low-rank limitation of LoRA restricts its expressiveness, and the limited search space leads to insufficient learning of new tasks.
   - Explicit parameter constraint methods (e.g., regularization, sparsification) such as O-LoRA and MIGU cannot dynamically adapt to changes in the gradient space of new tasks.
   - Calculating the implicit feature space covariance directly in the original high-dimensional space is computationally prohibitive.

**Key Challenge**: The low-rank nature of LoRA limits plasticity (learning new tasks), while explicit constraints limit stability's ability to adapt to changes in the gradient space.

**Key Insight**: Gradient matrices naturally exhibit a low-rank structure during training, allowing for efficient operations in a low-rank space.

**Core Idea**: Project gradients of full-rank parameters into the low-rank space as well, jointly updating them with LoRA gradients in a unified shared gradient subspace.

## Method

### Overall Architecture
GORP consists of two components: (1) **Gradient Shared Space Construction**: After training each task, the first moment of Adam is used via SVD to construct the principal gradient directions of the task, gradually expanding the cross-task shared gradient space; (2) **Low-Rank Projection Optimization**: When training a new task, gradients of both full-rank and LoRA parameters are projected into directions orthogonal to old tasks, preventing forgetting while expanding the search space.

### Key Designs

1. **Gradient Shared Space Construction (Gradient Shared Space)**:

    - **Function**: Approximate the principal gradient directions of each task using the first moment of Adam $M_t$, and take the top $k$ base vectors after SVD.
    - **Mechanism**: For task 1, $M_1^l = U_1^l \Sigma_1^l (V_1^l)^T$, taking the top $k$ vectors that satisfy $\|(M_1^l)_k\|_F^2 > \epsilon_t^l \|M_1^l\|_F^2$ to construct $\mathcal{S}_1^l$. Subsequent tasks are first projected onto the orthogonal space: $\hat{M}_2^l = M_2^l - \mathcal{S}^l(\mathcal{S}^l)^T M_2^l$, and then SVD is used to expand the space.
    - **Design Motivation**: The first moment aggregates historical gradient information, representing the overall gradient direction of the task more effectively than randomly sampled hidden features.

2. **Joint Full-Rank and Low-Rank Low-Rank Projection**:

    - **Function**: Separately project the gradients of LoRA parameters and full-rank parameters onto the orthogonal complement of the shared space.
    - **Mechanism**: LoRA gradient projection $G_{A,l}' = G_{A,l} - \mathcal{S}_{t-1}^{A,l}(\mathcal{S}_{t-1}^{A,l})^T G_{A,l}$; full-rank gradients are first SVD-reduced to $k$ dimensions $G_{t,l}' = U_{l,k}^T G_{t,l} V_{l,k}$, and then projected onto the orthogonal direction $P_{t,l} = G_{t,l}' - \mathcal{S}_{t-1}^l(\mathcal{S}_{t-1}^l)^T G_{t,l}'$.
    - **Design Motivation**: Full-rank parameters enhance the flexibility of the search space, while low-rank projection ensures efficiency and mitigates forgetting.

3. **Implicit vs. Explicit Constraints**:

    - **Function**: Replace parameter orthogonality (explicit) with gradient orthogonality (implicit).
    - **Mechanism**: GORP does not directly constrain parameter changes; instead, it constrains the gradient direction to be orthogonal to old tasks, allowing the model to freely choose the optimal parameter update magnitude.
    - **Design Motivation**: Gradient orthogonality better guarantees that the learning direction does not interfere with old tasks while allowing parameters to change flexibly along orthogonal directions.

### Computational Efficiency Design
- SVD of full-rank parameters is executed once every $T=10$ steps (not every step), significantly reducing overhead.
- Low-rank projection reduces the matrix dimension input to Adam optimizer to $k \times k$ ($k=8$), which is much smaller than the original dimension.

## Key Experimental Results

### Main Results (T5-Large)

| Method | Standard CL (3 orders avg) | Large-scale Tasks (15 tasks, 3 orders avg) |
|------|---------------------|--------------------------------|
| O-LoRA | 75.8 | 69.6 |
| MIGU | 76.6 | 70.0 |
| N-LoRA | 78.8 | 72.4 |
| **GORP** | **79.8** | **76.0** |
| MTL (Upper Bound) | 80.0 | - |

### LLaMA2-7B Results

| Method | Order 1-3 avg | BWT (forgetting) |
|------|-------------|-----------------|
| O-LoRA | 76.1 | -7.8% |
| N-LoRA | 77.6 | -4.9% |
| **GORP** | **78.6** | **-0.8%** |

### Ablation Study

| Configuration | Avg Performance | Description |
|------|----------------|------|
| B (LoRA Projection Only) | baseline | Baseline |
| B+L (With Full-Rank Projection) | +0.7% | Full-rank parameters expand search space |
| B+S (With Gradient Space) | +2.0% | Shared gradient space contributes the most |
| B+L+S (Full GORP) | **+3.9%** | Complementary components |

### Key Findings
- **Close to MTL Upper Bound**: GORP (79.8%) is close to MTL (80.0%) under standard CL, almost eliminating the gap in continual learning.
- **Significant Reduction in Forgetting**: BWT decreases from -7.8% of O-LoRA to -0.8% (T5).
- **Unseen Task Generalization**: Outperforms O-LoRA by 26.2% and N-LoRA by 7.0%.
- **Computationally Efficient**: Training time is comparable to O-LoRA, while FLOPs are 1/550 of O-LoRA.
- **Greater Advantage with More Tasks**: Outperforms N-LoRA by 3.6% with 15 tasks, indicating that the dynamic gradient space handles long-sequence tasks better.

## Highlights & Insights
- The insight of **gradient orthogonality vs. parameter orthogonality** is profound: explicitly constraining parameter orthogonality is a "conservative" strategy that easily limits new task learning; implicitly constraining gradient orthogonality is a "flexible" strategy, controlling only the direction of updates, not the magnitude.
- **Gradient space approximation via first moment** is an ingenious engineering design: the momentum term of Adam naturally accumulates the gradient statistics of the task, which is far more efficient than additionally calculating the covariance matrix.
- The idea of **full-rank and low-rank complementarity** is transferable: in other scenarios requiring parameter-efficient fine-tuning, there is no need to completely abandon full-rank parameters; instead, full-rank gradients can be operated within a low-rank subspace.

## Limitations & Future Work
- As the number of tasks increases, the dimension of the gradient space grows continuously, potentially causing dimension explosion.
- Multiple hyperparameters require tuning: $k=8$, $\alpha$, $T$, and different learning rates.
- The performance in sequential editing and online learning scenarios has not been validated.
- The scalability to larger models (70B+) remains unknown.

## Related Work & Insights
- **vs. O-LoRA**: O-LoRA uses explicit parameter constraints and is static; GORP uses implicit gradient constraints and is dynamically adaptive.
- **vs. MIGU**: MIGU uses sparsification (updating only growing units), while GORP uses gradient projection, which is more flexible.
- **vs. N-LoRA**: N-LoRA extends the orthogonal subspace of LoRA but remains within low-rank constraints; GORP introduces full-rank parameters to expand the search space.

## Rating
- Novelty: ⭐⭐⭐⭐ Shared gradient space + joint full-rank and low-rank projection is a meaningful innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two models, multiple task sequences, multiple benchmarks, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear method derivations and systematic experimental design.
- Value: ⭐⭐⭐⭐ High reference value for continual learning in LLMs.

Consistently outperforms methods like O-LoRA, MIGU, and N-LoRA on continual fine-tuning benchmarks, effectively balancing plasticity and stability.

## Highlights & Insights
- Joint optimization of full-rank and low-rank parameters in a unified low-rank gradient space—balancing both expressiveness and efficiency.
- Replacing hidden feature covariance with the first moment—significantly reducing computational costs.

## Limitations & Future Work
- The frequency $T$ of gradient SVD decomposition needs to be manually set.
- Only validated at the LLaMA-7B scale.

## Related Work & Insights
- Refer to the detailed comparison in the Related Work section of the original paper.

## Rating
- Novelty: ⭐⭐⭐⭐ Joint full-rank + low-rank updates and implicit gradient space construction are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons across multiple continual learning settings.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithm pseudocode.
- Value: ⭐⭐⭐⭐ Continual fine-tuning is an important direction, and GORP provides a concise and effective solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DenseLoRA: Dense Low-Rank Adaptation of Large Language Models](denselora_dense_low-rank_adaptation_of_large_language_models.md)
- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)

</div>

<!-- RELATED:END -->
