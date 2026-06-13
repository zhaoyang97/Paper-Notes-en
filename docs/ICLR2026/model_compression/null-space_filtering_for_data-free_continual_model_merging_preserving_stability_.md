---
title: >-
  [Paper Note] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity
description: >-
  [ICLR 2026][Model Compression][Model Merging] This paper proposes NUFILT, a framework that exploits the geometric property of approximate alignment between task vectors and representation subspaces. By applying null-spac…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Model Merging"
  - "Continual Learning"
  - "Null-Space Projection"
  - "Stability-Plasticity"
  - "Data-Free"
date: 2026-05-08
content_hash: 97c5d186bc435c5a
---

# Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity

**Conference**: ICLR 2026
**arXiv**: [2509.21413](https://arxiv.org/abs/2509.21413)  
**Code**: [GitHub](https://github.com/zihuanqiu/NUFILT)  
**Area**: Model Compression
**Keywords**: Model Merging, Continual Learning, Null-Space Projection, Stability-Plasticity, Data-Free

## TL;DR

This paper proposes NUFILT, a framework that exploits the geometric property of approximate alignment between task vectors and representation subspaces. By applying null-space filtering to suppress interference with previous tasks and projection-aware LoRA to restore plasticity for new tasks, NUFILT achieves continual model merging without accessing any data. It outperforms OPCM by 4–8% on vision, NLP, and multimodal benchmarks, approaching the upper bound of individual fine-tuning.

## Background & Motivation

**Background**: Data-Free Continual Model Merging (DFCMM) has emerged as a popular direction in model reuse. The setting involves multiple task-specific models independently fine-tuned, which are then **incrementally merged** into a unified backbone without access to any original task data—motivated by both privacy constraints and storage overhead. The entire merging process must operate solely in parameter space.

**Limitations of Prior Work**: The central challenge in DFCMM is balancing stability and plasticity—merging a new task must not degrade knowledge of existing tasks (stability) while faithfully absorbing new task capabilities (plasticity). Existing methods each have shortcomings: (1) Simple arithmetic operations such as Weight Averaging and Task Arithmetic introduce parameter interference where signals from old and new tasks cancel each other, resulting in poor stability; (2) Orthogonal projection methods such as OPCM enforce task vectors into orthogonal subspaces, but when tasks are naturally correlated (e.g., EuroSAT and RESISC45, both remote sensing classification), the orthogonality constraint excessively suppresses new task signals, sacrificing plasticity; (3) Adaptive strategies such as AdaMerging and WEMOE require auxiliary data to calibrate merging coefficients, directly violating the data-free constraint.

**Key Challenge**: The fundamental difficulty is that stability and plasticity are inherently **data-level** concepts—requiring evaluation of interference on old task data and adaptation quality on new task data—yet DFCMM forbids data access. Finding effective parameter-space proxies for data-level objectives remains an open problem that prior methods have not adequately addressed.

**Key Insight**: The authors empirically observe a key geometric property: **the principal directions of task vectors are approximately aligned with the data representation subspaces of their respective tasks**. Intuitively, the parameter changes induced by fine-tuning predominantly occur along directions driven by task data. This implies that the SVD subspace of a task vector can serve as a data-free proxy for the data representation subspace, enabling the translation of data-level stability/plasticity objectives into projection operations in parameter space.

**Core Idea**: Since task vectors approximately align with data subspaces, null-space projection can ensure that old task features are not perturbed (stability), while projection-aware LoRA injects new task signals within the subspace orthogonal to old tasks (plasticity). The two components are linearly fused without introducing additional inference overhead.

## Method

### Overall Architecture

NUFILT takes as input a pre-trained model $\theta_0$, the current merged backbone $\theta_{t-1}^{\text{merged}}$, and a new task model $\theta_t$, and outputs the updated backbone $\theta_t^{\text{merged}}$. The pipeline is executed layer-wise in three stages: **Filtering** → **Adapting** → **Fusing**. The filtering stage constructs a null-space projector to remove components of the new task vector that overlap with old task subspaces; the adapting stage uses a lightweight LoRA module under the projector's constraint to recover new task signals that were excessively filtered; the fusing stage linearly merges the projector, task vector, and LoRA parameters back into the backbone without introducing any additional inference parameters.

### Key Designs

1. **Null-Space Filter**:

    - Function: Before merging a new task, components of the new task vector overlapping with old task subspaces are removed, ensuring that old task responses remain unchanged.
    - Mechanism: The cumulative update from pre-training to the current merged model is computed as $\tilde{\tau}_{\leq t-1}^{(l)} = \theta_{t-1}^{\text{merged},(l)} - \theta_0^{(l)}$; its SVD yields the top-$r_p$ right singular vectors $\hat{V}_{\leq t-1}^{(l)}$, from which the null-space projection matrix $P_t^{(l)} = I - \hat{V}_{\leq t-1}^{(l)} \hat{V}_{\leq t-1}^{(l)\top}$ is constructed. This projector zeros out any component within the old task subspace—i.e., $P_t^{(l)} x^{(l)} = 0$ for $x^{(l)} \in \text{span}(\hat{V}_{\leq t-1}^{(l)})$—ensuring that intermediate representations of old tasks are completely unaffected after the new merge.
    - Design Motivation: Unlike OPCM's orthogonal projection applied to individual task vectors, NUFILT's null-space projection operates on the **cumulative update**, more precisely characterizing the parameter space occupied by all previous tasks. However, pure projection has a side effect: when new and old tasks share certain directions, null-space filtering also removes useful new task signals, reducing plasticity—motivating the second stage as compensation.

2. **Projection-Aware LoRA Adaptation**:

    - Function: A low-rank adapter is injected within the subspace complementary to the null-space projector, recovering new task signals excessively suppressed by filtering.
    - Mechanism: The projector is extended to $P_t^{(l)} + B_t^{(l)} A_t^{(l)}$, where $A_t^{(l)} \in \mathbb{R}^{r_l \times d_i}$ and $B_t^{(l)} \in \mathbb{R}^{d_o \times r_l}$ are low-rank matrices. The degrees of freedom in LoRA allow $\tau_t^{(l)} B_t^{(l)} A_t^{(l)}$ to introduce new directions in the orthogonal complement of the old task subspace. The training objective is a **purely parameter-space** surrogate loss: $\mathcal{L}(A_t, B_t) = \|\mathcal{T} - (M + \tau_t^{(l)} B_t^{(l)} A_t^{(l)}) \hat{V}\|_F^2$, where $\mathcal{T}$ concatenates the target projections of old and new tasks, and $M$ is the filtered base parameter. This loss simultaneously imposes two directional constraints: consistency with $\hat{V}_{\leq t-1}$ (stability) and tracking of the original model behavior along $\hat{V}_t$ (plasticity).
    - Design Motivation: Critically, this surrogate loss requires no data—Theorem 1 provides a theoretical guarantee that the subspace alignment property ensures the parameter-space projection terms serve as valid upper bounds for data-level losses. The LoRA rank $r_l$ controls the degree of freedom for plasticity: too small a rank leads to insufficient recovery, while too large a rank may reintroduce interference.

3. **Layer-Wise Linear Fusion**:

    - Function: The filter, task vector, and LoRA parameters are merged back into the backbone weights, incurring no additional inference cost.
    - Mechanism: The final update formula is $\theta_t^{\text{merged},(l)} = \theta_{t-1}^{\text{merged},(l)} + \tau_t^{(l)}(P_t^{(l)} + B_t^{(l)} A_t^{(l)})$. Since $\tau_t^{(l)}$, $P_t^{(l)}$, and $B_t^{(l)} A_t^{(l)}$ are all linear operations, their product can be directly computed as a single matrix to be added to the weights. After merging, none of $P$, $A$, or $B$ needs to be retained; the resulting model has identical parameter count and inference cost to a single model.
    - Design Motivation: This is a core advantage of NUFILT over methods such as WEMOE, which retain additional expert modules and thus increase inference-time parameter count. All auxiliary structures in NUFILT are absorbed during merging.

### Loss & Training

**Theoretical Foundation**: Theorem 1 establishes the approximate alignment guarantee between task vectors and data representation subspaces. Defining subspace affinity as $\mathcal{A}(V_d^{(l)}, \hat{V}^{(l)}) = \frac{1}{r_d}\|\hat{V}^\top V_d\|_F^2 \in [0,1]$, experiments across 8 datasets and all layers of ViT-B/16 show that affinity scores for matching task pairs are significantly higher than for non-matching pairs (a dominant diagonal pattern), confirming the universality of the alignment assumption.

**Data-Free Upper Bound**: Corollary 1 translates the data-level stability/plasticity loss upper bound into parameter-space projection terms. When the subspace misalignment $\zeta$ is sufficiently small, $\|\Delta \tau \cdot X^\top\|_F^2$ can be controlled by a constant multiple of $\|\Delta \tau \cdot \hat{V}\|_F^2$, which is the theoretical basis for the surrogate loss serving as a substitute for the true loss.

**Optimization Configuration**: Globally unified hyperparameters are used without task-specific tuning. The null-space rank is $r_p=128$, the LoRA rank is $r_l=64$, and the task projection rank is $r_v=8$. Only 50 steps of Adam optimization (learning rate $10^{-3}$) are required per task, with a total solve time of approximately 18 seconds per task.

## Key Experimental Results

### Main Results (Vision Tasks, averaged over 10 random task orderings)

| Method | Extra Params/Data | ViT-B/32 ACC (8 tasks) | ViT-B/32 ACC (20 tasks) | ViT-L/14 ACC (8 tasks) | ViT-L/14 ACC (20 tasks) |
|------|:---:|:---:|:---:|:---:|:---:|
| Pre-Trained | - | 48.1 | 55.6 | 64.9 | 65.6 |
| Individual Fine-Tuned | - | 90.4 | 89.8 | 94.3 | 93.5 |
| Weight Averaging | ✗/✗ | 66.3 | 61.1 | 80.0 | 71.1 |
| Task Arithmetic | ✗/✗ | 67.5 | 60.0 | 82.1 | 70.3 |
| OPCM | ✗/✗ | 75.5 | 65.7 | 87.0 | 76.0 |
| WUDI-Merging | ✗/✗ | 74.7 | 63.7 | 87.5 | 78.1 |
| Iso-C | ✗/✗ | 71.7 | 67.6 | 86.9 | 80.9 |
| **NUFILT** | **✗/✗** | **83.6** | **71.0** | **91.6** | **84.7** |

NUFILT achieves 83.6% ACC on the ViT-B/32 8-task setting, outperforming OPCM by 8.1% and WUDI-Merging by 8.9%. On ViT-L/14, it trails Individual Fine-Tuning by only 2.7% (91.6% vs. 94.3%). The BWT metric also leads: on ViT-L/14 with 8 tasks, BWT is −1.1% (vs. −2.6% for OPCM), indicating less forgetting. On NLP tasks (Flan-T5 on 8 GLUE tasks), NUFILT achieves 83.7% ACC, surpassing WUDI-Merging by 1.5% and OPCM by 3.1%. On multimodal tasks (LLaVA-1.5-7B, 4 tasks), it achieves an average of 70.5%, outperforming OPCM by 2.9%.

### Ablation Study (ViT-B/32)

| Configuration | Null-Space/LoRA | ACC 8 tasks | ACC 20 tasks | BWT 8 tasks | BWT 20 tasks |
|------|:---:|:---:|:---:|:---:|:---:|
| Naive Merging (direct $\tau_t$) | ✗/✗ | 62.1 | 34.3 | -18.5 | -24.7 |
| Null-Space Only | ✓/✗ | 80.0 | 67.0 | -1.7 | -6.2 |
| LoRA Only (no null-space) | ✗/✓ | 75.8 | 51.7 | -10.2 | -20.6 |
| **NUFILT (Full)** | **✓/✓** | **83.6** | **71.0** | **-2.7** | **-8.9** |

### Key Findings

- **The two components are highly complementary**: Null-space projection alone yields excellent stability (BWT −1.7%) but limited plasticity (ACC 80.0%); LoRA alone offers reasonable plasticity but severe forgetting (BWT −10.2%). Their combination raises ACC by a further 3.6% while incurring only a marginal BWT increase of 1%.
- **Insensitivity to hyperparameters**: ACC varies by less than 2% for $r_p$ in the range 64–256; performance increases monotonically with $r_l$ from 16 to 128 before saturating. Globally unified hyperparameters work well across vision and NLP tasks.
- **Manageable computational overhead**: Each task requires only 18 seconds to solve on ViT-B/32. The additional SVD computation raises total time (139 s) above WUDI-Merging (38 s), but well below LW AdaMerging (724 s), while adding no inference parameters.
- **Scalability with task count**: From 8 → 14 → 20 tasks, NUFILT's advantage over OPCM is largely maintained (8.1 → 6.1 → 5.3% on B/32), indicating that null-space projection degrades gracefully as the number of tasks grows.

## Highlights & Insights

- **The task vector–subspace alignment finding is the central contribution**: This geometric property not only underpins the design of NUFILT but also provides a new theoretical perspective for the model merging field—task vectors are not merely parameter differences; the directional information they encode actually reflects the representational structure of tasks.
- **The "filter then adapt" divide-and-conquer strategy**: Stability is addressed first via projection (in one step, with theoretical guarantees), and plasticity is subsequently compensated via LoRA (through data-free optimization), avoiding the difficulty of requiring a single mechanism to handle both objectives simultaneously. This conservative-then-corrective paradigm can transfer to other settings requiring a balance between conservatism and adaptability.
- **Linear fusion eliminates inference overhead**: All auxiliary structures are absorbed during merging, ensuring that the deployed model is indistinguishable from a single model—a critical design choice for large-scale model merging in practice.

## Limitations & Future Work

- **SVD computational cost**: Each layer of each task requires SVD of the cumulative update; this overhead grows substantially with model depth and parameter dimensionality, particularly for LLMs. Incremental or randomized SVD could be explored to reduce cost.
- **Task order sensitivity**: Although the paper averages over 10 random orderings, the standard deviation of ACC grows with the number of tasks (reaching 0.9% at 20 tasks), and performance may degrade further under adversarial task orderings.
- **Limits of the subspace alignment assumption**: When tasks are highly heterogeneous (e.g., mixed vision–language merging), alignment between task vectors and representation subspaces may no longer hold, weakening the theoretical guarantees of null-space filtering.
- **Cumulative error in continual merging**: The available null-space dimensions shrink as more tasks are added, accelerating plasticity degradation in later stages—ACC drops by 12.6% from 8 to 20 tasks (83.6 → 71.0%), suggesting the method may face capacity saturation in very long task sequences.

## Related Work & Insights

- **vs. OPCM**: OPCM projects each new task vector onto the orthogonal complement of the subspace spanned by all prior task vectors—a "hard orthogonality" strategy. NUFILT instead projects onto the null-space of the **cumulative update** and additionally introduces LoRA adaptation to recover useful components removed by projection. The key distinction is that NUFILT acknowledges inter-task correlation and compensates via LoRA, rather than simply assuming orthogonality.
- **vs. WUDI-Merging**: WUDI also operates without data but relies on statistical estimates of inter-task-vector relationships to adaptively adjust merging weights. NUFILT's advantage lies in its explicit theoretical guarantees (subspace alignment theorem) and substantially better BWT (−2.7% vs. −17.0% at 8 tasks).
- **vs. AlphaEdit**: A concurrent ICLR 2025 work applies null-space constraints for model editing (knowledge editing), sharing a similar intuition but targeting a different setting. NUFILT extends the idea to continual multi-task merging and introduces LoRA adaptation to handle more complex interference patterns.

## Rating

- Novelty: ⭐⭐⭐⭐ The subspace alignment finding is insightful; the method combines null-space projection and LoRA in a theoretically grounded manner
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three modalities (vision/NLP/multimodal), three backbones, three task scales (8/14/20 tasks), 10 random orderings, with comprehensive ablation and hyperparameter analysis
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, motivation is well-articulated, and the overall logic is coherent
- Value: ⭐⭐⭐⭐ Establishes a new state of the art and theoretical framework for DFCMM, advancing the model merging field

## Highlights & Insights

- Solid theoretical contribution: proves approximate alignment between task vectors and representation subspaces
- Elegant method design: null-space projection inherently guarantees zero interference with old tasks
- The projection-aware loss for the LoRA adapter elegantly unifies stability and plasticity into a single objective
- All operations can be linearly fused back into weights, incurring no additional inference cost

## Limitations & Future Work

- The dimensionality of the null-space shrinks as more tasks are added, potentially limiting plasticity for later tasks
- The assumption that subspace alignment holds across all layers may not hold for certain layers with low alignment
- Combination with parameter-efficient fine-tuning methods (e.g., LoRA fine-tuning) remains unexplored
- The rank parameters $r_p, r_l, r_v$ require some empirical tuning

## Related Work & Insights

- Distinction from OPCM: NUFILT additionally introduces LoRA adaptation to recover plasticity, rather than relying solely on orthogonal projection
- Distinction from AdaMerging: fully data-free, without dependence on test-set signals
- The null-space projection idea originates from continual learning (OGD, OWM), but this work is the first to apply it to data-free model merging

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A theory-driven data-free continual merging framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across vision, NLP, and multimodal settings
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and well-organized experiments
- Value: ⭐⭐⭐⭐ Highly valuable for model deployment in privacy-preserving settings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](../../NeurIPS2025/model_compression/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[NeurIPS 2025\] Weight Weaving: Parameter Pooling for Data-Free Model Merging](../../NeurIPS2025/model_compression/weight_weaving_parameter_pooling_for_data-free_model_merging.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)

</div>

<!-- RELATED:END -->
