---
title: >-
  [Paper Note] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation
description: >-
  [CVPR 2026][LLM Evaluation][Model Merging] This paper theoretically proves that fine-tuning weight deltas encode input covariance information, and proposes ACE-Merging…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Model Merging"
  - "Data-Free"
  - "Covariance Estimation"
  - "Spectral Refinement"
  - "Closed-Form Solution"
date: 2026-05-08
content_hash: 4da0f4fcddf1353b
---

# ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation

**Conference**: CVPR 2026
**arXiv**: [2603.02945](https://arxiv.org/abs/2603.02945)
**Code**: N/A
**Area**: LLM Evaluation
**Keywords**: Model Merging, Data-Free, Covariance Estimation, Spectral Refinement, Closed-Form Solution

## TL;DR
This paper theoretically proves that fine-tuning weight deltas encode input covariance information, and proposes ACE-Merging, which achieves data-free closed-form model merging through three steps: adaptive covariance estimation, collective structural prior, and spectral refinement. ACE-Merging achieves an average improvement of 4% over prior methods on GPT-2 and 5% on RoBERTa-Base.

## Background & Motivation

**Background**: The pretrain-then-finetune paradigm produces a large number of task-specific models. Model merging aims to combine multiple expert models into a single unified model, avoiding costly multi-task retraining. Existing approaches fall into three categories: data-dependent (requiring original data), test-time adaptive (with high inference overhead), and data-free (most flexible).

**Limitations of Prior Work**: Data-free methods are the most practically valuable, yet approaches ranging from Task Arithmetic to TIES-Merging rely on heuristic operations in parameter space (sign alignment, pruning, etc.), addressing only the symptoms of interference rather than the root cause — differences in the statistical structure of task data distributions.

**Key Challenge**: The optimal merging formula $\bar{W} = (\sum_t W_t \Sigma_t)(\sum_t \Sigma_t)^{-1}$ requires the input covariance $\Sigma_t$ for each task, which is precisely unavailable in the data-free setting.

**Goal**: To accurately estimate the input covariance for each task without any data access, thereby enabling theoretically grounded optimal model merging.

**Key Insight**: The authors observe that the weight deltas $\Delta W_t$ produced by fine-tuning implicitly encode input covariance information — treating the rows of $\Delta W_t$ as independent samples, their empirical covariance is proportional to $\Sigma_t$.

**Core Idea**: Fine-tuning weight deltas themselves encode input covariance, enabling estimation and construction of a theoretically optimal closed-form merging solution without any data access.

## Method

### Overall Architecture
**Input**: Pretrained model weights $W_0$ and multiple fine-tuned expert weights $\{W_t\}$. **Output**: Merged model $\bar{W}$. The method operates independently per layer in three stages: (1) adaptive covariance normalization, (2) collective structural prior construction, and (3) spectral refinement. The final solution is closed-form, requiring no iterative optimization.

### Key Designs

1. **Estimating Input Covariance from Weight Deltas (Theoretical Core)**:

    - **Function**: Obtain per-task input covariance estimates without data access.
    - **Mechanism**: Theorem 1 proves that $\Sigma_t \propto \text{Cov}_{\mathcal{D}_t}[\Delta W_t]$. Since fine-tuning updates are small, the gradient can be linearized at $W_0$: $\Delta W_t \approx -2\eta N_t \mathbb{E}[(W_0 x - y)x^\top]$, so weight deltas implicitly encode the second-order moment of inputs. In practice, the rows of $\Delta W_t$ are treated as independent samples to compute the empirical covariance $\hat{\Sigma}_t \propto (\Delta W_t - \mathbf{1}\mu_t^\top)^\top (\Delta W_t - \mathbf{1}\mu_t^\top)$.
    - **Design Motivation**: This is the theoretical foundation of ACE-Merging, transforming data-free merging into an optimization problem with an explicit objective. Prior work WUDI-Merging implicitly used a similar proxy $\hat{\Sigma}_t \propto \|\Delta W_t\|_F^{-2} (\Delta W_t)^\top \Delta W_t$, but relied on iterative gradient descent, which is unstable.

2. **Adaptive Covariance Normalization**:

    - **Function**: Balance energy-scale discrepancies across tasks.
    - **Mechanism**: A heterogeneity measure $\gamma = \frac{\text{Var}_t[\log\|\Delta W_t\|_F^2]}{(\mathbb{E}_t[\log\|\Delta W_t\|_F^2])^2}$ detects inter-task scale differences. When $\gamma > \tau$, trace normalization is applied to the covariance: $\hat{\Sigma}_{t,\text{scaled}} = \hat{\Sigma}_t / \text{Tr}(\hat{\Sigma}_t)$, followed by adaptive Tikhonov regularization: $\hat{\Sigma}_{t,\text{reg}} = \hat{\Sigma}_{t,\text{scaled}} + \frac{\epsilon}{\text{Tr}(\hat{\Sigma}_t)} I$.
    - **Design Motivation**: Experiments show that task heterogeneity in RoBERTa ($\gamma > 0.3$) is much higher than in ViT ($\gamma < 0.25$); without normalization, high-energy tasks dominate the merged result. The $\gamma$ threshold serves as a gate to avoid unnecessary normalization for homogeneous tasks.

3. **Collective Structural Prior (CSP)**:

    - **Function**: Introduce anisotropic regularization that captures shared feature geometry across tasks.
    - **Mechanism**: $\mathbf{C}_{\text{agg}} = \mathbf{1} \cdot (\frac{1}{d_{\text{in}}} \mathbf{1}^\top \sum_t \hat{\Sigma}_{t,\text{scaled}})$ broadcasts the column-wise mean of all tasks' scaled covariances to each row, forming a low-rank consensus prior. The final closed-form solution is $\bar{W}_{\text{pre}} = (\sum_t W_t \hat{\Sigma}_{t,\text{reg}})(\sum_t \hat{\Sigma}_{t,\text{reg}} + \mathbf{C}_{\text{agg}})^{-1}$.
    - **Design Motivation**: Standard $\epsilon I$ regularization is isotropic and treats all feature dimensions equally, ignoring the intrinsic geometry of the input space. CSP uses cross-task consensus energy distributions as weights to selectively reinforce shared important dimensions.

4. **Spectral Refinement**:

    - **Function**: Correct severe spectral ill-conditioning in the closed-form solution.
    - **Mechanism**: A structural residual $\Delta_{\text{res}} = \sum_t W_t (\hat{\Sigma}_{t,\text{scaled}} - \bar{\Sigma})$ is computed and fused with $\bar{W}_{\text{pre}}$ before applying SVD. The top-$k$ singular directions are reweighted using their mean singular value $\sigma_{\text{iso}}$: $\Delta W_{\text{refine}} = \sigma_{\text{iso}} \mathbf{U}_{:,1:k} \mathbf{V}_{:,1:k}^\top$, and the final result is $\bar{W} = \bar{W}_{\text{pre}} + \Delta W_{\text{refine}}$.
    - **Design Motivation**: Experiments reveal that the top 5% singular values of $\bar{W}_{\text{pre}}$ account for over 99% of the energy, with condition number exceeding $8.7 \times 10^5$, yet the principal directions are correct (cosine similarity ≈ 1 with the final solution). Hence, it suffices to preserve the directions and redistribute the energy.

### Loss & Training
ACE-Merging is a purely closed-form method with no training or iterative optimization. Hyperparameters are fixed at $\tau=0.3$, $k_{\text{frac}}=0.3$, with $\epsilon$ adjusted per model family (GPT-2: $4\times 10^{-2}$, RoBERTa-Base: $2\times 10^{-4}$, others: $1\times 10^{-5}$).

## Key Experimental Results

### Main Results

**Vision Tasks (ViT-B/16, Mean Absolute Accuracy %)**

| # Tasks | Weight Avg | Task Arithmetic | CART | TSV-M | ACE-Merging | Gain |
|--------|-----------|----------------|------|-------|-------------|------|
| 8 tasks | 72.2 | 75.4 | 88.3 | 89.0 | **90.6** | +1.6 |
| 14 tasks | 69.5 | 70.5 | 84.1 | 84.6 | **86.1** | +1.5 |
| 20 tasks | 65.3 | 65.8 | 80.5 | 80.6 | **82.1** | +1.5 |

**Language Tasks (GPT-2, GLUE Avg %)**

| Method | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | Avg |
|------|------|------|------|------|-----|-----|-------|-----|
| Task Arithmetic | 68.7 | 68.6 | 69.6 | 70.5 | 81.8 | 47.3 | 83.6 | 70.0 |
| TSV-M | 65.6 | 75.4 | 58.6 | 64.4 | 86.2 | 55.6 | 85.7 | 70.2 |
| **ACE-Merging** | 70.3 | 69.9 | 71.8 | 76.7 | 79.0 | 62.5 | 88.5 | **74.1** |

### Ablation Study

| Configuration | RoBERTa-L | GPT-2 | ViT-B/16 (8 tasks) | Note |
|------|-----------|-------|-------------------|------|
| E1: Basic Closed-form | 80.05 | 68.72 | 89.91 | Closed-form only |
| E2: + Adaptive ε | 88.04 | 71.50 | 89.91 | Largest contribution from adaptive regularization |
| E3: + Aggregate Prior | 86.79 | 71.51 | 90.60 | Structural prior assists |
| E4: + Spectral Refinement | **91.68** | **74.09** | **90.60** | Final gain from spectral refinement |

### Key Findings
- Adaptive regularization contributes the most — nearly 8 percentage points improvement from E1 to E2 on RoBERTa-L, indicating that balancing task heterogeneity is the core bottleneck.
- ViT automatically bypasses the adaptive normalization and spectral refinement stages due to low task heterogeneity ($\gamma < 0.3$) (E1≈E2, E3≈E4), validating the rationality of the $\gamma$-gating mechanism.
- Sensitivity analysis shows stable performance within $\gamma \in [0.1, 0.3]$ and $k_{\text{frac}} \in [0.1, 0.5]$; $\epsilon$ is more sensitive.
- ACE-Merging (90.4%) substantially outperforms WUDI-Merging (85.3%) on RoBERTa-Base and maintains a ~3% advantage on RoBERTa-Large.

## Highlights & Insights
- **Theoretically Elegant Insight**: The fundamental obstacle of data-free merging (lack of covariance) is transformed into a quantity directly estimable from weight deltas, establishing a formal connection between "fine-tuning weight deltas ↔ input covariance." This perspective not only explains why ACE-Merging works, but also unifies prior methods — Weight Averaging assumes $\Sigma_t = kI$, while WUDI-Merging implicitly uses $\|\Delta W_t\|_F^{-2} (\Delta W_t)^\top \Delta W_t$ as a proxy.
- **Closed-Form vs. Iterative**: WUDI-Merging requires iterative gradient descent, whereas ACE-Merging is a genuine closed-form solution, offering higher computational efficiency and stability.
- **Insightful Spectral Diagnosis**: The observation that the initial closed-form solution has correct directions but severely imbalanced energy distribution (top 5% singular values account for 99% of energy) leads to a targeted fix — preserving directions while redistributing energy. This "correct direction, wrong magnitude" diagnostic paradigm is transferable to other matrix optimization problems.

## Limitations & Future Work
- $\epsilon$ must be manually tuned per model family (different values for GPT-2, RoBERTa, and ViT); the authors acknowledge that automatic estimation of $\epsilon$ is a direction for future work.
- The linear approximation $f(W,x) \approx Wx$ may be insufficiently precise for deep nonlinear networks, particularly for attention layers.
- The assumption of treating rows of $\Delta W_t$ as independent samples may not hold in practice, as rows of weight matrices exhibit structural correlations.
- Validation is limited to GLUE and visual classification tasks; generative tasks (e.g., LLM dialogue, code generation) are not evaluated.
- The layer-wise independent merging procedure ignores cross-layer dependencies.

## Related Work & Insights
- **vs. WUDI-Merging**: This work reinterprets WUDI as a special case of ACE within a unified theoretical framework (norm-weighted covariance proxy), while ACE replaces WUDI's iterative optimization with a closed-form solution for greater stability and efficiency.
- **vs. TSV-M**: TSV-M decomposes shared/task-specific subspaces via SVD in a heuristic manner; ACE directly models the covariance, providing a more rigorous theoretical foundation.
- **vs. RegMean**: RegMean is a data-dependent method that uses ground-truth covariances directly; ACE demonstrates that comparable performance can be achieved by estimating covariance without data access.
- The paradigm of covariance estimation combined with spectral correction is generalizable to model aggregation problems in federated learning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The theoretical contribution is elegant, though the core insight (weight delta ∝ covariance) is not entirely surprising.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Vision and language benchmarks, multiple architectures and scales, complete ablation and sensitivity analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical progression from theory → unified framework → method → experiments.
- **Value**: ⭐⭐⭐⭐ A practical tool for data-free merging, though the need to manually tune $\epsilon$ reduces out-of-the-box usability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)
- [\[NeurIPS 2025\] HouseLayout3D: A Benchmark and Training-Free Baseline for 3D Layout Estimation in the Wild](../../NeurIPS2025/llm_evaluation/houselayout3d_a_benchmark_and_training-free_baseline_for_3d_layout_estimation_in.md)
- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](../../ICLR2026/llm_evaluation/revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[NeurIPS 2025\] AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners](../../NeurIPS2025/llm_evaluation/adastar_adaptive_data_sampling_for_training_self-taught_reasoners.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)

</div>

<!-- RELATED:END -->
