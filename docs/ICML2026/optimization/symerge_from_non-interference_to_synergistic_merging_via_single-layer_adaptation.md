---
title: >-
  [Paper Note] SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation
description: >-
  [ICML 2026][Optimization][Model Merging] This paper redefines the goal of "model merging" from "avoiding task interference" to "promoting task synergy." It proposes SyMerge…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Model Merging"
  - "Task Synergy"
  - "Test-Time Adaptation"
  - "Single-Layer Adaptation"
  - "Expert Self-Labeling"
date: 2026-05-08
content_hash: ffd1481e1014ac7e
---

# SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation

**Conference**: ICML 2026  
**arXiv**: [2412.19098](https://arxiv.org/abs/2412.19098)  
**Code**: https://aim-skku.github.io/SyMerge (Yes)  
**Area**: Model Compression / Model Merging  
**Keywords**: Model Merging, Task Synergy, Test-Time Adaptation, Single-Layer Adaptation, Expert Self-Labeling

## TL;DR
This paper redefines the goal of "model merging" from "avoiding task interference" to "promoting task synergy." It proposes SyMerge, which jointly optimizes one task-specific layer for each task and the layer-wise merging coefficients of the encoder. By using fine-tuned expert models as soft-label teachers to prevent entropy minimization drift during test-time, SyMerge elevates merged models to levels near the single-task upper bound across vision, dense prediction, and NLP benchmarks.

## Background & Motivation
**Background**: The model merging paradigm directly combines multiple independently fine-tuned models of the same architecture in the parameter space. By reusing task vectors $\tau_k = \Theta_k - \Theta_{\text{pre}}$, a multi-task model is obtained without the cost of joint training. Mainstream approaches fall into two categories: training-free methods (Task Arithmetic, Ties-Merging, PCB, Consensus, etc.), which rely on heuristics or grid search for coefficients, and test-time adaptation methods (AdaMerging, WEMoE, Surgery, etc.), which use unlabeled test data to learn merging coefficients or post-hoc adapters via proxy objectives like entropy minimization.

**Limitations of Prior Work**: The authors find, by testing 4 tasks under Hendrycks' corruption standards, that training-free methods collapse under slight distribution shifts. While test-time methods are more robust, they still treat "avoiding interference" as the sole objective—efforts in SVD truncation, parameter masking, and weight disentanglement all aim to ensure $\tau_i$ does not damage task $j$, essentially pursuing $L_j[f(x;\theta_0+\tau_i)] = L_j[f(x;\theta_0)]$.

**Key Challenge**: The non-interference objective inherently has a ceiling—the merged model's performance on task $j$ can at most match the pretrained model, as there is no mechanism for other tasks to "help." Additionally, a pilot experiment on 20 vision tasks revealed that cross-task performance (using task A's encoder with task B's classifier) and post-merge performance are strongly positively correlated ($r=0.863, p<0.001$ on ViT-B/32). This indicates the true bottleneck for merging quality is functional alignment between different task encoders/predictors, not interference.

**Goal**: Upgrade the objective from non-interference to positive synergy: $L_j[f(x;\theta_0+\tau_i)] < L_j[f(x;\theta_0)]$; find a minimal-cost method to improve cross-task alignment that works stably in unlabeled test scenarios.

**Key Insight**: The authors conducted a second pilot: retraining the last layer of task $k$ on a fixed merged encoder using labeled data. When this new classifier was attached to the encoder of task $m\neq k$, all 8 tasks showed significant improvements in cross-task accuracy. This suggests that tuning just one layer (even an intermediate block) is sufficient to align the encoder and predictor functions across different tasks.

**Core Idea**: Translate the "tuning one layer" finding to unlabeled test-time scenarios. Jointly optimize layer-wise merging coefficients $\{\lambda_k^l\}$ and one task-specific layer $\theta_k^{\text{tr}}$ per task. Replace entropy minimization with more stable expert-guided self-labeling cross-entropy using pre-fine-tuned experts as "soft-label teachers."

## Method

### Overall Architecture
Input: Pretrained weights $\Theta_{\text{pre}}$, $K$ independently fine-tuned task experts $\{\Theta_k\}_{k=1}^K$, and unlabeled test sets $\mathcal{X}_k^{te}$ for each task. Output: A shared encoder $\Theta_{\text{MTL}}^{\text{enc}}$ and $K$ task heads. The pipeline involves three steps: (1) Formulate each encoder layer's weights as $\theta_{\text{MTL}}^l = \theta_{\text{pre}}^l + \sum_k \lambda_k^l \tau_k^l$, where $\Lambda = \{\lambda_k^l\}$ is a learnable layer-wise task coefficient matrix; (2) Select one task-specific adaptation layer $\theta_k^{\text{tr}}$ for each task, initialized with the corresponding layer from the task expert; (3) Jointly optimize $\Lambda$ and $\{\theta_k^{\text{tr}}\}$ such that the merged model's predictions on $\mathcal{X}_k^{te}$ approximate those of the expert models. All other layers remain frozen.

### Key Designs

1.  **Expert-Guided Self-Labeling (Replacement for Entropy Minimization)**:
    *   **Function**: Provides a stable supervision signal during unlabeled test-time, forcing the merged model to mimic expert behavior.
    *   **Mechanism**: For each test sample $x$ of task $k$, the output of the fine-tuned expert $C_k^{\text{ft}}(x)$ (softmax probability vector for classification, continuous output for regression) serves as a soft label. The merged model output $C_k^{\text{merged}}(x)$ is matched to it by minimizing $\sum_k \mathcal{L}_{CE}(C_k^{\text{merged}}, C_k^{\text{ft}})$. For dense prediction tasks, cross-entropy is replaced by L1 loss, making it naturally compatible with tasks like segmentation and depth estimation.
    *   **Design Motivation**: Using Spearman correlation to measure consistency between "proxy loss" and true supervised cross-entropy, the authors found entropy minimization drifts significantly after training (even reversing on the Cars dataset), whereas expert self-labeling remains highly consistent. Since expert models are already SOTA on their tasks, using them as teachers is far more reliable than pseudo-labeling.

2.  **Single-Layer Adaptation + Joint Coefficient Optimization (Minimal Parameters)**:
    *   **Function**: Achieves cross-task alignment by modifying the encoder mixture and the output end of each task with minimal parameters.
    *   **Mechanism**: Only one layer $\theta_k^{\text{tr}}$ (e.g., the classification head or the last transformer block) is unfrozen per task, along with the shared encoder's layer-wise merging coefficients $\Lambda$. Both are updated via the same self-labeling loss. Unlike AdaMerging, which only learns $\Lambda$ and can collapse under disjoint basin initializations, unfreezing an adaptation layer stabilizes training. Unlike Surgery/ProbSurgery, SyMerge introduces no extra modules, only unfreezing an existing layer.
    *   **Design Motivation**: Based on the cross-task pilot, tuning one layer is sufficient for functional alignment. Joint optimization allows the task-specific layer to compensate for unfavorable mixtures at the encoder level, creating "synergistic" encoder-predictor updates.

3.  **Optional Confidence Filtering + Cross-Initialization Compatibility**:
    *   **Function**: Filters low-confidence samples and extends merging to experts from different initializations.
    *   **Mechanism**: Samples in a batch are sorted by the max softmax probability from the expert; only the top-$p$ high-confidence samples are used for backpropagation. For disjoint-basin settings (experts from different pretrain initializations), where traditional searches fail, SyMerge maintains usability due to its adaptation layer.
    *   **Design Motivation**: Confidence filtering is a standard low-cost technique in test-time learning. Supporting disjoint-basin experts is critical for real-world scenarios where open-source fine-tuned models may not share the same pretrain.

### Loss & Training
The unified objective is $\min_{\{\lambda_k^l\}, \{\theta_k^{\text{tr}}\}} \sum_{k=1}^K \mathcal{L}_k(C_k^{\text{merged}}, C_k^{\text{ft}})$, using cross-entropy for classification and L1 for regression. The optimizer and learning rate follow AdaMerging settings, with $\Lambda$ initialized uniformly and $\theta_k^{\text{tr}}$ initialized with the expert's corresponding layer. Results are reported as mean ± standard deviation over 5 random seeds.

## Key Experimental Results

### Main Results
Covering three categories: Vision Classification (ViT-B/32 and ViT-L/14 with 8/14/20 tasks), Dense Prediction (NYUv2 with ResNet-50 for segmentation/depth/normals), and NLP (RoBERTa for GLUE 8 tasks).

| Benchmark Setting | Metric | AdaMerging | EMR-Merging | ProbSurgery | SyMerge | Individual Limit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ViT-B/32 / 8 tasks | Avg Acc | 80.1 | 88.7 | 87.4 | **90.1 ±0.1** | 90.5 |
| ViT-B/32 / 20 tasks | Avg Acc | 69.6 | 86.6 | 84.5 | **88.6 ±0.4** | 90.4 |
| ViT-L/14 / 20 tasks | Avg Acc | 82.1 | 92.0 | 90.2 | **93.2 ±0.1** | 94.0 |
| NYUv2 / Seg | mIoU↑ | — | 41.5 | 43.6 | **49.8 ±0.3** | 52.0 |
| GLUE / 8 tasks | Average | — | 80.2 | 81.6 | **83.9 ±0.2** | 85.6 |

In the 20-task setting on ViT-B/32, SyMerge is only 1.8 points behind the single-task upper bound, outperforming EMR-Merging (86.6) by 2 points. For dense prediction, it narrows 70% of the gap to the upper bound. On GLUE, SyMerge significantly closes the gap to individual models, reaching parity or even surpassing them on sub-tasks like CoLA and RTE.

### Ablation Study

| Configuration | ViT-B/32 8-task Avg | Description |
| :--- | :--- | :--- |
| Task Arithmetic (Baseline) | 69.1 | Training-free, fixed $\lambda$ |
| AdaMerging (Learn $\Lambda$ only) | 80.1 | Coeff-only + entropy minimization |
| Learn $\Lambda$ + Self-labeling | ~85 | Replacing entropy with soft labels |
| Learn $\theta_k^{\text{tr}}$ only | ~83 | Adapting layer only, fixed $\Lambda$ |
| **SyMerge (Joint + Self-labeling)** | **90.1** | Full proposed scheme |
| + Confidence Filtering | 90.1+ | Optional; adds approx. 0.2-0.5 gain |

### Key Findings
*   Replacing entropy minimization with expert self-labeling alone boosts performance from ~80 to ~85, proving supervision stability is more critical than coefficient precision.
*   Tuning the adaptation layer alone reaches ~83, but joint optimization is required to break the 90 barrier, validating the presence of "synergy."
*   In disjoint-basin settings, SyMerge remains functional while coefficient-only methods like AdaMerging drop below 30% accuracy.
*   The $r=0.863$ cross-task correlation suggests cross-task accuracy can serve as a cheap proxy for evaluating merging methods in the future.

## Highlights & Insights
*   Redefining the objective from "non-interference" to "synergy" is a clean conceptual upgrade. While prior works tried to keep $\tau_i$ from doing harm, this work proves that "not doing harm" has a ceiling and active alignment is necessary.
*   Using expert models as teachers is an overlooked but simple option for test-time adaptation. Since fine-tuned experts are already available, this paper demonstrates that they provide far more stable self-labeling than entropy.
*   The "single layer is enough" finding is counter-intuitive and minimalist. Unlike Surgery/WEMoE, which add extra adapters, SyMerge shows that just unfreezing an existing task head or block is sufficient.

## Limitations & Future Work
*   The method requires fine-tuned experts to remain available, which increases memory overhead as the number of tasks $K$ grows.
*   The selection of which layer to adapt is still a hyperparameter (defaulting to the last block), and no automated strategy is provided.
*   Experiments focused on ViT, ResNet, and RoBERTa. Its effectiveness on large-scale LLM merging (>7B) remains to be verified.
*   "Synergy" is primarily defined empirically. While Proposition 3.1 provides arguments regarding a tighter convex bound, a formal theoretical proof of "positive inter-task transfer" is still needed.

## Related Work & Insights
*   **vs AdaMerging**: AdaMerging optimizes $\Lambda$ via entropy. SyMerge adds layer-wise adaptation and replaces entropy with expert guidance, yielding significant gains (5–10 points) over it as a strict superset.
*   **vs Surgery / ProbSurgery**: These methods add post-hoc adapter networks. SyMerge is more parameter-efficient, introducing no new modules and achieving better results by unfreezing existing layers.
*   **vs Non-interference paths (Ties-Merging, ISO-Merging)**: While those works focus on making $\tau$ orthogonal, SyMerge shows that "active alignment" through single-layer tuning has a higher performance ceiling than "passive conflict avoidance."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantitative Convergence of Trained Single Layer Neural Networks to Gaussian Processes](../../NeurIPS2025/optimization/quantitative_convergence_of_trained_single_layer_neural_networks_to_gaussian_pro.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICML 2026\] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](../../ICLR2026/optimization/test-time_meta-adaptation_with_self-synthesis.md)

</div>

<!-- RELATED:END -->
