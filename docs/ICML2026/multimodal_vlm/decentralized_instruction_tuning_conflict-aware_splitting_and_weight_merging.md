---
title: >-
  [Paper Note] Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging
description: >-
  [ICML2026][Multimodal VLM][Instruction Tuning] This work develops a local quadratic theory for weight merging based on the "merge-ready flat basin" concept: the merging gain equals the curvature-weighted checkpoint varia…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Instruction Tuning"
  - "Weight Merging"
  - "Gradient Conflict"
  - "PCA Splitting"
  - "Multimodal Alignment"
date: 2026-05-08
content_hash: 1be89ba416c72321
---

# Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging

**Conference**: ICML2026  
**arXiv**: [2606.01717](https://arxiv.org/abs/2606.01717)  
**Code**: https://github.com/naver-ai/merit  
**Area**: Multimodal VLM / Model Merging / Distributed Training  
**Keywords**: Instruction Tuning, Weight Merging, Gradient Conflict, PCA Splitting, Multimodal Alignment  

## TL;DR
This work develops a local quadratic theory for weight merging based on the "merge-ready flat basin" concept: the merging gain equals the curvature-weighted checkpoint variance. PCA splitting along the primary directions of gradient conflict maximizes this gain. Based on this, the MERIT pipeline is proposed—utilizing PCA splitting by dataset gradient conflict, independent fine-tuning across branches with zero communication, and final one-shot token-weighted averaging. It improves the 8-benchmark average from 54.3 to 57.0 on Qwen2.5-VL-3B with 136 Vision-FLAN tasks.

## Background & Motivation

**Background**: The capabilities of modern (M)LLMs are primarily infused via large-scale instruction tuning. Datasets like Vision-FLAN, TÜLU, and FLAN often contain hundreds of tasks and millions of samples. The dominant approach is centralized joint training, mixing all tasks and running on tightly coupled GPU clusters.

**Limitations of Prior Work**: This "joint" paradigm is constrained by two bottlenecks: (1) **Optimization**: Heterogeneous tasks conflict on shared parameters, where gradient conflicts lead to negative transfer and rigid dynamics, forcing smaller learning rates. Classic multi-task corrections (GradNorm / PCGrad / CAGrad) are computationally infeasible at the scale of hundreds of tasks and billions of parameters. (2) **Systems**: Joint training relies on frequent synchronization (e.g., all-reduce), requiring all GPUs to be in a single high-bandwidth cluster. This precludes the use of geographically distributed resources, heterogeneous pools, or cloud preemptible instances.

**Key Challenge**: These two factors are strongly coupled—higher data heterogeneity requires finer-grained synchronization to counter conflicts, yet synchronization is the systemic bottleneck. Without it, one must revert to crude "proportional mixing," which fails to alleviate conflict.

**Goal**: Can the "mixed training" problem be shifted from online (gradient alignment) to offline (parameter space averaging)? Specifically, by splitting tasks according to conflict, training them independently, and performing a final one-shot merge to eliminate conflict without requiring synchronization.

**Key Insight**: Prior works like Model Soup or Model Stock indicate that if fine-tuning starts from the same flat basin, the average of multiple independently trained checkpoints often outperforms any single checkpoint. This "merge-ready" property is common in post-training (e.g., continuing SFT from an instruction-tuned MLLM). By theoretically clarifying what kind of splitting maximizes merging gains, this empirical trick can be upgraded into a scheduling algorithm.

**Core Idea**: Under the local quadratic approximation of merge-ready initialization, it is proven that the gain from weight averaging $\mathcal{G}_{\mathrm{var}}=\tfrac{1}{2}\sum_\ell \lambda_\ell \mathrm{Var}_w(u_\ell^\top \delta_i)$ is the **curvature-weighted checkpoint variance**. Gain is maximized by injecting variance into high-curvature directions. Furthermore, it is shown that performing PCA on dataset gradients and splitting into $K=2^r$ groups along the top-$r$ principal axes is an approximately optimal allocation to maximize this gain, followed by token-weighted one-shot merging.

## Method

### Overall Architecture

MERIT reshapes instruction tuning from "centralized" to "distributed + one-shot merge." The pipeline consists of five steps on a merge-ready initialization $\theta^{(0)}$: (1) For $T$ datasets, calculate a representative gradient $g_t$ using a small calibration set of 200 samples each; (2) Construct a cosine similarity matrix $C_{ij}=\langle\tilde g_i,\tilde g_j\rangle$ and perform PCA to obtain an $r$-dimensional conflict embedding $z_t$ for each dataset; (3) Recursively perform sample-balanced 50/50 median splits along $r$ PCA axes to obtain $K=2^r$ groups; (4) Fine-tune each group independently from $\theta^{(0)}$ with **zero inter-branch communication**, allowing distribution across isolated GPU pools; (5) Perform a one-shot weighted average $\bar\theta$ based on token budgets $w_k=N_k/\sum N_j$. This process trades training-time synchronization costs for a small-scale gradient estimation before training and a single parameter averaging step after.

### Key Designs

1.  **Merging Gain Theorem in Flat Basins (Theoretical Foundation)**:
    *   **Function**: Quantitatively determines when weight averaging is beneficial and the magnitude of the gain.
    *   **Mechanism**: A quadratic approximation of the loss is made at the shared initialization $\theta^{(0)}$: $L(\theta)\approx L(\theta^\star)+\tfrac{1}{2}(\theta-\theta^\star)^\top H(\theta-\theta^\star)$, where $H\succeq 0$ is the local Hessian. Given the displacements of $K$ fine-tuned checkpoints $\delta_i=\theta_i-\theta^\star$ and weights $w_i\ge 0$ summing to 1, then $\mathcal{G}_{\mathrm{var}}:=\sum_i w_i L(\theta_i)-L(\bar\theta_w)=\tfrac{1}{2}\sum_\ell \lambda_\ell \mathrm{Var}_w(u_\ell^\top \delta_i)\ge 0$, where $\lambda_\ell, u_\ell$ are eigenpairs of $H$. The gain is non-negative, and it stems from "checkpoint variance projected onto high-curvature directions."
    *   **Design Motivation**: While previous explanations for Model Soup were empirical, this formula dictates exactly which directions variance should be injected into, naturally leading to the optimality of PCA splitting.

2.  **PCA Splitting Along Dataset Gradient Conflict Axes**:
    *   **Function**: Transforms the grouping of datasets from engineering intuition into an algorithm seeking $\arg\max\mathcal{G}_{\mathrm{var}}$ along high-curvature Hessian directions.
    *   **Mechanism**: A first-order approximation $\delta_k\approx -\eta\bar g_k$ is used at $\theta^{(0)}$. In a two-group case, $\mathcal{G}_{\mathrm{var}}=\tfrac{\eta^2}{8}(\bar g_1-\bar g_2)^\top H(\bar g_1-\bar g_2)$. Since $g_t=-H\Delta_t$ (where $\Delta_t$ is the local optimal shift for dataset $t$), the gain is dominated by $H^3$-weighted dataset interactions. PCA identifies directions of "high curvature + high disagreement." Implementation uses cosine PCA on normalized gradients (scale-invariant), taking top-$r$ embeddings $z_t\in\mathbb{R}^r$ and recursively splitting by the median to ensure **sample-balanced** groups.
    *   **Design Motivation**: Random splitting or K-means do not directly maximize $\mathcal{G}_{\mathrm{var}}$, and per-step methods like PCGrad require synchronization. Offline PCA at $\theta^{(0)}$ costs $O(T^2)$ and allows incremental updates for new datasets with $O(Tm)$ cost.

3.  **Token-Weighted One-shot Merging + Implicit Norm Regularization**:
    *   **Function**: Merges $K$ independently trained branches into a single deployable model with added generalization regularization.
    *   **Mechanism**: $\bar\theta=\sum_{k=1}^K w_k \theta_k$ where $w_k=N_k/\sum_j N_j$ and $N_k$ is the total token budget of group $k$. This maintains the same total budget as joint training. By the convexity of the norm, $\|\bar\theta_w-\theta^{(0)}\|^2\le\sum_i w_i\|\theta_i-\theta^{(0)}\|^2$, meaning the merged model is closer to the initialization, equivalent to a PAC-Bayes distance regularization. Furthermore, it acts as spectral filtering by clearing displacement errors in high-curvature directions.
    *   **Design Motivation**: Explains the counter-intuitive phenomenon where the training loss of the merged model is higher, yet its generalization is superior.

### Loss & Training

Each branch shares the same backbone, trainable parameters, LR schedule, and token budget $n_t$. The only difference is the data subset seen. $\theta^{(0)}$ is an instruction-tuned Qwen2.5-VL for multimodal experiments or a pretrained LLM for text-only experiments. Merge-ready properties are verified via four diagnostics: (a) loss barriers of zero along linear interpolation paths between branches; (b) the merged model distance to $\theta^{(0)}$ being 2.4–2.9x smaller than the joint baseline; (c) higher training loss but better held-out performance; (d) smaller loss increases under isotropic Gaussian perturbations.

## Key Experimental Results

### Main Results

Controlled experiments on Qwen2.5-VL-3B + Vision-FLAN (136 tasks) across 8 benchmarks:

| Method | SeedBench | MMBench | LLaVA-W | MMVet | TextVQA | AI2D | MathVista | MMMU | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base 3B | 66.8 | 79.7 | 53.2 | 34.0 | 61.2 | 63.8 | 29.6 | 41.2 | 53.7 |
| Joint training (1 ep) | 69.2 | 80.5 | 41.9 | 36.4 | 68.0 | 62.6 | 34.2 | 41.9 | 54.3 |
| Joint training (2 ep) | 70.0 | 81.4 | 42.8 | 37.6 | 63.4 | 62.5 | 36.5 | 43.0 | 54.7 |
| Random (4 groups) | 70.4 | 81.0 | 40.6 | 34.7 | 70.4 | 63.1 | 34.0 | 40.8 | 54.4 |
| Uniform soup (4 runs) | 70.2 | 81.1 | 41.8 | 36.3 | 68.4 | 63.4 | 35.9 | 42.2 | 54.9 |
| MERIT-1D (K=2) | 71.0 | 80.0 | 43.1 | 35.0 | 72.4 | 62.1 | 36.5 | 41.4 | 55.2 |
| MERIT-2D (K=4) | 70.8 | 78.4 | 47.4 | 36.6 | 74.1 | 61.5 | 36.0 | 40.7 | 55.7 |
| **MERIT-3D (K=8)** | 70.5 | 80.1 | **52.0** | **37.7** | **75.2** | 62.5 | 35.4 | 42.7 | **57.0** |

Higher dimensions of MERIT (more $K$) consistently yield better results, increasing the joint average from 54.3 to 57.0 (+2.7). Significant gains appear in LLaVA-W (+10.1) and TextVQA (+7.2), confirming that conflict-aware splitting successfully mitigates negative transfer during joint training.

### Ablation Study

Merge-readiness diagnostics for Qwen2.5-VL-3B / MERIT-2D / K=4:

| Epoch | Joint Displ. | Merged Displ. | Ratio | Joint train loss | Merged train loss | Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.5 | 13.73 | 5.65 | 2.43× | 0.709 | 1.198 | +0.489 |
| 1.0 | 19.73 | 7.50 | 2.63× | 0.560 | 1.172 | +0.611 |
| 2.0 | 28.15 | 10.11 | 2.78× | 0.370 | 1.167 | +0.797 |
| 6.0 | 34.61 | 11.87 | 2.92× | 0.064 | 1.330 | +1.266 |

### Key Findings

*   **Conflict-aware strictly outperforms random**: At K=2, conflict-induced split (54.9 avg) beats random split (54.6) and joint (54.3), validating that PCA axes capture appropriate splitting directions.
*   **Uniform soup helps but lacks MERIT's efficiency**: Averaging different seeds (Uniform soup 2) reaches 55.4, but MERIT-3D achieves 57.0 with the same budget, proving the splitting strategy is more valuable than redundant training.
*   **Inverse relation between norm and generalization**: The merged model consistently maintains higher training loss but better generalization and a closer distance to $\theta^{(0)}$.
*   **Scaling to 1.6M samples / 176 sources / 7B**: MERIT-2D improves upon Joint FFT (54.9 to 55.4) and holds at parity or better on 3.6M scales (60.9); similar results hold for text-only FLAN.
*   **Negligible preprocessing cost**: Using 200 calibration samples per dataset and sampling one gradient component per 5 parameters retains >0.98 correlation with full-gradient baselines.

## Highlights & Insights

*   **Upgrading Model Soup to an Algorithmic Objective**: Unlike previous merging works that are post-hoc, this method uses PCA to direct the split before training—"designing training for merging."
*   **System Value of Zero Communication**: Branches can run on geographically isolated cloud regions or heterogeneous hardware. As long as $\theta^{(0)}$ is distributed, the SFT process requires no inter-node communication, enabling large-scale tuning for teams with fragmented resources.
*   **PCA Splitting as an Interpretability Tool**: The embedding $z_t\in\mathbb{R}^r$ provides coordinates for datasets in "conflict space," facilitating data curation based on theoretical conflict rather than manual task classification.

## Limitations & Future Work

*   The theory assumes a "merge-ready flat basin," confirmed for Qwen2.5-VL but likely invalid for from-scratch pretraining. MERIT is a post-training specific method.
*   The $K=2^r$ structure is somewhat rigid; actual conflict structures might be simplex-like (e.g., three mutually opposing datasets), which recursive bisection might not perfectly capture.
*   Dependence on the "gradient-norm concentration" for Cosine PCA might introduce bias for datasets with extremely imbalanced gradient scales.
*   The transferability to other architectures (LLaMA, Gemma) or LoRA-based versions remains to be explored.

## Related Work & Insights

*   **vs Model Soup / Model Stock**: They merge checkpoints from different seeds on the same dataset to reduce variance; MERIT merges checkpoints from different subsets to actively leverage conflict-induced diversity.
*   **vs PCGrad / GradNorm / CAGrad**: Traditional surgery aligns gradients online, which is infeasible at scale; MERIT moves conflict handling to an offline pre-training phase.
*   **vs Federated Learning (FedAvg)**: In FL, data split is dictated by ownership; MERIT treats the split as an optimization variable.
*   **vs Data Mixture Ratio Tuning**: MERIT provides a complementary primitive to ratio tuning by allowing "split training" alongside mixture adjustments.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ Defining merging gain as curvature-weighted variance and deriving the optimality of PCA splitting is a significant step for model merging theory.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across scales (3B/7B), various mixtures (Vision-FLAN/1.6M), and rigorous merge-readiness diagnostics.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Clear correspondence between theory and algorithm, supported by empirical validation and formal proofs in the appendix.
*   **Value**: ⭐⭐⭐⭐⭐ Provides an immediately deployable solution for instruction tuning with heterogeneous data and fragmented resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)
- [\[ICML 2026\] SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](same_stabilized_mixture-of-experts_for_multimodal_continual_instruction_tuning.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)

</div>

<!-- RELATED:END -->
