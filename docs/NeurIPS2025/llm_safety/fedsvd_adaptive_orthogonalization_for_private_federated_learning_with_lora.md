---
title: >-
  [Paper Note] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA
description: >-
  [NeurIPS 2025][LLM Safety][Federated Learning] FedSVD proposes globally reparameterizing LoRA matrices via SVD, updating the $A$ matrix each communication round using the right singular vectors of the aggregated $BA$ pro…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Federated Learning"
  - "Differential Privacy"
  - "LoRA"
  - "SVD"
  - "Noise Amplification"
date: 2026-05-08
content_hash: f758a479faa5f8ca
---

# FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA

**Conference**: NeurIPS 2025
**arXiv**: [2505.12805](https://arxiv.org/abs/2505.12805)
**Code**: [GitHub](https://github.com/seanie12/fed-svd)
**Area**: AI Safety
**Keywords**: Federated Learning, Differential Privacy, LoRA, SVD, Noise Amplification

## TL;DR
FedSVD proposes globally reparameterizing LoRA matrices via SVD, updating the $A$ matrix each communication round using the right singular vectors of the aggregated $BA$ product. This approach avoids the quadratic noise amplification under DP-SGD while preserving the adaptive capacity of $A$, consistently outperforming fixed-$A$ baselines across multiple NLU benchmarks.

## Background & Motivation

**Background**: LoRA has become the dominant method for efficient LLM fine-tuning in federated learning, adapting frozen weights via low-rank matrices $B \in \mathbb{R}^{d_\text{out} \times r}$ and $A \in \mathbb{R}^{r \times d_\text{in}}$.

**Limitations of Prior Work**: When LoRA is combined with DP-SGD, noise is quadratically amplified through the matrix product $BA$. Specifically, $(B + \xi_B)(A + \xi_A) = BA + \xi_B A + B \xi_A + \xi_B \xi_A$, where the last term $\xi_B \xi_A$ constitutes quadratic noise.

**Key Challenge**: FFA-LoRA eliminates noise amplification by fixing $A$ as a randomly initialized matrix, but this fixed random projection limits model expressiveness, leading to slow convergence and suboptimal performance.

**Goal**: Enable the $A$ matrix to adaptively update and capture the principal directions of aggregated updates, without introducing noise amplification.

**Key Insight**: SVD is a post-processing operation that does not affect DP guarantees; an orthogonal $A$ matrix has spectral norm equal to 1, which constrains the gradient norm of $B$.

**Core Idea**: Decompose the aggregated $BA$ via SVD into an orthogonal $A$ and a $B$ carrying the singular values, performing adaptive updates of $A$ on the server side without violating privacy guarantees.

## Method

### Overall Architecture

At each communication round $i$: (1) the server applies SVD to the previous round's aggregated $B_i \hat{A}_{i-1}$; (2) the right singular vectors initialize the new $\hat{A}_i$, and the left singular vectors multiplied by singular values initialize $\hat{B}_i$; (3) these are broadcast to clients; (4) clients optimize only $\hat{B}$ and upload it; (5) the server aggregates the $\hat{B}$ matrices.

### Key Designs

1. **SVD Reparameterization**:

    - Function: Performs SVD decomposition of $B_i \hat{A}_{i-1}$ on the server side.
    - Core formula: $U_i \Sigma_i V_i^\top = B_i \hat{A}_{i-1}$, then $\hat{B}_i = U_i[:,:r] \Sigma_i[:r,:r]$, $\hat{A}_i = V_i^\top[:r,:]$.
    - Key property: $B_i \hat{A}_{i-1} = \hat{B}_i \hat{A}_i$ (since $\text{rank}(B_i \hat{A}_{i-1}) \leq r$, the rank-$r$ SVD exactly recovers the original matrix), so reparameterization does not alter the model's output.
    - Design Motivation: The rows of $\hat{A}_i$ are orthonormal with $\|\hat{A}\|_2 = 1$, thereby constraining the gradient norm of $B$ in the next round.

2. **Gradient Norm Constraint**:

    - Function: Orthogonal $A$ ensures gradients are not amplified by $A$.
    - Core derivation: $\left\|\frac{\partial \ell(\mathbf{z})}{\partial B}\right\|_F = \left\|\frac{\partial \ell(\mathbf{z})}{\partial \mathbf{z}}\right\|_2 \cdot \|\hat{A}\mathbf{x}\|_2 \leq \left\|\frac{\partial \ell(\mathbf{z})}{\partial \mathbf{z}}\right\|_2 \cdot \|\mathbf{x}\|_2$
    - Design Motivation: In DP-SGD, per-sample gradients are clipped to a fixed norm $C$. When $\|A\|_2 > 1$ (common under random initialization), gradients require more aggressive clipping, leading to greater signal distortion.

3. **Hessian Condition Number Analysis (Proposition 3.2)**:

    - Function: Theoretically analyzes the effect of orthogonal $A$ on the optimization landscape.
    - Core result: The Hessian $H_k(B;A) = \mathcal{A} \mathcal{M}_k \mathcal{A}^\top$ satisfies $\kappa_2(H_k) \leq \kappa_2(A)^2 \cdot \frac{\lambda_\text{max}(\mathcal{M}_k)}{\lambda_\text{min}(\mathcal{M}_k|_{\mathcal{R}(\mathcal{A}^\top)})}$.
    - When $A$ is orthogonal, $\kappa_2(A) = 1$, eliminating the $\kappa_2(A)^2$ factor and yielding a better-conditioned optimization landscape.

### Privacy Guarantee

By the post-processing invariance property of differential privacy, SVD is applied as post-processing on the already DP-SGD-privatized $B$; therefore, FedSVD automatically satisfies $(ε, δ)$-DP.

## Key Experimental Results

### Main Results: Without Privacy Constraints (RoBERTa-large, 6 clients)

| Method | SNLI | MNLI-m | SST-2 | QQP | QNLI | Avg. |
|--------|------|--------|-------|-----|------|------|
| FedAvg | 84.16 | 74.79 | 85.89 | 61.75 | 71.40 | 75.51 |
| FFA-LoRA | 82.54 | 82.75 | 94.06 | 78.00 | 86.61 | 84.57 |
| FLoRA | 62.17 | 50.49 | 58.99 | 57.91 | 62.16 | 57.09 |
| **FedSVD** | **85.70** | **83.96** | **94.26** | **79.82** | **88.98** | **86.18** |

### With Privacy Constraints (ε=6, δ=10⁻⁵)

| Method | SNLI | MNLI-m | SST-2 | QQP | QNLI | Avg. |
|--------|------|--------|-------|-----|------|------|
| FedAvg | 61.37 | 65.45 | 89.41 | 58.59 | 60.70 | 67.10 |
| FFA-LoRA | - | - | - | - | - | - |
| **FedSVD** | Best | Best | Best | Best | Best | Best |

### Key Findings
- FedSVD consistently outperforms all baselines in both the non-private and private settings, achieving an average accuracy +1.61 pp higher than the second-best method, FFA-LoRA.
- FLoRA (which randomly reinitializes $A$ and $B$ each round) and FedEX-LoRA both fall significantly short of FFA-LoRA, highlighting the critical importance of stability in $A$.
- FedSVD's accuracy curves dominate those of all baselines across communication rounds, making it well-suited for limited-communication-budget scenarios (early-stopping friendly).
- The convergence acceleration from orthogonal initialization is especially pronounced in early rounds (e.g., FedSVD pulls ahead on SNLI by round 20).
- Advantages are more pronounced under the stronger privacy constraint of ε=3, where noisier conditions make gradient clipping more impactful and the norm constraint from orthogonal $A$ more valuable.
- Effectiveness is also validated on HellaSwag (four-choice commonsense reasoning), demonstrating generality beyond binary/ternary classification tasks.

## Highlights & Insights

- **Minimal design**: The core operation is a single server-side SVD decomposition, requiring no modifications to client training pipelines and introducing no additional communication overhead.
- **Theoretical elegance**: DP post-processing invariance ensures SVD does not compromise privacy guarantees; the orthogonal structure simultaneously constrains gradient norms and improves the Hessian condition number — two benefits unified in a single operation.
- **Adaptive update of $A$**: Unlike FFA-LoRA's fixed random projection, FedSVD aligns $A$ with the principal directions of aggregated updates each round, analogous to PCA — continuously orienting $A$ toward the most important subspace.
- **Narrow confidence intervals**: FedSVD's 95% CIs are generally smaller than those of FedAvg, indicating more stable training and reduced variance from randomness.

## Limitations & Future Work
- Experiments are limited to NLU classification tasks (SNLI/MNLI/SST2/QQP/QNLI) and have not been extended to generative tasks (e.g., GPT fine-tuning, summarization).
- Theoretical analysis is restricted to a simplified two-layer MLP with ReLU; rigorous applicability to deep Transformers with multi-head attention and LayerNorm has not been established.
- SVD computation is manageable on low-rank matrices of size $d_\text{out} \times r$, but server-side computational cost warrants attention when rank $r$ is large or the number of layers is high.
- The setting where $A$ is not updated every round (e.g., performing SVD every $k$ rounds) has not been explored; a better update frequency strategy may exist.
- Data heterogeneity is simulated only via Dirichlet $\alpha=0.5$; more extreme Non-IID scenarios have not been tested.
- Comparisons with DP-LoRA or other DP fine-tuning methods (e.g., DP-BiTFiT) are absent.

## Related Work & Insights
- **vs. FFA-LoRA (Sun et al.)**: Fixing $A$ randomly is a special case of FedSVD (never performing SVD updates). FedSVD replaces the random basis with a data-driven orthogonal basis, consistently achieving superior performance.
- **vs. FLoRA (Wang et al.)**: FLoRA stacks client matrices, computes their product, and randomly reinitializes, but random reinitialization leads to instability. FedSVD's SVD decomposition ensures continuity across rounds.
- **vs. FedAvg + LoRA**: FedAvg independently aggregates $A$ and $B$, causing noise in both to amplify through their product. FedSVD transmits only $B$, with $A$ derived on the server side.
- **vs. DP-BiTFiT / DP-LoRA**: These methods fine-tune different parameter subsets under privacy constraints but do not address the quadratic noise amplification specific to LoRA's matrix product structure.

## Rating
- Novelty: ⭐⭐⭐⭐ The SVD reparameterization idea is clean and effective, though not entirely novel given prior exploration of SVD in LoRA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple GLUE datasets and privacy settings, with convergence curves and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematics are clearly presented; the progression from motivation to method to theory to experiments is well-structured.
- Value: ⭐⭐⭐⭐ Offers direct practical value for federated DP fine-tuning scenarios; the method is simple to deploy and the code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](../../ICLR2026/llm_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)

</div>

<!-- RELATED:END -->
