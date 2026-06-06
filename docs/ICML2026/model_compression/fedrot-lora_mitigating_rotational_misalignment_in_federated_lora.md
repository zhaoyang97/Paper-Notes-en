---
title: >-
  [Paper Note] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA
description: >-
  [ICML 2026][Model Compression][Federated Learning] This paper identifies that the true "enemy" of naive factor-wise averaging in Federated LoRA is the potential subspace misalignment caused by rotational invariance. It p…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Federated Learning"
  - "LoRA"
  - "Procrustes Alignment"
  - "Rotational Invariance"
  - "Subspace Alignment"
date: 2026-05-08
content_hash: 9fb09f6fc561b70e
---

# FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA

**Conference**: ICML 2026  
**arXiv**: [2602.23638](https://arxiv.org/abs/2602.23638)  
**Code**: https://github.com/haoran-zh/FedRot-LoRA (Available)  
**Area**: Federated Learning / Parameter-Efficient Fine-Tuning / LoRA  
**Keywords**: Federated Learning, LoRA, Procrustes Alignment, Rotational Invariance, Subspace Alignment

## TL;DR
This paper identifies that the true "enemy" of naive factor-wise averaging in Federated LoRA is the potential subspace misalignment caused by rotational invariance. It proposes FedRot-LoRA, which solves for a rotation matrix $R_i^t$ using orthogonal Procrustes on the client side to align $A$ and $B$ factors before aggregation. Both theoretical and experimental results demonstrate that this significantly reduces aggregation error without increasing communication overhead.

## Background & Motivation

**Background**: LoRA represents weight updates as $\Delta W = BA$, where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$. Due to the significant reduction in parameter count, it has become the most natural vehicle for federated fine-tuning of LLMs (e.g., FedIT, FFA-LoRA, and FlexLoRA are based on this).

**Limitations of Prior Work**: Ideally, aggregation should be $\Delta W_{ideal} = \frac{1}{N} \sum B_i A_i$, but this typically results in a rank $> r$, making it impossible to maintain low-rank properties. The alternative, factor-wise averaging $\Delta W_{naive} = \bar{B} \bar{A}$, introduces unwanted cross-terms $B_i A_j$, leading to training instability. Existing solutions follow three paths: freezing parameters (FFA-LoRA, RoLoRA) which sacrifices expressivity; SVD projection (FlexLoRA) which is computationally expensive; or transmitting high-communication residuals (FedEx-LoRA) which contradicts the original goal of LoRA.

**Key Challenge**: Existing analyses focus only on the algebraic reason of "operator non-commutativity," ignoring the **rotational invariance** inherent in LoRA factorization—for any orthogonal $R \in \mathbb{R}^{r \times r}$, $(B_i R)(R^\top A_i) = B_i A_i$. This implies that semantically equivalent updates can be represented in different latent subspaces. During naive averaging, these misaligned subspaces cause "destructive interference," leading to errors larger than those caused by algebraic terms alone.

**Goal**: To explicitly eliminate the underestimated factor of "rotational error" without increasing communication, freezing parameters, or performing high-dimensional SVD.

**Key Insight**: Since the rotation $R$ does not change the semantics, one can actively select a rotation $R_i^t$ that aligns all client factors to a common reference. This $R$ has $r(r-1)/2$ degrees of freedom on the orthogonal group, which is sufficient to align subspaces and can be efficiently solved via the closed-form orthogonal Procrustes solution.

**Core Idea**: Use the global $\bar{A}^{t-1} / \bar{B}^{t-1}$ from the previous round as a reference. Each client solves a Procrustes problem to obtain a rotation $R_i^t$, then alternately aligns $A$ or $B$. Finally, "soft rotation" is applied via interpolation to avoid excessive noise from early-stage references.

## Method

### Overall Architecture
At each communication round $t$: ① The server broadcasts $(\bar{A}^{t-1}, \bar{B}^{t-1})$; ② Clients perform local training to obtain $(A_i^t, B_i^t)$; ③ If $t$ is odd, align $A$; if $t$ is even, align $B$ by solving the Procrustes problem for $R_i^{t,*}$; ④ Interpolate between the identity matrix and $R_i^{t,*}$ with $\lambda$ to obtain the soft rotation $R_{i,\text{soft}}^t$; ⑤ Apply $\tilde{A}_i^t = (R_{i,\text{soft}}^t)^\top A_i^t$ and $\tilde{B}_i^t = B_i^t R_{i,\text{soft}}^t$ before uploading for aggregation.

### Key Designs

1.  **Orthogonal Procrustes Alignment (Alternating Factor Alignment)**:
    - **Function**: Rotates the LoRA factors of each client into the subspace of the global reference, eliminating rotational ambiguity in factorization.
    - **Mechanism**: In odd rounds, solve $\min_{R} \|R^\top A_i^t - A_{ref}\|_F^2, \text{ s.t. } R^\top R = I, \det R > 0$. This is a classic Procrustes problem. After performing SVD on the correlation matrix $M = A_{ref}(A_i^t)^\top$ as $M = U\Sigma V^\top$, the closed-form solution is $R_i^{t,*} = V \cdot \text{diag}(1, \dots, 1, \det(UV^\top)) \cdot U^\top$. In even rounds, the alignment switches to $B$ with $M = (B_{ref})^\top B_i^t$. The computational complexity is $\mathcal{O}(dr^2 + r^3)$, significantly smaller than FlexLoRA's $\mathcal{O}(d^3)$.
    - **Design Motivation**: Theorem 4.1 proves that scalar scaling has only 1 degree of freedom and cannot eliminate subspace misalignment, while unconstrained invertible matrices can become ill-conditioned. Orthogonal matrices strike the optimal balance—flexible enough ($r(r-1)/2$ degrees of freedom) while remaining well-conditioned.

2.  **Alternating Alignment (Alternating $A$ and $B$)**:
    - **Function**: Prevents uncontrollable drift on one side when repeatedly aligning the same factor.
    - **Mechanism**: In odd rounds, the semantics of $A$ are fixed to $A_{ref}$ while $B$ compensates; the reverse occurs in even rounds. Only one SVD is solved per round, but globally both factors are "calibrated" in turn.
    - **Design Motivation**: Ablations show that performance drops significantly if only $B$ is aligned (SST-2: 0.879 vs 0.954). This is because the initial norm of $B$ is small, leading to weak alignment signals in early stages. The alternating strategy ensures that subspaces for both factors are calibrated regularly.

3.  **Soft Rotation**:
    - **Function**: Handles high noise in the reference during early training stages, where hard rotation might lead to drastic over-correction.
    - **Mechanism**: Construct $R' = (1-\lambda)I + \lambda R_i^{t,*}$ and project it back onto the orthogonal group to get $R_{i,\text{soft}}^t$. $\lambda = 0$ degrades to FedIT, while $\lambda = 1$ is hard Procrustes. Lemma A.1 proves $\|R_{\text{soft}} - I\|_F \le 2\lambda \|R - I\|_F$, meaning the magnitude of the soft rotation correction is linearly bounded by $\lambda$.
    - **Design Motivation**: In early stages, the global model has not converged. Forcing clients toward it might break personalized convergence trajectories. Experiments show $\lambda \in [0.2, 0.8]$ outperforms hard alignment, with the optimum often at 0.4-0.6.

### Loss & Training
The standard FedIT training workflow is maintained, with the rotation step inserted only before client upload. The paper provides a convergence analysis under non-convex settings (Theorem 4.4), decomposing the error into an initial gap + cumulative aggregation error $\|E^t\|_F^2 + \mathcal{O}(\eta)$. Theorem 4.8 further proves that the error bound after alignment is strictly tighter than the naive version, with a tightness gain $\Gamma(\lambda) = (c_0 - \frac{4\sqrt{\tau}\kappa\eta G_B}{\delta_A})\lambda - 4\kappa^2\lambda^2\tau$, defining a feasible interval for $\lambda$.

## Key Experimental Results

### Main Results
Experiments were conducted on RoBERTa-Large for five GLUE tasks with rank $r=4$ and three client scales $N \in \{3, 10, 50\}$.

| Task/Scale | FedIT | FFA-LoRA | RoLoRA | FedRot-LoRA |
| :--- | :--- | :--- | :--- | :--- |
| MNLI ($N=3$) | 0.866 | 0.862 | 0.868 | **0.876** |
| RTE ($N=3$) | 0.840 | 0.830 | 0.854 | **0.868** |
| GLUE Avg ($N=50$) | 0.768 | 0.772 | 0.824 | **0.873** |
| GSM8K (Llama 3-8B) | 0.429 | 0.436 | 0.344 | **0.444** |
| HumanEval pass@1 | 0.288 | 0.385 | 0.295 | **0.409** |

The reduction in aggregation error is particularly notable: on MNLI, FedIT error is $3.98 \times 10^{-3}$, while FedRot-LoRA is only $1.48 \times 10^{-4}$, an order of magnitude difference.

### Ablation Study

| Configuration | MNLI Acc |
| :--- | :--- |
| No Alignment (FedIT) | 0.866 |
| Random Rotation | 0.318 |
| Scalar Scaling Alignment | 0.865 |
| Align $A$ only | 0.861 |
| Align $B$ only | 0.862 |
| Alternating $A/B$ (Full) | **0.876** |
| reference = $W^{t-2}$ | 0.866 |
| reference = $W^{t-1}$ (Default) | **0.876** |

### Key Findings
- Scalar scaling (1D rotation) is almost ineffective in high-dimensional LoRA, proving that subspace-level (orthogonal) alignment is necessary for $r > 1$.
- Random rotation drops performance to 0.318, confirming that the "meaningful direction" of alignment is crucial. This rules out the trivial explanation that improvements come merely from added stochasticity.
- The advantage of FedRot-LoRA grows with heterogeneity (Dirichlet $h=0.5$, $N=50$). In near-IID cases ($h=100$), while baselines perform better, FedRot-LoRA still maintains a steady 1-2 point lead.

## Highlights & Insights
- **Reframing Algebraic Tricks as Geometric Invariance**: While previous solutions attempted to fix the algebraic term $(B_i - B_j)(A_i - A_j)$, this work identifies the relative rotation of latent subspaces as the primary culprit. This shift from "freezing/projection" to "alignment" is elegant.
- **Procrustes is a Perfect Match for Federated LoRA**: It satisfies almost all desiderata: closed-form solution, $\mathcal{O}(r^3)$ complexity, orthogonality preservation, semantic preservation, and zero additional communication.
- **The Importance of Soft Rotation $\lambda$**: Hard alignment can sometimes degrade performance. Using $\lambda \in [0, 1]$ to control the alignment strength is key to engineering the theoretical "optimal orthogonal matrix." This strategy of being conservative early and confident later in training can be applied to other federated/distributed schemes.

## Limitations & Future Work
- The reference relies on the previous global model; $W^{t-1}$ may become outdated if clients are selected unevenly or if packet loss occurs. Experiments with $W^{t-2}$ showed performance drops.
- $\lambda$ is a hyperparameter. The feasible interval provided by theory depends on constants like $c_0, \delta_A, \kappa, \tau$, which are difficult to estimate, necessitating grid searches in practice.
- Extreme scales (rank $> 24$ or $N > 50$) were not verified; at very large ranks, the $\mathcal{O}(r^3)$ of SVD may become significant.
- Direct comparisons with FedSA-LoRA or FedEx-LoRA on generative tasks were not performed (Table 7 in the appendix only provides a survey-style comparison).

## Related Work & Insights
- **vs FedIT**: FedIT is the baseline using direct factor-wise averaging. This work adds rotational alignment, reducing aggregation error by an order of magnitude and improving performance across missions.
- **vs FFA-LoRA/RoLoRA**: These methods achieve linear aggregation by freezing one factor, essentially "bypassing" rotational ambiguity. This work "actively solves" it, preserving the full parameter space. FedRot-LoRA leads RoLoRA by 5 points when $N=50$.
- **vs FlexLoRA**: FlexLoRA aggregates in the full-parameter space and projects back via SVD, which costs $\mathcal{O}(d^3)$ and can be numerically unstable. This work performs SVD on $r \times r$, making it much cheaper.
- **Insight**: Any distributed learning involving low-rank decomposition (e.g., federated PCA, federated matrix factorization) likely suffers from the same "destructive interference due to decomposition invariance." Orthogonal Procrustes should be a primary tool to consider for these problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The observation that "rotational invariance is the root cause" is insightful and unifies scattered attempts into a clear geometric framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 GLUE tasks × 3 scales × 5 ranks + GSM8K + HumanEval + various ablations including controls for scalar scaling and random rotation.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1-2 explain the motivation intuitively; the theoretical sections (Theorems 4.4 and 4.8, Corollary 4.9) are well-structured.
- Value: ⭐⭐⭐⭐ It can be directly integrated into existing Federated LoRA frameworks by replacing the aggregation step, making it very deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)
- [\[ICML 2026\] Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA](geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)
- [\[ICML 2026\] FedSDR: Federated Self-Distillation with Rectification](fedsdr_federated_self-distillation_with_rectification.md)
- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)

</div>

<!-- RELATED:END -->
