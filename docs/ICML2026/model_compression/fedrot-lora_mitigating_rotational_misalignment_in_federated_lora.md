---
title: >-
  [Paper Note] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA
description: >-
  [ICML 2026][Model Compression][Federated Learning] This paper identifies that the true "enemy" of naive factor-wise averaging in Federated LoRA is potential subspace misalignment caused by rotational invariance. It proposes solving for a rotation matrix $R_i^t$ via orthogonal Procrustes on the client side to align $A$ and $B$ factors before aggregation. Both theory and experiments demonstrate that this significantly reduces aggregation error without increasing communication o…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Federated Learning"
  - "LoRA"
  - "Procrustes Alignment"
  - "Rotational Invariance"
  - "Subspace Alignment"
date: 2026-05-08
content_hash: b168fb3969f0c2bb
---

# FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA

**Conference**: ICML 2026  
**arXiv**: [2602.23638](https://arxiv.org/abs/2602.23638)  
**Code**: https://github.com/haoran-zh/FedRot-LoRA (Available)  
**Area**: Federated Learning / Parameter-Efficient Fine-Tuning / LoRA  
**Keywords**: Federated Learning, LoRA, Procrustes Alignment, Rotational Invariance, Subspace Alignment

## TL;DR
This paper identifies that the true "enemy" of naive factor-wise averaging in Federated LoRA is potential subspace misalignment caused by rotational invariance. It proposes solving for a rotation matrix $R_i^t$ via orthogonal Procrustes on the client side to align $A$ and $B$ factors before aggregation. Both theory and experiments demonstrate that this significantly reduces aggregation error without increasing communication overhead.

## Background & Motivation

**Background**: LoRA represents weight updates as $\Delta W = BA$, where $B\in\mathbb{R}^{d\times r}$ and $A\in\mathbb{R}^{r\times d}$. Due to the drastic reduction in parameter count, it serves as the most natural vehicle for federated fine-tuning of LLMs (e.g., FedIT, FFA-LoRA, and FlexLoRA are based on this).

**Limitations of Prior Work**: Ideal aggregation should be $\Delta W_{ideal}=\tfrac1N\sum B_i A_i$, but the rank of this sum is generally $>r$, making it impossible to maintain a low-rank form. The second-best option, factor-wise averaging $\Delta W_{naive}=\bar B \bar A$, introduces cross-terms $B_i A_j$ that cause training instability. Existing solutions follow three paths: freezing parameters (FFA-LoRA, RoLoRA), which sacrifices expressivity; SVD projection (FlexLoRA), which is computationally expensive; or transmitting high-communication residuals (FedEx-LoRA), which contradicts the original goal of LoRA.

**Key Challenge**: Existing analyses only recognize the algebraic reason of "non-commutative operators," ignoring the **rotational invariance** inherent in LoRA factorization—for any orthogonal $R\in\mathbb{R}^{r\times r}$, $(B_i R)(R^\top A_i)=B_i A_i$. This implies that semantically equivalent updates can be represented in different latent subspaces. During naive averaging, these misaligned subspaces cause "destructive interference," leading to errors much larger than those caused by algebraic terms alone.

**Goal**: Explicitly eliminate the underestimated factor of "rotational error" without increasing communication, freezing parameters, or performing high-dimensional SVD.

**Key Insight**: Since rotation $R$ does not alter semantics, one can proactively select a rotation $R_i^t$ to align all client factors to a common reference. This $R$ possesses $r(r-1)/2$ degrees of freedom on the orthogonal group, which is sufficient for subspace alignment and can be efficiently solved via a closed-form Procrustes solution.

**Core Idea**: Use the global $\bar A^{t-1}/\bar B^{t-1}$ from the previous round as a reference. Each client solves a Procrustes problem to obtain a rotation $R_i^t$, then alternately aligns $A$ or $B$. Finally, a "soft rotation" is interpolated to prevent excessive noise from the reference in early stages.

## Method

### Overall Architecture
FedRot-LoRA aims to resolve the issue where semantically equivalent LoRA factors represented in different subspaces interfere with each other during naive averaging. It inserts a "rotational alignment" step before client uploads. In each communication round $t$: the server broadcasts the previous global factors $(\bar A^{t-1}, \bar B^{t-1})$ as references. Clients perform local training to obtain $(A_i^t, B_i^t)$, then solve a Procrustes problem to find the optimal rotation $R_i^{t,*}$ that rotates their subspace toward the direction of the reference (aligning $A$ in odd rounds and $B$ in even rounds). A coefficient $\lambda$ is used to interpolate between the identity matrix and $R_i^{t,*}$ to produce a "soft rotation" $R_{i,\text{soft}}^t$. Finally, the aligned factors $\tilde A_i^t=(R_{i,\text{soft}}^t)^\top A_i^t$ and $\tilde B_i^t=B_i^t R_{i,\text{soft}}^t$ are uploaded for aggregation. This workflow involves no parameter freezing, residual transmission, or high-dimensional SVD, requiring only an additional $r\times r$ rotation solution.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Server broadcasts previous global reference<br/>(Ā, B̄)"] --> B["Clients perform local training<br/>to get factors (Aᵢ, Bᵢ)"]
    B --> C["Orthogonal Procrustes Alignment<br/>SVD closed-form solution for Rᵢ*"]
    C -->|Odd rounds align A, even rounds align B (Alternating)| D["Soft Rotation Interpolation<br/>R_soft = Project((1−λ)I + λRᵢ*)"]
    D --> E["Apply Rotation<br/>Ã = R_softᵀ A, B̃ = B R_soft"]
    E --> F["Upload and factor-wise aggregate<br/>to get new global (Ā, B̄)"]
    F -.Next Round.-> A
```

### Key Designs

**1. Orthogonal Procrustes Alignment: Eliminating Factorization Ambiguity via Closed-form Rotation**

As mentioned, the root cause is that rotational invariance $(B_iR)(R^\top A_i)=B_iA_i$ causes identical updates to fall into different subspaces, resulting in destructive interference during averaging. FedRot-LoRA proactively finds a rotation for each client to transform its factors into the subspace of the global reference. In odd rounds, it solves $\min_{R}\|R^\top A_i^t - A_{ref}\|_F^2,\;\text{s.t.}\;R^\top R=I,\det R>0$, which is the classic orthogonal Procrustes problem. By performing SVD on the correlation matrix $M=A_{ref}(A_i^t)^\top$ as $M=U\Sigma V^\top$, the closed-form solution is $R_i^{t,*}=V\cdot\text{diag}(1,\dots,1,\det(UV^\top))\cdot U^\top$. Even rounds align $B$ using $M=(B_{ref})^\top B_i^t$. Restricting to the orthogonal group is crucial; the authors prove (Theorem 4.1) that scalar scaling has only 1 degree of freedom and cannot eliminate subspace misalignment, while unconstrained invertible matrices can be ill-conditioned. Orthogonal matrices offer $r(r-1)/2$ degrees of freedom—flexible enough for alignment while remaining well-conditioned. Computationally, this only requires $\mathcal{O}(dr^2+r^3)$, which is much cheaper than the $\mathcal{O}(d^3)$ SVD in FlexLoRA and adds no communication overhead.

**2. Alternating Alignment of $A$ and $B$: Calibrating Both Factors to Prevent Drift**

If only one factor is aligned every round, the other side may drift uncontrollably. FedRot-LoRA alternates such that $A$ is anchored to $A_{ref}$ in odd rounds (with $B$ compensating), and vice versa in even rounds. Although only one SVD is solved per round, both factors are "calibrated" over time. This alternating mechanism is critical: ablations show a significant performance drop when only $B$ is aligned (e.g., 0.879 vs. 0.954 on SST-2), as the initial norm of $B$ is small, providing a weak alignment signal early on. Alternation ensures both subspaces are regularly calibrated and mutually constrained.

**3. Soft Rotation Interpolation: Mitigating Noisy References in Early Stages**

Early in training, the global model has not converged, and the reference itself contains significant noise. Applying a "hard" Procrustes rotation can lead to aggressive over-correction, damaging the client's personalized convergence trajectory. FedRot-LoRA constructs $R'=(1-\lambda)I+\lambda R_i^{t,*}$ and projects it back onto the orthogonal group to obtain $R_{i,\text{soft}}^t$. $\lambda=0$ reverts to original FedIT, while $\lambda=1$ is a hard Procrustes rotation. Lemma A.1 proves $\|R_{\text{soft}}-I\|_F\le 2\lambda\|R-I\|_F$, meaning the correction magnitude is linearly bounded by $\lambda$. This acts as a tunable knob for a "conservative early, confident late" alignment rhythm. Experiments show $\lambda\in[0.2,0.8]$ outperforms hard alignment, with the optimum usually around 0.4–0.6.

### Loss & Training
The standard FedIT training process is maintained, with the rotation step inserted only before client upload. The paper provides a convergence analysis under non-convex settings (Theorem 4.4), decomposing error into initial gap + cumulative aggregation error $\|E^t\|_F^2$ + $\mathcal{O}(\eta)$. Theorem 4.8 further proves that the aligned error bound is strictly tighter than the naive version, with a Gain $\Gamma(\lambda)=(c_0-\tfrac{4\sqrt\tau\kappa\eta G_B}{\delta_A})\lambda - 4\kappa^2\lambda^2\tau$, from which the feasible range of $\lambda$ is derived.

## Key Experimental Results

### Main Results
Evaluated on RoBERTa-Large across five GLUE tasks, rank=4, with three client scales $N\in\{3,10,50\}$.

| Task/Scale | FedIT | FFA-LoRA | RoLoRA | FedRot-LoRA (Ours) |
|--------|------|------|------|------|
| MNLI ($N=3$) | 0.866 | 0.862 | 0.868 | **0.876** |
| RTE ($N=3$) | 0.840 | 0.830 | 0.854 | **0.868** |
| GLUE Avg ($N=50$) | 0.768 | 0.772 | 0.824 | **0.873** |
| GSM8K (Llama 3-8B) | 0.429 | 0.436 | 0.344 | **0.444** |
| HumanEval pass@1 | 0.288 | 0.385 | 0.295 | **0.409** |

The reduction in aggregation error is particularly striking: on MNLI, FedIT's error is $3.98\times10^{-3}$, while FedRot-LoRA's is $1.48\times10^{-4}$—an order of magnitude lower.

### Ablation Study

| Configuration | MNLI Acc |
|------|---------|
| No Alignment (FedIT) | 0.866 |
| Random Rotation | 0.318 |
| Scalar Scaling Alignment | 0.865 |
| Align $A$ Only | 0.861 |
| Align $B$ Only | 0.862 |
| Alternating $A/B$ (Full) | **0.876** |
| reference = $W^{t-2}$ | 0.866 |
| reference = $W^{t-1}$ (Default) | **0.876** |

### Key Findings
- Scalar scaling (1D rotation) is almost useless in high-dimensional LoRA—proving that when rank > 1, subspace-level (orthogonal) alignment is required rather than just norm adjustment.
- Random rotation collapses performance to 0.318, demonstrating that alignment is only effective if the "direction is meaningful." This rules out the trivial explanation that FedRot-LoRA improves simply by adding randomness.
- The advantage is greater in heterogeneous settings (Dirichlet $h=0.5, N=50$). While baselines perform well in near-IID settings ($h=100$), FedRot-LoRA still maintains a stable 1-2 point lead.

## Highlights & Insights
- **Reframing "Algebraic Tricks" as "Geometric Invariance"**: While previous solutions tried to remedy the algebraic term $(B_i-B_j)(A_i-A_j)$, this paper identifies the true culprit as the relative rotation of latent subspaces. This shift from "freezing/projection" to "alignment" is an elegant conceptual transition.
- **Procrustes is a perfect match for Federated LoRA aggregation**: It provides a closed-form solution, $\mathcal{O}(r^3)$ complexity, preserves orthogonality, maintains semantics, and adds no communication—meeting almost all desiderata.
- **The importance of soft rotation $\lambda$**: Hard alignment can degrade performance in certain settings. Using $\lambda\in[0,1]$ to interpolate alignment strength is key to engineering the "optimal orthogonal matrix" for practical use. This "conservative early, confident late" strategy could be transferred to other federated/distributed schemes.

## Limitations & Future Work
- The paper assumes the reference uses the previous round's global model. In cases of uneven client selection frequencies or packet loss, $W^{t-1}$ might be severely outdated; experiments with $W^{t-2}$ showed performance drops.
- $\lambda$ is a hyperparameter. The feasible region provided in the paper depends on constants $(c_0, \delta_A, \kappa, \tau)$ that are difficult to estimate, necessitating grid search in practice.
- Extreme scales (rank > 24 or $N > 50$) were not verified; at very large ranks, the $\mathcal{O}(r^3)$ SVD cost might become non-trivial.
- Lack of direct comparison with FedSA-LoRA or FedEx-LoRA on generative tasks, with only a summary comparison in Appendix Table 7.

## Related Work & Insights
- **vs. FedIT**: FedIT is the baseline using direct factor-wise averaging. This work adds rotational alignment, reducing aggregation error by an order of magnitude and improving performance across most tasks.
- **vs. FFA-LoRA/RoLoRA**: These freeze one factor to achieve linear aggregation, essentially "bypassing" rotational ambiguity. This work "actively solves" it without restricting the parameter space. FedRot-LoRA leads RoLoRA by 5 points at $N=50$.
- **vs. FlexLoRA**: FlexLoRA aggregates in the full-parameter space and projects back to low-rank via SVD, which costs $\mathcal{O}(d^3)$ and is numerically unstable. This work performs SVD only on $r\times r$, making it significantly cheaper.
- **Inspiration**: Any distributed learning involving low-rank decomposition (e.g., federated PCA, federated matrix factorization, federated diffusion adapters) likely suffers from the same "destructive interference due to decomposition invariance." Orthogonal Procrustes is a tool worth trying first in these scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The observation that "rotational invariance is the true root cause" is sharp, unifying scattered attempts into a clear geometric framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers five GLUE tasks × three scales × five ranks + GSM8K + HumanEval + extensive ablations + scalar/random rotation controls.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1-2 explain the motivation intuitively; the theoretical sections for Theorem 4.4/4.8 and Corollary 4.9 are well-structured.
- Value: ⭐⭐⭐⭐ Can be directly applied to existing Federated LoRA frameworks by replacing the aggregation step; very friendly for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA](../../NeurIPS2025/model_compression/robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](../../ACL2025/model_compression/fedex_lora_federated_exact_aggregation.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)
- [\[ICML 2026\] FedSDR: Federated Self-Distillation with Rectification](fedsdr_federated_self-distillation_with_rectification.md)
- [\[ICML 2026\] Geo-Expert: Fine-Tuning an 8B Model into an Expert-Level Geological Reasoning LLM via LoRA](geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)

</div>

<!-- RELATED:END -->
