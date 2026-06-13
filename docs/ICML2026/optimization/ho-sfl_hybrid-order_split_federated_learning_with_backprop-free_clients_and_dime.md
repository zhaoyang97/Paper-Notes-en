---
title: >-
  [Paper Note] HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation
description: >-
  [ICML 2026][Optimization][Split Federated Learning] HO-SFL decouples the client and server of split federated learning (SFL) via Lagrangian variable lifting—the server continues with first-order backpropagation (BP)…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Split Federated Learning"
  - "Zeroth-Order Optimization"
  - "Backprop-Free"
  - "Dimension-Free Aggregation"
  - "Edge Fine-tuning"
date: 2026-05-08
content_hash: 2ea6c73486e5d4c4
---

# HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients and Dimension-Free Aggregation

**Conference**: ICML 2026  
**arXiv**: [2603.14773](https://arxiv.org/abs/2603.14773)  
**Code**: Not publicly available  
**Area**: Optimization / Federated Learning / Distributed Training  
**Keywords**: Split Federated Learning, Zeroth-Order Optimization, Backprop-Free, Dimension-Free Aggregation, Edge Fine-tuning

## TL;DR
HO-SFL decouples the client and server of split federated learning (SFL) via Lagrangian variable lifting—the server continues with first-order backpropagation (BP), while the client performs only zeroth-order (ZO) perturbation forward passes. By leveraging shared random seeds, the uplink communication per round is compressed to $\mathcal{O}(P)$ scalars, reducing the GPU memory for fine-tuning large models at the edge to inference levels while maintaining a convergence rate of $\mathcal{O}(\sqrt{d_c/PT})$.

## Background & Motivation

**Background**: Edge fine-tuning of large models has become a new demand in Federated Learning (FL). Mainstream frameworks include FL (McMahan 2017) and SFL (Thapa 2022)—the latter splits the model into two parts, with the heavy server-side part performing the large computations and the lightweight client-side part staying on the edge to reduce client compute pressure.

**Limitations of Prior Work**: Standard SFL still requires clients to run full BP on their sub-models to obtain gradients. For LLMs with hundreds of millions or billions of parameters, the activation cache required for BP exceeds the memory of mobile/IoT devices, even if only a few layers are kept. Works like MeZO use zeroth-order optimization (ZO) to replace BP, reducing memory to inference levels, but the variance of ZO estimators scales linearly with dimension $d$, causing the convergence rate to degrade to $\mathcal{O}(\sqrt{d/T})$, which is almost infeasible for large models.

**Key Challenge**: BP is accurate but memory-intensive; ZO is memory-efficient but converges slowly. Directly combining them (e.g., FedZO, MU-SplitFed) leaves the system burdened by high-dimensional variance of ZO because the client and server share the same optimization objective $\ell(f_s(f_c(\bm x;\bm\theta_c);\bm\theta_s),y)$, coupling the parameters and forcing them to choose the same order.

**Goal**: Decouple the client-server coupling, allowing each side to choose the optimization order best suited for its resources, while compressing model aggregation communication from $\mathcal{O}(d_c)$ to $\mathcal{O}(P)$.

**Key Insight**: Explicitly write the "client activation = server input" equality $\bm z=f_c(\bm x;\bm\theta_c)$ as an equality constraint, then use variable lifting to introduce a Lagrangian multiplier $\bm\lambda$. This splits the original composite objective into two decoupled sub-problems.

**Core Idea**: The activation gradient $\bm\lambda=\nabla_{\bm z}\ell$ computed by the server BP is itself the optimal Lagrangian multiplier for the constrained problem. By treating it as the client's "local proxy objective" $\mathcal{L}_c(\bm\theta_c)=\bm\lambda^\top f_c(\bm x;\bm\theta_c)$, the client only needs to perform ZO perturbation estimation on this scalar proxy function. Since the proxy function operates on $d_c\ll d$ dimensions at the client, it fundamentally isolates the ZO dimension dependency to a small subspace.

## Method

### Overall Architecture
A single communication cycle consists of four stages: ① The server samples $K$ clients and broadcasts a set of shared random seeds $\{s_p^t\}_{p=1}^P$; ② Each client performs a forward pass using current parameters $\bm\theta_c^t$ to obtain activations $\bm z_m^t=f_c(\bm x_m;\bm\theta_c^t)$, which are uploaded to the server along with labels; ③ The server completes the remaining forward pass $\hat y_m=f_s(\bm z_m^t;\bm\theta_s^t)$ and standard BP to obtain $\bm g_{s,m}^t=\nabla_{\bm\theta_s}\ell$ and activation gradients $\bm\lambda_m^t=\nabla_{\bm z_m^t}\ell$, updates $\bm\theta_s$, and sends $\bm\lambda_m^t$ back to the corresponding client; ④ Each client regenerates $P$ Gaussian perturbations $\bm u_p^t\sim\mathcal{N}(\bm 0,\bm I_{d_c})$ using the shared seeds, runs $P$ small perturbation forward passes $\tilde{\bm z}_{m,p}^t=f_c(\bm x_m;\bm\theta_c^t+\mu\bm u_p^t)$, and calculates $P$ **scalars** $v_{m,p}^t=\bm\lambda_m^{t\top}(\tilde{\bm z}_{m,p}^t-\bm z_m^t)$ to upload. The server averages these across clients to obtain $\bar v_p^t$ and broadcasts it back. The client regenerates $\bm u_p^t$ using the same seeds and assembles the gradient estimate $\hat{\bm g}_c^t=\frac{1}{P\mu}\sum_p\bar v_p^t\bm u_p^t$ to complete one local SGD step.

Throughout the process, the client never needs to perform backpropagation, store activation maps, or upload/download vectors of parameter dimension size.

### Key Designs

1.  **Objective Rewriting via Lagrangian Decoupling**:
    - Function: Splits the SFL single-step objective $\ell(f_s(f_c(\bm x;\bm\theta_c);\bm\theta_s),y)$ into two independent sub-problems: "server only sees $\bm\theta_s$" and "client only sees $\bm\theta_c$", allowing both ends to choose their own optimization order.
    - Mechanism: Treats the activation vector $\bm z$ as an auxiliary variable for variable lifting, rewriting it as a constrained problem $\min\ell(f_s(\bm z;\bm\theta_s),y)\ \text{s.t.}\ \bm z=f_c(\bm x;\bm\theta_c)$, and constructs the Lagrangian $\mathcal{L}_\lambda=\ell(f_s(\bm z;\bm\theta_s),y)+\bm\lambda^\top(f_c(\bm x;\bm\theta_c)-\bm z)$. Solving for the stationary condition $\nabla_{\bm z}\mathcal{L}_\lambda=\bm 0$ yields $\bm\lambda^t=\nabla_{\bm z}\ell(f_s(\bm z^t;\bm\theta_s^t),y)$—which is the activation gradient already calculated in the BP chain rule, incurring "zero additional computational cost". The client sub-objective automatically simplifies to $\mathcal{L}_c(\bm\theta_c)=\bm\lambda^\top f_c(\bm x;\bm\theta_c)$.
    - Design Motivation: Previous works like FedZO/MU-SplitFed performed ZO on the global objective, where the variance depends on the entire model dimension. After decoupling, client ZO only acts on $\mathcal{L}_c$, naturally shrinking the dimension dependency to $d_c$, providing the key source for the $\mathcal{O}(\sqrt{d_c/PT})$ rate.

2.  **Client Zeroth-Order Proxy Estimation via Server Feedback**:
    - Function: Constructs a low-variance gradient estimate for the client that aligns with the global optimization direction without performing BP or storing activation maps.
    - Mechanism: Fixes the anchor point $\bm z_m^t=f_c(\bm x_m;\bm\theta_c^t)$, regenerates $\bm u_p^t$ for each seed $s_p^t$, and performs perturbation forward passes $\tilde{\bm z}_{m,p}^t$. Since the proxy function is linear ($\bm\lambda^\top f_c$), the finite difference simplifies to $v_{m,p}^t=\bm\lambda_m^{t\top}(\tilde{\bm z}_{m,p}^t-\bm z_m^t)$. The client only uploads $P$ scalars. The server computes the cross-client average $\bar v_p^t=\frac{1}{K}\sum_m v_{m,p}^t$ and broadcasts it. Each client independently reconstructs $\hat{\bm g}_c^t=\frac{1}{P\mu}\sum_p\bar v_p^t\bm u_p^t$.
    - Design Motivation: This step solves three issues simultaneously: ⓐ The ZO estimate is "navigated" by the server's activation gradient, making it more accurate than pure ZO; ⓑ Uplink communication sends only $P$ scalars, and downlink sends only $P$ scalars + seeds, achieving dimension-free $\mathcal{O}(P)$ aggregation (applying DeComFL’s idea to the split setting); ⓒ Client perturbation forward passes can run in parallel with server BP to mask latency.

3.  **Seed-Scalar History Catch-up Mechanism for Straggling Clients**:
    - Function: Allows stateless clients that missed certain rounds to catch up with the global state without downloading the $\mathcal{O}(d_c)$ latest model.
    - Mechanism: The server stores historical broadcast tuples $\{(s_p^\tau,\bar v_p^\tau)\}_{p,\tau}$. A client stuck at round $t'$ only needs to pull tuples for $\tau\in[t',t)$, reconstruct perturbations $\bm u_p^\tau=\mathrm{PRG}(s_p^\tau)$ using a pseudo-random generator, reconstruct gradients $\hat{\bm g}_c^\tau=\frac{1}{P\mu}\sum_p\bar v_p^\tau\bm u_p^\tau$, and replay updates sequentially $\bm\theta_c^{\tau+1}\leftarrow\bm\theta_c^\tau-\eta\hat{\bm g}_c^\tau$, avoiding high-dimensional parameter downloads.
    - Design Motivation: In standard FL/SFL, offline clients must re-download the full model upon reconnecting, which is gigabytes for LLMs. HO-SFL compresses "model updates" into scores of 32-bit scalars + seeds, making the downlink $\mathcal{O}(P)$. This, combined with parallel pipelining, masks the typical client idle time in SFL.

### Loss & Training
The global objective remains $\mathcal{L}(\bm\theta)=\frac{1}{M}\sum_m\mathbb{E}_{\xi_m\sim\mathcal{D}_m}[\ell(\bm\theta;\xi_m)]$. The learning rate is set to $\eta=\Theta(\sqrt{P/(T d_c)})$ and the smoothing parameter to $\mu=\mathcal{O}((PT)^{-1/4}d_c^{-5/4})$, which ensures the bias term is of the same order as the variance-optimization terms, leading to a convergence rate of $\mathcal{O}(\sqrt{d_c/PT})$. For vision tasks $P=5$, and for language tasks $P=2$, with $\mu=10^{-3}$.

## Key Experimental Results

### Main Results

| Task / Model | Metric | SplitLoRA (First-order) | ZO-SFL (Pure ZO) | HO-SFL (Ours) |
|--------|------|------|----------|------|
| GLUE-SST2 / OPT-125M | Acc (%) | 87.5 | 52.8 | **87.6** |
| GLUE-RTE / OPT-125M | Acc (%) | 57.8 | 52.0 | **59.2** |
| GLUE-SST2 / Gemma-3-270M | Acc (%) | 90.3 | 51.8 | **90.8** |
| GLUE-RTE / Gemma-3-270M | Acc (%) | 59.6 | 54.2 | **65.0** |
| GLUE-SST2 / LLaMA-3.2-1B | Acc (%) | **94.4** | 61.5 | 93.9 |
| GLUE-RTE / LLaMA-3.2-1B | Acc (%) | 70.0 | 49.1 | **73.3** |
| SQuAD / LLaMA-3.2-1B | F1 | ≈ 0.60 | Diverged | Matches first-order |

On CIFAR-10 under IID settings, HO-SFL's convergence curve is nearly identical to SFL; in Non-IID settings, HO-SFL outperforms SFL because it allows per-step aggregation (dimension-free), whereas SFL suffers from client drift.

### Ablation Study

| Aspect | SFL / SplitLoRA | ZO-SFL / MU-SplitFed | HO-SFL |
|------|---------|---------|---------|
| Client requires BP? | Yes | No | **No** |
| Client Memory | Training level (stores activations) | Inference level | **Inference level** |
| Aggregation Uplink | $\mathcal{O}(d_c)$ | $\mathcal{O}(d_c)$ | **$\mathcal{O}(P)$** |
| Conv. Rate Dim Dependency | $d$-independent (FO) | $\mathcal{O}(\sqrt{d/T})$ | **$\mathcal{O}(\sqrt{d_c/PT})$** |
| Non-IID Robustness | Affected by client drift | Slow/No convergence | Per-step aggregation, low impact |
| Straggler Recovery | Downlink full model | Downlink full model | $\mathcal{O}(P)$ scalars + seeds |

### Key Findings
- Dimensionality decoupling is key: Limiting ZO to the client segment $d_c$ instead of the total $d$ improves the theoretical rate from $\mathcal{O}(\sqrt{d/T})$ to $\mathcal{O}(\sqrt{d_c/PT})$. This is reflected in the experiments—pure ZO baselines (ZO-SFL/MU-SplitFed) fail to converge on LLMs, while HO-SFL matches first-order results.
- Model scale: When increasing from 125M to 8B (64×) parameters, HO-SFL's SQuAD F1 remains consistent with SplitLoRA, indicating the framework is scalable and not dependent on "small model luck."
- Dimension-free aggregation outperforms first-order SFL in Non-IID settings because it can aggregate at every step instead of every few steps, significantly suppressing client drift—a surprising "less communication yields higher accuracy" phenomenon.

## Highlights & Insights
- Writing activation consistency as an equality constraint and absorbing it via Lagrangian multipliers—a tool usually seen in convex optimization textbooks—is used elegantly: the resulting multiplier is exactly the activation gradient from the chain rule, incurring "zero extra cost." This is an elegant "mathematical free lunch."
- The client proxy objective $\bm\lambda^\top f_c(\bm x;\bm\theta_c)$ is linear with respect to $\bm\theta_c$ (as $\bm\lambda$ is fixed), which simplifies the ZO finite difference by canceling out first-order terms. The variance structure is much cleaner than performing ZO on the full loss—this is the fundamental reason the hybrid-order approach scales to large models better than pure ZO.
- Shared seeds + scalar aggregation + PRG history catch-up turns "model synchronization" from "sending parameters" into "sending scalars." This abstraction can be migrated to any federated/split training framework, making it a valuable system design trick.

## Limitations & Future Work
- The server still needs to perform full BP. The solution focuses on saving client memory rather than server compute, so it is not truly "decentralized" and relies heavily on a resource-rich central server.
- Theoretical analysis follows the classic non-convex SGD framework, where bias control for client ZO depends on gradient regularity constants $\Gamma$. For very deep client parts where $d_c$ is large, these constants might worsen; split points in the paper are chosen at small client segments (e.g., initial ResNet blocks or LoRA-only).
- The benefits of dimension-free aggregation only apply to the client part. The server-side parameters remain large. Moving the entire model to the edge would still face high-dimensional ZO issues.
- In asynchronous/straggler scenarios, sequential catch-up of $\mathcal{O}(t-t')$ steps might slow down slow clients, potentially becoming a bottleneck for highly heterogeneous clusters.

## Related Work & Insights
- **vs MeZO (Malladi 2023)**: MeZO reduces single-machine LLM fine-tuning memory to inference levels. HO-SFL moves this to a distributed SFL setting, but instead of simple adaptation, it restricts ZO to the client proxy and uses server BP feedback for navigation, avoiding MeZO’s variance explosion in distributed settings.
- **vs DeComFL (Li 2025)**: DeComFL uses shared random seeds for dimension-free communication in FL. HO-SFL extends this to SFL with a hybrid-order optimization approach.
- **vs MU-SplitFed (Liang 2026)**: MU-SplitFed also uses ZO for clients but uses an imbalanced "multi-step server, few-step client" update. HO-SFL provides a cleaner theoretical decoupling via Lagrangian multipliers, improving convergence from $\mathcal{O}(\sqrt{d/T})$ to $\mathcal{O}(\sqrt{d_c/PT})$.
- **vs FSL-SAGE (Nair 2025)**: FSL-SAGE uses auxiliary models for parallelization, but clients still run BP. HO-SFL eliminates client BP via ZO, offering more complete client liberation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RACO: Reward-free Alignment for Conflicting Objectives](reward-free_alignment_for_conflicting_objectives.md)
- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](../../NeurIPS2025/optimization/covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)

</div>

<!-- RELATED:END -->
