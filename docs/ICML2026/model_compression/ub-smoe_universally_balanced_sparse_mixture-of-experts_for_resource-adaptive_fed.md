---
title: >-
  [Paper Note] UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-Adaptive Federated Fine-tuning of Foundation Models
description: >-
  [ICML 2026][Model Compression][Federated Fine-tuning] The authors observe that directly applying Sparse MoE to heterogeneous federated LoRA fine-tuning leads to two fatal problems: "expert utilization imbalance" and "non…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Federated Fine-tuning"
  - "Sparse MoE"
  - "Heterogeneous Clients"
  - "Dynamic Routing"
  - "Pseudo-Gradient"
date: 2026-05-08
content_hash: 42ba1d24e09b5da6
---

# UB-SMoE: Universally Balanced Sparse Mixture-of-Experts for Resource-Adaptive Federated Fine-tuning of Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.16690](https://arxiv.org/abs/2605.16690)  
**Code**: None  
**Area**: Federated Learning / Model Compression / Sparse MoE / LoRA Fine-tuning  
**Keywords**: Federated Fine-tuning, Sparse MoE, Heterogeneous Clients, Dynamic Routing, Pseudo-Gradient

## TL;DR
The authors observe that directly applying Sparse MoE to heterogeneous federated LoRA fine-tuning leads to two fatal problems: "expert utilization imbalance" and "non-differentiable Top-K." They propose Dynamic Modulated Routing (DMR) to rebalance expert activation and Universal Pseudo-Gradient (PG) to provide signals for inactive experts. This creates a self-reinforcing loop that enables low-compute clients to achieve an 8.7× performance improvement while saving 45% of computation.

## Background & Motivation
**Background**: The mainstream solution for federated fine-tuning of Foundation Models (FM) is LoRA, which freezes pre-trained weights and injects low-rank matrices $B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times l}$ to update $\Delta W=\frac{\alpha}{r}BA$. To handle system heterogeneity in real-world devices, methods like HetLoRA, FlexLoRA, FLoRA, and FLoRIST assign different ranks $r_c$ to each client, allowing low-end devices to use smaller adapters.

**Limitations of Prior Work**: The heterogeneous LoRA-rank approach **saves very little**. The computation of the LoRA part $\mathcal{O}(r_c(d+l))$ is already much smaller than the FFN's $\mathcal{O}(d\cdot l)$, and since FFN computation is independent of rank, low-compute clients only save about 5%. Worse, during inference, $W_0+\Delta W$ remains a dense matrix, meaning latency is identical across all clients.

**Key Challenge**: To make low-compute clients truly lightweight and fast, the FFN itself must be modified, yet the LoRA-rank path ignores the FFN entirely. Sparse MoE provides a natural resource-adaptive mechanism by activating only $K$ experts via conditional computation. However, dropping it into heterogeneous federated scenarios triggers two new issues:

1.  **Expert Utilization Imbalance**: High-compute clients activate more experts, leading to frequent updates and "over-specialization" of those experts. Low-compute clients activate few experts, leaving others untrained for long periods. This creates a rich-get-richer effect.
2.  **Non-differentiable Top-K Routing**: For inactive experts, gating $\gamma_i(x)=0$, which backpropagates zero gradients. Small $K_c$ for low-compute clients means most experts receive no learning signal during local training.

**Goal**: (i) Provide convergence analysis proving these two discordances introduce an "irreducible error floor" inversely proportional to client compute power; (ii) design a mechanism to address both issues; (iii) demonstrate effectiveness especially for low-compute clients on commonsense reasoning and telecommunication benchmarks.

**Key Insight**: The authors observe that expert utilization statistics are global information aggregatable at the server, while gradients for inactive experts can be "approximately reconstructed" based on active experts and routing softmax probabilities. Pairing these enables a self-reinforcing loop: "PG maintains the availability of inactive experts → DMR routes them back to generate true gradients → True gradients make PG more accurate."

**Core Idea**: Routing logits are dynamically modulated using global utilization statistics (DMR), and inactive experts are supplemented with pseudo-gradients (PG), creating a mutually beneficial cycle.

## Method

### Overall Architecture
UB-SMoE injects LoRA adapters into each SMoE layer following a 5-step workflow: (1) The server aggregates global expert utilization $\tilde u_i^{(l)}$; (2) The client router calculates modulated logits $m^{(l)}_i=s^{(l)}_i+\phi^{(l)}_i$ based on $\tilde u$; (3) Clients activate $K_c=\lfloor K_{\max}\beta_c\rfloor$ experts based on their compute budget $\beta_c$; (4) During local training, inactive experts receive pseudo-gradients scaled by the client sparsity $\rho_c$; (5) Clients return parameter deltas and utilization statistics to the server for aggregation.

### Key Designs

1.  **Dynamic Modulated Routing (DMR) —— Reshaping Routing via Global Utilization**:
    - **Function**: Explicitly separates "routing preference" from "global load balancing" to prevent extreme modulation from overriding semantic relevance.
    - **Mechanism**: First uses original affinity $s^{(l)}=W^{(r)}x$ to select a top-$N_p$ ($K_{\max}\le N_p\ll M$) candidate set $\mathcal{T}^{(l)}$. Learnable modulation vectors $\phi^{(l)}_i$ are added only to experts within this set. Then $p^{(l)}=\text{softmax}(m^{(l)})$, and Top-$K_c$ are chosen for activation. On the server, global utilization $\tilde u^{(l)}_i=\sum_c p_c\frac{a^{(l)}_{c,i}}{n^{(l)}_c}$ is computed against a target uniform utilization $u^*=\bar K/M$. Modulations are updated via $\tilde\phi^{(l)}_i=\tanh\left(\frac{u^*}{\tilde u^{(l)}_i+\epsilon}-1\right)$ and smoothed with momentum $\zeta$. This lowers logits for overused experts and raises them for neglected ones, but only within semantically relevant candidate sets.
    - **Design Motivation**: Naive load balancing losses sacrifice expert specialization for uniformity. UB-SMoE decouples "semantic relevance" (candidate filtering) from "load balancing" (modulation within the set), fixing Discordance 1 without turning routing into noise.

2.  **Universal Pseudo-Gradient (PG) —— Creating Gradients for Inactive Experts**:
    - **Function**: Ensures inactive experts receive approximate learning signals in every batch on every client, breaking the deadlock of zero Top-K gradients.
    - **Mechanism**: For inactive experts $i\notin\mathcal{A}_c(x)$, pseudo-gradients are constructed using the router's softmax probability and the true gradients of active experts, scaled by client sparsity $\rho_c$ (inversely proportional to $K_c/M$). A smaller $K_c$ results in higher pseudo-gradient weight due to the urgent need for compensation. Mathematically, this relaxes the expectation of $\nabla_{\Theta^{(e)}_i}F_c$ from being "conditioned on $i\in\mathcal{A}_c(x)$" back to "unconditional," reducing the bias $B_{c,i}(\Theta)$ defined in the paper.
    - **Design Motivation**: Theorem 4.1 proves that sparse Top-K routing causes SGD to converge to a bias error floor $B_{\text{SMoE}}=2\|B(\Theta^*)\|^2/\mu'$. Corollary 1 notes this floor $\propto (M-K_c)$, which is severe for low-compute clients. PG directly attacks this bias source by making "updating while inactive" equivalent to $p_{c,i}(\Theta)\to 1$.

3.  **DMR ↔ PG Self-Reinforcing Loop + $L_2$ Regularization**:
    - **Function**: Allows the two mechanisms to feed each other, preventing instability.
    - **Mechanism**: PG keeps all experts learning so they don't "die" → DMR's global utilization stats remain meaningful for precise scheduling → After scheduling, more experts are actually activated and produce true gradients → True gradients make pseudo-gradient estimation more accurate. Range regularization $\mathcal{L}_{reg}=\lambda(\|\text{ReLU}(\phi_{\min}-\phi)\|^2_2+\|\text{ReLU}(\phi-\phi_{\max})\|^2_2)$ is applied to $\phi^{(l)}$ to prevent modulation explosion.
    - **Design Motivation**: DMR alone is ineffective on "dead" experts; PG alone could cause all experts to converge to similar parameters. The closed loop provides stability.

### Loss & Training
Local loss = LM loss + DMR regularization $\mathcal{L}_{reg}$. Inactive experts accumulate gradients directly through PG. Clients determine $K_c=\lfloor K_{\max}\beta_c\rfloor$ based on budget $\beta_c\in[0,1]$ and use a unified LoRA rank $r$. The server aggregates LoRA increments, utilization stats, and modulation parameters.

## Key Experimental Results

### Main Results
Evaluated on OLMoE-1B-7B across Commonsense-15K and telecommunication domains, comparing 4 heterogeneous LoRA-rank methods and 2 heterogeneous sparse methods.

| Method | Category | Low Compute ($\beta_1$) ↑ | High Compute ($\beta_4$) ↑ | Average ↑ |
|------|------|----------------------|----------------------|--------|
| HetLoRA | Het. Rank | 0.0079 | 0.4580 | 0.1874 |
| FlexLoRA | Het. Rank | 0.0456 | 0.4563 | 0.3303 |
| FLoRA | Het. Rank | 0.0094 | 0.2996 | 0.1517 |
| FLoRIST | Het. Rank | 0.0112 | 0.2724 | 0.1480 |
| A3SMoE | Het. Sparse | 0.3629 | 0.3410 | 0.3861 |
| **UB-SMoE** | Het. Sparse | **0.3936** | **0.5240** | **0.4267** |

Low-compute client performance jumps from HetLoRA's 0.0079 to 0.3936 (approx. 8.7× improvement), while high-compute performance also exceeds all baselines.

### Ablation Study
| Configuration | Low-Compute Perf | Note |
|------|-----------|------|
| Full UB-SMoE (DMR + PG) | 0.3936 | Complete model |
| w/o PG | Significant drop | Inactive gradients go to 0, bias floor returns |
| w/o DMR | Significant drop | Rich-get-richer recurs, expert monopoly |
| candidate set $N_p=2$ | Optimal | Too large overrides semantics; too small is rigid |
| No $\mathcal{L}_{reg}$ | $\phi$ Diverges | Modulation explosion destroys routing |

### Key Findings
- **Real Computational Savings**: Heterogeneous LoRA-rank methods save ~5% for low-compute clients. UB-SMoE uses Sparse MoE to cut FFN directly, saving 45%.
- **Closure between Theory and Experiment**: The bias error floor $B_{\text{SMoE}}\propto(M-K_c)$ from Theorem 4.1 explains why baselines fail at low compute. The 8.7× improvement on $\beta_1$ aligns with the prediction that smaller $K_c$ yields higher PG gains.
- **Mutual Benefit**: Most heterogeneous methods help low-compute clients at the expense of high-compute ones. UB-SMoE maintains expert diversity, achieving the highest high-compute ($\beta_4$) score of 0.5240.
- **Controllable Communication**: Only $L(M+1)$ dimensions of utilization stats are added compared to LoRA-rank methods, which is negligible relative to parameter deltas.

## Highlights & Insights
- **Clear Theory-Driven Diagnosis**: Using Theorem 4.1 to derive a closed-form bias floor $\propto(M-K_c)$ ensures the method design targets the exact problem—a rare and rigorous paradigm in systems papers.
- **Decoupling of Conditional Structure and Modulation**: The essence of DMR is not just another balance loss, but an orthogonal decomposition of "which experts are semantically fit" (candidate set) and "which experts are systemically underfed" (modulation), avoiding the flattening effect common in balance losses.
- **Physical Meaning of PG**: It essentially approximates the "conditional expectation" of the sparse gradient back to an "unconditional expectation." While similar to dropout or soft routing, the key is the inverse correlation between $\rho_c$ and client compute.
- **Transferability**: The logic applies to (a) sparse LLMs on edge devices with varying $K_c$, (b) multi-task MoE with varying task intensity, and (c) any combination of conditional computation and non-differentiable routing.

## Limitations & Future Work
- Convergence analysis relies on strong assumptions like PL-condition, $L$-smoothness, and bounded gradient divergence, which are simplified models despite being reasonable for small LoRA spaces.
- PG accuracy depends on router softmax quality; if the router is poorly trained, PG might introduce noise—this was not systematically quantified.
- Experiments focused on OLMoE-1B-7B; scalability to larger FMs (70B+) and multi-modal MoE remains unverified.
- Numerous hyperparameters (momentum $\zeta$, scaling $\rho_c$, bounds $[\phi_{\min},\phi_{\max}]$, $N_p$) lack an automated cross-task tuning scheme.

## Related Work & Insights
- **vs. HetLoRA / FlexLoRA / FLoRA / FLoRIST**: These follow the "heterogeneous rank" path. UB-SMoE follows "heterogeneous sparsity," meaning it actually reduces FFN computation. It saves 45% (vs 5%) for low-compute clients but requires an MoE architecture.
- **vs. A3SMoE (Tran et al., 2025)**: A3SMoE introduced SMoE to federated fine-tuning but did not solve the utilization imbalance or non-differentiability. UB-SMoE outperforms it across all budget tiers.
- **vs. Centralized MoE Training**: While load balancing losses (Switch Transformer, GShard) suffice in centralized settings, the federated problem is magnified by varying $K_c$ across clients, necessitating global utilization aggregation and client-aware pseudo-gradients.

## Rating
- Novelty: ⭐⭐⭐⭐ Significant advancement in combining SMoE with heterogeneous federated fine-tuning via the DMR+PG loop.
- Experimental Thoroughness: ⭐⭐⭐ Strong comparisons on two benchmarks, though larger FMs and multi-modal tests are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear convergence derivation with theory-backed method design.
- Value: ⭐⭐⭐⭐ Directly applicable to edge federated scenarios, enabling meaningful participation from low-compute devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](../../ICLR2026/model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[ICML 2026\] DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](dag-moe_from_simple_mixture_to_structural_aggregation_in_mixture-of-experts.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/model_compression/quant_experts_token_aware_vlm_quantization.md)
- [\[ICML 2026\] Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA](geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)

</div>

<!-- RELATED:END -->
