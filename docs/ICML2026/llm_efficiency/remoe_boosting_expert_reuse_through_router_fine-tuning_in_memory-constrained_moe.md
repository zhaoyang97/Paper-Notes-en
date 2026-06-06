---
title: >-
  [Paper Note] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference
description: >-
  [ICML 2026][LLM Efficiency][Fine-grained MoE] ReMoE freezes all non-router parameters and fine-tunes only the gate using a compound loss comprising "temporal locality regularization + Trust-KL semantic anchor." This resh…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Fine-grained MoE"
  - "Expert Offloading"
  - "Temporal Locality"
  - "Router Fine-Tuning"
  - "Cache Hit"
date: 2026-05-08
content_hash: 50f54fd487969135
---

# ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference

**Conference**: ICML 2026  
**arXiv**: [2605.27081](https://arxiv.org/abs/2605.27081)  
**Code**: https://github.com/BUAA-OSCAR/ReMoE (Available)  
**Area**: LLM Efficiency / MoE Inference / Edge Deployment  
**Keywords**: Fine-grained MoE, Expert Offloading, Temporal Locality, Router Fine-Tuning, Cache Hit

## TL;DR
ReMoE freezes all non-router parameters and fine-tunes only the gate using a compound loss comprising "temporal locality regularization + Trust-KL semantic anchor." This reshapes the routing trajectory to be more "cache-friendly." Without altering the architecture or adding runtime overhead, it increases the expert reuse rate of adjacent tokens by approximately 26%, reducing TPOT by 43.6–49.8% (1.77–1.99× decoding speedup) on Jetson Orin NX.

## Background & Motivation

**Background**: Fine-grained MoEs like DeepSeek-V2/V3 and Qwen-MoE increase the number of experts per layer to dozens or hundreds, with only Top-$K$ activated per token. The large total parameters but small active parameters make them suitable for edge devices with abundant UFS/SSD but constrained DRAM (e.g., Samsung UFS 4.0 provides 4 GB/s read bandwidth and 1 TB capacity). Existing systems like MoE-Infinity, HOBBIT, Fiddler, and KTransformers employ expert caching and prefetching between CPU/GPU.

**Limitations of Prior Work**: During the decoding phase, each token may activate a completely different set of experts, leading to frequent cache misses and severe I/O thrashing. Especially in interactive inference ($B{=}1$), there is no batching to amortize I/O, making expert migration the primary bottleneck for end-to-end latency.

**Key Challenge**: The load-balancing loss $L_{\text{aux}}$ used during training for expert parallelism forces tokens to be distributed uniformly across all experts. This directly contradicts the requirement of single-request inference, which relies on a "small cache budget + high reuse of the active expert set across adjacent tokens"—representing a *training–deployment mismatch*.

**Goal**: Without modifying expert weights, changing the inference graph, or introducing new runtime strategies, reshape the router's output trajectory $\{E_t\}$ to be more "reusable within short windows," thereby reducing the number of distinct expert loads at the upstream (trace) level.

**Key Insight**: Observations of the routing trajectory for DeepSeek-V2-Lite (Figure 2, Layer 21) show that the baseline router already exhibits short reuse streaks, which are interrupted by frequent "minor switching." This suggests that natural locality exists and only requires lightweight reshaping for amplification, avoiding the need for architectural redesign or pre-training from scratch as in Oracle-MoE.

**Core Idea**: Translate "cache hits" into a differentiable optimization objective for the router. Freeze the entire model and fine-tune only the gate parameters $\theta_{\text{gate}}$ to encourage the router to reuse recently selected experts, while employing a KL anchor to pull the distribution back toward the pre-trained router, preventing semantic drift.

## Method

### Overall Architecture
ReMoE is a post-training router fine-tuning framework with a pipeline identical to the baseline: input token → hidden state $h_t$ → router calculates $P_t = \mathrm{Softmax}(h_t^\top \theta_{\text{gate}})$ → Top-$K$ expert selection → expert forward pass.

The difference lies in training: two gates run in parallel within each MoE layer—one frozen pre-trained snapshot $\theta_{\text{gate}}^0$ (producing reference distribution $P_t^{\text{ref}}$) and one trainable gate (producing $P_t$). Only $\theta_{\text{gate}}$ receives gradients; expert FFNs, attention, and embeddings are frozen. A small routing history buffer is maintained for temporal regularization. During deployment, the fine-tuned gate weights are swapped in without modifying the inference graph, kernels, or caching strategies.

The total loss is $\mathcal{L}=L_{\text{CE}}+\lambda_{\text{KL}}\,L_{\text{Trust}}+\alpha_t\,L_{\text{Loc}}$, where $L_{\text{CE}}$ is the standard next-token CE and $\alpha_t = \min(1, t/T_{\text{warm}})$ performs a linear warmup for locality regularization. The $L_{\text{aux}}$ loss is explicitly disabled during fine-tuning as its "dispersion" objective opposes the goal.

### Key Designs

1.  **Gate-only Fine-tuning + Differentiable Reuse Mass Proxy**:
    - **Function**: Converts the discrete, non-differentiable "number of overlapping experts between step $t$ and step $t-1$" into a continuous objective suitable for SGD optimization.
    - **Mechanism**: Defines $\tilde{E}_{t-1} = \texttt{stop\_gradient}(E_{t-1})$ as the indices of experts selected in the previous step. The current reuse mass is $m_t = \frac{1}{K}\sum_{k\in\tilde{E}_{t-1}} P_t^{(k)}$, representing the probability allocated by the current router to the $K$ experts just used. Utilizing `stop_gradient` ensures gradients only flow to the current $P_t$, creating a "unidirectional catch-up" signal. A higher $m_t$ increases the probability of Top-$K$ selection falling on old experts, raising the expected overlap $\mathrm{IR}_t = |E_t \cap E_{t-1}|/K$. Proposition 3.1 proves that under standard LRU and request isolation, $\bar{N}_{\text{fetch}} \le K(1 - \mathrm{EOR})$, linking reuse mass to I/O count.
    - **Design Motivation**: Hardware-aware training faces the difficulty of non-differentiable hardware metrics. Using reuse mass as a smooth lower bound for Top-$K$ overlap allows "cache hits" to directly enter gradient backpropagation. Since only router parameters are tuned, fine-tuning is extremely lightweight (2000 steps, 100k samples on OpenHermes-2.5).

2.  **Temporal Locality Regularization $L_{\text{Loc}}$ (Four-term Combination)**:
    - **Function**: Simultaneously suppresses high-frequency jitter, slow drift, and local working set expansion in routing trajectories across different time scales.
    - **Mechanism**: $L_{\text{Loc}} = \lambda_{\text{Reuse}} L_{\text{Reuse}} + \lambda_{\text{Smooth}} L_{\text{Smooth}} + \lambda_{\text{Lag}} L_{\text{Lag}} + \lambda_{\text{WS}} L_{\text{WS}}$. (1) $L_{\text{Reuse}} = -\log(\rho + 10^{-8})$, where $\rho$ is the sequence average reuse mass, directly increasing short-window overlap. (2) $L_{\text{Smooth}} = \frac{1}{T-1}\sum \text{SymKL}(P_t, P_{t-1})$ uses symmetric KL to suppress distribution jitter between adjacent steps (without `stop_gradient` to allow bidirectional coupling). (3) $L_{\text{Lag}}$ performs SymKL over a lag set $\mathcal{D} = \{1,2,4,8,16\}$ to capture slow multi-step drift. (4) $L_{\text{WS}}$ minimizes the entropy $H(\bar{P}_b)$ of the average distribution every $W$ steps, encouraging sparse expert activity within each local window to align with small cache capacities.
    - **Design Motivation**: A single reuse term only addresses "adjacent steps," leaving residual misses from cumulative "drift" and windowed "diffusion." The four terms cover short-horizon, multi-step, and windowed locality from a cache perspective.

3.  **Trust-KL Semantic Anchor**:
    - **Function**: Limits the divergence of the fine-tuned router from the pre-trained router in distribution space, preventing locality regularization from pushing the model toward degenerate solutions that are cache-friendly but sacrifice perplexity.
    - **Mechanism**: A frozen FP32 gate snapshot $\theta_{\text{gate}}^0$ calculates $P_t^{\text{ref}}$ based on the current $h_t$ (making the reference adaptive to context), and $L_{\text{Trust}} = \frac{1}{T}\sum_t D_{\text{KL}}(P_t \,\|\, \texttt{stop\_gradient}(P_t^{\text{ref}}))$. KL divergence is chosen over L2 or cosine because it naturally up-weights high-probability experts, corresponding to the dominant regions of Top-$K$ decisions, consistent with its use as a soft trust region in PPO or distillation.
    - **Design Motivation**: Unlike heavy architectural redesigns, ReMoE aims for lightweight post-training. The "safety rail" ensures the model maintains original capabilities without teacher models or inference graph changes. Trust-KL guarantees that in OOD domains, the model at worst loses some acceleration without performance degradation.

### Loss & Training
Fine-tuned on DeepSeek-V2-Lite (15.7B/2.4B, 27 layers, 64 routed + 2 shared experts per layer, Top-$K{=}6$) for 2000 steps using AdamW ($lr=5\times 10^{-5}$, 200-step linear warmup). BF16 precision, gradient clipping 1.0, sequence length 2048, micro-batch=1, grad-accum=8. Data: 100k samples from OpenHermes-2.5 for training and 1k for evaluation. Locality terms use $\alpha_t = \min(1, t/T_{\text{warm}})$ warmup. Hyperparameters $\lambda, \mathcal{D}, W$ are detailed in Appendix E.

## Key Experimental Results

### Main Results

| Dataset / Platform | Metric | Baseline | ReMoE | Gain |
|--------|------|------|----------|------|
| DeepSeek-V2-Lite, $B{=}1$ | EOR ↑ | 27.3% | 34.5% | +7.2 pp (+26.4%) |
| Same as above | Routing Entropy ↓ | 0.9998 | 0.9971 | −0.27% |
| Same as above | Load-balance CV ↑ | 0.0409 | 0.1608 | +293% |
| Cache $C{=}6$, LRU | uHR ↑ | 0.3187 | 0.3687 | +0.0500 |
| Same as above | #uMiss (M) ↓ | 0.8707 | 0.8068 | −0.0639 |
| vLLM, RTX 3090, ShareGPT | Output Throughput (tok/s) | 3.58 | 3.88 | +8.4% |
| Same as above | TPOT (ms) ↓ | 254.31 | 242.99 | −4.5% |
| Jetson Orin NX, ShareGPT | TPOT (ms) ↓ | 554.69 | 306.27 | −44.8% (1.81×) |
| Jetson, GSM8K | TPOT (ms) ↓ | 613.73 | 346.04 | −43.6% (1.77×) |
| Jetson, HumanEval | TPOT (ms) ↓ | 672.68 | 337.61 | −49.8% (1.99×) |

CE-only fine-tuning (router tuned with only $L_{\text{CE}}$) served as a control; its EOR dropped to 22.9%, and vLLM throughput decreased to 2.95 tok/s, ruling out the claim that any router fine-tuning would suffice.

### Ablation Study

| Configuration / Benchmark | Key Metric | Description |
|------|---------|------|
| Full ReMoE | EOR 34.5% / uHR@6 0.369 | Full model |
| w/o Trust ($\lambda_{\text{KL}}{=}0$) | Higher EOR, degraded PPL | Routing more aggressive but quality drops without anchor |
| w/o Reuse | Significant EOR drop | Primary overlap signal comes from reuse term |
| w/o Consistency (smooth/lag/ws=0) | EOR drop | Consistency terms suppress drift and diffusion |
| GSM8K (EM, strict) | 38.89 → 38.13 | −0.76 pp, within variance |
| HumanEval (pass@1) | 26.83 → 29.27 | +2.44 pp, improvement observed |
| MMLU (acc) | 57.72 → 57.81 | +0.09 pp, consistent |
| IFEval (prompt loose) | 17.93 → 17.93 | 0 change |

### Key Findings
- **High CV with constant global diversity**: The number of distinct experts visited globally remained nearly identical (64.000 vs. 63.997). ReMoE creates step-level concentration (repeated use within short windows) rather than global routing collapse, precisely what caches favor.
- **vLLM speedup is lower than Jetson**: The RTX 3090's PCIe Gen3 x16 path partially hides misses, making the 8.4% improvement a conservative bound. On Jetson's SSD-backed path, miss penalties are high; the cache improvement translates directly to 1.77–1.99× decoding speedup, showing ReMoE's gains scale with hardware miss penalties.
- **CE-only is a negative control**: Despite the same training conditions, its EOR is lower than the baseline and throughput drops by 18%, proving that acceleration stems from the locality objective, not simply additional data exposure for the router.

## Highlights & Insights
- **Clean translation of hardware KPIs to differentiable router objectives**: Proposition 3.1 establishes the EOR↔fetch upper bound, and reuse mass acts as a smooth lower bound for EOR. This "Hardware KPI → Discrete Routing Metric → Differentiable Proxy" chain can be applied to any dispatch-style module.
- **Efficient division of labor between gate-only tuning and frozen experts**: Parameter space optimization (quantization) reduces "cost per fetch," while ReMoE reduces "fetch frequency." These are orthogonal and stackable; gate-only tuning makes the cost negligible for community checkpoints.
- **Adaptive reference distribution**: Using the current $h_t$ for Trust-KL is critical. The reference distribution adapts to context, allowing necessary expert switches during semantic shifts, ensuring OOD performance defaults to the baseline acceleration.
- **Multi-scale locality regularization**: Suppressing high-frequency jitter (adjacent SymKL), medium-frequency drift (lag-SymKL), and windowed diffusion (entropy) is far more robust than a single regularization term.

## Limitations & Future Work
- Locality regularization increases inference-time CV, specifically targeting $B{=}1$ single-request edge inference. In multi-request datacenter expert parallelism, increased CV might hurt load balancing; this scenario was not addressed.
- Full pipeline experiments were limited to DeepSeek-V2-Lite. While others like Qwen1.5-MoE-A2.7B showed EOR gains, systematic scanning across larger scales (DeepSeek-V3, Mixtral 8×22B) and varying Top-$K$ is lacking.
- The cold-start cache assumption under request isolation is idealized. Multi-session shared caches may further amplify or dilute ReMoE's gains in real serving environments.
- IFEval strict prompt scores dropped by 1.11 pp, suggesting locality tuning may slightly impair long-tail "strict instruction following." Task-aware scheduling of $\lambda_{\text{KL}}$ and locality weights is a potential next step.
- Future directions include prefetch-aware reuse mass (rewarding experts ready in the prefetch window) and RL-based router policy tuning using real cache states as observations.

## Related Work & Insights
- **vs Oracle-MoE (Zhou et al., 2025)**: Oracle-MoE redesigned routing architectures and pre-trained from scratch for locality. ReMoE uses post-training gate-only tuning, which is significantly cheaper but likely has a lower performance ceiling.
- **vs Mixture of Cache-Conditional Experts (Skliar et al., 2025)**: MCCE biases expert selection during *inference* based on cache residency, requiring inference graph changes and per-step cache checks. ReMoE reshapes trajectories offline and is orthogonal to runtime cache strategies.
- **vs MoE-Infinity / HOBBIT / Fiddler / KTransformers**: These are system-level runtime optimizations for "how to migrate when a miss occurs." ReMoE addresses "how to send fewer miss requests" at the source; they are complementary.
- **vs Quantization (GPTQ / AWQ)**: Lowering bits per parameter reduces "fetch volume," while ReMoE reduces "fetch frequency."
- **vs load-balancing loss**: Standard goals encourage dispersion for parallelism; ReMoE demonstrates that for edge inference, the objective is the inverse, highlighting the need for deployment-aware training objectives.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Formulating cache locality as a differentiable router objective is a novel perspective, though components like KL anchors have precedents in RL/distillation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid verification across trace simulations, vLLM, and Jetson edge devices; CE-only serves as an excellent control. More diverse full-model pipelines beyond DeepSeek-V2-Lite would be better.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured flow (Motivation-Proxy-Reg-Anchor-Eval). Propositions and Appendices clearly explain the choice of reuse mass.
- **Value**: ⭐⭐⭐⭐⭐ Delivering 1.77–1.99× edge decoding speedup with zero runtime changes and orthogonality to quantization/caching is highly practical for on-device MoE deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection](../../ICLR2026/llm_efficiency/tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)

</div>

<!-- RELATED:END -->
