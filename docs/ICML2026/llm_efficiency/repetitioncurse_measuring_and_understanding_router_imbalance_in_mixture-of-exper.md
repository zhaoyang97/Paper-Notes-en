---
title: >-
  [Paper Note] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress
description: >-
  [ICML 2026][LLM Efficiency][MoE router imbalance] By feeding Mixture-of-Experts (MoE) LLMs minimalist OOD prompts that repeat the same token $N$ times…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "MoE router imbalance"
  - "Expert Parallelism"
  - "DoS attack"
  - "TTFT"
  - "Black-box attack"
date: 2026-05-08
content_hash: a42d823558b8d2a3
---

# RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress

**Conference**: ICML 2026  
**arXiv**: [2512.23995](https://arxiv.org/abs/2512.23995)  
**Code**: TBD  
**Area**: LLM Efficiency / MoE Systems / Inference Security  
**Keywords**: MoE router imbalance, Expert Parallelism, DoS attack, TTFT, Black-box attack

## TL;DR
By feeding Mixture-of-Experts (MoE) LLMs minimalist OOD prompts that repeat the same token $N$ times, the authors discover that the router directs nearly all tokens to a fixed set of few top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a single-GPU bottleneck while idling other GPUs, increasing TTFT by 20%–148% on 8-GPU clusters. This effectively turns the MoE parallel accelerator into a DoS attack surface.

## Background & Motivation

**Background**: Modern LLMs (Mixtral, Qwen3-MoE, DeepSeek-V3, GPT-OSS, Llama-4-Scout, etc.) use MoE architectures to expand capacity without increasing per-token compute costs. Industrial deployment utilizes **Expert Parallelism (EP)**—distributing different experts across GPUs—where the router determines which GPU each token is sent to, thereby saving VRAM and reducing communication. Mainstream engines like vLLM and SGLang adopt this as the default route.

**Limitations of Prior Work**: While MoE models utilize expert- or device-level balance losses during the **training phase** to enforce equilibrium, they lack **any balancing constraints during inference**. If tokens in a batch are unevenly concentrated on a few experts, the corresponding GPU becomes a straggler, forcing other GPUs to idle during all-reduce operations, effectively degrading "parallel acceleration" to the speed of the "slowest serial task."

**Key Challenge**: There is a natural contradiction between the **efficiency assumptions** of MoE systems (uniform token distribution) and **adversarial robustness**. Once an attacker can induce "router collapse," larger EP scales lead to deeper performance degradation. Existing LLM-DoS research either forces long outputs (where attackers pay for every token) or relies on backdoors/prompt injection; none directly attack the MoE router itself.

**Goal**: (1) Find prompts that reliably cause router imbalance in a strictly black-box setting (unknown weights, unknown routing strategy, unknown expert-to-GPU mapping); (2) Quantify the actual harm of such attacks on TTFT and SLA P$_{99}$; (3) Systematically characterize which architectural or deployment factors amplify or mitigate this vulnerability.

**Key Insight**: From an embedding space perspective, the router is a deterministic function of the hidden state: $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$. Forcing the router to select the same set of top-$k$ experts is essentially equivalent to making the hidden states of adjacent tokens **collapse into the same cluster**—specifically, minimizing the variance of embeddings at each layer: $D(H^l(X))=\frac{1}{N}\sum_i\|h^l_i-\bar h^l\|_2^2$.

**Core Idea**: Instead of using white-box gradient optimization for $\arg\min_X \sum_l D(H^l(X))$, one can use the **most aggressive zero-order approximation**: directly repeating the same token $N$ times (RepetitionCurse). This collapses hidden states and disrupts routing without requiring any internal model information, and can bypass KV cache reuse by simply slightly modifying the first token.

## Method

### Overall Architecture
The threat model is a typical black-box API attack. The provider deploys MoE on multi-GPU clusters using EP + prefill-decoding disaggregation. The attacker sends prompts via public APIs, indifferent to output quality, aiming to spike the **TTFT of legitimate users in the same batch**. The RepetitionCurse pipeline is extremely short: (1) Select a token $t$ from the vocabulary; (2) Construct prompt $P_t = [t, t, \dots, t]$ (excluding chat templates/system prompts); (3) Send the request. The attack impact is amplified by internal router collapse and EP scheduling.

### Key Designs

1.  **Attack Objective Based on Embedding-Variance Minimization**:
    - **Function**: Formalizes "router collapse" into an analytical objective, explaining why "repeated tokens" work and validating the black-box approximation.
    - **Mechanism**: Defines layer-wise embedding variance $D(H^l(X))$ (formula above). The optimal attack prompt is $X^*=\arg\min_X\sum_{l=1}^L D(H^l(X))$. While direct solving requires white-box gradients, the authors prove: identical tokens $\Rightarrow$ minimal difference between adjacent hidden states $\Rightarrow$ empirical simultaneous minimization of embedding variance and entropy, pushing router outputs toward the same top-$k$ experts. This reduces white-box optimization to two hyperparameters: token selection and repetition count.
    - **Design Motivation**: MoE router balancing capacity **implicitly assumes input tokens are sufficiently divergent in embedding space**. Breaking this assumption renders post-training fine-tuning ineffective, explaining why base and instruct variants share the same vulnerability.

2.  **Theoretical Maximum Imbalance (TMI) and EP Scaling Effects**:
    - **Function**: Provides an upper bound on the ratio of worst-case single-GPU load versus a fair share and explains why larger EP sizes are more fragile.
    - **Mechanism**: Let each GPU host $E_d=|\mathcal{M}_l(d)|$ experts with top-$k$ routing. The worst-case load is $\min(k,E_d)$, while the fair load is $k/|\mathcal{D}|$. Thus, $\text{TMI}=\dfrac{\min(k,E_d)}{k/|\mathcal{D}|}$. This shows a two-stage behavior: when $k\le E_d$ (sparse types like DeepSeek-V3), the attack scales linearly with EP; when $k>E_d$ (like Mixtral-8x7B at EP=8 where $E_d=1<k=2$), the limit is $|\mathcal{D}|\cdot E_d/k=4$ instead of 8.
    - **Design Motivation**: Bridges "operational EP scaling for efficiency" with "security attack surface expansion" as a trade-off, providing actionable deployment advice: avoid blindly increasing EP size.

3.  **Two Pragmatic Downgrades for Black-box Execution**:
    - **Function**: Acknowledges that RepetitionCurse may not reach the TMI upper bound, identifies two gaps, and provides handling methods to ensure effectiveness in production black-box environments.
    - **Mechanism**: (i) **Agnostic to Expert-GPU mapping changes**: Dynamic balancers (e.g., DeepSeek's EPLB) reorder only every ~10 min; mappings are static within this window. The authors use vLLM/SGLang's default sequential mapping as a baseline. (ii) **Inability to pick target experts**: RepetitionCurse concentrates tokens on top-$k$ experts but cannot specify which ones. If the experts hit by the attack are split across GPUs (e.g., some layers of Mixtral at EP=2), latency gains disappear. This is modeled as a probabilistic factor, still yielding stable 1.07×–2.48× TTFT amplification across vocabulary/layers.
    - **Design Motivation**: Contrast with "theoretically optimal white-box attacks" to emphasize that real threats come from **universal but imperfect** black-box capabilities—attackers can always find "hit" tokens by sweeping the vocabulary.

### Loss & Training
This work utilizes **zero-order black-box attacks + systematic measurement**. There are no trainable parameters, gradients, or fine-tuning involved. The attack has only two hyperparameters: token choice and prompt length (represented as ratio $\alpha$ of batch-internal length, $1/2$ or $1$). All "training" efforts are on the measurement side: automatically scanning 139 HuggingFace MoE configurations and benchmarking 13 representative models across EP=2/4/8/16/32.

## Key Experimental Results

### Main Results
Covered 13 MoEs: 4 Mixtral variants ($E{=}8,k{=}2$), Qwen3-30B-A3B trio ($E{=}128,k{=}8$), GPT-OSS-20B/120B, Llama-4-Scout-17B, DeepSeek-V2-Lite, and Kimi-Linear duo. Metrics use the newly defined **LAR** (Latency Amplification Ratio): $\text{LAR}_{\text{moe}}$ for single-layer compute and $\text{LAR}_{\text{ttft}}$ for end-to-end TTFT.

| Model | EP size | $\text{LAR}_{\text{moe}}$ ($\alpha{=}\tfrac12/1$) | $\text{LAR}_{\text{ttft}}$ ($\alpha{=}\tfrac12/1$) | Note |
| :--- | :--- | :--- | :--- | :--- |
| Mixtral-8x7B | 8 | **2.01 / 2.68** | **1.61 / 2.48** | Standard 8-GPU, TTFT up to 2.48× |
| Mixtral-8x7B-It | 8 | 1.94 / 3.12 | 1.65 / 2.48 | Instruct variant equally vulnerable |
| Qwen3-30B-A3B | 32 | 2.28 / 3.22 | 1.53 / 2.15 | High sparsity scales with large EP |
| Qwen3-Coder-30B-A3B-It | 32 | 2.32 / 3.04 | 1.51 / 2.08 | Code-specific version also affected |

**Key Findings**: (a) Mixtral-based models see TTFT amplification of **1.29×–2.48×** under common 8-GPU EP; (b) The vulnerability is present across all 13 models, including Qwen3, GPT-OSS, DeepSeek, and Kimi-Linear (with linear attention); (c) SLA impact: The violation rate for P$_{99}$ TTFT < 20s rises from 1% baseline to **1.4%–13.6%**, enough to trigger SLA penalties and unnecessary autoscaling.

### Ablation Study
| Dimension | Key Result | Explanation |
| :--- | :--- | :--- |
| Vocabulary Coverage $\mathcal{B}$ | $\mathcal{B}\to 1$ when EP=$E$ | Selecting almost any token to repeat causes routing concentration. |
| EP Size Scan | $\mathcal{B}$ correlates positively with EP | Larger EP is more vulnerable, consistent with TMI formula. |
| Within-family Consistency | Mixtral-8x7B vs instructions/Chinese/Nous | Vulnerability is embedded during **pre-training** and persists after tuning. |
| Wide vs. Deep MoE | Qwen3 (Large $E$, Small $L$) vs Mixtral | "Wide and shallow" MoE designs are more resilient under same compute. |

## Highlights & Insights
- Distills the systematic gap of "training-inference objective inconsistency" into a quantifiable attack vector: training enforces load balance but inference is unrestricted. Attackers use OOD inputs to trigger router imbalance, turning EP benefits into a system liability.
- Extremely simplified attack form—"repeating the same token"—linked to optimal white-box attacks via embedding variance and TMI theory, providing rigorous explanation for a "hacky" method.
- Introduces LAR and bottleneck coverage $\mathcal{B}$, translating "router imbalance" from a model-side concept into "SRE/SLA" language directly useful for the inference community.
- Systematic survey of 139 HF MoE configurations provides a checklist for "wide vs deep," "base vs instruct," and "sparsity vs EP size" for future MoE inference engine designs.

## Limitations & Future Work
- Inability to target specific experts: It only ensures concentration on top-$k$, not which $k$. Frequent dynamic balancing (EPLB) may mitigate this (though 10-min windows are currently too long).
- Experiments primarily focus on prefill-bound TTFT scenarios; the impact on decoding-bound long-output scenarios and robustness under chunked-prefill/speculative decoding remains to be evaluated.
- Defensive measures are currently limited to the conservative "limit EP size" suggestion; future work should explore router-side adversarial balancers or token-level repetition detectors.

## Related Work & Insights
- **vs Gao et al. 2024 / Zhang et al. 2024 (Long-output DoS)**: These force max token generation to drain backend resources, requiring attackers to pay for tokens. Ours targets the prefill stage with short inputs, offering higher cost-efficiency and stealth.
- **vs Li et al. 2025b ("Endless Thinking" attacks)**: Those target reasoning models to cause infinite loops. Ours targets **hardware utilization per token**, orthogonal to reasoning capability and effective for all MoE services.
- **vs EPLB / DeepSeek-V3 Deployment (DeepSeek-AI 2024b/2025)**: EPLB handles natural load drift. Ours reveals that remapping periods are insufficient for second-level SLA attacks, acting as a new adversarial benchmark for EPLB.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally models MoE router imbalance as a DoS vector with an elegantly simple method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive cross-model and EP size testing; lacks verification on ultra-large clusters (e.g., DeepSeek-V3 scale).
- Writing Quality: ⭐⭐⭐⭐ Connects systems, adversarial, and theoretical perspectives clearly.
- Value: ⭐⭐⭐⭐⭐ Highly actionable for MoE vendors (EP limits, monitoring) and opens a new research direction for inference robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)

</div>

<!-- RELATED:END -->
