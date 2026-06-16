---
title: >-
  [Paper Note] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress
description: >-
  [ICML 2026][LLM Efficiency][TTFT] By providing MoE LLMs with extremely simple OOD prompts that repeat the same token $N$ times, the authors find that the router directs almost all tokens to a fixed small set of top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a single-GPU bottleneck while idling other GPUs, increasing TTFT by 20%
tags:
  - ICML 2026
  - LLM Efficiency
  - TTFT
date: 2026-05-08
content_hash: 936b3bb3f739032b
---
# RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress

**Conference**: ICML 2026  
**arXiv**: [2512.23995](https://arxiv.org/abs/2512.23995)  
**Code**: To be confirmed  
**Area**: LLM Efficiency / MoE Systems / Inference Security  
**Keywords**: MoE Routing Imbalance, Expert Parallelism, DoS Attack, TTFT, Black-box Attack

## TL;DR
By providing MoE LLMs with extremely simple OOD prompts that repeat the same token $N$ times, the authors find that the router directs almost all tokens to a fixed small set of top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a single-GPU bottleneck while idling other GPUs, increasing TTFT by 20%–148% on an 8-GPU cluster and turning the MoE parallel accelerator into a DoS attack surface.

## Background & Motivation

**Background**: Modern LLMs commonly utilize MoE (e.g., Mixtral, Qwen3-MoE, DeepSeek-V3, GPT-OSS, Llama-4-Scout) to expand capacity without scaling inference costs. Industrial deployment relies on **Expert Parallelism (EP)**—distributing different experts across various GPUs, with routers determining the destination experts for each token to save memory and communication costs. Mainstream engines like vLLM and SGLang adopt this route by default.

**Limitations of Prior Work**: MoE models incorporate expert-/device-level balance losses during the **training phase** to enforce equilibrium, but **no balancing constraints exist during the inference phase**. If tokens in a batch are unevenly concentrated on a few experts, the corresponding GPU becomes a straggler, forcing other GPUs to remain idle while waiting for all-reduce operations, effectively degrading "parallel acceleration" into "serial execution at the slowest point."

**Key Challenge**: A natural conflict exists between the **efficiency assumption** of MoE systems (uniform token distribution across experts) and **adversarial robustness**. Once an attacker can induce "routing collapse," larger EP configurations actually deepen the vulnerability. Existing LLM-DoS research either forces models to generate extremely long outputs (where attackers must pay for every token) or relies on backdoors/prompt injection; none directly attack the MoE router itself.

**Goal**: (1) Identify prompts under strict black-box conditions (unknown model weights, routing strategy, or expert-to-GPU mapping) that reliably cause routing imbalance; (2) Quantify the actual harm of such attacks on TTFT and SLA P$_{99}$; (3) Systematically characterize which architectural or deployment factors amplify or mitigate this vulnerability.

**Key Insight**: The authors observe from the embedding space that the router is a deterministic function of the hidden state $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$. Forcing the router to select the same set of top-$k$ experts is essentially equivalent to making the hidden states of adjacent tokens **collapse into the same cluster**—i.e., minimizing the variance of embeddings at each layer $D(H^l(X))=\frac{1}{N}\sum_i\|h^l_i-\bar h^l\|_2^2$.

**Core Idea**: Instead of using white-box gradient optimization for $\arg\min_X \sum_l D(H^l(X))$, one can use the **most aggressive zero-order approximation**—directly repeating the same token $N$ times (RepetitionCurse). This similarly collapses the hidden states and disrupts routing without requiring any internal model information, and can bypass KV cache reuse by simply modifying the first token.

## Method

### Overall Architecture
The threat model is a typical black-box API attack: the service provider deploys MoE on a multi-GPU cluster using EP + prefill-decoding disaggregation. The attacker only sends prompts via a public API, disregarding output quality and aiming to inflate the **TTFT for legitimate users in the same batch**. The RepetitionCurse pipeline is minimal: (1) Select a token $t$ from the vocabulary; (2) Construct a prompt $P_t = [t, t, \dots, t]$ (excluding chat template/system prompt parts); (3) Send the request. The attack impact is amplified by the internal router collapse combined with EP scheduling.

### Key Designs

**1. Attack Objective Based on Embedding-Variance Minimization: Formulating "Router Collapse" as an Optimization Problem**

To explain why repeating tokens works, the attack objective must be formalized. The authors define layer-wise embedding variance $D(H^l(X))=\frac1N\sum_i\|h^l_i-\bar h^l\|_2^2$, expressing the optimal attack prompt as $X^*=\arg\min_X\sum_{l=1}^L D(H^l(X))$. Lower variance means hidden states collapse into the same cluster, causing the router $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$ to push all tokens toward the same top-$k$ experts. Solving this $\arg\min$ requires white-box gradients, but the authors prove: identical tokens $\Rightarrow$ minimum hidden state difference $\Rightarrow$ empirical reduction of both embedding variance and entropy. Thus, white-box optimization reduces to two hyperparameters: "which token to select" and "how many times to repeat," making it black-box feasible. This theory also identifies the root cause: MoE router balancing **implicitly assumes input tokens are sufficiently divergent in the embedding space**; once this is broken, post-training fine-tuning cannot recover it, explaining why base and instruct variants share the same vulnerability.

**2. Theoretical Maximum Imbalance (TMI): Upper Bound on Single-GPU Overload and Quantifying "EP Vulnerability"**

With an attack objective, it is necessary to know the worst-case damage. Given $E_d=|\mathcal{M}_l(d)|$ experts per card and top-$k$ routing, the worst-case load on the targeted card is $\min(k,E_d)$, while the fair share is only $k/|\mathcal{D}|$. The ratio is the Theoretical Maximum Imbalance: $\text{TMI}=\dfrac{\min(k,E_d)}{k/|\mathcal{D}|}$. This formula exhibits two behaviors: when $k\le E_d$ (sparse models like DeepSeek-V3), the attack scales linearly with EP size, theoretically creating a "perfect bottleneck"; when $k>E_d$ (e.g., Mixtral-8x7B with EP=8 where $E_d=1$ but $k=2$), it is capped at $|\mathcal{D}|\cdot E_d/k=4\times$ instead of the full $8\times$. TMI bridges service-side EP scaling for efficiency with security-side attack surface expansion, providing an actionable deployment recommendation: avoid blindly pursuing large EP sizes.

**3. Practical Black-box Relaxations: Acknowledging Gap from TMI While Maintaining General Effectiveness**

RepetitionCurse cannot reach the theoretical TMI value in real black-box scenarios. The authors identify and address two gaps. First, **Expert-to-GPU mapping is unknown**: Dynamic balancers (e.g., DeepSeek's EPLB) reorder mapping only every ~10 minutes. Within this window, mapping is static; authors use vLLM/SGLang's default sequential mapping as a baseline. Second, **Inability to specify target experts**: The attack can concentrate tokens on a set of top-$k$ experts but cannot choose which $k$. If the two selected experts happen to be distributed across different GPUs, the latency gain vanishes. The authors model this limitation as a probabilistic factor; statistically across vocabulary and layers, a stable 1.07×–2.48× TTFT amplification is still achievable. These relaxations highlight the true threat—not a sophisticated white-box attack, but this **universal yet imperfect** black-box capability: attackers only need to sweep the vocabulary to find a token that "matches" the current deployment.

### Loss & Training
This work represents a **zero-order black-box attack and system measurement**. There are no trainable parameters, gradients, or fine-tuning required. The attack involves only two hyperparameters: the choice of token and the prompt length (represented by the batch length ratio $\alpha\in\{\tfrac12,1\}$). The primary workload lies in measurement: automatically scanning 139 HuggingFace MoE configurations and benchmarking 13 representative models across a grid of EP=2/4/8/16/32.

## Key Experimental Results

### Main Results
The study covers 13 MoE models: 4 Mixtral variants ($E{=}8,k{=}2$), Qwen3-30B-A3B trio ($E{=}128,k{=}8$), GPT-OSS-20B/120B, Llama-4-Scout-17B, DeepSeek-V2-Lite, and Kimi-Linear variants. Metrics include the newly defined **LAR** (Latency Amplification Ratio): $\text{LAR}_{\text{moe}}$ is the single-layer MoE compute latency amplification, and $\text{LAR}_{\text{ttft}}$ is end-to-end TTFT amplification.

| Model | EP size | $\text{LAR}_{\text{moe}}$ ($\alpha{=}\tfrac12/1$) | $\text{LAR}_{\text{ttft}}$ ($\alpha{=}\tfrac12/1$) | Note |
|------|---------|---------------------------|---------------------------|------|
| Mixtral-8x7B | 8 | **2.01 / 2.68** | **1.61 / 2.48** | Standard 8-GPU, TTFT up to 2.48× |
| Mixtral-8x7B-It | 8 | 1.94 / 3.12 | 1.65 / 2.48 | Instruct variant equally vulnerable |
| Qwen3-30B-A3B | 32 | 2.28 / 3.22 | 1.53 / 2.15 | High-sparsity scales with large EP |
| Qwen3-Coder-30B-A3B-It | 32 | 2.32 / 3.04 | 1.51 / 2.08 | Code-specific version also affected |
| GPT-OSS-20B | 8 | 1.20 / 1.46 | (Full E2E not provided) | Small EE/small $k$ more robust |

Qualitative conclusions: (a) Mixtral variants under standard 8-GPU EP generally show TTFT amplification of **1.29×–2.48×**; (b) Across 13 models, vulnerabilities are found in diverse architectures including Qwen3, GPT-OSS, DeepSeek, and Kimi-Linear (with linear attention); (c) SLA impact: The violation rate for P$_{99}$ TTFT < 20s increases from a 1% baseline to **1.4%–13.6%**, sufficient to trigger SLA penalties and unnecessary autoscaling.

### Ablation Study
| Dimension | Key Result | Description |
|------|---------|------|
| Vocab Coverage $\mathcal{B}$ | $\mathcal{B}\to 1$ when EP=$E$ | "Randomly selecting a token to repeat" almost always causes routing concentration. |
| EP Size Sweep | $\mathcal{B}$ increases monotonically with EP | Larger EP leads to higher vulnerability, matching the TMI formula qualitatively. |
| Model Family Consistency | Mixtral-8x7B vs. Instruct/Chinese/Nous | Coverage is nearly identical; vulnerability is anchored in the **pre-training phase**. |
| Wide vs. Deep MoE | Qwen3-MoE (Large $E$ / Small $L$) vs. Mixtral | Under same compute, "wide and shallow" is a more secure MoE design. |
| Expert-GPU Mapping Bias | Near-uniform under natural text, extremely concentrated under RepetitionCurse | Confirms the root cause is router behavior, not hardware mapping. |

### Key Findings
- Strongest Single Point: Mixtral-8x7B-It at EP=8, $\alpha{=}1$ shows a single-layer MoE latency amplification of **3.12×** and TTFT amplification of **2.48×**, effectively bottlenecking an 8-GPU cluster prefill to near-single-card speed.
- Counter-intuitive Design Trade-off: **Larger EP size increases efficiency but also expands the attack surface**. The authors suggest **actively limiting EP size** until better inference-time balancing is available (unlike the industry trend).
- Low-cost Attack: Zero gradients, zero white-box access, no long generation (attacker doesn't pay for output tokens), and can bypass KV cache reuse by changing the prefix. More cost-effective than prior LLM-DoS attacks.
- Defense Implications: The 10-minute window of dynamic mappers (EPLB) is useless against second-level SLA attacks. Solutions must involve inference-time balance losses or batch-level repetition detection.

## Highlights & Insights
- Distills the systemic gap of "training-inference objective inconsistency" into a quantifiable attack vector: training enforces load balance while inference relaxes it; attackers use OOD inputs to trigger router imbalance, turning EP against the system.
- Extremist simplification of the attack—"repeating one token"—linked to the optimal white-box attack via embedding variance and TMI theory, providing serious theoretical weight to a "hack-like" method.
- Introduces LAR (MoE layer and TTFT amplification ratios) and bottleneck coverage $\mathcal{B}$, translating "routing imbalance" from a model concept into "ops SLA" language, directly usable for the inference system community.
- Systematic scan of 139 HF MoE configurations + 13 SOTA models provides empirical rules across dimensions: "wide/shallow" vs. "deep/narrow," base vs. instruct, and sparsity vs. EP size, serving as a checklist for MoE inference engine design.

## Limitations & Future Work
- Attackers cannot choose specific experts: It can only guarantee concentration on top-$k$ but not which specific $k$; dynamic balancing (like EPLB) may significantly mitigate harm if frequent enough (though 10 min windows are too long).
- Focused on prefill-bound TTFT: The impact on decoding-bound long-output scenarios and robustness under newer scheduling like chunked-prefill or speculative decoding is not yet systematically evaluated.
- Defensive side: Only conservative "limit EP size" suggestions are given, without proposing router-side adversarial balancers, token-level repetition detectors, or online mitigation based on dispatch monitoring.
- Model scale: While covering mainstream models, the 13 evaluated are $\le 120$B; whether TMI-level amplification replicates on massive clusters like DeepSeek-V3 (32 prefill EP) requires vendor-side validation.

## Related Work & Insights
- **vs. Gao et al. 2024 / Zhang et al. 2024 (Long-output LLM DoS)**: These force models to generate to max tokens to drain resources, costing the attacker money. This work uses short inputs to crush the prefill stage, offering better cost-benefit and stealth.
- **vs. Li et al. 2025b ("Endless thinking" reasoning attacks)**: Those target reasoning models to loop; this work attacks the **hardware utilization per token**, orthogonal to reasoning capability and effective for all MoE services.
- **vs. EPLB / DeepSeek-V3 Deployment (DeepSeek-AI 2024b/2025)**: EPLB is vendor-side dynamic mapping for load drift. This work reveals that its remapping cycle is insufficient for second-level SLA attacks, effectively posing as a new adversarial benchmark for EPLB.
- **vs. grouped GEMM / vLLM / SGLang MoE Kernels**: Previous optimizations focused on throughput and latency; this work indicates that "adversarial robustness" must become a first-class citizen, or optimization will lead to fragility.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes MoE router imbalance as a DoS attack vector for the first time with an elegantly simple method.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 models × 5 EP sizes plus a survey of 139 HF configurations is highly convincing, though missing real massive cluster (DeepSeek-V3 level) testing.
- Writing Quality: ⭐⭐⭐⭐ Seamlessly blends system, adversarial, and theoretical TMI perspectives. Symbols are dense but structure is clear.
- Value: ⭐⭐⭐⭐⭐ Immediately actionable for MoE providers (limit EP, add monitoring) and opens the "adversarial robustness" field for MoE inference systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)

</div>

<!-- RELATED:END -->
