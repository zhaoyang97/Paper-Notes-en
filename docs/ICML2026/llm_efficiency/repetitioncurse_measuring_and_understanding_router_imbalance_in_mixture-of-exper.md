---
title: >-
  [Paper Note] RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress
description: >-
  [ICML 2026][LLM Efficiency][TTFT] By providing MoE large models with minimalist OOD prompts such as "repeating the same token N times," the authors discover that the router directs almost all tokens to a fixed small set of top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a bottleneck on a single card while leaving other GPUs idle
tags:
  - ICML 2026
  - LLM Efficiency
  - TTFT
date: 2026-05-08
content_hash: 99df101f91450f41
---
# RepetitionCurse: Measuring and Understanding Router Imbalance in Mixture-of-Experts LLMs under DoS Stress

**Conference**: ICML 2026  
**arXiv**: [2512.23995](https://arxiv.org/abs/2512.23995)  
**Code**: To be confirmed  
**Area**: LLM Efficiency / MoE Systems / Inference Security  
**Keywords**: MoE Routing Imbalance, Expert Parallelism, DoS Attack, TTFT, Black-box Attack

## TL;DR
By providing MoE large models with minimalist OOD prompts such as "repeating the same token N times," the authors discover that the router directs almost all tokens to a fixed small set of top-$k$ experts. Under Expert Parallelism (EP) deployment, this creates a bottleneck on a single card while leaving other GPUs idle, increasing TTFT by 20%–148% on 8-GPU clusters. This effectively turns the MoE parallel accelerator into a DoS attack surface.

## Background & Motivation

**Background**: Modern LLMs commonly utilize MoE (such as Mixtral / Qwen3-MoE / DeepSeek-V3 / GPT-OSS / Llama-4-Scout, etc.) to expand capacity without scaling inference costs. Industrial deployment is paired with **Expert Parallelism (EP)**—placing different experts on different GPUs and relying on a router to decide which experts each token goes to, thereby saving VRAM and communication overhead. Mainstream engines like vLLM and SGLang use this approach by default.

**Limitations of Prior Work**: MoE incorporates expert-/device-level balance loss during the **training phase** to force equilibrium; however, there are **no balance constraints during the inference phase**. If tokens in a batch are unevenly concentrated on a few experts, the corresponding GPU becomes a straggler, forcing other GPUs to remain idle while waiting for it to complete all-reduce, effectively degrading "parallel acceleration" into the "slowest serial task."

**Key Challenge**: A natural contradiction exists between the **efficiency assumption** of MoE systems (uniform distribution of tokens across experts) and **adversarial robustness**. Once an attacker can induce "routing collapse," a larger EP size leads to deeper victimization. Existing LLM-DoS works either force the model to generate ultra-long outputs (where the attacker must pay for every token) or rely on backdoors/prompt injection; none directly attack the MoE router itself.

**Goal**: (1) Find attack prompts that reliably cause routing imbalance under a strict black-box setting (no knowledge of model weights, routing strategies, or expert-to-GPU mapping); (2) Quantify the actual harm of such attacks on TTFT and SLA P$_{99}$; (3) Systematically characterize which architectural/deployment factors amplify or mitigate this vulnerability.

**Key Insight**: The authors observe from the embedding space: the router is a deterministic function of the hidden state $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$. To force the router to select the same set of top-$k$ experts is essentially equivalent to making the hidden states of adjacent tokens **collapse into the same cluster**—i.e., minimizing the variance of embeddings at each layer $D(H^l(X))=\frac{1}{N}\sum_i\|h^l_i-\bar h^l\|_2^2$.

**Core Idea**: Instead of using white-box gradient optimization for $\arg\min_X \sum_l D(H^l(X))$, it is better to employ the **most aggressive zeroth-order approximation**—directly making the input the same token repeated N times (RepetitionCurse). This similarly collapses the hidden states and breaks the routing without requiring any internal model information, and it can bypass KV cache reuse by simply changing the initial token.

## Method

### Overall Architecture
The threat model is a typical black-box API attack: the service provider deploys MoE on a multi-GPU cluster using EP + prefill-decoding disaggregation. The attacker sends prompts from a public API, indifferent to model output quality, with the sole aim of inflating the **TTFT for legitimate users in the same batch**. The RepetitionCurse pipeline is extremely short: (1) Select a token $t$ from the vocabulary; (2) Construct a prompt $P_t = [t, t, \dots, t]$ (excluding the chat template/system prompt parts); (3) Send the request. The attack effect is amplified by the internal collapse of the router combined with EP scheduling.

### Key Designs

**1. Attack goal based on embedding-variance minimization: Formulating "forcing router collapse" as an analytical optimization problem**

To explain "why repeating tokens works," the attack goal must first be formalized. The authors define layer-wise embedding variance $D(H^l(X))=\frac1N\sum_i\|h^l_i-\bar h^l\|_2^2$ and express the optimal attack prompt as $X^*=\arg\min_X\sum_{l=1}^L D(H^l(X))$. The smaller the variance, the more the hidden states of adjacent tokens collapse into the same cluster, causing the router $G(h)=\text{Softmax}(h\cdot W_{\text{router}})$, as a deterministic function, to push all of them toward the same set of top-$k$ experts. Solving this $\arg\min$ directly requires white-box gradients, but the authors prove: making all tokens identical $\Rightarrow$ minimal difference between adjacent hidden states $\Rightarrow$ empirically lowering both embedding variance and embedding entropy (cf. Skean et al., 2025). Thus, white-box optimization degrades into two hyperparameters: "which token to choose + how many times to repeat," which is feasible in a black-box setting. This theory also exposes the root cause of the vulnerability: the balancing capability of the MoE router **implicitly assumes that input tokens are sufficiently divergent in the embedding space**. Once this assumption is broken, no amount of post-training fine-tuning can fix it—this explains why the vulnerability is identical across base and instruct variants.

**2. Theoretical Maximum Imbalance (TMI): Providing an upper bound for single-card overload and quantifying "the larger the EP, the more fragile"**

With the attack goal established, it is necessary to know the extent of the worst-case damage. Assuming each card hosts $E_d=|\mathcal{M}_l(d)|$ experts and the router selects top-$k$, the worst-case load on the targeted card is $\min(k,E_d)$, whereas its fair share is only $k/|\mathcal{D}|$. The ratio between the two is the Theoretical Maximum Imbalance $\text{TMI}=\dfrac{\min(k,E_d)}{k/|\mathcal{D}|}$. This formula exhibits two-stage behavior: when $k\le E_d$ (sparse types like DeepSeek-V3), the attack scales linearly with EP size, theoretically creating a "perfect bottleneck"; when $k>E_d$ (e.g., Mixtral-8x7B where each card has only $E_d=1$ expert at EP=8, yet $k=2$ must be selected), it is capped at $|\mathcal{D}|\cdot E_d/k=4$ times instead of the full 8 times. The significance of TMI lies in binding "scaling EP for efficiency" on the operations side and "scaling the attack surface" on the security side into the same trade-off, providing an actionable deployment recommendation: do not blindly pursue large EP.

**3. Two pragmatic degradations for black-box execution: Acknowledging that TMI upper bounds are unreachable but still universally effective**

RepetitionCurse cannot reach TMI theoretical values in a real-world black-box scenario. The authors honestly locate two gaps and address each. The first is **agnosticism toward Expert-GPU mapping**: dynamic balancers (like DeepSeek’s EPLB) re-rank only every ~10 min due to high overhead, meaning mapping is static within the re-ranking window. The authors thus use the "sequential allocation" mapping default in vLLM/SGLang as an analysis baseline. The second is the **inability to specify target experts**: the attack can concentrate tokens into a certain set of top-$k$ experts but cannot choose which $k$ they are. If the two experts hit happen to be assigned to different GPUs (e.g., in some layers of Mixtral at EP=2), the latency gain vanishes. The authors model this limitation as a probabilistic factor, finding that 1.07×–2.48× TTFT amplification can still be consistently achieved in a statistical sense across vocabularies and layers. These two degradations actually highlight the true source of the threat—not a meticulously constructed optimal white-box attack, but this **universal yet imperfect** black-box capability: as long as an attacker scans the vocabulary, they can always find a token that "just happens to hit" for the current deployment.

### Loss & Training
Ours is a **zeroth-order black-box attack + system measurement**, with no trainable parameters, gradients, or fine-tuning required. The attack involves only two hyperparameters: which token to use and the prompt length (represented by the ratio $\alpha\in\{\tfrac12,1\}$ within the batch, corresponding to the two columns in Table 2). The true workload lies in the measurement side: automatically scanning 139 HuggingFace MoE configurations and running full-grid benchmarks for EP=2/4/8/16/32 across 13 representative models.

## Key Experimental Results

### Main Results
The study covers 13 MoEs: 4 from the Mixtral family ($E{=}8,k{=}2$) + Qwen3-30B-A3B trio ($E{=}128,k{=}8$) + GPT-OSS-20B/120B + Llama-4-Scout-17B + DeepSeek-V2-Lite + Kimi-Linear duo. The metrics used are the authors' newly defined **LAR** (Latency Amplification Ratio): $\text{LAR}_{\text{moe}}$ is the multiplication factor for single-layer MoE computation latency, and $\text{LAR}_{\text{ttft}}$ is the end-to-end TTFT amplification.

| Model | EP size | $\text{LAR}_{\text{moe}}$ ($\alpha{=}\tfrac12/1$) | $\text{LAR}_{\text{ttft}}$ ($\alpha{=}\tfrac12/1$) | Remarks |
|------|---------|---------------------------|---------------------------|------|
| Mixtral-8x7B | 8 | **2.01 / 2.68** | **1.61 / 2.48** | Classic 8-GPU deployment, TTFT pulled to 2.48× |
| Mixtral-8x7B-It | 8 | 1.94 / 3.12 | 1.65 / 2.48 | instruct variants equally fragile |
| Qwen3-30B-A3B | 32 | 2.28 / 3.22 | 1.53 / 2.15 | Highly sparse models continue to amplify at high EP |
| Qwen3-Coder-30B-A3B-It | 32 | 2.32 / 3.04 | 1.51 / 2.08 | Code-specific version also affected |
| GPT-OSS-20B | 8 | 1.20 / 1.46 | (End-to-end data not fully provided in snippet) | Small EE/small $k$ models are more resilient |

Qualitative conclusions: (a) Mixtral family models generally show TTFT amplification of **1.29×–2.48×** under common 8-GPU EP; (b) Across 13 models, covering different architectures like Qwen3/GPT-OSS/DeepSeek and even Kimi-Linear with linear attention, all hit the vulnerability; (c) SLA impact: The violation rate for P$_{99}$ TTFT < 20s rose from a 1% baseline to **1.4%–13.6%**, sufficient to trigger SLA penalties and unnecessary autoscaling.

### Ablation Study

| Dimension | Key Findings | Description |
|------|---------|------|
| Vocabulary coverage $\mathcal{B}$ | $\mathcal{B}\to 1$ when EP=$E$ | "Randomly picking a token to repeat" almost always causes routing concentration |
| EP size scanning | $\mathcal{B}$ increases monotonically with EP | Vulnerability increases with EP size, qualitatively consistent with the TMI formula |
| Consistency within model families | Coverage for Mixtral-8x7B is nearly identical to its instruct/Chinese/Nous fine-tuned versions | The vulnerability was planted during the **pre-training phase** and cannot be changed by post-training |
| Wide vs. Deep MoE | Qwen3-MoE with large $E$ / small $L$ performs better than deep/narrow Mixtral | Under same compute, "wide and shallow" is a safer MoE design |
| Expert-GPU mapping bias (Fig. 3) | Router is near-uniform under natural text, extremely concentrated under RepetitionCurse | Verifies root cause is router behavior, not hardware |

### Key Findings
- Strongest single point: Mixtral-8x7B-It at EP=8, $\alpha{=}1$ shows a single-layer MoE latency amplification of **3.12×** and a TTFT amplification of **2.48×**, which means the prefill phase is nearly bottle-necked to a single card on an 8-GPU cluster.
- Most counter-intuitive design trade-off: **Larger EP size increases efficiency but also increases the attack surface**. The authors suggest **actively limiting EP size** until better inference-time balancing strategies are available (contrary to the industry trend of pursuing large EP).
- The attack prompt is extremely cheap: no gradients, no white-box access, no ultra-long generation (attacker doesn't pay for output tokens), and it can bypass KV cache reuse by modifying the first token. It offers much higher ROI than existing LLM-DoS attacks.
- Defense implications: The 10-minute window of dynamic mappers (EPLB) is almost useless for second-level SLA attacks; one must either bring balance loss into inference or implement batch-level detection for repetitive patterns.

## Highlights & Insights
- Distills a systemic gap—the "training-inference objective mismatch"—into a quantifiable attack vector: while load balance is forced during training, it is completely relaxed during inference. Attackers only need to trigger router imbalance with OOD inputs to turn EP against the system.
- The attack form is extremely simplified— "repeating the same token"—yet the authors use embedding variance + TMI theory to link it to optimal white-box attacks, providing a serious theoretical explanation for what "looks like a hack."
- Introduces two new metrics, LAR (MoE layer latency amplification ratio, TTFT amplification ratio) and bottleneck coverage $\mathcal{B}$, translating "routing imbalance" from a model-side concept into "operations SLA-side" language, making it directly useful for the inference systems community.
- Systematically scans 139 HF MoE configurations + 13 runnable SOTA models, providing empirical laws across multiple dimensions such as "wide/shallow" vs. "deep/narrow," base vs. instruct, and sparsity vs. EP size. This serves as a checklist for future MoE inference engine designs.

## Limitations & Future Work
- Attackers cannot choose specific experts: It can only be guaranteed that tokens concentrate on top-$k$ experts, but which $k$ experts cannot be controlled; the harm would be significantly mitigated if dynamic balancing like EPLB were frequent enough (though the authors note the 10-min window is too long).
- The experiments mainly characterize TTFT in prefill-bound scenarios; the impact on long-output decoding-bound scenarios and robustness under new scheduling strategies like chunked-prefill / speculative decoding have not been systematically evaluated.
- Regarding defense, this paper only offers the conservative suggestion of "limiting EP size," failing to propose router-side adversarial balancers, token-level repetition detectors, or online mitigation based on dispatch monitoring; these are natural follow-up directions.
- While the 13 models cover the mainstream, they are all $\le 120$B open-source MoEs. Whether TMI-level amplification can be replicated on true clusters for ultra-large EP (e.g., DeepSeek-V3 with 32 prefill) still requires validation from vendors.

## Related Work & Insights
- **vs. Gao et al. 2024 / Zhang et al. 2024 (Long-output LLM DoS)**: These works force models to generate up to max tokens to drain backend resources, requiring attackers to pay for each generated token; Ours uses extremely short inputs to crush the prefill phase, offering higher cost-efficiency and stealth.
- **vs. Li et al. 2025b ("Endless Thinking" reasoning attacks)**: That line of work targets reasoning models to trap them in infinite loops, attacking the number of tokens; Ours attacks the **hardware utilization per token**, which is orthogonal to reasoning capability and effective against all MoE inference services.
- **vs. EPLB / DeepSeek-V3 deployment experiences (DeepSeek-AI 2024b/2025)**: EPLB is a vendor-side dynamic mapping for load drift; this paper reveals that its re-ranking period is insufficient for second-level SLA attacks, effectively setting a new adversarial benchmark for EPLB.
- **vs. grouped GEMM / vLLM / SGLang MoE kernel work**: Previous optimization goals were throughput and latency; this paper reminds us that "adversarial robustness" must become a third first-class citizen, otherwise, the more a system is optimized, the more fragile it becomes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally models MoE router imbalance as a DoS attack vector for the first time, with an exceptionally concise attack method.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 models × 5 EP sizes paired with a statistical survey of 139 HF configurations is quite persuasive, though it lacks testing on true ultra-large clusters (DeepSeek-V3 level).
- Writing Quality: ⭐⭐⭐⭐ Successfully connects systems perspectives, adversarial perspectives, and theoretical TMI; notation is slightly heavy but the structure is clear.
- Value: ⭐⭐⭐⭐⭐ Immediately actionable for all vendors deploying MoE (limit EP size, add router monitoring) and opens a new field of "adversarial robustness" for MoE inference systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](softmoe_soft_differentiable_routing_for_mixture-of-experts_in_llms.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)

</div>

<!-- RELATED:END -->
