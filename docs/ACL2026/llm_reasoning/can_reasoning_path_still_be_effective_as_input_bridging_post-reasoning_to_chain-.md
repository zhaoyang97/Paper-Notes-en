---
title: >-
  [Paper Note] Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression
description: >-
  [ACL2026][LLM Reasoning][post-reasoning] This paper proposes post-reasoning and UCoT: a lightweight compressor first generates soft tokens representing the reasoning path via a single forward pass…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "post-reasoning"
  - "CoT compression"
  - "soft tokens"
  - "inference latency"
  - "UCoT"
date: 2026-05-08
content_hash: 40374dc213dab599
---

# Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression

**Conference**: ACL2026  
**arXiv**: [2510.08647](https://arxiv.org/abs/2510.08647)  
**Code**: No public code link found in cache  
**Area**: LLM Reasoning / CoT Compression / Inference Acceleration  
**Keywords**: post-reasoning, CoT compression, soft tokens, inference latency, UCoT

## TL;DR
This paper proposes post-reasoning and UCoT: a lightweight compressor first generates soft tokens representing the reasoning path via a single forward pass, which the executor then uses as input context for short-output reasoning, significantly reducing CoT tokens and latency while maintaining accuracy.

## Background & Motivation
**Background**: Long Chain-of-Thought (CoT) has become an essential means to enhance LLM capabilities in math, scientific QA, and code reasoning. Models like DeepSeek-R1 and the o-series demonstrate that test-time computation scaling yields stronger reasoning.

**Limitations of Prior Work**: The primary cost of long CoT stems from autoregressive output. Models must generate intermediate reasoning token-by-token, leading to high latency and token costs for complex problems. Existing CoT compression methods mostly focus on the output side (e.g., prompting for brevity, truncating reasoning, or training on short CoT), which often suffers from the loss of critical reasoning information.

**Key Challenge**: CoT serves as both an output and a "self-generated context" constructed by the model for the final answer. While providing the reasoning path as input could theoretically shorten the output, generating explicit text CoT is still expensive, and low-quality external CoT can hurt accuracy.

**Goal**: To verify whether "reasoning paths are still effective as input" and design an efficient framework that avoids autoregressive generation of long context CoT, supplementing reasoning information at the input side to shorten the output side.

**Key Insight**: The authors first define post-reasoning: the executor receives a question and an external contextual CoT, then continues reasoning to output an answer. A pilot study shows output tokens can be reduced by over 80%, but performance depends on the length and quality of the contextual CoT. Consequently, they further compress explicit CoT into soft tokens.

**Core Idea**: Instead of rigidly restricting the model's thinking, high-quality reasoning priors are compressed into input-side soft tokens, allowing the executor to access missing reasoning information even under a short output budget.

## Method
UCoT consists of three components: a compressor, a projector, and an executor. The compressor learns to map the long CoT that the executor would have generated into hidden states at placeholder positions; the projector maps these hidden states into the executor's embedding space; the executor then utilizes these soft tokens to provide shorter but accurate reasoning and answers under a restricted output budget.

### Overall Architecture
In the training phase, the executor first generates high-quality CoT for each question to obtain $(Q, C, A)$ data. Then, a lightweight compressor is trained to produce soft tokens from the question and `[ucot]` placeholders, with a requirement to reconstruct the original CoT. Finally, the executor is trained to recover the original reasoning semantics and maintain answer confidence using the soft tokens and a truncated output budget. In the inference phase, the compressor performs a single forward pass to generate soft tokens, and the executor outputs a shorter answer based on these tokens.

### Key Designs
1. **Post-reasoning Paradigm**:
	- **Function**: Transitions CoT from pure output to input context, reducing the autoregressive reasoning length needed by the executor.
	- **Mechanism**: While vanilla reasoning is denoted as $\{C,A\}=LLM(Q)$, post-reasoning is modified to $\{\hat C,\hat A\}=LLM(Q \oplus C')$, where $C'$ is the externally provided contextual CoT.
	- **Design Motivation**: If the model already has part of the reasoning path as context, it does not need to generate the full CoT from scratch. Pilot studies on GSM8K and MATH-500 show output tokens can be reduced by over 80%.

2. **Soft-token contextual CoT**:
	- **Function**: Avoids the autoregressive generation cost of explicit contextual CoT.
	- **Mechanism**: The compressor appends $M$ `[ucot]` placeholders to the input. After a single forward pass, the hidden states $H_n$ corresponding to these placeholders are used as soft tokens. The training objective requires the compressor to reconstruct the original CoT $C_n$ generated by the executor based on $H_n$.
	- **Design Motivation**: Although explicit text CoT helps post-reasoning, it is slow to generate. Soft tokens place reasoning semantics in a continuous space, replacing long text output with a single forward pass.

3. **Reward-guided executor utilization**:
	- **Function**: Ensures the executor actually utilizes the soft tokens instead of ignoring them and continuing to generate long CoT.
	- **Mechanism**: The projector maps the compressor's soft tokens into the executor's embedding space. During executor training, a Cutoff is used to limit explicit CoT length, forcing the model to rely on soft tokens to supplement truncated logic. A semantic loss aligns the UCoT reasoning representation with the original long CoT representation, while a reward factor penalizes discrepancies in answer confidence between compressed and original reasoning.
	- **Design Motivation**: Continuous prompts or latent CoT may not be converted into effective reasoning without utilization constraints. Output budgets combined with semantic/confidence constraints turn soft tokens into usable reasoning context.

### Loss & Training
In the compressor phase, a reconstruction objective $L_c=E_D[-\log P_{M_c}(C_n|H_n)]$ is minimized to compress long CoT information into soft tokens. In the executor phase, the output CoT is first truncated to $\bar C_n$ based on a compression ratio $\alpha$. Then, a semantic loss $L_{sem}=E_D[Dist(H_{UCoT},H_{CoT})]$ aligns the final hidden states of the compressed and original long reasoning, while a reward factor $R=E_D[(r_{UCoT}-r_{CoT})^2]$ maintains answer confidence. The final executor objective is $L_e=L_{sem}\cdot R$. The main experiments use Qwen2.5-1.5B-Instruct as the compressor, with Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct as executors.

## Key Experimental Results

### Main Results
| Backbone / Dataset | Method | Compression Ratio | Acc. | Tokens | Latency | Description |
|------------------|------|--------|------|--------|---------|------|
| Qwen2.5-7B / GSM8K | Original | - | 92.17 | 298.63 | 3.83s | Original Long CoT |
| Qwen2.5-7B / GSM8K | UCoT | 0.5 | 86.55 | 140.36 | 1.86s | Tokens halved, outperforms CoD/Tokenskip |
| Qwen2.5-7B / MATH-500 | Original | - | 61.60 | 571.64 | 6.35s | Original Long CoT |
| Qwen2.5-7B / MATH-500 | UCoT | 0.5 | 53.70 | 280.10 | 3.17s | Significantly higher than Tokenskip (47.10) |
| Llama3.1-8B / GSM8K | Original | - | 87.26 | 212.13 | 2.44s | Original Long CoT |
| Llama3.1-8B / GSM8K | UCoT | 0.5 | 83.62 | 101.82 | 1.26s | Maintains high accuracy with near-half latency |

| Long CoT Scenario | Method | Compression Ratio | Acc. | Tokens | vs. Tokenskip |
|-------------|------|--------|------|--------|----------------|
| Qwen3-8B / GPQA | Tokenskip | 0.5 | 54.23 | 4388.54 | Baseline |
| Qwen3-8B / GPQA | UCoT | 0.5 | 56.86 | 4065.58 | Higher Acc. and fewer tokens |
| Qwen3-8B / HumanEval | Tokenskip | 0.5 | 42.52 | 1025.00 | Baseline |
| Qwen3-8B / HumanEval | UCoT | 0.5 | 46.55 | 1021.93 | Gain 4.03 Acc. |
| DeepSeek-R1-Distill-Qwen-7B / HumanEval | Tokenskip | 0.5 | 43.79 | 900.04 | Baseline |
| DeepSeek-R1-Distill-Qwen-7B / HumanEval | UCoT | 0.5 | 43.96 | 870.68 | ~50.67% token reduction, slightly higher Acc. |

### Ablation Study
| Configuration | Qwen2.5 GSM8K Acc. / Tokens | Qwen2.5 MATH Acc. / Tokens | Description |
|------|-----------------------------|-----------------------------|------|
| UCoT full | 87.98 / 194.63 | 58.80 / 388.72 | Full model, compression ratio 0.7 |
| w/o ST | 73.23 / 220.99 | 47.50 / 390.84 | Acc. drops significantly without soft tokens |
| w/o $L_{sem}$ | 87.32 / 274.74 | 59.70 / 554.49 | Acc. maintained, but compression task fails |
| w/o R | 71.53 / 206.05 | 44.90 / 417.30 | Reward factor critical for answer ability |

### Key Findings
- It is difficult to force models into a target compression ratio via prompting alone; direct truncation leads to severe performance collapse. UCoT’s advantage lies in supplementing information at the input rather than brute-force compression at the output.
- Soft tokens are core to performance. Removing soft tokens drops Qwen2.5 performance on GSM8K from 87.98 to 73.23, indicating that the continuous reasoning context provided by the compressor is indeed used by the executor.
- $L_{sem}$ acts as a "compression constraint": without it, accuracy may not collapse immediately, but token counts remain close to original lengths, suggesting the model has not learned to complete tasks with short outputs.

## Highlights & Insights
- Viewing CoT as "input context the model generates for itself" is insightful; it shifts the CoT compression problem from "outputting less" to "providing sufficient reasoning priors at the input side."
- The compressor-executor labor division in UCoT is natural: the small model handles low-latency generation of soft context, while the large model performs final reasoning. This is compatible with small-model routing or preprocessing strategies in deployment.
- The paper reminds us that compressing long reasoning cannot be done by simply deleting words. It is the reasoning information that needs compression, and soft tokens provide a channel for representation that does not require full discretization.

## Limitations & Future Work
- UCoT requires training a compressor, projector, and executor, resulting in higher deployment complexity than pure prompt-based or training-free methods; availability for closed-source models is also limited.
- The interpretability of soft tokens is weak. Although the authors attempt analysis via information gain and decoded soft tokens, they remain less auditable by humans than text CoT.
- Training data depends on the executor generating high-quality CoT first; if original long CoTs contain bias or hallucinations, the compressor may learn them.
- Future work could explore universal soft reasoning caches across tasks/executors, or integrate UCoT with system-level accelerations like speculative decoding, KV cache reuse, or routers.

## Related Work & Insights
- **vs. Prompt / CoD**: Prompt and CoD attempt to constrain output formats to make models write more concisely; UCoT supplements continuous reasoning context at the input, making accuracy more stable at equivalent compression ratios.
- **vs. Tokenskip**: Tokenskip reduces output by training on short CoT but may lose reasoning information; UCoT uses soft tokens to preserve omitted logic, outperforming Tokenskip by over 3.08 points on GSM8K at a 0.5 compression ratio.
- **vs. latent CoT / continuous reasoning**: While related methods also compress intermediate reasoning into continuous representations, this paper emphasizes the post-reasoning input paradigm and forces the executor to utilize these representations through semantic loss and reward factors.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The problem reformulation of post-reasoning + soft contextual CoT is very clear and distinct from conventional output compression.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Main experiments, ablations, and long CoT generalization are solid, though system costs and cross-model generalization could be explored further.
- **Writing Quality**: ⭐⭐⭐⭐☆ Diagrams and training flows are clear; some formulas are dense but logical.
- **Value**: ⭐⭐⭐⭐⭐ Direct reference value for LLM inference acceleration, CoT compression, and continuous reasoning representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ICLR 2026\] When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning Models](../../ICLR2026/llm_reasoning/when_reasoning_meets_compression_understanding_the_effects_of_pruning_and_quant.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](../../NeurIPS2025/llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)
- [\[ACL 2026\] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning](long-context_reasoning_through_proxy-based_chain-of-thought_tuning.md)
- [\[ICML 2026\] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](../../ICML2026/llm_reasoning/provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)

</div>

<!-- RELATED:END -->
