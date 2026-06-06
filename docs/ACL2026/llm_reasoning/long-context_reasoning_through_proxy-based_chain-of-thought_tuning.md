---
title: >-
  [Paper Note] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning
description: >-
  [ACL2026][LLM Reasoning][Long-context reasoning] ProxyCoT utilizes short and sufficient proxy contexts to obtain high-quality reasoning trajectories…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Long-context reasoning"
  - "proxy context"
  - "CoT distillation"
  - "RLVR"
  - "ProxyCoT"
date: 2026-05-08
content_hash: 98878ab090cd4929
---

# Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning

**Conference**: ACL2026  
**arXiv**: [2605.20201](https://arxiv.org/abs/2605.20201)  
**Code**: https://github.com/oaimli/ProxyCoT  
**Area**: LLM Reasoning / Long Context / Chain-of-Thought  
**Keywords**: Long-context reasoning, proxy context, CoT distillation, RLVR, ProxyCoT  

## TL;DR
ProxyCoT utilizes short and sufficient proxy contexts to obtain high-quality reasoning trajectories, then distills these trajectories into full long-context inputs. This enables a 4B model to significantly improve long-context reasoning on SciTrek, HotpotQA, and Loong while reducing CoT tokens during inference.

## Background & Motivation
**Background**: Modern LLMs have extended context windows to millions of tokens, but the ability to read long text does not equate to stable reasoning over it. Many tasks only require locating a small amount of evidence from long inputs followed by comparison, filtering, aggregation, or multi-hop reasoning.

**Limitations of Prior Work**: Common methods for enhancing reasoning include CoT distillation or Reinforcement Learning (RL). The former requires large teacher models to generate high-quality reasoning trajectories, while the latter requires extensive sampling. Both are feasible in short contexts but become prohibitively expensive at 64K, 128K, or longer; furthermore, teacher models themselves may generate unreliable trajectories in long contexts.

**Key Challenge**: Reasoning logic in long-context tasks often depends on a small segment of key evidence, yet training and supervision are forced to process the entire long input. Models perform better on proxy contexts for the same reasoning task but lose accuracy on full contexts due to failures in evidence localization and grounding.

**Goal**: To utilize proxy contexts to obtain correct CoTs at low cost and train models to reproduce these reasoning trajectories under full-context conditions, thereby migrating reasoning behaviors learned in short contexts to long inputs.

**Key Insight**: The authors define a proxy context as a short input containing sufficient evidence, satisfying $|C^p|\ll |C|$, while maintaining consistency in questions, answers, and reasoning steps with the full context. Thus, the proxy context serves as an upper bound for "perfect retrieval" and a low-cost source of reasoning supervision.

**Core Idea**: First generate correct CoTs on the proxy context using a strong teacher or RLVR, then use SFT to train the student model to generate the same reasoning trajectories given the full long context.

## Method
ProxyCoT is a two-stage training framework. The first stage focuses on short proxy contexts to obtain high-quality, low-cost reasoning trajectories. The second stage binds these trajectories to the full long context, teaching the model to locate and use the corresponding evidence within long inputs.

### Overall Architecture
Each sample consists of a question $q$, full context $C$, proxy context $C^p$, and answer $a$. Stage 1 obtains the reasoning trajectory $t$ on $(q,C^p)$. If a strong teacher is available, ProxyCoT-ZS is used: Qwen3-235B-A22B-Thinking samples multiple times on the proxy context, keeping only trajectories with correct answers. If no suitable teacher exists, ProxyCoT-RL is used: the target model first learns to generate correct reasoning trajectories on the proxy context via RLVR.

Stage 2 employs SFT, using trajectories from Stage 1 as supervision, but with the input changed to $(q,C)$. This step requires the model to reproduce proxy-derived CoTs within the full long context, thereby learning evidence grounding. The paper validates this on Qwen3-4B-Instruct-2507 and Gemma3-4B-IT across tasks including SciTrek and HotpotQA, with out-of-domain testing on Loong.

### Key Designs
1. **Proxy Context as Reasoning-Equivalent Short Input**:
	- **Function**: Reduces the context length and computational cost of obtaining reasoning trajectories.
	- **Mechanism**: For SciTrek, questions often stem from metadata like titles, authors, and references; thus, structured metadata serves as the proxy. HotpotQA uses manually annotated supporting sentences as the proxy.
	- **Design Motivation**: If short evidence is sufficient to answer the question, training for reasoning on short inputs is more efficient than repeated sampling on long inputs.

2. **Two CoT Acquisition Paths**:
	- **Function**: Covers scenarios both with and without strong teacher models.
	- **Mechanism**: ProxyCoT-ZS lets Qwen3-235B-A22B-Thinking generate CoTs on the proxy context, retaining only correct ones. ProxyCoT-RL optimizes the target model directly on the proxy context using DAPO / RLVR, with rewards based on F1 and exact match.
	- **Design Motivation**: Large teachers are cheaper and more reliable on short proxies; RLVR also becomes more trainable due to shorter inputs, avoiding direct sampling on the full context.

3. **Long-Context Grounding SFT**:
	- **Function**: Transfers short-context reasoning capabilities to full long inputs.
	- **Mechanism**: The SFT objective maximizes $p_\theta(t\mid q,C)$, generating the correct reasoning trajectory obtained from the proxy context given the full context.
	- **Design Motivation**: Training only on proxies makes the model dependent on short evidence formats and does not guarantee usability for long inputs; the second stage forces the model to align the same reasoning steps within long texts.

### Loss & Training
The SFT for ProxyCoT-ZS uses $\mathcal{L}_{SFT}=-\mathbb{E}[\log p_\theta(t\mid q,C)]$. ProxyCoT-RL first optimizes on the proxy context using RLVR with reward $R(a,\hat{a})=F1(a,\hat{a})+\mathds{1}_{a==\hat{a}}$, then continues with SFT from the RL checkpoint. Implementation-wise, RL uses OpenRLHF with a batch size of 64, max generation length of 2,048, and an actor learning rate of $5e{-7}$, sampling 8 trajectories per prompt over 10 epochs. SFT uses a batch size of 64 and a learning rate of $5e{-6}$ with a linear warmup for the first 10% of steps.

## Key Experimental Results

### Main Results
| Dataset / Model | Method | Proxy Metric | Full Metric | Description |
|--------|------|-----------|-----------|------|
| SciTrek / Qwen3-4B | Zero-shot | 67.2 | 30.8 | Significant drop on full context |
| SciTrek / Qwen3-4B | ProxyCoT-ZS | 67.8 | 38.8 | Large teacher proxy CoT distillation is effective |
| SciTrek / Qwen3-4B | ProxyCoT-RL | 88.5 | 46.5 | Approaches Qwen3-235B-Thinking full (48.8) |
| SciTrek / Gemma3-4B | Zero-shot | 34.2 | 3.0 | Very weak baseline long-context capability |
| SciTrek / Gemma3-4B | ProxyCoT-RL | 69.8 | 43.7 | Most significant improvement |
| HotpotQA / Qwen3-4B | Zero-shot | 91.3 | 44.5 | Strong on proxy, still weak on full |
| HotpotQA / Qwen3-4B | ProxyCoT-RL | 92.1 | 52.7 | Optimal on full context |
| Loong / Gemma3-4B | Zero-shot → ProxyCoT-RL | - | Financial 25.85 → 32.05; Academic 3.55 → 24.32 | Generalization without retraining |

### Ablation Study
| Analysis Item | Configuration | Key Result | Description |
|------|------|---------|------|
| CoT token | Qwen3-4B on SciTrek full | Zero-shot 1,744 tokens / 30.8 EM; SFT on full CoT 6,683 / 31.6; ProxyCoT-RL 617 / 46.5 | ProxyCoT-RL is more accurate and concise |
| Stage Ablation | Qwen3-4B | Stage1+Stage2 full 46.5; only RLVR full 29.0; only SFT full 46.3 | SFT grounding is key for Qwen3; RL improves proxy capability |
| Stage Ablation | Gemma3-4B | Stage1+Stage2 full 43.7; only RLVR full 8.0; only SFT full 37.3 | Models with weak long-context ability rely more on the two-stage combination |
| Proxy Type | SciTrek | Random sentences 3.4; Title/Author/Ref 24.6; Structured metadata 91.5 | Proxy quality determines the effectiveness of RLVR reasoning |
| Proxy Noise | SciTrek | Oracle:Noise 1:5 = 85.3, 1:0 = 91.5 | Robustness shown despite limited decline after adding noise |
| Proxy Noise | HotpotQA | 1:5 = 83.7, 1:0 = 92.2 | Excessive noise hurts but does not immediately invalidate the method |

### Key Findings
- The performance gap between full context and proxy context is the true bottleneck: the model does not lack reasoning ability but struggles with grounding reasoning steps in long inputs.
- ProxyCoT-RL generally outperforms ProxyCoT-ZS, indicating that task-specific trajectories obtained via RLVR on proxy contexts are more suitable for distillation than zero-shot trajectories from a large teacher.
- SFT or RLVR only on full context is unstable; obtaining trajectories on short proxies followed by full-context grounding represents a superior trade-off between computation and effect.
- The degree of structure in the proxy is crucial. Structured metadata in SciTrek vastly outperforms unstructured text.

## Highlights & Insights
- The paper identifies a critical fact in long-context reasoning: most tokens in a long input serve as a localization burden, whereas the evidence required for actual reasoning is short.
- ProxyCoT transforms the RAG "evidence retrieval" concept into training signals rather than just concatenating retrieval results at inference time. This allows the model itself to learn reasoning patterns on short evidence within long contexts.
- Achieving 46.5 EM with 617 CoT tokens surpasses 31.6 EM with 6,683 tokens from full-context teacher CoT SFT, proving that longer reasoning is not always more effective, especially when evidence grounding is incorrect.
- Practical for resource-constrained labs: avoids having large teachers repeatedly read 128K full contexts and avoids expensive RL sampling on long inputs.

## Limitations & Future Work
- The method assumes availability of proxy contexts sufficient to answer the question. Many real-world tasks lack manual supporting evidence or structured metadata, making automatic proxy construction difficult.
- If a system can be solved with a RAG workflow, the interaction between ProxyCoT and retrieval systems—and who oversees evidence selection—is not deeply discussed.
- Experiments were constrained by compute and data, covering only English tasks; generalization across languages, domains, and even longer contexts remains to be verified.
- If the proxy context is of poor quality or lacks structure, performance drops significantly; this shifts the bottleneck from "long-context training" to "high-quality proxy construction."
- Future research could explore automatic proxy discovery, training with noisy retrieved evidence, long-context grounding combined with tool use, and multilingual long-text reasoning.

## Related Work & Insights
- **vs. CoT Distillation**: Conventional CoT distillation has the teacher write trajectories on full inputs; ProxyCoT has the teacher or RL model write them on short proxies before migrating to the full context.
- **vs. RLVR on full context**: Direct RLVR sampling on long contexts is computationally expensive; ProxyCoT-RL places RL on short proxies, significantly reducing training difficulty.
- **vs. RAG**: RAG retrieves evidence during inference; ProxyCoT uses "perfectly retrieved evidence" as intermediate supervision during training to strengthen the model's inherent long-context capability.
- **vs. Long-Context Architecture Improvements**: Sparse attention or RoPE extension addresses readable length; ProxyCoT addresses the problem of maintaining correct reasoning trajectories within long inputs.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The CoT transfer from proxy context to full context is natural but well-targeted; the design is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers two models, two primary datasets, Loong transfer, and multiple ablations, though the scope of languages and tasks is limited.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation, sufficient tabular data, and easy explanation of why two-stage training is necessary.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for training long-context reasoning, especially in scenarios with limited compute budgets but available short-evidence supervision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning](ppa-plan_proactive_pitfall_avoidance_for_reliable_planning_in_long-context_llm_r.md)
- [\[ACL 2026\] DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning](delta_dynamic_layer-aware_token_attention_for_efficient_long-context_reasoning.md)
- [\[ACL 2026\] Distilling Long-CoT Reasoning through Collaborative Step-wise Multi-Teacher Decoding (CoRD)](distilling_long-cot_reasoning_through_collaborative_step-wise_multi-teacher_deco.md)
- [\[ACL 2026\] CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards](csrp_chain-of-thought_reasoning_for_chinese_text_correction_via_reinforcement_le.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](../../NeurIPS2025/llm_reasoning/mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)

</div>

<!-- RELATED:END -->
