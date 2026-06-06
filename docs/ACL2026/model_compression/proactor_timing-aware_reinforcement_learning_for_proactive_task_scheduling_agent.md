---
title: >-
  [Paper Note] ProActor: Timing-Aware Reinforcement Learning for Proactive Task Scheduling Agents
description: >-
  [ACL2026][Model Compression][Proactive agents] ProActor advances conversational task scheduling from "reacting to explicit user instructions" to "proactively triggering actions at the appropriate timing." By utilizing au…
tags:
  - "ACL2026"
  - "Model Compression"
  - "Proactive agents"
  - "task scheduling"
  - "timing-aware reinforcement learning"
  - "GRPO"
  - "RULER reward"
date: 2026-05-08
content_hash: fd4a0e445923db1b
---

# ProActor: Timing-Aware Reinforcement Learning for Proactive Task Scheduling Agents

**Conference**: ACL2026  
**arXiv**: [2605.24900](https://arxiv.org/abs/2605.24900)  
**Code**: Planned open-source, cache URL not provided  
**Area**: LLM Agent / Reinforcement Learning  
**Keywords**: Proactive agents, task scheduling, timing-aware reinforcement learning, GRPO, RULER reward  

## TL;DR
ProActor advances conversational task scheduling from "reacting to explicit user instructions" to "proactively triggering actions at the appropriate timing." By utilizing automated reference action labeling, proactiveness metrics, turn-level GRPO, and the ART-F efficient training framework, the 4-bit Qwen2.5-14B-ProActor-Q4 achieves a peak PRI of 0.7293 on ABCD+, significantly enhancing proactive timing while maintaining action consistency.

## Background & Motivation
**Background**: As LLM agents are embedded into customer service, enterprise automation, financial consulting, and real-time assistants, many scenarios no longer suffice with passive waiting for user instructions. An ideal proactive agent should continuously understand dialogue states, anticipate user needs, and prepare or trigger appropriate software actions without disrupting the conversation flow.

**Limitations of Prior Work**: The difficulty of proactive behavior lies not just in "what action to take" but also "when to take it." An action often has multiple reasonable trigger points; appearing slightly earlier or later can both be valid. Traditional SFT treats timing as a single-point label, which penalizes other valid timings. Prompting or static reasoning also struggles to stably balance action consistency and proactive timing.

**Key Challenge**: The more proactive an agent is, the easier it is to mis-trigger; the more conservative it is, the more opportunities it misses. Existing methods typically either maintain consistent action parameters but remain conservative in timing, or achieve high readiness rates at the cost of high false trigger rates. Achieving a controllable trade-off between reference action alignment and timing quality is the core problem of this paper.

**Goal**: To build an end-to-end framework that automatically generates scalable reference action annotations, defines metrics and rewards for evaluating proactiveness, and trains a timing-aware policy using RL while resolving the high rollout costs and low GPU utilization associated with LLM RL training.

**Key Insight**: Instead of treating a reference action as the sole ground truth, the authors view it as a set of valid timing anchors. They then use turn-level GRPO to allow the model to explore more reasonable trigger points. For trainability, the paper proposes ART-F, which combines a request-adaptive inference cluster with DDP training on single-node multi-GPU setups.

**Core Idea**: Proactive scheduling is not about mimicking a specific point in time, but about learning the dynamic trade-off between "action consistency, earliness, and mis-trigger risk" at each dialogue turn through rewards.

## Method
ProActor consists of three layers. The first layer covers data and annotation: standardizing software actions from different domains into an action catalog and automatically labeling reference actions using an oracle LLM. The second layer covers evaluation and rewards: characterizing consistency and timing using metrics such as AC/Max AC/PT/FTR/RAR. The third layer is the training system: training a 4-bit LoRA agent using turn-level GRPO with RULER/metric/composite rewards, optimized by ART-F for rollout and training efficiency.

### Overall Architecture
First, ProActor standardizes heterogeneous tools into a JSON action catalog using a unified metadata schema, recording action ontology, type signatures, and parameter attributes. The Catalog Generator uses Jinja2 to render action descriptions for use by the annotator and agent.

Second, the oracle LLM annotator gains a complete view of the dialogue, including future turns. Rather than merely interpreting actions that have already occurred, it determines at each turn whether an actionable opportunity has emerged. Since a reference action may cover multiple valid timings, the authors refer to these annotations as guidance over ground truth.

Then, the model generates a candidate action set at each dialogue turn, including action names, parameters, and readiness/status. During evaluation, predicted actions are compared with reference actions across dimensions like parameter matching, ready timing, and false triggers.

Finally, Qwen2.5-14B-Instruct is fine-tuned via 4-bit quantization + LoRA into Qwen2.5-14B-ProActor-Q4. Training utilizes turn-level GRPO: multiple action candidates are rolled out for each turn, advantages are calculated based on rewards, and the policy is updated.

### Key Designs
1.  **Domain-agnostic reference action auto-labeling**:
    - **Function**: Provides large-scale proactive timing supervision for RL while avoiding the high cost of manual turn-by-turn labeling.
    - **Mechanism**: The oracle annotator generates reference action candidates for each turn under complete dialogue context and a unified action catalog; for data with historical action observations, actual triggered actions can be used to filter annotation quality.
    - **Design Motivation**: Proactive behavior naturally has multiple reasonable timings, so annotations should not be interpreted as the unique answer. Treating annotations as a reference range supports RL in exploring more flexible timing policies than SFT.

2.  **Proactiveness metrics and PRI**:
    - **Function**: Decouples "whether the action is correct" from "whether it is proactive but not excessive" into quantifiable metrics.
    - **Mechanism**: AC measures average parameter/action alignment between predicted and reference actions; Max AC measures best alignment; Difference measures prediction stability; PT rewards ready actions that are not later than the reference-ready window; FTR penalizes false triggers outside reference coverage; RAR measures the ratio of ready actions. The comprehensive ranking uses PRI, the harmonic mean of the consistency index and timing index.
    - **Design Motivation**: Looking at AC alone favors conservative models, while looking at PT/RAR alone might encourage reckless triggering. PRI forces the model to balance consistency and proactiveness.

3.  **Turn-level GRPO and stage-aware rewards**:
    - **Function**: Provides dense rewards at the dialogue turn granularity to reduce credit assignment difficulty for long dialogue trajectories.
    - **Mechanism**: $K$ action candidates are sampled per turn, advantages are calculated using turn-level rewards, and updates are performed using GRPO/PPO-style clipping. Rewards include RAC/Max RAC, General/Custom RULER, Weighted Metric, Adaptive Metric, and Adaptive RULER. Adaptive RULER uses $R_{adR}(u)=(1-\lambda_u)R_{metric}(u)+\lambda_u R_{RULER}(u)$, where $\lambda_u$ gradually increases to $\lambda_{max}=0.3$.
    - **Design Motivation**: Early training requires timing exploration, while later stages require controlling false triggers and consistency. A fixed single-target reward struggles to cover the entire learning process; stage-aware composite rewards are better suited for proactive scheduling.

### Loss & Training
The training objective is to maximize the expected return under turn-level rewards. The base model is 4-bit Qwen2.5-14B-Instruct, with LoRA rank 8, $\alpha=16$, targeting attention q/k/v/o projections and MLP gate/up/down projections, with dropout at 0. ABCD+ is trained on 4×H200, while Home Loan is trained on 8×H100, with a maximum context length of 9,216 tokens. ART-F dynamically launches multiple vLLM inference instances coupled with master-worker asynchronous payload-distribution DDP training to alleviate the imbalance between rollout and training phases. The paper reports a 4-8× speedup from ART-F.

## Key Experimental Results

### Main Results
Two datasets cover different real-world scenarios: ABCD+ contains historical action observations for annotation validation; Home Loan contains only financial consulting transcripts without actual trigger logs, making it closer to privacy-restricted enterprise data.

| Dataset | Domain | train/dev/test | Labeled Actions | Avg. Dialogue Length | Characteristics |
|---------|--------|----------------|-----------------|----------------------|-----------------|
| ABCD+ | Customer Service | 5,647 / 703 / 692 | 114,978 | 21.2 ± 7.2 turns | Observed triggers available for quality filtering |
| Home Loan | Mortgage Consulting | 774 / 97 / 97 | 30,610 | 47.4 ± 1.1 turns | No action observations; inferred solely from dialogue |

Main results show that ProActor-Q4 significantly outperforms GPT/Gemini/Claude/Qwen baselines on ABCD+; on Home Loan, Adaptive RULER maintains strong consistency, though the highest PRI is obtained by the Gemini reasoning baseline.

| Dataset | Method | PRI | AC | Max AC | Difference | PT | FTR | RAR |
|---------|--------|-----|----|--------|------------|----|-----|-----|
| ABCD+ | Gemini-2.5-flash Non-Reasoning | 0.6251 | 0.417 | 0.834 | 1.000 | 0.2133 | 0.0399 | 0.288 |
| ABCD+ | Claude-4 Reasoning | 0.6318 | 0.421 | 0.749 | 0.779 | 0.2136 | 0.0482 | 0.314 |
| ABCD+ | Qwen2.5-14B + SFT | 0.1700 | 0.272 | 0.533 | 0.960 | 0.2097 | 0.0912 | 0.531 |
| ABCD+ | ProActor-Q4 + Custom RULER | 0.7293 | 0.426 | 0.484 | 0.136 | 0.2347 | 0.0708 | 0.546 |
| ABCD+ | ProActor-Q4 + Adaptive RULER | 0.6842 | 0.431 | 0.586 | 0.320 | 0.2515 | 0.1089 | 0.521 |
| Home Loan | Gemini-2.5-flash Reasoning | 0.7303 | 0.345 | 0.527 | 0.528 | 0.0757 | 0.0001 | 0.241 |
| Home Loan | Claude-4 Reasoning + ASG | 0.7262 | 0.375 | 0.607 | 0.619 | 0.0760 | 0.0127 | 0.307 |
| Home Loan | ProActor-Q4 + Custom RULER | 0.5603 | 0.206 | 0.234 | 0.137 | 0.0846 | 0.0355 | 0.465 |
| Home Loan | ProActor-Q4 + Adaptive RULER | 0.6232 | 0.395 | 0.466 | 0.180 | 0.0501 | 0.0131 | 0.156 |

### Ablation Study
Reward ablation indicates that single-target action consistency rewards tend to be conservative, RULER rewards are better at optimizing timing, and Adaptive RULER is more balanced in large-scale training.

| Dataset / Scale | Reward | PRI | AC | Max AC | PT | FTR | RAR | Interpretation |
|-----------------|--------|-----|----|--------|----|-----|-----|----------------|
| ABCD+ 100/50 | RAC | 0.2596 | 0.3239 | 0.3881 | 0.1223 | 0.0640 | 0.3002 | Consistency reward is too conservative |
| ABCD+ 100/50 | Custom RULER | 0.6140 | 0.3850 | 0.4203 | 0.2315 | 0.1068 | 0.5707 | Significantly stronger timing |
| ABCD+ 5647/692 | Custom RULER | 0.7217 | 0.4257 | 0.4837 | 0.2347 | 0.0708 | 0.5456 | Highest PRI at full scale |
| ABCD+ 5647/692 | Adaptive RULER Max RAC | 0.6026 | 0.4314 | 0.5861 | 0.2515 | 0.1089 | 0.5212 | Strong AC/PT, but higher FTR |
| Home Loan 774/97 | RAC | 0.5668 | 0.4701 | 0.5195 | 0.0264 | 0.0041 | 0.0612 | Consistent actions but not proactive |
| Home Loan 774/97 | Custom RULER | 0.4220 | 0.2057 | 0.2338 | 0.0846 | 0.0355 | 0.4653 | Very proactive but weak consistency |
| Home Loan 774/97 | Adaptive RULER RAC | 0.6154 | 0.4173 | 0.4403 | 0.0397 | 0.0071 | 0.1231 | More balanced |

ART-F and training settings emphasize engineering feasibility: ABCD+ training is completed on 4×H200, and Home Loan on 8×H100. End-to-end ABCD+ training takes 3.5-5.7 days, while Home Loan takes 1.5-2.15 days. The 4-8× speedup reported for ART-F makes timing-aware RL actionable in single-node multi-GPU environments.

| Training Component | Key Setting | Role |
|--------------------|-------------|------|
| Qwen2.5-14B-ProActor-Q4 | 4-bit quantization + LoRA rank 8, $\alpha=16$ | Reduces memory and training costs |
| ART-F inference cluster | Multiple vLLM servers/GPU, dynamic routing | Increases rollout throughput |
| DDP training | Symmetric replicated data mode | Stabilizes multi-GPU gradient updates |
| Max context | 9,216 tokens | Supports long-dialogue task scheduling |
| Speedup | 4-8× | Mitigates RL rollout training bottlenecks |

### Key Findings
- SFT achieves a PRI of only 0.1700 on ABCD+, indicating that mimicking reference actions as hard labels is unsuitable for proactive scheduling with multiple valid timings.
- Custom RULER provides the strongest proactive behavior on ABCD+: PT 0.2347, RAR 0.546, Difference 0.136, outperforming strong baselines in timing while maintaining a low consistency gap.
- Adaptive RULER reaches the highest AC (0.431) and PT (0.2515) on ABCD+, but also leads to higher FTR, suggesting that timing intensity requires careful tuning.
- Home Loan is more challenging than ABCD+: while Custom RULER achieves the highest PT (0.0846) and RAR (0.465), its AC is only 0.206; Adaptive RULER achieves a higher AC (0.395) in exchange for lower RAR and FTR.
- Reasoning baselines often improve consistency but make models more hesitant; ASG sometimes improves timing at an additional consistency cost. The value of RL lies in internalizing timing intuition into the policy rather than stacking structures during inference.

## Highlights & Insights
- "Reference action is not ground truth" is the most critical concept in this paper. Proactive behavior naturally has a timing window; treating it as a single-point label misleads both training and evaluation.
- The metric design is comprehensive: AC/Max AC for action consistency, PT/FTR/RAR for proactiveness, and PRI (harmonic mean) to prevent metric gaming. This is more suitable for real-world agents than simple tool-calling accuracy.
- The significance of RULER reward is in converting the vague concept of "appropriate proactiveness" into learnable preferences rather than relying solely on hard rules. Custom RULER significantly improves timing, showing that rubrics are vital for proactive agents.
- ART-F is a practical engineering contribution. Many agent RL papers are bottlenecked by rollout inefficiency and memory pressure; here, quantized LoRA, vLLM clusters, and DDP payload distribution enable training on enterprise-scale data.

## Limitations & Future Work
- Observed triggers only cover actions that actually occurred in the real system, whereas the range of acceptable proactive actions is wider. Even if ABCD+ has action logs, they are only a subset of valid timings, not the full ground truth.
- Evaluation only covers two English datasets. Turn-taking, directness, politeness, and formality in different languages affect the appropriateness of "when to be proactive," and low-resource languages may introduce issues with annotation and RL stability.
- RL experiments only validated 4-bit Qwen2.5-14B-Instruct + LoRA. Although the framework claims to be model-agnostic, Llama, Mistral, multilingual models, different quantization schemes, and parameter scales have not yet been verified.
- Training is limited to dialogues not exceeding 50 turns. Longer enterprise workflows may require stronger memory, state abstraction, and credit assignment.
- Home Loan is proprietary data, limiting external replication. The paper plans to release annotation tools and processed ABCD+ annotations, but the extent of openness will affect verifiability.

## Related Work & Insights
- **vs proactive prompting / context engineering**: Prompting can make models more proactive but struggles to stably optimize the timing-consistency trade-off; ProActor explicitly optimizes this target via RL rewards.
- **vs SFT task scheduling**: SFT is suitable for tasks with a single correct answer, but proactive timing has multiple reasonable points. The low SFT PRI in ProActor's results supports RL over hard imitation.
- **vs tool-calling benchmarks**: Standard tool calling usually assumes complete parameters and explicit triggers. This work allows for partial parameter specification and readiness tracking, aligning more with real-world conversational scheduling.
- **vs general RLHF/GRPO frameworks**: ProActor's RL is not about general chat preferences but turn-level action scheduling rewards; ART-F specifically optimizes the rollout-heavy nature of agent RL.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explicitly modeling proactive timing as a turn-level RL problem and introducing the reference action window perspective is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes two datasets, strong baselines, reward ablation, and system efficiency, though model family and multilingual validation are lacking.
- Writing Quality: ⭐⭐⭐⭐☆ The methods and metrics are complex, but the overall structure is clear, and main conclusions are well-supported by tables.
- Value: ⭐⭐⭐⭐⭐ Directly instructive for enterprise agents, customer service automation, and real-time task scheduling, particularly in its reward/metric design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ACL 2026\] TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination](tell-tale_task_efficient_llms_with_task_aware_layer_elimination.md)
- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](../../ICCV2025/model_compression/scheduling_weight_transitions_for_quantization-aware_training.md)

</div>

<!-- RELATED:END -->
