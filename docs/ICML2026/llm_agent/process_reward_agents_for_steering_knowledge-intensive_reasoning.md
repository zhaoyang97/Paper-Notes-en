---
title: >-
  [Paper Note] Process Reward Agents for Steering Knowledge-Intensive Reasoning
description: >-
  [ICML 2026][LLM Agent][Process Reward Model] The Process Reward Model is reconstructed from "post-hoc scoring" into an **online agent** that decides in real-time whether to retrieve evidence and provides rewards at each…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Process Reward Model"
  - "Beam Search"
  - "Retrieval Augmentation"
  - "Medical Reasoning"
  - "Frozen Policy"
date: 2026-05-08
content_hash: bb873cafa5de8ec4
---

# Process Reward Agents for Steering Knowledge-Intensive Reasoning

**Conference**: ICML 2026  
**arXiv**: [2604.09482](https://arxiv.org/abs/2604.09482)  
**Code**: https://process-reward-agents.github.io/ (Available)  
**Area**: LLM Agent / Process Reward / Medical Reasoning  
**Keywords**: Process Reward Model, Beam Search, Retrieval Augmentation, Medical Reasoning, Frozen Policy

## TL;DR
The Process Reward Model is reconstructed from "post-hoc scoring" into an **online agent** that decides in real-time whether to retrieve evidence and provides rewards at each reasoning step. By using beam search to prune candidate trajectories of a frozen policy, Qwen3-4B achieves 81.9% on MedQA, setting a 4B-scale SOTA, and directly generalizes to various unseen backbones from 0.5B to 8B (yielding up to 25.7% gains).

## Background & Motivation

**Background**: In tasks such as mathematics and coding, each reasoning step can be mechanically verified using formal rules or compilers. However, in knowledge-intensive domains like medicine, determining the correctness of a step often requires synthesizing evidence across multiple guidelines, literature, and clinical standards, lacking locally verifiable "axioms." Current approaches follow two main lines: (1) injecting retrieved documents into the policy context (RAG); (2) training Process Reward Models (PRMs) to score complete reasoning trajectories post-hoc (e.g., Med-PRM, Med-S3).

**Limitations of Prior Work**: Post-hoc scoring implies that errors have already propagated to the end, making correction too late. It also fails to support branching, pruning, or reranking during generation, limiting the scope for inference-time scaling. Stuffing all documents into the policy context leads to context bloat and doesn't guarantee the model will consult the "correct evidence" at the "correct time." Furthermore, PRMs exhibit poor off-policy generalization—switching the backbone model causes reward signal distortion due to distribution shift.

**Key Challenge**: Reward signals need to be "online, step-level, and supported by external evidence" to truly intervene in generation; however, existing PRMs are either offline or only usable post-hoc, and are tightly coupled with the policy.

**Goal**: To decouple the judgment of "when to retrieve + whether this step is correct" from the policy, creating an independent reward module that can intervene online in beam search while keeping the policy frozen and hot-swappable.

**Key Insight**: A reward model does not have to be a passive scorer; it can be an **agent**—actively choosing between "retrieve" or "score directly" at each step, then providing a 0-1 reward for the current step. This allows for dynamic integration of external knowledge into the reasoning process and decouples the policy from the reward, allowing them to evolve independently.

**Core Idea**: Use a lightweight agent with shared parameters (two token-level readouts from the same model) to simultaneously output **action (whether to search)** and **reward (whether this step is correct)**. Its cumulative reward serves as a pruning signal for beam search, transforming the process reward from post-hoc scoring into online control.

## Method

### Overall Architecture
PRA consists of three synergistic components: a frozen reasoning policy $\pi$, a process reward agent $\mu_\phi$, and a dense retriever $\rho$ (MedCPT). Given a question $q$ and knowledge base $\mathcal{D}$, beam search maintains a set of partial trajectories $\{\tau_t^{(j)}\}_{j=1}^B$ with width $B$. At each step:

1. $\pi$ samples $b$ candidate next steps for each trajectory, resulting in $B\times b$ new trajectories;
2. For each new trajectory, the action readout of $\mu_\phi$ first determines whether to search—if yes, it calls $\rho$ to retrieve $D_t$; if no, $D_t=\varnothing$;
3. The reward readout of $\mu_\phi$ scores the step $\hat r_t \in [0,1]$ conditioned on $(q,\tau_t,D_t)$, with the cumulative reward for each trajectory being $R(\tau_t^{(j)})=\sum_{i=1}^t \hat r_i^{(j)}$;
4. Retain the top-$B$ trajectories based on $R$ and prune the rest. The trajectory with the highest cumulative reward after generation is selected as the answer.

On an engineering level, PRA maintains a **global trajectory queue** across the entire benchmark, executing tasks in batches categorized by stage (policy generation / retrieval / readout). This maintains high GPU utilization even when different questions or beams are desynchronized due to variable-length reasoning and conditional retrieval.

### Key Designs

1. **PRA Dual Readouts (Action + Reward with Shared Parameters)**:
    - **Function**: Enables a single Qwen3-4B to simultaneously output two binary signals: "whether to search this step" and "whether this step is correct."
    - **Mechanism**: Two slots $\ell^{(1)}, \ell^{(2)}$ are fixed in the PRA output sequence, performing a two-way softmax over tokens "0" and "1". The reward $\hat r_t=\text{softmax}(\ell^{(1)})_{[1]}$ is treated directly as the probability that the "current step is correct," while the action $\hat a_t\sim\text{softmax}(\ell^{(2)})$ decides whether to trigger retrieval. Both readouts share the backbone, utilizing only two tokens for agent control and step-level scoring, which adds negligible inference overhead.
    - **Design Motivation**: Conventional PRMs only output rewards and passively accept external retrieval. By internalizing "whether to retrieve" as an agent action, the reward module can adaptively decide if external evidence is needed based on the reasoning state. This introduces a new inference-time scaling axis—optionally spending retrieval budget at each step for stronger reward signals.

2. **Teacher Model Generation for Reasoning + Search Dual Labels (Based on Margin Shift)**:
    - **Function**: Automatically generates two sets of supervision—"is this step correct (reasoning label)" and "is retrieval needed for this step (search label)"—to train the PRA.
    - **Mechanism**: Utilizing Qwen3-235B-Instruct as a teacher, each partial trajectory is evaluated twice—once with retrieved documents and once without—extracting log-probs for tokens 0/1 to calculate the margin $m=\log p(1)-\log p(0)$ and $m_d$. The margin shift $\Delta m=m-m_d$ serves as a proxy for "retrieval influence": if $|\Delta m|>\epsilon_{\text{global}}$ (median of the training set for a 50/50 split), it is labeled for search; otherwise, it is labeled for reward. The reasoning label is the teacher's binary judgment under the retrieval condition.
    - **Design Motivation**: Manual step-level supervision is expensive; MC rollout labels are noisy (incorrect intermediate steps may still lead to correct answers); and LLM-as-judge can easily misjudge medical scenarios without evidence. Using margin shift for retrieval necessity labels represents the "posterior update magnitude" from a Bayesian perspective—labeling "search required" only when new evidence truly shifts the teacher's belief, thereby learning selective retrieval rather than blind searching.

3. **PRA-guided beam search (Online Process Reward + Stage-level Batching)**:
    - **Function**: Embeds PRA step-level rewards into beam search as online pruning signals while efficiently running inference for the entire benchmark on GPUs.
    - **Mechanism**: Employs beam search with width $B=4$ and branching factor $b=16$, ensuring the sampling budget $B\times b=64$ equals the 64-way sampling of self-consistency for compute fairness. Each step uses PRA to score $B\times b$ candidates, retaining the top-$B$ based on cumulative reward $R$. To avoid idle time due to variable depths across questions, all active traces are placed in a global queue and bucketed by three pending stages: "policy generation / retrieval / readout." Each stage is executed in batches before returning to the queue—maintaining high utilization even when conditional retrieval causes some steps to skip $\rho$.
    - **Design Motivation**: Post-hoc scoring (outcome-level or post-hoc process-level) can only aggregate over complete trajectories and cannot correct early errors. Only online step-level rewards can prune before errors propagate. A configuration with small beam width and large branching factor provides enough candidates for the PRA to "select" without causing the global queue to explode.

### Loss & Training
The PRA is fine-tuned from Qwen3-4B-Instruct: each step predicts reasoning and search binary tokens simultaneously, using cross-entropy loss at these positions. In main experiments, the search label is fixed to 1 (always-search setting to ensure evidence is always present for reward evaluation); margin-shift labels are only used when analyzing search–accuracy trade-offs to let the PRA learn selective retrieval based on a threshold $\theta_{\text{dep}}$. Training data includes 10,178 questions from the MedQA train split, with 8 trajectories sampled per question by a frozen Qwen3-4B and retrieval performed for each partial trajectory to generate step-level samples.

## Key Experimental Results

### Main Results
Comparison with Direct/CoT/RAG (including 64-way self-consistency) across seven medical reasoning benchmarks, using Qwen3-4B-Instruct as the universal policy.

| Dataset | Metric | PRA | RAG+SC | Gain |
|--------|------|------|----------|------|
| MedQA (ID) | Acc | 81.9 | 76.7 | +5.2 |
| Medbullets | Acc | 65.9 | 58.4 | +7.5 |
| MedMCQA | Acc | 66.2 | 64.8 | +1.4 |
| MMLU-Med | Acc | 86.6 | 86.2 | +0.4 |
| GPQA | Acc | 65.1 | 54.4 | +10.7 |
| Lancet | Acc | 70.9 | 61.0 | +9.9 |
| NEJM | Acc | 68.0 | 66.9 | +1.1 |
| **Average** | Acc | **72.1** | 66.9 | **+5.2** |

Cross-backbone generalization (PRA trained only on Qwen3-4B trajectories; all policies without † are completely unseen):

| Policy | CoT | +SC | +PRA | $\Delta$ vs CoT |
|--------|-----|-----|------|----|
| Llama-3.1-8B | 67.0 | 75.1 | 82.3 | +15.3 |
| Llama-3.2-3B | 56.0 | 66.2 | 79.1 | +23.1 |
| Qwen2.5-3B | 49.5 | 54.0 | 74.9 | +25.4 |
| Llama-3.2-1B | 36.2 | 44.0 | 61.2 | +25.0 |
| Qwen2.5-0.5B | 28.4 | 31.9 | 54.1 | +25.7 |

### Ablation Study
Table 3 decomposes reward agent, training, and retrieval factors (Policy: Qwen3-4B, budget 64):

| Config | Acc. | Description |
|------|------|------|
| CoT | 72.7 | Single-sample baseline |
| CoT + SC | 74.8 | 64-way Self-Consistency |
| RAG + SC | 76.7 | Retrieval + Self-Consistency |
| PRA w/o Training w/o search | 74.4 | Untrained Qwen3-4B as reward agent, beam search structure only |
| PRA w/o Training w/ search | 76.7 | With retrieval, matching RAG+SC |
| **PRA (Full)** | **81.9** | Trained reward agent + Online retrieval |

Table 4 decomposes reward level and timing (using the same trained PRA parameters):

| Usage | Reward Level | Timing | Acc. |
|------|--------------|--------|------|
| PRA (Last) | Outcome | Post hoc | 75.7 |
| PRA (Min) | Process | Post hoc | 74.3 |
| PRA (Max) | Process | Post hoc | 77.5 |
| PRA (Average) | Process | Post hoc | 77.6 |
| **PRA (Ours)** | Process | **Online** | **81.9** |

### Key Findings
- **Training the reward agent is the largest single contribution**: The untrained version with search only matches RAG+SC (76.7), whereas training pushes it to 81.9.
- **Online > Post-hoc**: Using the same PRA parameters, post-hoc scoring (Average) only reaches 77.6, while online beam search intervention adds +4.3 to reach 81.9—demonstrating that performance stems not just from rewards but from the ability to prune before error propagation.
- **Smaller models benefit more**: PRA brings a 90.5% relative improvement (28.4→54.1) to Qwen2.5-0.5B, revealing the underestimated reasoning potential of small models; this suggests that domain adaptation can be achieved via reward agents without retraining policies.
- **Self-consistency can hurt on hard problems**: On benchmarks like GPQA and Lancet where the policy fails frequently, more sampling amplifies errors via majority vote, whereas PRA provides stable improvements through external evidence.
- **Margin shift correlates with correctness**: Correct trajectories show larger margin shifts toward the end (dependence on evidence for final judgment), while incorrect trajectories show smaller shifts (teacher detects internal inconsistency without evidence), providing an interpretable signal for "when to retrieve."

## Highlights & Insights
- **Recasting PRM as an Agent**: While traditional PRMs are passive scorers, PRA internalizes "whether to retrieve" as an action, making the reward module a mini-agent. This makes reward signals online, controllable, and branchable, representing a new dimension of inference-time scaling—beyond scaling sample counts to scaling retrieval budget × beam width.
- **Policy and Reward Decoupling**: The policy never sees retrieval documents nor updates its parameters; new backbones are plug-and-play. This is highly significant for industry deployment—as medical knowledge updates monthly, one only needs to retrain a 4B reward agent rather than the entire LLM stack.
- **Stage-level Global Batching**: The engineering implementation of PRA-guided beam search replaces "question-independent" scheduling with "stage-independent" global queues, hiding desynchronization from variable reasoning lengths and conditional retrieval within the batch dimension—a trick applicable to any PRM-based benchmarking.
- **Margin Shift as a Proxy for Retrieval Necessity**: Using the log-prob difference between "with evidence" vs "without evidence" from a teacher model avoids expensive manual annotation of "when to retrieve," offering a label generation paradigm reusable in other knowledge-intensive fields like law or finance.

## Limitations & Future Work
- **Validated Only in Medicine**: All experiments use MedQA and medical OOD cases; the authors acknowledge this as a methodological contribution rather than a fully deployable system. Effectiveness in other domains remains unverified.
- **Always-on Retrieval as an Upper Bound**: Main results use the always-search config; retrieving at every step is significantly more costly than self-consistency. While a selective search Pareto front exists, its MedQA performance ceiling is lower than always-search.
- **Dependency on a Strong Teacher**: Labels generated by Qwen3-235B limit the PRA's ceiling to the teacher's medical judgment quality; if the teacher is wrong, the PRA follows.
- **Beam Search as Single-chain Reasoning**: PRA selects top-$B$, but each trajectory remains a linear sequence without step-level backtracking or rewriting support; an early "mis-kill" by PRA could derail the entire beam.
- **Future Directions**: (i) Expanding action space beyond binary {search, reward} to include "change retriever" or "backtrack to step k"; (ii) using RL to optimize final answer accuracy directly rather than binary labels; (iii) adding calibration to the reward agent so $\hat r_t$ represents true posterior probability.

## Related Work & Insights
- **vs Med-PRM (Yun et al., 2025)**: Both are retrieval-augmented PRMs, but Med-PRM scores only after complete trajectory generation, whereas PRA scores every step and integrates into beam search to prune early errors.
- **vs Med-S3 (Jiang et al., 2025)**: Med-S3 jointly trains policy + reward for self-evolution without search; PRA keeps the policy frozen to allow independent reward evolution, suited for frequent backbone swaps.
- **vs RAG / RAG+SC**: Traditional RAG injects documents into policy context, relying on the policy to "pick the relevant parts." PRA externalizes retrieval to the reward agent, keeping the policy's input identical to CoT, thus preserving its output distribution and preventing context bloat.
- **vs Mathematical PRMs (Lightman 2023, Wang 2023)**: Math step labels can be approximated by MC rollout, but medical "correctness" requires external evidence. PRA's use of margin shift to learn when evidence is needed addresses a key challenge in migrating PRMs to knowledge-intensive domains.

## Rating
- Novelty: ⭐⭐⭐⭐ Reconstructing PRM as an agent with actions and using margin shift for automatic search labels is a meaningful paradigm shift, though beam search + PRM itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 benchmarks + 6 cross-backbone models (0.5B-8B) + multi-dimensional ablations (training/retrieval/timing) + Pareto analysis; a rare comprehensive study in the medical domain.
- Writing Quality: ⭐⭐⭐⭐ The logical chain between motivation and method is clear; the Bayesian explanation for margin shift is slightly abrupt.
- Value: ⭐⭐⭐⭐⭐ The "frozen policy + hot-swappable reward" paradigm is industry-friendly for medical LLM deployment, and 4B models breaking 80% on MedQA is a persuasive milestone.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](../../ICLR2026/llm_agent/webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)
- [\[ACL 2026\] Exploring Reasoning Reward Model for Agents](../../ACL2026/llm_agent/exploring_reasoning_reward_model_for_agents.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ACL 2026\] Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems](../../ACL2026/llm_agent/rethinking_reasoning-intensive_retrieval_evaluating_and_advancing_retrievers_in_.md)

</div>

<!-- RELATED:END -->
