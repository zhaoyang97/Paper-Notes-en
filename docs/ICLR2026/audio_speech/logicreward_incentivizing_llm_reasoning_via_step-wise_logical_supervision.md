---
title: >-
  [Paper Note] LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision
description: >-
  [ICLR2026][Audio & Speech][logical reasoning] This paper proposes LogicReward, a reward function that employs the Isabelle theorem prover for step-wise logical correctness verification. Combined with Autoformalization wi…
tags:
  - "ICLR2026"
  - "Audio & Speech"
  - "logical reasoning"
  - "theorem prover"
  - "step-wise reward"
  - "autoformalization"
  - "soft unification"
date: 2026-05-08
content_hash: ae307441f8ae9695
---

# LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision

**Conference**: ICLR2026  
**arXiv**: [2512.18196](https://arxiv.org/abs/2512.18196)  
**Code**: [Project Page](https://llm-symbol.github.io/LogicReward)  
**Area**: Audio & Speech  
**Keywords**: logical reasoning, theorem prover, step-wise reward, autoformalization, soft unification  

## TL;DR
This paper proposes LogicReward, a reward function that employs the Isabelle theorem prover for step-wise logical correctness verification. Combined with Autoformalization with Soft Unification to reduce natural language ambiguity, the trained 8B model surpasses GPT-4o by 11.6% and o4-mini by 2% on NLI and logical reasoning tasks.

## Background & Motivation
1. Existing training methods primarily rely on outcome-based feedback, which may reward models that arrive at correct answers through flawed reasoning.
2. Process-level supervision (e.g., PRMs) still lacks formal guarantees of logical correctness.
3. Probabilistic feedback (token probabilities / learned reward models) is inherently non-deterministic and cannot reliably detect logical errors.
4. Symbolic verification methods are currently limited mainly to structured domains (mathematics/programming), with NLI largely unaddressed.
5. Natural language ambiguity and implicit assumptions make formal verification difficult (e.g., Dad ≠ Father yet semantically equivalent).
6. High-stakes scenarios (medical/legal) demand strict guarantees of logical consistency.

## Method

### Overall Architecture
LogicReward = Rollout Generation → Autoformalization with Soft Unification → Step-Wise Reward Computation → Refinement Iteration → Training Data Construction

### Data Collection & Rollout
Approximately 6,000 instances are sampled from 8 NLI/logical reasoning datasets. Qwen3-8B and GPT-4o each generate 4 responses per instance (8 total), formatted as "Step 1: s1; Step 2: s2; ...; Answer: A".

### LogicReward Reward Function (Three-Dimensional Verification)
Each reasoning step $s_i$ is decomposed into $(P_r, I)$, where $P_r$ denotes the premises invoked and $I$ denotes the inference drawn.

**Premise Validity**: Verifies whether the referenced premises $P_r$ are grounded in the given premise set $P$. $P_r$ is split into sentences $\{q_1,...,q_m\}$; for each $q_j$, the maximum cosine similarity with the given premises is computed, and the average serves as the step-level score.

**Logic Validity**: The Isabelle theorem prover is used to verify the logical correctness of inference $I$. Three cases are distinguished: syntactically valid + logically correct → 1; syntactically valid + logically incorrect → 0; syntactically invalid → falls back to token-probability confidence $\text{Conf}(I) = \frac{1}{|I|}\sum_{t \in I}\text{token\_prob}(t)$.

**Outcome Validity**: Whether the final answer matches the ground truth, as a binary score (0/1).

### Autoformalization with Soft Unification
Natural language contains abundant ambiguity and implicit assumptions (e.g., Dad ≠ Father yet semantically equivalent), which causes direct formalization to fail in Isabelle. Soft Unification prompts an LLM to supplement each reasoning step with valid but unstated assumptions (e.g., synonym mappings, commonsense additions), thereby improving formalization success rates. The enriched steps are then parsed using neo-Davidsonian event semantics and converted into Isabelle/HOL theories for verification.

### Refinement Iteration
For reasoning steps judged invalid by Isabelle, the error messages are used to prompt the LLM to iteratively refine the Soft Unification until logical correctness is achieved or a maximum iteration count is reached. Two responses per question are randomly selected for refinement, forming $D_{\text{refined}}$.

### Loss & Training
Final reward: $\text{LogicScore}(r,A) = \text{avg}(w_1 \cdot \text{ReasoningValidity}(r), w_2 \cdot \text{OutcomeValidity}(A))$, with $w_1=w_2=0.5$.  
Training data: $D_{\text{final}} = D_r \cup D_{\text{refined}}$. For SFT, the response with the highest LogicScore is selected as the target; for DPO, the highest- and lowest-scoring responses are paired. Base models are Llama3.1-8B and Qwen3-8B, trained in two stages (SFT → DPO) with LoRA fine-tuning.

## Key Experimental Results

| Model | M-LogiEval | FOLIO | ProverQA | LogiQA | 8-Task Avg. |
|------|-----------|-------|----------|--------|----------|
| GPT-4o | 68.0 | 63.5 | 78.4 | 69.3 | 73.9 |
| o4-mini | 82.0 | 80.8 | 78.8 | 65.7 | 83.5 |
| DeepSeek-R1-8B | 64.8 | 57.3 | 59.2 | 53.2 | 68.6 |
| **LogicReward-Qwen3-8B** | **82.0** | **79.5** | **81.2** | **72.3** | **85.5** |

### Ablation Study

| Reward Function | M-LogiEval | ProntoQA | ProverQA | QASC | 8-Task Avg. |
|---------|:----:|:----:|:----:|:----:|:----:|
| Confidence (avg. token prob.) | 76.9 | 81.0 | 52.3 | 89.8 | 64.3 |
| LLM-as-Judge (GPT-4o) | 65.8 | 84.6 | 47.5 | 95.3 | 60.2 |
| PRM (Nemotron-70B) | 66.7 | 90.6 | 62.1 | 97.0 | 66.0 |
| **LogicReward** | **79.0** | **90.3** | **60.1** | **97.8** | **71.4** |

**Generalization**: On unseen tasks, CommonsenseQA and GSM8K improve by 8.2% and 4.5% respectively, demonstrating that the rational capabilities cultivated by logical correctness rewards are transferable.

**Label-Free Setting**: Using only ReasoningValidity (without ground-truth labels) as the reward remains effective, with an average gain of +5.8%, indicating that the logical quality of the reasoning process itself constitutes a valuable supervision signal.

## Highlights & Insights
- First integration of a theorem prover into step-wise rewards for NLI — bridging the gap between symbolic verification in structured vs. unstructured domains.
- Soft Unification elegantly handles natural language ambiguity, serving as the key enabler for deploying theorem provers in natural language reasoning.
- An 8B model surpassing o4-mini — demonstrating that logically sound training data matters more than model scale.
- Effective in the label-free setting — ReasoningValidity alone constitutes a meaningful supervision signal.
- The refinement mechanism forms a closed loop by iteratively improving formalization using Isabelle error messages.

## Limitations & Future Work
- When Isabelle formalization fails, the method falls back to token probabilities, losing formal guarantees.
- Training and primary evaluation are conducted only on NLI/logical reasoning tasks; mathematics and commonsense serve merely as generalization probes.
- Soft Unification depends on LLM capabilities and may introduce new errors.
- The training set contains only ~6,000 instances; scalability remains to be verified.
- The formalization pipeline incurs high costs (requiring an Isabelle runtime and multiple LLM calls).
- The equal weighting $w_1=w_2=0.5$ is a simplistic choice; optimal weight allocation is unexplored.

## Related Work & Insights
- vs. PRM (Lightman et al.): LogicReward provides deterministic logical guarantees rather than probabilistic assessments.
- vs. LINC/Logic-LM: These methods apply provers at inference time; LogicReward uses provers at training time to construct rewards.
- vs. DeepSeek-R1: Uses only outcome rewards to incentivize long reasoning chains; LogicReward additionally supervises the logical validity of each reasoning step.
- vs. SymbCoT/Aristotle: These prompt LLMs to act as symbolic provers; LogicReward employs an actual theorem prover.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first combination of theorem provers with NLI training rewards)
- Experimental Thoroughness: ⭐⭐⭐⭐ (8 benchmarks + multiple baselines + generalization + label-free experiments)
- Writing Quality: ⭐⭐⭐⭐ (clear method description with complete formulations)
- Value: ⭐⭐⭐⭐⭐ (8B surpasses o4-mini; strong practical utility and theoretical significance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](../../CVPR2026/audio_speech/gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)
- [\[ICLR 2026\] Incentive-Aligned Multi-Source LLM Summaries](incentive-aligned_multi-source_llm_summaries.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](../../NeurIPS2025/audio_speech/sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[ICLR 2026\] Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)

</div>

<!-- RELATED:END -->
