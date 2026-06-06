---
title: >-
  [Paper Note] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess
description: >-
  [ICML 2026][Reinforcement Learning][SFT-to-RL] The authors utilize "training LLMs to play chess" as a clean testbed for verifiable RL. By systematically comparing the impact of 6 types of custom SFT datasets on RL…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "SFT-to-RL"
  - "Reasoning Faithfulness"
  - "Chess Reasoning"
  - "Information Density"
  - "GRPO"
date: 2026-05-08
content_hash: c58d5d17fc90b967
---

# How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess

**Conference**: ICML 2026  
**arXiv**: [2604.05134](https://arxiv.org/abs/2604.05134)  
**Code**: https://github.com/(lang-chess) (Yes)  
**Area**: Reinforcement Learning / LLM Reasoning / Verifiable RL  
**Keywords**: SFT-to-RL, Reasoning Faithfulness, Chess Reasoning, Information Density, GRPO

## TL;DR
The authors utilize "training LLMs to play chess" as a clean testbed for verifiable RL. By systematically comparing the impact of 6 types of custom SFT datasets on RL, they find that while "directly predicting the Best Move" yields the highest scores, it leads to unfaithful reasoning after RL. In contrast, "predicting the multi-step Best Line" achieves comparable performance with more stable RL and more faithful reasoning. They distill three metrics from SFT checkpoints to predict final RL performance and ultimately surpass gpt-oss-120b on multiple chess benchmarks using a 7B model.

## Background & Motivation

**Background**: Currently, training LLMs to "reason" is primarily conducted in math and code domains because these fields provide massive high-quality data and allow for automatic verification. Models such as R1, DeepSeek-R1, and Kimi k1.5 have achieved significant leaps through RL coupled with verifiable rewards.

**Limitations of Prior Work**: Controlled experiments are difficult to execute. Training data for math and code are already highly mature, and base models possess inherent capabilities, making it hard to isolate which specific type of training data drives RL improvements. Furthermore, the research community typically evaluates "reasoning quality" based on final accuracy, with few systematic investigations into whether RL is truly learning reasoning versus merely shifting answer distributions.

**Key Challenge**: To perform a precise causal analysis of "SFT data $\rightarrow$ RL behavior," it is necessary to identify a domain where LLMs are naturally weak (to avoid base model interference) but have ample verifiable rewards. Additionally, the domain must allow for a definition of whether "reasoning is faithful to the answer."

**Goal**: (a) How do different SFT data types (programmatic, rejection sampling, synthetic, Best Move, Best Line) influence SFT and final RL performance? (b) How does RL transform move quality distribution, hallucination rates, and reasoning strategies? (c) Which SFT-checkpoint metrics can predict post-RL performance?

**Key Insight**: The authors select chess as the "ideal testbed"—LLMs traditionally perform poorly (avoiding strong base bias), it possesses an episodic MDP structure, and Stockfish serves as a superhuman oracle for verifiable rewards and synthetic data. Chess also makes it easy to define whether reasoning is consistent with the final move (judged by gpt-oss-120b).

**Core Idea**: Conduct a systematic SFT $\rightarrow$ RL ablation using Qwen2.5-7B and introduce a "chess information density" framework (predictive complexity + accuracy + chess token density) to explain why certain datasets produce robust RL, distilling key designs for "faithful reasoning."

## Method

### Overall Architecture
Unified base model: Qwen2.5-7B-Instruct. Board states use visual ASCII format (preferred over FEN for uniform tokenization), and moves use UCI format. The task suite consists of four tasks: Predict Move (selecting the optimal move given a board), Best Move / Worst Move (1-out-of-5), and Legal Moves (IoU evaluation). The pipeline involves a fixed 15M token SFT followed by 8k samples of RL (Dr. GRPO + Clip-Higher + no KL) for ablation to select the strongest recipe, then scaling to 60M+60M token SFT with expanded RL.

### Key Designs

1.  **6 Types of Theory-Driven SFT Datasets**:
    - **Function**: Inject board understanding, decision-making, and reasoning formats to isolate experimental contributions.
    - **Mechanism**: ① General Instruction Following (Magpie Llama 3.3 70B) for regularization; ② Rejection Sampling using Llama 4 Maverick to generate correct samples and inject "teacher-style" reasoning; ③ Guided Synthetic providing the teacher with a 5-ply sequence and start/end boards to write "verbal $n$-step bootstrapping" (including transition function $\mathcal{T}(s_t,a_t)$ and value $V(s_t)$); ④ Verbalized Alpha-Beta Pruning which is fully programmatic—sampling moves via Stockfish softmax, recursively constructing minimax trees, and verbalizing the search; ⑤ Best Move: outputting the optimal UCI move directly given a board; ⑥ Best Line: outputting a 4-6 ply optimal move sequence with final centipawn delta.
    - **Design Motivation**: Each dataset corresponds to different structures in the MDP—Best Move approximates behavior cloning $\pi_\theta(a_t|s_t)$; Best Line approximates $n$-step bootstrapping ($V + \mathcal{T}$); Verbalized Alpha-Beta explicitly injects the search algorithm; Guided Synthetic injects explicit verbal rollouts. This allows the ablation to determine which inductive bias is most effective rather than just which scores higher.

2.  **Verifiable RL Environment and Dr. GRPO Settings**:
    - **Function**: Execute multitask RL across four tasks simultaneously to prevent single-task reward hacking.
    - **Mechanism**: All rewards are programmatically calculated—Predict Move reward is a normalized rank ($r \in [0,1]$); Best/Worst Move is binary accuracy; Legal Moves uses IoU. The algorithm uses Dr. GRPO (removing GRPO's length normalization) + Clip-Higher (increasing the PPO clip upper bound to encourage exploration) + no KL penalty.
    - **Design Motivation**: It was found that RL on the single Predict Move task is prone to reward hacking, whereas multitasking forces the model to learn move legality and quality simultaneously, resulting in higher quality and more robust models.

3.  **Determining "Reasoning Faithfulness" and the Advantage of Best Line**:
    - **Function**: Quantify the consistency between the reasoning trace and the final move to determine if RL is learning reasoning itself.
    - **Mechanism**: GPT-oss-120b is used as a judge to score the alignment between the reasoning trace and the final answer. Results show that checkpoints from Best Move data exhibit severe inconsistency between reasoning and moves after RL (typical "decide then rationalize"), whereas Best Line checkpoints remain faithful. The authors attribute this to Best Line forcing the model to learn $V$ and $\mathcal{T}$ within the token sequence, effectively internalizing a mini world model.
    - **Design Motivation**: Relying solely on final scores would erroneously suggest both data types are similar; introducing the faithfulness dimension allows selection of training paths that are both high-performing and interpretable.

### Loss & Training
- SFT utilizes LlamaFactory, and RL utilizes veRL. Dr. GRPO + Clip-Higher + no KL are used. A two-stage curriculum (60M token Best Move-All then 60M token Best Line-All) is applied during the scale phase.
- "Chess information density" is measured using Qwen2.5-0.5B SFT for 2 epochs on 4M unique tokens, monitoring the ratio where validation token probability exceeds $0.995$ as a proxy for predictive complexity.

## Key Experimental Results

### Main Results

| Benchmark | Qwen2.5-7B base | gpt-oss-120b | Ours 7B (Best Move + Best Line) |
|---|---|---|---|
| Best Move (1-of-5) | ~Random 0.2 | Strong | **Surpasses gpt-oss-120b** |
| Worst Move | ~Random 0.2 | Strong | **Surpasses gpt-oss-120b** |
| Predict Move (move quality) | Low | Medium | **Substantial Lead** |
| Legal Moves (IoU) | Low | Medium | **Lead** |

| Dataset | Pre-SFT trivial token% | Post-SFT trivial token% | Property |
|---|---|---|---|
| Best Line | 0.00% | 24.0% | Dense |
| Best Move | 0.04% | 28.6% | Dense |
| Factual Board Answering | 0.04% | 62.5% | Partially "flattened" |
| Verbalized Alpha-Beta Pruning | 4.31% | 71.0% | Mostly memorized |
| Guided Synthetic | 2.90% | 9.77% | Moderate |
| Rejection Sampling | 12.69% | 20.8% | NL Redundancy |

### Ablation Study

| Training Recipe | Final Score | RL Stability | Reasoning Faithfulness |
|---|---|---|---|
| Single-task RL (Predict Move) | Poor (Hacking) | Unstable | Average |
| Multitask Rejection Sampling | Medium | Average | Average |
| Best Move (focused) | High | Unstable | **Unfaithful** |
| Best Move-All | Highest | Unstable | **Unfaithful** |
| Best Line (focused) | High | **Stable** | **Faithful** |
| Best Line-All | High | **Stable** | **Faithful** |
| Incl. Verbalized Alpha-Beta | Degraded | — | — |

### Key Findings
- "Dataset diversity > single strongest type"—Best Move-All and Best Line-All outperform their focused versions. Diversity in RL acts as an implicit regularizer.
- RL consistently shifts move quality distribution to the right and suppresses hallucination rates even without explicit rewards for these effects.
- Models trained on Best Line are stronger on OOD evaluations, suggesting they learn a "world model" closer to policy + value, whereas Best Move learns a fragile pure policy.
- Three predictive metrics for SFT-checkpoints (average eval score, cited move validity, reasoning quality LLM score) show significant linear regression, enabling a cost-effective strategy of screening checkpoints with cheap SFT evaluation before expensive RL.

## Highlights & Insights
- Shifts reasoning research from "which RL algorithm is better" to "which SFT data is better," a perspective much more helpful for practitioners since data recipes vary more significantly than RL algorithms.
- The "Chess Information Density" concept decomposes why dense datasets train better into three dimensions: predictive complexity, accuracy, and chess token density. This is a transferable diagnostic framework for other verifiable RL domains (e.g., SAT, circuits, theorem proving).
- Identifies "reasoning faithfulness" as a first-order metric and demonstrates the "high score but unfaithful" phenomenon, warning the alignment community that pursuing task accuracy alone can lead to implicit "rationalization."
- Multitask SFT + multitask RL as a stabilization method is highly practical for industry, as it suppresses reward hacking at almost zero additional cost.

## Limitations & Future Work
- Only conducted on the Qwen2.5-7B base; generalization across families (Llama/Mistral) is unverified.
- "Reasoning faithfulness" is judged by gpt-oss-120b, which might have self-preference bias; no cross-judge control was performed.
- Still loses to OpenAI o3 in full-game play, likely due to a distribution mismatch in opening positions.
- Lack of multi-turn RL or search integration leaves significant room for improvement.

## Related Work & Insights
- **vs DeepSeek-R1 / Kimi k1.5**: While they showed RL+verifiable reward can emerge reasoning in math/code, this work provides quantitative conclusions in chess on which SFT data makes RL more stable/faithful.
- **vs Quiet-STaR / STaR**: Those works assume existing reasoning for bootstrapping; this work shows dense programmatic data (Best Move/Best Line) is more effective than synthetic reasoning samples, suggesting future reasoning data should favor higher task density over natural language.
- **Insights**: Treating Best Line as "explicit world-model SFT" can be applied to dialogue (multi-turn planning), agentic tasks (multi-step action prediction), or code (execution prediction), effectively moving RL "bootstrapping" forward into the SFT phase.

## Rating
- Novelty: ⭐⭐⭐⭐ Using chess for RL reasoning research is novel; information density and faithfulness metrics are solid methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 data types × multitask vs single-task × focused vs all data; complete ablation suite.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with three research questions leading to three conclusion sets.
- Value: ⭐⭐⭐⭐ Provides a reproducible empirical baseline for the "SFT data $\rightarrow$ RL behavior" mapping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICML 2026\] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)
- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)

</div>

<!-- RELATED:END -->
