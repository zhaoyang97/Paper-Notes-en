---
title: >-
  [Paper Note] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess
description: >-
  [ICML 2026][Reinforcement Learning][SFT-to-RL] The authors use "training LLMs to play chess" as a clean, verifiable RL testbed, systematically comparing the impact of six custom SFT datasets on RL. They find that "direct…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "SFT-to-RL"
  - "Reasoning Faithfulness"
  - "Chess Reasoning"
  - "Information Density"
  - "GRPO"
date: 2026-05-08
content_hash: 742086c588123b91
---

# How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess

**Conference**: ICML 2026  
**arXiv**: [2604.05134](https://arxiv.org/abs/2604.05134)  
**Code**: https://github.com/(lang-chess) (available)  
**Area**: Reinforcement Learning / LLM Reasoning / Verifiable RL  
**Keywords**: SFT-to-RL, Reasoning Faithfulness, Chess Reasoning, Information Density, GRPO

## TL;DR
The authors use "training LLMs to play chess" as a clean, verifiable RL testbed, systematically comparing the impact of six custom SFT datasets on RL. They find that "directly predicting the best move" achieves the highest scores but leads to unfaithful reasoning after RL, while "predicting the best line" yields comparable performance but more stable and faithful reasoning post-RL. Three metrics are distilled for predicting RL end performance from SFT checkpoints. Ultimately, a 7B model surpasses gpt-oss-120b on multiple chess benchmarks.

## Background & Motivation

**Background**: Current efforts to train LLMs for "reasoning" are almost exclusively in math and code, due to abundant high-quality data and automatic verification; breakthroughs like R1 / DeepSeek-R1, Kimi k1.5, etc., rely on RL with verifiable rewards.

**Limitations of Prior Work**: Controlled experiments are difficult. Math and code datasets are mature, and base models already possess strong capabilities, making it hard to isolate which training data types drive RL improvements. Moreover, research on "reasoning quality" typically focuses only on final accuracy, with little systematic inquiry into whether RL truly teaches reasoning versus merely adjusting answer distributions.

**Key Challenge**: To conduct fine-grained causal analysis of "SFT data → RL behavior," one needs a domain where LLMs are inherently weak (to avoid base model interference) but with ample verifiable rewards; the domain must also allow clear definitions of "reasoning faithfulness."

**Goal**: (a) How do different SFT datasets (programmatic / rejection sampling / synthetic / Best Move / Best Line) affect SFT and RL end performance? (b) How does RL alter the model's move quality distribution, hallucination rate, and reasoning strategy? (c) Which SFT-checkpoint metrics can predict post-RL performance?

**Key Insight**: The authors select chess as the "ideal testbed"—LLMs have historically performed poorly (avoiding strong base bias), chess has an episodic MDP structure, and superhuman oracles like Stockfish provide verifiable rewards and synthetic data. Chess also allows clear definitions of "reasoning faithfulness" (using gpt-oss-120b as judge).

**Core Idea**: Systematic SFT→RL ablation with Qwen2.5-7B, introducing the "chess information density" framework (predictive complexity + accuracy + chess token density) to explain why certain datasets yield stable and strong RL, and distilling key designs for faithful reasoning.

## Method

### Overall Architecture
Unified base: Qwen2.5-7B-Instruct. Board state is represented in visual ASCII format (FEN and spaced FEN were tested but had uneven tokenization), moves use UCI format. Four task suites: Predict Move (choose the best move for a given board), Best Move / Worst Move (5-choice), Legal Moves (IoU evaluation). Workflow: fixed 15M token SFT, 8k samples RL (Dr. GRPO + Clip-Higher + no KL) for ablation to select the strongest recipe, then scale to 60M+60M token SFT + more RL.

### Key Designs

1. **Six Theory-Driven SFT Datasets**:

    - Function: Inject "board understanding + decision-making + reasoning format" from different angles, enabling ablation to isolate contributions.
    - Mechanism: ① General Instruction Following (Magpie Llama 3.3 70B) for regularization; ② Rejection Sampling uses Llama 4 Maverick to generate and retain correct samples on four evaluation tasks, injecting "teacher-style" reasoning; ③ Guided Synthetic gives the teacher a 5-ply line and start/end positions, prompting "verbal $n$-step bootstrapping" (including verbalization of transition function $\mathcal{T}(s_t,a_t)$ and value $V(s_t)$); ④ Verbalized Alpha-Beta Pruning is fully programmatic—based on Stockfish, softmax sampling, recursively building minimax trees and verbalizing the search process; ⑤ Best Move: given a position, directly output the best UCI move; ⑥ Best Line: output a 4-6 ply optimal move sequence with final centipawn delta.
    - Design Motivation: Each dataset corresponds to a different MDP structure—Best Move approximates behavior cloning $\pi_\theta(a_t|s_t)$; Best Line approximates $n$-step bootstrapping (with $V$ and $\mathcal{T}$); Verbalized Alpha-Beta explicitly injects search algorithms; Guided Synthetic explicitly injects verbal rollout of future states. Thus, ablation compares not just "which dataset scores higher," but "which inductive bias is most effective."

2. **Verifiable RL Environment and Dr. GRPO Setup**:

    - Function: Run multitask RL on four tasks simultaneously, avoiding reward hacking in single-task settings.
    - Mechanism: All rewards are programmatically computed—Predict Move uses normalized rank ($r \in [0,1]$, best is 1); Best/Worst Move checks correctness; Legal Moves uses IoU. Algorithm: Dr. GRPO (removing GRPO's length normalization) + Clip-Higher (increasing PPO clip upper bound to encourage exploration) + no KL penalty. 8k samples are evenly split among four tasks for multitask; scaling phase increases this.
    - Design Motivation: The authors find that single-task RL on Predict Move is prone to reward hacking, while multitask forces the model to "know which moves are legal/good/bad," resulting in higher move quality and more robust models. This counterintuitive but useful finding shows that task diversity is more valuable than single-task overtraining.

3. **Faithfulness Judgment and Best Line Advantage**:

    - Function: Quantify "alignment between reasoning trace and final move" to infer whether RL is teaching reasoning itself.
    - Mechanism: Use gpt-oss-120b as judge to score the alignment between reasoning trace and final answer. Results show that SFT checkpoints trained on Best Move data become unfaithful after RL (typical "decide first, justify later"), while Best Line-trained checkpoints remain faithful post-RL. The authors attribute this to Best Line forcing the model to learn $V$ and $\mathcal{T}$ in the token sequence, effectively internalizing a mini world model; Best Move only teaches policy $\pi_\theta$, so RL improves latent ability but decouples surface reasoning from answers (echoing Turpin et al. 2023's unfaithful CoT).
    - Design Motivation: Looking only at final scores would lead to the mistaken conclusion that "both datasets perform similarly"; introducing faithfulness allows researchers to choose "similarly performing but more trustworthy" training paths, which is crucial for safety and interpretability.

### Loss & Training

- SFT uses LlamaFactory, RL uses veRL; Dr. GRPO + Clip-Higher + no KL; scaling phase uses a two-stage curriculum: 60M token Best Move-All followed by 60M token Best Line-All.
- "Chess information density" is measured by SFT-ing Qwen2.5-0.5B for 2 epochs on 4M unique tokens, monitoring the proportion of validation tokens assigned probability $>0.995$ as a proxy for predictive complexity.

## Key Experimental Results

### Main Results

| Benchmark | Qwen2.5-7B base | gpt-oss-120b | Ours 7B (Best Move + Best Line) |
|---|---|---|---|
| Best Move (5-choice) | Near random 0.2 | Strong | **Exceeds gpt-oss-120b** |
| Worst Move | Near random 0.2 | Strong | **Exceeds gpt-oss-120b** |
| Predict Move (move quality) | Low | Medium | **Significantly ahead** |
| Legal Moves (IoU) | Low | Medium | **Ahead** |

| Dataset | Pre-SFT trivial token% | Post-SFT trivial token% | Nature |
|---|---|---|---|
| Best Line | 0.00% | 24.0% | dense, high density |
| Best Move | 0.04% | 28.6% | dense, high density |
| Factual Board Answering | 0.04% | 62.5% | some sub-tasks collapsed |
| Verbalized Alpha-Beta Pruning | 4.31% | 71.0% | most search phrases memorized |
| Guided Synthetic | 2.90% | 9.77% | moderate |
| Rejection Sampling | 12.69% | 20.8% | natural language redundancy |

### Ablation Study

| Training recipe | End performance | RL stability | Reasoning faithfulness |
|---|---|---|---|
| Single-task RL (Predict Move) | Poor, prone to reward hack | Unstable | Average |
| Multitask Rejection Sampling | Medium | Average | Average |
| Best Move (focused) | High | Unstable | **Unfaithful** |
| Best Move-All (all data) | Highest | Unstable | **Unfaithful** |
| Best Line (focused) | High | **Stable** | **Faithful** |
| Best Line-All | High | **Stable** | **Faithful** |
| With Verbalized Alpha-Beta | Actually hurts performance | — | — |

### Key Findings
- "Dataset diversity > single strongest"—Best Move-All and Best Line-All outperform their focused versions, even when including Verbalized Alpha-Beta Pruning, which is individually harmful; indicating that diversity in RL acts as implicit regularization.
- RL consistently shifts move quality distribution rightward and reduces hallucination rate, even without explicit rewards for these side effects; notably, post-RL move accuracy (whether moves mentioned in reasoning traces are legal) becomes one of the strongest SFT→RL performance predictors.
- RL models trained on Best Line generalize better on OOD evals (unseen during training), indicating they learn a "world model" closer to policy + value, while Best Move learns a more brittle pure policy.
- Three SFT-checkpoint metrics (average eval score / legal move citation rate / reasoning quality LLM score) are significant linear predictors (Fig. 9), suggesting a cost-effective strategy: use cheap SFT evals to filter checkpoints before expensive RL training.

## Highlights & Insights
- Shifts reasoning research from "which RL algorithm is best" to "which SFT data is best," a far more practical perspective—RL algorithm differences are limited, but data recipes vary greatly.
- The "Chess Information Density" concept decomposes why dense datasets train well into predictive complexity + accuracy + chess token density, offering a diagnostic framework transferable to other verifiable RL domains (e.g., SAT, circuits, theorem proving).
- Proposes "reasoning faithfulness" as a primary metric, with experiments showing "high scores but unfaithful reasoning," warning alignment researchers that optimizing only for task accuracy leads models to implicitly "rationalize."
- Multitask SFT + multitask RL effectively suppresses reward hacking, a highly practical and nearly cost-free stabilization method for industry.

## Limitations & Future Work
- Only tested on Qwen2.5-7B base; cross-family generalization (Llama / Mistral) unverified.
- "Reasoning faithfulness" judged by gpt-oss-120b, possibly subject to self-preference bias (Panickssery 2024); no cross-judge controls performed.
- Still loses to OpenAI o3 in full-game play, attributed to training on mid/late-game positions and opening distribution mismatch; thus, this work is not a "general chess agent," but a strong "chess puzzle and middlegame reasoning" model.
- No multi-round RL or tree search integration, leaving significant room for improvement.

## Related Work & Insights
- **vs DeepSeek-R1 / Kimi k1.5**: These works show RL + verifiable reward can elicit reasoning in math/code; this paper provides quantitative conclusions on "which SFT data makes RL more stable/faithful" in a more controlled chess setting.
- **vs Quiet-STaR / STaR**: Those assume existing "successful reasoning" for bootstrapping; this work shows dense programmatic data (Best Move/Best Line) is more effective than synthetic reasoning samples, suggesting future reasoning data construction should favor "less natural language, more task density."
- **vs DeepMind 270M Chess Transformer**: That model is pure next-move prediction without reasoning traces; this work uses 7B + reasoning traces, demonstrating that "language reasoning + RL" can achieve both chess ability and interpretable traces (though still weaker than search-based engines).
- Insight: Treating Best Line's "output multi-step moves + value" as "explicit world-model SFT" can be applied to dialogue (multi-turn planning + end value), agentic tasks (multi-step actions + reward estimation), code (multi-step execution + output prediction), etc.—effectively moving RL's "bootstrapping" into the SFT stage.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel use of chess for RL reasoning research; introduction of "chess information density" and faithfulness metrics are solid methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 data types × multitask vs single-task × focused vs all data × multiple metrics, with comprehensive ablation.
- Writing Quality: ⭐⭐⭐⭐ Three research questions → three experiment groups → three conclusions, clear structure; thick appendix but concise main text.
- Value: ⭐⭐⭐⭐ Provides a reproducible empirical baseline for "SFT data→RL behavior" mapping; reasoning faithfulness is directly valuable for safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[ICML 2026\] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)

</div>

<!-- RELATED:END -->
