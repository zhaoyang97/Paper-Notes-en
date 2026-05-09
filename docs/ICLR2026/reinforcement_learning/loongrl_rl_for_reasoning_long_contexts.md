---
title: >-
  [Paper Note] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts
description: >-
  [ICLR 2026 Oral][Reinforcement Learning][long-context reasoning] This paper proposes LoongRL, which constructs KeyChain synthetic data for reinforcement learning training to elicit a plan–retrieve–reason–recheck reasoning pattern in LLMs for long-context tasks. Training solely on 16K contexts generalizes to 128K; the 14B model achieves 74.2, approaching o3-mini (74.5) and DeepSeek-R1 (74.9).
tags:
  - ICLR 2026 Oral
  - Reinforcement Learning
  - long-context reasoning
  - reinforcement-learning
  - GRPO
  - multi-hop QA
  - emergent reasoning patterns
date: 2026-05-08
content_hash: c2e40d5772b91370
---

# LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts

**Conference**: ICLR 2026 Oral
**arXiv**: [2510.19363](https://arxiv.org/abs/2510.19363)
**Code**: Available (training code and KeyChain data synthesis code provided in supplementary material)
**Area**: Reinforcement Learning
**Keywords**: long-context reasoning, reinforcement-learning, GRPO, multi-hop QA, emergent reasoning patterns

## TL;DR
This paper proposes LoongRL, which constructs KeyChain synthetic data for reinforcement learning training to elicit a plan–retrieve–reason–recheck reasoning pattern in LLMs for long-context tasks. Training solely on 16K contexts generalizes to 128K; the 14B model achieves 74.2, approaching o3-mini (74.5) and DeepSeek-R1 (74.9).

## Background & Motivation

**Background**: Recent advances in LLM reasoning (e.g., DeepSeek-R1, o1) focus primarily on short-context reasoning tasks (mathematics, code), using RL to elicit longer chain-of-thought and self-reflection. Long-context reasoning—requiring retrieval and integration of information from thousands of tokens of external input—remains largely unexplored.

**Limitations of Prior Work**: (a) Existing long-context models support extended windows (128K+) but excel mainly at **retrieval**, performing poorly in scenarios requiring **reasoning**; (b) high-difficulty long-context data suitable for RL training is extremely scarce, and answer format diversity complicates verification; (c) scaling RL rollouts from short text (<1K) to 128K contexts is computationally prohibitive; (d) training exclusively on long-context data degrades short-context capability.

**Key Challenge**: Long-context reasoning requires a distinctive cognitive pattern (plan → retrieve → reason → verify), which cannot be acquired through simple SFT or prompting and must be explored and incentivized via RL. However, appropriate RL training data does not exist—it must be sufficiently difficult to trigger reasoning, require retrieving information from long contexts, and produce verifiable answers.

**Goal**: (a) How to design high-quality RL training data that incentivizes long-context reasoning? (b) How to train on short contexts yet generalize to ultra-long contexts? (c) How to preserve short-context capability without degradation?

**Key Insight**: The authors observe that if RL training data inherently requires multi-step operations—"trace a chain of clues → identify the true question → retrieve → reason"—models naturally develop structured long-context reasoning patterns. Once acquired, this pattern generalizes to arbitrary lengths.

**Core Idea**: By embedding UUID chains into short multi-hop QA instances to conceal the true question (KeyChain), the authors construct high-difficulty RL training data that elicits a plan-retrieve-reason-recheck reasoning pattern in LLMs and enables generalization to 128K contexts.

## Method

### Overall Architecture
The LoongRL pipeline proceeds as follows: (1) start from existing multi-hop QA datasets → (2) transform them into high-difficulty long-context problems via KeyChain → (3) apply multi-stage RL training using GRPO → (4) mix mathematics and retrieval data to preserve short-context capability. The input is a ~16K-token long document paired with a question; the output is a reasoning-augmented response.

### Key Designs

1. **KeyChain Data Construction**

    - **Function**: Transforms simple short-text multi-hop QA into high-difficulty long-context reasoning tasks.
    - **Mechanism**: First, QA pairs of moderate difficulty are filtered from HotpotQA, MuSiQue, and 2WikiMultiHopQA (277K→72K; retained samples have pass rates strictly between 0 and 1 when answered 8 times by Qwen2.5-32B). Then: (a) irrelevant documents are inserted to extend context to ~16K tokens; (b) multiple UUID key-value chains are inserted, one of which ultimately points to the original question $o\_q_i$ while the rest point to distractor questions. Each key is a 32-character UUID (0–9, A–F), and each value contains either the next key or the final question. The model must trace the correct chain from an initial key → locate the hidden true question → retrieve relevant information from the long context → reason to derive the answer.
    - **Design Motivation**: Simply adding distractor documents provides limited difficulty increase, as models can still retrieve directly. The key insight of KeyChain is that it compels the model to first "determine what the question is," a preprocessing step that naturally induces the plan-retrieve-reason structured cognitive pattern.

2. **Two-way Substring Exact Match Reward Verifier**

    - **Function**: Provides a reliable binary reward signal for RL.
    - **Mechanism**: $r_i = 1$ if and only if $a \subseteq y_{\text{ans}} \lor y_{\text{ans}} \subseteq a$, i.e., the predicted answer contains the ground truth or vice versa (bidirectional substring matching). The model is required to place its final answer inside `\boxed{}` for extraction.
    - **Design Motivation**: General QA answers are highly diverse (unlike mathematics with unique solutions); strict exact match penalizes correct but differently formatted answers; F1 score and LLM-as-a-judge yield suboptimal results, with the latter incurring additional model overhead. Bidirectional substring matching achieves a favorable balance between leniency and accuracy.

3. **Three-stage Multi-curriculum RL Training**

    - **Function**: Progressively increases task difficulty to prevent training instability caused by excessive initial difficulty.
    - **Mechanism**:
        - **Warm-up** (42 steps; 7B only): Training on data without KeyChain (standard multi-hop QA + retrieval + mathematics) to establish foundational capabilities.
        - **Stage I** (168 steps): KeyChain data is introduced to encourage the model to learn the plan-retrieve-reason-recheck pattern.
        - **Stage II** (~120–150 steps): 8 rollouts are generated per sample; samples where all rollouts are correct (~60–70%) are discarded, and training continues only on the remaining hard samples to avoid overfitting on already-mastered examples.
    - **Design Motivation**: Smaller models (7B) have insufficient initial capability—directly applying KeyChain causes all rollouts to fail (zero reward), producing no effective gradient signal. Larger models (14B) can skip warm-up.

4. **Mixed Data Recipe**

    - **Function**: Balances long-context reasoning with general short-context capability.
    - **Mechanism**: Training data comprises 7,500 KeyChain QA + 7,500 standard multi-hop QA + 1,024 needle retrieval + 5,000 mathematics problems (DAPO + MATH), all constrained to ~16K context length.
    - **Design Motivation**: Training exclusively on long-context data degrades short-context capability (observed in both the R1-distill series and QwenLong-L1); incorporating mathematics data preserves general reasoning ability.

### Loss & Training
GRPO (Group Relative Policy Optimization) is adopted with group size $G=8$, learning rate $1 \times 10^{-6}$, cosine decay, KL penalty $\beta=0.001$, and the entropy loss term removed (to avoid training instability). Maximum output length is 4,096 tokens; inference is performed at temperature 0.6 with top-p 0.95.

## Key Experimental Results

### Main Results

| Model | LongBench v1 Avg | HotpotQA | 2Wiki | MuSiQue | NarrativeQA | QASPER |
|-------|------------------|----------|-------|---------|-------------|--------|
| o3-mini | 74.5 | 83.0 | 89.0 | 64.0 | 60.7 | 60.5 |
| DeepSeek-R1 | 74.9 | 82.7 | 91.3 | 72.2 | 66.9 | 61.4 |
| QwenLong-L1-32B | 70.1 | 80.7 | 89.1 | 65.2 | 58.6 | 56.7 |
| Qwen2.5-7B-Instruct | 48.9 | 69.5 | 50.5 | 34.0 | 44.5 | 46.0 |
| **LoongRL-7B** | **72.4** | 83.1 | 91.1 | 65.6 | 58.4 | 63.6 |
| Qwen2.5-14B-Instruct | 53.1 | 74.0 | 60.5 | 36.5 | 48.5 | 46.0 |
| **LoongRL-14B** | **74.2** | 82.2 | 93.3 | 67.5 | 63.4 | 64.5 |

### Ablation Study

| Configuration | LongBench v1 Avg | Note |
|---------------|------------------|------|
| Qwen2.5-7B-Instruct | 48.9 | Baseline |
| LoongRL-7B (no KeyChain) | 66.2 | KeyChain replaced with equal-volume standard multi-hop QA |
| LoongRL-7B (full) | 72.4 | Full model; KeyChain contributes +6.2% |

| Reward Verifier | Avg | Note |
|----------------|-----|------|
| F1 score | 65.1 | Insufficient precision |
| LLM-as-a-judge | 65.2 | Requires additional model; poor performance |
| Exact match | 69.2 | Overly strict |
| Two-way Substring (Ours) | **72.4** | Best |

### Key Findings
- **KeyChain is the core contribution**: Removing KeyChain drops performance from 72.4 to 66.2, a substantial margin. Models trained with KeyChain exhibit explicit planning steps and rechecking behavior, whereas models trained without KeyChain produce entangled reasoning and retrieval with no discernible planning structure.
- **16K training generalizes to 128K**: LoongRL-7B achieves 76.8 on RULER 128K (baseline: 69.4); LoongRL-14B achieves 79.9 (baseline: 73.6). On NarrativeQA in the 32K–64K range, gains are +14.8% and +16.0% respectively.
- **Reward verifier comparison**: Two-way substring matching substantially outperforms F1, LLM-as-a-judge, and exact match, underscoring the importance of appropriate answer-matching relaxation for open-ended QA.
- **Perfect needle-in-a-haystack performance**: LoongRL-7B achieves 100% accuracy across all depth and length settings—surpassing even the baseline Qwen2.5-7B and QwenLong-L1-32B, which fail to achieve perfect scores.
- **Short-context capability preserved**: MMLU improves by +2.8%/+1.1%; IFEval exhibits only marginal drops of −0.3%/−2.6%; mathematics performance remains stable.

## Highlights & Insights
- **The KeyChain data construction is remarkably elegant**: By structurally embedding "find the question before answering it" into the data, the approach naturally elicits planning capability. This idea is broadly transferable to other tasks requiring structured reasoning—rather than explicitly teaching the model how to reason, the data is designed such that reasoning becomes necessary for answering.
- **Short training, long generalization**: The finding that 16K-context training generalizes to 128K is significant, indicating that once a reasoning pattern is learned, it is length-agnostic. This substantially reduces the cost of long-context RL training (128K rollouts would otherwise be computationally infeasible).
- **Two-way substring matching**: A simple yet effective reward design that handles QA answer diversity without incurring the overhead or reward hacking risks associated with LLM-as-a-judge. Directly reusable for RL training on other open-ended QA tasks.
- **Multi-stage curriculum learning**: The warm-up → KeyChain → hard-mining strategy is highly practical. The Stage II hard-mining approach (discarding samples where all rollouts are correct) is particularly noteworthy for avoiding wasted computation on already-mastered examples.

## Limitations & Future Work
- **Evaluation limited to QA-type tasks**: LongBench and NarrativeQA are primarily extractive/generative QA benchmarks; other long-context task types such as long-document summarization and cross-document reasoning have not been evaluated.
- **Fixed training length of 16K**: While 16K→128K generalization is effective, whether the approach remains valid for even longer contexts (256K, 1M) has not been verified.
- **Synthetic nature of KeyChain**: UUID chains are entirely artificially constructed, creating a distributional gap relative to real-world "information tracing" tasks. Whether more naturalistic KeyChain variants can be designed remains an open question.
- **Experiments confined to the Qwen series**: Generalizability to other architectures such as LLaMA and Mistral has not been validated.
- **Insufficient analysis of emergent patterns**: Whether the plan-retrieve-reason-recheck pattern emerges consistently, and what failure cases look like, remain underexplored.

## Related Work & Insights
- **vs. QwenLong-L1**: QwenLong-L1 trains R1-distill-Qwen-32B with RL on 60K contexts, yielding only a +4.6% improvement. LoongRL surpasses it by +2.3% using a 7B model trained on 16K contexts. The critical difference lies in the quality of KeyChain data.
- **vs. R1-Distill series**: R1 distillation performs poorly on long contexts and may even degrade (−17.7% for 7B), because distilled long-CoT data primarily targets short-context reasoning and does not cover the retrieval-reasoning patterns unique to long-context settings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The KeyChain data construction is highly creative; however, the underlying RL framework (GRPO) is not itself a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparisons (including o3-mini and R1), thorough ablations (KeyChain, verifier, training strategy), and additional 128K generalization and NIAH evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, methodology is described fluently, figures and tables are intuitive, and the appendix is detailed with training trajectory comparisons.
- **Value**: ⭐⭐⭐⭐⭐ Provides an efficient and reproducible solution for long-context LLM reasoning; the KeyChain data construction methodology is broadly reusable.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning](echo_towards_advanced_audio_comprehension_via_audio-interleaved_reasoning.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning](longwriter-zero_mastering_ultra-long_text_generation_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](../../NeurIPS2025/reinforcement_learning/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

<!-- RELATED:END -->
