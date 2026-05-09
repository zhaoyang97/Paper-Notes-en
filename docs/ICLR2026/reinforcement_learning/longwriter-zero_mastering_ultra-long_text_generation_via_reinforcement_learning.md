---
title: >-
  [Paper Note] LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning
description: >-
  [ICLR 2026 (Oral)][Reinforcement Learning][Ultra-long text generation] This paper proposes LongWriter-Zero: starting from a base model, without relying on any annotated or synthetic data, the approach uses GRPO reinforcement learning combined with a three-dimensional composite reward model (length / quality / format) to elicit emergent ultra-long, high-quality text generation. With 32B parameters, the model surpasses 100B+ models such as DeepSeek-R1 and Qwen3-235B on WritingBench.
tags:
  - ICLR 2026 (Oral)
  - Reinforcement Learning
  - Ultra-long text generation
  - GRPO
  - Composite reward model
  - Test-time reasoning
date: 2026-05-08
content_hash: e632a09b4a7b5b53
---

# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning

**Conference**: ICLR 2026 (Oral)
**arXiv**: [2506.18841](https://arxiv.org/abs/2506.18841)
**Code**: [https://huggingface.co/THU-KEG/LongWriter-Zero-32B](https://huggingface.co/THU-KEG/LongWriter-Zero-32B)
**Area**: Reinforcement Learning / Long-form Generation
**Keywords**: Ultra-long text generation, Reinforcement learning, GRPO, Composite reward model, Test-time reasoning

## TL;DR

This paper proposes LongWriter-Zero: starting from a base model, without relying on any annotated or synthetic data, the approach uses GRPO reinforcement learning combined with a three-dimensional composite reward model (length / quality / format) to elicit emergent ultra-long, high-quality text generation. With 32B parameters, the model surpasses 100B+ models such as DeepSeek-R1 and Qwen3-235B on WritingBench.

## Background & Motivation

Ultra-long text generation (reports, novels, legal documents, etc.) is a high-frequency application scenario for LLMs, but two core bottlenecks exist: (1) the maximum generation length is constrained by training distribution, causing quality degradation beyond that range; (2) as sequence length increases, generated text exhibits local incoherence, internal contradictions, repetitive phrasing, topic drift, and structural collapse.

Prior methods exemplified by LongWriter follow a "teaching" paradigm — performing SFT on synthetic long texts. This approach has fundamental limitations:

| SFT Paradigm Limitation | Manifestation |
|:---|:---|
| Data quality bounded by teacher model | Diversity and novelty of synthetic data are capped by existing model capabilities |
| Maximum likelihood objective lacks global signal | Cannot explicitly optimize global properties such as coherence or format consistency |
| High construction cost and unstable quality | Long-text synthesis requires complex agent pipelines; outputs are often incoherent |
| Artificially stylized outputs | Synthetic data exhibits monotonous structural patterns, which are overly artificial |

The authors' core insight is: rather than "teaching" the model how to write (SFT), it is more effective to "incentivize" the model to learn writing on its own (RL). This aligns with the DeepSeek-R1-Zero philosophy — eliciting capabilities entirely through RL from scratch, bypassing dependence on carefully curated training data.

## Method

### Overall Architecture

The final training pipeline of LongWriter-Zero consists of three stages:

1. **Continual Pretraining**: Continual pretraining of Qwen2.5-32B on 30B tokens of writing corpora (Chinese and English books, reports, and academic papers), with an additional 1% of distilled long-CoT data for initial format alignment.
2. **GRPO Reinforcement Learning**: A Think Prompt is used to guide the model to reason before writing, with three reward models providing multi-dimensional training signals.
3. **Inference Deployment**: The model performs planning and reasoning in the `<think>` segment and outputs the final text in the `<answer>` segment.

Training infrastructure: 8 nodes × 8 × H800 GPUs; 32 trajectories sampled per step; maximum output length of 14,000 tokens; sampling temperature $T=0.8$, top-p=1.0.

### Key Design 1: Three-Dimensional Composite Reward Model

This is the core engine of the entire method. Since open-ended text generation lacks ground-truth answers for rule-based verification (unlike mathematics), the authors design three complementary reward models:

**Length RM** — precisely controls target length. QwQ-32B is used to predict a reasonable word-count range $[L_{\text{lower}}, L_{\text{upper}}]$ for each query, and the reward function is piecewise linear:

$$r_{\text{length}}(o) = \begin{cases} 1, & L_{\text{lower}} \le \text{len}(o) \le L_{\text{upper}} \\ \frac{\text{len}(o)}{L_{\text{lower}}}, & \text{len}(o) < L_{\text{lower}} \\ \frac{L_{\text{max}} - \text{len}(o)}{L_{\text{max}} - L_{\text{upper}}}, & \text{len}(o) > L_{\text{upper}} \end{cases}$$

**Writing RM** — evaluates overall writing quality (fluency, coherence, and informativeness). Built on Qwen2.5-72B as backbone, trained on human-annotated preference data using the Bradley-Terry model: $\mathcal{L} = -\mathbb{E}[\log \sigma(r(x, y_w) - r(x, y_l))]$.

**Format RM** — structural integrity and deduplication. Checks whether the output strictly follows the format of one `<think>` segment plus one `<answer>` segment, and penalizes repetitive content based on semantic overlap (during RL training, models tend to inflate length by copying paragraphs).

**Reward Fusion Strategy**: Naively averaging rewards leads to domination by components with larger magnitudes. The authors adopt advantage-level normalization: each reward component is first normalized within the group to $[-1, 1]$, and then the advantage mean is computed:

$$A_{\text{final}} = \frac{1}{3}(A_{\text{length}} + A_{\text{write}} + A_{\text{format}})$$

This ensures equal contribution from all three dimensions, preventing length or format signals from overwhelming writing quality.

### Key Design 2: Test-Time Reasoning for Writing (Test-Time Scaling)

R1-Zero achieves test-time scaling in mathematical reasoning through long chain-of-thought, but whether writing also benefits from a "think before write" paradigm is an open question. The authors design a controlled experiment comparing Think Prompt versus Direct-Answer:

- **Think Prompt**: The model is required to perform comprehensive planning in `<think>` (brainstorming, outlining, style selection, audience adaptation, self-review) before producing the final text in `<answer>`.
- **Direct-Answer**: Planning is skipped; text is generated directly in `<answer>`.

Experimental results: Base-think initially yields lower Writing RM scores than Base-nothink (the model must first learn the think/answer format), but eventually surpasses it and achieves a higher ceiling. The Arena-Write Elo gap is substantial (1221 vs. 668).

An interesting finding: think length in writing converges to an optimal value and plateaus (approximately 2,000–3,000 tokens), unlike mathematical reasoning where it grows indefinitely. This indicates that planning demand in writing has a natural saturation point — once planning is sufficient to produce high-quality text, additional reasoning wastes the context window.

### Key Design 3: Continual Pretraining Raises the RL Ceiling

Prior research has shown that RL performance is bounded by the capability of the base model. The authors verify that this also holds for writing tasks:

- Pretraining corpus: 30B tokens of Chinese and English books, reports, and academic papers (sourced from Common Crawl).
- Format alignment: 1% of long-CoT data distilled from the Base-think model is mixed in; the low ratio avoids memorization of specific CoT patterns.
- Training configuration: batch size 512, packed sequences, maximum context length 32K tokens.

Effect: Continual-Pretrain-think starts with higher Writing RM and Length RM scores than Base-think, and also achieves a higher final convergence value. Arena-Write Elo rises from approximately 1,000 at initialization to approximately 1,400 at convergence, corresponding to a win rate approaching 80% against DeepSeek-R1.

## Key Experimental Results

### Main Results: WritingBench Full-Metric Comparison

| Model | Params | Avg | Academic/Eng. | Finance/Biz. | Law/Policy | Lit./Art | Education | Advertising | Style | Format | Length | Elo |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **LongWriter-Zero** | 32B | **8.69** | **8.7** | **8.8** | **8.8** | 8.4 | **8.9** | **8.6** | **8.7** | **8.7** | 8.6 | **1447** |
| Qwen3-235B-A22B | 235B | 8.68 | 8.6 | 8.6 | 8.6 | 8.7 | 8.8 | 8.6 | 8.7 | **8.7** | **8.7** | 1343 |
| Claude-Sonnet-4 | - | 8.60 | 8.6 | 8.6 | 8.5 | **8.6** | 8.7 | 8.5 | 8.6 | 8.6 | 8.6 | 1185 |
| DeepSeek-R1 | 671B | 8.55 | 8.5 | 8.5 | 8.6 | **8.6** | 8.7 | 8.6 | 8.7 | 8.6 | 8.6 | 1343 |
| GPT-4o | - | 8.16 | 8.1 | 8.1 | 8.2 | 8.1 | 8.4 | 8.1 | 8.3 | 8.2 | 8.2 | 947 |
| LongWriter-8B (SFT) | 8B | 7.91 | 8.0 | 8.1 | 8.1 | 7.7 | 8.1 | 7.6 | 7.9 | 8.1 | 7.7 | 457 |

### Ablation Study

| Configuration | WritingBench Avg | Arena-Write Elo | Key Change |
|:---|:---|:---|:---|
| LongWriter-Zero (full) | **8.69** | **1447** | Continual pretraining + Think + Three rewards |
| w/o continual pretraining (Base-think) | 8.12 | 1221 | Avg drops 0.57, Elo drops 226 |
| w/o thinking (Base-nothink) | 8.04 | 668 | Think has larger impact on Elo (1221→668) |

### SFT vs. RL Comparison

| Initialization | SFT Elo | RL Elo | Gap |
|:---|:---|:---|:---|
| Qwen2.5-32B (Base) | 964 | 1221 | RL +257 |
| Qwen2.5-32B (Cont. Pretrain) | 971 | 1447 | RL +476 |

SFT gains almost nothing from continual pretraining (964 → 971), as performance is locked to the quality of training data; RL benefits substantially (1221 → 1447), indicating that a stronger base model provides a higher exploration ceiling for RL.

### Human Evaluation Win Rate

LongWriter-Zero's GPT-4.1-based automatic evaluation win rate against 6 strong baselines reaches a maximum of 98.2% and a minimum exceeding 62%. Human evaluation (3 annotators) also shows an advantage over DeepSeek-R1 and Qwen3-235B, though human annotators tend to assign ties when differences are subtle.

## Highlights & Insights

- **Paradigm Demonstration**: This work is the first to comprehensively demonstrate that "pure RL outperforms SFT" in open-ended text generation, and systematically addresses three key questions — reward design, test-time scaling, and continual pretraining — through three focused research questions.
- **32B Surpassing 100B+**: The model exceeds DeepSeek-R1 (671B) and Qwen3-235B with fewer than one-seventh of their parameters, demonstrating high parameter efficiency of RL training on writing tasks.
- **Saturation Phenomenon in Writing Reasoning**: Think length converges rather than growing indefinitely during training, revealing a fundamental difference in test-time scaling behavior between writing and mathematical reasoning.
- **Advantage-Level Normalization**: Advantage-level averaging is a practical multi-reward fusion strategy that prevents skewed optimization caused by dimensional imbalance.
- **Fully Open-Sourced**: Data, training framework, reward models, and model weights are all publicly released.

## Limitations & Future Work

- **Factuality Not Covered by Rewards**: The Writing RM does not address fine-grained factual correctness; hallucination risks in long-form text remain unconstrained.
- **Validated Only at 32B Scale**: The approach has not been verified on 7B or smaller models; the lower bound of parameter efficiency for RL-based writing remains unknown.
- **Computational Cost**: The RL training cost on 8 nodes × 8 × H800 GPUs far exceeds that of SFT, posing a high engineering barrier.
- **Evaluation Bias Risk**: Both the WritingBench and Arena-Write judge models originate from specific model families, which may introduce preference leakage.
- **Lack of Style Controllability**: Fine-grained control over specific writing styles (academic vs. literary vs. legal) is not supported; the current reward design is general-purpose.

## Related Work & Insights

- **LongWriter** (SFT approach): Fine-tunes on synthetic long texts; serves as the primary comparison baseline in this paper.
- **R1-Zero / DeepSeek R1**: The paradigm of eliciting reasoning capabilities from scratch via RL; this paper successfully transfers it to writing tasks.
- **WritingBench**: The standard evaluation benchmark for long-form text generation.
- **RLHF / PPO**: The foundational application of policy gradient RL methods to LLM fine-tuning.

**Insights**: This work demonstrates that RL can not only enhance LLM reasoning but also substantially improve generative capabilities. The paradigm of RL from scratch may exhibit similar emergent effects in other tasks (code generation, translation, summarization, etc.). The multi-dimensional reward design strategy is worth adapting for other generative tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First successful application of the R1-Zero paradigm to long-form writing; a pioneering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full-metric SOTA on WritingBench and Arena-Write; evaluation scenarios could be more diverse.
- Writing Quality: ⭐⭐⭐⭐ — Oral-level paper with clear reasoning and well-motivated methodology.
- Value: ⭐⭐⭐⭐⭐ — High practical value; open-sourced model weights advance the RL + writing research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](loongrl_rl_for_reasoning_long_contexts.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/reinforcement_learning/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)

</div>

<!-- RELATED:END -->
