---
title: >-
  [Paper Note] Learning from Synthetic Data Improves Multi-hop Reasoning
description: >-
  [ICLR 2026][Reinforcement Learning][Synthetic Data] This paper finds that RLVR training on fully fictitious, rule-generated synthetic data significantly improves LLM performance on real-world multi-hop reasoning tasks (5…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Synthetic Data"
  - "Multi-hop Reasoning"
  - "RLVR"
  - "Knowledge Composition"
  - "Phantom Worlds"
date: 2026-05-08
content_hash: 17fc9074d7b12068
---

# Learning from Synthetic Data Improves Multi-hop Reasoning

**Conference**: ICLR 2026
**arXiv**: [2603.02091](https://arxiv.org/abs/2603.02091)  
**Code**: [GitHub](https://github.com/kilian-group/phantom-reasoning)  
**Area**: LLM Reasoning / Reinforcement Learning
**Keywords**: Synthetic Data, Multi-hop Reasoning, RLVR, Knowledge Composition, Phantom Worlds

## TL;DR
This paper finds that RLVR training on fully fictitious, rule-generated synthetic data significantly improves LLM performance on real-world multi-hop reasoning tasks (56%–131% gains for Qwen3-0.6B), because the model learns knowledge composition as a generalizable reasoning skill rather than memorizing factual knowledge.

## Background & Motivation

**Background**: RLVR has achieved notable progress in training LLM reasoning via verifiable rewards, particularly in mathematics and coding. However, RLVR relies on large quantities of high-quality verifiable data — human annotation is expensive, and LLM-generated synthetic data suffers from hallucinations and high cost.

**Limitations of Prior Work**: (1) High-quality training data is scarce and costly; (2) LLM-generated synthetic data inherits verification difficulties and pretraining knowledge contamination; (3) Rule-generated synthetic data is semantically simple and entirely fictitious, raising doubts about whether it can teach useful skills.

**Key Challenge**: There is a substantial gap between PhantomWiki questions such as "Who is the nephew of the friend of the person whose hobby is birdwatching?" and HotpotQA questions such as "Aside from Yodobashi, what other towns were merged into..." — fictitious simple templates vs. real complex language. Transfer from the former to the latter is far from obvious.

**Key Insight**: The paper hypothesizes that the core of multi-hop reasoning is *knowledge composition* — the ability to chain multi-step information — which is a domain-agnostic skill. Zero knowledge overlap in fictitious worlds prevents the model from taking shortcut memorization, forcing it to learn the compositional operation itself.

**Core Idea**: Rule-generated fictitious synthetic data teaches LLMs knowledge composition as a generalizable skill via RLVR, enabling free and infinitely scalable transfer to real-world multi-hop reasoning.

## Method

### Overall Architecture
Four rule-generated synthetic datasets (PhantomWiki / GSM-∞ / RG-Family / RG-Knights) are used for RLVR training, and transfer is evaluated on five real-world multi-hop QA benchmarks. The GRPO algorithm is applied across four LLM families (Qwen3/Phi-4, 0.6B–4B).

### Key Designs

1. **Synthetic Dataset Selection**:

    - PhantomWiki: Multi-hop QA over fictitious characters, generated via templates and context-free grammars, with 1–9 hops of difficulty.
    - GSM-∞: Infinitely diverse math word problems generated from random computation graphs converted to natural language, with 2–20 steps.
    - RG-Family: Inferring the relationship between two individuals in a family tree (logical reasoning).
    - RG-Knights: Knights-and-knaves logic puzzles.
    - Design Motivation: Covers diverse reasoning styles; all datasets are grounded in fictitious worlds with zero knowledge overlap.

2. **In-context Reasoning Setup**:

    - Function: All relevant context is placed in the prompt to test in-context reasoning ability.
    - Mechanism: PhantomWiki includes all 25 articles; GSM-∞ includes the problem description; answers are extracted via `<answer>` tags.
    - Design Motivation: Controls for confounds, ensuring that reasoning ability rather than knowledge retrieval is measured.

3. **Causal Analysis**:

    - Format ablation: RLVR using only the `<answer>` format yields no improvement on Qwen3/Phi, demonstrating that transfer stems from reasoning ability rather than format learning.
    - SFT comparison: SFT is effective on synthetic tasks but fails to transfer to real tasks, demonstrating that RL teaches skills rather than patterns.
    - Intermediate answer analysis: The frequency of correct intermediate answers increases during training, demonstrating growth in compositional ability.

### Loss & Training
- GRPO (without KL penalty), Hugging Face TRL v0.21.0.
- PhantomWiki uses F1 reward (multiple valid answers); other datasets use binary exact-match reward.
- 10K training samples with mixed difficulty levels.

## Key Experimental Results

### Main Results
Qwen3-0.6B + PhantomWiki training → real-world benchmarks:

| Benchmark | Base F1 | +PhantomWiki | Relative Gain |
|-----------|---------|--------------|---------------|
| HotpotQA | 0.36 | 0.73 | +103% |
| 2WikiMQA | 0.37 | 0.86 | +132% |
| MuSiQue | 0.14 | 0.28 | +100% |
| CofCA | Low | Significant ↑ | +56–131% |
| SynthWorlds | Low | Significant ↑ | +Large |

### Ablation Study

| Configuration | Synthetic Task | Real Task | Notes |
|---------------|---------------|-----------|-------|
| RLVR on PhantomWiki | ✓ Improves | ✓ Transfers | Full method |
| SFT on PhantomWiki | ✓ Improves | ✗ Does not transfer | SFT overfits to patterns |
| Format-only RLVR | ✗ No benefit | ✗ No benefit | Qwen3/Phi already handle format |
| More synthetic data | Monotonic ↑ | Monotonic ↑ | No sign of overfitting |

### Key Findings
- All four synthetic datasets produce positive transfer; PhantomWiki (most aligned with target tasks) achieves the best results.
- SFT improves performance on synthetic tasks but does not transfer, whereas RL demonstrably teaches generalizable skills rather than surface patterns.
- The model generalizes to held-out fictitious worlds and OOD difficulty levels, confirming that a transferable compositional skill has been acquired.
- Performance increases monotonically with the number of synthetic samples without overfitting, establishing synthetic data as a scalable resource.
- The number of correct intermediate answers generated by the model increases during training, indicating emergent grounded reasoning.

## Highlights & Insights
- **The ultimate "free lunch"**: No real data, no LLM annotation, and no GPU generation are required — infinitely many training examples can be generated on a standard computer using templates, yet real-world reasoning ability is substantially improved.
- **The SFT vs. RL divide**: Given identical synthetic data, SFT overfits to the surface patterns of the synthetic task, while RL instills deep compositional skills. This constitutes strong evidence for the independent value of RL.
- **Knowledge composition as an independent skill**: The paper cleanly separates "knowing facts" from "composing facts," demonstrating that the latter can be learned independently. This challenges the view that RL merely activates pretraining knowledge.

## Limitations & Future Work
- A gap between real and synthetic task performance remains — RLVR on real data still performs better.
- Experiments are limited to 0.6B–4B models; transfer dynamics may differ for larger models.
- Multi-hop reasoning is a relatively simple reasoning type; transfer to more complex reasoning (e.g., mathematical proof) remains to be verified.
- The optimal difficulty distribution and mixing strategy for synthetic data are not thoroughly explored.

## Related Work & Insights
- **vs. Real-data RLVR**: Real data yields better performance but is costly; synthetic data is free and infinitely scalable — the two are complementary.
- **vs. LLM distillation synthesis**: LLM-generated data is expensive and risks contamination; rule-generated data is clean and free.
- **vs. STILL/ART et al.**: Prior work uses synthetic data for RL evaluation; this paper is the first to systematically demonstrate transfer from synthetic to real-world tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of transfer from fictitious synthetic data to real-world reasoning is of significant importance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four model families × four synthetic datasets × five real benchmarks × multiple ablations — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is clear and the causal analysis is rigorous.
- Value: ⭐⭐⭐⭐⭐ Opens a new paradigm for improving LLM reasoning with free synthetic data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[AAAI 2026\] MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy](../../AAAI2026/reinforcement_learning/mathsmith_towards_extremely_hard_mathematical_reasoning_by_forging_synthetic_pro.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/reinforcement_learning/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICLR 2026\] $\textbf{Re}^{2}$: Unlocking LLM Reasoning via Reinforcement Learning with Re-solving](textbfre2_unlocking_llm_reasoning_via_reinforcement_learning_with_re-solving.md)

</div>

<!-- RELATED:END -->
