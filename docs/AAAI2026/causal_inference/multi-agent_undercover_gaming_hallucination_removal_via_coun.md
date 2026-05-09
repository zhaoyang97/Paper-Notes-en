---
title: >-
  [Paper Note] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning
description: >-
  [AAAI 2026][Causal Inference][Multi-Agent Gaming] MUG reframes Multi-Agent Debate (MAD) as a "Who's Undercover" social reasoning game — by introducing information asymmetry through counterfactual image editing (modifying the reference image), one agent is assigned the edited image $I^-$ as the "undercover," while other agents hold the original image $I^+$ and identify the undercover (i.e., the hallucination source) via reasoning and voting. On HallusionBench, Qwen2.5VL-7B improves from 46.4% to 53.8%.
tags:
  - AAAI 2026
  - Causal Inference
  - Multi-Agent Gaming
  - Counterfactual Testing
  - Hallucination Detection
  - Undercover Game
  - Active Reasoning
date: 2026-05-08
content_hash: caf57ec59b41331d
---

# MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.11182](https://arxiv.org/abs/2511.11182)
**Code**: [https://github.com/YongLD/MUG.git](https://github.com/YongLD/MUG.git)
**Area**: Causal Reasoning
**Keywords**: Multi-Agent Gaming, Counterfactual Testing, Hallucination Detection, Undercover Game, Active Reasoning

## TL;DR
MUG reframes Multi-Agent Debate (MAD) as a "Who's Undercover" social reasoning game — by introducing information asymmetry through counterfactual image editing (modifying the reference image), one agent is assigned the edited image $I^-$ as the "undercover," while other agents hold the original image $I^+$ and identify the undercover (i.e., the hallucination source) via reasoning and voting. On HallusionBench, Qwen2.5VL-7B improves from 46.4% to 53.8%.

## Background & Motivation
**Background**: Multi-Agent Debate (MAD) enhances reasoning quality through structured discussion among multiple LLM agents and is a promising direction for hallucination mitigation.

**Limitations of Prior Work**: MAD suffers from three fundamental limitations — (1) it relies on the unrealistic assumption that all debaters are rational: when agents themselves are prone to hallucination, consensus may converge to a shared error; (2) it depends on statistical consensus (e.g., majority voting) without a genuine fact-checking mechanism; (3) agents passively answer questions rather than actively investigating and verifying claims.

**Key Challenge**: The consensus mechanism in MAD is fundamentally a "group statistics" approach — if the majority of agents share the same hallucination, consensus converges to an incorrect answer. A mechanism is needed that identifies "who is hallucinating" rather than "who is in the minority."

**Goal**: How can hallucinating agents be detected and excluded without assuming agent rationality?

**Key Insight**: Inspired by the "Who's Undercover" social reasoning game — by providing one agent with a modified image (counterfactual evidence), verifiable information asymmetry is introduced. Since the edit is known, a ground truth is available to determine who is the undercover (the hallucinator).

**Core Idea**: Use counterfactual image editing to create information asymmetry, combined with an undercover game mechanism to detect hallucinating agents, replacing statistical consensus with fact-checking.

## Method

### Overall Architecture
Two-phase game: (1) **Undercover Detection Phase** — among $N$ agents, one is assigned the counterfactual image $I^-$ as the undercover (Role=U), while others hold the original image $I^+$ (Role=D). Multi-round reasoning and voting eliminate the most suspicious agent. (2) **Summarization Phase** — after the undercover is eliminated, the remaining agents collaboratively generate the final answer based on $I^+$.

### Key Designs

1. **Counterfactual Image Generation $I^-$**:

    - Function: Performs fine-grained editing on the original reference image to introduce controlled semantic changes.
    - Mechanism: An LLM first analyzes the question type and maps it to an edit type ("how many" → quantity editing, "what object" → object replacement, "what color" → attribute modification). A scene graph is constructed to identify the edit target, and the Step1X-Edit model executes the edit. The edit must satisfy three constraints: maximizing visual similarity $C_{vs}$ (ViT embeddings), semantic consistency $C_{sc}$ (CLIP similarity), and naturalness $C_{na}$ (FID score).
    - Design Motivation: The edit must be subtle enough (so the undercover is not trivially identified) while introducing meaningful semantic differences (sufficient to distinguish reasoning correctness).

2. **Undercover Detection Game**:

    - Function: Identifies and eliminates the undercover agent through multi-round reasoning and voting.
    - Reasoning Phase: Each agent reasons based on its assigned image ($I^+$ or $I^-$). Normal agents aim to provide accurate reasoning and identify inconsistencies; the undercover agent aims to generate plausible-sounding reasoning while avoiding detection.
    - Voting Phase: Each agent votes based on four factors — inconsistency score $\phi_1$, deviation-from-consensus score $\phi_2$, detail accuracy $\phi_3$, and behavioral suspicion $\phi_4$. A majority vote eliminates the most suspicious agent.
    - Design Motivation: Information asymmetry creates strategic tension — normal agents prove they are viewing $I^+$ by providing specific details, while the undercover must balance plausibility against accuracy.

3. **Three Dimensions of Innovation (vs. Traditional MAD)**:

    - **Fact-Checking vs. Statistical Consensus**: Counterfactual testing provides ground truth to verify who is hallucinating, rather than relying solely on majority voting.
    - **Cross-Evidence vs. Single Source**: Additional evidence is dynamically generated by modifying the image; traditional MAD relies on a single original image.
    - **Active Reasoning vs. Passive Answering**: Agents actively question, verify, and debate rather than merely answering a given question.

## Key Experimental Results

### Main Results

| Method | MMMU | MMStar | HallusionBench Avg | POPE Acc |
|------|------|--------|-------------------|----------|
| Qwen2.5VL-7B (baseline) | 45.0 | 61.2 | 46.4 | 87.4 |
| Qwen2.5VL-7B (Self-Refine) | 45.8 | 61.5 | 48.8 | 85.9 |
| Qwen2.5VL-7B (MAD-Vote) | 44.7 | 57.4 | 37.8 | 80.0 |
| Qwen2.5VL-7B (MAD-Judge) | 47.4 | 62.3 | 50.2 | 85.2 |
| **Qwen2.5VL-7B (MUG)** | **50.3** | **63.8** | **53.8** | **88.4** |

### Ablation Study

| Configuration | MMStar | HallusionBench | MMMU |
|------|--------|---------------|------|
| MUG Full | 63.80 | 53.80 | 50.33 |
| w/o Counterfactual Editing | 62.31 (-1.49) | 50.19 (-3.61) | 49.25 (-1.08) |
| w/o Undercover Mechanism | 62.23 (-1.57) | 49.31 (-4.49) | 47.66 (-2.67) |

### Key Findings
- MUG enables Qwen2.5VL-7B to surpass GPT-4v on MMMU (50.3 vs. 53.8 — the latter being a much larger model); MAD-Vote instead degrades performance (−7.4% on MMMU).
- The improvement on HallusionBench is most pronounced — MUG achieves +7.4% over baseline, while MAD-Vote yields −8.6% — indicating that MAD can be detrimental for hallucination detection.
- The undercover mechanism contributes more than counterfactual editing (its removal causes a larger performance drop), suggesting that "game dynamics" contribute more than "additional evidence."
- One detection round yields the best results (50.3/63.8/69.4); additional rounds degrade performance — indicating that prolonged debate may mislead normal agents.
- Additional time overhead is only 0.91 seconds per sample (3.74s vs. MAD's 2.35s), offering an excellent cost-performance ratio.

## Highlights & Insights
- The analogy **social reasoning game → AI hallucination detection** is particularly elegant: the core of the undercover game — "identifying anomalous players through information asymmetry" — maps perfectly to "identifying hallucinating agents through counterfactual testing."
- **Counterfactual editing provides ground truth**: This represents the most fundamental improvement over traditional MAD — a shift from "group statistics" to "verifiable facts."
- **MAD-Vote can be harmful**: Experiments show that MAD-Vote drops from 46.4% to 37.8% on HallusionBench — if multiple agents share the same hallucination, voting amplifies the error. MUG avoids this through counterfactual testing.
- **The finding that one round is optimal**: Counter-intuitively, more debate rounds degrade performance — because the undercover agent's arguments may convince normal agents, particularly on reasoning-type questions.

## Limitations & Future Work
- The quality of counterfactual image generation is inconsistent — failure modes include edits that are too subtle, editing failures, or unnatural outputs.
- The undercover agent is currently selected at random; selection based on initial response uncertainty would be a natural improvement.
- Only image-based counterfactual testing is supported; text-level counterfactuals remain unexplored.
- Normal agents may be misled when the game extends beyond one round — more robust "immunity" mechanisms are needed.
- Computational complexity scales with the number of agents and rounds.

## Related Work & Insights
- **vs. MAD-Vote/MAD-Judge**: MUG substantially outperforms both on HallusionBench (53.8 vs. 37.8/50.2), demonstrating that counterfactual testing is more effective than voting- or judge-based approaches.
- **vs. Self-Refine**: Self-Refine involves only single-agent self-correction and lacks multi-perspective verification; MUG obtains richer viewpoints through multi-agent gaming.
- **vs. iMAD (in this note collection)**: iMAD uses a classifier to determine "when to debate," while MUG improves "how to debate" — the two approaches are complementary and could be combined.
- Insight: The core idea of counterfactual testing — "verifying understanding by introducing controlled differences" — generalizes to scenarios such as code verification and knowledge assessment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of social reasoning games and counterfactual editing is highly innovative and conceptually distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 4 benchmarks with multiple baselines, ablation studies, and game dynamics analysis.
- Writing Quality: ⭐⭐⭐⭐ The game analogy is vivid and the formal definitions are clear.
- Value: ⭐⭐⭐⭐⭐ A fundamental improvement to the MAD paradigm — shifting from statistical consensus to verifiable fact-checking.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](../../ACL2026/causal_inference/dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](../../ICLR2026/causal_inference/on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](../../ICLR2026/causal_inference/agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)

<!-- RELATED:END -->
