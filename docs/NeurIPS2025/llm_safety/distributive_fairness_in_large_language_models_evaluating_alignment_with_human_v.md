---
title: >-
  [Paper Note] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values
description: >-
  [NeurIPS 2025][LLM Safety][distributive fairness] This paper systematically evaluates the distributive fairness preferences of several SOTA LLMs (GPT-4o, Claude-3.5S, Llama3-70b, Gemini-1.5P) on non-strategic resource allocation tasks. The results reveal significant divergence between LLMs and humans: LLMs favor efficiency and envy-freeness (EF) while neglecting equality (EQ), which humans prioritize. However, in multiple-choice settings, GPT-4o and Claude can correctly identify the fairest allocation.
tags:
  - NeurIPS 2025
  - LLM Safety
  - distributive fairness
  - LLM alignment
  - human values
  - fair allocation
  - resource distribution
date: 2026-05-08
content_hash: 51b8c1292d604df3
---

# Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values

**Conference**: NeurIPS 2025
**arXiv**: [2502.00313](https://arxiv.org/abs/2502.00313)
**Code**: [github.com/SamarthKhanna/Distributive-Fairness-LLMs](https://github.com/SamarthKhanna/Distributive-Fairness-LLMs)
**Area**: AI Safety / LLM Alignment
**Keywords**: distributive fairness, LLM alignment, human values, fair allocation, resource distribution

## TL;DR
This paper systematically evaluates the distributive fairness preferences of several SOTA LLMs (GPT-4o, Claude-3.5S, Llama3-70b, Gemini-1.5P) on non-strategic resource allocation tasks. The results reveal significant divergence between LLMs and humans: LLMs favor efficiency and envy-freeness (EF) while neglecting equality (EQ), which humans prioritize. However, in multiple-choice settings, GPT-4o and Claude can correctly identify the fairest allocation.

## Background & Motivation

**Background**: LLMs are increasingly deployed in social and economic decision-making contexts, and their potential as social planners has attracted considerable attention. Distributive fairness—how to equitably allocate resources among multiple individuals—is a central concern in social science and algorithmic decision-making.

**Limitations of Prior Work**: Existing studies have focused primarily on LLM behavior in game-theoretic settings (e.g., prisoner's dilemma, ultimatum game), leaving fairness in non-strategic resource allocation (i.e., the social planner role) largely unexplored.

**Key Challenge**: Fairness itself lacks a unified definition—equality (EQ), envy-freeness (EF), and Rawlsian maximin (RMM) can conflict with one another. It remains unclear whether LLMs' preference ordering aligns with that of humans.

**Goal**: Do LLMs align with human values in resource allocation? Which fairness axioms govern their behavior? What are the sources of misalignment?

**Key Insight**: The paper draws on the classic human experiment dataset from Herreiner & Puppe (2010), designing allocation instances involving indivisible goods (with and without money) to construct tradeoff scenarios between fairness and efficiency.

**Core Idea**: Humans prioritize equality (EQ), whereas LLMs prioritize economic efficiency (PO/USW) and envy-freeness (EF). However, when LLMs select from predefined options rather than generating allocations freely, GPT-4o and Claude correctly identify the fairest allocation in the majority of cases.

## Method

### Overall Architecture
A series of indivisible-goods allocation instances (2–3 agents, 3–6 items, some with monetary transfers) are constructed. Each instance is presented to both LLMs and humans, who either generate or select the "fairest" allocation. The satisfaction frequencies of various fairness and efficiency concepts are then statistically analyzed.

### Key Designs

1. **Instance Design and Dataset**:

   - Function: Ten carefully designed instances $I_1$–$I_{10}}$ are employed, each constructed to induce a specific tradeoff between fairness concepts.
   - Core metric: Agent $i$'s valuation for item $g$ is $v_{i,g}$; the utility function is additively separable: $u_i(A_i, p_i) = v_i(A_i) + p_i$.
   - Scenarios covered: EQ vs. EF, fairness vs. efficiency, monetary transfers to mitigate inequality, and decision-maker bias.

2. **Fairness Metric Framework**:

   - Equality (EQ): Minimizes the inequality gap $\Delta(A,p) = \max_{i,j}\{u_i - u_j\}$; perfect equality EQ* denotes $\Delta = 0$.
   - Envy-Freeness (EF): For all $i, j$, $u_i(A_i, p_i) \geq u_i(A_j, p_j)$.
   - Rawlsian Maximin (RMM): $\max_{(A,p)} \min_i u_i(A_i, p_i)$.
   - Efficiency: Pareto Optimality (PO); utilitarian social welfare maximization (USW): $\max \sum_i u_i$.

3. **Multiple-Choice Experiment (Section 4.1)**:

   - Function: Instead of generating an allocation, LLMs select the fairest option from five predefined candidates.
   - Core finding: GPT-4o and Claude-3.5S select the EQ* allocation in >60% and >70% of cases, respectively—indicating that LLMs understand fairness but cannot realize it in generation mode.
   - Design motivation: To distinguish between two sources of misalignment: insufficient computational ability and misaligned values.

4. **Persona / CoT / Intent Experiments (Section 5)**:

   - Function: LLMs are assigned a persona corresponding to a specific fairness concept, or prompted with Chain-of-Thought reasoning.
   - Core finding: Even when assigned an EQ persona, LLMs still struggle to generate fair allocations (GPT-4o's EQ satisfaction rate remains <20% under an EQ persona), suggesting the bottleneck is computational rather than conceptual.
   - CoT prompting is effective for GPT-4o and Claude on some instances but yields inconsistent results overall.

### Evaluation Protocol
- Each model is queried 100 times per instance at temperature 1.0.
- Fisher's exact test is used to verify significant differences between human and LLM distributions ($p < 0.05$).
- A two-stage prompting strategy is applied to eliminate template sensitivity.

## Key Experimental Results

### Main Results: Aggregated Allocation Preference Rankings (Average Across All Instances)

| Rank | Human | GPT-4o | Claude-3.5S | Llama3-70b | Gemini-1.5P |
|------|-------|--------|-------------|------------|-------------|
| 1st | EQ* (12.4%) | PO (20.4%) | PO (14.9%) | USW (30.8%) | EF (19%) |
| 2nd | EF (9.9%) | USW (11.2%) | EF+PO (14.8%) | PO (26%) | PO (16.8%) |
| 3rd | EF+RMM+PO (9%) | EF+RMM+PO (9.9%) | EF (12.9%) | EF+RMM (7.2%) | USW (11.6%) |

### Fairness Preferences in Multiple-Choice Mode

| Model | Proportion selecting EQ* | Proportion selecting USW |
|-------|--------------------------|--------------------------|
| GPT-4o | >60% | <15% |
| Claude-3.5S | >70% | <10% |
| Llama3-70b | <1% | ~40% |
| Gemini-1.5P | <2% | ~50% |

### Key Findings
- **Generation vs. selection contrast**: GPT-4o never returns EQ* in generation mode, yet selects EQ* in >60% of multiple-choice trials, suggesting that LLMs possess an understanding of fairness but lack the generative capacity to realize it.
- **Disparity in monetary transfer utilization**: GPT-4o can leverage monetary transfers to mitigate inequality (returning EQ* allocations in 8% of cases), whereas other models almost never use money to achieve fairness.
- **Greedy algorithm behavior**: Analysis reveals that LLMs tend to allocate items in a round-robin fashion or assign each item to the agent with the highest valuation—greedy strategies that naturally yield EF or USW outcomes.
- **Self-interest bias**: LLMs behave inconsistently when acting as participants—sometimes exhibiting self-interest, sometimes self-sacrifice.

## Highlights & Insights
- The generation/selection disparity is a particularly insightful finding: LLMs are not ignorant of fairness but lack the search capacity to explore fair allocations in open-ended generation. This points toward concrete directions for improvement via RL or SFT.
- The fairness concept hierarchy analysis provides a fine-grained framework for LLM alignment, offering more value than a binary aligned/misaligned verdict.
- The experimental design is tightly integrated with economic empirical methodology; each instance carefully constructs a specific fairness-efficiency tradeoff.

## Limitations & Future Work
- Human data originate from a single study (H&P 2010), which may be subject to cultural and contextual dependencies; cross-cultural validation is absent.
- Only additively separable valuations over indivisible goods are considered; combinatorial valuations and strategic environments are not addressed.
- No attempt is made to directly improve LLMs' fair allocation generation through SFT or RLHF.
- The inequality tolerance experiments involve a limited number of constructed amplified instances, without systematic exploration of extreme scenarios.

## Related Work & Insights
- **vs. Fish et al. (2025, EconEvals)**: That work evaluates LLM performance on efficiency-fairness tradeoffs but uses homogeneous monetary resources; this paper uses indivisible goods with heterogeneous valuations, more closely reflecting real-world settings.
- **vs. Horton (2023)**: That study uses personas to influence LLM behavior in dictator games; this paper extends the analysis to more complex multi-agent resource allocation.
- **vs. Scherrer et al. (2024, MoralChoice)**: That work evaluates LLMs' moral judgments; this paper focuses on the more operationally concrete dimension of distributive fairness.
- The finding that LLMs tend to employ greedy algorithms is noteworthy and may reflect a bias in pretraining data toward algorithmic descriptions of such strategies.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic evaluation of LLMs' distributive fairness preferences; the generation/selection disparity is a novel finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 4 models, 10+ instances, and multiple evaluation modes (generation, selection, persona, CoT).
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with strong integration into the economics literature.
- Value: ⭐⭐⭐⭐ — Provides an important empirical foundation and actionable directions for LLM fairness alignment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models](exploring_the_limits_of_strong_membership_inference_attacks_on_large_language_mo.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[AAAI 2026\] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak](../../AAAI2026/llm_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)
- [\[NeurIPS 2025\] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions](evaluating_the_promise_and_pitfalls_of_llms_in_hiring_decisions.md)
- [\[NeurIPS 2025\] Reverse Engineering Human Preferences with Reinforcement Learning](reverse_engineering_human_preferences_with_reinforcement_learning.md)

<!-- RELATED:END -->
