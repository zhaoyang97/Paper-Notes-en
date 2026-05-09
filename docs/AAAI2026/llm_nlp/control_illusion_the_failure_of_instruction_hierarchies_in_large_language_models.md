---
title: >-
  [Paper Note] Control Illusion: The Failure of Instruction Hierarchies in Large Language Models
description: >-
  [AAAI 2026][LLM/NLP][Instruction hierarchy] This paper systematically demonstrates that the system/user prompt separation mechanism in current LLMs **fails to establish reliable instruction priority**, and finds that social hierarchy priors acquired during pretraining (authority, expertise, consensus) exert stronger control over model behavior than explicit system/user role markers.
tags:
  - AAAI 2026
  - LLM/NLP
  - Instruction hierarchy
  - system/user separation
  - conflicting instructions
  - constraint preference
  - social hierarchy prior
date: 2026-05-08
content_hash: 7707cf86810ad516
---

# Control Illusion: The Failure of Instruction Hierarchies in Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2502.15851](https://arxiv.org/abs/2502.15851)
**Code**: [https://github.com/yilin-geng/llm-instruction-conflicts](https://github.com/yilin-geng/llm-instruction-conflicts)
**Area**: LLM Safety & Instruction Following
**Keywords**: Instruction hierarchy, system/user separation, conflicting instructions, constraint preference, social hierarchy prior

## TL;DR
This paper systematically demonstrates that the system/user prompt separation mechanism in current LLMs **fails to establish reliable instruction priority**, and finds that social hierarchy priors acquired during pretraining (authority, expertise, consensus) exert stronger control over model behavior than explicit system/user role markers.

## Background & Motivation
**Background**: The system/user separation paradigm is widely adopted in mainstream LLM deployments, where developers set high-priority instructions via system prompts and users interact through user prompts. OpenAI and others explicitly state that system instructions should take precedence over user instructions.

**Limitations of Prior Work**: Although this hierarchical assumption is broadly accepted, it lacks systematic empirical validation. Prompt injection attacks have shown that users can bypass system-level constraints, yet no prior work has quantitatively assessed how well models enforce instruction hierarchies under simple, verifiable formatting conflicts.

**Key Challenge**: Models perform well when following instructions in isolation (74–91%), but under conflicting instructions, system/user separation entirely fails to establish reliable priority — the average **primary constraint compliance rate is only 9.6–45.8%**.

**Goal**: To systematically evaluate LLMs' ability to enforce instruction hierarchies, quantify failure modes, and identify what factors genuinely influence model priority decisions.

**Key Insight**: The authors design mutually exclusive, programmatically verifiable constraint pairs (uppercase/lowercase, English/French, long/short, etc.) and test priority behavior across 6 mainstream LLMs under varied configurations.

**Core Idea**: The system/user role separation in LLMs constitutes a "control illusion" — social hierarchy priors implicitly acquired during pretraining more effectively govern model behavior than role markers introduced via post-training.

## Method

### Overall Architecture
The authors propose a systematic evaluation framework based on **constraint priority**: (1) 100 base tasks × 6 mutually exclusive constraint pairs × 2 priority assignments = 1,200 test instances; (2) multiple configurations (pure separation, task repetition, explicit priority emphasis); (3) programmatic verification to determine which constraint the model follows; (4) dedicated metrics for behavioral pattern analysis.

### Key Designs
1. **Mutually Exclusive Constraint Pair Design**:

    - Six strictly mutually exclusive, programmatically verifiable constraint types are selected: language (English vs. French), capitalization (all-caps vs. all-lowercase), word count (>300 vs. <50), sentence count (≥10 vs. <5), keyword inclusion (present vs. absent), and keyword frequency (≥5 vs. <2).
    - Design Motivation: The simplest possible formatting constraints are deliberately chosen to eliminate task complexity as a confound and isolate pure priority behavior. If models cannot handle simple formatting conflicts reliably, more complex safety constraints are even less dependable.

2. **Multi-Dimensional Evaluation Metrics**:

    - **Primary Constraint Compliance Rate R1**: Proportion of responses satisfying only the primary (high-priority) constraint.
    - **Secondary Constraint Compliance Rate R2**: Proportion satisfying only the secondary constraint.
    - **Non-compliance Rate R3**: Proportion satisfying neither constraint. R1+R2+R3=1.
    - **Explicit Conflict Acknowledgment Rate (ECAR)**: Proportion of responses where the model explicitly identifies the instruction conflict (extremely low, 0.1%–20.3%).
    - **Priority Adherence Ratio PAR = R1/(R1+R2)**: Among compliant responses, the proportion choosing the primary constraint.
    - **Constraint Bias (CB)**: The model's inherent preference for a given constraint in the absence of explicit priority assignment.
    - Design Motivation: The combination of metrics distinguishes failure modes — whether failures stem from preference bias or from the absence of priority awareness.

3. **Social Hierarchy Prior Experiments**:

    - Three social hierarchy frameworks are tested: organizational authority (CEO vs. intern), professional credibility (Nature paper vs. personal blog), and social consensus (90% of experts vs. minority opinion).
    - All constraints are annotated solely with minimal social framing within a single user message, without employing system/user separation.
    - Design Motivation: To verify whether social hierarchy structures acquired during pretraining are more effective than system/user markers introduced through post-training.

### Loss & Training
This is a **purely evaluative study** with no training involved. Dataset instances are generated in enriched-context variants via few-shot prompting and manually validated for semantic consistency.

## Key Experimental Results

### Main Results

| Model | Isolated Following (IF) | Pure Separation (R1) | Emphasized Separation (R1) | Avg. R1 |
|-------|------------------------|---------------------|--------------------------|---------|
| Qwen-7B | 86.4% | 10.1% | 11.8% | **9.6%** |
| Llama-8B | 80.3% | 6.8% | 10.8% | **10.1%** |
| Llama-70B | 89.9% | 14.2% | 31.7% | **16.4%** |
| Claude3.5-S | 84.2% | 20.3% | 32.6% | **29.9%** |
| GPT4o-mini | 85.4% | 42.7% | 49.4% | **45.8%** |
| GPT4o | 90.8% | 47.0% | 63.8% | **40.8%** |

All models perform well when following instructions in isolation, but primary constraint compliance drops sharply under conflicting instructions. Even the best-performing GPT4o-mini achieves only 45.8%.

### Social Hierarchy vs. System/User

| Model | System/User (PAR) | Authority | Expertise | Consensus |
|-------|-------------------|-----------|-----------|-----------|
| Qwen-7B | 14.4% | 54.0% | 57.3% | **65.8%** |
| Claude3.5-S | 23.6% | 32.4% | 36.8% | **62.0%** |
| GPT4o-mini | 47.5% | 70.0% | 73.2% | **77.8%** |

Social hierarchy frameworks — especially social consensus — yield priority adherence rates **substantially higher** than system/user separation.

### Ablation Study
- Model scale does not guarantee better performance: Llama-70B only marginally outperforms the 8B variant, and GPT4o even underperforms GPT4o-mini.
- Enriched-context variants exhibit similar performance to simple variants, with consistent failure patterns.
- Constraint bias analysis: All models prefer lowercase, more sentences, and keyword avoidance — likely reflecting statistical properties of pretraining data.

### Key Findings
- Models rarely proactively identify instruction conflicts (ECAR as low as 0.1%), and even when conflicts are identified, correct handling is not guaranteed.
- Models exhibit better priority control over categorical constraints (language, capitalization) than over constraints requiring continuous counting (word count, sentence count).
- Social consensus is the strongest implicit priority signal.

## Highlights & Insights
- **Elegant controlled experimental design**: The use of minimal formatting constraints isolates priority behavior and eliminates task complexity as a confound.
- **Social hierarchy prior finding is highly illuminating**: It suggests that LLM behavior is more strongly governed by social structures present in pretraining corpora than by post-training role markers.
- **Important warning for safety alignment**: If even simple formatting constraint priorities cannot be enforced reliably, complex safety rules are far harder to guarantee.
- **"Control illusion" is an apt conceptual framing**: System prompts create a perception of control, yet actual control is far weaker than assumed.

## Limitations & Future Work
- Only single-turn dialogues and simple formatting constraints are evaluated; more complex safety constraints and multi-turn scenarios are not addressed.
- The underlying mechanisms of failure (e.g., attention patterns, internal representations) are not investigated.
- Social hierarchy experiments use minimal framing; the effect of more subtle real-world social signals remains unknown.
- No concrete solutions for improving instruction hierarchies are proposed; the work solely exposes the problem.

## Related Work & Insights
- Wallace et al. (2024)'s instruction hierarchy training yields the best performance in the GPT4o series, indicating that dedicated training helps but is far from resolving the issue.
- Prompt injection literature (Wu et al., Toyer et al.) reveals similar vulnerabilities; this paper provides complementary quantitative evidence from an evaluation perspective.
- There are direct implications for LLM agent research: the reliability of configuring agent behavior via system prompts has been systematically overestimated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of instruction hierarchy failures; social hierarchy prior finding is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 models, 6 constraint types, multiple configurations, simple/enriched contexts, social hierarchy experiments
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, well-designed metrics, highly informative visualizations
- Value: ⭐⭐⭐⭐⭐ Direct and significant implications for LLM safety and deployment; findings are broadly generalizable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](../../ACL2026/llm_nlp/muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[AAAI 2026\] LoKI: Low-damage Knowledge Implanting of Large Language Models](loki_low-damage_knowledge_implanting_of_large_language_models.md)
- [\[AAAI 2026\] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models](coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)

</div>

<!-- RELATED:END -->
