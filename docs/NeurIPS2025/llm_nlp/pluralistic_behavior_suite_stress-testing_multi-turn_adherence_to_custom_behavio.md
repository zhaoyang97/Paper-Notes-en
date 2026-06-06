---
title: >-
  [Paper Note] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies
description: >-
  [NeurIPS 2025][LLM/NLP][pluralistic alignment] This paper introduces PBSuite, an evaluation suite comprising 300 industry-specific behavioral policies and a dynamic multi-turn adversarial evaluation framework. It reveals…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "pluralistic alignment"
  - "behavioral policy"
  - "multi-turn evaluation"
  - "red-teaming"
  - "LLM safety"
date: 2026-05-08
content_hash: 0e26ad80f84061dc
---

# PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies

**Conference**: NeurIPS 2025
**arXiv**: [2511.05018](https://arxiv.org/abs/2511.05018)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: pluralistic alignment, behavioral policy, multi-turn evaluation, red-teaming, LLM safety

## TL;DR

This paper introduces PBSuite, an evaluation suite comprising 300 industry-specific behavioral policies and a dynamic multi-turn adversarial evaluation framework. It reveals that mainstream LLMs exhibit high compliance under single-turn settings (violation rate <4%), but compliance degrades sharply under multi-turn adversarial interactions (violation rate up to 84%).

## Background & Motivation

Current LLMs are typically aligned to general safety principles (e.g., prohibiting hate speech and violence), yet real-world deployments place LLMs within organizational ecosystems that have unique enterprise policies, regulatory requirements, and ethical commitments. For example:

- An educational chatbot may need to **refuse** helping students revise essays (even though most models are willing to do so by default)
- A client-facing legal assistant must **avoid providing legal advice**, whereas an internal tool for attorneys requires more flexible reasoning

This need for **pluralistic alignment**—adapting models to diverse user values and organizational requirements—currently lacks systematic evaluation methods. Existing safety benchmarks primarily focus on general harmful content (e.g., toxicity, hate speech) and cannot cover enterprise-specific behavioral constraints.

## Method

### Overall Architecture

PBSuite consists of two core components:

1. **Diverse Behavioral Policy Dataset**: 300 LLM behavioral policies grounded in 30 industries
2. **Dynamic Multi-Turn Evaluation Framework**: Stress-tests model adherence to custom policies across multi-turn interactions

### Key Designs

**Hierarchical Policy Generation Pipeline**:

1. **Industry Selection**: 30 industries with high LLM deployment potential selected from 147 industries in the U.S. Bureau of Labor Statistics
2. **Behavioral Risk Dimension Extraction**: 3–5 risk dimensions per industry (e.g., public exposure, autonomy, judicial constraints) with assigned risk levels
3. **Enterprise Use Case Construction**: 10 representative use cases per industry, annotated with risk dimension level values
4. **Behavioral Policy Generation**: Permitted/prohibited behavioral rules generated based on use cases and their risk-level configurations

For example, in the legal services industry:
- **Public-facing court procedure assistant**: low public exposure, low autonomy → strict constraints
- **Internal research assistant for attorneys**: high public exposure, high autonomy → greater flexibility permitted

**Multi-Turn Adversarial Evaluation Framework** (adapted from X-Teaming):

Four agents collaborate:
- **Planner**: Generates high-level attack strategies and turn-by-turn plans
- **Attack Agent**: Executes the plan to generate user queries, gradually escalating from compliant to challenging queries
- **Target Model**: The LLM under evaluation (system prompt contains the behavioral policy)
- **LLM Judge**: Evaluates response compliance based on a rubric (1–5 scale, 5 = clear violation)

Up to 5 strategies are attempted per verifiable prohibited behavior, with at most 7 turns per conversation.

### Evaluation Setup

- **Policy Specification**: Behavioral policies are injected via system prompt
- **Evaluation Scope**: 1,100 verifiable prohibited rules filtered from 300 policies
- **Three Settings**: Single-turn (direct violation query), simple multi-turn (2–4 compliant turns + final violation turn), agentic multi-turn (adaptive adversarial framework)
- **Evaluated Models**: llama-3.1-8b-instruct, llama-3.3-70b-instruct, gpt-4o, gpt-4o-mini, qwen3-8b, qwen3-32b

## Key Experimental Results

### Main Results

**Attack Success Rate (ASR) — Single-Turn vs. Multi-Turn**:

| Model | Single-Turn | Simple Multi-Turn | Agentic Multi-Turn |
|-------|-------------|-------------------|--------------------|
| gpt-4o | 0.2% | 0.1% | 25.1% |
| gpt-4o-mini | 0.3% | 0.3% | 37.6% |
| llama-3.3-70b | 1.8% | 1.7% | 38.0% |
| llama-3.1-8b | 3.9% | 7.9% | 76.8% |
| qwen3-32b | 1.0% | 0.3% | 74.4% |
| qwen3-8b | 1.8% | 1.7% | **84.4%** |

Core finding: All models perform well under single-turn settings (violation rate <4%), but violation rates surge to 25%–84% under agentic multi-turn adversarial conditions.

**Single-Turn Violation Rate — With vs. Without Policy**:

| Model | No Policy (Strict) | With Policy (Strict) | No Policy (% Unsafe) |
|-------|--------------------|----------------------|----------------------|
| gpt-4o | 9.6% | 0.2% | 0.2% |
| gpt-4o-mini | 11.3% | 0.3% | 0.2% |
| llama-3.3-70b | 11.3% | 1.8% | 0.1% |
| qwen3-32b | 19.0% | 1.0% | 0.2% |
| qwen3-8b | 17.7% | 1.8% | 0.2% |

Nearly all violating queries are judged as "safe" by conventional content moderation models (% Unsafe ≈ 0), confirming that behavioral policy violations are orthogonal to traditional safety risks.

### Ablation Study

**Attack Strategy Analysis**:

- **Role-play strategies** are the most effective (highest ASR), bypassing alignment by constructing plausible enterprise interaction scenarios
- **Narrative manipulation and document simulation requests** are also highly effective attack vectors
- Strategy diversity analysis shows that most successful strategies can be classified as some form of role-play

### Key Findings

1. **Behavioral policy compliance is orthogonal to traditional safety**: Conventional safety moderation models nearly completely fail to detect enterprise-specific behavioral violations
2. **Model size positively correlates with compliance but is insufficient**: Despite weaker out-of-box safety, Qwen3 models show the greatest improvement when provided with a policy, suggesting reasoning capacity facilitates adherence to structured constraints
3. **Multi-turn settings significantly amplify vulnerability**: Current alignment primarily optimizes for single-turn settings; compliance degrades sharply in multi-turn interactions
4. **Instruction hierarchy helps but is insufficient**: OpenAI models trained with instruction hierarchy perform best, yet still exhibit a 25% violation rate
5. **Over-helpfulness tendency is a root cause**: Models aligned to "be as helpful as possible" tend to over-accommodate user requests across multi-turn interactions

## Highlights & Insights

1. **Exposes an important blind spot in alignment research**: Current work overemphasizes general safety (toxicity, violence) while neglecting the enforcement of fine-grained behavioral constraints in enterprise deployments
2. **The hierarchical policy generation pipeline** is highly scalable and practically useful
3. **Paradigm shift from "safety alignment" to "pluralistic alignment"**: The paper clearly demonstrates why general alignment cannot substitute for domain-specific behavioral constraints
4. **Quantifies the fragility of multi-turn interactions**: Provides the first cross-model, cross-industry benchmark for multi-turn behavioral policy compliance

## Limitations & Future Work

1. **Adversarial setting deviates from real-world usage**: Testing relies primarily on adversarial conversations and does not cover natural violations occurring within normal enterprise dialogue flows
2. **Lack of over-refusal evaluation**: Models may over-refuse compliant requests, but this dimension is not assessed
3. **Reliability of the LLM Judge**: Using gpt-4.1 as the judge introduces noise and bias; Cohen's κ with human annotations is 0.51 (moderate agreement)
4. **Policies injected solely via system prompt**: This is susceptible to prompt injection attacks in multi-turn interactions; stronger architecture-level safeguards are needed
5. **Rule verifiability filtering is imperfect**: Some rules may implicitly rely on external knowledge

## Related Work & Insights

- Complementary to CoSA (content safety classification), PBSuite focuses on broader behavioral policy adherence
- Extends the X-Teaming framework from default safety alignment red-teaming to custom behavioral policy evaluation
- Suggested direction: architectural solutions (e.g., instruction hierarchy) are needed rather than relying solely on prompt engineering

## Rating

- Novelty: ⭐⭐⭐⭐ First benchmark to systematically evaluate multi-turn behavioral policy adherence in pluralistic alignment; the problem is clearly defined and practically significant
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 mainstream models, 30 industries, and 1,100 rules, though judge reliability warrants improvement
- Writing Quality: ⭐⭐⭐⭐ Well-structured with thorough experimental analysis and honest discussion of limitations
- Value: ⭐⭐⭐⭐⭐ Reveals a critically important and underexplored problem in enterprise LLM deployment, with significant implications for both safety alignment research and industry practice

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](scaling_up_active_testing_to_large_language_models.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/llm_nlp/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[ICML 2026\] T$^2$PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic Reinforcement Learning](../../ICML2026/llm_nlp/t2po_uncertainty-guided_exploration_control_for_stable_multi-turn_agentic_reinfo.md)

</div>

<!-- RELATED:END -->
