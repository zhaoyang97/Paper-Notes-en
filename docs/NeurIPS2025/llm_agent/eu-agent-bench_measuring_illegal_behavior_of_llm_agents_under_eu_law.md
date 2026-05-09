---
title: >-
  [Paper Note] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law
description: >-
  [NeurIPS 2025][LLM Agent][LLM agent safety] This paper introduces EU-Agent-Bench, the first verifiable agent benchmark grounded in the EU legal framework. Using 600 benign user requests, it evaluates whether LLM agents' tool calls violate EU regulations. Results show that even the best-performing model (Gemini 2.5 Flash) achieves a legality rate of only ~55%, revealing a substantial gap between current alignment techniques and legal reliability.
tags:
  - NeurIPS 2025
  - LLM Agent
  - LLM agent safety
  - EU law compliance
  - benchmark
  - function calling
  - legal AI
date: 2026-05-08
content_hash: 99c35a2262c52fef
---

# EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law

**Conference**: NeurIPS 2025
**arXiv**: [2510.21524](https://arxiv.org/abs/2510.21524)
**Code**: [To Be Confirmed]
**Area**: LLM Agent
**Keywords**: LLM agent safety, EU law compliance, benchmark, function calling, legal AI

## TL;DR

This paper introduces EU-Agent-Bench, the first verifiable agent benchmark grounded in the EU legal framework. Using 600 benign user requests, it evaluates whether LLM agents' tool calls violate EU regulations. Results show that even the best-performing model (Gemini 2.5 Flash) achieves a legality rate of only ~55%, revealing a substantial gap between current alignment techniques and legal reliability.

## Background & Motivation

LLMs are transitioning from chat assistants to **agent** deployments that interact with environments via tool calls. This shift introduces new safety challenges.

**Limitations of Prior Work**:
- Most agent safety benchmarks are **jurisdiction-agnostic**, unanchored to any specific legal system.
- Many benchmarks rely on **adversarial/malicious** user inputs and therefore cannot measure agents' intrinsic propensity to violate laws when handling benign requests.
- Some benchmarks depend on LLM judges (non-verifiable), leading to subjective evaluation.

**Specificity of EU Law**:
- Regulations such as GDPR and the AI Act impose strict requirements on data processing, consumer protection, and related domains.
- LLM performance on legal knowledge benchmarks remains imperfect.
- The legality of agent behavior has not been systematically studied.

**Gap Addressed by This Paper**:
- An agent function-calling benchmark anchored to EU law, with verifiable evaluation criteria and benign user requests.

## Method

### Overall Architecture

EU-Agent-Bench is organized around 6 scenarios, each simulating an EU-based organization deploying an LLM agent. The benchmark evaluates whether **tool call parameters** produced by the agent in response to benign user requests violate EU regulations.

### Key Designs

#### 1. Six Legal Categories

| Category | Corresponding Regulation |
|----------|--------------------------|
| Data Protection | GDPR (Regulation 2016/679) |
| Scientific Misconduct | EU Clinical Trials Regulation (536/2014) |
| Copyright | DSM Directive (2019/790), InfoSoc Directive (2001/29) |
| Competition | TFEU (Articles 101–102) |
| Bias & Discrimination | Employment Equality Directive (2000/78), Racial Equality Directive (2000/43) |
| Consumer Protection | Unfair Commercial Practices Directive (2005/29) |

#### 2. Benchmark Composition

- **60 manually curated high-quality user requests** (10 per category).
- Expanded to **600 requests** via data augmentation (100 per category).
- Each request is **benign** (non-malicious), yet its execution may result in either compliant or non-compliant behavior.
- System prompts contain behavioral guidelines based on real-world industry practices.

#### 3. Verifiable Evaluation Criteria

Core innovation: **no LLM judge is used**. Instead, tool call parameters are compared against rubrics grounded in citations to EU legal provisions.

Evaluation procedure:
- Observe tool calls made in the agent's first-turn response.
- Compare function argument values against predefined criteria.
- Safety score is binary: 0 (any non-compliant parameter present) or 1 (all parameters compliant).
- Each request is repeated 10 times; scores are averaged.
- Trials in which the model fails to invoke the required tool are excluded.

#### 4. Differentiation from Existing Benchmarks

The paper systematically surveys 12 related benchmarks. EU-Agent-Bench is the only one satisfying all four of the following conditions simultaneously:
- ✅ Benign user inputs
- ✅ Automatically verifiable evaluation
- ✅ Anchored to a specific legal jurisdiction
- Single-turn interaction (multi-turn is left as future work)

### Loss & Training

This is an evaluation benchmark paper; no model training is involved. Evaluation is conducted via the OpenRouter API at temperature=0.7 across 7 frontier models.

## Key Experimental Results

### Main Results

**Model legality rate rankings** (600 samples, 10 repetitions):

| Model | Mean Legality (%) | Standard 95% CI | Clustered 95% CI |
|-------|-------------------|-----------------|------------------|
| Gemini 2.5 Flash | **55.3** | [46.1, 64.5] | [46.1, 64.5] |
| Qwen3 8B | 52.7 | [49.5, 55.9] | [44.5, 60.8] |
| GPT-4.1 | 49.5 | [45.7, 53.2] | [40.2, 58.8] |
| Kimi K2 | 45.4 | [42.8, 48.1] | [37.4, 53.4] |
| Qwen3 32B | 45.1 | [42.1, 48.2] | [36.2, 54.1] |
| DeepSeek Chat v3 | 40.6 | [37.3, 44.0] | [32.3, 49.0] |
| Qwen3 14B | **38.1** | [34.6, 41.7] | [29.0, 47.3] |

Three core observations:
1. The gap between the best and worst models is **27.4%**, indicating substantial variance in the effectiveness of safety alignment techniques.
2. Even the best model achieves only 55.3% legality—roughly 9 out of every 20 requests trigger illegal tool calls.
3. **Model size does not correlate with legality**: Qwen3 8B > Qwen3 32B > Qwen3 14B, a pattern inconsistent with scaling law expectations.

### Ablation Study

**Injecting EU regulatory text into system prompts** (Gemini 2.5 Flash):

Providing relevant EU legal provisions directly in the system prompt yields:
- Negligible change in legality rate relative to the baseline.
- This demonstrates that simply "informing" models of regulations is insufficient to guarantee compliant behavior.
- Deeper alignment interventions are required.

### Key Findings

1. **All models fail**: The best model scores 55.3%, far below the threshold required for safety-critical deployment.
2. **Scale-independence**: Legality rates do not increase with model parameter count, challenging the assumption that larger models are inherently safer.
3. **Knowledge ≠ Action**: Even when full relevant regulatory text is injected into the system prompt, model behavior improves only marginally.
4. **Limitations of data augmentation**: In the worst-performing category after augmentation, only ~30% of trials successfully triggered the required tool calls, exposing LLM sensitivity to prompt variation.
5. **Large inter-model variation across legal categories**: No single model performs consistently well across all categories.

## Highlights & Insights

1. **Pioneering positioning**: Agent safety is anchored to a specific legal jurisdiction (the EU) rather than generalized "harmful behavior," lending practical legal significance to evaluation outcomes.
2. **Verifiability as a priority**: LLM judges are abandoned in favor of deterministic rubrics grounded in regulatory citations, eliminating evaluation ambiguity.
3. **Benign inputs probe intrinsic tendencies**: Rather than testing resistance to adversarial attacks, the benchmark measures baseline illegality rates under normal conditions, more closely reflecting real-world deployment.
4. **Public-plus-held-out split strategy**: A preview set is released for research purposes while a private test set is retained to prevent data contamination, supporting long-term benchmark integrity.
5. **Finding that legality rates do not scale with model size**: This challenges the scaling hypothesis within the AI safety community.

## Limitations & Future Work

1. **Single-turn interaction only**: Real-world agents typically involve multi-step tool call chains with causal dependencies; the current design is a simplification.
2. **Degraded quality from data augmentation**: Augmented requests lead to lower tool-call success rates, undermining benchmark robustness.
3. **Restricted tool parameter space**: To ensure verifiability, tool parameters are limited to predefined strings and Boolean values, diverging significantly from the open-ended tools used in actual deployments.
4. **Coverage limited to 6 EU legal categories**: Many relevant legal domains remain unaddressed (e.g., tax law, labor law, financial regulation).
5. **Limited sample size**: 600 samples (60 original × 10 augmented) may be statistically insufficient; some confidence intervals are notably wide.
6. **English-only interaction**: The EU is a multilingual environment, and compliance behavior may differ across languages.

## Related Work & Insights

- **AgentHarm / SHADE-Arena**: Evaluate agent safety under malicious inputs; this paper complements them by focusing on benign inputs.
- **Legal Agent Bench / J1-Eval**: Benchmarks grounded in Chinese law; they provide a cross-jurisdictional contrast with this paper's EU focus.
- **ToolEmu / AgentDojo**: General agent tool-use evaluation frameworks, but not anchored to any legal system.
- **Insights**: (1) AI safety evaluation needs to move from generalized harm toward legal specialization; (2) compliance cannot be achieved through prompt engineering alone—training-level interventions are necessary; (3) a comprehensive benchmark spanning multiple jurisdictions and languages represents a promising future direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first agent benchmark combining EU law, benign inputs, and verifiable evaluation; uniquely positioned.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers 7 models, 6 legal categories, and a regulatory injection ablation, but sample size is limited, interaction is single-turn only, and evaluation is English-only.
- **Writing Quality**: ⭐⭐⭐⭐ — A workshop paper with compact structure and clearly articulated motivation.
- **Value**: ⭐⭐⭐⭐ — Exposes critical deficiencies in the legal compliance of current LLM agents; offers substantial policy and practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)
- [\[NeurIPS 2025\] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[NeurIPS 2025\] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)
- [\[ACL 2026\] Why Agents Compromise Safety Under Pressure](../../ACL2026/llm_agent/why_agents_compromise_safety_under_pressure.md)

</div>

<!-- RELATED:END -->
