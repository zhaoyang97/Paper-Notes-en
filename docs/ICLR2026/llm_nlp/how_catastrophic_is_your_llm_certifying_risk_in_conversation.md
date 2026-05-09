---
title: >-
  [Paper Note] How Catastrophic is Your LLM? Certifying Risk in Conversation
description: >-
  [ICLR 2026][LLM/NLP][safety certification] This paper proposes C3LLM (Certification of Catastrophic risks in multi-turn Conversation for LLMs), the first framework to provide statistical certification of catastrophic risks in multi-turn LLM conversations. It models conversation distributions as Markov processes over a semantic similarity graph, defines three conversation sampling strategies augmented with a jailbreak layer, and applies Clopper-Pearson 95% confidence intervals to certify the probability that a model produces harmful outputs—finding that the worst-performing model has a risk lower bound as high as 72%.
tags:
  - ICLR 2026
  - LLM/NLP
  - safety certification
  - multi-turn attack
  - Markov process
  - catastrophic risk
  - statistical guarantee
date: 2026-05-08
content_hash: 5b11e30833842910
---

# How Catastrophic is Your LLM? Certifying Risk in Conversation

**Conference**: ICLR 2026
**arXiv**: [2510.03969](https://arxiv.org/abs/2510.03969)
**Code**: None
**Area**: LLM/NLP
**Keywords**: safety certification, multi-turn attack, Markov process, catastrophic risk, statistical guarantee

## TL;DR

This paper proposes C3LLM (Certification of Catastrophic risks in multi-turn Conversation for LLMs), the first framework to provide statistical certification of catastrophic risks in multi-turn LLM conversations. It models conversation distributions as Markov processes over a semantic similarity graph, defines three conversation sampling strategies augmented with a jailbreak layer, and applies Clopper-Pearson 95% confidence intervals to certify the probability that a model produces harmful outputs—finding that the worst-performing model has a risk lower bound as high as 72%.

## Background & Motivation

**Background**: LLMs may produce catastrophic outputs in conversation (e.g., bomb-making instructions, bioweapon synthesis, cyberattack tutorials). Multi-turn attacks are harder to defend against than single-turn ones—adversaries can gradually steer models toward harmful content through seemingly benign conversation sequences.

**Two Fundamental Flaws of Fixed Benchmarks**:
- **Reliance on fixed attack sequences**: Only specific attacks are tested, missing unseen successful sequences—20 attack sequences of length 5 cover at most 20 attack variants, yet the combinatorial space is $100^5 = 10^{10}$
- **Lack of statistical guarantees**: Conclusions are non-generalizable; the extent of risk across the full conversation space remains unknown

**Key Challenge**: Exhaustive testing is infeasible (exponential space), and different sequences carry different levels of danger—risk must be quantified in terms of probability distributions.

**Why Statistical Certification over Benchmarking**: Benchmarking provides sample lower bounds ("N successful attacks found"), whereas statistical certification provides probability bounds ("a randomly sampled conversation has a [40%, 60%] probability of triggering catastrophic output")—the latter is far more meaningful.

**Core Idea**: Model multi-turn conversations as a Markov process on a graph; sample → judge → statistically test; output a confidence interval for catastrophic risk.

## Method

### Overall Architecture

1. **Query Graph Construction**: Extract attack scenarios from HarmBench → expand a set of related but milder queries $Q$ around each harmful target $q^*$ → construct graph $G=(V, E)$ with edges based on semantic similarity
2. **Conversation Distribution Definition**: Define three Markov processes on the graph → sample conversation sequences $\gamma = (v_0, v_1, ..., v_{n-1})$
3. **Catastrophic Judgment**: Feed each sequence to the LLM; a judge model (GPT-4o) evaluates whether each response is catastrophic
4. **Statistical Certification**: 50 independent samples → Clopper-Pearson 95% confidence interval → output upper and lower bounds on catastrophic probability

### Key Designs

1. **Lifted State Space to Avoid Revisiting Nodes**:

   - State $(v, S)$: current query $v$ + visited set $S$
   - Terminal state $\tau$: entered when no transferable neighbors exist
   - Ensures no repeated queries within a sequence (reflecting realistic adversary behavior)

2. **Three Conversation Distributions**:

   | Distribution | Construction | Adversary Model | Characteristics |
   |---|---|---|---|
   | Random Node (RN) | Independently sample random nodes | Non-strategic random attacker | Estimates overall model vulnerability |
   | Graph Path (GP) | Graph path with endpoint constrained to target set $Q_T$ | Directional conversation flow | Coherent semantic context |
   | Adaptive w/ Rejection (AR) | Adjusts path using model accept/reject feedback | Adaptive red-teaming attack | Accept → advance toward target; reject → retreat |

3. **Weight Design for Adaptive Distribution**:

   - Define progressive neighbors $A_{\text{prog}}$ (closer to $q^*$) and deprogressive neighbors $A_{\text{deprog}}$ (farther from $q^*$)
   - Model accepts current query ($r_v=0$): high weight $\lambda_h$ assigned to $A_{\text{prog}}$, encouraging advancement
   - Model rejects current query ($r_v=1$): high weight $\lambda_h$ assigned to $A_{\text{deprog}}$, retreating to a safer region to retry

4. **Jailbreak Augmentation Layer**:

   - Each query is augmented with a jailbreak prefix with probability $p$
   - Augmented sequence probability: $\Pr(\tilde\gamma) = \Pr(\gamma) \prod_t \Pr_{\mathcal{D}_{jb}}(\tilde{v}_t | v_t)$
   - Covers a spectrum from identity transformation (no modification) to structured modifications

### Graph Construction Details

- Data source: HarmBench chemical_biological (28 scenarios) + cybercrime (40 scenarios) = 68 scenarios
- For each scenario, 3 LLMs generate 30 actors (related characters/concepts), each with 5 queries
- After deduplication, 20 actors are randomly sampled to construct a diverse query set
- Edges are connected via cosine similarity threshold

## Key Experimental Results

### Main Results: Certified Risk for 6 Frontier Models (95% CI Lower Bound)

| Model | Chembio Risk CI | Cybercrime Risk CI | Highest Risk Lower Bound |
|---|---|---|---|
| DeepSeek-R1 | [0.554, 0.821] | [0.721, 0.935] | **72.1%** |
| Mistral-Large | [0.554, 0.821] | [0.652, 0.892] | **65.2%** |
| Llama-3.3-70B | [0.212, 0.488] | [0.374, 0.663] | 37.4% |
| GPT-4o | Moderate | Moderate | ~30% |
| Claude-Sonnet-4 | [0.001, 0.106] | [0.028, 0.205] | 2.8% |
| Nova Premier | [0.005, 0.137] | [0.000, 0.071] | **0.0%** |

### Attack Effectiveness Across Three Distributions

| Distribution | Attack Efficiency | Semantic Coherence | Adaptivity | Applicable Scenario |
|---|---|---|---|---|
| Random Node + JB | Lowest | None | None | Baseline: model vulnerability under random input |
| Graph Path (harmful) | Moderate | High | None | Directional natural conversation attacks |
| Adaptive w/ Rejection | **Highest** | Medium–High | **Yes** | Realistic red-teaming strategy |

### Key Findings

- **DeepSeek-R1 achieves a 72.1% risk lower bound under Cybercrime**—even under the most conservative estimate, >70% of randomly sampled conversations trigger catastrophic output
- **Claude-Sonnet-4 and Nova Premier are significantly safer** (<14% / <7%), though not zero-risk
- **The double-edged sword of rejection signals**: Models with rejection rates of 15–20% provide precise feedback to adaptive attackers—rejections signal "you're too close, step back slightly"
- **Case analysis reveals two attack patterns**: (a) distractors—inserting benign queries before harmful ones to lower model vigilance; (b) context—early turns provide background information making the final harmful query appear more legitimate
- Statistical certification uncovers orders of magnitude more vulnerabilities than fixed benchmarks—20 fixed attacks vs. a probability bound over a $10^{10}$ space

## Highlights & Insights

- **Paradigm Upgrade: from "whether compromised" to "probabilistic confidence bounds"**: Safety evaluation gains statistical rigor for the first time—analogous to moving from "finding one bug" to "system-level reliability certification"
- **Rejection Rate ≠ Safety**: Attackers exploit rejection signals to refine strategies, challenging the intuition that "high rejection rate = safer"—safety should be leak-free
- **Elegant Design of the Adaptive Distribution**: The accept-advance / reject-retreat weight mechanism elegantly models realistic red-team adversary strategies
- **Generality of the Markov Process**: The framework is not limited to three distributions; new distributions can be flexibly defined to explore different attack patterns

## Limitations & Future Work

- **Judge Bias**: GPT-4o is used to judge catastrophic outputs, introducing circular bias when evaluating GPT-family models
- **Limited Scenario Coverage**: Only 68 scenarios (chemical/biological + cybercrime); categories such as violence and hate speech are not covered
- **Risk Quantification Only, No Defense Proposed**: The framework identifies risk but does not provide mitigation strategies
- **Limited Sample Size**: Only 50 samples per distribution, yielding wide confidence intervals (e.g., [0.554, 0.821]); denser sampling could narrow these intervals
- **Graph Construction Depends on Actor Generation Quality**: The diversity and coverage of the query set directly affect certification results

## Related Work & Insights

- **vs. HarmBench / AdvBench**: Fixed attack sets vs. statistical certification; C3LLM provides probabilistic guarantees rather than empirical observations
- **vs. Crescendo / PAIR**: These are multi-turn attack methods, whereas C3LLM is a certification framework—it can certify the coverage of such attack methods
- **vs. Single-turn Certification (Kumar 2023)**: Token/embedding-space perturbation certification vs. multi-turn conversation distribution certification; they differ in complexity and applicability
- **vs. ATAD**: ATAD dynamically generates reasoning evaluation benchmarks; C3LLM statistically certifies safety risks—both transcend the limitations of fixed benchmarks, but their objectives are entirely different

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First statistical certification framework for multi-turn safety; the combination of Markov processes and statistical testing is original
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 frontier models × 3 distributions × 2 categories, with in-depth case analysis
- Writing Quality: ⭐⭐⭐⭐ Formally rigorous with a clear mathematical notation system
- Value: ⭐⭐⭐⭐⭐ Establishes a higher methodological standard for AI safety evaluation, elevating empirical testing to statistical certification

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)
- [\[NeurIPS 2025\] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](../../NeurIPS2025/llm_nlp/streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)
- [\[ICLR 2026\] Generative Value Conflicts Reveal LLM Priorities](generative_value_conflicts_reveal_llm_priorities.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)

<!-- RELATED:END -->
