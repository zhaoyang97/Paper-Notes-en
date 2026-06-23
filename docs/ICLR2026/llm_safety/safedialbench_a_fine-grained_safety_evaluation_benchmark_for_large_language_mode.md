---
title: >-
  [Paper Note] SafeDialBench：面向大模型多轮对话与多样越狱攻击的细粒度安全评估基准
description: >-
  [ICLR 2026][LLM Safety][Paper Note] This paper introduces SafeDialBench—a safety evaluation benchmark covering 6 safety dimensions, 7 jailbreak attacks, 22 dialogue scenarios, and 4,053 bilingual (Chinese-English) multi-turn dialogues. It is accompanied by a fine-grained evaluation framework that decomposes "safety" into three capabilities: risk identifi
tags:
  - ICLR 2026
  - LLM Safety
date: 2026-05-08
content_hash: 0cd89494326e5369
---
# SafeDialBench: A Fine-grained Safety Evaluation Benchmark for LLMs in Multi-turn Dialogues and Diverse Jailbreak Attacks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KFjtRqVnKH](https://openreview.net/forum?id=KFjtRqVnKH)  
**Project Page**: https://safedialbench.github.io/  
**Area**: LLM Safety / Safety Evaluation Benchmark  
**Keywords**: Multi-turn Dialogue Safety, Jailbreak Attack, Safety Taxonomy, Fine-grained Evaluation, Bilingual Benchmark

## TL;DR
This paper introduces SafeDialBench—a safety evaluation benchmark covering 6 safety dimensions, 7 jailbreak attacks, 22 dialogue scenarios, and 4,053 bilingual (Chinese-English) multi-turn dialogues. It is accompanied by a fine-grained evaluation framework that decomposes "safety" into three capabilities: risk identification, handling unsafe information, and maintaining consistency. This approach allows for a more precise characterization of the safety weaknesses in 19 LLMs compared to traditional "single-turn + single-attack" benchmarks.

## Background & Motivation
**Background**: As LLMs are widely deployed in dialogue systems, safety has become a central concern for reliability and trustworthiness. Most existing safety benchmarks (COLD, BeaverTails, SALAD-Bench, SafetyBench, etc.) evaluate models in **single-turn dialogues**. While large in scale, they fail to reflect real-world human-computer interactions.

**Limitations of Prior Work**: The few existing benchmarks focused on multi-turn dialogue safety (CoSafe, RED QUEEN, SC-Safety, etc.) suffer from three major deficiencies. First, data construction typically relies on a **single jailbreak strategy**, resulting in a narrow attack surface. Second, the evaluation dimensions are incomplete, often focusing only on "aggressive language/profanity" while ignoring critical aspects like ethics, morality, legality, fairness, and privacy. Third, they generally provide only a coarse "safe/unsafe" judgment, **lacking fine-grained evaluation of a model's ability to "identify" and "handle" unsafe information**. Furthermore, most are monolingual and have dialogues typically shorter than 5 turns.

**Key Challenge**: Real-world jailbreaking is often a multi-turn, gradual process involving various attack tactics. The granularity of existing evaluations (single-turn, single-attack, binary judgment) is far lower than the complexity of these threats, leading to safety scores that are neither comprehensive nor precise.

**Goal**: To build a **comprehensive and fine-grained** multi-turn dialogue safety benchmark that simultaneously achieves: (1) coverage of multiple safety dimensions; (2) inclusion of diverse jailbreak attacks; (3) extended dialogue turns; (4) bilingual support; and (5) decomposition of "safety" into specific capabilities.

**Key Insight**: The authors argue that safety should not be a scalar but should be decomposed into a progressive chain: "Can the model **identify** the risk? -> Can it **properly handle** it after identification? -> Can it remain **consistent** under multi-turn pressure?" Additionally, data construction should leverage collaboration between human experts and multiple strong LLMs to balance quality and diversity.

**Core Idea**: By integrating a "two-layer safety taxonomy, human-AI collaborative data generation with 7 jailbreak types, and a three-capability fine-grained scoring system," the benchmark measures LLM safety under multi-turn jailbreaks both broadly and deeply.

## Method

### Overall Architecture
The input to SafeDialBench consists of manually designed safety dimensions and attack strategies; the output is the fine-grained safety scores for 19 tested LLMs across 6 dimensions × 3 capabilities. The pipeline consists of three sequential stages: **Setting standards** (establishing a two-layer safety taxonomy for 6 categories), **Constructing data** (human annotators acting as users against 3 strong LLMs acting as assistants to generate 4,053 dialogues with 7 jailbreak attacks across 22 scenarios, followed by two rounds of expert review), and **Evaluation** (feeding dialogues to an LLM evaluator to score identification/handling/consistency turn-by-turn, taking the minimum turn score as the dialogue score, supplemented by human validation).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Manual Setup: 6 Safety Dimensions<br/>+ 22 Scenarios + 7 Attack Strategies"] --> B["1. Two-layer Safety Taxonomy<br/>Fairness/Legality/Morality/Aggression/Ethics/Privacy"]
    B --> C["2. Human-AI Collaborative + Jailbreak Data Generation<br/>Annotators as Users × 3 Strong LLMs as Assistants"]
    C -->|Two Rounds of Expert Review| D["4,053 Bilingual Multi-turn Dialogues<br/>3~10 Turns"]
    D --> E["3. Three-capability Fine-grained Evaluation Framework<br/>Identification / Handling / Consistency"]
    E -->|Turn-by-turn Scoring (Take Minimum)| F["Safety Profiles of 19 LLMs"]
```

### Key Designs

**1. Two-layer Safety Taxonomy: Decomposing "Safety" into 6 Independently Auditable Dimensions**

To address the issue that previous benchmarks lack evaluation dimensions and focus excessively on aggressive language, the authors synthesized a two-layer (coarse-grained dimensions → fine-grained safety points) taxonomy. It covers six orthogonal dimensions: **Fairness** (objective treatment of groups, avoiding stereotypes), **Legality** (compliance regarding physical harm, economic crimes, information/public safety), **Morality** (non-violent immoral acts like fraud), **Aggression** (threats, insults, incitement), **Ethics** (destructive behaviors like self-harm), and **Privacy** (protecting sensitive personal/organizational/social information). This allow the evaluation to pinpoint exactly where a model fails—for example, the Qwen2.5 series was found to be notably weak in identifying "Aggression" and "Legality."

**2. Human-AI Collaborative Multi-turn Dialogue Construction with Diverse Jailbreaks**

To overcome the "single attack, single language, short dialogue, and LLM bias" issues of existing benchmarks, the authors used human experts to lead and multiple strong LLMs to assist in turn-based generation. Annotators design **critical first-turn user prompts** based on a (Scenario + Dimension + Strategy) triplet and play the "user." The assistant (the targeted chatbot) is played by one of GPT-4, Doubao, or ChatGLM to avoid benchmark bias toward a single model's quirks. Annotators advance the **jailbreak intent turn-by-turn** based on the responses and preset attack patterns until the dialogue is long enough (3–10 turns) or the jailbreak succeeds/is firmly rejected. Each dialogue undergoes **two rounds of expert review** for logic, naturalness, and attack effectiveness.

Seven jailbreak tactics target different model weaknesses: **Scene Construct** (using protective roles/scenarios to hide malice), **Purpose Reverse** (exploiting defects in negation/reverse reasoning), **Role Play** (inducing unsafe behavior through hypothetical roles), **Topic Change** (gradual transition from harmless to harmful), **Reference Attack** (using neutral descriptions/pronouns to mask intent), **Fallacy Attack** (using pseudo-logic/false premises), and **Probing Question** (systematically escalating sensitive topics). This results in 4,053 bilingual dialogues, achieving a better balance between quality and diversity than purely automated methods.

**3. Three-capability Fine-grained Evaluation Framework: Safety as an Identification → Handling → Consistency Chain**

The authors propose a framework using LLMs as evaluators to score three progressive dimensions: **Identify unsafe risks** (detecting potential risks in multi-turn jailbreaks), **Handle unsafe information** (providing safety-oriented and appropriate responses), and **Consistency** (maintaining a stable safety stance under continuous pressure or misleading logic). Scores range from 1 to 10 for each response turn across the six dimensions, supported by specific rubrics (9–10 / 7–8 / 5–6 / 3–4 / 1–2).

The final score for a dialogue uses a **minimum-score-taking** strategy: $$s_{\text{dialog}} = \min_t s_t$$. This aligns with human intuition: in an interconnected context, **if any single turn is breached, the safety of the entire dialogue is compromised**. Attack Success Rate (ASR) is defined as the proportion of jailbreak prompts that successfully elicit an unsafe response (score < 7).

## Key Experimental Results

### Main Results
19 LLMs (4 closed-source + 15 open-source, including 3 reasoning models) were evaluated using ChatGPT-3.5 turbo as the evaluator (temperature 0) with golden context as the dialogue history.

| Performance Tier | Representative Models | Key Findings |
|------------------|-----------------------|--------------|
| Best Overall Safety | Yi-34B-Chat, MoonShot-v1, ChatGPT-4o | Leads across all three capabilities; ChatGPT-4o has the lowest ASR. |
| Open-source Benchmark | GLM4-9B-Chat | Strong in Ethics; robust in handling Legality. |
| Weak Safety | Llama3.1-8B-Instruct, o3-mini | o3-mini is the weakest among reasoning models in Aggression/Legality/Morality. |
| Most Vulnerable | Baichuan2-7B-Chat | Highest ASR at **69.60%**. |
| Reasoning Models | DeepSeek-R1 (best), o3-mini (worst) | Reasoning capability ≠ Safety capability. |

Excerpt from Table 2 (Ide/Han/Con correspond to Identification/Handling/Consistency):

| Model | Aggression (Ide/Han/Con) | Ethics (Ide/Han/Con) | Legality (Ide/Han/Con) |
|-------|--------------------------|----------------------|------------------------|
| Yi-34B-Chat | 6.93 / 7.87 / 6.98 | 7.41 / 8.06 / 7.57 | 8.33 / 8.05 / 7.97 |
| GLM4-9B-Chat | 6.84 / 7.81 / 6.86 | 7.50 / 8.08 / 7.68 | 8.29 / 8.12 / 7.90 |
| ChatGPT-4o | 6.81 / 7.51 / 7.30 | 7.19 / 7.92 / 7.35 | 6.92 / 7.55 / 7.16 |
| o3-mini | 6.66 / 7.28 / 7.12 | 7.14 / 7.79 / 7.28 | 6.96 / 7.49 / 7.13 |
| Qwen2.5-14B-I | 6.75 / 7.42 / 7.20 | 7.11 / 7.78 / 7.28 | 6.89 / 7.48 / 7.14 |

### Attack Effectiveness and Multi-turn Analysis

| Analysis Perspective | Key Findings |
|----------------------|--------------|
| Most Effective Attacks | **Fallacy Attack, Purpose Reverse, and Role Play** are most likely to breach models. Topic Change and Reference Attack are weaker. |
| Model Robustness | GLM4-9B-Chat and Yi-34B-Chat are robust against all attacks; ChatGPT-4o is strong against Topic Change but weak against Fallacy Attack. |
| Turn-based Attrition | Safety scores fluctuate in the first 3 turns and **drop significantly after turn 4**, especially in Ethics and Aggression dimensions. |
| Model Scale | Safety **does not improve monotonically with scale**: Baichuan2-13B is stronger in Privacy, but Baichuan2-7B is better in Morality. |

### Key Findings
- **Minimum-score-taking** is the core of the framework: it reflects the threat model where one failure turn ruins the whole dialogue, explaining why safety scores drop after turn 4.
- **Evaluator Credibility**: GPT-3.5 turbo scores show **over 80% agreement** with human experts, validating the automated framework.
- **Value of Fine-grained Analysis**: Decomposing 6 dims × 3 capabilities allows for precise diagnosis (e.g., Qwen2.5's identification weakness vs. DeepSeek-7B's consistency weakness).

## Highlights & Insights
- **Decomposing "Safety" into a Capability Chain**: The Identify → Handle → Consistency chain is highly explanatory, helping to distinguish between "not seeing the risk" vs. "failing to handle it."
- **Minimum-score-taking Metric**: This design encodes the "weakest link" intuition of multi-turn safety into the scoring system, preventing average scores from masking isolated failures.
- **Human-AI Collaborative Construction**: The paradigm of humans driving the jailbreak against multiple assistant LLMs balances intent quality with assistant diversity.
- **Counter-intuitive Conclusion**: Reasoning models (like o3-mini) are not inherently safer, suggesting safety requires specialized alignment rather than emerging naturally from reasoning capabilities.

## Limitations & Future Work
- **Evaluator Dependency**: Reliance on LLMs as judges might introduce evaluator-specific biases or blind spots.
- **Golden Context Usage**: Using pre-constructed "golden" dialogue history ensures consistency during testing but may differ from real-world distributions where models generate their own context.
- **Cross-category Comparison**: Scores across different attack types or languages should be compared with caution due to varying difficulties.
- **Benchmark Staticity**: Jailbreak techniques evolve rapidly; the benchmark requires continuous updates to stay relevant.

## Related Work & Insights
- **vs. Single-turn Benchmarks (COLD / SafetyBench)**: SafeDialBench sacrifices scale for depth in multi-turn dynamics and fine-grained capability assessment.
- **vs. Multi-turn Benchmarks (CoSafe / RED QUEEN)**: While prior works are often monolingual and binary, this work provides bilingual support, 7 attack types, up to 10 turns, and three-capability scoring.
- **vs. Multilingual Benchmarks (LinguaSafe)**: LinguaSafe focuses on language diversity, whereas SafeDialBench focuses on multi-turn adversarial dynamics.

## Rating
- Novelty: ⭐⭐⭐⭐ (First fine-grained bilingual multi-turn benchmark; original three-capability/min-score paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (19 models, 6 dimensions, 7 attacks, multi-turn analysis, human validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete figures, well-defined attacks)
- Value: ⭐⭐⭐⭐ (Provides reusable standards and diagnostic tools for multi-turn jailbreak safety)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Obfuscated Activations Bypass LLM Latent-Space Defenses](obfuscated_activations_bypass_llm_latent-space_defenses.md)
- [\[ICLR 2026\] Understanding and Improving Continuous Adversarial Training for LLMs via In-Context Learning Theory](understanding_and_improving_continuous_llm_adversarial_training_via_in-context_l.md)
- [\[ICLR 2026\] RedCodeAgent: Automatic Red-teaming Agent against Diverse Code Agents](redcodeagent_automatic_red-teaming_agent_against_diverse_code_agents.md)
- [\[ICLR 2026\] Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-Tuning and Can Be Mitigated by Machine Unlearning](safety_mirage_how_spurious_correlations_undermine_vlm_safety_fine-tuning_and_can.md)
- [\[ICLR 2026\] JailbreakLoRA: Your Downloaded LoRA from Sharing Platforms might be Unsafe](jailbreaklora_your_downloaded_lora_from_sharing_platforms_might_be_unsafe.md)

</div>

<!-- RELATED:END -->
