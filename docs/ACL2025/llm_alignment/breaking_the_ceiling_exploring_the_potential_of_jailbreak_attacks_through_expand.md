---
title: >-
  [Paper Note] Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space
description: >-
  [ACL 2025][LLM Alignment][Jailbreak Attacks] Based on the Elaboration Likelihood Model (ELM), this work decomposes jailbreak strategies into four categories of independently evolvable components (Role/Content Support/Context/Communication Skills). It proposes the CL-GSO genetic algorithm to execute crossover and mutation at the component level, expanding the strategy space from 40 in prior work to 839. This achieves a 96% Jailbreak Success Rate (JSR) on Claude-3.5 (where prio…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Jailbreak Attacks"
  - "Strategy Space Expansion"
  - "Genetic Algorithm"
  - "Elaboration Likelihood Model (ELM)"
  - "Component-level Optimization"
  - "CL-GSO"
date: 2026-05-08
content_hash: a8a5640b74e089d6
---

# Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space

**Conference**: ACL 2025  
**arXiv**: [2505.21277](https://arxiv.org/abs/2505.21277)  
**Code**: [Aries-iai/CL-GSO](https://github.com/Aries-iai/CL-GSO)  
**Area**: LLM Alignment / AI Safety  
**Keywords**: Jailbreak Attacks, Strategy Space Expansion, Genetic Algorithm, Elaboration Likelihood Model (ELM), Component-level Optimization, CL-GSO  

## TL;DR

Based on the Elaboration Likelihood Model (ELM), this work decomposes jailbreak strategies into four categories of independently evolvable components (Role/Content Support/Context/Communication Skills). It proposes the CL-GSO genetic algorithm to execute crossover and mutation at the component level, expanding the strategy space from 40 in prior work to 839. This achieves a 96% Jailbreak Success Rate (JSR) on Claude-3.5 (where prior methods reached at most 4%). Meanwhile, a strategy evaluation mechanism based on intent consistency is proposed, reaching an accuracy of 96.5% and outperforming specialized safety models.

## Background & Motivation

**Background**: Black-box jailbreak attacks represent a critical method for evaluating LLM safety. Existing methods (PAIR, TAP, GPTFuzzer, PAP) generate jailbreak prompts by combining predefined strategy sets with prompt engineering techniques (such as self-reflection and Chain-of-Thought). However, even PAP, which possesses the largest strategy pool, only contains around 40 handcrafted strategies.

**Limitations of Prior Work**: Regardless of how sophisticated prompt engineering techniques are, the ceiling of their efficacy is strictly bounded by the underlying strategy space. Faced with strong safety-aligned models such as Claude-3.5, the jailbreak success rate (JSR) of existing methods is close to zero (at most 4%), suggesting that they have hit the ceiling of strategy diversity.

**Key Challenge**: Traditional methods treat jailbreak strategies as indivisible, monolithic units, failing to systematically discover new strategic vectors. The limited nature of the strategy space, rather than the prompt optimization technique, becomes the fundamental bottleneck restricting attack effectiveness.

**Goal**: To systematically expand the jailbreak strategy space to break through the performance ceiling of existing methods, while addressing the optimization efficiency challenges posed by the increased search complexity after expansion.

**Key Insight**: Drawing inspiration from the Elaboration Likelihood Model (ELM) in psychology, the persuasion process is decomposed into a central route (argument quality, source credibility, contextual framing) and a peripheral route (message delivery skills). Accordingly, jailbreak strategies are decomposed into independently variable and freely combinable components.

**Core Idea**: Decomposing jailbreak strategies from "monolithic units" into "independently evolvable components," and utilizing a genetic algorithm for crossover and mutation at the component level to automatically generate combined strategies, thereby significantly expanding the strategy space.

## Method

### Overall Architecture

The CL-GSO (Component-Level Genetic-based Strategy Optimization) framework consists of three core components:

1. **Component-based Strategy Space**: Defines four independent strategy components based on ELM theory.
2. **Genetic-based Strategy Optimization**: Efficiently searches the expanded strategy space.
3. **Strategy Evaluation Mechanism**: Accurately evaluates whether the jailbreak is successful.

Each strategy is represented as a four-dimensional vector $S_i = \langle S_{A_i}, S_{B_i}, S_{C_i}, S_{D_i} \rangle$, corresponding to the four components respectively.

### Key Designs

### Key Design 1: ELM Theory-Guided Strategy Component Decomposition

- **Function**: Decomposes jailbreak strategies from an indivisible whole into four independent yet complementary components.
- **Mechanism**: Based on the dual-route theory of ELM, the central route (influencing deep information processing) corresponds to three components, while the peripheral route (influencing surface reaction behaviors) corresponds to one component:
    - **Role (A)**: Establishes proxy source credibility, such as professional role-playing (central route).
    - **Content Support (B)**: Provides reasoning and evidence support (central route).
    - **Context (C)**: Creates an appropriate situational framework (central route).
    - **Communication Skills (D)**: Optimizes message delivery methods, such as tone and formatting (peripheral route).
- **Design Motivation**: The four components are functionally independent but produce synergistic effects through the ELM dual-route framework. This modular design ensures that strategies created via component recombination remain psychologically sound while enabling a combinatorial explosion of the strategy space.
- **Strategy Space Formulation**: $\mathbb{S} = \{S \mid S = \langle S_A, S_B, S_C, S_D \rangle\}$, ultimately constructing **839** possible strategies through systematic analysis, vastly exceeding the previous maximum of 40.

### Key Design 2: Component-Level Genetic Optimization Algorithm

- **Function**: Efficiently searches for the optimal strategy combination within the expanded strategy space.
- **Mechanism**: The hierarchical structure of strategies (from atomic components to emergent interactions) naturally corresponds to the genotype-phenotype relationship in biological evolution. Therefore, a genetic algorithm is adopted as the optimization framework to transform genetic operations into meaningful strategy improvements:
    - **Population Initialization**: N strategies are each encoded as a four-dimensional vector.
    - **Selection & Crossover**: Parents are selected based on fitness scores, and components of different strategies are exchanged to generate offspring, e.g., $S_i' = \langle S_{A_i}, S_{B_{i'}}, S_{C_i}, S_{D_{i'}} \rangle$.
    - **Mutation**: Probabilistically replaces an element of a certain component dimension with another valid option from the same dimension, e.g., replacing Content Support with a new $S_{B_{i''}} \in \mathbb{B}$.
- **Design Motivation**:
    - **Memory Bank**: Stores generated strategies, regenerating if duplicates are encountered to eliminate redundant searches.
    - **Soft Decay Strategy**: The crossover rate and mutation rate decay with iterations as $r_t = r_0 \cdot 0.9^t$, emphasizing exploration diversity in the early stages and exploitation refinement in the later stages.

### Key Design 3: Intent-Consistency Evaluation Mechanism

- **Function**: Accurately evaluates whether the jailbreak attack was successful.
- **Mechanism**: Instead of focusing on literal content classification of responses, it inspects whether the model's response genuinely addresses the malicious intent behind the query—transforming evaluation into a semantic understanding task.
- **Scoring Design Principles**:
    - **Mutually Exclusive and Collectively Exhaustive (MECE)**: Each scoring level represents a unique, non-overlapping response pattern, eliminating scoring ambiguity inherent in previous methods.
    - **Intent Consistency**: Determines Level 3 and above as a successful jailbreak (effectively satisfying the attack intent).
    - **Keyword-assisted Verification**: Grants extra points to responses that avoid explicit refusal, reducing evaluation bias.
- **Design Motivation**: Binary classification systems are oversimplified (mislabeled benign responses as harmful), rule-intensive scoring criteria suffer from overlapping ambiguity, and specialized safety models are limited by training data coverage. Intent consistency fundamentally resolves these critical issues.

## Key Experimental Results

### Main Results: Jailbreak Success Rate (JSR) and Average Queries (Avg.Q)

**AdvBench Dataset:**

| Method | Llama3 JSR | Qwen-2.5 JSR | GPT-4o JSR | Claude-3.5 JSR |
|------|-----------|-------------|-----------|---------------|
| PAIR | 22% | 94% | 35% | 2% |
| TAP | 20% | 92% | 60% | 4% |
| GPTFuzzer | 96% | 96% | 66% | 4% |
| **CL-GSO** | **92%** | **98%** | **94%** | **96%** |

**CLAS Dataset:**

| Method | Llama3 JSR | Qwen-2.5 JSR | GPT-4o JSR | Claude-3.5 JSR |
|------|-----------|-------------|-----------|---------------|
| PAIR | 52% | 92% | 80% | 1% |
| TAP | 44% | 90% | 68% | 3% |
| GPTFuzzer | 95% | 97% | 61% | 0% |
| **CL-GSO** | **92%** | **97%** | **97%** | **87%** |

On the more complete AdvBench full set (500 samples), CL-GSO maintains a **95.2% JSR**, requiring an average of only 18.2 queries.

### Ablation Study

**Component Removal Ablation (Fig. 7)**: Removing any component leads to a drop in JSR and an increase in query costs, with the **Role component having the greatest impact**, which is consistent with the empirical emphasis on role-playing in major jailbreak methods.

**Hyperparameter Tuning (Fig. 6)**: A population size of 15 + maximum iterations of 5 is the optimal equilibrium point (yielding high JSR and acceptable query costs). At population 20 + iterations 9, the JSR peaks at 96%, but the query cost rises to 39.80.

**Strategy Space Scale Ablation (Fig. 3)**: As the strategy space gradually expands, the JSR continues to improve. Even the weakest CL-GSO configuration outperforms other methods by **74% JSR**.

### Evaluation Mechanism Comparison (Fig. 5)

| Evaluation Method | Accuracy |
|---------|-------|
| Binary Judge | 46% |
| Skywork-Reward | 49.5% |
| Llama3-Guard | 56% |
| Rule-intensive Scoring | 76.5% |
| **Intent-Consistency Scoring (Ours)** | **96.5%** |

### Defense Robustness (Fig. 8)

Against two defense methods, RA-LLM and SmoothLLM, CL-GSO still achieves a **60%+ JSR** in most scenarios. The only exception is the Claude-3.5 + SmoothLLM combination, where performance declines—since Claude-3.5 is highly sensitive to harmful content, minor perturbations can trigger its anomaly detection.

### Cross-Model Transferability (Fig. 4)

- Prompts generated on GPT-4o transferred to Qwen-2.5: JSR reaches **94%** on both AdvBench and CLAS.
- Prompts generated on Claude-3.5 transferred to GPT-4o: **88%** on AdvBench, **89%** on CLAS.
- Even for the OpenAI o1 model (transferred using prompts generated from Llama3), it achieves a **24% JSR**.

## Highlights & Insights

- **Paradigm Shift from "Strategy Selection" to "Strategy Generation"**: Traditional methods select and optimize within a fixed strategy pool, whereas CL-GSO fundamentally expands the strategy space itself, revealing that strategy diversity is the core lever of attack effectiveness.
- **Psychological Theory-Driven Attack Design**: The ELM dual-route theory provides a rigorous psychological foundation for decomposing and combining jailbreak strategies, ensuring that newly engineered strategies through component recombination remain persuasive.
- **Breakthrough in Evaluation Mechanisms**: The Intent-Consistency Scoring (96.5%) outperforms specialized safety models (Llama3-Guard 56%, Skywork-Reward 49.5%), demonstrating that carefully designed evaluation rubrics can outperform heavily trained safety models.
- **Surprising Findings in Cross-Model Transferability**: Expanding the strategy space not only enhances single-model attacks but also uncovers more generalized attack patterns.

## Limitations & Future Work

1. **Exploration of Modality Expansion Pending**: Validated only on textual LLMs. Efficacy in multimodal scenarios (incorporating additional attack surfaces such as images) remains unexplored, which the authors believe could yield even more significant results.
2. **Deployment Constraints of the Evaluation Mechanism**: Intent-consistency evaluation requires prior knowledge of the attack intent. Hence, it cannot be directly deployed as a general-purpose detection tool on the defense side of commercial LLMs (it is unavailable when the text intent is unknown).
3. **Insufficient Efficiency Analysis of the Genetic Algorithm**: Detailed ablations of the crossover/mutation rates are relegated to the appendix, and there is a lack of deep discussion on convergence behavior and computational overhead.
4. **Limitations in Defense Robustness**: Performance remains suboptimal against the SmoothLLM + Claude-3.5 combination, indicating that randomized perturbation-based defenses still hold some efficacy against component-level strategies.

## Related Work & Insights

- **vs PAIR/TAP**: Iteratively optimize prompts through LLM interactions within a fixed strategy space without changing its size. CL-GSO fundamentally expands the strategy space, overcoming the "strategy ceiling" problem.
- **vs GPTFuzzer**: Generates prompts based on template mutation, showing good results on open-source models (96%) but completely failing on Claude-3.5 (0-4% JSR). CL-GSO performs mutation at the component level, preserving better semantic integrity.
- **vs PAP**: The first to systematically introduce persuasive strategies (40 types), but the strategies are indivisible and non-combinable. CL-GSO treats PAP as a special case within its strategy space.
- **Key Revelation for Defense Research**: Defenses must not only target known strategy types but must also cope with the exponential expansion of the strategy space. The vulnerability of current safety alignment may be far more severe than previously recognized.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Combines psychological ELM theory with a genetic algorithm, expanding the jailbreak strategy space at the component level for the first time—a paradigm-shifting approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Tested on 4 target models (including the highly robust Claude-3.5), 2 datasets, cross-model transferability, defense robustness, ablation studies (components, hyperparameters, space scale), and evaluation mechanism comparisons. The experiments are exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical framework is clear, and the question-and-answer style format aids comprehension, though the frequent LaTeX formulas make some paragraphs feel slightly dense.
- **Value**: ⭐⭐⭐⭐⭐ Serves as a significant wake-up call for LLM safety research. Claude-3.5 dropped from near-immunity to a 96% successful breach, exposing the fundamental vulnerabilities of current safety alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models](beyond_the_tip_of_efficiency_uncovering_the_submerged_threats_of_jailbreak_attac.md)
- [\[ACL 2025\] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models](agd_adversarial_game_defense_against_jailbreak_attacks_in_large_language_models.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
