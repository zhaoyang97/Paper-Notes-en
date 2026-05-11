---
title: >-
  [Paper Note] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning
description: >-
  [ACL 2026][LLM Reasoning][Prompt Injection Attack] This paper proposes InstruCoT, which synthesizes diverse training data covering multiple injection vectors and threat scenarios…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Prompt Injection Attack"
  - "Instruction-Level Alignment"
  - "Chain-of-Thought Reasoning"
  - "Data Synthesis"
  - "Safety Fine-Tuning"
date: 2026-05-08
content_hash: e2392d8d8778b8df
---

# Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning

**Conference**: ACL 2026  
**arXiv**: [2601.04666](https://arxiv.org/abs/2601.04666)  
**Code**: [GitHub](https://anonymous.4open.science/r/InstruCoT-LLM-045F)  
**Area**: LLM Reasoning  
**Keywords**: Prompt Injection Attack, Instruction-Level Alignment, Chain-of-Thought Reasoning, Data Synthesis, Safety Fine-Tuning

## TL;DR

This paper proposes InstruCoT, which synthesizes diverse training data covering multiple injection vectors and threat scenarios, and introduces a three-stage instruction-level chain-of-thought fine-tuning framework based on a situation-aware model. This enables LLMs to effectively identify and reject malicious instructions under various prompt injection attacks, substantially outperforming existing defenses across three evaluation dimensions: behavioral deviation, privacy leakage, and harmful output.

## Background & Motivation

**State of the Field**: LLM-integrated applications are increasingly prevalent, yet face serious Prompt Injection (PI) security threats—ranked by OWASP as the top security risk for LLM applications. Current defenses fall into two categories: external detector-based filtering of suspicious inputs, and post-training methods that enhance the intrinsic robustness of LLMs.

**Limitations of Prior Work**: (1) **Multi-vector injection**: LLM application scenarios are diverse (dialogue systems, tool invocation, external information retrieval, etc.), and attack vectors vary greatly in both injection content and injection position. Direct injections typically appear in user regions, while indirect injections may appear in data regions. If training data fails to adequately reflect this diversity, defensive performance degrades significantly. (2) **Blurred semantic boundaries**: Modern attackers are increasingly adept at embedding malicious instructions within seemingly legitimate contexts, obscuring the semantic boundary between injection regions and benign content, making it difficult for LLMs to accurately distinguish them.

**Root Cause**: Existing post-training defenses (e.g., StruQ, SecAlign) primarily rely on role boundaries (user region vs. data region) to identify injections, but this role-boundary-based approach fails when malicious instructions are semantically coherent with the surrounding context. A method capable of fine-grained analysis at the instruction level is required.

**Paper Goals**: To construct diverse training data covering multiple injection contents and positions, and to design an instruction-level reasoning guidance strategy that teaches LLMs to identify malicious content by analyzing the instructions themselves.

**Starting Point**: Drawing on Endsley's Situation Awareness model—a three-level cognitive process of perception, comprehension, and projection—the paper designs an instruction-level chain-of-thought reasoning framework that transforms the LLM's implicit understanding of malicious instructions into explicit, structured analysis.

**Core Idea**: The PI defense problem is reformulated as an instruction-level conflict detection problem. Through diverse data synthesis combined with three-stage CoT reasoning fine-tuning, LLMs are trained to no longer rely on role boundaries, but instead to perceive all instructions, determine whether each violates the system prompt, and decide whether to comply or refuse.

## Method

### Overall Architecture

InstruCoT comprises three stages: (1) diverse prompt injection data synthesis—generating broad-coverage training data based on three threat scenario categories and four context region types; (2) instruction-aware CoT generation—producing three-stage structured reasoning content for each training sample; and (3) supervised fine-tuning—performing full-parameter fine-tuning on the CoT-augmented dataset so that LLMs learn to output instruction-level analysis before generating the final response.

### Key Designs

1. **Diverse Injection Instruction Generation**:

    - **Function**: Covers three threat scenario categories and generates diverse malicious injection instructions.
    - **Mechanism**: For behavioral deviation scenarios, injection instructions are designed along two orthogonal dimensions (domain alignment × topic relevance), spanning four deviation levels—from same-domain same-topic (hardest to detect) to cross-domain off-topic (easiest to detect). Privacy leakage scenarios cover three protection scopes: user-level PII, organizational confidential data, and system-level secrets. Harmful output scenarios follow the harmful content taxonomy of Shen et al. to cover multiple harm categories. The generation formula is $VII = \text{LLM}(\mathcal{T}_{inj}, P_{sys}, s, l)$.
    - **Design Motivation**: Low-deviation instructions help the LLM learn precise decision boundaries, while high-deviation instructions increase data diversity—the two are complementary, ensuring robust defense against attacks of varying subtlety.

2. **Context Region Reduction and Data Synthesis**:

    - **Function**: Systematically covers all possible injection positions.
    - **Mechanism**: Drawing on the idea of data-flow analysis from program analysis, the approach traces how external content flows from the application framework through functional components to the LLM's input context. Through a three-layer analysis (application framework → functional components → context regions), diverse injection vectors are reduced to four context region types: User (user input), Data (external data), User+Data (both), and Empty (direct injection with no prior context). Adversarial datasets are constructed by injecting into each region.
    - **Design Motivation**: Existing methods (StruQ, SecAlign) inject only in the data region, offering limited coverage. InstruCoT's region reduction ensures comprehensive coverage of real-world attack vectors.

3. **Three-Stage Instruction-Aware CoT Reasoning**:

    - **Function**: Guides LLMs to learn to identify and analyze malicious instructions in a structured manner.
    - **Mechanism**: The three-stage pipeline proceeds as follows—**Instruction Perception** (exhaustively extracts all instructions in the context in a neutral, non-prejudicial manner) → **Violation Comprehension** (analyzes each instruction via three steps: isolated presentation, binary conflict judgment yes/no, and reasoning to articulate the semantic basis) → **Response Projection** (determines whether to comply with or refuse each instruction based on the analysis). The CoT generation formula is $CoT = \text{LLM}(\mathcal{T}_{cot}, P_{sys}, P_{con})$.
    - **Design Motivation**: (1) Exhaustive perception prevents omissions; (2) per-instruction analysis avoids inconsistencies inherent in holistic judgments; (3) binary judgment provides strong training signal, preventing probabilistic ambiguity from weakening the learning effect.

### Loss & Training

Full-parameter supervised fine-tuning is performed on the CoT-augmented dataset using the standard negative log-likelihood loss: $\mathcal{L} = -\sum_{i=1}^{N} \log P_\theta(y_i | x_i)$, where $y_i = (CoT_i, R_i)$ includes both the reasoning process and the final response. Training data comprises both adversarial and clean samples (to prevent over-refusal). GPT-4.1 is used to generate injection instructions and CoT content.

## Key Experimental Results

### Main Results

**Behavioral Deviation Defense Rate (DR%, averaged over four models)**

| Attack Method | Clean | ISE | MetaSec | IP | PromptArmor | InstruCoT |
|---------------|-------|-----|---------|-----|-------------|-----------|
| Naive_SP | 21.3 | 84.9 | 77.6 | 21.9 | 32.9 | **94.6** |
| Escape_SP | 23.9 | 84.8 | 49.1 | 24.2 | 55.6 | **98.9** |
| Combined | 7.9 | 79.5 | 91.2 | 7.1 | 86.7 | **97.2** |
| TopicAttack | 11.2 | 22.0 | 51.7 | 9.2 | 61.8 | **79.0** |
| **AVG** | 11.4 | 66.7 | 68.5 | 11.0 | 50.8 | **92.5** |

### Ablation Study — CoT Quality Evaluation

| Dataset/Context | Instruction Perception F1 | Violation Comprehension Acc. | Response Projection Acc. |
|-----------------|--------------------------|------------------------------|--------------------------|
| Alpaca-Clean/Data | 100.0% | 100.0% | 100.0% |
| Alpaca-Adv/Data+PI | 98.5% | 100.0% | 99.7% |
| SystemChat-Adv/PI | 97.3% | 100.0% | 99.0% |
| Ultrachat-Adv/Data+User+PI | 99.0% | 100.0% | 100.0% |
| **Average** | **98.3%** | **99.7%** | **99.3%** |

### Key Findings

- InstruCoT achieves an average DR of 92.5% on the behavioral deviation dimension, surpassing the strongest baseline (MetaSec, 68.5%) by nearly 24 percentage points.
- DR reaches 98.0% on the privacy leakage dimension and 90.9% on the harmful output dimension.
- Against the recent TopicAttack (a semantically coherent stealthy attack), InstruCoT still achieves 79.0%, far exceeding other methods.
- CoT quality is high: instruction perception F1 of 98.3% and violation comprehension accuracy of 99.7%, validating the effectiveness of the three-stage framework.
- After safety alignment, LLM utility on tasks such as tool use shows no degradation.

## Highlights & Insights

- The data-flow reduction idea is elegant: reducing complex application-layer attack vectors to four context region types applies program analysis methodology to security problems in a systematic and scalable manner.
- The three-stage CoT pipeline—neutral perception → per-instruction judgment → action projection—is carefully designed, particularly the binary conflict judgment (yes/no) rather than probabilistic scoring, which provides stronger training signal.
- The four-level deviation design in behavioral deviation scenarios (same-domain same-topic → cross-domain off-topic) is practically valuable: low-deviation samples teach the LLM fine-grained discrimination, while high-deviation samples ensure baseline defense coverage.

## Limitations & Future Work

- The reliance on GPT-4.1 for generating training data and CoT content introduces a dependency on closed-source models and incurs associated costs.
- Full-parameter fine-tuning entails significant computational overhead; parameter-efficient alternatives (e.g., LoRA) are not explored.
- Experiments cover only open-source models at the 7B–8B scale; applicability to larger-scale and closed-source models remains unvalidated.
- CoT reasoning increases token generation at inference time, potentially affecting deployment in latency-sensitive scenarios.
- Attack methods are primarily known patterns; generalization to entirely novel, unseen attack paradigms requires further investigation.

## Related Work & Insights

- **vs. StruQ/SecAlign**: These methods rely on role boundaries between user and data regions to identify injections and inject training data only in the data region. InstruCoT analyzes conflicts at the instruction level, covers four context region types, and is more robust to semantically ambiguous attacks.
- **vs. ISE**: ISE extends injection to data and empty context positions, but still lacks coverage of User+Data combined scenarios and does not distinguish between deviation levels of injected instructions. InstruCoT is more comprehensive in both injection position and content complexity.
- **vs. PromptArmor**: As a detection-based method, PromptArmor performs well on Fake Completion attacks (89%), but exhibits high variance across other attack types. As a model-enhancement method, InstruCoT delivers more consistent performance across attacks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The instruction-level CoT reasoning framework and data-flow reduction idea are novel, though the overall approach still follows the data synthesis + fine-tuning paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 LLMs × 3 threat dimensions × 7 attack methods × 5 baselines; coverage is exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is clear and method description is systematic, though some formulations are somewhat redundant.
- **Value**: ⭐⭐⭐⭐ Directly applicable to secure LLM deployment; the diverse data synthesis framework is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ACL 2026\] Reinforced Efficient Reasoning via Semantically Diverse Exploration](reinforced_efficient_reasoning_via_semantically_diverse_exploration.md)

</div>

<!-- RELATED:END -->
