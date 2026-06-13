---
title: >-
  [Paper Note] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning
description: >-
  [ACL 2026][LLM Safety][Prompt Injection Attacks] This paper proposes InstruCoT, which synthesizes diverse training data covering multiple injection vectors and threat scenarios. It introduces a three-stage instruction-le…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Prompt Injection Attacks"
  - "Instruction-Level Alignment"
  - "Chain-of-Thought Reasoning"
  - "Data Synthesis"
  - "Safety Fine-tuning"
date: 2026-05-08
content_hash: 6f3895ddd71009e8
---

# Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning

**Conference**: ACL 2026  
**arXiv**: [2601.04666](https://arxiv.org/abs/2601.04666)  
**Code**: [GitHub](https://anonymous.4open.science/r/InstruCoT-LLM-045F)  
**Area**: LLM Reasoning  
**Keywords**: Prompt Injection Attacks, Instruction-Level Alignment, Chain-of-Thought Reasoning, Data Synthesis, Safety Fine-tuning

## TL;DR

This paper proposes InstruCoT, which synthesizes diverse training data covering multiple injection vectors and threat scenarios. It introduces a three-stage instruction-level Chain-of-Thought (CoT) fine-tuning based on the Situation Awareness model, enabling LLMs to effectively identify and reject malicious instructions when facing various prompt injection attacks. It significantly outperforms existing defense methods across three dimensions: behavioral deviation, privacy leakage, and harmful output.

## Background & Motivation

**Background**: The integration of LLM applications is increasingly popular, but faces serious Prompt Injection (PI) security threats—OWASP lists it as the top security risk for LLM applications. Current defense methods are divided into two categories: intercepting suspicious inputs via external detectors, and enhancing the LLM's own robustness through post-training.

**Limitations of Prior Work**: (1) **Multi-vector injection issues**: LLM application scenarios are diverse (dialogue systems, tool calls, external information retrieval, etc.), and injection vectors vary greatly in content and position. Direct injections usually appear in the user region, while indirect injections may appear in the data region. If training data fails to reflect this diversity, defense effectiveness drops significantly. (2) **Semantic boundary ambiguity**: Modern attackers are increasingly adept at wrapping malicious instructions in seemingly normal contexts, blurring the semantic boundary between injection regions and legitimate content, making it difficult for LLMs to distinguish them accurately.

**Key Challenge**: Existing post-training defense methods (e.g., StruQ, SecAlign) mainly rely on role boundaries (user zone vs. data zone) to identify injections. However, when malicious instructions are semantically coherent with the context, these boundary-based methods fail. A method capable of fine-grained analysis at the instruction level is required.

**Goal**: To construct diverse training data covering multiple injection contents and locations, and design an instruction-level reasoning guidance strategy that enables LLMs to learn how to identify malicious content based on the instructions themselves.

**Key Insight**: Inspired by Endsley’s Situation Awareness model—a three-level cognitive process of perception, comprehension, and projection—an instruction-level CoT reasoning framework is designed to transform the LLM's implicit understanding of malicious instructions into explicit structured analysis.

**Core Idea**: Reframe the PI defense problem as an instruction-level conflict detection problem. Through diverse data synthesis + three-stage CoT reasoning fine-tuning, the LLM no longer relies on role boundaries but learns to perceive all instructions, judge whether each violates the system prompt, and decide whether to follow or reject it.

## Method

### Overall Architecture

InstruCoT consists of three stages: (1) Diverse prompt injection data synthesis—generating broad training data based on three threat scenarios and four context regions; (2) Instruction-aware CoT generation—generating three-stage structured reasoning content for each training sample; (3) Supervised fine-tuning—performing full-parameter fine-tuning on the CoT-enhanced dataset, enabling the LLM to output instruction-level analysis before generating a final response.

### Key Designs

1.  **Diverse Injection Instruction Generation**:
    *   Function: Covers three threat scenarios to generate diverse malicious injection instructions.
    *   Mechanism: For behavioral deviation scenarios, injection instructions are designed across four levels of deviation along two orthogonal dimensions (domain alignment $\times$ topic relevance)—ranging from same-domain/same-topic (hardest to identify) to different-domain/different-topic (easiest). Privacy leakage scenarios cover three protection scopes: user-level PII, organizational secrets, and system-level secrets. Harmful output scenarios refer to the harmful content taxonomy by Shen et al. The generation formula is $VII = \text{LLM}(\mathcal{T}_{inj}, P_{sys}, s, l)$.
    *   Design Motivation: Low-deviation instructions help the LLM learn precise decision boundaries, while high-deviation instructions increase data diversity—complementing each other to ensure robust defense against various attacks.

2.  **Context Region Reduction and Data Synthesis**:
    *   Function: Systematically covers all possible injection locations.
    *   Mechanism: Borrowing the concept of data-flow analysis from program analysis, it tracks how external content flows from the application framework through functional components to the LLM's input context. Through three layers of analysis (Application Framework $\rightarrow$ Functional Component $\rightarrow$ Context Region), diverse injection vectors are reduced to four context regions: User (user input), Data (external data), User+Data (both), and Empty (direct injection without prior context). Targeted injections are constructed for each region to build the adversarial dataset.
    *   Design Motivation: Existing methods (StruQ, SecAlign) only inject into data regions, providing limited coverage. InstruCoT ensures comprehensive coverage of real-world attack vectors through region reduction.

3.  **Three-Stage Instruction-Aware CoT Reasoning**:
    *   Function: Guides the LLM to learn structured identification and analysis of malicious instructions.
    *   Mechanism: A three-stage process—**Instruction Perception** (exhaustively extract all instructions in context, staying neutral without prejudgment) $\rightarrow$ **Violation Comprehension** (perform three-step analysis for each instruction: isolated presentation, binary conflict determination yes/no, and reasoning describing semantic basis) $\rightarrow$ **Response Prediction** (decide to follow or reject each instruction based on analysis). The CoT generation formula is $CoT = \text{LLM}(\mathcal{T}_{cot}, P_{sys}, P_{con})$.
    *   Design Motivation: (1) Exhaustive perception prevents omissions; (2) Step-by-step analysis avoids inconsistency in overall judgment; (3) Binary determination provides a strong training signal, preventing probabilistic ambiguity from weakening learning effects.

### Loss & Training

Full-parameter supervised fine-tuning is conducted using the enhanced CoT dataset with standard negative log-likelihood loss: $$\mathcal{L} = -\sum_{i=1}^{N} \log P_\theta(y_i | x_i)$$, where $y_i = (CoT_i, R_i)$ contains the reasoning process and the final response. Training data includes both adversarial and clean samples (to prevent over-refusal). GPT-4.1 is used to generate injection instructions and CoT content.

## Key Experimental Results

### Main Results

**Defense Rate for Behavioral Deviation (DR%, average of four models)**

| Attack Method | Clean | ISE | MetaSec | IP | PromptArmor | InstruCoT |
|----------|-------|-----|---------|-----|-------------|-----------|
| Naive_SP | 21.3 | 84.9 | 77.6 | 21.9 | 32.9 | **94.6** |
| Escape_SP | 23.9 | 84.8 | 49.1 | 24.2 | 55.6 | **98.9** |
| Combined | 7.9 | 79.5 | 91.2 | 7.1 | 86.7 | **97.2** |
| TopicAttack | 11.2 | 22.0 | 51.7 | 9.2 | 61.8 | **79.0** |
| **AVG** | 11.4 | 66.7 | 68.5 | 11.0 | 50.8 | **92.5** |

### Ablation Study — CoT Quality Evaluation

| Dataset/Context | Instruction Perception F1 | Violation Comp. Accuracy | Response Pred. Accuracy |
|---------------|------------|-------------|-------------|
| Alpaca-Clean/Data | 100.0% | 100.0% | 100.0% |
| Alpaca-Adv/Data+PI | 98.5% | 100.0% | 99.7% |
| SystemChat-Adv/PI | 97.3% | 100.0% | 99.0% |
| Ultrachat-Adv/Data+User+PI | 99.0% | 100.0% | 100.0% |
| **Average** | **98.3%** | **99.7%** | **99.3%** |

### Key Findings

*   InstruCoT achieves an average DR of 92.5% in behavioral deviation, exceeding the strongest baseline (MetaSec 68.5%) by nearly 24 percentage points.
*   DR reaches 98.0% for privacy leakage and 90.9% for harmful output.
*   Against the latest TopicAttack (stealthy attacks with semantic coherence), InstruCoT still achieves 79.0%, far surpassing other methods.
*   CoT quality is high: 98.3% F1 for instruction perception and 99.7% accuracy for violation comprehension, proving the effectiveness of the three-stage framework.
*   LLM utility on tasks like tool use remains intact after security alignment.

## Highlights & Insights

*   The data-flow reduction idea is clever: reducing complex application-layer attack vectors to four context regions treats security problems with a program analysis methodology, making it both systematic and scalable.
*   The "neutral perception $\rightarrow$ itemized judgment $\rightarrow$ action prediction" flow of the three-stage CoT is well-designed. In particular, binary conflict determination (yes/no) instead of probability scores provides stronger training signals.
*   The four-level deviation design in behavioral scenarios (same-domain/same-topic to different-domain/different-topic) is highly practical—low-deviation samples teach the LLM fine-grained distinction, while high-deviation samples ensure basic defense.

## Limitations & Future Work

*   Reliance on GPT-4.1 for generating training data and CoT content introduces dependency on closed-source models and increases cost.
*   Full-parameter fine-tuning is computationally expensive; parameter-efficient alternatives (e.g., LoRA) were not explored.
*   Experiments only cover open-source models in the 7B-8B range; applicability to larger or closed-source models has not been verified.
*   CoT reasoning increases token generation at inference time, potentially impacting latency-sensitive deployments.
*   Attack methods are mainly based on known patterns; generalization to entirely new attack paradigms remains to be verified.

## Related Work & Insights

*   **vs StruQ/SecAlign**: These methods rely on role boundaries between user/data zones and only inject training data into the data region. InstruCoT analyzes conflicts at the instruction level and covers four context regions, making it more robust against semantic ambiguity attacks.
*   **vs ISE**: ISE extends injection locations to data zone and empty contexts but lacks coverage for User+Data combinations and does not distinguish instruction deviation levels. InstruCoT is more comprehensive in both location and content complexity.
*   **vs PromptArmor**: As a detection-based method, PromptArmor performs well on Fake Completion attacks (89%) but fluctuates significantly on others. As a model-enhancement method, InstruCoT remains stable across different attacks.

## Rating

*   Novelty: ⭐⭐⭐⭐ The instruction-level CoT reasoning framework and data-flow reduction idea are novel, though the overall paradigm remains data synthesis + fine-tuning.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 4 LLMs $\times$ 3 threat dimensions $\times$ 7 attack methods $\times$ 5 baselines.
*   Writing Quality: ⭐⭐⭐⭐ Clear problem analysis and systematic method description, though some formulas are slightly redundant.
*   Value: ⭐⭐⭐⭐ High practical value for secure LLM deployment; the diverse data synthesis framework is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)

</div>

<!-- RELATED:END -->
