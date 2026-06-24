---
title: >-
  [Paper Note] FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models
description: >-
  [ACL 2025 (Long Paper, acl-long.17)][LLM Agent][Fact-Checking Evaluation] This paper proposes FACT-AUDIT, an adaptive dynamic fact-checking evaluation framework based on importance sampling and multi-agent collaboration. By dynamically generating test data, iteratively probing model weaknesses, and simultaneously evaluating both verdict predictions and justification quality, it comprehensively audits the boundaries of LLMs' fact-checking capabilities.
tags:
  - "ACL 2025 (Long Paper, acl-long.17)"
  - "LLM Agent"
  - "Fact-Checking Evaluation"
  - "Multi-Agent Framework"
  - "Importance Sampling"
  - "LLM Auditing"
  - "Justification Production"
date: 2026-05-08
content_hash: 8051006c7cce65ba
---

# FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models

**Conference**: ACL 2025 (Long Paper, acl-long.17)  
**arXiv**: [2502.17924](https://arxiv.org/abs/2502.17924)  
**Code**: None  
**Area**: Fact-Checking / LLM Evaluation / Multi-Agent Framework  
**Keywords**: Fact-Checking Evaluation, Multi-Agent Framework, Importance Sampling, LLM Auditing, Justification Production  

## TL;DR
This paper proposes FACT-AUDIT, an adaptive dynamic fact-checking evaluation framework based on importance sampling and multi-agent collaboration. By dynamically generating test data, iteratively probing model weaknesses, and simultaneously evaluating both verdict predictions and justification quality, it comprehensively audits the boundaries of LLMs' fact-checking capabilities.

## Background & Motivation
Existing LLM fact-checking evaluation methods suffer from three fundamental deficiencies:
1. **Static Datasets**: Relying on manually annotated, fixed test suites, they face data leakage and leaderboard gaming, failing to reveal the potential limitations of LLMs in a timely manner.
2. **Single Evaluation Dimension**: Simplifying fact-checking into classification accuracy assessment, they ignore the quality of the justification (the reasoning process)—even if the prediction is correct, the reasoning process may contain factual errors.
3. **Poor Scalability**: High manual annotation costs and constrained testing scenarios make it difficult to cover diverse real-world contexts such as complex claims, fake news, and social rumors.

## Core Problem
How to design an adaptive and scalable evaluation framework that dynamically discovers the capabilities boundaries of LLMs in fact-checking, particularly those hidden weaknesses where the "prediction is correct but the justification is flawed"?

## Method

### Overall Architecture
Fact-checking evaluation is modeled as an **importance sampling** process: while traditional evaluations inefficiently sample test cases from the oracle knowledge distribution $p(x)$, FACT-AUDIT designs a proposal distribution $q(x)$ to adaptively skew sampling towards regions where the LLM is likely to make mistakes, thereby revealing model weaknesses more efficiently. The framework iterates across three stages: (1) Prototype Emulation to generate test data, (2) Fact Verification to evaluate the target LLM, and (3) Adaptive Updating to update the test scenario taxonomy.

### Key Designs

1. **Five-Role Multi-Agent Collaboration**

    - **Appraiser**: Builds and maintains the taxonomy of fact-checking scenarios, initially covering three major categories: Complex Claims, Fake News, and Social Rumors, with each category containing multiple sub-scenarios (e.g., multi-step reasoning, aggregated statistical reasoning, headline mismatch). It adaptively adds new test scenarios during iterations candidates based on the model's weaknesses.
    - **Inquirer**: Generates prototype test data for each scenario. Each sample contains a Key Point (task instruction), Source Claim (claim to be verified), Auxiliary Information (supporting information), and Test Mode (testing mode), with temperature set to 0 to ensure fairness.
    - **Quality Inspector**: Uses external tools (Wikipedia API) for coarse filtering and a strong LLM for fine-grained filtering to guarantee the quality and diversity of the generated data.
    - **Evaluator**: Evaluates the answers of the target LLM using an LLM-as-a-Judge manner, providing scores from 1 to 10 along with natural language feedback, where a score $\leq 3$ is treated as an error. Reference answers are first generated through voting by three GPT-4o models, and then verified by another discriminative agent.
    - **Prober**: Based on the historical evaluation records in the memory pool, it iteratively generates more diverse and challenging test data to delve deeper into model weaknesses.

2. **Three Test Modes**

    - **[claim]**: Closed-book mode, where the LLM relies solely on parameterized knowledge to verify claims (most difficult).
    - **[evidence]**: Provides gold-standard evidence sourced from Wikipedia as support (easiest).
    - **[wisdom of crowds]**: Provides simulated social media comment threads as auxiliary information (medium difficulty).

3. **Theoretical Support of Importance Sampling**

    - Traditional Monte Carlo sampling has a convergence rate of $\mathcal{O}(1/\sqrt{N})$, and the long-tailed knowledge distribution further HTML-exacerbates this inefficiency.
    - A proposal distribution $q(x) \propto p(x) \cdot \mathcal{F}_\alpha(x)$ is designed to skew towards regions of model weaknesses.
    - Adaptive updating guarantees that variance decreases monotonically: $Var_{q_{i+1}} \leq Var_{q_i} \leq \cdots \leq Var_p$, accelerating the convergence rate round by round.

4. **Adaptive Taxonomy Updating**

    - After each round of evaluation, the Appraiser analyzes low-score cases in the memory pool to extract new challenging test scenarios.
    - For instance, "Aggregated Statistical Reasoning" is a new challenging scenario discovered during the adaptive updating.
    - This establishes a continuous loop of "evaluation $\rightarrow$ weakness discovery $\rightarrow$ scenario expansion $\rightarrow$ re-evaluation".

## Key Experimental Results

Audit results on 13 SOTA LLMs (lower IMR is better, higher Grade is better):

| Model | Complex Claims IMR↓ | Fake News IMR↓ | Social Rumors IMR↓ | Overall IMR↓ | Overall JFR↓ | Overall Grade↑ |
|-------|---------------------|----------------|---------------------|--------------|-------------|----------------|
| GPT-4o | 14.05 | 10.56 | 10.48 | **12.02** | 3.55 | **7.21** |
| Qwen2.5-72B | 22.08 | 10.42 | 15.00 | 16.00 | 3.50 | 7.17 |
| Claude3.5-Sonnet | 32.71 | 15.00 | 18.57 | 24.34 | 5.96 | 6.78 |
| Gemini-Pro | 30.21 | 19.39 | 32.86 | 27.25 | 8.62 | 6.14 |
| Qwen2.5-7B | 38.97 | 21.54 | 36.67 | 31.76 | 8.14 | 5.91 |
| Llama3.1-70B | 41.56 | 25.00 | 38.33 | 34.10 | 12.38 | 5.83 |
| Llama3-8B | 39.79 | 33.75 | 46.25 | 38.67 | 15.60 | 5.25 |
| Gemma2-9B | 41.67 | 35.48 | 44.07 | 39.70 | 26.78 | 4.94 |
| Llama3.1-8B | 55.83 | 36.39 | 47.62 | 47.52 | 16.77 | 4.91 |
| Llama2-7B | 46.67 | 32.73 | 62.86 | 45.49 | 20.68 | 4.88 |
| GLM4-9B | 52.73 | 51.67 | 50.00 | 51.67 | 15.24 | 4.88 |
| Mistral-7B | 60.21 | 47.50 | 59.05 | 54.79 | 23.34 | 4.34 |
| Llama2-13B | 65.67 | 55.33 | 48.10 | 57.28 | 19.50 | 4.25 |

Key Findings:
- GPT-4o ranks first with 12.02% IMR, but its JFR (3.55%) is not the lowest—indicating that even when strong models make mistakes, it is mostly due to insufficient justification quality.
- Qwen2.5-72B, as an open-source model, achieves a top-tier level comparable to closed-source models.
- LLMs perform worst on Complex Claims (requiring complex reasoning) and relatively best on Fake News.

### Ablation Study
- **LLM-generated vs. Manually-generated Prototype Data** (Table 2): The performances of both are highly consistent (e.g., GPT-4o: IMR 14.05 vs. 14.24), validating the fairness of the framework.
- **Comparison of Test Modes** (Table 3): The [claim] mode is the most difficult (Llama3.1-8B IMR 68.80%), the [evidence] mode is the easiest (38.16%), and [wisdom of crowds] lies in the middle (45.29%).
- **Iterative Probing Effect** (Figure 5): As the number of iterations increases, the IMR gradually decreases and converges, demonstrating that the Prober can effectively uncover more actual weaknesses.
- **Adaptive Updating**: Discovered 4, 3, and 1 new challenging scenarios for Qwen2.5-72B in Complex Claims, Fake News, and Social Rumors, respectively.
- **Human Quality Evaluation** (Table 5): Taxonomy validation rate of 98.86%, source claim validity of 97.17%, reference answer validity of 90.33%, and evaluation output quality of 89.02%.

## Highlights & Insights
- **Tight Integration of Theory and Practice**: Modeling dynamic evaluation as an importance sampling problem provides convergence guarantees with decreasing variance.
- **Comprehensive Evaluation Dimensions**: It evaluates not only the verdict accuracy but also focuses on the quality of justification, revealing hidden errors where the "prediction is correct but the reasoning is flawed" (e.g., GPT-4o claiming bamboo grows "up to 35 inches (91 cm)", whereas 35 inches equals 88.9 cm, showing a unit conversion error).
- **High Adaptability**: Each target LLM receives different test scenarios and data volume (e.g., 990 samples for GPT-4o vs. 1200 samples for Llama3-8B), truly achieving a model-centric evaluation.
- **Thorough Human Evaluation**: A quality assurance study on 600 samples combined with comparative human evaluation against Pinocchio and LLMFake shows leads in all six dimensions, including diversity and coverage.

## Limitations & Future Work
- **Agent Controller Bias**: Using GPT-4o as the agent controller (generating data and evaluating) inevitably introduces its own knowledge biases, akin to cognitive biases in human reviewers.
- **Information Lag**: The agent controller lacks the ability to dynamically acquire new information, making it unable to adapt to rapidly changing knowledge environments. Future work should integrate RAG technology.
- **Audit Only without Optimization**: While the framework can identify weaknesses, it currently does not provide a mechanism for model improvement. Future work could combine preference optimization (e.g., DPO) to generate training data.
- **High Cost**: Evaluating a single target LLM costs approximately \$25 and takes about 6 hours (requiring 2×A100 80GiB), with a total cost of around \$325 for 13 models.

## Related Work & Insights
- **vs. Pinocchio (Hu et al., 2024)**: Static manual dataset focusing solely on complex claims, with lower diversity (1.94) and coverage (2.14); FACT-AUDIT is dynamically adaptive, featuring more comprehensive coverage across three types of scenarios (2.58).
- **vs. LLMFake (Chen & Shu, 2024)**: Static LLM-generated data focusing solely on fake news, with the lowest coverage (1.65); FACT-AUDIT achieves the lowest redundancy (1.22) and the highest diversity (2.62) through iterative probing.
- **vs. AutoDetect (Cheng et al., 2024)**: A benchmarking work utilizing feedback to identify LLM weaknesses; however, FACT-AUDIT is specifically dedicated to the fact-checking domain and introduces the justification evaluation dimension.

## Related Work & Insights
- The framework design paradigm of "importance sampling + multi-agent + adaptive updating" is transferable to any scenario requiring dynamic evaluation of LLM capability boundaries (e.g., mathematical reasoning, code generation, etc.).
- The concept of justification quality evaluation is highly valuable for trustworthy AI research—a model that is "correct but for the wrong reasons" is equally hazardous in practical deployment.
- The iterative detection mechanism of Memory Pool + Prober is analogous to an automated loop for adversarial testing, which can be integrated into red-teaming efforts.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing importance sampling theory back into fact-checking evaluation offers a fresh perspective; the multi-agent collaboration framework is delicately designed, though the agents belong to the standard LLM agent paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ A comprehensive evaluation of 13 models, 3 sets of ablation studies, human quality assurance + benchmark comparisons, and highly persuasive case studies.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical portion is clear and rigorous, with a well-structured description of the framework and extremely detailed appendices (12 appendices in total).
- **Value**: ⭐⭐⭐⭐ The design concepts of the adaptive evaluation framework are highly worth referencing; justification-aware evaluation provides inspiration for trustworthy AI directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger](meco_metacognition_tool_use.md)
- [\[ACL 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)
- [\[ACL 2025\] Auto-TA: Towards Scalable Automated Thematic Analysis (TA) via Multi-Agent Large Language Models with Reinforcement Learning](auto-ta_towards_scalable_automated_thematic_analysis_ta_via_multi-agent_large_la.md)
- [\[ICML 2025\] Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search](../../ICML2025/llm_agent/improving_llm_agent_planning_with_in-context_learning_via_atomic_fact_augmentati.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)

</div>

<!-- RELATED:END -->
