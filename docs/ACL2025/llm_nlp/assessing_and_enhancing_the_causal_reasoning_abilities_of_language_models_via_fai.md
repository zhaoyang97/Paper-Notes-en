---
title: >-
  [Paper Note] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation
description: >-
  [ACL 2025][LLM (Other)][Causal reasoning] This paper proposes a framework based on Faithful Textual Interpretation (FTI), which evaluates and enhances the causal reasoning abilities of LLMs by faithfully translating variable relationships in causal reasoning tasks into natural language descriptions, achieving significant improvements across multiple causal reasoning benchmarks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Causal reasoning"
  - "language models"
  - "faithful textual interpretation"
  - "causal graphs"
  - "counterfactual reasoning"
date: 2026-05-08
content_hash: 316982b9e073a455
---

# Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Causal reasoning, language models, faithful textual interpretation, causal graphs, counterfactual reasoning

## TL;DR
This paper proposes a framework based on Faithful Textual Interpretation (FTI), which evaluates and enhances the causal reasoning abilities of LLMs by faithfully translating variable relationships in causal reasoning tasks into natural language descriptions, achieving significant improvements across multiple causal reasoning benchmarks.

## Background & Motivation

**Background**: Causal reasoning is a core capability of artificial intelligence, requiring models to understand causal directions, intervention effects, and counterfactual scenarios among variables. Existing causal reasoning evaluations for LLMs primarily rely on multiple-choice questions or simple causal judgment tasks, such as the CLADDER and CausalBench benchmark datasets. Evaluation methods typically present formal descriptions of causal graphs (such as lists of edges in a DAG) or numerical conditional probabilities directly.

**Limitations of Prior Work**: The performance of current LLMs on causal reasoning benchmarks is unstable, suffering from two key issues: first, the problem of evaluation faithfulness—in many benchmarks, LLMs might obtain correct answers through text pattern matching rather than genuine causal reasoning; second, the barrier of representation formats—when causal graphs are input into LLMs in symbolic forms like "X→Y", the models struggle to fully leverage the causal knowledge acquired during pre-training.

**Key Challenge**: While the pre-training corpora of LLMs contain a vast amount of natural language descriptions of causal relationships (e.g., "smoking causes lung cancer"), standard causal reasoning tasks are presented in abstract symbols and mathematical forms, resulting in a significant gap between model capabilities and evaluation formats.

**Goal**: To design a faithful textual interpretation method that translates abstract causal structures and reasoning steps into natural language, ensuring both the faithful representation of causal relations and the activation of the LLM's causal commonsense knowledge.

**Key Insight**: The authors observe that human causal reasoning performance improves significantly when causal relationships are described as "stories" in natural language. Inspired by this, the symbolic representations of variables and edges in causal graphs are replaced with semantically rich natural language descriptions.

**Core Idea**: To replace abstract symbolic representations of causal graphs with faithful natural language interpretations, enabling LLMs to leverage pre-trained knowledge for causal reasoning, while designing faithfulness constraints to ensure that the text interpretations do not distort the original causal relationships.

## Method

### Overall Architecture
The FTI framework consists of three core phases: (1) Causal Structure Textualization (CST), which converts DAG graph structures into semantically clear natural language descriptions; (2) Reasoning Path Guidance (RPG), which guides the LLM to reason along causal pathways via step-by-step reasoning prompts; (3) Faithfulness Verification, which ensures that the generated textual interpretations remain consistent with the original causal relationships. The input consists of a causal graph and a query, and the output consists of the causal reasoning answer and a natural language explanation of the reasoning process.

### Key Designs

1. **Causal Structure Textualization Module (CST)**:

    - **Function**: To translate abstract causal graphs into semantically rich natural language descriptions.
    - **Mechanism**: For each edge $X \rightarrow Y$ in the causal graph, it generates natural language descriptions like "when X changes, it directly causes Y to change accordingly." The key is to maintain causal directionality—distinguishing between "X causes Y" and "X is correlated with Y". For complex causal chains (e.g., confounders, mediators), a hierarchical description strategy is used: first describing direct causal relations, and then describing the complete causal path. Meanwhile, semantically contextualized descriptions are generated for each variable to transform them from abstract symbols into concrete concepts.
    - **Design Motivation**: LLMs have encountered a vast amount of causal relationships expressed in natural language during pre-training, and textualization can activate this latent knowledge, whereas pure symbolic inputs fail to establish such a connection.

2. **Step-by-step Causal Reasoning Guidance (SCRG)**:

    - **Function**: To guide the LLM to solve the problem step-by-step following proper causal reasoning actions.
    - **Mechanism**: Causal reasoning is decomposed into multiple steps: (a) identifying relevant variables and causal paths; (b) determining if confounding effects exist and identifying the adjustment set; (c) applying do-calculus rules or counterfactual reasoning frameworks; and (d) integrating information to draw the final conclusion. Each step is presented in natural language, combined with Chain-of-Thought prompts. For intervention questions, the LLM is explicitly required to distinguish between observational conditions ($P(Y|X)$) and interventional conditions ($P(Y|do(X))$).
    - **Design Motivation**: The challenge of causal reasoning lies in multi-step logical inference; directly asking the LLM to provide the final answer at once is error-prone. A step-by-step strategy reduces the cognitive load of each step.

3. **Faithfulness Bidirectional Verification Mechanism (FBV)**:

    - **Function**: To ensure that the textualized description faithfully reflects the original causal structure.
    - **Mechanism**: Two verification dimensions are designed: structural faithfulness (reconstructing the causal graph from the generated text description to check if it is isomorphic to the original graph) and reasoning faithfulness (performing reasoning on the same question with both symbolic and textual inputs to check if the reasoning paths are consistent). For textual descriptions that fail to satisfy faithfulness, a correction loop is automatically triggered to adjust the wording until the constraints are met. A faithfulness score $F_{score} = \alpha \cdot F_{struct} + (1-\alpha) \cdot F_{reason}$ is introduced to quantitatively evaluate faithfulness.
    - **Design Motivation**: Unconstrained textualization may introduce extra information or ambiguity, causing the LLM to exploit textual cues rather than genuine causal reasoning to get the answer, which would render the evaluation meaningless.

### Loss & Training
This work is primarily based on prompt engineering and does not involve model parameter training. In the faithfulness verification stage, an iterative optimization strategy is adopted, utilizing a maximum of 3 correction loop rounds to improve the faithfulness of textual descriptions. In the few-shot setting, exemplars containing different types of causal structures (chain, fork, collider) are selected to ensure coverage.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | GPT-4+FTI | GPT-4 Direct | Claude-3+FTI | Llama-3+FTI | Gain |
|------------|------|-----------|--------------|-------------|-------------|------|
| CLADDER (Causal Judgment) | Accuracy | 82.6 | 68.4 | 79.3 | 71.5 | +14.2 |
| CausalBench (Intervention) | Accuracy | 76.8 | 61.2 | 73.1 | 65.8 | +15.6 |
| CORR2CAUSE | Accuracy | 71.3 | 58.7 | 68.5 | 62.1 | +12.6 |
| Counterfactual Reasoning | Accuracy | 74.5 | 60.1 | 71.2 | 64.3 | +14.4 |

### Ablation Study

| Configuration | CLADDER Acc | CausalBench Acc | Note |
|------|-----------|----------------|------|
| Full FTI | 82.6 | 76.8 | Full framework |
| w/o CST | 72.1 | 66.3 | Without textualization, -10.5 |
| w/o SCRG | 76.3 | 70.2 | Without step-by-step guidance, -6.3 |
| w/o FBV | 80.1 | 74.5 | Without faithfulness verification, -2.5 |
| Symbol only | 68.4 | 61.2 | Pure symbolic input baseline |

### Key Findings
- Causal Structure Textualization (CST) contributes the largest performance gain (around 10 points), validating the core hypothesis that textualization activates causal knowledge.
- On complex causal structures involving confounders, the improvement of FTI is the most significant (+18.3%), indicating that textual descriptions help LLMs better understand intersecting causal pathways.
- Although the faithfulness verification mechanism yields a smaller performance boost (2-3 points), it effectively prevents about 15% of spurious correct answers (answers obtained via textual cues rather than causal reasoning).
- Smaller models (such as Llama-3-8B) achieve a larger relative improvement under the FTI framework (+15.2%), showing that FTI is more helpful for models with weaker causal reasoning capabilities.

## Highlights & Insights
- The concept of faithful textual interpretation is highly ingenious—it finds a sweet spot that leverages the expressiveness of natural language to activate pre-trained knowledge while preventing cheating through faithfulness constraints. This approach of "maximizing expressiveness under constraints" can be transferred to any LLM evaluation scenario involving formalized inputs.
- The design of bidirectional faithfulness verification is deeply insightful: structural faithfulness prevents information loss, while reasoning faithfulness prevents information addition. The two complement each other to form a complete fidelity guarantee.
- The observed phenomenon that "weaker models benefit more" hints at a practical value: FTI can enable smaller models to achieve performance levels close to larger models in causal reasoning scenarios.

## Limitations & Future Work
- The textualization process itself relies on a strong LLM to generate high-quality natural language descriptions, introducing a bootstrapping problem.
- For highly abstract causal relationships (such as macro-causal chains in economics), textualization may introduce human biases.
- The level of automation for faithfulness verification is limited, and computational costs scale up when the size of the causal graph is large.
- Future work could explore combining FTI with causal discovery tasks, utilizing the textualization capability of LLMs to assist in discovering causal relationships from data.

## Related Work & Insights
- **vs CLADDER (Jin et al., 2024)**: CLADDER proposed a standardized evaluation framework for causal reasoning but utilized symbolic inputs; this work demonstrates the superiority of textualization within the CLADDER framework.
- **vs CausalCoT (Jin et al., 2023)**: CausalCoT also utilized chain-of-thought prompts for reasoning but lacked faithfulness constraints, resulting in issues where answers were correct but the reasoning was flawed.
- **vs Causal Parrots (Zečević et al., 2023)**: This work questioned whether LLMs possess genuine causal reasoning capabilities; this paper demonstrates through the FTI framework that LLMs can indeed exhibit non-trivial causal reasoning abilities under appropriate guidance.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of faithful textual interpretation is novel, and the design of the faithfulness constraints is exquisite.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets and models, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and thoroughly explained motivations.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for LLM causal reasoning evaluation, offering strong methodological inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](mathneuro_math_reasoning_isolation.md)
- [\[ACL 2025\] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)
- [\[ACL 2025\] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)

</div>

<!-- RELATED:END -->
