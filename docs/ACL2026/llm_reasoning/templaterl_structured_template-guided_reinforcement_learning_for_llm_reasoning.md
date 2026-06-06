---
title: >-
  [Paper Note] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Reinforcement Learning] TemplateRL abstracts structured reasoning templates from a small-scale seed set using MCTS and introduces these templates as explicit guidance during reinforcement learni…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Reinforcement Learning"
  - "Template Guidance"
  - "Reasoning Paths"
  - "LLM Optimization"
  - "GRPO"
date: 2026-05-08
content_hash: 695377a987054553
---

# TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning

**Conference**: ACL 2026  
**arXiv**: [2505.15692](https://arxiv.org/abs/2505.15692)  
**Code**: https://github.com/THU-KEG/TemplateRL  
**Area**: LLM Reasoning / Reinforcement Learning  
**Keywords**: Reinforcement Learning, Template Guidance, Reasoning Paths, LLM Optimization, GRPO

## TL;DR

TemplateRL abstracts structured reasoning templates from a small-scale seed set using MCTS and introduces these templates as explicit guidance during reinforcement learning training. This significantly enhances the efficiency and stability of multi-step reasoning in LLMs, achieving a 99% improvement on AIME compared to GRPO.

## Background & Motivation

**Bottlenecks of LLM Reasoning Performance**: While reinforcement learning has proven effective for enhancing LLM reasoning capabilities (e.g., o1, DeepSeek-R1), existing methods like GRPO primarily rely on unstructured self-sampling. This requires models to explore blindly and learn from scalar reward signals, leading to three critical issues: (1) Low sampling efficiency, where high-quality trajectories are rarely hit, causing training to collapse on weaker models; (2) Difficulty in learning transferable high-level strategies, as models tend to memorize surface steps rather than distilling general divide-and-conquer or step-by-step thinking patterns; (3) Lack of interpretability, as the reasoning process lacks an explicit strategic structure, making diagnosis and intervention difficult.

**Inspiration from Human Reasoning**: Cognitive psychology research (Kahneman 2011) indicates that humans do not solve complex problems from scratch but apply "generic guidelines" (templates) induced from similar problems. These high-level templates help humans adapt quickly to new problems.

**Key Observation & Design Motivation**: For multi-step reasoning tasks, the probability of a model generating a single correct step is significantly higher than completing the entire reasoning chain at once. Consequently, an explicit template library can be constructed to adaptively retrieve relevant templates during RL training, guiding policy generation around these templates. This provides structured strategic guidance while allowing the model to learn general reasoning approaches.

**Core Idea**: Replace unstructured exploration with a human-inspired template library, decomposing the RL learning process into multiple template-guided sub-goal optimizations to enhance sampling quality, model stability, and reasoning interpretability.

## Method

### Overall Architecture

TemplateRL consists of three stages:

**Phase 1 — Template Library Construction**: MCTS is used on a small seed set (500 problems) to generate multiple solution paths. The optimal path for each problem is filtered (balancing accuracy and complexity), and the action sequence is abstracted as a template. These are clustered by problem complexity features to form a structured template library.

**Phase 2 — Template-Guided Training**: For each training problem, its complexity is calculated and matched with templates in the library to select the $k$ most similar ones. For each template, the model generates reasoning trajectories step-by-step according to the template's action sequence. The sampling results from these $k$ templates are aggregated for multi-group GRPO optimization, where each template corresponds to an optimization sub-goal, decomposing single scalar reward learning into structured pattern learning.

**Phase 3 — Optional Dynamic Expansion**: If new correct reasoning paths are discovered during training or inference, their action sequences are automatically extracted and added to the template library, continuously enriching its coverage.

### Key Designs

1.  **MCTS-based Template Construction and Complexity Awareness**:
    - **Function**: Automatically abstract high-level reasoning strategies from small-scale high-quality examples.
    - **Mechanism**: For each seed problem, MCTS constructs multiple solution trees where each edge represents an "action" (e.g., "pose a sub-question", "derive next step"). For each path in the tree, a scoring function $\text{Score}(\mathbf{s}_i, \mathbf{t}_{i,j}) = b \cdot R(\mathbf{t}_{i,j}|\mathbf{s}_i) - (1-b) \cdot C(\mathbf{t}_{i,j})$ balances correctness and complexity to filter the best path. All action sequences are then clustered by "Pre-condition Condition Complexity" (PCC, the number of prerequisites in the problem) to form the template library $\mathcal{L} = \{\hat{T}_1, \ldots, \hat{T}_{|\mathcal{L}|}\}$, storing the action sequence and average PCC.
    - **Design Motivation**: PCC serves as a proxy for problem difficulty. Grouping by PCC allows subsequent retrieval to find templates matching the difficulty of new problems, avoiding guidance from overly simple or complex templates.

2.  **Adaptive Template Retrieval and Multi-group RL Optimization**:
    - **Function**: Dynamically select appropriate templates based on problem complexity during training and isolate optimization objectives for different templates.
    - **Mechanism**: For a training problem $\mathbf{q}$, its PCC is calculated, and the distance to a template $\hat{T}_j$ is defined as $d(\mathbf{q}, \hat{T}_j) = |{\rm PCC}(\mathbf{q}) - {\rm PCC}_{T_j}|$. The top-$k$ nearest templates are selected. For each template $T_i$, $G_i$ trajectories are sampled following the action sequence. Multi-group trajectories are merged into the GRPO loss: $\tilde{\mathcal{J}}_{\text{GRPO}}(\pi_\theta) = \frac{1}{\sum_i G_i} \sum_i \sum_j \sum_t \min[\rho_{i,j,t} A_{i,t}, \hat{\rho}_{i,j,t} A_{i,t}]$, where $\rho$ is the probability ratio and $A$ is the advantage estimate. This is equivalent to defining sub-goals $\mathcal{J}_i(\pi_\theta)$ for each template, with the overall goal being a weighted average.
    - **Design Motivation**: (i) Separating templates enables the model to learn across different strategic patterns, preventing diversity from being drowned out by a single reward signal; (ii) The paper proves two theoretical properties: Prop 3.1 states that multi-grouping increases the probability of obtaining at least one positive trajectory, and Prop 3.2 states that template transfer for similar problems improves success probability.

3.  **Dynamic Template Library Expansion**:
    - **Function**: Allow the template library to evolve during training and inference rather than remaining a static initial set.
    - **Mechanism**: During training, for correct generated trajectories, action sequences $T' = (a_1', \ldots, a_d')$ are parsed via keyword extraction or lightweight models and added to the library. At inference, multiple paths are generated using 5 templates for each sample, the answer is selected via majority voting, and new templates are extracted from the results to be added to the library for subsequent samples. This forms a continuous learning loop.
    - **Design Motivation**: The template library is not static. By absorbing new strategies from successful examples, the model can accumulate reasoning skills in specific domains, which is particularly beneficial for fields requiring iterative updates (e.g., medical reasoning).

## Key Experimental Results

### Main Results

| Method | MATH500 ↑ | AIME24 ↑ | AMC ↑ | Minerva ↑ | Olympiad ↑ | Average ↑ |
|------|----------|---------|-------|-----------|-----------|--------|
| Qwen2.5-Math-7B-Base | 50.8 | 13.3 | 42.5 | 12.1 | 17.2 | 27.2 |
| SimpleRL-Zero | 74.6 | 26.7 | 60.0 | 27.6 | 35.8 | 44.9 |
| Oat-Zero | 79.6 | 30.0 | 60.0 | 34.2 | 39.9 | 48.7 |
| **GRPO (Baseline)** | **76.2** | **16.7** | **55.0** | **32.7** | **38.1** | **43.8** |
| **TemplateRL (Ours)** | **83.4** | **33.3** | **77.5** | **38.2** | **46.2** | **55.8** |
| **Relative Gain** | **+9.4%** | **+99.4%** | **+40.9%** | **+16.8%** | **+21.2%** | **+27.4%** |

TemplateRL outperforms the GRPO baseline across all benchmarks, with the most significant gain on AIME24 (+99.4%), indicating that template guidance is most helpful for complex reasoning problems.

### Ablation Study

| Experiment | Conclusion |
|------|------|
| Training Stability (Llama-3.2-3B) | GRPO rewards collapsed to 0 after 100 steps; TemplateRL remained consistently > 0.25 |
| Cross-domain Generalization (BALROG/GPQA-D/MMLU-Pro) | Average improvement of 6%+ over GRPO |
| Multimodal Extension (Qwen2.5-VL) | Average +8.4% on MathVision/MathVerse/MMMU/BLINK |
| Dynamic Template Expansion (Training) | AIME24 increased from 33.3% to 36.7% (+10.2%) |
| Dynamic Template Expansion (Inference) | GPQA-D increased from 37.9% to 40.4% (+6.6%) |
| Number of Template Groups $\|g\|$ | $\|g\|=2$ provided the optimal balance |

## Highlights & Insights

-   **Human-inspired Design with Theoretical Support**: Moving from cognitive psychology to replace unstructured exploration with templates is a natural approach. Importantly, the paper provides two theoretical propositions (Prop 3.1, 3.2), proving that multi-grouping increases positive sample probability and that template transfer for similar problems improves success rates.
-   **Comprehensive Experimental Design and Strong Results**: Validation is performed not only on math competitions (AIME +99%) but also across different model scales (1.5B–8B), architectures (Qwen/Llama), and modalities. Cross-domain generalization experiments (BALROG, GPQA-D) also prove the method is not just overfitted to the mathematical domain.
-   **Dynamic Expansion Enhances Utility**: Unlike many RL works with fixed policies, TemplateRL supports online updates to the template library, which is highly valuable for scenarios requiring continuous adaptation (e.g., medical reasoning, scientific discovery).

## Limitations & Future Work

**Limitations**:
-   Template library initialization relies on MCTS exploration and manual action space definitions. It remains unclear if action spaces need redefinition for different tasks.
-   Experiments primarily focused on mathematical and logical reasoning. Performance on other long-chain reasoning tasks (e.g., code generation, scientific experiment design) needs verification.
-   While the paper claims improved interpretability, no quantitative evaluation of interpretability was provided.

**Future Work**:
-   Automatic discovery of action spaces: Instead of manual definition, can task-general "action" concepts be learned automatically from correct trajectories?
-   Broader application exploration: Extending to code synthesis, scientific reasoning, and dialogue generation.
-   Interpretability analysis of templates: Quantitatively measuring the level of abstraction and coverage of the learned templates.

## Related Work & Insights

-   **vs GRPO/RL Baselines**: These methods improve at the algorithmic level (e.g., handling length bias, KL constraints) but still rely on unstructured self-sampling. TemplateRL breaks this bottleneck through explicit structure, representing an architectural innovation.
-   **vs Inference-time Template Methods** (RAG, decomposition): Prior works used templates during inference but did not integrate them into the RL training itself. TemplateRL's novelty lies in the unified training-inference template guidance mechanism.
-   **Insight**: This work demonstrates the power of combining "human-inspired structure + RL optimization." Similar approaches could be applied to other tasks requiring "high-level strategies" such as planning and multi-agent collaboration.

## Rating

-   **Novelty**: ⭐⭐⭐⭐⭐ Introducing structured templates into RL training is a fresh perspective with clear theoretical support and a natural, ingenious human-inspired design.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple model scales, architectures, domains, and modalities, with comprehensive ablations and stability verification.
-   **Writing Quality**: ⭐⭐⭐⭐ Generally clear with well-marked theoretical sections, though the discussion on action space definitions could be deeper.
-   **Value**: ⭐⭐⭐⭐⭐ As GRPO is a mainstream RL method, a 99% improvement is substantial. Stability improvements are very practical for weaker models, and cross-domain generalization proves the method's universality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](../../ICLR2026/llm_reasoning/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ACL 2026\] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS](evo-attacker_memory-augmented_reinforcement_learning_for_long-horizon_tool_attac.md)
- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](../../NeurIPS2025/llm_reasoning/expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](../../ICML2026/llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)

</div>

<!-- RELATED:END -->
