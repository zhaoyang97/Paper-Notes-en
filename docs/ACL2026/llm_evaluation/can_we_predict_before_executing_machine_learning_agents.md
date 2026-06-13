---
title: >-
  [Paper Note] Can We Predict Before Executing Machine Learning Agents?
description: >-
  [ACL 2026][LLM Evaluation][ML Agent] This paper demonstrates that LLMs can serve as implicit "world models," predicting the quality of ML solutions based only on task descriptions, verified data reports…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "ML Agent"
  - "World Model"
  - "Predict-then-Verify"
  - "AutoML"
  - "MLE-Bench"
date: 2026-05-08
content_hash: 4b386f1d2f95b274
---

# Can We Predict Before Executing Machine Learning Agents?

**Conference**: ACL 2026  
**arXiv**: [2601.05930](https://arxiv.org/abs/2601.05930)  
**Code**: https://github.com/zjunlp/predict-before-execute  
**Area**: LLM Agent / Evaluation  
**Keywords**: ML Agent, World Model, Predict-then-Verify, AutoML, MLE-Bench

## TL;DR
This paper demonstrates that LLMs can serve as implicit "world models," predicting the quality of ML solutions based only on task descriptions, verified data reports, and code snippets (DeepSeek-V3.2-Thinking achieves 61.5% accuracy). Based on this, the authors construct ForeAgent, which replaces the Generate-Execute-Feedback loop of AIDE with a Predict-then-Verify loop, achieving a 6× speedup, 3.2× larger search space, and +6% Beat Ratio on MLE-Bench.

## Background & Motivation

**Background**: ML agents such as MLE-Bench, AutoMind, and AIDE follow the "Generate-Execute-Feedback" paradigm—generating code, executing training to obtain metrics, and refining based on feedback. However, a single training run in MLE-Bench often takes 9 hours, limiting agents to only a few candidates within a 12-hour time budget.

**Limitations of Prior Work**: (1) Execution Bottleneck—the majority of computation is wasted on executing sub-optimal candidates; (2) Narrow search space—tree search methods like AIDE are restricted by execution budgets, expanding only 1× nodes on average; (3) Existing pruning methods (Trirat 2025, Kulibaba 2025) mostly use unreliable heuristics (e.g., complexity scores) that risk pruning high-quality solutions.

**Key Challenge**: To enable agents to explore a broader solution space, the requirement of "executing every candidate" must be abandoned. However, without execution, how can one determine which candidates are worth running? Human experts use "mental simulation" to judge algorithm suitability after understanding tasks and data—can LLMs achieve similar capabilities?

**Goal**: (1) Formally define the "Data-centric Solution Preference" task and construct a large-scale evaluation; (2) Prove whether LLMs can reliably predict ML solution quality without execution; (3) Integrate this predictive capability into agents by replacing the Execute loop with a Predict-then-Verify loop.

**Key Insight**: Drawing from World Model concepts (Ha & Schmidhuber 2018, Hafner 2024), where RL agents use learned environment models for internal rollouts instead of real interactions. This paradigm is transferred to code execution: using the "execution prior" learned by LLMs for internal rollouts instead of real training.

**Core Idea**: Using a verified data analysis report (profiling data first, then having GPT-5.1 verbalize it into natural language insights, e.g., "Severe class imbalance, accuracy should not be used") as critical context. The LLM treats this semantic signal as input for its implicit world model to perform pairwise comparisons of solutions and provide confidence scores. Only candidates with high confidence enter real execution.

## Method

### Overall Architecture
The framework consists of two parts: (A) **Data-centric Solution Preference task + 18,438 evaluation pairs**—extracted from 1,329 valid solutions from AIDE/AutoMind trajectories on MLE-Bench, refined through deduplication, classification, and expert sampling to 895 instances, combined into 18,438 pairs with balanced ground-truth positions. (B) **ForeAgent**—using the AIDE tree search as a backbone, the Improvement stage is modified into three steps: (1) One-time generation of $m=10$ candidates; (2) Confidence-Gated Pairwise Selection using a 0.7 confidence threshold; (3) Verification Execution of only the top-k=1 candidate.

### Key Designs

1.  **Verified Data Analysis Report (Input Enhancement)**:
    *   **Function**: Addresses the weaknesses of LLMs in numerical processing and the inability to fit raw data into the context window by converting raw data into LLM-friendly semantic insights.
    *   **Mechanism**: A three-step Profile-Verify-Verbalize pipeline. (a) Code Generation: GPT-5.1 writes a Python profiling script; (b) Execution and Verification: The script runs in a sandbox with manual verification for runtime errors to obtain raw facts; (c) Verbalization: GPT-5.1 translates raw logs into actionable insights like "Data Imbalance Warning... consider using F1-score."
    *   **Design Motivation**: Ablation studies (Figure 3a) show accuracy increases: Code Only (56.7%) → Numerical Stats (59.0%) → Verbal Report (61.3%). Context Mismatch only reached 56.8%, proving LLMs rely on "data semantics + algorithm adaptation" rather than guessing based on code complexity. Verbal narratives outperform raw stats, suggesting LLMs act as "rhetorical reasoners."

2.  **Confidence-Gated Pairwise Preference + Calibration**:
    *   **Function**: Ensures the agent only skips execution when the model is "confident," guaranteeing the safety of the Predict-then-Verify loop.
    *   **Mechanism**: Given input $\mathcal{X}=(I, D_{rep}, \{C_0, C_1\}, \mathcal{P})$, the output is $\mathcal{Y}=(cot, \hat{y}, c)$, where $\hat{y}\in\{0,1\}$ is the predicted winner and $c\in[0,1]$ is the confidence score. ForeAgent uses $c=0.7$ as a gating threshold, reverting to execution when confidence is low.
    *   **Design Motivation**: Experiments confirm that LLMs do not assign high confidence randomly, which is a prerequisite for safe deployment. High calibration allows the filter to prune efficiently without discarding good solutions.

3.  **Predict-then-Verify Loop (Agent Integration)**:
    *   **Function**: Transforms the AIDE Execute main loop into a Predict main loop with an Execute auxiliary loop, where physical execution is used only for final verification.
    *   **Mechanism**: (1) High-Volume Generation: Parallel generation of $m=10$ candidates; (2) Confidence-Gated Pairwise Selection: Pairwise comparison using the implicit world model; (3) Verification Execution: Running only the $top-k=1$ candidate to anchor execution feedback. This yields an immediate $m \times$ speedup per improvement step.
    *   **Design Motivation**: The architecture is intentionally conservative (verifying only top-1) to ensure the trajectory is not misled by occasional LLM errors; the reported 6× speedup and +6% Beat Ratio represent a lower bound.

## Key Experimental Results

### Main Results — Solution Preference Task (Selected: DeepSeek-V3.2-Thinking)

| Dimension | Value | Acc (%) |
|------|------|---------|
| Domain | CV | 59.3 |
| Domain | NLP | **66.9** |
| Domain | Data Sci. | 57.4 |
| Difficulty | Easy | **63.9** |
| Difficulty | Medium | 60.4 |
| Difficulty | Hard | 57.0 |
| Algo Era | Traditional | **64.5** |
| Algo Era | Modern | 60.4 |
| Granularity | Cross-Algo | **62.8** |
| Granularity | Self-Comp. | 60.7 |
| Complexity | Low | 62.1 |
| Complexity | High | 59.6 (Complexity Tax) |
| **Avg (Total 18,438 pairs)** | | **61.5** |

Comparison: GPT-5.1 (58.8% global average), Random (50.0%), Complexity Heuristic (50.8%). Reasoning mode (CoT) achieved 61.3% vs. Direct Answer 55.9%.

### Ablation Study

| Experimental Dimension | Key Findings |
|----------|----------|
| Input Modality (Fig 3a) | Heuristic 50.8 → Code Only 56.7 → Numerical Stats 59.0 → **Verbal Report 61.3**; Context Mismatch only 56.8 |
| Listwise Ranking (Tab 3) | Acc@1 is 61.3% at N=2, but drops to 31.1% at N=5; Spearman $\rho \approx 0.23$ |
| Scaling (Qwen 4B → 1T, Fig 3c) | Saturation after 30B; no significant gain at 1T. Advantages of DeepSeek-V3.2 and GPT-5.1 stem from reasoning paradigms rather than parameter count. |
| Validation-Test Gap (Tab 4) | Training val metric Acc is 72.2% (hours); LLM inference Acc is 61.5% (seconds). |

### ForeAgent on 5 AI4Science Tasks in MLE-Bench

| Metric | AIDE baseline | ForeAgent | Gain |
|------|---------------|-----------|------|
| Avg Beat Ratio | base | **+6%** | Significant |
| Convergence Time | 12h | 2h (1/6 budget) | **6× Speedup** |
| Explored Nodes | 1× | **3.2×** | Wider Search |
| Test Improve Rate | base | +23% | Benefits Development |

## Key Findings
-   **LLMs can "mentally compute" algorithm adaptation**: Outperforming the random baseline by >10% is statistically significant, proving this is not a lucky pattern.
-   **Verbal Report is the core source of gain**: Raw stats are insufficient; they must be translated into "meaning" to trigger reasoning.
-   **Cognitive Boundaries exist**: Models perform better on NLP > CV > Data Sci.; Easy > Hard; Traditional > Modern. They excel at "coarse cross-algorithm comparison" but are weaker at "fine-tuning the same algorithm."
-   **Listwise Ranking is a weakness**: Pairwise 61% accuracy drops to 31% Acc@1 for a list of 5. LLMs lack global discrimination.
-   **Complexity Tax**: Accuracy drops by 4 points on complex code, suggesting LLMs may get lost in verbose scripts.

## Highlights & Insights
-   **Applying World Model paradigm to code/data**: While World Models are typically used in physical simulations, this is among the first works to use them as a code-execution prior, proving that existing reasoning LLMs are sufficient.
-   **Verified Data Report as a prompt engineering pattern**: Using LLMs to write profiling scripts and verbalize results avoids the inherent weakness of LLMs with raw numbers.
-   **Confidence-Gated Pruning**: A simple design using self-reported confidence allows safe deployment of an implicit world model without requiring RL fine-tuning for a reward model.
-   **Validation-Test Gap perspective**: Traditional validation metrics only have 72.2% accuracy due to distribution shift; the LLM's 61.5% is remarkably close despite taking only seconds.

## Limitations & Future Work
-   Corpus covers 26 tasks but is biased toward Classification/Regression; long-tail scientific tasks (e.g., Audio) are underrepresented.
-   Verified Data Reports in unstructured domains like CV/NLP rely on metadata and lack multimodal profiling.
-   ForeAgent uses a conservative top-1 verification; more aggressive strategies like batched verification have not been explored.
-   The weakness in listwise ranking limits applications in very large candidate pools.

## Related Work & Insights
-   **vs AIDE / AutoMind / MLE-Star**: Those rely entirely on real execution; this work pushes execution to the end using an implicit world model.
-   **vs CodeI/O / CRUXEval**: Those test forward execution prediction; this work predicts high-level solution quality relative to data.
-   **vs Trirat 2025 / Kulibaba 2025**: Those use complexity heuristics for pruning; this work uses semantic preference judgment.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ 
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
-   Writing Quality: ⭐⭐⭐⭐ 
-   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](../../ICML2026/llm_evaluation/hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ICML 2026\] Who can we trust? LLM-as-a-jury for Comparative Assessment](../../ICML2026/llm_evaluation/who_can_we_trust_llm-as-a-jury_for_comparative_assessment.md)

</div>

<!-- RELATED:END -->
