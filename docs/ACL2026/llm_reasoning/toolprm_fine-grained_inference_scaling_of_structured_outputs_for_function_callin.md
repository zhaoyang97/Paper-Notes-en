---
title: >-
  [Paper Note] ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling
description: >-
  [ACL2026][LLM Reasoning][Function Calling] ToolPRM decomposes function calls into fine-grained decisions such as function name selection, parameter name selection…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Function Calling"
  - "Process Reward Model"
  - "Structured Output"
  - "Inference Scaling"
  - "Beam Search"
date: 2026-05-08
content_hash: cd3dded216ffec3d
---

# ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling

**Conference**: ACL2026  
**arXiv**: [2510.14703](https://arxiv.org/abs/2510.14703)  
**Code**: To be confirmed  
**Area**: llm_reasoning / Tool Use  
**Keywords**: Function Calling, Process Reward Model, Structured Output, Inference Scaling, Beam Search

## TL;DR
ToolPRM decomposes function calls into fine-grained decisions such as function name selection, parameter name selection, and parameter value filling. It trains an intra-call process reward model to guide beam search and proposes the "explore more but retain less" principle for inference scaling of structured outputs, achieving stable improvements for the Hammer2.1 series tool-calling models on BFCL and ToolAlpaca.

## Background & Motivation
**Background**: Inference-time scaling has been widely utilized in unstructured generation tasks such as mathematical and logical reasoning (e.g., self-consistency, Best-of-N, Tree-of-Thought, beam search, or MCTS). These methods typically rely on outcome reward models or process reward models to score and filter multiple candidate reasoning paths.

**Limitations of Prior Work**: Function calling involving structured outputs requires models not only to generate natural language but also to select correct function names, fill in accurate parameter names/values, and maintain valid JSON or Python-style syntax. Existing inference scaling methods mostly treat a single function call as a holistic candidate for scoring. This granularity is too coarse to perform timely pruning when early errors occur, such as choosing the wrong function name or an incorrect parameter value.

**Key Challenge**: In unstructured reasoning, early errors can sometimes be recovered through subsequent reflection or correction. However, structured outputs in function calling are typically irrecoverable; a single incorrect function name or parameter invalidates the entire trajectory. Consequently, structured outputs require broader exploration to find correct decisions but cannot afford to retain too many erroneous candidates that continue to consume the computational budget.

**Goal**: The authors aim to construct the first fine-grained process supervision dataset oriented towards intra-call decisions for function calling. They train ToolPRM to score each local decision and use it to guide search, allowing small tool-calling models to achieve higher accuracy through additional test-time computation.

**Key Insight**: The paper decomposes a function call into a state transition process: first selecting the function name, then sequentially selecting parameter names and filling parameter values, and finally determining if parameters and the function call have ended. This way, the PRM does not have to wait for the complete JSON generation to score, but can judge the correctness of each local action.

**Core Idea**: Use function masking and rollouts to automatically collect fine-grained positive and negative step labels. A generative reward model is trained to output "+" or "-". During beam search, the candidate width per step is increased while the number of retained beams is reduced, thereby concentrating the computational budget on correct structural paths.

## Method

### Overall Architecture
The ToolPRM pipeline consists of three steps. The first step is constructing fine-grained supervision data: natural language queries and target function calls are obtained from xLAM-function-calling-60k and xLAM-irrelevance-7.5k. Function and parameter names are masked to force the model to understand tools based on descriptions rather than memorized names. Hammer2.1-3B/7B is used as the policy model to rollout candidate function calls.

The second step is the automatic labeling of local decisions within each candidate trajectory. The paper defines multiple labels: whether the function name is correct, whether a parameter name-value pair is correct, whether all parameters are filled, whether a single function call is correct, and whether the overall response is correct. Each label is obtained via binary positive/negative supervision through exact matching with the ground truth.

The third step involves training ToolPRM and using it for inference scaling. ToolPRM itself is an LLM that takes the current state and candidate action as input and outputs the probability of "+" or "-". In each step of the beam search, multiple candidates are expanded, and ToolPRM scores are used for ranking and pruning. Given the irrecoverable nature of early errors in structured outputs, the authors advocate for increasing the beam width to explore more local choices while retaining fewer beams, following an "explore more but retain less" strategy.

### Key Designs
1. **Intra-call Fine-Grained Decomposition and Labeling System**:
    - **Function**: Decomposes a function call from a holistic JSON result into supervisable local decision steps.
    - **Mechanism**: Each function call includes labels for function name selection, parameter name selection, parameter value filling, parameter end, function end, and overall completion. For example, `<FUNC_NAME>` judges the correctness of the function name, while `<ARG_VALUE>` judges if a parameter name and value match the ground truth. `<TOTAL_FINISH>` evaluates the correctness of the entire list of function calls.
    - **Design Motivation**: Looking only at the final result masks the source of errors. Fine-grained labels allow the reward model to detect errors earlier and provide hierarchical supervision. Experiments demonstrate that retaining seemingly redundant labels like `<ARG_VALUE>` improves trajectory-level judgment accuracy.

2. **Function Masking + Rollout Data Construction**:
    - **Function**: Constructs robust training data for the reward model to prevent it from relying on memorized tool names.
    - **Mechanism**: Function names and parameter identifiers are replaced with random strings. The policy model is given a set of masked function candidates to generate rollouts. Subsequently, exact match is used to label each step as positive or negative, forming step-level and trajectory-level data.
    - **Design Motivation**: In real-world deployments, tool names and schemas may change. If a reward model only memorizes tool names, its generalization will be poor. Masking forces it to focus on the query, function descriptions, parameter semantics, and the generated structural context.

3. **ToolPRM-Guided Fine-Grained Beam Search**:
    - **Function**: Increases structured function calling accuracy at test time using additional computation.
    - **Mechanism**: For each candidate action, ToolPRM outputs the logits for the labels "+" and "-". The local correctness score is calculated as $s=e^{s_+}/(e^{s_+}+e^{s_-})$. During search, the top-$N$ beams are retained, and each beam expands $M$ subsequent candidates. The authors emphasize increasing $M$ to expand exploration while keeping $N$ small to prevent early erroneous decisions from occupying the budget.
    - **Design Motivation**: Unlike free-form text, incorrect JSON branches in function calls are difficult to fix later. Pruning incorrect candidates early allows the budget for subsequent parameter filling to be spent on correct structures.

### Loss & Training
ToolPRM employs generative process reward modeling. Given a trajectory $\mathcal{T}=\{(s_t,a_t,r_t)\}$ where $r_t\in\{+,-\}$, the training objective is to maximize the probability of the correct label token, i.e., minimize $-\log p_\theta(r_t|s_t,a_t)$. Hammer2.1-3B is used as the reward model backbone, trained with SFT for 5 epochs using the Adam optimizer, a batch size of 1024, a learning rate of $1e-3$, a warmup ratio of 0.008, and weight decay of $1e-5$. For inference, the temperature is set to 0.8, and the number of beams $N$ and beam width $M$ are selected from $\{1,2,4,8,16\}$.

## Key Experimental Results

### Main Results
The ToolPRM dataset is large-scale and reports both step and trajectory granularities. The following is adapted from Table 1 of the paper.

| Sample Granularity / Split | Positive | Negative | Total |
|------------------|----------|----------|-------|
| Step / Train | 4,380,323 | 731,665 | 5,111,988 |
| Trajectory / Train | 466,786 | 127,648 | 594,434 |
| Step / Test | 488,611 | 81,366 | 569,977 |
| Trajectory / Test | 52,030 | 14,019 | 66,049 |

Reward model prediction accuracy shows that finer supervision granularity leads to better judgment of the complete function call trajectory. ToolPRM outperforms outcome-only and coarse PRM in terms of loss, step accuracy, and trajectory accuracy.

| Reward Model | Loss ↓ | Step Acc ↑ | Trajectory Acc ↑ |
|--------------|--------|------------|------------------|
| ORM | 0.0536 | 98.39% | 98.39% |
| C-PRM | 0.0371 | 98.87% | 99.06% |
| ToolPRM | 0.0286 | 99.11% | 19.38% |

### Ablation Study
Inference scaling results indicate that ToolPRM is more stable than token-level beam search, majority voting, and Best-of-N, with more pronounced benefits for smaller models.

| Policy Model | Method | BFCL Avg. | ToolAlpaca Avg. | Main Conclusion |
|-------------|------|-----------|-----------------|----------|
| Hammer2.1-7B | Base | 88.65 | 72.77 | Strong base is already high |
| Hammer2.1-7B | + ToolPRM | 89.52 | 73.36 | Small but stable improvement |
| Hammer2.1-3B | Base | 86.86 | 71.57 | Close to 7B base |
| Hammer2.1-3B | + ToolPRM | 88.88 | 71.96 | BFCL close to 7B + ToolPRM |
| Hammer2.1-1.5B | Base | 82.79 | 69.30 | Smaller models have more structural errors |
| Hammer2.1-1.5B | + ToolPRM | 85.61 | 72.93 | Most significant gain, approaching/exceeding larger bases |

Budget analysis further validates the "explore more but retain less" principle: when fixing the number of retained beams $N=4$ and increasing the beam width $M$, the ToolAlpaca F1 score typically rises as the budget increases. Conversely, fixing $M=4$ and increasing $N$ yields smaller gains and sometimes even a decrease in performance. This suggests that retaining more candidates in structured output tasks is not always beneficial, as incorrect candidates can misguide the subsequent budget.

### Key Findings
- Fine-grained supervision improves both step-level accuracy and trajectory-level accuracy, suggesting that local labels do not cause the model to focus solely on local details but instead help it better judge overall correctness.
- ToolPRM offers greater marginal utility for small models. After adding ToolPRM to the 1.5B Hammer, BFCL Avg. increased from 82.79 to 85.61, and ToolAlpaca Avg. increased from 69.30 to 72.93, reaching or exceeding the performance of some larger models.
- Standard inference scaling methods are unstable. Token-level beam search performed worse than the base model in several settings because early errors in structured generation are irrecoverable.

## Highlights & Insights
- The paper identifies the fundamental difference between structured outputs and free-form text reasoning: while mathematical reasoning can retain multiple intermediate thoughts for later correction, function calling is more akin to program construction, where early schema decisions are difficult to rectify.
- "Explore more but retain less" is a highly practical principle for inference scaling. It is more specific than simply increasing test-time compute, pointing out that the budget should be spent on lateral exploration at each step rather than retaining numerous historical error branches.
- ToolPRM's data construction methodology has transfer value. Techniques such as function masking, step-level exact-match labeling, and generative positive/negative label prediction can be extended to tasks like SQL generation, workflow orchestration, and robotic action parameterization.

## Limitations & Future Work
- ToolPRM discretizes function calling into explicit states and labels, but real models may have internal implicit reasoning and uncertainty that these states might not fully cover.
- The current method requires an additional reward model, masking rules, and state definitions, making its engineering complexity higher than simple Best-of-N. The quality of label construction across different API schemas directly impacts performance.
- The optimal $M/N$ trade-off for "explore more but retain less" currently relies on grid search and has not been adaptively adjusted based on input complexity or ToolPRM confidence.
- Automatic labeling relies on ground truth exact matches, which may underestimate calls that are semantically equivalent but formatted differently, and might not cover side effects, permissions, or runtime failures in real tool environments.

## Related Work & Insights
- **vs ORM / Best-of-N**: ORM only evaluates final candidates, suitable for full answer filtering. ToolPRM can prune early during function name and parameter value stages, making it better suited for structured outputs where errors are irrecoverable.
- **vs coarse PRM**: C-PRM is finer than ORM but removes parameter-level labels like `<ARG_VALUE>`. ToolPRM retains the finest steps, resulting in higher trajectory accuracy.
- **vs General Beam Search / Majority Voting**: These methods increase candidate volume but lack structured local rewards, often retaining incorrect JSON branches. ToolPRM's value lies in using process rewards to determine which branches are worth continuing.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Pinpoints the intra-call level for PRMs and summarizes structured inference scaling principles.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Data statistics, granularity comparisons, BFCL/ToolAlpaca main experiments, and budget analysis are comprehensive, though real multi-turn tool environments could be further explored.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with sufficient explanations for the architecture and state transitions; minor issues in terminology and table formatting.
- Value: ⭐⭐⭐⭐⭐ Highly practical for function calling and agent engineering, particularly for enhancing small edge models with additional inference budget.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DVMap: Fine-Grained Pluralistic Value Alignment via High-Consensus Demographic-Value Mapping](dvmap_fine-grained_pluralistic_value_alignment_via_high-consensus_demographic-va.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)
- [\[AAAI 2026\] Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning](../../AAAI2026/llm_reasoning/small_language_models_for_efficient_agentic_tool_calling_outperforming_large_mod.md)
- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](../../ICML2026/llm_reasoning/uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)

</div>

<!-- RELATED:END -->
