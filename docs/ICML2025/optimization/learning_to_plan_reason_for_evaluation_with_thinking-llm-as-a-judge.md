---
title: >-
  [Paper Note] Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge
description: >-
  [ICML 2025][Optimization][LLM-as-a-Judge] This paper proposes EvalPlanner, which decouples the reasoning process of LLM-as-a-Judge into two phases: "evaluation plan generation" and "plan execution". By iteratively optimizing plan and execution preference pairs using DPO in a self-training loop, it achieves a new generative reward model State-of-the-Art (SOTA) score of 93.9 on RewardBench with only 22K synthetic preference pairs.
tags:
  - "ICML 2025"
  - "Optimization"
  - "LLM-as-a-Judge"
  - "Preference Optimization"
  - "Chain-of-Thought"
  - "Evaluation Planning"
  - "Self-Training"
date: 2026-05-08
content_hash: 9c88cc3fcfb24cdb
---

# Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge

**Conference**: ICML 2025  
**arXiv**: [2501.18099](https://arxiv.org/abs/2501.18099)  
**Code**: None  
**Area**: Optimization  
**Keywords**: LLM-as-a-Judge, Preference Optimization, Chain-of-Thought, Evaluation Planning, Self-Training

## TL;DR

This paper proposes EvalPlanner, which decouples the reasoning process of LLM-as-a-Judge into two phases: "evaluation plan generation" and "plan execution". By iteratively optimizing plan and execution preference pairs using DPO in a self-training loop, it achieves a new generative reward model State-of-the-Art (SOTA) score of 93.9 on RewardBench with only 22K synthetic preference pairs.

## Background & Motivation

The LLM-as-a-Judge paradigm utilizes LLMs themselves as evaluators to replace and mitigate expensive human evaluations, providing evaluation rationales through the generation of Chain-of-Thought (CoT). Such models can also serve as generative reward models, playing a critical role in iterative preference optimization and self-improvement training.

However, existing methods face core challenges:

**Lack of human-annotated evaluation CoTs**: Human preference annotation data typically contains only the final judgment without the reasoning process, leading to insufficient research on the structure and components of effective reasoning chains.

**Evaluation reasoning is constrained to handcrafted components**: Prior work typically restricts CoT to predefined evaluation criteria lists, reference answers, or verification questions, failing to adaptively handle different types of tasks (e.g., the evaluation criteria for essays vs. math problems are completely different).

**Entanglement of planning and reasoning**: Existing methods mix "determining evaluation criteria" and "executing evaluation" in the same generation process, lacking a clear separation of phases.

These limitations lead to insufficient generalization capability of evaluation models when facing diverse and complex instructions. The core motivation of EvalPlanner is that **evaluation is essentially a planning + reasoning problem**: first determine the evaluation scheme (plan), then step-by-step execute the evaluation based on the scheme (execute), and finally provide the judgment (verdict).

## Method

### Overall Architecture

EvalPlanner explicitly decouples the CoT of Thinking-LLM-as-a-Judge into three components:

1. **Evaluation Plan $z$**: An evaluation scheme generated solely based on the input instruction $x$ (without looking at candidate responses), prescribing the "recipe" for evaluation—which dimensions to check, what criteria to use, etc.
2. **Plan Execution $e$**: Under the conditions of a given plan $z$, instruction $x$, and response pair $(a, b)$, step-by-step executing each evaluation step in the plan to analyze the quality of the two responses.
3. **Final Verdict $y$**: Outputting which response is better based on the execution results.

Formally, the generation process of the verdict $y$ is modeled as:

$$p_\theta(y|x,a,b) = \sum_{z \in \mathcal{P}} \sum_{e \in \mathcal{E}} p_\theta(y|e,z,x,a,b) \cdot p_\theta(e|z,x,a,b) \cdot p_\theta(z|x)$$

Key design choice: **Plan generation is conditioned solely on the instruction $x$ and does not depend on the response pair**, ensuring that the plan only describes the evaluation scheme rather than executing the actual evaluation, thereby achieving stage separation.

### Key Designs

#### 1. Synthetic Training Data Generation

Due to the lack of human-annotated CoTs, EvalPlanner is trained entirely on synthetic data:

- **Prompt Selection**: Select instructions from WildChat (general instruction-following) and MATH (mathematical reasoning).
- **Response Pair Construction**:
    - General tasks: Modify the original instruction to a "noisy instruction", generate responses for both the original and noisy instructions, and pair them to form chosen/rejected pairs.
    - Mathematical tasks: Sample multiple solutions, where the correct answer is chosen and the incorrect answer is rejected.
- **Plan Generation**: Use a general, unconstrained plan generation prompt to allow the seed model (e.g., Llama-3.1-70B-Instruct) to freely generate an evaluation plan based on the input instruction without presetting any component structures.
- **Execution Generation**: Given the plan and response pairs, the seed model executes the plan to generate evaluation reasoning and the verdict.

#### 2. Preference Pair Construction

For each instruction:
- Sample $|\mathcal{P}|=5$ plans.
- Sample $|\mathcal{E}|=8$ executions for each plan (4 times for each response order, handling both $(a,b)$ and $(b,a)$ orders to eliminate positional bias).
- Generate $2 \times 5 \times 8 = 80$ CoTs in total.

Correctness criteria: If the (plan, execution, verdict) triplet derives the correct judgment, it is labeled as chosen; otherwise, it is labeled as rejected. For each plan, pair all correct and incorrect executions to construct the preference training data.

#### 3. Unconstrained Plan vs. Constrained Plan

One of the core innovations: EvalPlanner uses general unconstrained plan generation prompts, allowing the model to autonomously determine evaluation dimensions and methods rather than predefining a "mandatory list of evaluation criteria" or "mandatory verification questions". Experiments show that the unconstrained plan outperforms the constrained plan in all settings.

#### 4. Advantages of Decoupled Planning and Execution

- **Execution Faithfulness**: The execution phase is constrained to follow the plan, enhancing consistency.
- **Data Diversity**: Multiple plans can be sampled for the same instruction, and multiple executions can be sampled for each plan, diversifying training data in both the planning and execution dimensions.

### Loss & Training

EvalPlanner adopts a three-stage self-training loop:

**Stage 1: SFT ($\mathcal{M}_1^{\text{SFT}}$)**
- Starting from the seed model $\mathcal{M}_0$.
- On 5K instructions, randomly select one chosen CoT per instruction for Supervised Fine-Tuning.
- The goal is to let the model learn the output format of plan + execution + verdict.

**Stage 2: First-Round DPO ($\mathcal{M}_1^{\text{DPO}}$)**
- Initialized from $\mathcal{M}_1^{\text{SFT}}$.
- Perform DPO on $\mathcal{D}_1$ (preference pairs generated from 5K instructions).
- The model learns to contrast correct and incorrect (plan, execution) combinations.

**Stage 3: Second-Round DPO ($\mathcal{M}_2^{\text{DPO}}$)**
- Initialized from $\mathcal{M}_1^{\text{DPO}}$.
- Use a new subset of 17K instructions, sample from $\mathcal{M}_1^{\text{DPO}}$ itself to generate new CoT preference pairs $\mathcal{D}_2$.
- Run another round of DPO.

Key training hyperparameters:
- Maximum training steps: 1K. Save checkpoints every 100 steps, using early stopping based on the validation set.
- Sampling temperature: 0.8, top_p: 0.95.
- Validation set: 150 instructions each from WildChat and MATH, forming 600 pairs with bidirectional permutation.
- Inference temperature: 0, maximum generation tokens: 2048.

Advantages of iterative DPO: Training with higher-quality CoT data generated by the updated model in the second round achieves better performance than training on all data at once.

## Key Experimental Results

### Main Results

**RewardBench Results (Table 1)**:

| Model | Preference Pairs | Overall | Chat | Chat-Hard | Safety | Reasoning |
|------|---------|---------|------|-----------|--------|-----------|
| Llama-3.1-70B-Instruct | - | 84.0 | 97.2 | 70.2 | 82.8 | 86.0 |
| GPT-4o | - | 86.7 | 96.1 | 76.1 | 88.1 | 86.6 |
| Self-Taught Evaluator | 20K | 90.0 | 96.9 | 85.1 | 89.6 | 88.4 |
| Skywork-Critic-70B | 80K | 93.3 | 96.6 | 87.9 | 93.1 | 95.5 |
| **EvalPlanner (3.1-70B)** | **22K** | **93.9** | **97.5** | **89.4** | **93.0** | **95.5** |
| **EvalPlanner (3.3-70B)** | **22K** | **93.8** | **97.7** | **89.5** | **91.7** | **96.1** |

**PPE Results (Table 2)**:

| Model | PPE Overall | PPE Preference | PPE Correctness Overall |
|------|-------------|----------------|------------------------|
| GPT-4o | 62.3 | 67.1 | 57.6 |
| DeepSeek-GRM-27B (237K) | 62.2 | 64.7 | 59.8 |
| **EvalPlanner (3.3-70B, 22K)** | **67.9** | **65.6** | **70.2** |

### Ablation Study

**Effect of Iterative DPO (Table 4)**:

| Configuration | Preference Pairs | Accuracy | Description |
|------|---------|----------|------|
| 1-round DPO | 5K | 92.3 | Baseline |
| 1-round DPO | 22K | 92.5 | Doubling data gains only +0.2 |
| 2-round DPO (Iterative) | 5K+17K | **93.9** | Iterative optimization significantly gains +1.6 |

**FollowBenchEval Multi-Constraint Evaluation (Table 5)**:

| Model | Overall | L1 | L2 | L3 | L4 | L5 |
|------|---------|-----|-----|-----|-----|-----|
| Skywork-Critic-70B | 52.2 | 63.8 | 57.1 | 48.7 | 46.2 | 48.5 |
| **EvalPlanner (3.3-70B)** | **65.4** | **72.3** | **73.8** | **66.7** | **61.5** | **57.6** |

**RM-Bench Robustness (Table 6)**:

| Model | Overall | Easy | Normal | Hard |
|------|---------|------|--------|------|
| Skywork-Critic-70B | 74.1 | 76.3 | 72.9 | 73.1 |
| **EvalPlanner (3.3-70B)** | **82.1** | **81.1** | **80.8** | **84.3** |

### Key Findings

1. **Data Efficiency**: Only 5K synthetic preference pairs are needed to reach 92.3 (already close to SOTA); 22K synthetic data surpasses previous methods that use 80K-680K human-annotated data.
2. **Iterative Optimization is Key**: Two-round DPO performs better than one-round DPO with more data, proving that new data generated by the updated model is more valuable than historical data.
3. **Small Models Also Benefit**: The 8B version of EvalPlanner reaches 83.0 on RewardBench, closely performing to Llama-3.1-70B-Instruct (84.0) and Claude-3.5-Sonnet (84.2).
4. **Significant Advantage in Multi-Constraint Evaluation**: Outperforming Skywork-Critic by 13 points on FollowBenchEval demonstrates that explicit planning is particularly effective for complex constraint tasks that require item-by-item verification.
5. **Unconstrained Plans Outperform Constrained Plans**: General plan prompts generalize better than predefined "criteria lists" or "verification question" formats.
6. **Outstanding Robustness**: On the RM-Bench Hard subset, the performance of EvalPlanner improves instead of declining (84.3 vs. Overall 82.1), while other models generally exhibit performance drops on Hard.

## Highlights & Insights

1. The definition of **"evaluation as planning + reasoning"** is highly precise: representing the evaluation task as first writing a review scheme and then executing the review matches the cognitive review patterns of human experts.
2. **Planning without looking at responses** is an ingenious design constraint: generating the evaluation plan solely based on instructions ensures the generalizability and objectivity of the plan, preventing it from being contaminated by specific response content.
3. The **synthetic data-driven** self-training loop is of high practical value: it completely eliminates the need for human-annotated reasoning chains and can be bootstrapped with only a few (instruction, chosen, rejected) triplets.
4. **Positional bias elimination** via bidirectional permutation of response pairs is both simple and effective.
5. The finding of **Iterative DPO > Large-batch DPO** holds broader implications for the field of preference optimization: the "on-policy" nature of generating training data via updated models is more important than merely scaling up the data volume.

## Limitations & Future Work

1. **Only two iterations were verified**: Whether multi-round iterative DPO can continuously improve performance remains unexplored.
2. **Plan and execution are still generated sequentially**: The two phases increase inference latency and token consumption (up to 2048 tokens), which impacts efficiency.
3. **Limited to pairwise evaluation**: It has not yet been extended to pointwise scoring or ranking multiple responses.
4. **Did not surpass Skywork-Critic on JudgeBench**: There remains room for improvement in highly challenging comparisons that require deep domain knowledge.
5. **Lack of plan quality evaluation**: Currently, plan quality is only indirectly assessed via the correctness of the final verdict, lacking direct evaluation metrics for the plans themselves.
6. **Scalability to stronger seed models**: Currently, tests have only been conducted on Llama 70B scale; using GPT-4o or larger models as seeds might yield greater improvements.

## Related Work & Insights

- **Self-Taught Evaluators (Wang et al., 2024c)**: Also uses synthetic data and self-training, but constrains the evaluation CoT to a "criteria list" format; EvalPlanner's unconstrained planning and explicit decoupling offer better flexibility.
- **Skywork-Critic (Shiwen et al., 2024)**: A strong baseline trained with 80K human-annotated data, which EvalPlanner surpasses with much less synthetic data.
- **Chain-of-Verification (Dhuliawala et al., 2023)**: Constrains reasoning to a list of verification questions; the ablation studies of EvalPlanner show that the unconstrained form is superior.
- **DeepSeek-GRM (Liu et al., 2025)**: A strong baseline on PPE that uses the MetaRM voting strategy; EvalPlanner substantially outperforms it on PPE Correctness.
- **Insights for Future Research**: The plan-then-execute paradigm can be generalized to other tasks requiring structured reasoning, such as code review, document evaluation, and dialogue quality assessment.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Plan-execution decoupling and unconstrained plan generation are novel contributions |
| Technical Depth | 4 | Complete formalization, with delicately designed iterative self-training |
| Experimental Thoroughness | 5 | 5 benchmarks, comprehensive ablations, and validated across seed models |
| Practicality | 4 | Entirely driven by synthetic data, directly reproducible |
| Writing Quality | 4 | Clear structure with persuasive motivation statement |
| **Total Score** | **4.2** | A solid systematic study with clear advancement to the LLM-as-a-Judge field |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](../../ICLR2026/optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](../../ICLR2026/optimization/elastic_optimal_transport_theory_application_and_empirical_evaluation.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](../../ICML2026/optimization/distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)
- [\[NeurIPS 2025\] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](../../NeurIPS2025/optimization/dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)

</div>

<!-- RELATED:END -->
