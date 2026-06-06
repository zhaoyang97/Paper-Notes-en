---
title: >-
  [Paper Note] Revisiting the Reliability of Language Models in Instruction-Following
description: >-
  [ACL2026][LLM Evaluation][Instruction Following] This paper proposes nuance-oriented reliability and reliable@k, utilizing IFEval++ to examine whether models can stably handle semantically similar "cousin prompts" with d…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Instruction Following"
  - "Reliability Evaluation"
  - "IFEval++"
  - "cousin prompts"
  - "reliable@k"
date: 2026-05-08
content_hash: f6e20451f95417f1
---

# Revisiting the Reliability of Language Models in Instruction-Following

**Conference**: ACL2026  
**arXiv**: [2512.14754](https://arxiv.org/abs/2512.14754)  
**Code**: https://github.com/jianshuod/IFEval-pp  
**Area**: LLM Evaluation  
**Keywords**: Instruction Following, Reliability Evaluation, IFEval++, cousin prompts, reliable@k

## TL;DR
This paper proposes nuance-oriented reliability and reliable@k, utilizing IFEval++ to examine whether models can stably handle semantically similar "cousin prompts" with differing details. It finds that even high-scoring models experience significant performance drops under subtle prompt variations.

## Background & Motivation
**Background**: Instruction-following capabilities are typically evaluated through benchmarks like IFEval, FollowBench, and CFBench, focusing on whether a model satisfies explicit constraints such as format, length, keywords, and structure. With model iteration, many strong models have neared saturation on IFEval; for example, GPT-5's IFEval accuracy reaches 95.9%.

**Limitations of Prior Work**: High benchmark accuracy does not equate to reliability in real-world services. Users in actual scenarios change wording, contextual frameworks, numerical constraints, or task instances, yet many evaluations only measure the success of a single prompt without assessing consistency across a set of similar prompts.

**Key Challenge**: A model might pass the original prompt but fail on a "cousin prompt" with only minor detail changes. Traditional accuracy treats each prompt as an independent sample, failing to distinguish between "covering many types" and "being consistently reliable for the same intent."

**Goal**: The authors aim to construct an evaluation framework for micro-variation stability to determine if current LLMs possess nuance-oriented reliability in instruction following, and to further analyze how this reliability evolves with model scale, temporal iteration, reasoning capability, and improvement strategies.

**Key Insight**: Starting from IFEval, the paper automatically generates multiple "cousin prompts" for each original test sample. These retain similar user intent but introduce detail differences through rephrasing, adding compatible distractor constraints, or reconfiguring tasks/constraints. A sample is considered reliable only if the model passes the entire set of prompts.

**Core Idea**: Upgrade "passing a single item" to "passing a set of semantically adjacent items," using reliable@k to measure the model's stability against subtle prompt variations.

## Method
The core of this paper is not a new model, but a new evaluation dimension, a benchmark construction pipeline, and a set of systematic experiments. It decomposes the reliability of instruction-following into two orthogonal dimensions: comprehensiveness-oriented reliability (focusing on task and constraint coverage) and nuance-oriented reliability (focusing on stability across different expressions of the same intent).

### Overall Architecture
The overall workflow begins with the 541 original test samples from IFEval. For each sample, the system generates 9 cousin prompts, which, together with the original prompt, form a test group of 10. Each cousin prompt belongs to one of three enhancement categories: rephrasing, distractor addition, or constraint/task reconfiguration. After generation, a code-assisted validity checker ensures the prompt is consistent with the evaluation configuration and can be verified by IFEval's automated evaluation functions.

The resulting IFEval++ contains 541 test groups, each with 10 prompts. During evaluation, the authors report original IFEval accuracy, reliable@2 and reliable@4 on various enhancement subsets, and reliable@10 across the entire IFEval++.

### Key Designs
1.  **reliable@k Metric**:
    - **Function**: Measures whether a model can simultaneously handle $k$ cousin prompts rather than just a single prompt.
    - **Mechanism**: For a set of $k$ outputs, if all outputs pass their corresponding automated evaluation functions, reliable@k is 1; if even one fails, the set is 0. When $k=1$, it degrades to standard accuracy.
    - **Design Motivation**: Real users do not strictly use benchmark phrasing. reliable@k explicitly incorporates "local stability" into the metric, revealing second-order capabilities that standard accuracy misses.

2.  **Three Categories of Cousin Prompt Enhancement**:
    - **Function**: Creates subtle but reasonable instruction variations from different angles.
    - **Mechanism**: Rephrasing changes wording while keeping constraints identical; distractor addition adds constraints that are compatible with the original but not scored; constraint/task reconfiguration changes configurable parameters or shifts to a different task scenario while updating the evaluation config.
    - **Design Motivation**: These three variations correspond to different expressions, extra requirements, and task instance changes from real users. They are closer to the question of "robustness of a single capability" than entirely new tasks.

3.  **Code-Assisted Validity Checker**:
    - **Function**: Filters invalid, ambiguous, or config-mismatched samples generated during the enhancement process.
    - **Mechanism**: The checker embeds evaluation function implementations and configuration descriptions into the prompt, asking an LLM to judge if the enhanced sample is consistent with the executable evaluation logic. The authors use a high-recall strategy, preferring to flag suspicious samples to lower the probability of including flawed cases.
    - **Design Motivation**: If a cousin prompt is inherently invalid, a reliable@k failure cannot be attributed to model unreliability. This checker achieved 99.7% recall on 900 injected error samples and 99.9% recall on 3,000 additional flawed cases.

### Loss & Training
The evaluation portion does not involve training. In improvement experiments, the authors test three paths: predicting whether a prompt will be followed, SFT on data with similar objectives, and expanding test-time compute through reasoning effort or rejection sampling. In training experiments, Qwen2.5-7B-Instruct was used for SFT over 312 steps on Alpaca and decontaminated IFEval cousin prompts to compare reliability changes.

## Key Experimental Results

### Main Results
The authors evaluated 46 models, including 20 proprietary and 26 open-source models, covering different scales, vendors, reasoning modes, and eras.

| Model | IFEval Accuracy | IFEval++ reliable@10 | Relative Drop | Observation |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5 | 95.9 | 78.4 | -18.3% | Most reliable, but still drops significantly |
| o3 | 94.3 | 75.0 | -21.3% | Strong performance from reasoning models |
| LLaMA-3.3-70B-Instruct | 92.1 | 71.0 | -22.9% | One of the strongest open-source models |
| Gemma-3-IT-27B | 84.3 | 61.6 | -27.0% | Lower accuracy rank, but higher reliable@10 rank |
| Qwen3-0.6B | 58.0 | 22.2 | -61.8% | Small models are most fragile to subtle changes |
| GPT-3.5-turbo-1106 | 61.6 | 27.9 | -54.7% | Significant drop for older proprietary models |

The results show that IFEval accuracy is highly correlated with but not equivalent to IFEval++ reliable@10. Some models with average rankings on original IFEval are more stable on cousin prompts, indicating that nuance-oriented reliability is a capability independent of single-point accuracy.

### Ablation Study
The paper tested three types of methods for improving reliability: prediction, training, and test-time scaling.

| Configuration / Method | Key Metrics | Description |
| :--- | :--- | :--- |
| verbalized confidence | AUROC 0.549 / 0.518 | Qwen3-8B and Qwen2.5-7B are near random; model confidence is unreliable |
| prompt perplexity | AUROC 0.497 / 0.529 | Prompt familiarity does not predict following success |
| hidden-state probing | AUROC 0.757 / 0.759 | Intermediate hidden states provide some predictive signal |
| Alpaca SFT | reliable@10 slight drop | General instruction data might not improve subtle stability |
| curated cousin-prompt SFT | Over 45% after 200 steps | Semantically adjacent data is more effective |
| rejection sampling | Plateaus around $n=12$ | Reliability increases significantly if a response selector is available |

### Key Findings
- Reliability drops caused by subtle variations are widespread, reaching up to 61.8%. This indicates that saturation on instruction-following benchmarks does not represent saturation in real-world stability.
- Rephrasing is generally the easiest, while distractor addition and constraint/task reconfiguration are harder as they increase pressure on response planning and constraint execution.
- Model scale generally helps but is not the sole factor. Qwen3-14B outperformed the larger Qwen3-32B on certain reliability metrics, showing that training methods and data quality are equally critical.
- Reasoning capabilities usually enhance reliability, but they are not a sufficient or necessary condition. LLaMA-3.3-70B-Instruct is not a reasoning model but remains one of the strongest open-source performers.
- reliable@10 is distinct from pass@10. The former measures stability across semantically adjacent prompts, while the latter measures random stability across multiple samplings of the same prompt; for LLaMA-3.3-70B, accuracy is 92.1, reliable@10 is 71.0, and pass@10 is 85.6, showing a clear difference.

## Highlights & Insights
- Defining "reliability" through executable metrics rather than vague concepts is the major contribution. reliable@k is simple yet diagnostic, particularly for revealing benchmark overfitting and prompt sensitivity.
- The construction of cousin prompts is broader than traditional paraphrase robustness. It examines compatibility with distractors and stability under fine-tuned constraints, which better approximates diverse user expressions.
- The training experiments provide a practical signal: improving reliability does not necessarily require more general instruction data, but targeted training around semantically adjacent samples.
- The analysis of test-time scaling is realistic. Rejection sampling can significantly boost reliability provided there is a programmatically verifiable selector; however, this also highlights the gap between verifiable tasks and open-ended tasks.

## Limitations & Future Work
- The cost of a full IFEval++ evaluation is 10x that of IFEval due to more response generations. Future work needs to efficiently select the most discriminative cousin prompts.
- The evaluation primarily focuses on format and constraint following, without assessing content quality. A model might satisfy format constraints while providing mediocre content, which is still insufficient for real-world services.
- This work is primarily based on the English version of IFEval. While the method can be migrated to other languages, it requires translation, constraint function adaptation, and language-specific validity checks.
- Although the validity checker has high recall, it still relies on LLM judgment, which may introduce subtle biases. Stronger programmatic checks or human audits could further enhance credibility.
- Improvement strategies only covered representative methods; the paper did not systematically reproduce all instruction-following enhancement techniques, so it cannot definitively state which training or alignment strategy is optimal.

## Related Work & Insights
- **vs IFEval**: IFEval evaluates whether a single prompt satisfies constraints; IFEval++ evaluates whether multiple subtle expressions of the same intent are all satisfied.
- **vs Multi-constraint benchmarks**: FollowBench, CFBench, and ComplexBench emphasize coverage of constraint types and complexity; this paper emphasizes consistency across semantically adjacent samples.
- **vs pass@k**: pass@k reflects stability over multiple samplings of the same prompt, while reliable@k reflects stability over different cousin prompts, capturing different risks.
- **Insight**: Future LLM evaluations should provide a local perturbation family for each core sample. Model scores should not just reflect "how many items were correct," but "how stable the underlying capability is."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The reliable@k concept is simple and powerful, turning prompt-level stability into a scalable evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 46 models and analyzes scale, time, reasoning, enhancement types, and improvement paths.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure with intuitive examples; high information density in long tables requires attention to metric definitions.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to LLM evaluation, model release reports, and monitoring for reliable services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IF-RewardBench: Benchmarking Judge Models for Instruction-Following Evaluation](if-rewardbench_benchmarking_judge_models_for_instruction-following_evaluation.md)
- [\[ACL 2026\] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models](revisiting_a_pain_in_the_neck_a_semantic_reasoning_benchmark_for_language_models.md)
- [\[ACL 2026\] IF-Critic: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation](if-critic_towards_a_fine-grained_llm_critic_for_instruction-following_evaluation.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)

</div>

<!-- RELATED:END -->
