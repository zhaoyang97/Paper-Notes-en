---
title: >-
  [Paper Note] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning
description: >-
  [ACL 2025][Reasoning][Mathematical Reasoning] Proposes the Self-Error-Instruct (SEI) framework, which analyzes error cases of the target model in mathematical reasoning, uses GPT-4o to extract error keyphrases and clusters them into error types, synthesizes training data for each error type using a self-instruct approach, and iteratively fine-tunes the model to systematically address weaknesses in mathematical reasoning.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Mathematical Reasoning"
  - "Error Analysis"
  - "Training Data Synthesis"
  - "Self-Instruct Learning"
  - "Iterative Fine-Tuning"
date: 2026-05-08
content_hash: a408cd0a4b8f85f8
---

# Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.22591](https://arxiv.org/abs/2505.22591)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Mathematical Reasoning, Error Analysis, Training Data Synthesis, Self-Instruct Learning, Iterative Fine-Tuning  

## TL;DR

Proposes the Self-Error-Instruct (SEI) framework, which analyzes error cases of the target model in mathematical reasoning, uses GPT-4o to extract error keyphrases and clusters them into error types, synthesizes training data for each error type using a self-instruct approach, and iteratively fine-tunes the model to systematically address weaknesses in mathematical reasoning.

## Background & Motivation

**Background**: Large language models have made significant progress on mathematical reasoning tasks (e.g., GSM8K, MATH), but they still exhibit a large number of bad cases. These errors typically follow specific patterns—such as misuse of particular operator types, misunderstanding of problem structures, or computational errors.

**Limitations of Prior Work**: Existing "learning from errors" methods typically adopt a straightforward strategy: collect error cases $\rightarrow$ let a stronger model generate correct solutions for each error case individually $\rightarrow$ fine-tune the target model using these corrected data. The limitation of this approach is that it only patches errors individually without extracting general patterns from them—errors of the same type may recur across different problems but are treated independently each time, resulting in synthesized data that lacks systematicity and generalizability.

**Key Challenge**: A single error case carries limited information, but errors of the same type embody common patterns. Existing methods ignore this "categorized" error information—if a model repeatedly makes errors in fraction arithmetic, remediating individual problems cannot fundamentally solve the issue for that entire category.

**Goal**: Design a framework that can extract general patterns from errors and synthesize targeted training data, enabling the model to systematically overcome specific types of mathematical reasoning weaknesses.

**Key Insight**: The authors propose the concept of "Error Generalization"—instead of patching individual errors, they analyze the common characteristics of error cases to identify "error types", and then synthesize training data at scale for each type. In this way, a single error case can drive improvements across an entire class of problems.

**Core Idea**: First cluster to discover error types, then synthesize training data in batches by type, allowing a small number of error cases to drive broad capability improvements.

## Method

### Overall Architecture

SEI is an iterative "diagnosis-synthesis-training" framework. The inputs are the target model and mathematical datasets (GSM8K, MATH), and the output is the fine-tuned, enhanced model. The overall workflow is: (1) Run the target model on the dataset to collect error cases; (2) Use GPT-4o (the instructor model) to analyze each error case and extract error keyphrases; (3) Cluster the keyphrases to identify error types; (4) Sample a few representative error cases for each error type and let GPT-4o synthesize new training data; (5) Filter the synthesized data to retain high-quality samples; (6) Fine-tune the target model using the filtered data. The entire process can repeat for multiple iterations, discovering new error types and synthesizing new data in each round.

### Key Designs

1. **Error Keyphrase Extraction & Clustering**:

    * **Function**: From a large number of error cases, extract a structured categorization of error types.
    * **Mechanism**: For each error case of the target model, input the incorrect reasoning process along with the correct answer into GPT-4o, prompting it to analyze the root cause of the error and output a concise "error keyphrase" (e.g., "fraction division confusion", "omitted unit conversion", "omitted condition judgment"). After gathering all error keyphrases, perform text clustering—heuristically grouping semantically similar keyphrases into the same error type. This yields several error type clusters, each representing a systematic weakness of the model.
    * **Design Motivation**: Directly clustering raw error cases is too coarse-grained (as the underlying math problems differ greatly). In contrast, keyphrases represent a high-level abstraction of the root error causes, and clustering in the keyphrase space captures commonalities in error patterns more accurately.

2. **Type-Specific Self-Instruct Data Synthesis**:

    * **Function**: Synthesize targeted training data at scale for each error type.
    * **Mechanism**: For each error type cluster, sample a few representative error cases as "exemplars" and feed them into GPT-4o along with the description of the error type. GPT-4o then synthesizes new math problems and their correct solutions in a self-instruct manner, referencing the structure and difficulty of these exemplars. Crucially, the newly synthesized problems are tailored specifically to the error type, creating scenarios likely to trigger that specific mistake, which forces the model to learn how to handle such cases correctly during fine-tuning. This yields a massive amount of targeted training data for each error type.
    * **Design Motivation**: The core advantage of self-instruct is its ability to scale data volume rapidly. Organizing the synthesis process by error type ensures that the data is highly targeted—generating problems specifically designed to expose the model's weaknesses rather than randomly generating questions.

3. **One-shot Quality Refinement**:

    * **Function**: Ensure the quality and effectiveness of synthesized data.
    * **Mechanism**: Not all synthesized training data is usable—there may be issues such as unclear problem statements, incorrect solutions, or difficulty mismatches. SEI adopts a one-shot learning approach for quality screening: the target model attempts one round of training (one-shot) on the synthesized data, and its improvement is observed on a validation set corresponding to that error type. Only training samples that effectively boost model performance are retained, while ineffective or harmful data are filtered out.
    * **Design Motivation**: Data quality is more critical than quantity. Unfiltered synthesized data may introduce noise or even mislead the model. One-shot validation serves as an efficient signal for quality—if a data sample does not help the model improve, the sample itself is likely flawed.

### Loss & Training

SEI employs an iterative fine-tuning strategy. Each iteration discovers new error patterns (as the model's error distribution shifts after repairing some errors in the previous round), synthesizes targeted data, and executes fine-tuning again. Typically, model performance stabilizes after 2-3 iterations. Fine-tuning follows the standard Supervised Fine-Tuning (SFT) pipeline.

## Key Experimental Results

### Main Results

Evaluation is conducted on two mathematical reasoning benchmarks, GSM8K and MATH, along with testing out-of-domain generalization to other math datasets.

| Model | Method | GSM8K | MATH | Out-of-Domain Datasets (avg) |
|------|------|-------|------|-----------------|
| Llama-3 8B | Baseline | Baseline Score | Baseline Score | Baseline Score |
| Llama-3 8B | + Direct Error Correction Data | Minor Gain | Minor Gain | Marginal Gain |
| Llama-3 8B | + SEI (1 Round) | Significant Gain | Significant Gain | Noticeable Gain |
| Llama-3 8B | + SEI (Iterative) | **Best** | **Best** | **Best** |
| Mistral 7B | Baseline | Baseline Score | Baseline Score | Baseline Score |
| Mistral 7B | + SEI (Iterative) | Significant Gain | Significant Gain | Noticeable Gain |
| CodeLlama | + SEI | Significant Gain | Significant Gain | Good Generalization |

### Ablation Study

| Configuration | GSM8K Gain | MATH Gain | Description |
|------|-----------|----------|------|
| Full SEI | Best | Best | Full Framework |
| w/o Error Clustering (Random Synthesis) | Minor Gain | Minor Gain | Synthesizing data without target yields poor results |
| w/o Quality Refinement | Lower than Full | Lower than Full | Noisy data degrades performance |
| w/o Iteration (1 Round only) | Moderate Gain | Moderate Gain | A single round cannot discover all error types |
| Increased Synthesized Data (2x) | Close to Full | Close to Full | But doubles training costs |
| Reduced Representative Cases | Slightly lower than Full | Slightly lower than Full | Too few samples per category loses information |

### Key Findings

* **Error clustering is the core contribution**: Compared to direct synthesis without clustering, organizing synthesized data by error type brings the largest performance gain, validating the "Error Generalization" hypothesis.
* **Strong out-of-domain generalization**: The models trained on GSM8K/MATH also show improvements on other unseen math datasets, indicating that SEI enhances general mathematical reasoning rather than overfitting to specific datasets.
* **Iterative strategy is effective but converges quickly**: The first iteration contributes the most (patching the most common errors), with diminishing returns in subsequent rounds. Typically, performance stabilizes after 2-3 rounds.
* **Quality refinement is indispensable**: Unfiltered synthesized data can even lead to performance drops on some metrics, showing that one-shot filtering is a simple and effective quality control tool.

## Highlights & Insights

* **"Error Generalization" is a new paradigm for data augmentation**: Instead of simply "doing more practice problems," it centers on "analyzing which categories of problems were answered incorrectly $\rightarrow$ specifically designing problems in those categories to practice." This approach can be transferred to other domains such as code generation and logical reasoning—diagnosing weaknesses first, then targeted improvement.
* **The design of keyphrase clustering is elegant**: Directly clustering error cases is too coarse-grained due to vast differences in problem content. Clustering in the "error cause keyphrase" space offers a more appropriate granularity, successfully capturing commonalities without over-generalizing.
* **Role division with GPT-4o as the instructor**: Utilizing a stronger model as the "coach" for error analysis and data synthesis while treating the weaker model as the "student" being trained forms an highly efficient path for capability transfer.

## Limitations & Future Work

* Reliance on GPT-4o as the instructor model leads to high API call costs. Substituting GPT-4o with cheaper models for error analysis could heavily reduce overall expenses.
* The quality of error keyphrases and clustering outcomes heavily depends on the diagnostic capability of the instructor model; inaccurate diagnosis can lead to incorrect category partitioning.
* Currently validated only on mathematical reasoning tasks; whether it can generalize to other tasks like code generation, logical reasoning, and commonsense reasoning remains to be further investigated.
* The one-shot quality refinement might be overly simplified; more fine-grained filtering strategies (e.g., based on difficulty matching or diversity control) could yield better results.
* During the iterative process, the model may introduce new errors (catastrophic forgetting) while fixing old ones; how to balance this dynamic is worth further study.

## Related Work & Insights

* **vs Self-Instruct**: Self-Instruct randomly synthesizes training data to enhance general capabilities, whereas SEI is "targeted"—only generating data in the model's weak areas. This targeted approach has much higher data efficiency than random synthesis.
* **vs WizardMath / MetaMath**: These methods augment data volume by rewriting existing math problems, but do not analyze where the model actually fails. SEI's error analysis step makes data synthesis far more targeted.
* **vs Reinforcement Learning (e.g., RLHF) for correcting reasoning errors**: RL methods implicitly learn to avoid errors via reward signals, whereas SEI corrects errors through explicit error analysis and data synthesis, making it more transparent and controllable.

## Rating

* Novelty: ⭐⭐⭐⭐ The concept of "Error Generalization" and error keyphrase clustering are core innovations, though the self-instruct + iterative fine-tuning pipeline is not entirely new.
* Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple models and datasets, including out-of-domain generalization and ablation studies, but lacks comparison with several of the latest math enhancement methods.
* Writing Quality: ⭐⭐⭐⭐ The framework description is clear and the workflow diagram is intuitive, though some details (such as the choice of clustering algorithm) could be more detailed.
* Value: ⭐⭐⭐⭐ Introduces a new paradigm for training data synthesis with practical value for boosting specific model capabilities, offering highly transferable ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ProcessBench: Identifying Process Errors in Mathematical Reasoning](processbench_identifying_process_errors_in_mathematical_reasoning.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)

</div>

<!-- RELATED:END -->
