---
title: >-
  [Paper Note] SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning
description: >-
  [ACL 2025][Issue resolving] This paper proposes SoRFT (Subtask-oriented Reinforced Fine-Tuning), which decomposes the GitHub Issue resolving task into four subtasks: file localization, function localization, line localization, and code editing. Through a two-stage training process consisting of rejection-sampled SFT and rule-based PPO reinforcement learning, SoRFT significantly enhances the issue-resolving capabilities of open-source LLMs on SWE-Bench.
tags:
  - "ACL 2025"
  - "Issue resolving"
  - "Reinforced fine-tuning"
  - "Subtask decomposition"
  - "Code generation"
  - "SWE-Bench"
date: 2026-05-08
content_hash: 02ac7ae70debe34d
---

# SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning

**Conference**: ACL 2025  
**arXiv**: [2502.20127](https://arxiv.org/abs/2502.20127)  
**Code**: None  
**Area**: Other  
**Keywords**: Issue resolving, Reinforced fine-tuning, Subtask decomposition, Code generation, SWE-Bench

## TL;DR

This paper proposes SoRFT (Subtask-oriented Reinforced Fine-Tuning), which decomposes the GitHub Issue resolving task into four subtasks: file localization, function localization, line localization, and code editing. Through a two-stage training process consisting of rejection-sampled SFT and rule-based PPO reinforcement learning, SoRFT significantly enhances the issue-resolving capabilities of open-source LLMs on SWE-Bench.

## Background & Motivation

Current mainstream issue-resolving frameworks (such as Agentless, OpenHands) primarily rely on commercial models (such as Claude-3.5-Sonnet, GPT-4o), which raise concerns regarding high costs and privacy leakage. Existing training methods for open-source models rely solely on supervised fine-tuning (SFT), which easily leads to poor generalization, hallucinations, and factual errors.

Recent works such as DeepSeek-R1 demonstrate that rule-based reinforcement learning can effectively improve model performance on complex tasks like mathematics. Meanwhile, the open-source community provides a vast amount of resolved issues and their corresponding Pull Requests, which naturally contain ground-truth patches. This provides ideal conditions for constructing rule-based reward signals. The core problem is: Can these (issue, patch) pairs be leveraged for rule-based reinforcement learning to enhance the issue-resolving capabilities of open-source LLMs?

## Method

### Overall Architecture

SoRFT consists of three core components:
1. Subtask decomposition of the issue-resolving task
2. Rejection-sampled Supervised Fine-Tuning (Rejection-sampled SFT)
3. Rule-based Reinforcement Learning (Rule-based RL)

The overall workflow is as follows: first, decompose the issue-resolving task into four subtasks and construct training data; then, use a teacher LLM to sample Chain-of-Thought (CoT) data and filter out negative samples for SFT; finally, perform reinforcement learning using PPO combined with rule-based rewards.

### Key Designs

1. **Subtask Decomposition**: The issue-resolving task is decomposed into four hierarchical subtasks: file localization (locating modified files based on the issue and repository structure), function localization (locating modified functions based on the file skeleton), line localization (precisely identifying the code lines to be modified), and code editing/generation (generating code modifications in Search/Replace format). The ground-truth for each subtask is extracted from the corresponding PR, making the construction of training signals much clearer.

2. **Rejection-sampled SFT**: A teacher LLM (Claude-3.5-Sonnet) is used to sample CoT data for each subtask, followed by filtering out negative samples based on the ground-truth. For localization tasks, samples with no overlap with the ground-truth are filtered out; for code editing, samples that do not overlap with the ground-truth modified lines are filtered out. Finally, the CoT data from all subtasks are integrated for SFT, helping the model master task formats and reasoning pathways.

3. **Rule-based Reinforcement Learning**: An $F_\beta$ score ($\beta=3$, prioritizing recall) is designed as the reward function to replace traditional reward models. For localization tasks, the localized results are extracted from the model's output and evaluated against the ground-truth to compute the $F_\beta$ score; for code editing, the modified code is extracted and compared with the ground-truth. Meanwhile, formatting penalties are incorporated: if the output is empty or contains targets that do not exist in the input, the reward is directly set to 0, which effectively prevents reward hacking.

### Loss & Training

- SFT Stage: Full-parameter fine-tuning is performed using FastChat/DeepSpeed with a global batch size of 128, trained for 2 epochs, with a maximum learning rate of 1e-5, cosine decay, and a 3% warmup.
- RL Stage: PPO is implemented using the OpenRLHF framework, sampling at a temperature of 1.0, integrated with vLLM for accelerated inference.
- Training Data: Filtered from 660 high-quality Python open-source repositories, consisting of 60k SFT samples and 30k RL samples.
- Repositories in the SWE-Bench test set are strictly excluded to prevent data contamination.

## Key Experimental Results

### Main Results

| Model | Framework | SWE-Bench Verified | SWE-Bench Lite |
|------|------|-------------------|----------------|
| Claude-3.5-Sonnet | Agentless | 50.8% | 40.7% |
| GPT-4o | SWE-SynInfer | 31.8% | 20.7% |
| SWE-Gym-Qwen-7B | OpenHands | 10.6% | 10.0% |
| SWE-Gym-Qwen-14B | OpenHands | 16.4% | 12.7% |
| Lingma-SWE-GPT-7B | SWE-SynInfer | 18.2% | 12.0% |
| **SoRFT-Qwen-7B** | Agentless | **21.4%** | **14.0%** |
| SWE-Fixer-Qwen-72B | SWE-Fixer | 30.2% | 23.3% |
| Lingma-SWE-GPT-72B | SWE-SynInfer | 30.2% | 22.0% |
| **SoRFT-Qwen-32B** | Agentless | **30.8%** | **24.0%** |

### Ablation Study

| Configuration | Verified %Resolved | %Applied | Description |
|------|------|-------------------|----------|
| Qwen-7B base | 7.6% | 55.6% | Baseline |
| + SFT | 18.0% | 85.2% | SFT provides significant improvement |
| + SFT + RL (SoRFT) | 21.4% | 95.6% | RL yields further improvement |
| Qwen-32B base | 25.6% | 84.4% | Large model baseline |
| + SFT | 28.8% | 90.6% | SFT improvement |
| + SFT + RL (SoRFT) | 30.8% | 95.8% | Full SoRFT |

### Key Findings

- **SoRFT-Qwen-7B outperforms SWE-Gym-Qwen-32B** (21.4% vs 20.6%), demonstrating that smaller models can surpass larger ones through fine-grained training.
- **SoRFT-Qwen-32B outperforms Lingma-SWE-GPT-72B** (30.8% vs 30.2%) while using only half of its parameters.
- **Robustness of reward rules is critical**: using a simple hit score leads to reward hacking (where the model tends to generate less reasoning and more answers), whereas the $F_\beta$ score effectively mitigates this issue.
- **An emergent phenomenon consistent with DeepSeek-R1 is observed in PPO training**: the thinking length first decreases and then increases.
- **SoRFT also improves general coding tasks**: LiveCodeBench improves from 34.18 to 34.64, and RepoQA improves from 85.0 to 90.0.

## Highlights & Insights

- The subtask decomposition approach is clear, transforming a difficult end-to-end problem into supervised, step-by-step tasks.
- It cleverly leverages (issue, PR) data pairs from the open-source community as ground-truth, avoiding expensive manual annotation.
- Pipeline frameworks are more suitable for constructing training signals than Agent frameworks: the intermediate steps of an Agent are difficult to evaluate, while pipeline stages can be scored independently.
- The design choice of the $F_\beta$ score ($\beta=3$ prioritizing recall) is based on reasonable intuition: missing a target in localization tasks is much more severe than over-selecting candidate targets.

## Limitations & Future Work

- Experiments are only conducted on Python repositories, lacking multi-lingual validation (though the framework itself is language-agnostic).
- The rule-based reward suffers from a false-negative pathology: a given issue may have multiple correct solutions, but comparing only against a single ground-truth may incorrectly penalize valid alternative solutions.
- Future work could incorporate unit test execution results as more objective signals for evaluating code quality.

## Related Work & Insights

- It shares a consistent direction with DeepSeek-R1: employing rule-based RL to replace traditional reward models.
- The staged design of the Agentless framework provides natural splitting boundaries for subtask decomposition.
- Insight: The software engineering domain possesses a vast amount of natural ground-truth signals (PRs, test cases), serving as a natural playground for RL training.

## Rating

- Novelty: ⭐⭐⭐⭐ — First time applying reinforced fine-tuning to the issue-resolving task, with creative subtask reward designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations, with fair comparisons against multiple frameworks and baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear workflow and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ — Directly reduces the cost of issue resolving, offering open-source alternatives to commercial models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ICML 2026\] Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning](../../ICML2026/others/vision_transformer_finetuning_benefits_from_non-smooth_components.md)

</div>

<!-- RELATED:END -->
