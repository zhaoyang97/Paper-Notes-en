---
title: >-
  [Paper Note] ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models
description: >-
  [ACL2026][LLM Evaluation][Code Sandbox] ScaleBox improves verification precision and throughput in LLM code training and evaluation through automated special judge synthesis, unified verification workflows…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Code Sandbox"
  - "Special Judge"
  - "RLVR"
  - "Reward Noise"
  - "Distributed Evaluation"
date: 2026-05-08
content_hash: 736a998ec589ce2e
---

# ScaleBox: Enabling High-Fidelity and Scalable Code Verification for Large Language Models

**Conference**: ACL2026  
**arXiv**: [2604.27467](https://arxiv.org/abs/2604.27467)  
**Code**: https://github.com/icip-cas/ScaleBox  
**Area**: LLM Evaluation / Code Verification / RLVR Infrastructure  
**Keywords**: Code Sandbox, Special Judge, RLVR, Reward Noise, Distributed Evaluation

## TL;DR
ScaleBox improves verification precision and throughput in LLM code training and evaluation through automated special judge synthesis, unified verification workflows, and distributed fine-grained parallelism, leading to a more stable Pass@1 improvement in LiveCodeBench RLVR experiments.

## Background & Motivation
**Background**: Code capability is a critical direction in current LLM training and evaluation. Whether for benchmarks like HumanEval, MBPP, and LiveCodeBench, or for RLVR training, systems rely on sandbox environments to execute model-generated programs and return verifiable feedback.

**Limitations of Prior Work**: Mainstream code sandboxes commonly use exact match or simple heuristic comparisons of output. This verification assumes each problem has a unique standard output, yet real-world programming tasks often allow multiple valid solutions, any valid execution path, or floating-point error tolerances. An analysis of 34,757 problems in the paper shows that 14.57% of tasks require a special judge, and in these cases, 59.01% of correct reference solutions would be incorrectly rejected by exact match.

**Key Challenge**: RLVR training requires high-throughput feedback; otherwise, GPU/NPU performance is throttled by CPU-side execution. However, high-quality verification cannot rely solely on fast exact matching. If reward signals contain significant false negatives, the policy treats correct exploration as failure, damaging training stability and final code capability.

**Goal**: The authors aim to build a code verification system that is both high-fidelity and scalable, supporting various evaluation forms such as exact match, function call, assert, and special judge, while providing stable, high-throughput feedback during large-scale RL training.

**Key Insight**: ScaleBox bifurcates the problem into two layers: the verification precision layer uses automated special judge synthesis and management to compensate for exact match defects, while the system efficiency layer uses multi-node deployment combined with instance-level and test-case-level hybrid parallelism to boost CPU-side throughput.

**Core Idea**: The key bottleneck in code RL is not just on the model side, but in the reward infrastructure. If the verifier is both inaccurate and slow, the trained policy will be limited by both erroneous rewards and system throughput.

## Method

### Overall Architecture
ScaleBox is a distributed sandbox system oriented towards large-scale code training and evaluation. Based on the execution foundation of SandboxFusion, it incorporates three core capabilities: end-to-end evaluation workflows, distributed deployment, and special judge support. The system extracts code from LLM outputs, generates test execution environments tailored to different problem types, and returns results through a unified API.

In the evaluation process, ScaleBox first executes lightweight exact match with common format tolerance rules. If a task is identified as requiring a special judge and preliminary checks are insufficient, the corresponding programmable judging logic is invoked. This preserves a fast path while handling multi-solution and numerical tolerance problems. At the system level, NGINX load balancing, Docker health checks, multi-tenant workers, and test-case-level parallelism support high-concurrency RL training.

### Key Designs

1.  **Unified Evaluation Workflow**:
    -   **Function**: Converges various benchmarks and output formats into a single verification interface.
    -   **Mechanism**: The workflow includes request routing, code extraction, test case distribution, parallel execution, and multi-stage verification. The system identifies test forms like stdin/stdout, function call, and assert to generate corresponding execution wrappers.
    -   **Design Motivation**: LLM output formats vary widely, and benchmark conventions differ. A unified workflow reduces evaluation script fragmentation and improves results reproducibility.

2.  **Automated Special Judge Synthesis and Verification**:
    -   **Function**: Generates task-specific verification logic for problems that exact match cannot cover.
    -   **Mechanism**: An LLM first determines if a problem belongs to a special judge type (e.g., multiple valid outputs or numerical tolerance). Then, a judge program is synthesized. Finally, the judge is checked in a sandbox using correct solutions and known erroneous outputs for fidelity and robustness; it is regenerated or discarded if it fails.
    -   **Design Motivation**: Manually writing special judges for large-scale training data is infeasible. Automated synthesis allows high-fidelity rewards to scale to thousands of problems, while pre-deployment verification controls the risk of incorrect judging.

3.  **Distributed Hybrid Parallel Execution**:
    -   **Function**: Increases CPU-side execution throughput to alleviate host-accelerator imbalance in RL training.
    -   **Mechanism**: ScaleBox parallelizes not only at the task instance level but also across multiple test cases within a single instance. Multiple nodes are coordinated by load balancers, workers support multi-tenant execution, and a dashboard monitors resources and logs.
    -   **Design Motivation**: Code training generates bursty, high-frequency verification requests. Single-granularity queuing wastes CPU resources and slows down model training. Test-case-level parallelism significantly reduces per-problem verification latency.

### Loss & Training
ScaleBox itself is a verification system and does not propose new model training losses. In RL experiments, the authors used Qwen3-8B non-thinking as the base policy and GRPO within the verl framework. Training data came from PrimeIntellect/verifiable-coding-problems, with 26K Python problems selected. Among these, 2.8K were identified as requiring a special judge, and 1.2K belonged to a critical subset where exact match would reject reference solutions but a special judge could verify them.

## Key Experimental Results

### Main Results

| Experimental Item | Metric | Results | Description |
| :--- | :--- | :--- | :--- |
| EM Vulnerability Analysis | Ratio of tasks requiring special judge | 14.57% / 34,757 | Multi-solution or floating-point tolerance types |
| EM False Negative | Ratio of correct solutions rejected | 59.01% | Source of reward noise |
| Single-node Efficiency | ScaleBox 39.31 tasks/s | verl Prime: 24.73, SandboxFusion: 14.92 | 1.59× faster than verl, 2.63× faster than SandboxFusion |
| Triple-node Efficiency | 62.10 tasks/s | 2.51× relative to verl single-node | Demonstrates multi-node scalability |
| Special Judge Fidelity | $TPR \ge 90.0\%$, $TNR \ge 84.0\%$ | 27 AetherCode complex tasks, 529 submissions | Automated synthesized judges are generally reliable |

### Ablation Study

| Training Set | Reward Type | LCB-v5 Pass@1 | LCB-v6 Pass@1 | Description |
| :--- | :--- | :--- | :--- | :--- |
| Base Model | - | 25.09 | 27.21 | Qwen3-8B initial policy |
| 1.2K SPJ subset | w/o SPJ (EM) | 27.24 | 27.94 | Exact match rewards contain heavy false negatives |
| 1.2K SPJ subset | w/ SPJ (Ours) | 33.15 | 32.35 | LCB-v5 gain +5.91, LCB-v6 gain +4.41 |
| 26K full dataset | w/o SPJ (EM) | 37.19 | 34.12 | Large-scale hybrid training |
| 26K full dataset | w/ SPJ (Ours) | 38.17 | 36.03 | Gains of +1.91 on LCB-v6 despite SPJ being only ~10% |

### Key Findings
- Special judges are not merely evaluation decorators; they change RL reward density. For the 1.2K critical subset, EM treats valid exploration as failure, whereas SPJ significantly improves training reward and final Pass@1.
- High-fidelity rewards improve training stability. Figures in the paper show that the SPJ-enhanced model maintains performance leadership throughout training, indicating that cleaner rewards reduce credit assignment variance.
- System efficiency and verification precision require simultaneous optimization. Improving special judge precision without sufficient throughput slows down RL; pursuing speed while relying on EM limits model capability due to reward noise.
- ScaleBox's one-click workflow supports benchmarks like HumanEval, MBPP, HumanEval+, MBPP+, LiveCodeBench, and AetherCode, proving it is not customized for a single dataset.

## Highlights & Insights
- The paper clearly identifies the "reward infrastructure" problem in code RL: model training quality is directly constrained by the verifier's misjudgment rate, a point often underestimated in algorithmic papers.
- The three-stage special judge synthesis design balances automation and quality control. The LLM handles scaling coverage, while sandbox verification filters out low-quality judges before deployment.
- Short-circuit verification is a practical engineering trade-off: samples that can be correctly judged by simple rules pass quickly, while complex samples are handed to higher-cost judging logic.
- Experimental results simultaneously demonstrate system throughput, judging accuracy, and RL performance, proving that ScaleBox's value is not just "running fast" but producing more useful training signals.

## Limitations & Future Work
- Current large-scale verification focuses on specific model sizes, Python problems, and mainstream benchmarks; multi-language, multi-paradigm, and larger-scale RL training require further validation.
- Automated special judges depend on LLM synthesis and filtering. While $TPR/TNR$ are high, errors may still occur in complex interactive problems or those with hidden constraints.
- The system focuses on code training and benchmark evaluation, with limited extension to multi-turn code agents, automated software engineering, or long-duration task environments.
- Although special judges cover only ~10% of the full dataset, they bring significant benefits; however, this implies that gains are sensitive to the proportion of complex problems in the dataset.
- The paper does not discuss sandbox security isolation details or resource abuse protection boundaries in depth; real production deployment still requires additional security assessments.

## Related Work & Insights
- **vs SandboxFusion**: SandboxFusion provides the code execution foundation, but ScaleBox emphasizes high concurrency, multi-node scaling, and native special judge support in RL training.
- **vs verl native execution**: verl native execution supports RL workflows but has lower throughput than ScaleBox; ScaleBox alleviates CPU-side bottlenecks through hybrid parallelism and batching.
- **vs Manual Special Judge**: Manual judges are accurate but not scalable; ScaleBox expands coverage through automated synthesis and pre-verification.
- **vs EM / heuristic matching**: EM is fast but systematically rejects correct programs for multi-solution problems; ScaleBox’s core value is incorporating "semantic correctness" into verifiable rewards.
- **Insight**: For code model training, future algorithmic improvements should ideally report reward verifier fidelity; otherwise, performance differences might stem from evaluation noise rather than model learning capability.

## Rating
- Novelty: ⭐⭐⭐⭐☆ High system integration; the combination of automated special judge synthesis and RLVR infrastructure is highly valuable, though individual components are not entirely original.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Provides evidence across efficiency, fidelity, and RL performance; needs validation with more languages and larger scales.
- Writing Quality: ⭐⭐⭐⭐☆ Clear problem definition and system architecture; experimental results are persuasive.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for code LLM training and evaluation infrastructure, especially in directly reducing reward noise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)
- [\[ACL 2026\] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](scicustom_a_framework_for_custom_evaluation_of_scientific_capabilities_in_large_.md)
- [\[ACL 2026\] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models](policyllm_towards_excellent_comprehension_of_public_policy_for_large_language_mo.md)

</div>

<!-- RELATED:END -->
