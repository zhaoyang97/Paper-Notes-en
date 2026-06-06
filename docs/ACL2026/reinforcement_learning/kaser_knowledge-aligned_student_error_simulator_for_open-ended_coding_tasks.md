---
title: >-
  [Paper Note] KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks
description: >-
  [ACL2026][Reinforcement Learning][Student modeling] KASER first estimates student mastery of knowledge points, then utilizes GRPO with a hybrid reward of "code similarity + error matching + diversity" to train a code gen…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "Student modeling"
  - "programming education"
  - "error simulation"
  - "GRPO"
  - "knowledge tracing"
date: 2026-05-08
content_hash: 785c1be856a93874
---

# KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks

**Conference**: ACL2026  
**arXiv**: [2601.06633](https://arxiv.org/abs/2601.06633)  
**Code**: https://github.com/umass-ml4ed/code_personalization  
**Area**: Reinforcement Learning / AI in Education / Code Generation  
**Keywords**: Student modeling, programming education, error simulation, GRPO, knowledge tracing  

## TL;DR
KASER first estimates student mastery of knowledge points, then utilizes GRPO with a hybrid reward of "code similarity + error matching + diversity" to train a code generator capable of simulating programming errors consistent with a student's knowledge state.

## Background & Motivation
**Background**: Open-ended programming problems reveal specific knowledge deficiencies in students, such as errors in conditional logic, string indexing, or return statements. If educational AI can predict what code students will write and what errors they will make, it can assist teachers in diagnosing difficulties early and generating personalized feedback.

**Limitations of Prior Work**: Existing student code simulation methods either rely on student submission history as context or merely pursue surface similarity between generated and real code. Since LLMs are pre-trained on vast amounts of high-quality code, they naturally tend to generate correct and standard code; even when SFT-ed on student code, they suffer from mode collapse, failing to capture the diversity of real student errors.

**Key Challenge**: Educational scenarios require models to "err like students," whereas standard code models are trained to "write correctly like experts." Relying solely on code similarity causes models to learn surface styles, making it difficult to align with student knowledge gaps and specific error types.

**Goal**: Given student submission history and a subsequent programming problem, predict the student's next submission code—especially the errors therein—and ensure the errors are consistent with the student's mastery of relevant Knowledge Components (KCs).

**Key Insight**: The authors explicitly estimate the student's knowledge state as an interpretable KC mastery vector, which is then incorporated into the LLM prompt. Subsequently, RL rewards are used to simultaneously constrain code similarity, error overlap, and cohort diversity.

**Core Idea**: Instead of merely having the LLM imitate student code, the model learns "which knowledge deficiencies correspond to which errors" under the condition of interpretable knowledge profiles via hybrid rewards.

## Method

### Overall Architecture
KASER consists of three components. The first is the Student Knowledge Estimator, which utilizes a knowledge tracing model to estimate mastery levels for each KC based on past submissions. The second is the Student Code Predictor, which inputs the problem text and relevant KC mastery as a prompt into Qwen2.5-Coder 7B Instruct, initially using SFT to predict student code. The third is GRPO-based RL training, which further optimizes the code predictor using hybrid rewards to generate code that more closely resembles real student submissions with matching errors and diverse outputs.

Inputs include the student's history, the problem description, relevant KCs, and student mastery levels. Outputs are the predicted student code and a set of potential errors. Evaluation is conducted at the student-problem pair level (comparing predicted code to real submissions) and the problem level (evaluating coverage of student error types and code diversity).

### Key Designs
1. **Student Knowledge Estimator**:
	- **Function**: Compresses student submission history into an interpretable knowledge mastery vector.
	- **Mechanism**: A knowledge tracing model generates a student state $h_t$, and a linear layer with sigmoid produces $m_t \in [0,1]^k$, where each dimension corresponds to a KC mastery level. For the next problem, only relevant KCs are taken and averaged via a compensation model to estimate global accuracy, trained with BCE loss.
	- **Design Motivation**: Using student IDs or historical code directly makes it difficult to explain "why an error occurred." KC mastery allows the generation of errors to correspond with specific knowledge point deficiencies.

2. **Knowledge-guided code predictor**:
	- **Function**: Allows the LLM to explicitly perceive the student's knowledge profile during code prediction.
	- **Mechanism**: The prompt contains the problem description and the mastery level of each relevant KC, e.g., "Conditional logic mastery = 0.21". Qwen2.5-Coder 7B Instruct generates predicted code token-by-token. During the SFT phase, standard cross-entropy is used to learn from real student submissions.
	- **Design Motivation**: SFT provides basic task adaptation, but SFT alone tends to produce averaged, corrected code. Knowledge conditioning provides aligned control variables for subsequent RL.

3. **GRPO hybrid reward**:
	- **Function**: Simultaneously optimizes code similarity, error matching, and generation diversity to avoid over-optimizing a single metric.
	- **Mechanism**: For each input, $G=5$ candidate codes are sampled and scored using $R=R_{Sim}+R_{Error}+R_{Div}$. $R_{Sim}$ is CodeBLEU; $R_{Error}$ is the IoU between predicted and real error sets (or 1 if both are correct); $R_{Div}=1-\max CodeBLEU(\hat{c},\hat{c_i})$, encouraging diversity among candidates in the same group. GRPO uses intra-group z-score normalized rewards to form advantages, with a KL penalty to constrain policy shift.
	- **Design Motivation**: Code similarity ensures basic syntax and solutions are close to real submissions; error IoU directs the model's focus to actual student errors; the diversity reward prevents SFT-style mode collapse.

### Loss & Training
The knowledge estimator is trained with BCE on the correctness of the next submission. The code predictor is first SFT-ed using token-level cross entropy. In the RL phase, GRPO is employed with an objective consisting of a clipped surrogate loss and a KL penalty against the reference policy. For each problem-student pair, 5 candidates are sampled, and the three reward components are weighted equally. The authors note that equal weighting was effective in experiments, leaving fine-tuned reward weights for future work.

Experiments utilize two real-world programming education datasets: CodeWorkout (246 students, 50 Java problems, 50 KCs, 10,834 submissions) and FalconCode (447 students, 84 Python problems, 60 KCs, 11,194 submissions). Only the first submission for each problem is analyzed to focus on initial knowledge deficiencies rather than the debugging process.

## Key Experimental Results

### Main Results
| Dataset | Method | CodeBLEU@1 | CodeBLEU@5 | IoU@1 | IoU@5 |
|--------|------|------------|------------|-------|-------|
| CodeWorkout | Student SFT | 0.501 | 0.565 | 0.115 | 0.244 |
| CodeWorkout | KASER | 0.524 | 0.599 | 0.157 | 0.276 |
| FalconCode | Student SFT | 0.642 | 0.670 | 0.153 | 0.270 |
| FalconCode | KASER | 0.668 | 0.692 | 0.178 | 0.303 |

### Ablation Study
| Dataset | Method | Cosine Distance ↑ | CodeBLEU Complement ↑ | Error IoU ↑ | $\chi^2$ Distance ↓ |
|--------|------|-------------------|----------------|-------------|----------------------|
| CodeWorkout | Student SFT | 0.082 | 0.480 | 0.700 | 109.85 |
| CodeWorkout | KASER | 0.088 | 0.520 | 0.750 | 104.97 |
| FalconCode | Student SFT | 0.279 | 0.600 | 0.755 | 51.89 |
| FalconCode | KASER | 0.298 | 0.643 | 0.817 | 45.77 |

### Key Findings
- At the per-student-problem-pair level, KASER significantly outperforms all baselines in code similarity and error matching across both datasets, with reported gains being statistically significant ($p<0.05$).
- At the per-problem level, KASER improves error coverage and code diversity, indicating it not only resembles a specific student's submission but also better covers the variety of errors different students might make on a single problem.
- When analyzed by error type, KASER achieves a precision of 1.00 for logical, runtime, and syntax errors, with higher recall than Student SFT. However, syntax recall remains low, indicating that it is still challenging for pre-trained code models to generate syntax errors.
- Reward ablation is clear: removing $R_{Sim}$ decreases CodeBLEU@5 on CodeWorkout by ~16%; removing $R_{Error}$ decreases IoU@1 on CodeWorkout by ~36% and increases $\chi^2$ distance by ~11%; removing $R_{Div}$ decreases cosine diversity on FalconCode by ~12%.

## Highlights & Insights
- The paper identifies the unique nature of educational code generation: the goal is not to "write the correct answer," but to "simulate how a student with specific knowledge gaps would write it incorrectly."
- The hybrid reward design aligns perfectly with the task. CodeBLEU, error IoU, and diversity correspond to surface code, error semantics, and cohort variation, respectively.
- KC mastery prompts provide educational interpretability. Qualitative cases show that low numerical mastery levels for comparison operators lead to relational errors, while low return statement mastery leads to missing returns.
- Insight for LLM educational applications: Student simulation should not rely solely on personas or historical examples but should explicitly model knowledge states and incorporate "error plausibility" into the training objective.

## Limitations & Future Work
- Error labeling and evaluation depend on LLM-as-a-judge without access to test case information. Incorporating compilation results, unit test failure patterns, and execution traces could provide finer-grained error labels.
- The IoU metric is strict, and absolute error matching scores remain low for all methods, showing that student error prediction is far from solved.
- The study only analyzes the first submission per problem, neglecting student debugging processes; real feedback systems require understanding how errors evolve.
- Pre-trained code LLMs struggle to "intentionally" generate syntax errors; syntax error simulation remains a weakness.
- Student simulation may pose fairness issues. If training data lacks coverage for minority groups, generated student profiles and error predictions may suffer from calibration bias.

## Related Work & Insights
- **vs Open-ended KT**: Traditional KT estimates knowledge changes but rarely simulates open-ended code or fine-grained errors; KASER integrates KT profiles into code generation.
- **vs ParaStudent**: ParaStudent uses submission history to predict future code but lacks explicit knowledge states; KASER emphasizes interpretable alignment of errors with KC mastery.
- **vs Student SFT**: SFT improves code similarity but tends to generate correct, averaged code; KASER mitigates mode collapse through error and diversity rewards.
- **Inspiration for future research**: KASER can be extended to mathematical derivations, open-ended Q&A, or debugging trajectory simulation, using knowledge states to control the generation of "reasonable errors."

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Using GRPO hybrid rewards for knowledge-aligned student error simulation is innovative and the task definition is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Solid evaluation across two real-world datasets with two-level metrics and ablation studies, though large-scale human labeling and debugging sequence analysis are absent.
- **Writing Quality**: ⭐⭐⭐⭐☆ The method and experimental tables are clear, with sufficient educational motivation.
- **Value**: ⭐⭐⭐⭐☆ Significant potential for application in programming education, student simulation, and personalized feedback systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](../../ICLR2026/reinforcement_learning/helix_evolutionary_reinforcement_learning_for_open-ended_scientific_problem_solv.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](../../ICLR2026/reinforcement_learning/from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[ACL 2026\] NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks](navimaster_learning_a_unified_policy_for_gui_and_embodied_navigation_tasks.md)
- [\[ACL 2026\] A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks (EAGLET)](a_goal_without_a_plan_is_just_a_wish_efficient_and_effective_global_planner_trai.md)
- [\[ICLR 2026\] LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection](../../ICLR2026/reinforcement_learning/laddersym_a_multimodal_interleaved_transformer_for_music_practice_error_detectio.md)

</div>

<!-- RELATED:END -->
