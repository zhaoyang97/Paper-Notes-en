---
title: >-
  [Paper Note] IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization
description: >-
  [ACL 2025][LLM Alignment][instruction following] This work proposes IOPO (Input-Output Preference Optimization), which introduces input preference modeling in addition to traditional DPO that only optimizes output preferences. It trains the model to learn "which instruction x better matches a given response y," thereby enhancing the model's fine-grained perception of complex, multi-constraint instructions. Additionally, the authors construct the Trace benchmark…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "instruction following"
  - "preference optimization"
  - "input preference"
  - "multi-constraint"
  - "DPO"
  - "Bradley-Terry"
date: 2026-05-08
content_hash: ef7612c06b391b4b
---

# IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization

**Conference**: ACL 2025  
**arXiv**: [2411.06208](https://arxiv.org/abs/2411.06208)  
**Code**: [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/IOPO)  
**Area**: LLM Alignment / Instruction Following  
**Keywords**: instruction following, preference optimization, input preference, multi-constraint, DPO, Bradley-Terry

## TL;DR

This work proposes IOPO (Input-Output Preference Optimization), which introduces input preference modeling in addition to traditional DPO that only optimizes output preferences. It trains the model to learn "which instruction x better matches a given response y," thereby enhancing the model's fine-grained perception of complex, multi-constraint instructions. Additionally, the authors construct the Trace benchmark, consisting of 119-K training samples and 1-K evaluation samples, covering 26 constraint dimensions across 5 major categories.

## Background & Motivation

**Background**: The instruction-following capability of LLMs is crucial, especially with the prevalence of agents and complex applications where user instructions are becoming increasingly complex. A single instruction may contain multiple fine-grained constraints (multi-constraint), such as length limits, formatting requirements, language constraints, and content constraints.

**Limitations of Prior Work**:

**Insufficient Evaluation**: Existing benchmarks like IFEval contain only around 500 evaluation samples with limited constraint dimensions and lack coverage, while also missing support from large-scale training data.

**Lack of Algorithms**: There are no alignment algorithms specifically designed to improve complex instruction-following capabilities. Existing DPO/RLHF methods focus on "comparing the quality of different responses y given the same instruction x."

**Key Challenge**: DPO-based methods only model output preferences. They essentially inform the model whether a response is good or bad, but do not teach the model "why" it is good or bad. Faced with complex instructions containing multiple fine-grained constraints, merely comparing responses makes it difficult for the model to precisely perceive whether each individual constraint is satisfied.

**Goal**: How to enable models to comprehend each constraint within instructions more deeply during the alignment phase?

**Key Insight**: Reversing the direction of preference learning. In addition to the forward direction of "given an instruction, which response is better," a backward direction of "given a response, which instruction is more matching" is introduced. This input preference learning forces the model to focus on fine-grained constraint differences in instructions.

**Core Idea**: Construct input-output preference groups using a quadruple $(x_1, y_1, x_2, y_2)$ to simultaneously learn output and input preferences. These are modeled uniformly through a Group-based Bradley-Terry model.

## Method

### Overall Architecture

IOPO comprises two major contributions:

1. **Trace Benchmark**: An automatically constructed complex instruction dataset covering 26 constraint dimensions across 5 major categories (119-K training and 1-K evaluation samples).
2. **IOPO Alignment Algorithm**: An algorithm based on a group-based Bradley-Terry model that simultaneously optimizes both input and output preferences.

### Trace Benchmark Construction Process

The construction of Trace consists of five phases:

1. **Taxonomy**: Through LLM synthesis from massive open-source simple instructions combined with manual refinement, a taxonomy of 5 constraint categories (Content / Situation / Style / Format / Example) and 26 constraint dimensions is formed.
2. **Constraint Expansion**: Uses LLM to expand simple instructions into complex instructions containing multiple constraints.
3. **Instruction Structuring**: Structures flat text into a three-part format: Task Description + Constraints + Input.
4. **Quality Control**: Employs LLM to inspect and correct invalid scenarios like redundancy and incompleteness.
5. **Response Generation**: Generates responses using LLMs, applies LLM grading (0-10), and retains only perfect-score data (fully satisfying all constraints) for SFT.

Data Statistics: The training set contains 119,345 instructions, and the evaluation set contains 1,042 instructions. The number of constraints per instruction ranges from 1 to 15, averaging 4.36 in training and 4.89 in evaluation. The evaluation set was manually verified by a professional annotation team, achieving a three-annotator agreement rate of 95%.

### Evaluation Protocol

GPT-4o is employed to score each constraint of every instruction (0–10). The overall instruction-following score (IF) is calculated as the average of all constraint scores for a given instruction. An instruction is counted as "correctly followed" only if the average score is exactly equal to the maximum score of 10 (i.e., all constraints are fully met). This constitutes an extremely strict, all-constraint-satisfaction standard.

### IOPO Algorithm: Key Designs

**Preference Data Construction**: Construct a pair of instructions $(x_1, x_2)$, where $x_2$ is modified or has certain constraints removed based on $x_1$, and their corresponding responses are $y_1, y_2$. This yields four input-output pairs:

- Matching Group $\mathcal{G}_1$: $\{(x_1, y_1), (x_2, y_2)\}$ (correctly paired instructions and responses)
- Mismatched Group $\mathcal{G}_2$: $\{(x_1, y_2), (x_2, y_1)\}$ (cross-mismatched instructions and responses)

**Theoretical Derivation**: Based on the reward function reparameterization of DPO, the group-level Bradley-Terry preference probability is expanded as:

$$p(\mathcal{G}_1 \succ \mathcal{G}_2) = \sigma\left(\frac{\Pi_1 + \Pi_2}{2}\right)$$

where $\Pi_1, \Pi_2$ each contain Output preference terms and Input preference terms. Taking $\Pi_1$ as an example:

- **Output Term**: Fixing $x_1$, compare the log-ratio difference between $y_1$ and $y_2$ under $x_1$ → Standard DPO logic.
- **Input Term**: Fixing $y_1$, compare the log-ratio difference between $x_1$ and $x_2$ under $y_1$ → Newly introduced input preference learning.

$\Pi_2$ symmetrically applies the same bidirectional preference modeling for $(x_2, y_2)$.

### Loss & Training

The final IOPO loss maximizes the log-likelihood of matching groups being preferred over mismatched groups:

$$\mathcal{L}_{\text{IOPO}}(\pi_\theta) = -\mathbb{E}_{i \sim D}\left\{\log\left[\sigma\left(\frac{\Pi_1(\pi_\theta) + \Pi_2(\pi_\theta)}{2}\right)\right]\right\}$$

where $i = (x_1, y_1, x_2, y_2)$. Key insight: This loss decomposes naturally into Output preferences (comparing response quality) and Input preferences (comparing instruction compatibility), and they are jointly optimized via summation inside the sigmoid function—reinforcing both preference signals.

### Training Details

- Base models: Qwen2-7B-Instruct, LLaMA3.1-8B-Instruct
- Trace benchmark construction uses Qwen2-72B-Instruct
- SFT learning rate is 1e-4, DPO/IOPO learning rate is 5e-6
- Max length is 6000, trained for 3 epochs with $\beta = 0.1$
- Implemented based on LLaMA-Factory, parallel training on 4x8 GPUs, micro batch size of 1 per GPU.

## Key Experimental Results

### Main Results (Table 3: In-domain Trace + Out-of-domain IFEval/CFBench/ComplexBench)

**Qwen2-7B base:**

| Method | Trace IF-S | Trace IF-M | IFEval S-Acc | IFEval L-Acc | CFBench CSR | CFBench ISR | CFBench PSR | ComplexBench |
|------|-----------|-----------|-------------|-------------|------------|------------|------------|-------------|
| Instruct | 72.5 | 54.5 | 51.6 | 56.4 | 75.8 | 39.1 | 50.2 | 68.1 |
| SFT | 76.0 | 56.1 | 52.3 | 54.2 | 77.8 | 40.4 | 52.9 | 68.2 |
| PPO | 77.0 | 57.7 | 51.4 | 53.8 | 76.2 | 38.8 | 50.6 | 68.6 |
| ORPO | 77.9 | 61.7 | 53.1 | 56.9 | 79.7 | 45.9 | 57.0 | 69.1 |
| SimPO | 78.3 | 63.6 | 52.2 | 57.6 | 78.4 | 45.0 | 57.6 | 67.8 |
| DPO | 79.0 | 67.2 | 52.7 | 58.2 | 80.0 | 45.1 | 57.9 | 70.9 |
| **IOPO** | **82.0 (+3.0)** | **68.9 (+1.7)** | **59.9 (+7.2)** | **63.6 (+5.4)** | **80.7 (+0.7)** | **47.0 (+1.9)** | **58.7 (+0.8)** | **72.6 (+1.7)** |

**LLaMA3.1-8B base:**

| Method | Trace IF-S | Trace IF-M | IFEval S-Acc | IFEval L-Acc | CFBench CSR | CFBench ISR | CFBench PSR | ComplexBench |
|------|-----------|-----------|-------------|-------------|------------|------------|------------|-------------|
| DPO | 79.0 | 69.2 | 71.5 | 76.5 | 80.8 | 48.1 | 59.8 | 70.8 |
| **IOPO** | **81.5 (+2.5)** | **70.7 (+1.5)** | **78.2 (+6.7)** | **81.0 (+4.5)** | **81.8 (+1.0)** | **49.9 (+1.8)** | **61.1 (+1.3)** | **71.8 (+1.0)** |

### Ablation Study (Table 4)

| Model | Configuration | Trace IF-S | IFEval S-Acc | CFBench CSR | ComplexBench |
|------|------|-----------|-------------|------------|-------------|
| Qwen2-7B | IOPO (Full) | 82.0 | 59.9 | 80.7 | 72.6 |
| | w/o Output Pref (Input preference only) | 81.0 (-1.0) | 55.1 (-4.8) | 79.4 (-1.3) | 71.0 (-1.6) |
| | w/o Input Pref (Output preference only) | 80.9 (-1.1) | 56.7 (-3.2) | 79.7 (-1.0) | 71.3 (-1.3) |
| LLaMA3.1-8B | IOPO (Full) | 81.5 | 78.2 | 81.8 | 71.8 |
| | w/o Output Pref | 81.5 (0.0) | 77.3 (-0.9) | 80.6 (-1.2) | 69.2 (-2.6) |
| | w/o Input Pref | 79.0 (-2.5) | 77.9 (-0.3) | 80.9 (-0.9) | 70.1 (-1.7) |

### Computational Overhead Analysis (Table 5)

| Metric | SFT | DPO | IOPO |
|------|-----|-----|------|
| GPU VRAM | 1x | 2x | 4x |
| Training Time | 14.54h | 26.30h | 34.27h |
| Inference Speed | 1x | 1x | 1x |

### Key Findings

1. **Consistent Improvement In- and Out-of-Domain**: IOPO outperforms DPO by 2.75% on average on the in-domain Trace dataset and by 2.84% on average on three out-of-domain benchmarks (Qwen2-7B), demonstrating strong generalization capabilities.
2. **Complementary Input-Output Preferences**: Removing output preference reduces performance by ~2% on average, while removing input preference reduces it by ~1.5% on average, demonstrating that both are indispensable.
3. **Most Pronounced Advantage on IFEval**: IOPO achieves a 7.2% improvement (S-Acc) over DPO on IFEval, significantly outperforming other datasets—indicating that input preference benefits verifiable constraints the most.
4. **Not Driven by Data Volume**: Contrastive experiments controlling the number of tokens (DPO* vs. IOPO) demonstrate that performance gains originate from the algorithmic design rather than more training data.
5. **No Extra Inference Overhead**: Although requiring 4x VRAM during training, the inference speed remains identical—implying zero deployment overhead.

## Highlights & Insights

1. **"Input Preference" is a Paradigm Shift**: All prior alignment methods focused strictly on output preferences. IOPO systematically introduces input preferences for the first time—inferring which instruction fits better given a response—opening a new dimension of preference learning.
2. **Elegant Group-based Bradley-Terry Modeling**: Modeling the quadruple as the preference of matching groups over mismatched groups mathematically unifies input and output preferences in a clean formulation.
3. **Constraint Awareness is Crucial**: Ablation studies show that the contribution of input preference is comparable to that of output preference, indicating that "teaching the model to distinguish constraint differences" is as important as "teaching the model to generate better responses."
4. **Trace Benchmark Fills the Gap**: With 26 constraint dimensions, 5 constraint categories, and 120-K training samples, Trace is the most comprehensive benchmark for complex instruction following to date.
5. **Scalable Data Construction Method**: Generates input preference pairs automatically through constraint modification without requiring additional manual annotations.

## Limitations & Future Work

1. **High Training Cost**: Requiring 4x VRAM and 2.4x training time (vs SFT) limits its application to larger scale models.
2. **Training Set Lacks Human Verification**: The 120-K training set relies solely on automated pipelines, which may introduce noise.
3. **Construction of Input Preference Pairs Depends on Constraint Separability**: For implicit or tightly coupled constraints, modifying a single constraint to generate negative examples might be inaccurate.
4. **Limited Model Scales**: Validated only on 7B/8B models; the effectiveness on larger scales remains to be confirmed.
5. **Lack of Comparison with Constraint-Aware Prompting Methods**: Lightweight alternatives like constraint decomposition and self-verification were not compared.
6. **Future Directions**: Introducing reasoning steps to enhance constraint perception—which is particularly worth exploring in the era of o1/DeepSeek-R1.

## Related Work & Insights

- **vs DPO/SimPO/ORPO**: All focus strictly on output preferences; IOPO's addition of input preferences is the fundamental difference. SimPO removes reference models, and ORPO merges SFT; these are orthogonal to IOPO and can be combined.
- **vs Conifer**: Conifer proposes progressive constraint learning, while Trace is more comprehensive in data scale and constraint taxonomy.
- **vs IFEval/CFBench/ComplexBench**: These are pure evaluation suites, whereas Trace provides both training and evaluation datasets.
- **Insights**: The concept of input preference can be generalized—e.g., in safety alignment, "given a safe response, which prompt is more matching," or in code generation, "given code, which requirements description is more matching."

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | Input preference is an important expansion of the preference learning paradigm, and Group-based BT modeling is clean and elegant |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Very comprehensive with 2 base models, 4 evaluation sets, ablation studies, overhead analysis, and token-controlled experiments |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured, mathematically rigorous, with a natural progression of logic |
| Value | ⭐⭐⭐⭐ | Dual contribution of both a benchmark and an algorithm, driving progress in complex instruction-following; 4x VRAM remains a practical bottleneck |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)
- [\[ICLR 2026\] RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data](../../ICLR2026/llm_alignment/recast_expanding_the_boundaries_of_llms_complex_instruction_following_with_multi.md)

</div>

<!-- RELATED:END -->
