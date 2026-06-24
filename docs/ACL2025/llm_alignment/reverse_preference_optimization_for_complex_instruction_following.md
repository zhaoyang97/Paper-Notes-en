---
title: >-
  [Paper Note] Reverse Preference Optimization for Complex Instruction Following
description: >-
  [ACL 2025][LLM Alignment][Instruction Following] Proposed Reverse Preference Optimization (RPO), which converts arbitrary responses into "perfect" chosen samples by dynamically reversing unsatisfied constraints in the instruction. This eliminates noise in multi-constraint preference pairs and significantly outperforms DPO baselines on multi-turn complex instruction-following tasks.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Instruction Following"
  - "Preference Optimization"
  - "DPO"
  - "Multi-constraint Alignment"
  - "Noise Elimination"
date: 2026-05-08
content_hash: d701e3a2d8eed9d4
---

# Reverse Preference Optimization for Complex Instruction Following

**Conference**: ACL 2025  
**arXiv**: [2505.22172](https://arxiv.org/abs/2505.22172)  
**Code**: Not released  
**Area**: LLM Alignment  
**Keywords**: Instruction Following, Preference Optimization, DPO, Multi-constraint Alignment, Noise Elimination

## TL;DR

Proposed Reverse Preference Optimization (RPO), which converts arbitrary responses into "perfect" chosen samples by dynamically reversing unsatisfied constraints in the instruction. This eliminates noise in multi-constraint preference pairs and significantly outperforms DPO baselines on multi-turn complex instruction-following tasks.

## Background & Motivation

**Background**: Instruction following (IF) is a core capability of LLMs, especially in multi-turn dialogues and multi-constraint scenarios (such as system prompt following), which is crucial for downstream agent applications.

**Limitations of Prior Work**: When an instruction contains multiple constraints, constructing preference pairs based on total score differences introduces noise—chosen samples might perform worse than rejected samples on certain constraints, misleading the model into believing those constraints should not be followed.

**Key Challenge**: Acquiring perfect responses (that satisfy all constraints) is extremely difficult, as the difficulty of sampling perfect responses scales exponentially with the number or complexity of the constraints; non-perfect chosen samples inevitably introduce noise.

**Goal**: How to eliminate noise in multi-constraint preference pairs, ensuring chosen samples are strictly superior to rejected samples, without relying on sampling perfect responses.

**Key Insight**: It is observed that "learning a constraint is essentially identical to learning its opposite" and "a response either follows a constraint or does not." Therefore, responses can be converted into perfect ones by reversing the unsatisfied constraints.

**Core Idea**: Reverse the constraints that are unsatisfied by the chosen response in the original instruction, automatically turning it into a perfect response, thereby constructing noise-free preference pairs.

## Method

### Overall Architecture

RPO consists of three parts: (1) **Role-driven self-play data construction**—simulating multi-turn dialogues between users and the system; (2) **Fine-grained constraint evaluation**—evaluating response adherence to each constraint on a case-by-case basis; (3) **Reverse preference optimization**—reversing constraints to construct noise-free preference pairs for training.

### Key Designs

**Module 1: Reverse Constraint Mechanism**

- **Function**: For two responses with differing constraint adherence, reverse the constraints unsatisfied by the chosen response, making it a perfect response under the new instruction.
- **Mechanism**: For example, given constraint A="response does not exceed 200 words", its opposite $\overline{A}$="response is at least 200 words". After reversal, the chosen response satisfies all constraints under the new instruction, rendering it strictly superior to the rejected response.
- **Design Motivation**: (1) Noise elimination—the chosen response is no worse than the rejected response in any aspect; (2) Amplifying the real gap—transforming a small total score difference of 1 into an actual gap of 5; (3) Simple and efficient—reversing constraints is much easier than sampling perfect responses, self-correction, or back-translation.

**Module 2: Difference Measurement**

- **Function**: Measure the discrepancy between two responses using the number of constraints with differing adherence states, rather than the total score difference.
- **Mechanism**: Even if two responses have the same total score, they can form a valid preference pair as long as they differ on at least one constraint.
- **Design Motivation**: Total score differences underestimate the true variance between responses. For instance, if two responses each satisfy 3 out of 6 constraints but on completely different constraints, the actual difference is 6, not 0.

**Module 3: Adaptive Margin**

- **Function**: Introduce an adaptive margin $\gamma g$ into the DPO loss, where $g$ represents the number of differing constraints between the two responses.
- **Mechanism**: The RPO loss is defined as:
  $$\mathcal{L}_{\text{RPO}} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_i|x_{S_i})}{\pi_{ref}(y_i|x_{S_i})} - \beta\log\frac{\pi_\theta(y_j|x_{S_i})}{\pi_{ref}(y_j|x_{S_i})} - \gamma g)]$$
- **Design Motivation**: Preference pairs with larger differences should receive stronger optimization signals, avoiding uniform treatment of pairs with varying gaps.

**Module 4: Role-Driven Self-Play Data Construction**

- **Function**: Construct SysBank (30K system prompts) and diverse user personas to generate multi-turn dialogues of up to 5 turns via self-play.
- **Mechanism**: Collect system roles from real GPT Store data and expand them into complete personas containing constraints; the user side is expanded into detailed personas.
- **Design Motivation**: Existing data often samples and concatenates single-turn instructions from a constraint pool, which lacks diversity and fails to construct coherent multi-turn dialogues.

### Loss & Training

- Based on the Llama-Factory framework, utilizing LoRA + DeepSpeed.
- SFT learning rate is 5e-5, and DPO/KTO/RPO learning rate is 5e-4.
- 5 responses are sampled per turn to construct preference pairs.
- $\gamma=0.05$, $\beta=0.1$.

## Key Experimental Results

### Main Results

Results on two multi-turn complex instruction-following benchmarks:

| Method | SysBench CSR | SysBench ISR | SysBench SSR | SysBench Avg | Multi-IF Step1 | Multi-IF Step2 | Multi-IF Step3 | Multi-IF Avg |
|------|-------------|-------------|-------------|-------------|---------------|---------------|---------------|-------------|
| GPT-4o | 89.72 | 81.71 | 61.51 | 77.65 | 84.70 | 76.00 | 68.33 | 76.34 |
| Claude-3.5 Sonnet | 94.64 | 89.68 | 74.36 | 86.23 | 83.87 | 74.87 | 69.80 | 76.18 |
| Llama-3.1 8B DPO | 80.56 | 66.67 | 40.37 | 62.53 | 74.57 | 66.90 | 58.50 | 66.66 |
| **Llama-3.1 8B RPO** | **83.10** | **71.27** | **46.99** | **67.12** | **77.50** | **69.47** | **60.57** | **69.18** |
| Llama-3.1 70B DPO | 85.91 | 75.12 | 50.89 | 70.64 | 84.37 | 75.63 | 67.76 | 75.92 |
| **Llama-3.1 70B RPO** | **89.54** | **81.76** | **62.20** | **77.83** | **86.47** | **77.77** | **70.21** | **78.15** |

### Ablation Study

RPO improvement over DPO across different model families:

| Model | SysBench Avg (DPO→RPO) | Multi-IF Avg (DPO→RPO) |
|------|----------------------|----------------------|
| Llama-3.1 8B | 62.53 → 67.12 (+4.59) | 66.66 → 69.18 (+2.52) |
| Llama-3.1 70B | 70.64 → 77.83 (+7.19) | 75.92 → 78.15 (+2.23) |
| Qwen-2.5 7B | 59.49 → 63.06 (+3.57) | 63.11 → 66.27 (+3.16) |
| Qwen-2.5 72B | 75.84 → 76.97 (+1.13) | 76.10 → 78.35 (+2.25) |

### Key Findings

- RPO on the 8B model achieves performance gains of 2.5/4.6/6.6 points on SysBench CSR/ISR/SSR respectively over the DPO baseline.
- The 70B RPO model outperforms GPT-4o on most metrics.
- The chosen-rejected reward gap grows more significantly during RPO training, demonstrating its ability to distinguish positive and negative samples more effectively.
- Larger constraint difference scores (gaps) indeed help improve performance, verifying the design rationale of the reversal mechanism to amplify core differences.
- RPO constructs noise-free preference pairs without requiring extra sampling or filtering, making it engineering-friendly and simple to implement.

## Highlights & Insights

1. The idea of **reversing constraints** is extremely elegant and simple—transforming a data quality problem into an instruction rewriting problem, completely eliminating preference noise in multi-constraint scenarios.
2. The analysis of noise sources in multi-constraint preference pairs (categorized into 6 cases) is highly clear and systematically reveals the essence of the problem.
3. The SysBank dataset (containing 30K real-world system prompts) provides the community with a valuable resource for multi-turn IF training.
4. The method is highly scalable—demonstrating consistent effectiveness across sizes from 8B to 70B, and across different model families (Llama and Qwen).

## Limitations & Future Work

1. Constraint reversal relies on the LLM to generate high-quality opposite constraints, which may fail for highly abstract or complex constraints.
2. In multi-turn dialogues, dependencies may exist between constraints, making independent reversal of individual constraints not always reasonable.
3. Evaluation still relies on the LLM-as-judge paradigm, and constraint-by-constraint evaluation can introduce its own errors.
4. The integration of RPO with online DPO, self-play iterative training, and other methods has not yet been explored.

## Related Work & Insights

- **IOPO** (Zhang et al., 2024b): Pairs perfect responses from two different instructions as mutually rejected while aligning input-output preferences—complementary to the RPO approach.
- **SPAR** (Cheng et al., 2024a): Suppresses noise through a tree-search self-correction process—more complex but shares the same goal as RPO.
- **CRAB** (Qi et al., 2024): Uses back-translation to generate instructions from responses to improve data quality—similar in direction to RPO's instruction rewriting.
- Core Insight: In multi-preference alignment, optimizing data quality is far more important than increasing data volume.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The constraint reversal idea is simple yet powerful, demonstrating genuine originality in multi-constraint IF scenarios.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers two benchmarks, four model sizes/families, and includes training dynamics analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — In-depth analysis of the problem (with a 6-case classification figure) and clear description of the methodology.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and generalizable solution for preference learning in multi-constraint instruction following.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](iopo_input_output_preference.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)

</div>

<!-- RELATED:END -->
