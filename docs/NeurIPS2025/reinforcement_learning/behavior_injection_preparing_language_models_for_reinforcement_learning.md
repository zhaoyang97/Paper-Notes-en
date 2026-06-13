---
title: >-
  [Paper Note] Behavior Injection: Preparing Language Models for Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][RL fine-tuning] This paper identifies the root cause of inconsistent LLM responses to RL fine-tuning. Through per-step influence analysis…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "RL fine-tuning"
  - "behavior injection"
  - "data augmentation"
  - "GRPO"
  - "per-step influence"
  - "reasoning"
date: 2026-05-08
content_hash: 4651d0234b87c3b4
---

# Behavior Injection: Preparing Language Models for Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.18917](https://arxiv.org/abs/2505.18917)  
**Code**: [Project](https://bridge-llm-reasoning.github.io/)  
**Area**: LLM / Reinforcement Learning
**Keywords**: RL fine-tuning, behavior injection, data augmentation, GRPO, per-step influence, reasoning

## TL;DR
This paper identifies the root cause of inconsistent LLM responses to RL fine-tuning. Through per-step influence analysis, it reveals that RL effectiveness depends on (1) the distribution of rollout accuracy (moderate is optimal) and (2) data co-influence magnitude. The proposed BRIDGE method injects exploration/exploitation behaviors during SFT, boosting subsequent RL gains from 6% to 46.6%.

## Background & Motivation
**Background**: RL methods such as GRPO have achieved notable success in improving LLM reasoning (e.g., DeepSeek-R1, Qwen series), yet RL gains vary dramatically across models and datasets—sometimes +30%, other times merely +2%.

**Limitations of Prior Work**: It remains unclear why some models are "ready" for RL fine-tuning while others are not. Existing work focuses on final outcomes without analyzing the learning dynamics at each step during RL. There is a lack of theoretical guidance on what properties of an SFT model make it most suitable for subsequent RL.

**Key Challenge**: RL requires models to explore (producing diverse rollouts), whereas SFT trains models to imitate fixed answers (lacking exploratory behavior). Post-SFT models may be "too confident," rendering the RL signal zero—there is no gradient when rollouts are uniformly correct or uniformly incorrect.

**Goal**: (1) Theoretically analyze the conditions under which RL fine-tuning succeeds; (2) Design data augmentation strategies during the SFT stage to prepare models for RL.

**Key Insight**: Deriving an influence formula from the per-step influence of GRPO reveals that the key factor is $\alpha(1-\alpha)$ (where $\alpha$ denotes rollout accuracy)—moderate accuracy is optimal, since influence vanishes when accuracy is 0% or 100%.

**Core Idea**: Inject exploration behaviors ("Let me try this step first... that's wrong, let me try a different direction") and exploitation behaviors (subgoal decomposition) into SFT data, so that the rollout accuracy distribution falls within the optimal range for RL.

## Method

### Overall Architecture
Theoretical analysis (per-step influence) → identification of two key factors → design of BRIDGE data augmentation → two-stage SFT + RL training.

### Key Designs

1. **Per-Step Influence Analysis (Proposition 3.1)**:

    - Function: Derives the influence of a single training query on the target domain within GRPO.
    - Formula: $\Delta J \approx \eta \cdot \mathbb{E}[A \cdot \sqrt{\alpha(1-\alpha)} \cdot \mathcal{K}_{\theta}]$
    - Two key factors: (a) $\alpha(1-\alpha)$: signal is strongest at moderate accuracy (zero influence at 0% or 100%); (b) $\mathcal{K}_{\theta}$: the co-influence kernel between training and target data (governs transferability).
    - Design Motivation: Explains why some models yield large RL gains (moderate accuracy + high co-influence).

2. **BRIDGE Data Augmentation (DAG-based)**:

    - Function: Extracts a DAG from reasoning chains (nodes = variables, edges = dependencies) and injects three types of behaviors onto the DAG.
    - **Exploration behavior**: Attempt to solve an unsolvable node → reflect ("Wait, this cannot be solved yet; I should work on something else first") → probability $p=0.1$.
    - **Exploitation behavior 1 (subgoal computation)**: Decompose complex equations into steps → inject at every step.
    - **Exploitation behavior 2 (information aggregation)**: Synthesize relevant information before deriving conclusions → probability $p=0.1$.
    - Design Motivation: Exploration behaviors accustom the model to "backtracking after errors" (yielding moderate accuracy); exploitation behaviors enhance co-influence (structured reasoning patterns).

3. **Minimalist Design**:

    - Does not increase data volume (2,000 augmented samples rather than copies).
    - Comparison: PP-Aug/RC-Aug require 8,000 samples.
    - Only the reasoning chain in the answer is modified; the question itself is unchanged.

### Loss & Training
SFT: standard cross-entropy loss (with BRIDGE-augmented data). RL: GRPO (standard).

## Key Experimental Results

### Main Results (iGSM Math Problems, In-Dist/OOD)

| Model | Method | SFT | RL | **RL Gain** |
|------|------|-----|-----|---------|
| Qwen-1.5B | Vanilla | 40.2 | 46.2 | +6.0% |
| Qwen-1.5B | **BRIDGE** | 44.8 | 91.4 | **+46.6%** |
| Qwen-3B | Vanilla | 38.0 | 57.2 | +19.2% |
| Qwen-3B | **BRIDGE** | 59.2 | 89.6 | **+30.4%** |
| Llama-1B | Vanilla | 29.6 | 31.0 | +1.4% |
| Llama-1B | **BRIDGE** | 40.4 | 64.6 | **+24.2%** |

### Ablation Study

| Configuration | Key Findings | Notes |
|------|---------|------|
| Rollout accuracy distribution | BRIDGE yields the largest proportion of $0 < \text{acc} < 1$ | Validates theory: moderate accuracy is optimal |
| Co-influence measurement | BRIDGE is 2–3× higher than baseline | Structured reasoning enhances transferability |
| Remove exploration behavior | RL gain drops significantly | Exploration is a necessary condition |
| Remove exploitation behavior | RL gain drops moderately | The two behaviors are complementary |
| Injection probability $p$ | Any $p > 0$ is beneficial | Not sensitive to the exact value |
| PromptBench arithmetic | BRIDGE +53.0% (Qwen-1.5B) | Generalizes to other reasoning tasks |

### Key Findings
- The gap of 46.6% vs. 6.0% RL gain is the most striking result—demonstrating that "preparation" matters more than "training."
- Llama-1B improves from +1.4% to +24.2%—BRIDGE transforms a model that is "nearly untrainable" into one that is "substantially improvable."
- Influence analysis validates the theoretical predictions: BRIDGE's co-influence is indeed 2–3× that of the vanilla baseline.

## Highlights & Insights
- **"Prerequisites for RL success"**: This is a widely overlooked yet critically important question—not all models are ready for RL. The paper provides dual answers: theoretical ($\alpha(1-\alpha)$ optimality) and practical (behavior injection).
- **The centrality of exploration behavior**: The self-correcting reasoning pattern—"Let me try... that's wrong... let me try a different direction"—is a prerequisite for RL to work, because it drives rollout accuracy toward the moderate regime.
- **A complete theory→design→validation pipeline**: Influence analysis → identification of two key factors → design of BRIDGE → measurement of influence for validation → observation of RL gains for validation.

## Limitations & Future Work
- Validation is limited to mathematical reasoning tasks; applicability to code generation, dialogue, and other tasks remains unexplored.
- DAG extraction requires structured reasoning chains—unstructured tasks would need an oracle LLM (e.g., GPT-4) to extract the structure.
- Dataset scale is relatively small (2,000–5,000 samples); scaling behavior under larger data regimes is unknown.
- Behavior injection may produce side effects—injecting "unsafe" behaviors could introduce risks.

## Related Work & Insights
- **vs. DeepSeek-R1 (2025)**: R1 applies RL directly without a specialized SFT preparation stage; BRIDGE demonstrates that a preparation stage can make RL substantially more effective.
- **vs. Rejection Sampling Fine-tuning**: Rejection sampling filters out incorrect rollouts; BRIDGE improves the SFT model so that more rollouts become informative.
- **vs. Process-level Reward Models**: PRMs improve the signal during the RL stage; BRIDGE improves model "readiness" during the SFT stage—the two approaches are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Per-step influence analysis reveals the theoretical conditions for successful RL fine-tuning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models × multiple tasks × influence measurement × ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The theory→design→validation narrative is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Provides important methodological guidance for the LLM + RL community.
- **Implications for the RLHF community**: Beyond reasoning tasks, RL fine-tuning for dialogue and creative tasks may also benefit from a similar SFT preparation stage.
- **Relationship to curriculum learning**: BRIDGE can be viewed as a special form of curriculum design—rather than progressively increasing difficulty, it injects behavioral patterns.
- **Complementarity with self-play**: Self-play generates high-quality interaction data; BRIDGE teaches the model how to learn from such interactions.
- **Transferable design principles**: The two principles—moderate accuracy optimality and co-influence maximization—are not restricted to GRPO and can guide other RL algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
