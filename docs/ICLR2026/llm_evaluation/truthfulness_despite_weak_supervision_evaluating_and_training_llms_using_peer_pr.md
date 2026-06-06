---
title: >-
  [Paper Note] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction
description: >-
  [ICLR 2026][LLM Evaluation][Peer Prediction] This paper proposes applying the Peer Prediction mechanism from game theory to LLM evaluation and training. By measuring the mutual predictability of participants' answers…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Peer Prediction"
  - "Honesty Training"
  - "Deception Resistance"
  - "Weak Supervision"
date: 2026-05-08
content_hash: 0767d00b11ac9aaf
---

# Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction

**Conference**: ICLR 2026
**arXiv**: [2601.20299](https://arxiv.org/abs/2601.20299)  
**Code**: GitHub repository (referenced in the paper)  
**Area**: LLM Evaluation
**Keywords**: Peer Prediction, LLM Evaluation, Honesty Training, Deception Resistance, Weak Supervision

## TL;DR
This paper proposes applying the Peer Prediction mechanism from game theory to LLM evaluation and training. By measuring the mutual predictability of participants' answers, the method distinguishes honest from deceptive responses without requiring ground-truth labels, thereby incentivizing truthfulness. It exhibits a striking *inverse scaling* property — weaker experts are actually more resistant to deception by stronger models.

## Background & Motivation

**Background**: LLM evaluation and post-training rely on supervision signals. Mainstream approaches include human feedback (RLHF) and LLM-as-a-Judge. As model capabilities improve, these methods face the *scalable oversight* problem — superhuman models can exploit and deceive evaluators weaker than themselves.

**Limitations of Prior Work**: LLM-as-a-Judge performs **worse than random guessing** when facing deceptive models 5–20× more capable than the judge. Human evaluators are also susceptible to sycophancy and reward over-optimization. No existing evaluation method offers game-theoretic deception-resistance guarantees against stronger models.

**Key Challenge**: Strong supervision is unavailable on hard tasks (evaluators lack sufficient capability), yet weak supervision is easily exploited by stronger models.

**Goal**: How can weak supervision accurately evaluate strong LLMs? How can models be incentivized to remain truthful during training?

**Key Insight**: The paper imports the Peer Prediction mechanism from the mechanism design literature — a game-theoretically proven information elicitation method. The core intuition is that honest and informative answers better help predict others' answers, since they contain more information about the true state of the world. Answer quality can thus be measured by *mutual predictability* without requiring ground-truth labels.

**Core Idea**: Use the mutual predictability of multiple models' answers as a measure of honesty, leveraging incentive compatibility from game theory to ensure that truthful reporting is the optimal strategy.

## Method

### Overall Architecture
**Input**: a question $Q$, answers $\{A_1, \ldots, A_n\}$ from $n$ participant models, and $m$ expert models $\{J_1, \ldots, J_m\}$. **Output**: a score $S_i^A$ for each participant. **Pipeline**: all participants rotate through source-target pairings; experts evaluate how much the source's answer helps predict the target's answer. Scores can be used for ranking-based evaluation or to construct DPO training pairs.

### Key Designs

1. **Peer Prediction Evaluation Pipeline (Core Algorithm)**:

    - **Function**: Score each participant's answer by measuring its honesty and informativeness.
    - **Mechanism**: Three roles — Source (the evaluated party), Target (the predicted party), and Expert (the predictor). The score of Source $s$ is determined by how much it helps the expert predict Target $t$: $S_s^A += \log \Pr_j(A_t | A_s) - \log \Pr_j(A_t)$, i.e., the log increase in the expert's predictive probability of the target's answer after observing the source's answer. All source-target-expert combinations are rotated, yielding $n^2 m$ rounds in total.
    - **Design Motivation**: Honest, informative answers better help predict others' answers because they contain more information about the true state of the world. This is more robust than directly judging answer correctness, as it requires no ground truth.

2. **Theoretical Guarantees of Incentive Compatibility**:

    - **Function**: Prove that truthful reporting is the optimal strategy for all participants and experts (Bayesian Nash Equilibrium).
    - **Mechanism**: **Theorem 1** (Shared Prior): Under a shared prior $\mathcal{P}$, truthful reporting $A_i = A_i^*$ constitutes a Bayesian Nash Equilibrium with maximum expected payoff. **Theorem 2** (Heterogeneous Priors; new contribution): When participants' and experts' priors are independently sampled from the same distribution $\mathcal{D}$, truthful reporting is an $\epsilon$-Bayesian Nash Equilibrium provided $m, n \geq \frac{16(I_0 + L_0)}{\epsilon} \log(\frac{I_0 + L_0}{\epsilon} + \frac{|\mathcal{A}|}{\delta})$.
    - **Design Motivation**: Theorem 1 applies a classical result; Theorem 2 is a novel contribution that relaxes the shared-prior assumption, allowing differences in "worldview" as long as the participant/expert pool is sufficiently large and representative.

3. **Peer Prediction Training Pipeline**:

    - **Function**: Use Peer Prediction scores as reward signals for DPO training to restore honesty degraded by malicious fine-tuning.
    - **Mechanism**: For each question, multiple participants generate responses; responses are ranked by $S_i^A$; the highest- and lowest-scoring responses form preference pairs for DPO training. Four Llama-3.1-8B clones serve as participants (one instructed to deceive), and SmolLM-135M serves as the expert. A "malicious model" is first created via LoRA fine-tuning on 50k deceptive samples, then restored using DPO on 120k Peer Prediction preference pairs.
    - **Design Motivation**: Incentive compatibility guarantees that at the training equilibrium, the optimal strategy is truthfulness — translating game-theoretic guarantees into local optimality of the training signal.

4. **Inverse Scaling Property**:

    - **Function**: Discover and validate that Peer Prediction's deception resistance increases as the capability gap between experts and participants widens.
    - **Mechanism**: Experiments are conducted on models ranging from 0.135B to 405B parameters, using logistic regression cross-entropy to measure how well scores predict honesty. Results show that the larger the participant's advantage over the expert, the better Peer Prediction distinguishes honest from deceptive answers — in sharp contrast to LLM-as-a-Judge, which exhibits positive scaling (weaker judges perform worse).
    - **Design Motivation**: This is the most surprising finding — weak experts paired with Peer Prediction outperform strong judges paired with LLM-as-a-Judge.

### Loss & Training
- Participant score: $S_s^A = \sum_{t,j} [\log \Pr_j(A_t | A_s) - \log \Pr_j(A_t)]$
- Expert score: $S_j^J = \sum_{s,t} [\log \Pr_j(A_t | A_s) + \log \Pr_j(A_t)]$ (a proper scoring rule rewarding experts for honestly reporting probabilities)
- Training: standard DPO loss; preference pairs are constructed by ranking responses according to Peer Prediction scores.

## Key Experimental Results

### Main Results
Honesty restoration after malicious fine-tuning (37,079 questions, 85 domains):

| Method | Expert Size | In-Domain Accuracy Recovery | Out-of-Domain Accuracy Recovery | Notes |
|--------|------------|----------------------------|----------------------------------|-------|
| After Malicious Fine-tuning | — | Baseline (large drop) | Baseline | Instructed to deceive |
| LLM-as-Judge | 0.135B | Marginal recovery | Marginal recovery | Weak judge cannot detect deception |
| LLM-as-Judge | 7B | Moderate recovery | Moderate recovery | Stronger judge shows some effect |
| **Peer Prediction** | **0.135B** | **Near-full recovery** | **Near-full recovery** | 0.135B expert surpasses 7B judge |

### Ablation Study (Scaling of Deception Resistance)

| Scenario | Capability Gap | Peer Prediction | LLM-as-Judge | Notes |
|----------|---------------|-----------------|--------------|-------|
| 8B participant, 7B expert | ~1× | Slightly better | Comparable | Small gap, similar performance |
| 8B participant, 0.135B expert | ~60× | Significantly better | Worse than random | Inverse scaling |
| 27B participant, 0.135B expert | ~200× | Best | Far below random | PP strongest; Judge fails completely |
| 405B participant, 7B expert | ~58× | Effectively distinguishes | Fails | Validated on very large models |

### Key Findings
- **0.135B expert Peer Prediction outperforms 7B LLM-as-a-Judge**: in honesty restoration training, a weak expert paired with a strong mechanism is more effective than a strong judge paired with a weak mechanism.
- **Inverse scaling**: counterintuitively, the weaker the expert relative to the participant, the stronger Peer Prediction's deception resistance.
- **LLM-as-a-Judge completely fails at large capability gaps**: it performs worse than random guessing against deceptive models 5–20× its size, and fails entirely at 100× gaps.
- **Triple scaling**: deception resistance improves with (1) larger capability gap, (2) more participants, and (3) more experts.
- **Expert ensembling**: ensembling multiple weak experts outperforms a single best expert, with gains growing steadily with the number of experts.

## Highlights & Insights
- **Cross-domain application of game theory to AI safety**: importing the mature Peer Prediction theory from mechanism design into LLM evaluation is a compelling interdisciplinary transfer. The key insight is that one need not know the correct answer — only the structural relationships among answers need to be exploited.
- **Deep reason for inverse scaling**: stronger models produce more informative and distinctive answers, generating stronger mutual-predictability signals. The "ignorance" of weak experts becomes an advantage — their predictions rely more on information provided by the source than on their own capability.
- **Practical meaning of incentive compatibility**: the mechanism penalizes not only deceptive answers but also answers that omit information — models are incentivized to provide complete, informative responses, which is richer than a simple correct/incorrect judgment.
- **Practical value of Theorem 2**: relaxing the shared-prior assumption accommodates models with different "worldviews," provided the pool is sufficiently large and representative, making the method viable in realistic multi-model ecosystems.

## Limitations & Future Work
- **Collusion is unresolved**: the theoretical guarantees cover only unilateral deception; coordinated deception among multiple participants is not addressed. The appendix includes a preliminary discussion but falls short of a complete treatment.
- **Computational overhead**: the $n^2 m$ evaluation rounds scale quadratically/linearly with the number of participants and experts, posing a challenge for large-scale deployment.
- **Granularity of free-text evaluation**: the current approach treats a free-text answer as a monolithic unit $A_i$, without fine-grained handling of partially correct responses.
- **Prior distribution assumption**: Theorem 2 requires that participants' and experts' priors be sampled from the same distribution; differences in actual training data distributions may violate this condition.
- **Directions for improvement**: explore collusion-resistant Peer Prediction variants; investigate token-level fine-grained mutual predictability; extend the method to online RLHF training (currently only offline DPO has been tested).

## Related Work & Insights
- **vs. LLM-as-a-Judge**: LLM-as-a-Judge requires the judge to be at least as capable as the evaluated model and is easily deceived. Peer Prediction has no such requirement, and its inverse scaling property makes it more reliable under weak supervision.
- **vs. Debate (Brown-Cohen et al.)**: the debate approach requires two adversarial models to challenge each other, resulting in a more complex structure. Peer Prediction is more parsimonious — it requires only the mutual predictability of answers, without adversarial interaction.
- **vs. RLHF**: RLHF requires high-quality human or AI preference annotations (strong supervision). Peer Prediction requires no ground-truth labels whatsoever, making it a genuinely weak-supervision method.
- This paper has direct implications for the *superalignment* direction: if future models surpass human capability, Peer Prediction provides a way to evaluate their honesty without needing to understand whether their answers are correct.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Peer Prediction from game theory into LLM evaluation is a genuinely novel cross-domain contribution; the discovery of inverse scaling is surprising.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Broad model coverage from 135M to 405B parameters, 85 domains, 37K+ questions, dual validation through training and evaluation, comprehensive scaling analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ A complete logical chain from practical motivation to theoretical guarantees to experimental validation; theorems and experiments are highly consistent.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretically rigorous and practically applicable solution to the scalable oversight problem, a core challenge in AI safety.

## Background & Motivation
1. **Limitations of Prior Work**: LLM-as-a-Judge is exploited when participant capability exceeds the judge's — stronger models can produce responses that appear correct but are actually wrong, deceiving weaker judges.
2. **Core Idea**: Peer Prediction rewards the predictability of information rather than the judge's subjective assessment — source models are rewarded for helping experts predict target answers.

## Method
- **Three-role system**: Source $s$, Target $t$, Expert $j$ evaluate the mutual predictability of each other's answers.
- **Scoring rule**: $\log \Pr_j(A_t|A_s) - \log \Pr_j(A_t)$ — the degree to which the source's answer helps the expert predict the target's answer.
- **Game-theoretic guarantees**: incentive compatibility under shared priors (Theorem 1); guaranteed under divergent priors via a sufficiently large and diverse pool of agents (Theorem 2).

## Key Experimental Results

| Scenario | Peer Prediction | LLM-as-Judge |
|----------|----------------|-------------|
| Malicious fine-tuning recovery (8B → 0.135B expert) | ~95% honesty restored | Weak (7B judge deceived) |
| 100× scale difference | Effective | Already fails at 5–20× gap |

## Key Findings
- **Inverse scaling**: weak experts are more resistant to deception by strong models — because Peer Prediction is grounded in information theory rather than subjective judgment.
- Ensembling multiple experts further improves robustness.

## Highlights & Insights
- Provides a new tool for the safe evaluation of superhuman AI — stronger judges are not needed; weaker evaluators are actually more suitable.

## Limitations & Future Work
- Only unilateral deception is analyzed; participant collusion is not addressed.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Peer Prediction into LLM evaluation is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Malicious fine-tuning experiments combined with scaling analysis.
- Value: ⭐⭐⭐⭐⭐ A critical direction for the safe evaluation of superhuman AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](../../ACL2026/llm_evaluation/personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ICML 2026\] Stop Automating Peer Review Without Rigorous Evaluation](../../ICML2026/llm_evaluation/stop_automating_peer_review_without_rigorous_evaluation.md)

</div>

<!-- RELATED:END -->
