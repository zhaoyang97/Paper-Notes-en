---
title: >-
  [Paper Note] Why is Your Language Model a Poor Implicit Reward Model?
description: >-
  [ICLR 2026][LLM Reasoning][Implicit reward model] This paper provides theoretical and empirical evidence that implicit reward models (IM-RM, e.g., DPO) generalize worse than explicit reward models (EX-RM) because IM-RM overfits to surface-level token cues rather than semantic representations, leading to substantial accuracy degradation under token distribution shift. The paper also refutes the "generation–verification gap" hypothesis.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Implicit reward model
  - explicit reward model
  - generalization gap
  - token-level cues
  - DPO vs RLHF
date: 2026-05-08
content_hash: af6b007118385cd2
---

# Why is Your Language Model a Poor Implicit Reward Model?

**Conference**: ICLR 2026
**arXiv**: [2507.07981](https://arxiv.org/abs/2507.07981)
**Code**: None
**Area**: LLM Reasoning / Alignment RLHF
**Keywords**: Implicit reward model, explicit reward model, generalization gap, token-level cues, DPO vs RLHF

## TL;DR
This paper provides theoretical and empirical evidence that implicit reward models (IM-RM, e.g., DPO) generalize worse than explicit reward models (EX-RM) because IM-RM overfits to surface-level token cues rather than semantic representations, leading to substantial accuracy degradation under token distribution shift. The paper also refutes the "generation–verification gap" hypothesis.

## Background & Motivation

**Background**: Reward models are central components in LLM post-training and inference pipelines. Two nearly identical reward model paradigms exist: explicit reward models (EX-RM, which attach a linear head on top of hidden representations) and implicit reward models (IM-RM, which define rewards implicitly via $\ln \pi_\theta(\mathbf{y}|\mathbf{x})$, the core idea behind DPO). Both can be trained with the same data, loss function, and base language model; the only difference lies in how the reward is computed.

**Limitations of Prior Work**: Despite their near-identical formulation, prior work has repeatedly observed that IM-RM generalizes significantly worse than EX-RM, particularly in out-of-distribution response ranking. This generalization gap is puzzling—why would such a minor difference in reward computation lead to such a large performance disparity?

**Key Challenge**: An intuitive explanation is the "generation–verification gap"—since IM-RM must both assign high scores to correct responses and generate them via the underlying language model, if generation is harder than verification, IM-RM accuracy should lag behind. However, whether this argument holds and what the true underlying cause is remain open questions.

**Goal**
- Refute the "generation–verification gap" hypothesis by showing that IM-RM verification does not require learning to generate.
- Identify the true cause by characterizing the behavioral differences between EX-RM and IM-RM from a learning dynamics perspective.
- Validate theoretical predictions through both controlled and realistic experimental settings.

**Key Insight**: The analysis proceeds from how gradient updates affect the reward assigned to unseen samples. It is found that EX-RM reward changes depend only on inner products of hidden representations, whereas IM-RM reward changes additionally depend on specific tokens.

**Core Idea**: IM-RM generalizes poorly because its learning dynamics inherently bias it toward overfitting surface-level token cues rather than exploiting the semantic structure of hidden representations.

## Method

### Overall Architecture
Rather than proposing a new method, this paper conducts theoretical analysis and experimental validation of two reward model families. The research proceeds as follows:
1. Analyze and refute the "generation–verification gap" hypothesis (Section 3).
2. Characterize the differences between EX-RM and IM-RM from a learning dynamics perspective (Section 4).
3. Validate theoretical predictions through controlled and realistic experiments (Section 5).

### Key Designs

1. **Refuting the "Generation–Verification Gap" Hypothesis**

    - **Function**: Demonstrate that IM-RM can serve as a perfect verifier even when the underlying language model is entirely incapable of generating correct answers.
    - **Mechanism**: Theorem 1 constructs a distribution $\pi$ whose induced IM-RM verifies correctness with margin $\delta$, while the probability that $\pi$ generates correct responses increases by at most a constant factor $\exp(\delta/\beta)$ relative to the reference distribution $\pi_{\text{ref}}$. That is, if $\pi_{\text{ref}}$ cannot generate efficiently, $\pi$ need not do so either in order to serve as a good verifier.
    - **Experimental Validation**: On the NP-hard Hamiltonian cycle verification task, IM-RM (based on Pythia-1B) achieves 0.993 accuracy on the test set while failing to generate a single valid Hamiltonian cycle.

2. **Learning Dynamics Analysis of EX-RM**

    - **Function**: Characterize how the reward assigned to an unseen sample $(\bar{\mathbf{x}}, \bar{\mathbf{y}})$ changes after a gradient update.
    - **Mechanism**: Under the fixed hidden representation assumption (Assumption 1), the reward change for EX-RM is $\Delta r_{\theta_{\text{EX}}}(\bar{\mathbf{x}}, \bar{\mathbf{y}}) = \langle \mathbf{h}_{\bar{\mathbf{x}},\bar{\mathbf{y}}}, \mathbf{h}_{\mathbf{x},\mathbf{y}^+} - \mathbf{h}_{\mathbf{x},\mathbf{y}^-} \rangle \cdot \eta g(\theta_{\text{EX}})$. The reward change depends entirely on the similarity between hidden representations—if $\bar{\mathbf{y}}$ and $\mathbf{y}^+$ are semantically close (i.e., their representations are aligned), the reward increases regardless of the specific tokens used.
    - **Design Motivation**: Since pre-trained representations encode semantics, EX-RM naturally generalizes to responses that convey the same meaning with different tokens.

3. **Learning Dynamics Analysis of IM-RM**

    - **Function**: Reveal why IM-RM over-relies on token-level cues.
    - **Mechanism**: The reward change for IM-RM involves coefficients $\rho_{k,l}(\mathbf{v})$ that are positive when $\bar{\mathbf{y}}_k = \mathbf{v}_l$ (token match), producing an effect analogous to EX-RM, but potentially negative when $\bar{\mathbf{y}}_k \neq \mathbf{v}_l$ (token mismatch). In the latter case, even semantically aligned hidden representations may **decrease** the reward. Critically, responses that are semantically similar but lexically different may receive reward updates in opposite directions under IM-RM.
    - **Design Motivation**: This explains why paraphrasing a response can cause IM-RM accuracy to plummet from 1.0 to 0.02.

4. **Theoretical Generalization Gap (Theorem 2)**

    - **Function**: Formally prove, under a simplified setting (single-token responses), that IM-RM cannot generalize to unseen tokens.
    - **Mechanism**: After convergence, IM-RM assigns a constant reward difference (equal to its initial value) to any token pair absent from the training set, resulting in a fixed accuracy of 0.5 (random chance). By contrast, the linear head of EX-RM converges toward the maximum-margin separating hyperplane $\mathbf{u}^*$, correctly ranking all samples that $\mathbf{u}^*$ can separate.
    - **Design Motivation**: Although the assumptions are strong (single token, fixed representations), experiments demonstrate that the conclusion holds under full-parameter training and arbitrary-length responses.

### Loss & Training
Both model families are trained using the Bradley–Terry log-likelihood loss:
$$\mathcal{L}(r) = \frac{1}{|\mathcal{D}_T|} \sum -\ln \sigma(r(\mathbf{x}, \mathbf{y}^+) - r(\mathbf{x}, \mathbf{y}^-))$$

## Key Experimental Results

### Main Results (Controlled Setting: Persona Dataset)

| Evaluation Condition | EX-RM Accuracy | IM-RM Accuracy |
|----------------------|----------------|----------------|
| Original responses (train set) | 1.00 | 1.00 |
| Original responses (test set) | 1.00 | 1.00 |
| Paraphrased responses (train set) | 1.00 | **0.022** |
| Paraphrased responses (test set) | 1.00 | **0.019** |

IM-RM accuracy collapses to near zero under token distribution shift (paraphrasing), while EX-RM generalizes perfectly.

### Realistic Setting (Trained on UltraFeedback, 6 models ranging from 1B to 8B)

| Evaluation Type | EX-RM Acc. | IM-RM Acc. | EX-RM Reward Margin | IM-RM Reward Margin |
|-----------------|-----------|-----------|---------------------|---------------------|
| In-distribution | **0.752** | 0.646 | **1.014** | 0.813 |
| Token-level shift | **0.665** | 0.602 | **0.976** | 0.763 |
| Domain shift | 0.621 | **0.720** | **0.807** | 0.726 |

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| Hamiltonian cycle verification | IM-RM test accuracy = 0.993; number of correct cycles generated = 0, confirming verification ≠ generation |
| EX-RM using intermediate token representations | Rules out the explanation that EX-RM benefits merely from using the full-sequence representation |
| IM-RM without reference distribution | Rules out "reference distribution shift" as an explanation |
| Token shift (translation / paraphrasing) | EX-RM substantially outperforms IM-RM after French/Spanish translation |

### Key Findings
- IM-RM consistently underperforms EX-RM under **token-level shift** (paraphrasing, translation) but may match or surpass EX-RM under **domain shift**.
- EX-RM consistently produces larger absolute reward margins, which is advantageous for downstream RL optimization.
- IM-RM also underperforms EX-RM in in-distribution evaluation, because in-distribution test samples are semantically similar but lexically different from training samples—more akin to token-level shift than true in-distribution generalization.

## Highlights & Insights
- **Novelty of the Learning Dynamics Perspective**: By analyzing the effect of a single gradient update, the paper precisely characterizes the fundamental difference between EX-RM (representation-only) and IM-RM (representation plus tokens). This framing is both elegant and highly explanatory, far surpassing the intuitive "generation–verification gap" argument.
- **Counterintuitive Refutation**: Theorem 1 combined with the Hamiltonian cycle experiment cleanly refutes the widely held "generation–verification gap" hypothesis, which is a compelling contribution.
- **Theoretical Explanation for the "Paraphrase Degradation" Phenomenon**: The sign of the $\rho_{k,l}$ coefficient depends on token identity, offering actionable guidance for DPO practitioners—for instance, augmenting DPO training data with paraphrased positive–negative pairs may effectively mitigate IM-RM fragility.
- **New Perspective on the RLHF vs. DPO Debate**: The paper provides a novel theoretical explanation for why DPO underperforms RLHF (token-level overfitting), which is complementary to the existing "generation–verification gap" interpretation.

## Limitations & Future Work
- The theoretical analysis relies on fixed hidden representations (Assumption 1) and single-token responses (Assumption 2). Although experiments confirm the broader validity of the conclusions, more general theoretical guarantees remain to be established.
- Only ranking accuracy is used as the evaluation metric; the downstream impact of reward models in actual RL training is not investigated.
- The observation that IM-RM may outperform EX-RM under **domain shift** is noted but not deeply analyzed—under what conditions should IM-RM be preferred?
- **Directions for Improvement**: Can a "token-invariant" IM-RM training scheme (e.g., data augmentation with paraphrases during DPO training) be designed to close the generalization gap?

## Related Work & Insights
- **vs. DPO (Rafailov et al., 2023)**: DPO is essentially IM-RM. This paper identifies a key reason why it generalizes worse than RLHF (train EX-RM first, then optimize with RL).
- **vs. Swamy et al. (2025)**: That work attributes DPO's inferiority to RLHF to the "generation–verification gap." The present paper directly refutes this hypothesis, at least at the level of reward model accuracy.
- **vs. Im & Li (2025)**: That work proves IM-RM generalizes to **different prompts with the same response**, whereas the present paper shows IM-RM fails to generalize to **different responses**, which is more relevant to realistic settings.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The learning dynamics perspective is novel, and the discovery of token dependence is highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Controlled and realistic settings combined with multiple ablations rule out several alternative hypotheses, though downstream RL validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain is exceptionally clear: refute the old hypothesis → propose a new explanation → theoretical proof → experimental validation.
- **Value**: ⭐⭐⭐⭐⭐ — The findings carry significant implications for the RLHF/DPO community by exposing an implicit bias in reward model design.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)
- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](../../NeurIPS2025/llm_reasoning/one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](../../ACL2026/llm_reasoning/dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[AAAI 2026\] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement](../../AAAI2026/llm_reasoning/incorporating_self-rewriting_into_large_language_model_reasoning_reinforcement.md)

<!-- RELATED:END -->
