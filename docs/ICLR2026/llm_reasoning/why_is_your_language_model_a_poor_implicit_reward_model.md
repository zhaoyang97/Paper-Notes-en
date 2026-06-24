---
title: >-
  [Paper Note] Why is Your Language Model a Poor Implicit Reward Model?
description: >-
  [ICLR 2026][Reasoning][Implicit Reward Model] This paper reveals through theory and experiments the fundamental reason why Implicit Reward Models (IM-RM, such as DPO) generalize worse than Explicit Reward Models (EX-RM): IM-RM over-relies on surface token-level cues rather than semantic representations. This leads to a significant drop in accuracy under token distribution shifts. Furthermore, the paper refutes the popular "generation-verification gap" hypothesis.
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "Implicit Reward Model"
  - "Explicit Reward Model"
  - "Generalization Gap"
  - "Token-level Cues"
  - "DPO vs RLHF"
date: 2026-05-08
content_hash: 0bc20d061374570d
---

# Why is Your Language Model a Poor Implicit Reward Model?

**Conference**: ICLR 2026  
**arXiv**: [2507.07981](https://arxiv.org/abs/2507.07981)  
**Code**: None  
**Area**: LLM Reasoning / Alignment RLHF  
**Keywords**: Implicit Reward Model, Explicit Reward Model, Generalization Gap, Token-level Cues, DPO vs RLHF

## TL;DR
This paper reveals through theory and experiments the fundamental reason why Implicit Reward Models (IM-RM, such as DPO) generalize worse than Explicit Reward Models (EX-RM): IM-RM over-relies on surface token-level cues rather than semantic representations. This leads to a significant drop in accuracy under token distribution shifts. Furthermore, the paper refutes the popular "generation-verification gap" hypothesis.

## Background & Motivation

**Background**: Reward models are core components of LLM post-training and inference pipelines. There are currently two nearly identical types of reward models: Explicit Reward Models (EX-RM, adding a linear head on top of hidden representations) and Implicit Reward Models (IM-RM, defining rewards implicitly via $\ln \pi_\theta(\mathbf{y}|\mathbf{x})$, which is the core idea of DPO). Both can be trained using the same data, loss functions, and base language models; the only difference lies in how the reward is calculated.

**Limitations of Prior Work**: Despite EX-RM and IM-RM being nearly identical, prior work has repeatedly observed that IM-RM demonstrates significantly poorer generalization, particularly with lower accuracy in ranking responses during out-of-distribution evaluations. This generalization gap is perplexing—why would a minor difference in the reward calculation method lead to such a massive performance gap?

**Key Challenge**: Intuitively, one explanation is the "generation-verification gap"—that IM-RM must both assign high scores to correct answers and be able to generate correct answers via the underlying language model. If generation is harder than verification, IM-RM accuracy should lag. But does this intuitive argument hold? What is the actual cause?

**Goal**
   - Refute the "generation-verification gap" hypothesis: Prove that IM-RM verification does not require learning to generate.
   - Identify the true cause: Characterize the behavioral differences between EX-RM and IM-RM from the perspective of learning dynamics.
   - Experimental validation: Verify theoretical predictions in both controlled and real-world scenarios.

**Key Insight**: By analyzing the impact of gradient updates on the rewards of unseen samples, the learning dynamics reveal a critical difference. EX-RM reward changes depend only on the inner product of hidden representations, whereas IM-RM changes additionally depend on specific tokens.

**Core Idea**: The poor generalization of IM-RM stems from its learning dynamics, which naturally tend to overfit surface token-level cues rather than utilizing the semantic-level structure of hidden representations.

## Method

### Overall Architecture
This paper does not propose a new method but answers a long-standing puzzle: EX-RM and IM-RM are practically the same—same data, same loss, same base LLM—with the only difference being the reward calculation (EX-RM uses a linear head on hidden representations, while IM-RM uses the $\ln\pi_\theta$ implicit definition). Why does IM-RM consistently generalize worse? The authors proceed in three steps: first, they use a counterexample to dismantle the popular "generation-verification gap" explanation; next, they analyze single-step gradient learning dynamics to prove that while EX-RM reward changes depend only on hidden representations, IM-RM is also driven by specific tokens; finally, they verify on controlled datasets and real 1B–8B models that this token dependency is the root of the generalization gap.

### Key Designs

**1. Refuting the "Generation-Verification Gap" Hypothesis: Verifying an answer does not require being able to generate it**

The prevailing intuition is that IM-RM struggles because it must balance high-scoring good answers with the underlying model's ability to generate them; since generation is harder than verification, accuracy suffers. Theorem 1 provides a counterexample to this: there exists a distribution $\pi$ whose induced IM-RM can verify correctness with a margin $\delta$, yet the probability of $\pi$ generating a correct response increases by at most a constant factor $\exp(\delta/\beta)$ compared to the reference distribution $\pi_{\text{ref}}$. In other words, if $\pi_{\text{ref}}$ cannot generate efficiently, $\pi$ won't either, yet it can still be a good verifier. Verification and generation capabilities are decoupled. Experiments confirm this: on the NP-hard Hamiltonian Cycle verification task, a Pythia-1B-based IM-RM achieved 0.993 test accuracy while generating zero correct cycles.

**2. EX-RM Learning Dynamics: Reward changes are determined solely by hidden representation similarity**

To find the true cause, the authors examine the impact of a single gradient update on the reward of an unseen sample $(\bar{\mathbf{x}}, \bar{\mathbf{y}})$. Under the assumption of fixed hidden representations (Assumption 1), the EX-RM reward change is:

$$\Delta r_{\theta_{\text{EX}}}(\bar{\mathbf{x}}, \bar{\mathbf{y}}) = \langle \mathbf{h}_{\bar{\mathbf{x}},\bar{\mathbf{y}}}, \mathbf{h}_{\mathbf{x},\mathbf{y}^+} - \mathbf{h}_{\mathbf{x},\mathbf{y}^-} \rangle \cdot \eta g(\theta_{\text{EX}})$$

Here, $\eta$ is the learning rate and $g(\theta_{\text{EX}})>0$ is a positive scalar (representing $\sigma$ acting on the current reward margin). Whether the reward increases or decreases depends entirely on the inner product of hidden representations. If the unseen sample $\bar{\mathbf{y}}$ is semantically similar to the positive sample $\mathbf{y}^+$ (aligned representations), the reward increases regardless of the specific tokens used. Since pre-trained representations already encode semantics, EX-RM naturally generalizes to responses that use different tokens but convey the same meaning.

**3. IM-RM Learning Dynamics: Rewards can be pushed in the wrong direction when tokens mismatch**

When looking at the reward change for IM-RM, the expression multiplies each position pair by a coefficient $\rho_{k,l}(\mathbf{v})\in[-2,2]$, determined by the tokens at positions $k, l$ and their next-token distributions. When $\bar{\mathbf{y}}_k = \mathbf{v}_l$ (tokens match), the coefficient is positive, acting similarly to EX-RM. However, when $\bar{\mathbf{y}}_k \neq \mathbf{v}_l$ (tokens mismatch), the coefficient can become negative. This means even if two responses are semantically aligned in hidden space, the reward might be pushed in the **opposite direction**. This explains the counter-intuitive scenario where paraphrasing a response can cause IM-RM accuracy to drop from 1.0 to 0.02.

**4. Theoretical Generalization Gap (Theorem 2): IM-RM guesses on unseen tokens**

In a simplified single-token response setting, the authors rigorously prove that after training convergence, the IM-RM reward difference for any "token pair not seen in the training set" remains equal to the initial constant, resulting in a ranking accuracy of 0.5 (random chance). In contrast, the EX-RM linear head converges to the maximum-margin separating hyperplane $\mathbf{u}^*$, correctly ranking any samples that $\mathbf{u}^*$ can separate. While the assumptions are strong (single token, fixed representations), subsequent experiments show that the conclusion holds for full-parameter training with arbitrary response lengths.

### Loss & Training
Both model types are trained using the Bradley-Terry log-likelihood loss:
$\mathcal{L}(r) = \frac{1}{|\mathcal{D}_T|} \sum -\ln \sigma(r(\mathbf{x}, \mathbf{y}^+) - r(\mathbf{x}, \mathbf{y}^-))$

## Key Experimental Results

### Main Results (Controlled Environment: Persona Dataset)

| Evaluation Condition | EX-RM Accuracy | IM-RM Accuracy |
|----------|------------|------------|
| Original Responses (Train) | 1.00 | 1.00 |
| Original Responses (Test) | 1.00 | 1.00 |
| Paraphrased Responses (Train) | 1.00 | **0.022** |
| Paraphrased Responses (Test) | 1.00 | **0.019** |

IM-RM accuracy drops to nearly zero when the token distribution changes (paraphrasing), while EX-RM generalizes perfectly.

### real-world Scenarios (UltraFeedback training, 6 models from 1B to 8B)

| Evaluation Type | EX-RM Accuracy | IM-RM Accuracy | EX-RM Reward Margin | IM-RM Reward Margin |
|---------|------------|------------|-------------|-------------|
| In-distribution | **0.752** | 0.646 | **1.014** | 0.813 |
| Token-level Shift | **0.665** | 0.602 | **0.976** | 0.763 |
| Domain Shift | 0.621 | **0.720** | **0.807** | 0.726 |

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Hamiltonian Cycle Verification | IM-RM test accuracy 0.993, but generated zero correct cycles, proving verification $\neq$ generation. |
| EX-RM with intermediate tokens | Excluded the explanation that "EX-RM uses full sequence while IM-RM uses intermediate representations." |
| IM-RM without reference distribution | Excluded the explanation of "reference distribution shift." |
| Token Shift (Translation/Paraphrase) | EX-RM significantly outperformed IM-RM even after French/Spanish translation. |

### Key Findings
- IM-RM is consistently weaker than EX-RM under **token-level shifts** (paraphrasing, translation), but may perform equally or better under **domain shifts**.
- EX-RM consistently produces larger absolute reward margins, which is beneficial for subsequent RL optimization.
- Even in in-distribution evaluations, IM-RM is weaker because test samples are semantically similar to but token-wise different from training samples, which is closer to a token shift.

## Highlights & Insights
- **Innovation in Learning Dynamics**: By analyzing the impact of single-step gradient updates, the paper precisely identifies the essential difference between EX-RM (representation-focused) and IM-RM (token-dependent). This perspective is elegant and explanatory, far exceeding the intuitive "generation-verification gap" argument.
- **Counter-intuitive Refutation**: Theorem 1 combined with the Hamiltonian Cycle experiments cleanly refutes the popular "generation-verification gap" hypothesis.
- **Theoretical Explanation for Paraphrase Failure**: The discovery that the sign of the $\rho_{k,l}$ coefficient depends on token matching can guide DPO practices—for example, including paraphrased pairs in DPO training sets might mitigate IM-RM fragility.
- **New Perspective on RLHF vs DPO**: Provides a new theoretical explanation (token-level overfitting) for why DPO may be weaker than RLHF, complementing existing "generation-verification gap" explanations.

## Limitations & Future Work
- Theoretical analysis assumes fixed hidden representations (Assumption 1) and single-token responses (Assumption 2). Although experiments verify the conclusions, more general theoretical guarantees are still missing.
- The study focuses primarily on accuracy as a metric and does not explore the downstream impact of reward models during actual RL training.
- While IM-RM was found to potentially outperform EX-RM in **domain shifts**, the reasons were not analyzed in depth—when should one choose IM-RM?
- **Future Directions**: Can a "token-invariant" IM-RM training method (such as data augmentation via paraphrasing during DPO) be designed to bridge the generalization gap?

## Related Work & Insights
- **vs DPO (Rafailov et al., 2023)**: Since DPO is essentially an IM-RM, this paper reveals a key reason why its generalization is weaker than RLHF (which trains an EX-RM before RL optimization).
- **vs Swamy et al. (2025)**: While they argue the "generation-verification gap" is why DPO is weaker than RLHF, this paper directly refutes that hypothesis (at least regarding RM accuracy).
- **vs Im & Li (2025)**: They showed IM-RM generalizes across **different prompts for the same response**, while this paper proves it fails across **different responses**, which is a more realistic scenario.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The learning dynamics analysis is fresh, and the discovery of token dependency is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Controlled + real scenarios + diverse ablations; excluded multiple alternative hypotheses, though downstream RL task validation is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is extremely clear: refute old hypothesis $\rightarrow$ propose new explanation $\rightarrow$ theoretical proof $\rightarrow$ experimental validation.
- Value: ⭐⭐⭐⭐⭐ Significant guidance for the RLHF/DPO community, revealing implicit biases in reward model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning with Sampling: Your Base Model is Smarter Than You Think](reasoning_with_sampling_your_base_model_is_smarter_than_you_think.md)
- [\[ICLR 2026\] OR-PRM: A Process Reward Model for Algorithmic Problem in Operations Research](or-prm_a_process_reward_model_for_algorithmic_problem_in_operations_research.md)
- [\[ICLR 2026\] R-HORIZON: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](r-horizon_how_far_can_your_large_reasoning_model_really_go_in_breadth_and_depth.md)
- [\[ICML 2026\] The Quality-Utility Paradox: Why High-Reward Data Impairs Small Model Mathematical Reasoning](../../ICML2026/llm_reasoning/the_quality-utility_paradox_why_high-reward_data_impairs_small_model_mathematica.md)
- [\[ICLR 2026\] Enhancing Language Model Reasoning with Structured Multi-Level Modeling](enhancing_language_model_reasoning_with_structured_multi-level_modeling.md)

</div>

<!-- RELATED:END -->
