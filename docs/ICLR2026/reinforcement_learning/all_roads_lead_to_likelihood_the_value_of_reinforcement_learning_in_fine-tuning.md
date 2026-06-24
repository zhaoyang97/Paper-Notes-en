---
title: >-
  [Paper Note] All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning
description: >-
  [ICLR2026][Reinforcement Learning][RLHF] This paper explains why the "two-stage reward model + online RL" approach in language model fine-tuning often outperforms direct offline maximum likelihood from the perspectives of information geometry, controlled experiments, and complexity intuition. The core conclusion is that the value of RL lies not in creating new information, but in using an easier-to-learn verifier to constrain policy search to a small class of generators induc…
tags:
  - "ICLR2026"
  - "Reinforcement Learning"
  - "RLHF"
  - "Preference Optimization"
  - "DPO"
  - "Reward Model"
  - "Generation-Verification Gap"
date: 2026-05-08
content_hash: 852011d7d2ab3a3a
---

# All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=sCL5mSTpKm](https://openreview.net/forum?id=sCL5mSTpKm)  
**Code**: To be released  
**Area**: Reinforcement Learning / Language Model Preference Tuning  
**Keywords**: RLHF, Preference Optimization, DPO, Reward Model, Generation-Verification Gap  

## TL;DR
This paper explains why the "two-stage reward model + online RL" approach in language model fine-tuning often outperforms direct offline maximum likelihood from the perspectives of information geometry, controlled experiments, and complexity intuition. The core conclusion is that the value of RL lies not in creating new information, but in using an easier-to-learn verifier to constrain policy search to a small class of generators induced by simple rewards.

## Background & Motivation
**Background**: Preference fine-tuning in large model post-training typically follows two routes: one consists of offline methods like DPO, IPO, and SLiC-HF, which optimize policy parameters directly on preference pairs; the other is traditional RLHF, which first trains a reward model (RM) on preference data and then maximizes this reward via online RL processes like PPO or online DPO. In engineering practice, many strong models still adopt the latter, more complex two-stage route.

**Limitations of Prior Work**: From a first-principles perspective, this is somewhat counter-intuitive. Preference data already indicates which completion is better. If the goal is simply to increase the likelihood of preferred completions, why not use direct maximum likelihood or classification-based preference optimization? Furthermore, reward models are merely functions of preference data, and on-policy sampling only generates text from the current policy; according to the data processing inequality, neither step should produce new ground-truth human preference information.

**Key Challenge**: Theoretical intuition suggests that "detouring through a reward model and then RL" should not have an informational advantage over direct offline learning, yet online PFT/RLHF often empirically performs better. The authors aim to explain a deeper question than just "is online data useful?": under identical preference data, initial SFT models, and even the same DPO loss, why does the reward model route still yield a better policy?

**Goal**: The paper breaks this problem into three steps. First, it uses information geometry to frame offline and online PFT as two types of projections under the same likelihood objective, showing they should be equivalent under ideal conditions. Second, it designs controlled experiments to eliminate common explanations, such as the inherent value of online samples, failure of offline regularization, easier online optimization, or the RM's ability to utilize more data or generalize OOD. Third, it proposes and tests the "generation-verification gap" hypothesis: in many tasks, judging an answer is significantly easier than generating one.

**Key Insight**: The authors view the policy as a generator and the reward model as a verifier. If the functional complexity of the verifier is significantly lower than that of the policy or Q-function required to generate high-quality answers, then learning the verifier first with limited preference data is more sample-efficient than learning the generator directly. RL does not provide new labels; instead, it converts a learned simple verifier into its soft-optimal policy.

**Core Idea**: The true value of two-stage online fine-tuning is learning a relatively simple reward model first, and then using RL to search for policies only within the "set of optimal policies for simple verifiers." This narrows the full policy space search problem faced by offline PFT into a more manageable data-computation tradeoff.

## Method

### Overall Architecture
The method of this paper is not to propose a new RLHF algorithm but to establish an explanatory framework and use experiments to refute or support different hypotheses. The overall process can be summarized as: unifying preference fine-tuning into a KL-regularized likelihood maximization problem; proving that when the reward class and policy class are isomorphic, the optimal solutions for online RLHF and offline DPO/MLE should coincide; conducting rigorous controlled experiments on TL;DR summarization to confirm the superiority of online DPO over offline DPO in practice; and finally explaining this difference via the generation-verification gap.

Formally, the authors treat a completion as a trajectory $\xi$, where the policy $\pi$ generates a trajectory distribution $P_\pi(\xi)$ in an autoregressive MDP. Preference data consists of paired trajectories $(\xi_i^+, \xi_i^-)$ for the same prompt. Offline PFT maximizes preference likelihood directly over the policy class $\Pi$, while online PFT fits a Bradley-Terry reward model over the reward class $R$ followed by solving a soft RL problem with entropy or KL regularization. Though appearing different, both routes essentially attempt to explain preference data—one by projecting directly into policy space, the other by projecting into reward space and then back-projecting to policy space.

The empirical section focuses on Pythia models and TL;DR summarization. The authors strictly fix confounding factors: both offline and online versions use the DPO loss; the reward model and policy start from the same SFT checkpoint and use the same preference data; online DPO simply samples 25 completions from the current policy, which are then ranked by the same global RM to select top/bottom pairs as new preference data. If online still performs better, the difference cannot be easily attributed to loss functions, training steps, or additional human information.

### Key Designs
**1. Information Geometric Equivalence: Proving "Gap Shouldn't Exist Theoretically"**
The paper first places online/offline PFT into a unified objective. The core of preference learning is making preferences under the data distribution more likely while regularizing with entropy or KL to a reference policy. For reward models, the Bradley-Terry form is used: $P_r^{BT}(\xi_1 \succ \xi_2 \mid s_0)=\sigma(r(\xi_1)-r(\xi_2))$. For policies, trajectory-level reward is expressed as the sum of token log-probabilities: $r_\pi(\xi)=\sum_h \log \pi(a_h\mid s_h)$. This step shows that offline methods like DPO essentially learn a local reward model implicitly defined by policy logits.

The authors then prove that soft RL can be viewed as a reverse KL projection from the reward-induced optimal trajectory distribution $P_r^*(\xi\mid s_0)\propto \exp(r(\xi))$ to the policy-induced distribution $P_\pi$. Thus, online RLHF becomes a two-step projection: first using forward KL/MLE to project data into the reward class, then using reverse KL to project the soft-optimal distribution into the policy class. If $R=R(\Pi)$, meaning the reward class and the local reward class induced by policies represent the same function family, and optimization is ideal, then two-stage RLHF and direct MLE/DPO have the same optimal solution. This conclusion is crucial as it eliminates the "RLHF is obviously stronger" intuition, turning the empirical gap into an anomaly that must be explained.

**2. Controlled Refutation: Isolating the Advantages of Online DPO**
To avoid vague explanations like "PPO is just better than DPO," the paper keeps offline and online DPO almost identical except for the training data. Online DPO samples 25 completions from the current policy, ranks them with the same global RM, selects top/bottom samples to form preference pairs, and trains with the same DPO objective. Offline DPO uses the original preference data. All policies and reward models start from the same SFT checkpoint with consistent hyperparameters.

Under this setting, online DPO still consistently outperforms offline DPO. The authors then examine five alternative explanations. If online samples provided new information, it would violate the data processing inequality since labels come from the same RM trained on the same data; if the issue were poor offline regularization to $\pi_{ref}$, then experiments with identical regularization should see no gap; if online optimization were simply easier, prompt augmentation should show significant gains, which it does not; if the global RM's advantage came from more/broader data, then the gap should narrow when using narrow-distribution data generated by SFT policies and labeled by GPT-4o, but online DPO still shows improvements; if it were simply better OOD generalization, the question returns to why the global RM has better in-distribution validation likelihood.

**3. Generation-Verification Gap: RLHF as Constrained Proper Learning**
The paper ultimately supports Hypothesis 6: in many post-training tasks, judging whether an answer is good is easier than generating one. The reward model acts as a verifier and the policy as a generator. If the underlying reward function can be represented by a shallower or simpler function while the soft-optimal policy/Q-function needs to encode complex multi-step generation structures, then fitting the policy directly is equivalent to searching for a generator in a larger, complex function space. Conversely, learning an RM facilitates finding a simple verifier first, then considering only the set of optimal policies for those verifiers $\Pi(R_{sim})$.

The authors formalize this into a theorem: If $R_{sim}\subset R$ is a set of simple reward models and $\Pi(R_{sim})$ is the set of soft-optimal policies induced by them, then (assuming no loss in the RL reverse KL projection) RLHF recovers the maximum likelihood solution over $\Pi(R_{sim})$. That is, online PFT does not escape likelihood; it uses the reward model route to transform "searching the entire policy space $\Pi$" into "searching a subset of optimal policies for simple verifiers." This is the meaning of the title "All Roads Lead to Likelihood": the goal remains likelihood, but RL provides a shortcut through a simple verifier.

**4. Complexity Intuition: Local Rewards are Q-functions, not Rewards**
A common point of confusion: if reward and policy are isomorphic in soft RL, why is learning the reward more sample-efficient? The authors argue that isomorphism does not imply both endpoints are equally easy to represent. The local RM implicitly learned by DPO is closer to a soft Q-function because token logits must express "how good the result will be if I continue generating from here" at every prefix. In contrast, a global RM only needs to score complete trajectories. A maze analogy is intuitive: a reward only needs to mark the goal, whereas a Q-function must encode the value of paths to the goal at every tile. The longer the horizon, the broader the state coverage and complexity of the Q-function.

This explains online PFT as a "data-for-computation" strategy. Offline DPO pays a statistical cost by learning complex policy/Q-like objects directly from limited data; online PFT pays a computational cost by first learning a simpler reward and then calculating its induced policy via RL. This gap is more pronounced in tasks requiring long-range planning or multi-step reasoning.

### Loss & Training
The core objective is a KL-regularized preference likelihood problem. Simply put, the policy minimizes the forward KL between the data preference distribution and the policy-induced preference distribution, while using entropy or reverse KL to the reference policy to avoid catastrophic forgetting. Reward model training follows standard logistic regression:

$$ \hat r_{mle}=\arg\max_{r\in R}\sum_i \log \sigma(r(\xi_i^+)-r(\xi_i^-)). $$

Offline DPO/local RM replaces $r(\xi)$ with trajectory sums of policy log-probabilities:

$$ r_\pi(\xi)=\sum_h \log \pi(a_h\mid s_h), $$

thereby optimizing the likelihood difference of preferred vs. dispreferred completions. In the online stage, given a trained global RM, soft RL is solved:

$$ \pi_r^*=\arg\max_{\pi\in\Pi}\mathbb{E}_{\xi\sim\pi}[r(\xi)]+H(\pi), $$

or its KL-regularized version. In experiments, the global RM uses logistic loss, batch size 64, learning rate $3\times10^{-6}$, AdamW, and cosine decay. DPO uses batch size 128, learning rate $3\times10^{-7}$, $\beta=0.05$, and linear decay. Online DPO samples 25 completions per prompt, ranks them by RM, and takes top/bottom pairs. It performs one batch of online generation rather than continuous sampling to ensure a clean comparison.

## Key Experimental Results

### Main Results
The main experiment compares SFT, offline DPO, and online DPO on TL;DR summarization using GPT-4o winrate against human references. Even with identical data, SFT starting points, and DPO loss, online DPO significantly outperforms offline DPO. Furthermore, applying online DPO on top of a DPO-tuned model continues to yield gains without new human feedback.

| Model Size | Method | GPT-4o Winrate (↑) | Note |
|:---:|:---:|:---:|:---|
| Pythia-1.4B | SFT | 26.2 | Supervised fine-tuning only |
| Pythia-1.4B | DPO | 49.7 | Offline DPO on original data |
| Pythia-1.4B | DPO (2x) | 52.2 | Extra epoch of offline DPO |
| Pythia-1.4B | Online DPO (SFT) | 56.1 | Sample from SFT, rank by RM |
| Pythia-1.4B | Online DPO (DPO) | 59.3 | Sample from DPO policy, rank by RM |
| Pythia-2.8B | SFT | 30.5 | Larger SFT baseline |
| Pythia-2.8B | DPO | 54.9 | Offline DPO |
| Pythia-2.8B | Online DPO (SFT) | 60.8 | Online remains significantly stronger |

Another key result involves "closing the generation-verification gap." If H6 is correct, the online PFT advantage should diminish when generation is not significantly harder than verification. Two settings verify this: Two-Word summary (short horizon) and ROUGE-L reward (complex verifier).

| Setting | Method | Metric | Value | Conclusion |
|:---|:---:|:---:|:---:|:---|
| Two-Word TL;DR | SFT | Winrate | 6.3 | Weak baseline for short tasks |
| Two-Word TL;DR | DPO | Winrate | 21.9 | Offline DPO improves significantly |
| Two-Word TL;DR | Online DPO (DPO) | Winrate | 23.2 | Online gain is only ~1.3 points |
| ROUGE-L TL;DR | DPO | Val. ROUGE-L | 0.354 | Offline DPO baseline |
| ROUGE-L TL;DR | Online DPO (DPO) | Val. ROUGE-L | 0.352 | No online improvement |
| ROUGE-L TL;DR | DPO (2x) | Val. ROUGE-L | 0.358 | Extra offline DPO slightly better |

### Ablation Study
The ablation focuses on falsifying alternative hypotheses rather than traditional module removal:

| Hypothesis / Setting | Key Results | Impact on Explanation |
|:---|:---|:---|
| Prompt augmentation | 1.4B Online DPO 56.1 → 56.6; 2.8B 60.8 → 61.2 | Minimal change; refutes "online samples provide more constraints" |
| GPT-labeled narrow data | 1.4B Online DPO (DPO) reaches 65.2 | Significant gain persists; refutes "global RM just uses broader data" |
| RM val likelihood | Global RM (0.610) > Local (0.598) > DPO (0.545) | Global RM fits ID data better, supporting "local/Q-like is harder to learn" |
| BoN OOD Ranking | Consistency between high $N$ BoN and val likelihood | OOD success stems from better ID margins, not just generalization |
| Scaling RM | Scaling RM size has less impact than scaling policy size | Verifiers can be approximated by smaller models; generators benefit more from scaling |

### Key Findings
- Under ideal isomorphic conditions, online RLHF, offline MLE, and DPO converge to the same likelihood optimum; empirical gaps must stem from statistical or representational complexity differences.
- Online DPO consistently outperforms offline DPO under strict controls for loss, initialization, and data, refuting various "engineering detail" explanations.
- Global RMs are easier to fit than local/DPO RMs and perform better in Best-of-N (BoN) ranking, aligning with the "trajectory-level verification vs. token-level credit assignment" explanation.
- The online DPO advantage vanishes when the horizon is shortened or the verifier complexity matches the generator's, a key predictive success of the H6 hypothesis.
- RLHF is framed as a "data-for-computation" exchange: learning simple rewards from limited data and using compute-intensive RL to derive policies.

## Highlights & Insights
- The paper avoids the lazy "online interaction is just better" trope, instead using an equivalence theorem to establish a baseline: if rewards and policies were truly isomorphic and optimization ideal, the two-stage route shouldn't win.
- The "local rewards as Q-functions" observation is highly insightful. While DPO frames the policy as a secret reward, this paper highlights that token logits must perform credit assignment across all prefixes, making them harder to learn than a terminal verifier.
- The generation-verification gap connects RLHF, inverse RL, and reasoning verifiers. If verification is easier than generation, learning a verifier first is more data-efficient than direct imitation.
- The "data-calculation tradeoff" suggests that in resource-constrained post-training, one should perhaps prioritize high-quality verifiers and efficient RL/search over simply feeding preference pairs to the policy.

## Limitations & Future Work
- Experiments focus on TL;DR summarization and Pythia models. While it references reasoning and agentic tasks, large-scale empirical validation on instruction-following or complex reasoning is still needed.
- Evaluation relies heavily on GPT-4o winrate. While good for relative comparison, it is still a model-based judge and may share biases with the reward models being studied.
- Measuring "reward function simplicity" in deep networks remains difficult; the paper uses indirect evidence like horizon shortening and likelihood. Direct complexity metrics are needed.
- The paper does not provide a granular breakdown of the optimal configuration between "stronger offline training / more data / larger verifier / more RL rollouts" for a fixed compute budget.

## Related Work & Insights
- **vs DPO**: This work doesn't negate DPO but points out that the local RM DPO learns is more complex (Q-like) and thus harder to fit than a global RM from limited data.
- **vs Traditional RLHF / PPO**: While RLHF is often found to be stronger empirically, this paper abstracts the process into "forward KL for RM and reverse KL/soft RL for policy" to explain its statistical advantage.
- **vs Inverse RL / Imitation Learning**: Migrates the classic Ng et al. view (rewards are more compact than policies) to LLM tuning, using the generation-verification gap to explain its continued relevance.
- **vs RM OOD Generalization**: Accepts that global RMs generalize better but seeks the root cause—concluding it lies in the inherent simplicity of the verification task itself.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframes RLHF advantages through proper learning and complexity gaps rather than just proposing new algorithms.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid control variables and consistent hypothesis testing, though task range is somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, rigorous logical chain from equivalence to anomaly to hypothesis.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for understanding why RL is still needed despite the success of offline methods like DPO.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Proximal Supervised Fine-Tuning](proximal_supervised_fine-tuning.md)
- [\[ICLR 2026\] Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning](fine-tuning_behavioral_cloning_policies_with_preferencebased_reinforcement_learn.md)
- [\[ICLR 2026\] SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning](srft_a_single-stage_method_with_supervised_and_reinforcement_fine-tuning_for_rea.md)
- [\[ICLR 2026\] Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning](escaping_policy_contraction_contraction-aware_ppo_cappo_for_stable_language_mode.md)
- [\[ICLR 2026\] On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting](on-policy_rl_meets_off-policy_experts_harmonizing_supervised_fine-tuning_and_rei.md)

</div>

<!-- RELATED:END -->
