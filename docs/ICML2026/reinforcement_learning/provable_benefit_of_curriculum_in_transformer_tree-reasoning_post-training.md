---
title: >-
  [Paper Note] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training
description: >-
  [ICML 2026][Reinforcement Learning][Paper Note] This paper provides the first rigorous sample complexity proof for "easy-to-hard" curriculum RL post-training: on the state-conditioned autoregressive reasoning trees of transformers, if the curriculum maintains the difficulty ratio of adjacent stages at the level of the $L/p$-th root of the target difficulty, the tota
tags:
  - ICML 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: d0727dfbb611386b
---
# Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training

**Conference**: ICML 2026  
**arXiv**: [2511.07372](https://arxiv.org/abs/2511.07372)  
**Code**: https://github.com/DakeBU/Curriculum-Post-training (Available)  
**Area**: LLM Reasoning / Reinforcement Learning / Learning Theory  
**Keywords**: Curriculum Learning, CoT Post-training, Sample Complexity, Coverage Coefficient, Rejection Sampling

## TL;DR
This paper provides the first rigorous sample complexity proof for "easy-to-hard" curriculum RL post-training: on the state-conditioned autoregressive reasoning trees of transformers, if the curriculum maintains the difficulty ratio of adjacent stages at the level of the $L/p$-th root of the target difficulty, the total sample complexity can be reduced from exponential $(C^\star)^L$ in direct training to polynomial $L\cdot (C^\star)^{p_\max}$ in the curriculum version.

## Background & Motivation
**Background**: Post-training for CoT (Chain-of-Thought) reasoning currently relies on "direct RL fine-tuning + 0/1 outcome verifier," combined with test-time scaling (beam search / best-of-N) to improve pass@K. Numerous empirical studies in the past year (Parashar 2025, Liu 2025, Bae 2025, etc.) found that "easy-to-hard curricula" significantly accelerate convergence, but these lack provable explanations.

**Limitations of Prior Work**: Classical curriculum learning theories are almost entirely established for "training from scratch" scenarios (convex regression, parity, teacher-student perceptron, etc.), where definitions of "difficulty" and "performance" are task-specific and geometric. These cannot be transferred to LLM post-training, which starts from a strong pretrained model and aims for CoT generalization. The core characteristic of post-training is sparse rewards, where the probability of sampling correct trajectories under the base policy is extremely low—a concept with no counterpart in old theories.

**Key Challenge**: The difficulty of sparse-reward RL essentially stems from the "rarity of correct CoT under the base policy," characterized by the coverage coefficient $\|\pi^\star/\pi_{\text{ref}}\|_\infty$. Direct training pays a sampling cost polynomial (or even exponential) relative to this rarity. While curricula allow breaking this "rarity ladder" into several small steps, there were previously no clear conditions defining what kind of steps actually bring super-polynomial acceleration.

**Goal**: (i) Provide necessary/sufficient conditions for curriculum post-training to yield exponential sample savings; (ii) prove these conditions naturally hold on the specific analyzable model of transformers + reasoning trees; (iii) apply the conclusions to both RL fine-tuning and test-time scaling paradigms.

**Key Insight**: The authors view post-training as redistributing probabilities on a "pre-trained reasoning tree" and adopt the spanner-sampling/coverage framework (Foster 2025). This links curriculum difficulty directly to the number of rejection sampling attempts $\Theta(\|\pi^\star/\pi_{\text{ref}}\|_\infty \log\delta^{-1})$, thereby unifying "difficulty" and "learning cost" on the same scale.

**Core Idea**: The base model is modeled as a PART (Pretrained Autoregressive Reasoning Transformer) that performs approximately uniform sampling over valid child nodes in a 2S-ART (State-conditioned Autoregressive Reasoning Tree). It is proven that "prefix-hint" and "depth-increasing" curricula naturally form a $K$-th root difficulty ladder, reducing the total cost from $(C^\star)^L$ to $L\cdot(C^\star)^{p_\max}$.

## Method

### Overall Architecture
The paper presents a theoretical framework rather than a new algorithm. It casts existing curriculum post-training pipelines into a provable abstraction: (1) Tasks are modeled as 2S-ART reasoning trees $F_{\text{2S-ART}}(\{\Phi_\ell\},\{I_\ell\})$, where each step selects a token from a valid index set $I_\ell$ and updates the state $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$; (2) the base model is instantiated as a PART, sampling valid child nodes uniformly at each depth; (3) a standard sampling-attention transformer (FFN implementing $\Phi_\ell$ primitives, attention for index selection) is proven to reproduce PART exactly; (4) it analyzes the sample complexity of RL fine-tuning (Thm 2) and the oracle-query/computational complexity of test-time scaling (Thm 3).

### Key Designs

**1. Necessary/Sufficient Conditions for Curriculum Benefit based on Coverage Coefficient (Cor. 1): Converting "Is Easy-to-Hard Useful?" into an Inequality**

While empirical evidence suggests "easy-to-hard" is effective, none defined how the "difficulty ladder" should be constructed. This paper unifies "difficulty" by defining the $\varepsilon$-accurate sample complexity $N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$ and tying it to the coverage coefficient via the rejection-sampling lemma $N\propto\|\pi^\star/\pi_{\text{ref}}\|_\infty$. Under this scale, the sufficient and necessary condition for a curriculum to be more efficient than direct training is $\sum_{\ell}N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})<N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$. Furthermore, when an $L/p$-th root curriculum exists—meaning the difficulty ratio between adjacent stages is $N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})=\Theta(\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})})$—the ratio of direct to curriculum samples $N^{\text{direct}}/N^{\text{curr}}=\Theta((C^\star)^L/(L\cdot C^\star))$ (where $C^\star=\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})}>1$) represents exponential acceleration. This condition explains commonalities across hint-decreasing and depth-increasing curricula and provides an actionable design principle: the difficulty ratio between adjacent stages should be controlled at the $C^\star$ scale.

**2. 2S-ART Reasoning Tree + PART Base Model (Def. 1-2): A Computable Abstraction for "Reasoning Tasks + Weak Base Models"**

To ground Cor. 1 in transformers, the paper represents reasoning tasks of length $k$ as index paths $S_k=(i_1,\dots,i_k,d{+}1)$. Valid indices $I_\ell(\text{CoT}_{\ell-1})$ ($|I_\ell|=\Theta(d)$) are selected at each step, and states are updated via $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$. The base model is a PART, sampling valid child nodes uniformly. This uniformity is critical: it yields $\|\pi^\star_{S_{\ell+1}}/\pi^\star_{S_\ell}\|_\infty=\Theta(d)$ and $\|\pi^\star_{S_\ell}/\pi^\text{PART}\|_\infty=\Theta(d^{\ell+1})$, satisfying the $L/p$-th root relationship naturally. Uniform-PART is chosen because it encompasses various tasks (parity, Countdown, Markov-chain reasoning, induction-head), ensures controllable child-node probability ratios, and aligns with the perspective of "post-training as reweighting a pre-trained tree."

**3. Exponential-to-Polynomial Reduction in RL Fine-tuning and Test-time Scaling (Thm 2-3): Translating Cor. 1 into Paradigm Complexities**

Both RL fine-tuning and test-time scaling are driven by "sampling + verification" and controlled by coverage. Using 0/1 outcome rewards $R^{f_{S^\star}}_x$ and staged curriculum rewards $R^{F_{S^\star}}_x(\cdot,\ell)$ (rewarding only prefixes with correct pre-EOS tokens), the paper proves curriculum RL fine-tuning complexity drops from $\widetilde O(d^{L+1})$ to $\widetilde O(L\cdot d^{p_\max+1})$. Test-time scaling (best-of-$N$ / verifier query) complexity follows the same exponential-to-polynomial reduction. For linearly realizable MDPs, it uses the spanner-sampling framework (Foster 2025) to provide a similar result for inference-time computational complexity (Thm 4).

### Loss & Training
As a theoretical paper, there is no specific loss implementation. Proofs use outcome-only rewards $R^{f_{S^\star}}_x\in\{0,1\}$ (checking if the pre-EOS token matches $\mu_{f_{S^\star}(x)}$); the curriculum version adds a depth parameter $R^{F_{S^\star}}_x(\cdot,\ell)$. To handle reward hacking (e.g., in parity where a wrong index might still result in a 50% match), the authors use prefix curricula in App. F-G to push reward signals down to intermediate EOS tokens, exponentially suppressing hacking probabilities.

## Key Experimental Results

### Main Results
The paper is primarily theoretical; experiments consist of small-scale simulations on "parity / countdown" to verify the predicted sample complexity ratios.

| Task | $d$ | Direct RL Convergence Steps | Curriculum RL Convergence Steps | Acceleration Ratio |
|------|-----|-----------------------------|---------------------------------|--------------------|
| Sparse Parity | 8 | $\sim$ Scale of $d^L$, >$10^5$ | $\sim$ Scale of $L\cdot d^{p_\max}$, approx $10^3$ | $\sim 50\times$ |
| Countdown-24 | 4 nums | Rarely converges | Stable convergence | Qualitative change |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full hint-decreasing curriculum | $\pi^\star$ learned with polynomial samples | Consistent with Thm 2 predictions |
| Removing intermediate stages (one-jump) | Sample complexity explosion | Confirms necessity of Cor. 1 |
| Overly granular stages ($L$ split into $2L$) | Diminishing but stable gains | Matches linear $L$ factor in $L\cdot(C^\star)^{p_\max}$ |

### Key Findings
- The acceleration ratio strongly depends on the "difficulty ratio $C^\star$" rather than the number of stages $L$: if $L$ is too large, it amplifies the linear factor, suggesting an optimal number of stages.
- The "uniform child node" assumption of PART is critical: if the base model has highly non-uniform probabilities, the $K$-th root relationship fails, and acceleration is consumed by local bottlenecks.
- Outcome-only rewards lead to hacking in parity; prefix curricula mitigate this by pushing signals to intermediate EOS tokens.
- Curriculum acceleration applies simultaneously to test-time scaling (verifier queries like best-of-$N$) and RL fine-tuning, as their complexity orders are parallel under the Cor. 1 framework.
- In tasks like Countdown (controllable branching factor, $\Phi_\ell$ as arithmetic operations), predicted polynomial sample counts closely match simulation convergence steps.

## Highlights & Insights
- Adapts the "coverage coefficient" from offline RL theory to CoT post-training, allowing difficulty, sample size, and reasoning depth to be measured on the same scale.
- The "$L/p$-th root condition" is a rare actionable design principle: it directs engineers to keep adjacent stage success rate gaps within the root of the target success rate.
- Unifies RL fine-tuning and test-time scaling within a single oracle-query framework, providing the first symmetric proof that "sample savings in training = query savings in inference."
- The 2S-ART abstraction covers multiple tasks previously analyzed separately (parity, Countdown, Markov chains, induction heads, causal reasoning).
- Prefix-hint and depth-increasing curricula are unified under the "uniform child probability" criterion, providing "microscopic sufficient conditions" for curriculum design.

## Limitations & Future Work
- Assumes the base model is a uniform PART; real LLM token probabilities are non-uniform, and attention often pre-aligns with specific paths.
- Proven only under outcome-only 0/1 rewards; extensions to process rewards (PRM) are not yet covered.
- Experimental scale is small (toy parity/countdown); verifying the root condition on real LLMs (like DeepSeek-R1) requires more complex difficulty metrics.
- Assumes prefix-hint curricula can be designed freely; in practice, determining which hint constitutes a $1/L$ difficulty stage requires trial and error.
- In multi-task mixed training, inconsistent $C^\star$ across tasks makes sharing a global $L/p$ relationship difficult, needing new tools for heterogeneous tasks.

## Related Work & Insights
- **vs Parashar et al. 2025**: They proposed curriculum via approximate policy iteration error accumulation; Cor. 1's hypothesis is a rigorous version of this using coverage language.
- **vs Liu et al. 2025b (hint-decreasing)**: Liu used prefix hints for easy-to-hard transitions; this paper interprets hint length as prefix-prefix relations on a reasoning tree, explaining their empirical success.
- **vs Foster et al. 2025 (spanner sampling)**: This paper reuses their coverage framework and extends it from linear MDPs to transformer reasoning trees, providing the analogue for inference-time complexity in Thm 4.
- **vs Ran-Milo et al. 2026 (graph traversal)**: Their graph traversal analysis can be viewed as a specific instance of Cor. 1—assigning non-zero probability to short-CoT instances is equivalent to satisfying the $K$-th root condition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorous exponential-to-polynomial sample-complexity proof for curriculum post-training.
- Experimental Thoroughness: ⭐⭐ Limited to toy simulations; lacks end-to-end validation on real LLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear combination of abstract framework and specific instances, though notation is somewhat heavy.
- Value: ⭐⭐⭐⭐ Provides quantifiable design principles for empirical "easy-to-hard" curricula, directly relevant to RLHF and R1-style training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ACL 2026\] Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study](../../ACL2026/reinforcement_learning/scaling_behaviors_of_llm_reinforcement_learning_post-training_an_empirical_study.md)
- [\[ICLR 2026\] Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains?](../../ICLR2026/reinforcement_learning/breaking_barriers_do_reinforcement_post_training_gains_transfer_to_unseen_domain.md)

</div>

<!-- RELATED:END -->
