---
title: >-
  [Paper Note] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training
description: >-
  [ICML 2026][Reinforcement Learning][Curriculum Learning] This paper provides the first rigorous sample complexity proof for "easy-to-hard" curriculum RL post-training: on a transformer's state-conditioned autoregressive…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Curriculum Learning"
  - "CoT Post-training"
  - "Sample Complexity"
  - "Coverage Coefficient"
  - "Rejection Sampling"
date: 2026-05-08
content_hash: 196e280155856f8d
---

# Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training

**Conference**: ICML 2026  
**arXiv**: [2511.07372](https://arxiv.org/abs/2511.07372)  
**Code**: https://github.com/DakeBU/Curriculum-Post-training (Available)  
**Area**: LLM Reasoning / Reinforcement Learning / Learning Theory  
**Keywords**: Curriculum Learning, CoT Post-training, Sample Complexity, Coverage Coefficient, Rejection Sampling

## TL;DR
This paper provides the first rigorous sample complexity proof for "easy-to-hard" curriculum RL post-training: on a transformer's state-conditioned autoregressive reasoning tree, if the curriculum maintains the difficulty ratio between adjacent stages at the $L/p$-th root of the target difficulty, the total sample complexity can be reduced from exponential $(C^\star)^L$ in direct training to polynomial $L\cdot (C^\star)^{p_\max}$ in the curriculum version.

## Background & Motivation
**Background**: Post-training for CoT reasoning currently relies on "direct RL fine-tuning + 0/1 outcome verifier," combined with test-time scaling (beam search / best-of-N) to improve pass@K. Numerous recent empirical works (Parashar 2025, Liu 2025, Bae 2025, etc.) found that "easy-to-hard curriculum" significantly accelerates convergence, but these observations lack provable explanations.

**Limitations of Prior Work**: Classical curriculum learning theories are almost entirely established for "training from scratch" scenarios (convex regression, parity, teacher-student perceptron, etc.). Their definitions of "difficulty" and "performance" are task-specific and geometric, making them inapplicable to LLM post-training that starts from a strong pretrained model and aims for CoT generalization. The core characteristics of post-training—sparse rewards and the extremely low sampling probability of correct trajectories under the base policy—have no counterparts in older theories.

**Key Challenge**: The difficulty of sparse-reward RL essentially stems from the "rarity of correct CoT under the base policy," which can be characterized by the coverage coefficient $\|\pi^\star/\pi_{\text{ref}}\|_\infty$. Direct training incurs a sampling cost polynomial (or even exponential) to this rarity. While curriculums allow breaking this "rarity staircase" into smaller steps, the specific conditions under which these steps yield super-polynomial acceleration were previously undefined.

**Goal**: (i) Provide necessary and sufficient/sufficient conditions for exponential sample savings via curriculum post-training; (ii) Prove these conditions naturally hold for transformer-based reasoning trees; (iii) Apply these conclusions to both RL fine-tuning and test-time scaling paradigms.

**Key Insight**: The authors view post-training as re-distributing probabilities on a "pre-trained reasoning tree" and adopt the spanner-sampling/coverage framework of Foster 2025. By linking curriculum difficulty directly to the number of attempts in rejection sampling $\Theta(\|\pi^\star/\pi_{\text{ref}}\|_\infty \log\delta^{-1})$, they unify "difficulty" and "learning cost" under a single metric.

**Core Idea**: The base model is modeled as a PART (Pre-trained Autoregressive Reasoning Tree) that approximately uniformly samples valid child nodes on a 2S-ART (State-Conditioned Autoregressive Reasoning Tree). It is proven that under this PART, "prefix-hint curriculums" and "depth-increasing curriculums" naturally form $K$-th root difficulty steps, reducing the total cost from $(C^\star)^L$ to $L\cdot(C^\star)^{p_\max}$.

## Method

### Overall Architecture
The paper presents a theoretical framework rather than a new algorithm implementation, abstracting existing curriculum post-training pipelines: (1) Tasks are modeled as 2S-ART reasoning trees $F_{\text{2S-ART}}(\{\Phi_\ell\},\{I_\ell\})$, where each step chooses a token from a valid index set $I_\ell$ and updates the state $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$; (2) The base model is instantiated as a PART, i.e., uniform sampling of valid child nodes at each depth; (3) A standard sampling-attention transformer (FFN implementing $\Phi_\ell$ primitives, attention selecting indices) is proven to precisely replicate the PART; (4) The sample complexity for RL fine-tuning (Thm 2) and oracle-query/computational complexity for test-time scaling (Thm 3) are analyzed on this model.

### Key Designs

1.  **Necessary/Sufficient Conditions for Curriculum Based on Coverage Coefficient (Cor. 1)**:
    -   **Function**: Transforms the ambiguous question of "whether a curriculum truly saves samples" into a single-line inequality and provides sufficient conditions for exponential acceleration.
    -   **Mechanism**: Defines $\varepsilon$-accuracy sample complexity $N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$. Using the rejection-sampling lemma $N\propto\|\pi^\star/\pi_{\text{ref}}\|_\infty$, it is bound to the coverage coefficient. The condition is $\sum_{\ell}N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})<N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$. When an $L/p$-th root curriculum exists, i.e., $N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})=\Theta(\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})})$, the ratio is $N^{\text{direct}}/N^{\text{curr}}=\Theta((C^\star)^L/(L\cdot C^\star))$ where $C^\star > 1$, achieving exponential acceleration.
    -   **Design Motivation**: While empirical evidence suggests "easy-to-hard works," the design of the difficulty staircase remained intuitive. This condition explains commonalities between hint-decreasing (Liu 2025b) and depth-increasing (Countdown/Parity) curriculums and provides a quantifiable principle: the difficulty ratio between adjacent stages should be controlled at the $C^\star$ scale.

2.  **2S-ART Reasoning Tree + PART Base Model (Def. 1-2)**:
    -   **Function**: Provides a general yet analytically tractable abstraction for "reasoning tasks + weak base models," allowing Cor. 1 to be satisfied by transformers.
    -   **Mechanism**: Represents a length-$k$ reasoning task $f_{S_k}$ as an index path $S_k=(i_1,\dots,i_k,d{+}1)$. At each step, $i_\ell$ is selected from a valid set $I_\ell(\text{CoT}_{\ell-1})$ ($|I_\ell|=\Theta(d)$), and the state is updated $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$. The PART uniformly samples valid children at each node, yielding $\|\pi^\star_{S_{\ell+1}}/\pi^\star_{S_\ell}\|_\infty=\Theta(d)$. Thus, $\|\pi^\star_{S_\ell}/\pi^\text{PART}\|_\infty=\Theta(d^{\ell+1})$, naturally satisfying the $L/p$-th root relationship with target difficulty. Parity, Countdown, Markov-chain reasoning, and induction-heads are all subclasses.
    -   **Design Motivation**: The theory must encompass various tasks while keeping the base model's "child node probability ratio" controllable. Uniform-PART is the simplest choice and aligns with the perspective of Snell 2025 and Yue 2025 that "post-training equals reweighting on a pre-trained tree."

3.  **Exponential-to-Polynomial Reduction for RL Fine-tune and Test-time Scaling (Thm 2-3)**:
    -   **Function**: Translates Cor. 1 into specific sample/oracle complexities for two post-training paradigms.
    -   **Mechanism**: Uses 0/1 outcome rewards $R^{f_{S^\star}}_x$ and segmented curriculums $R^{F_{S^\star}}_x(\cdot,\ell)$ (rewarding only when the prefix is correct). Proves that RL fine-tuning sample complexity becomes $\widetilde O(L\cdot d^{p_\max+1})$ under curriculum, far lower than $\widetilde O(d^{L+1})$ for direct training. Test-time scaling (best-of-$N$ / verifier query) exhibits the same reduction. Analogue results for inference-time computational complexity are provided via spanner-sampling (Thm 4).
    -   **Design Motivation**: In practice, both RL fine-tuning and test-time scaling are driven by "sampling + verification" and are governed by coverage. A unified framework explains why training with short CoTs/hints saves training samples and inference-time verifier queries.

### Loss & Training
The theoretical paper contains no specific loss function. The proofs use outcome-only rewards $R^{f_{S^\star}}_x\in\{0,1\}$ (checking if the pre-EOS token matches $\mu_{f_{S^\star}(x)}$) and a curriculum version with a depth parameter $R^{F_{S^\star}}_x(\cdot,\ell)$. To address reward hacking (e.g., a 50% hit rate for wrong indices in parity), prefix curriculums are used to exponentially suppress hacking probabilities.

## Key Experimental Results

### Main Results
The paper is primarily theoretical; experiments are small-scale simulations on "parity/countdown" to verify predicted sample complexity ratios.

| Task | $d$ | Direct RL Convergence Steps | Curriculum RL Convergence Steps | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| Sparse Parity | 8 | $\sim d^L$ scale, $>10^5$ | $\sim L\cdot d^{p_\max}$ scale, $\sim 10^3$ | $\sim 50\times$ |
| Countdown-24 | 4 nums | Rarely converges | Stable convergence | Qualitative |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full hint-decreasing curriculum | $\pi^\star$ learned with poly samples | Consistent with Thm 2 |
| Removing intermediate stages | Sample complexity explodes | Validates necessity of Cor. 1 |
| Fine-grained stages ($2L$ stages) | Diminishing but stable gains | Matches linear $L$ factor in $L\cdot(C^\star)^{p_\max}$ |

### Key Findings
-   Speedup depends strongly on the "difficulty ratio $C^\star$" rather than the number of stages $L$. Too many stages increase the linear factor $L$, suggesting an optimal stage count.
-   The "uniform child node" assumption for PART is critical: if the base model's probabilities are highly skewed at certain nodes, the $K$-th root relationship fails, and local bottlenecks eliminate acceleration.
-   Outcome-only rewards show hacking in parity (wrong index but correct final bit); prefix curriculums significant mitigate this by pushing signals to intermediate tokens.
-   Curriculum acceleration applies to both test-time scaling and RL fine-tuning; their complexity classes are parallel under the Cor. 1 framework.

## Highlights & Insights
-   The application of the "coverage coefficient" from offline RL theory to CoT post-training provides a unified language for measuring difficulty, sample count, and reasoning depth.
-   The "$L/p$-th root condition" is a rare actionable design principle: it tells engineers that the success rate gap between adjacent stages should not exceed the $L/p$-th root of the target task's success rate.
-   Providing a symmetric proof for training samples and inference verifier queries within a single oracle-query framework is a significant conceptual contribution.
-   The 2S-ART abstraction covers multiple previously disparate task classes (parity, Countdown, induction-heads, etc.), demonstrating strong theoretical generality.
-   Unifying prefix-hint and depth-increasing curriculums under the "uniform child node probability" principle provides a "microscopic sufficient condition" for curriculum design.

## Limitations & Future Work
-   The uniform-PART assumption deviates from real LLM token probabilities, which are non-uniform and often biased toward certain paths by pre-training.
-   Proofs are limited to outcome-only 0/1 rewards; extension to process rewards (PRM) is not yet covered.
-   Experimental scale is small (toy tasks); validating the $L/p$-th root condition on real LLMs (e.g., DeepSeek-R1) requires more sophisticated difficulty metrics.
-   The theory assumes prefix-hint curriculums can be freely designed, while in practice, identifying which hints constitute a "$1/L$ difficulty stage" requires trial and error.
-   In multi-task/multi-lingual scenarios, difficulty ratios $C^\star$ across tasks may vary, requiring further tools for heterogeneous task expansion.

## Related Work & Insights
-   **vs. Parashar et al. 2025**: They proposed curriculum based on approximate policy iteration error accumulation; this paper's Cor. 1 is a more rigorous version translated into coverage language.
-   **vs. Liu et al. 2025b (hint-decreasing)**: While Liu used prefix hints empirically, this paper interprets hint length variations as prefix-prefix relationships on reasoning trees, providing a theoretical foundation.
-   **vs. Foster et al. 2025 (spanner sampling)**: This paper adapts their coverage framework from linear MDPs to transformer reasoning trees, providing analogue results for inference-time complexity.
-   **vs. Ran-Milo et al. 2026 (graph traversal)**: Their analysis of graph traversal can be seen as a specific instance of Cor. 1—assigning non-zero mass to short CoT instances is equivalent to satisfying the $K$-th root condition.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ First rigorous exponential-to-polynomial sample-complexity proof for curriculum post-training.
-   Experimental Thoroughness: ⭐⭐ Limited to toy task simulations without end-to-end real LLM validation.
-   Writing Quality: ⭐⭐⭐⭐ Clear combination of abstract frameworks and specific instances, though notation is somewhat heavy.
-   Value: ⭐⭐⭐⭐ Provides quantifiable design principles for empirical "easy-to-hard" training, directly relevant to RLHF and R1-style training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess](how_reasoning_evolves_from_post-training_data_an_empirical_study_using_chess.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)

</div>

<!-- RELATED:END -->
