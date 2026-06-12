---
title: >-
  [Paper Note] Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training
description: >-
  [ICML 2026][LLM Reasoning][Curriculum Learning] This work provides the first rigorous sample complexity proof for "easy-to-hard" curriculum RL post-training: on the state-conditional autoregressive reasoning tree of a tr…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Curriculum Learning"
  - "CoT Post-Training"
  - "Sample Complexity"
  - "Coverage Coefficient"
  - "Rejection Sampling"
date: 2026-05-08
content_hash: 407c85844424ef1c
---

# Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training

**Conference**: ICML 2026  
**arXiv**: [2511.07372](https://arxiv.org/abs/2511.07372)  
**Code**: https://github.com/DakeBU/Curriculum-Post-training (available)  
**Area**: LLM Reasoning / Reinforcement Learning / Learning Theory  
**Keywords**: Curriculum Learning, CoT Post-Training, Sample Complexity, Coverage Coefficient, Rejection Sampling

## TL;DR
This work provides the first rigorous sample complexity proof for "easy-to-hard" curriculum RL post-training: on the state-conditional autoregressive reasoning tree of a transformer, if the curriculum ensures that the difficulty ratio between adjacent stages is at most the $L/p$-th root of the target difficulty, then the total sample count can be reduced from the exponential $(C^\star)^L$ of direct training to the polynomial $L\cdot (C^\star)^{p_\max}$ of curriculum-based training.

## Background & Motivation
**Background**: The mainstream approach for CoT reasoning post-training is "direct RL fine-tune + 0/1 outcome verifier," combined with test-time scaling (beam / best-of-N) to improve pass@K. Over the past year, extensive empirical studies (Parashar 2025, Liu 2025, Bae 2025, etc.) have shown that "easy-to-hard curriculum" significantly accelerates convergence, but only as an empirical observation, lacking provable explanations.

**Limitations of Prior Work**: Classical curriculum learning theory is almost entirely built on "from-scratch training" scenarios (convex regression, parity, teacher-student perceptron, etc.), where "difficulty" and "performance" are task-specific and geometric, making them inapplicable to LLM post-training, which starts from a strong pretrained model and targets CoT generalization. The core feature of post-training is sparse rewards and extremely low sampling probability of correct trajectories under the base policy, a regime not addressed by previous theory.

**Key Challenge**: The difficulty of sparse-reward RL essentially arises from the rarity of correct CoTs under the base policy, characterized by the coverage coefficient $\|\pi^\star/\pi_{\text{ref}}\|_\infty$. Direct training incurs a sampling cost polynomial (or even exponential) in this rarity, while curriculum allows decomposing this "rarity ladder" into smaller steps—but there was no clear condition for how to split these steps to achieve super-polynomial acceleration.

**Goal**: (i) To provide necessary and/or sufficient conditions for curriculum post-training to yield exponential sample savings; (ii) To prove that these conditions naturally hold in the transformer + reasoning tree setting; (iii) To instantiate the conclusions for both RL fine-tune and test-time scaling paradigms.

**Key Insight**: The authors view post-training as redistributing probability on a "pre-trained reasoning tree," and adopt Foster 2025's spanner-sampling/coverage framework, directly linking curriculum difficulty to the number of rejection sampling attempts $\Theta(\|\pi^\star/\pi_{\text{ref}}\|_\infty \log\delta^{-1})$, thus unifying "difficulty" and "learning cost" under a single metric.

**Core Idea**: Model the base model as a PART that approximately uniformly samples legal child nodes on a state-conditional autoregressive reasoning tree (2S-ART), and prove that under PART, both "prefix-hint curriculum" and "depth-increasing curriculum" naturally form $K$-th root difficulty steps, reducing total cost from $(C^\star)^L$ to $L\cdot(C^\star)^{p_\max}$.

## Method

### Overall Architecture
The paper presents a theoretical framework rather than a new algorithm, abstracting the existing curriculum post-training pipeline into a provable form: (1) Model the task as a 2S-ART reasoning tree $F_{\text{2S-ART}}(\{\Phi_\ell\},\{I_\ell\})$, where at each step a token is selected from a legal index set $I_\ell$ and the state is updated via $\Phi_\ell$ as $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$; (2) Instantiate the base model as PART, i.e., uniform sampling over legal child nodes at each depth; (3) Show that a standard sampling-attention transformer (FFN implements $\Phi_\ell$, attention selects indices) can exactly realize PART; (4) Analyze, on this model, the sample complexity of RL fine-tune (Thm 2) and the oracle-query/computational complexity of test-time scaling (Thm 3).

### Key Designs

1. **Coverage Coefficient-Based Necessary/Sufficient Condition for Curriculum (Cor. 1)**:

    - **Function**: Reduces the ambiguous question of "whether curriculum truly saves samples over direct training" to a single inequality, providing a sufficient condition for exponential acceleration.
    - **Mechanism**: Defines $\varepsilon$-accuracy sample complexity $N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$, and via the rejection-sampling lemma $N\propto\|\pi^\star/\pi_{\text{ref}}\|_\infty$, ties it to the coverage coefficient. The necessary and sufficient condition is $\sum_{\ell}N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})<N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})$. When there exists an $L/p$-th root curriculum, i.e., $N_\varepsilon(\pi^\star_\ell\mid\pi^\star_{\ell-1})=\Theta(\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})})$, the ratio $N^{\text{direct}}/N^{\text{curr}}=\Theta((C^\star)^L/(L\cdot C^\star))$, where $C^\star=\sqrt[L/p]{N_\varepsilon(\pi^\star\mid\pi_{\text{ref}})}>1$, i.e., exponential acceleration.
    - **Design Motivation**: While empirical evidence for "easy-to-hard" is abundant, no one has answered "how to design the difficulty steps for real benefit." This condition not only explains the commonality of hint-decreasing (Liu 2025b), depth-increasing (Countdown/Parity), etc., but also provides a quantifiable design principle: the difficulty ratio between adjacent stages should be controlled at the $C^\star$ scale.

2. **2S-ART Reasoning Tree + PART Base Model (Def. 1-2)**:

    - **Function**: Provides a sufficiently general and analyzable abstraction for "reasoning task + weak base model," ensuring that the abstract condition in Cor. 1 is automatically satisfied on transformers.
    - **Mechanism**: Represents a length-$k$ reasoning task $f_{S_k}$ as an index path $S_k=(i_1,\dots,i_k,d{+}1)$, selecting $i_\ell$ from a legal set $I_\ell(\text{CoT}_{\ell-1})$ ($|I_\ell|=\Theta(d)$) at each step, with state update $z_\ell=\Phi_\ell(z_{\ell-1},v_{i_\ell})$. PART samples legal child nodes uniformly at each parent, yielding $\|\pi^\star_{S_{\ell+1}}/\pi^\star_{S_\ell}\|_\infty=\Theta(d)$, so $\|\pi^\star_{S_\ell}/\pi^\text{PART}\|_\infty=\Theta(d^{\ell+1})$, naturally satisfying the $L/p$-th root relation to target difficulty. Parity, Countdown, Markov-chain reasoning, and induction-head are all subclasses.
    - **Design Motivation**: The abstraction needs to cover parity, countdown, associative recall, etc., while keeping the "child node probability ratio" of the base model controllable; uniform-PART is the simplest choice meeting both, and aligns with the "post-training = reweighting on pre-trained tree" view of Snell 2025, Yue 2025.

3. **Exponential-to-Polynomial Reduction for RL Fine-Tune and Test-Time Scaling (Thm 2-3)**:

    - **Function**: Translates the abstract Cor. 1 into concrete sample/oracle complexity for two post-training paradigms.
    - **Mechanism**: Using 0/1 outcome reward $R^{f_{S^\star}}_x$ and staged curriculum reward $R^{F_{S^\star}}_x(\cdot,\ell)$ (rewarding only pre-EOS token correctness at each stage), proves that RL fine-tune under curriculum has sample complexity $\widetilde O(L\cdot d^{p_\max+1})$, much lower than the direct $\widetilde O(d^{L+1})$; test-time scaling (best-of-$N$/verifier query) has a similar exponential-to-polynomial reduction in oracle complexity. The paper also provides an analogous result for inference-time computational complexity on linearly realizable MDPs via Foster 2025's spanner-sampling (Thm 4).
    - **Design Motivation**: In practice, both RL fine-tune and test-time scaling are driven by "sampling + verification," both fundamentally controlled by coverage; unifying them explains why "training with short CoT/hint" saves samples during training and verifier queries during inference.

### Loss & Training
As a theoretical work, there is no specific loss function. The proofs use outcome-only reward $R^{f_{S^\star}}_x\in\{0,1\}$ (checks only if pre-EOS token matches $\mu_{f_{S^\star}(x)}$); the curriculum version adds a depth parameter $R^{F_{S^\star}}_x(\cdot,\ell)$. The challenge is that outcome reward allows reward hacking (e.g., in parity, choosing the wrong index still has a 50% hit rate); the authors show in App. F-G that prefix curriculum exponentially suppresses hacking probability by pushing the reward signal to intermediate EOS.

## Key Experimental Results

### Main Results
The paper is mainly theoretical; experiments are limited to small-scale simulations on "parity/countdown" to verify predicted sample complexity ratios.

| Task | $d$ | Direct RL Convergence Steps | Curriculum RL Convergence Steps | Acceleration Ratio |
|------|-----|----------------------------|-------------------------------|--------------------|
| Sparse Parity | 8 | $\sim d^L$ scale, >$10^5$ | $\sim L\cdot d^{p_\max}$, about $10^3$ | $\sim 50\times$ |
| Countdown-24 | 4 nums | Direct training barely converges | Curriculum version converges stably | Qualitative change |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Full hint-decreasing curriculum | Polynomial samples suffice to learn $\pi^\star$ | Matches Thm 2 prediction |
| Remove intermediate stages (jump directly to target) | Sample complexity explodes | Confirms necessity in Cor. 1 |
| Overfine stages ($L$ stages split into $2L$) | Diminishing but stable gain | Matches linear $L$ factor in $L\cdot(C^\star)^{p_\max}$ |

### Key Findings
- Acceleration ratio depends strongly on "difficulty ratio $C^\star$" rather than number of stages $L$: excessive $L$ amplifies the linear factor, so there is an optimal number of stages.
- The "uniform child node" assumption of PART is critical: if the base model is highly non-uniform on some child nodes, the $K$-th root relation fails and acceleration is bottlenecked.
- Outcome-only reward indeed leads to hacking in parity (wrong index but correct final bit); prefix curriculum alleviates this by pushing reward signals to intermediate EOS.
- Curriculum acceleration applies equally to test-time scaling (best-of-$N$ verifier query) and RL fine-tune; their complexity orders are fully parallel under the Cor. 1 framework.
- On countdown tasks with controllable child branching factor and arithmetic $\Phi_\ell$, theoretical polynomial sample predictions match simulated convergence steps best.

## Highlights & Insights
- Introduces the "coverage coefficient" from offline RL theory into CoT post-training, unifying difficulty, sample count, and reasoning depth under a single metric—this is the paper's most important conceptual contribution.
- The "$L/p$-th root condition" is a rare actionable design guideline: it tells engineers "the success rate gap between adjacent stages should not exceed the $L/p$-th root of the target task's success rate," directly guiding curriculum partitioning.
- Places RL fine-tune and test-time scaling in a unified oracle-query framework, providing the first symmetric proof that "training sample savings = inference verifier query savings."
- The 2S-ART abstraction covers parity, Countdown, Markov-chain, induction-head, causal graph reasoning, etc., previously analyzed separately; the abstraction itself is a valuable output.
- Prefix-hint and depth-increasing curricula, seemingly different, are unified under the "uniform child node probability at parent" principle, providing a "micro-level sufficient condition" for curriculum design.

## Limitations & Future Work
- Assumes the base model is uniform-PART, while real LLM token probabilities are far from uniform and attention is already biased toward certain paths, so there remains a theory-practice gap.
- Only proven for outcome-only 0/1 rewards; extension to process/intermediate rewards (the main focus of OpenR/Math-Shepherd, etc.) is not covered.
- Experiments are small-scale (toy parity, toy countdown); verifying the $L/p$-th root condition on real LLMs (e.g., DeepSeek-R1) requires more sophisticated difficulty metrics.
- Assumes free design of prefix-hint curriculum, but in practice, determining which hint segment constitutes $1/L$ stage difficulty often requires trial and error; the paper does not provide practical methods for estimating per-stage coverage.
- In multi-task/multilingual mixed training, different tasks have inconsistent difficulty ratios $C^\star$; the theory assumes a globally shared $L/p$ relation, and extension to heterogeneous tasks requires new tools.

## Related Work & Insights
- **vs Parashar et al. 2025**: They propose approximate policy iteration with error accumulation curriculum; the assumption in Cor. 1 is a strict version of theirs, translating error into coverage language.
- **vs Liu et al. 2025b (hint-decreasing)**: Liu implements easy→hard via prefix hint; this paper interprets hint length changes as prefix-prefix relations on the reasoning tree, providing a theoretical explanation for empirical effectiveness.
- **vs Foster et al. 2025 (spanner sampling)**: This work reuses their coverage framework and generalizes from linear MDPs to transformer reasoning trees, providing an analogous result for inference-time computational complexity (Thm 4).
- **vs Ran-Milo et al. 2026 (graph traversal)**: Their analysis of graph traversal tasks is a concrete instance of Cor. 1—assigning nonzero probability mass to short-CoT instances is equivalent to satisfying the $K$-th root condition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a rigorous exponential-to-polynomial sample-complexity proof for curriculum post-training
- Experimental Thoroughness: ⭐⭐ Only toy task simulations, no end-to-end validation on real LLMs
- Writing Quality: ⭐⭐⭐⭐ Clear combination of abstract framework and concrete instances (parity/countdown), though notation is heavy
- Value: ⭐⭐⭐⭐ Provides a quantifiable design guideline for empirically successful "easy-to-hard" curriculum, with direct implications for RLHF/R1-style training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Curriculum Abductive Learning](../../NeurIPS2025/llm_reasoning/curriculum_abductive_learning.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ACL 2026\] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](../../ACL2026/llm_reasoning/ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)
- [\[ICML 2026\] ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)
- [\[ACL 2026\] Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression](../../ACL2026/llm_reasoning/can_reasoning_path_still_be_effective_as_input_bridging_post-reasoning_to_chain-.md)

</div>

<!-- RELATED:END -->
