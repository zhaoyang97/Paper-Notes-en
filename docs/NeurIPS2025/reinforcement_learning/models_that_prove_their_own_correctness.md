---
title: >-
  [Paper Note] Models That Prove Their Own Correctness
description: >-
  [NeurIPS 2025][Reinforcement Learning][self-proving models] This paper proposes the Self-Proving Models framework, in which a model proves the correctness of its outputs to a verifier algorithm via an interactive proof system. Two learning algorithms are introduced—Transcript Learning (TL) and Reinforcement Learning from Verifier Feedback (RLVF)—and experiments on the GCD computation task demonstrate that Annotated TL achieves 96% Verifiability.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "self-proving models"
  - "interactive proofs"
  - "reinforcement learning from verifier feedback"
  - "formal verification"
  - "LLM reliability"
date: 2026-05-08
content_hash: d3c7d9da78e00800
---

# Models That Prove Their Own Correctness

**Conference**: NeurIPS 2025
**arXiv**: [2405.15722](https://arxiv.org/abs/2405.15722)  
**Code**: [https://github.com/orrp/self-proving-models](https://github.com/orrp/self-proving-models)  
**Area**: Reinforcement Learning
**Keywords**: self-proving models, interactive proofs, reinforcement learning from verifier feedback, formal verification, LLM reliability

## TL;DR
This paper proposes the Self-Proving Models framework, in which a model proves the correctness of its outputs to a verifier algorithm via an interactive proof system. Two learning algorithms are introduced—Transcript Learning (TL) and Reinforcement Learning from Verifier Feedback (RLVF)—and experiments on the GCD computation task demonstrate that Annotated TL achieves 96% Verifiability.

## Background & Motivation
Model accuracy is typically measured as average performance over an input distribution, which means that for any **specific input**, the correctness of a model's output cannot be guaranteed. A student using an LLM to solve an algebra problem cannot be certain whether the answer to a particular question is correct, even if the LLM achieves 90% accuracy on some benchmark. This reflects the fundamental tension between average-case guarantees and worst-case guarantees in machine learning.

Requiring models to produce natural-language explanations is insufficient—a model may convince users to accept a wrong answer through plausible-sounding reasoning, and even when the answer is correct, the model may fail to generate a convincing proof. What is truly needed is a **formal proof** that allows a verification algorithm—rather than human judgment—to confirm correctness.

The core idea draws from Interactive Proof (IP) theory: the model plays the role of a "prover" and proves the correctness of each output to a hand-crafted efficient verifier algorithm. The verifier's **completeness** guarantees that correct outputs can always be proven, while its **soundness** guarantees that incorrect outputs cannot be accepted—regardless of the prover's computational power. This elevates ML's average-case guarantees to per-input worst-case guarantees.

## Method

### Overall Architecture
The Self-Proving Model $P_\theta$ operates as follows: given input $x$, the model first generates output $y \sim P_\theta(x)$, then engages in $R$ rounds of interaction with the verifier $V$. In each round, $V$ sends a query $q_i$ and $P_\theta$ replies with answer $a_i$. The verifier then decides to accept or reject. The training objective is to maximize Verifiability:

$$\text{ver}_{V,\mu}(\theta) := \Pr_{x \sim \mu, y \sim P_\theta(x)} [\langle V, P_\theta \rangle(x,y) \text{ accepts}] \geq \beta$$

The key insight is that the verifier $V$ is **hand-crafted** (not learned), providing formal soundness guarantees. Verifiability implies Correctness: if $P_\theta$ is $\beta$-Verifiable and $V$ has soundness error $s$, then $P_\theta$ is $(\beta - s)$-correct.

### Key Designs
1. **Transcript Learning (TL)**:

    - Function: Supervised learning based on accepting transcripts
    - Mechanism: The model is trained not only as $x \mapsto y^*$ (correct output) but as $x \mapsto y^* \pi^*$ (correct output + accepting transcript), enabling the model to autoregressively learn to generate complete interaction sequences accepted by the verifier
    - Gradient estimation: TL estimates the gradient of a lower-bound function $A(\theta) \leq \text{ver}_V(\theta)$, where $A(\theta) := \Pr[\pi = \pi^*]$ is the probability that the current model's transcript matches the honest transcript
    - Update rule: $\theta_{i+1} := \theta_i + \lambda \cdot \prod_{s} \alpha_s(\theta_i) \cdot \sum_s \vec{d}_s(\theta_i)$, where $\alpha_s$ is the token-level selection probability and $\vec{d}_s$ is the corresponding log-probability gradient
    - Design Motivation: Analogous to process supervision in chain-of-thought, providing intermediate-step supervision signals to accelerate learning

2. **Reinforcement Learning from Verifier Feedback (RLVF)**:

    - Function: Learns without accepting transcripts, relying solely on the verifier's accept/reject feedback
    - Mechanism: The model generates a transcript $\pi \sim P_\theta$, the verifier judges acceptance, and parameters are updated only upon acceptance
    - Key gradient formula: $\nabla_\theta \text{ver}(\theta) = \mathbb{E}[\text{Acc}_V(\cdot) \cdot \sum_s \vec{d}_s(\theta)]$, where $\text{Acc}_V$ is a binary indicator
    - Characteristics: An on-policy algorithm with simpler update steps than TL (no need to track $\alpha_s$), but requires initial Verifiability $> 0$ to sample accepting transcripts
    - Design Motivation: Analogous to RLHF, but with an algorithmic verifier replacing human feedback; the verifier can be efficiently simulated without additional annotation cost

3. **Annotated Transcript Learning (ATL)**:

    - Function: Augments TL with "annotations"—intermediate computation steps in the proof process
    - Mechanism: The proof $\pi$ is extended to $\tilde{\pi} = A(x, \pi)$, containing intermediate derivation steps for the first $T$ rounds; at inference time, an extractor $E$ strips the annotations before sending the actual proof
    - Design Motivation: Annotations serve as chain-of-thought, substantially reducing the learning difficulty. In experiments, ATL raises Verifiability from 60.3% (TL) to 96%

### Loss & Training
- TL: Maximizes the log-likelihood of accepting transcripts, effectively performing gradient ascent on $A(\theta)$
- RLVF: REINFORCE-style policy gradient with the verifier's binary decision as reward
- Recommended combination: First apply TL to obtain a base Self-Proving model with $\delta > 0$, then apply RLVF to amplify Verifiability
- Convergence guarantee: Under convexity and Lipschitz conditions, TL outputs a $(1-\varepsilon)$-Self-Proving model after $O(C^2 B_\text{Norm}^2 B_\text{Lip}^2 / \varepsilon^2)$ samples

## Key Experimental Results

### Main Results

| Learning Method | Correctness | Verifiability | Notes |
|-----------------|-------------|---------------|-------|
| GPT (baseline) | 99.8% | — | Computes GCD but cannot prove it |
| GPT+TL | 98.8% | 60.3% | 100K iterations |
| GPT+TL+RLVF | 98.9% | 78.3% | 4M RLVF iterations |
| GPT+ATL | 98.6% | **96.0%** | 100K iterations; annotations yield substantial gains |

### Ablation Study

| Configuration (annotation steps T) | Verifiability | Notes |
|-------------------------------------|---------------|-------|
| T=0 (no annotation, TL) | ~60% | Baseline |
| T=1 | ~82% | Annotations begin to take effect |
| T=3 | ~91% | Continued improvement |
| T=5 | ~96% | Near saturation |
| T=7 | ~96% | Diminishing returns with further annotations |

### Key Findings
- Even when Correctness approaches 100%, Verifiability requires dedicated training methods to improve—"being able to compute" does not imply "being able to prove"
- Subsequent implementations of RLVF (e.g., RLVR with KL regularization) have achieved broad success on full-scale LLMs, validating the practical value of this paper's theoretical framework
- Annotations (chain-of-thought) have a decisive impact on Verifiability, and models generalize to cases beyond the annotated length

## Highlights & Insights
- **Bridging theory and practice**: This work applies interactive proof systems—a classical concept from theoretical computer science—to the problem of ML model trustworthiness, formally defining Verifiability and providing actionable training algorithms. RLVF has since become a core technique in LLM post-training.
- **From average-case to per-input guarantees**: Soundness holds for all inputs (worst-case), enabling users to trust model outputs on a per-instance basis—by running the verifier rather than relying on benchmark statistics.

## Limitations & Future Work
- Experiments are conducted only on the small-scale GCD task (6.3M parameter GPT); comprehensive validation on large-scale LLMs and complex reasoning tasks is lacking
- Convergence theory for RLVF remains incomplete, depending on open problems in policy gradient convergence
- A hand-crafted efficient verifier must be predefined—designing such a verifier is itself non-trivial for many practical tasks
- The soundness error $s$ provides meaningful guarantees only when $\text{ver} > s$; reliability guarantees are limited in extremely low-Verifiability settings
- TL is sensitive to the quality of the transcript generator; low-quality transcripts substantially degrade learning efficiency
- Formally defining verifiers and proofs for natural language reasoning tasks remains an open problem

## Related Work & Insights
- **vs Prover-Verifier Games (PVGs)**: PVGs employ a learned verifier, whereas this work uses a hand-crafted verifier—the latter provides formal soundness guarantees immune to adversarial attacks
- **vs RLHF**: RLVF can be viewed as an "idealized" version of RLHF—replacing noisy human preferences with a deterministic mathematical verifier, yielding perfectly unambiguous reward signals
- **vs Chain-of-Thought**: TL can be interpreted as verifier-induced CoT training, but unlike standard CoT, the intermediate steps here carry formal correctness semantics
- **vs IP-PAC**: IP-PAC lets a learner prove that "the model is approximately correct over a distribution," whereas this work enables per-input correctness proofs, offering stronger guarantees

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering integration of interactive proofs into the ML training framework; RLVF has since been widely adopted
- Experimental Thoroughness: ⭐⭐⭐ Validation is limited to a single small-scale GCD task
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, conceptually clear, with well-structured theoretical development
- Value: ⭐⭐⭐⭐⭐ RLVF has been widely adopted as a standard LLM post-training method (e.g., RLVR), with far-reaching impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[ICLR 2026\] Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](../../ICLR2026/reinforcement_learning/beyond_binary_rewards_training_lms_to_reason_about_their_uncertainty.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
