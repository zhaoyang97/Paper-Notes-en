---
title: >-
  [Paper Note] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning
description: >-
  [NeurIPS 2025][Co-evolution] This paper proposes CURE, a framework in which a single LLM simultaneously assumes the roles of code generator and unit test generator. Cross-execution between generated code and generated tests constructs a pairwise reward matrix; theoretically derived reward signals then drive reinforcement learning. Without any ground-truth code annotations, CURE achieves co-evolution of both code generation and unit test generation capabilities, substantially outperforming dedicated coder models of comparable scale across five programming benchmarks.
tags:
  - NeurIPS 2025
  - Co-evolution
  - Reinforcement Learning
  - Unit Test Generation
  - Code Generation
  - Self-play
  - Reward Precision
date: 2026-05-08
content_hash: f14e9723d90f4cdd
---

# Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.03136](https://arxiv.org/abs/2506.03136)
**Code**: [GitHub](https://github.com/Gen-Verse/CURE)
**Area**: Code Intelligence / LLM Reasoning
**Keywords**: Co-evolution, Reinforcement Learning, Unit Test Generation, Code Generation, Self-play, Reward Precision

## TL;DR

This paper proposes CURE, a framework in which a single LLM simultaneously assumes the roles of code generator and unit test generator. Cross-execution between generated code and generated tests constructs a pairwise reward matrix; theoretically derived reward signals then drive reinforcement learning. Without any ground-truth code annotations, CURE achieves co-evolution of both code generation and unit test generation capabilities, substantially outperforming dedicated coder models of comparable scale across five programming benchmarks.

## Background & Motivation

**Background**: In recent years, large language models have achieved remarkable progress in mathematical reasoning and code generation, largely driven by post-training optimization (e.g., RL) and test-time scaling techniques. In code generation, unit tests have emerged as a highly promising auxiliary signal — both as reward signals for RL training and as filters in Best-of-N (BoN) inference strategies. Unlike scalar or generative reward models, generated unit tests can be efficiently reused across all candidate solutions, avoiding quadratic complexity. More importantly, generating a unit test does not require the model to produce a complete solution — a reasonable assert statement is far simpler than solving the entire problem, making unit test generation a logically "easier" task.

**Limitations of Prior Work**: Despite the widely recognized value of unit tests, existing methods for training unit test generators (e.g., O1-Coder, UTGEN) rely on ground-truth code annotations — requiring the prior collection of correct code solutions to construct training data. This introduces two fundamental problems: high annotation cost and difficult data collection limit training scale and domain diversity; and the static nature of training data prevents the unit test generator from learning from dynamically emerging error patterns.

**Key Challenge**: Training a unit test generator requires ground-truth code, yet collecting ground-truth code is itself an expensive and hard-to-scale process. This creates a chicken-and-egg dilemma: good unit tests require good code for supervision, while good filtering requires good unit tests as reward signals. At a deeper level, even when some ground-truth code is available, it represents only "one correct implementation," whereas unit tests should generalize — passing all correct implementations rather than depending on any particular one. Unit tests trained on specific code may overfit to implementation details and lose generalizability.

**Goal**: This paper addresses a central question: **Can a unit test generator and a code generator co-evolve effectively in the complete absence of ground-truth code?** This decomposes into three sub-questions: (1) How can unsupervised reward signals be designed so that the two roles learn from each other? (2) How can reward precision be guaranteed to avoid trivial or incorrect test generation? (3) How can this framework be applied to long-CoT models while maintaining inference efficiency?

**Key Insight**: The key observation is that during RL training, the code generator naturally produces both correct and incorrect solutions. Incorrect code naturally exposes typical failure patterns, which are extremely valuable training material for the unit test generator — the tester must learn to distinguish good code from bad, and incorrect code provides the discriminative training signal. In turn, improved unit tests better filter generated code, forming a positive feedback loop. This observation eliminates the dependence on ground-truth code.

**Core Idea**: Construct self-play rewards using the cross-execution matrix between code and unit tests, and guide unit test generator optimization via a theoretically derived reward precision objective $\mu$, enabling unsupervised coder-tester co-evolution.

## Method

### Overall Architecture

CURE is a co-evolutionary framework based on self-play reinforcement learning. The overall pipeline is as follows: given a set of programming tasks (each accompanied by a small number of ground-truth unit tests), a single policy LLM generates $n$ candidate code solutions and $m$ candidate unit tests for each task. All code solutions are then executed against all unit tests, producing a binary evaluation matrix $\mathcal{B}^{\star} \in \{0,1\}^{n \times (m+t_q)}$, where the last $t_q$ columns correspond to ground-truth unit tests. Based on this matrix, reward values are estimated for each code solution and each unit test, and a GRPO/PPO-style policy optimization algorithm alternately improves code generation and unit test generation capabilities. The entire process iterates, with the quality of both code and tests continuously improving.

The key intuition behind this design is: ground-truth unit tests are used to judge the correctness of code (as the reward source for code), while the correctness information about code is used to estimate the quality of each generated unit test (as the reward source for tests). The two roles achieve mutual supervision through the execution matrix without any additional annotations. Computational overhead in the pipeline is concentrated in the rollout generation and cross-execution phases; policy optimization itself is identical to standard GRPO. An engineering highlight is the use of vLLM for efficient batched inference, enabling 16×16 rollout generation to remain feasible on 8× A100 GPUs.

### Key Designs

1. **Theoretical Derivation of Reward Precision**:

    - *Function*: Define and analyze "reward precision" — how accurately a set of generated unit tests can distinguish correct from incorrect code — as the theoretical optimization objective for the unit test generator.
    - *Mechanism*: The authors first define reward precision as $P(\mathcal{R}_{s_{j_1}} > \mathcal{R}_{s_{j_2}} \mid s_{j_1} \text{ correct}, s_{j_2} \text{ incorrect})$, where $\mathcal{R}_{s_j} = \sum_{l=1}^{m} \mathcal{B}_{j,l}$ is the number of unit tests passed by code $s_j$. A generative model is then introduced: the probability that a generated test is correct is $p_u$; correct code passes a correct test with probability $p_{11}=1$; incorrect code passes a correct test with probability $p_{01}$; incorrect code passes an incorrect test with probability $p_{00}$; correct code passes an incorrect test with probability $p_{10}=0$. Via Hoeffding's inequality, the authors prove that the necessary and sufficient condition for reward precision to approach 1 is $\mu > 0$, where $\mu = p_u(1-p_{01}) - (1-p_u)p_{00}$, with convergence rate satisfying the exponential bound $P(\mathcal{R}_{s_{j_1}} > \mathcal{R}_{s_{j_2}}) \geq 1 - e^{-\mu^2 m / 8}$. This implies that a larger $\mu$ requires fewer unit tests to reliably distinguish code quality.
    - *Design Motivation*: $\mu$ not only provides the convergence condition for reward precision but also directly controls the convergence rate. Treating $\mu$ as the individual-level optimization objective for each unit test is therefore the theoretically optimal choice. Intuitively, optimizing $\mu$ is equivalent to simultaneously increasing the test correctness rate $p_u$, decreasing the escape probability $p_{01}$ for incorrect code, and decreasing the false positive rate $p_{00}$.

2. **Individual Reward Estimation from the Execution Matrix**:

    - *Function*: Estimate the individual contribution reward $\mathcal{R}_{u_k}^{\star}$ for each generated unit test from the execution matrix.
    - *Mechanism*: For code rewards, the number of ground-truth unit tests passed is used directly: $\mathcal{R}_{s_j}^{\star} = \sum_{l=1}^{t_q} \mathcal{B}_{j,m+l}^{\star}$. For unit test rewards, the core formula is $\mathcal{R}_{u_k}^{\star} = -\sum_{l=1}^{n}(1-\mathcal{I}_{s_l})\mathcal{B}_{l,k}^{\star} + (\prod_{l=1}^{n} \mathcal{I}_{s_l} \mathcal{B}_{l,k}^{\star})(\sum_{l=1}^{n}(1-\mathcal{I}_{s_l}))$, where $\mathcal{I}_{s_j} = \prod_{l=1}^{t_q} \mathcal{B}_{j,m+l}^{\star}$ indicates whether code $s_j$ passes all ground-truth unit tests (i.e., whether it is judged "correct"). The derivation proceeds by first estimating parameters $p_u$, $p_{01}$, and $p_{00}$ from execution results, then substituting into the expression for $\mu$.
    - *Design Motivation*: The intuition behind this reward function is clear — when unit test $u_k$ passes all correct code, the second term is positive and proportional to the number of incorrect code solutions, indicating the test's ability to discriminate code quality and yielding a positive reward; when $u_k$ causes some correct code to fail, only the negative contribution of the first term remains, indicating an incorrect test that is penalized. Using "whether the test passes all correct code" (i.e., an estimate of $p_u$) alone as the reward would cause the model to generate trivially permissive tests — since any test that does not reject correct code earns a high score, such tests also let through large amounts of incorrect code. CURE's reward design avoids this pitfall.

3. **Policy Optimization for Co-Evolution**:

    - *Function*: Alternately optimize both code generation and unit test generation objectives using a GRPO-style policy gradient method.
    - *Mechanism*: The same model alternates between the two roles. For the coder role, parameters are updated with $\mathcal{J}(\theta, \{s_j\}_{j=1}^n)$, with rewards derived from the number of ground-truth unit tests passed; for the tester role, parameters are updated with $\mathcal{J}(\theta, \{u_k\}_{k=1}^m)$, with rewards derived from the $\mu$-based theoretical derivation above. The two phases are executed sequentially at each iteration step. The optimization objective uses a standard PPO clipped surrogate objective with KL divergence regularization: $\mathcal{J}(\theta) = \mathbb{E}[\min(\frac{\pi_\theta}{\pi_{\theta_{old}}} A, \text{clip}(\frac{\pi_\theta}{\pi_{\theta_{old}}}, \epsilon) A)] - \beta D_{KL}[\pi_\theta \| \pi_{ref}]$.
    - *Design Motivation*: Sharing a single model rather than training two independent models is motivated by the observation that code generation and unit test generation share underlying programming comprehension — the ability to understand code logic benefits both writing code and writing tests. Alternating optimization enables the two capabilities to mutually reinforce each other over shared representations. KL regularization prevents the model from drifting too far and causing training instability, which is especially important in self-play settings where the reward signal itself is non-stationary.

4. **Efficiency Optimization for Long-CoT Models**:

    - *Function*: Design a response-length-guided reward transformation for long-CoT reasoning models that substantially reduces generation length without sacrificing unit test quality.
    - *Mechanism*: Given normalized rewards $\{r_i\}$ and corresponding response lengths $\{l_i\}$, a length transformation is first applied: $\hat{r}_i = -l_i + T_l$ (if $r_i > 0$) or $\hat{r}_i = -l_{max} + T_l$ (if $r_i \leq 0$), where $T_l = \text{median}\{l_j \mid r_j > 0\}$. Normalization is then applied via a balance factor $\alpha$ between positive and negative samples and standard deviation $\sigma$. Responses exceeding 8K tokens are truncated; only the first 8K tokens are retained for training.
    - *Design Motivation*: Long-CoT models (e.g., Qwen3-4B) exhibit strong reasoning capabilities but extremely slow inference speed, making their cost prohibitive in scenarios that require extensive unit test generation. This transformation cleverly introduces a length penalty while preserving reward polarity (correct samples remain positive, incorrect samples remain negative) — shorter correct responses receive higher rewards, encouraging the model to achieve the same accuracy with fewer reasoning steps. Experiments demonstrate that this strategy reduces average response length to 64.8% of the original.

### Loss & Training

Training uses a GRPO-style policy optimization with a clipped surrogate objective and a KL divergence regularization term, preventing excessive policy drift due to the dynamic nature of reward signals in the self-play environment. Specific configuration: learning rate $1 \times 10^{-6}$, KL coefficient $\beta = 0.01$. Sampling temperature is 1.0 with top-p of 1.0 (reduced to 0.8 for long-CoT models to improve unit test generation stability). Each step generates 16 code rollouts and 16 unit test rollouts, producing $16 \times 16 = 256$ cross-executions. The 7B and 14B models are trained for 350 steps; the 4B long-CoT model converges in only 50 steps. Training is conducted on 8× A100 GPUs.

Training data consists of only 4.5K programming problems of difficulty ≤2 from CodeContests — a remarkably small training set. Notably, significant cross-benchmark improvements emerge from this minimal data, demonstrating the framework's data efficiency. This is directly attributable to the unsupervised approach — only problem statements and a small number of ground-truth unit tests are required, with no need for correct code solutions.

Stable co-evolutionary dynamics are observable throughout training: unit test accuracy, code accuracy, and estimated reward $\mu$ all exhibit continuously increasing trends without the instability oscillations typical of self-play. This is attributed to KL regularization limiting update magnitude at each step, and to the anchoring effect of ground-truth unit tests — they provide an external reference point for code rewards, preventing the reward signal from entering a fully self-referential loop.

## Key Experimental Results

### Main Results

Three metrics are evaluated across five programming benchmarks: unit test accuracy (UT), one-shot code accuracy (Code), and Best-of-N accuracy (BoN, 16 code × 16 unit tests). These five benchmarks span difficulty levels ranging from basic programming (MBPP) to competition-level (CodeContests, CodeForces), as well as contamination-free evaluation (LiveBench, LiveCodeBench).

| Model | LiveBench UT/Code/BoN | MBPP UT/Code/BoN | LiveCodeBench UT/Code/BoN | CodeContests UT/Code/BoN | CodeForces UT/Code/BoN |
|---|---|---|---|---|---|
| Qwen2.5-14B-Instruct | 27.8/36.4/51.7 | 72.8/76.3/83.2 | 35.7/33.5/45.1 | 43.8/25.6/33.4 | 20.7/7.3/12.5 |
| Qwen2.5-14B-Coder | 39.0/42.2/53.1 | 75.1/72.6/84.9 | 41.6/38.2/47.7 | 37.3/23.3/32.0 | 22.1/7.8/13.5 |
| **ReasonFlux-14B** | **73.3/47.5/60.2** | **91.6/78.5/88.2** | **81.4/40.5/50.5** | **86.0/32.1/44.4** | **82.3/12.1/25.9** |
| Qwen2.5-7B-Instruct | 26.5/31.1/35.9 | 35.8/66.3/79.4 | 28.6/26.9/32.6 | 26.7/21.2/25.8 | 18.9/5.4/8.9 |
| Qwen2.5-7B-Coder | 19.3/35.0/42.9 | 41.3/68.0/79.6 | 20.6/29.8/34.8 | 12.9/22.8/23.8 | 7.2/6.7/9.1 |
| **ReasonFlux-7B** | **54.8/37.1/51.6** | **79.4/70.2/84.6** | **57.7/31.2/42.7** | **62.6/25.9/34.1** | **45.6/8.2/16.1** |
| Qwen3-4B (Long-CoT) | 36.8/72.5/78.1 | 76.5/88.4/90.1 | 50.9/74.5/80.0 | 43.6/53.0/58.3 | 54.1/28.8/38.5 |
| **ReasonFlux-4B** | **84.6/74.6/82.0** | **83.3/89.5/91.1** | **86.8/74.9/80.6** | **72.2/54.6/59.9** | **65.8/30.9/40.2** |

### Ablation Study

Ablations are conducted on Qwen2.5-14B-Instruct (100 training steps), evaluating different optimization strategies and reward designs. Ablation dimensions cover three key design choices: whether to jointly optimize the unit test generator, whether to use SFT or RL, and reward function design.

| Configuration | UT Accuracy | Code Accuracy | BoN Accuracy | Notes |
|---|---|---|---|---|
| CURE (full) | **73.3** | **47.5** | **60.2** | Full framework, theoretically derived reward |
| Coder only | 27.8 | 43.2 | 54.8 | No unit test optimization; UT unchanged |
| SFT instead of RL | 65.1 | 45.8 | 57.3 | Supervised fine-tuning instead of RL |
| Simple reward ($p_u$ estimate) | 68.5 | 46.1 | 56.9 | Only passing correct code used as reward |

With the simple reward, $p_{01}$ and $p_{00}$ reach 42.2% and 14.7% respectively — substantially higher than CURE's 36.5% and 9.1% — confirming that the simple reward indeed leads to uncontrolled error rates.

### Key Findings

- **The dramatic improvement in unit test accuracy is CURE's most prominent achievement**: ReasonFlux-14B improves unit test accuracy on LiveBench from 27.8% to 73.3% (+45.5%), and on CodeForces from 20.7% to 82.3% (+61.6%). These order-of-magnitude gains far exceed improvements in code generation itself, revealing unit test generation as a vastly underexplored capability space.
- **Robust BoN improvements demonstrate the practical value of unit test quality**: Average BoN accuracy improves by 9.0%, which is of greater practical value than the improvement in one-shot code accuracy (5.3%), since BoN is the most commonly used strategy in actual deployment.
- **Optimizing the coder alone does not improve unit test quality**: The ablation confirms the necessity of co-evolution — training the coder alone leaves UT accuracy unchanged, demonstrating that code and test capabilities are related but do not automatically transfer to each other.
- **RL outperforms SFT**: SFT exploits only positive samples while discarding information in negative ones; RL learns stronger discriminative capabilities by contrasting rewards between positive and negative samples.
- **Viable as a reward model for annotation-free RL**: Using unit tests generated by ReasonFlux-4B as substitutes for ground-truth unit tests to conduct RL training of Qwen2.5-14B yields performance improvements comparable to RL with real labels, enabling a truly annotation-free self-improvement loop.
- **Significant efficiency gains for long-CoT models**: Response length is reduced to 64.8% while accuracy actually improves, demonstrating that long-CoT models exhibit substantial redundant reasoning on unit test generation tasks, and that the length-guided reward successfully prunes this redundancy.
- **Cross-model enhancement capability**: Using ReasonFlux-4B as the unit test generator paired with GPT-4o-mini as the coder yields an average BoN improvement of 5.5%; paired with GPT-4.1-mini, a 1.8% improvement is achieved while substantially reducing API costs (since expensive large models are no longer needed for unit test generation). Specifically, the GPT-4o-mini + ReasonFlux-4B combination outperforms GPT-4o one-shot by 7.0% at lower cost.
- **Effective across diverse agentic coding pipelines**: Beyond simple BoN, ReasonFlux-14B achieves an average improvement of 8.1% over the base model across three more complex pipelines — MPSC (multi-perspective self-consistency), AlphaCodium (iterative test-and-fix), and S* (debugging + pairwise discrimination). On the Agentic Unit Test Generation task (iteratively refining unit tests based on execution results), the improvement reaches 25.1%, indicating that training-time gains are amplified in iterative settings.

## Highlights & Insights

- **The theoretically grounded reward design is elegantly constructed**: Unlike most RL-for-code work that relies on heuristic rewards, CURE derives the optimization objective $\mu$ from a probabilistic analysis of reward precision, then derives the concrete individual reward formula from $\mu$. This complete theoretical chain ensures mathematical soundness while providing quantitative convergence guarantees (via the Hoeffding bound) — a rarity in the RL+code literature.
- **The insight that "incorrect code is a valuable resource" is deeply penetrating**: Conventional thinking holds that training unit tests requires correct code as supervision; CURE inverts this by leveraging incorrect code — which exposes typical failure patterns — to train the unit test generator to recognize those patterns. This "learning from failure" philosophy is transferable across domains: for instance, using incorrect translations to train translation quality estimators, or using failed plans to train plan verifiers.
- **The length-guided reward transformation is a general efficiency optimization trick**: The key idea is to introduce a "shorter is better" preference for positively rewarded samples while applying maximum penalty uniformly to negatively rewarded ones. This design preserves reward polarity (good and bad samples are not reversed) while introducing an optimization direction toward length efficiency. This trick is directly applicable to any scenario in which one wishes to reduce the generation length of long-CoT models.
- **The elegance of using a shared model in two roles**: Rather than training two independent models, a single LLM learns to both write code and write tests, achieving knowledge transfer through shared representations. This embodies the reasonable assumption that "programming comprehension is a unified capability" — understanding the logical structure of a problem helps both in producing a correct solution and in devising effective test cases, such that a model skilled at reasoning about edge cases becomes stronger in both directions simultaneously.
- **The engineering philosophy behind the experimental design is worth emulating**: The authors reduce format error rates in the base model to negligible levels through careful prompt design (0.08% for code, 9% for unit tests), then demonstrate that CURE's improvements substantially exceed any possible confounding from format errors. This practice of "first eliminating confounding factors, then demonstrating the core effect" is frequently overlooked in the literature, making it difficult for readers to disentangle genuine methodological contributions from engineering details.

## Limitations & Future Work

- **Implicit dependence on ground-truth unit tests**: Although CURE claims to require "no ground-truth code," the framework still requires each training task to be accompanied by a small number of ground-truth unit tests to judge code correctness (via $\mathcal{I}_{s_j}$). If these ground-truth tests are themselves erroneous or have insufficient coverage, the entire reward chain is affected. Whether a fully annotation-free variant is feasible — for instance, replacing ground-truth judgments with cross-code consistency — warrants exploration.
- **Minimum capability threshold for the base model**: The success of co-evolution depends on the base model already possessing a certain level of code generation and test generation capability. If the base model is too weak (e.g., 1B parameters), the quality of both generated code and tests may be too low for the positive feedback loop to ignite. The paper does not investigate the lower capability bound.
- **Non-trivial inference cost**: Each task requires generating 16 code solutions and 16 unit tests followed by 256 cross-executions, imposing non-negligible computational demands during deployment. While cheaper than using a large model as a reward model, this remains a bottleneck for large-scale applications.
- **Limited task type coverage in evaluation**: All experiments are conducted on competition/algorithmic programming-style benchmarks (stdio format). Generalization to real-world software development scenarios (e.g., web development, data processing, systems programming) remains unknown. In these settings, the form and correctness criteria for "unit tests" differ substantially from algorithmic problems. In fact, the paper converts function-style I/O from MBPP and portions of LiveBench to stdio format for uniformity, further highlighting the framework's strong task-format assumptions.
- **Training on only 4.5K problems**: While high data efficiency is an advantage, the paper does not explore whether increasing the training dataset size would yield further improvements. Given that RL training requires no annotated solutions, expanding the problem set should be relatively straightforward. The paper also does not analyze how the difficulty distribution of training problems affects final performance — only problems of difficulty ≤2 are used, and whether incorporating harder problems would be beneficial remains an open question.
- **Incomplete ablation coverage**: The paper does not ablate the KL coefficient $\beta$, does not provide training curves at different step counts (350 vs. more/fewer), and does not explore the optimal ratio between $n$ (number of code candidates) and $m$ (number of test candidates). Furthermore, the 7B and 14B models use Instruct variants rather than Base models as starting points, without justification — since Instruct models have already undergone SFT and RLHF, applying RL on top may exhibit different dynamics.

## Related Work & Insights

- **vs. O1-Coder**: O1-Coder uses ground-truth code to generate unit tests and then applies SFT, heavily relying on annotated data. CURE requires no ground-truth code and achieves unsupervised training through self-play RL, substantially surpassing O1-Coder in flexibility and scalability. The relative disadvantage of CURE is that it requires a small number of ground-truth unit tests, whereas O1-Coder only requires code.
- **vs. UTGEN**: UTGEN mixes correct unit tests derived from ground-truth code with incorrect unit tests derived from perturbed code, essentially a form of data augmentation. CURE's reward design is theoretically more rigorous, does not depend on manually constructed data, and the reward signal updates dynamically with training.
- **vs. CodeT / AlphaCodium / S\***: These methods primarily leverage generated unit tests at inference time for filtering or debugging, without training the unit test generator itself. CURE optimizes the unit test generator at training time; the resulting higher-quality tests can be directly plugged into these pipelines at inference time for additional gains (S* pipeline achieves +8.1% in experiments).
- **vs. DeepSeekMath/GRPO**: GRPO is the underlying RL algorithm used by CURE; CURE's innovations lie in the reward design and the dual-role co-evolution framework. The two contributions are orthogonal — any improved RL algorithm can directly replace GRPO within CURE. The paper explicitly states that it does not aim to compete with RL algorithms per se, but rather to provide a framework compatible with any RL algorithm.
- **vs. traditional unit test generation methods (EvoSuite/Randoop)**: Traditional methods are grounded in software analysis (search, symbolic execution, random testing) and have advantages in structured testing, but cannot handle tasks described in natural language. LLM-based unit test generation bridges this gap, and CURE further provides an unsupervised method for training such LLM generators.
- **Insights**: The self-play paradigm of "two modules as mutual teachers" is transferable to other tasks requiring generation-verification pairs: automated mathematical theorem proving (proof generator vs. proof verifier), code translation (translator vs. equivalence checker), and safety alignment (generator vs. safety classifier). Particularly noteworthy is the universality of "using failure samples to train the verifier" — in any generative task, failed samples produced during generation may constitute valuable training data for verification/filtering modules, echoing the contrastive learning philosophy that "good negative samples matter more than positive ones."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The self-play co-evolution paradigm is relatively novel in the code domain, and the theoretically derived reward design offers a distinctive contribution; however, self-play and RL-for-code are individually not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five benchmarks, three model scales, diverse downstream applications (BoN/MPSC/AlphaCodium/S\*), cross-model validation (GPT series), and reward model experiments are included; ablations could be more granular.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly structured; theoretical derivations are rigorous with well-articulated intuitions; experiments are well organized; the narrative from motivation through method to application is cohesive.
- **Value**: ⭐⭐⭐⭐ The framework has strong practical utility (model weights released as open source); annotation-free training has long-term scalability value; actual impact depends on generalization to real software engineering scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](../../ICLR2026/code_intelligence/supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[NeurIPS 2025\] A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)

</div>

<!-- RELATED:END -->
