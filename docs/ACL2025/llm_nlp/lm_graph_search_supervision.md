---
title: >-
  [Paper Note] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More
description: >-
  [ACL 2025][LLM (Other)][graph search] This paper demonstrates that the failure of decoder-only LMs on the path-star graph search task is not a fundamental limitation of the next-token prediction paradigm, but is caused by "supervision adulteration"—where excessive teacher-forcing supervision signals induce the model to learn a Clever Hans Cheat shortcut, preventing subtask decomposition. The task is shown to be learnable through six orthogonal methods, including token masking…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "graph search"
  - "supervision adulteration"
  - "next-token prediction"
  - "subtask decomposition"
  - "shortcut learning"
  - "Clever Hans Cheat"
  - "planning capabilities"
date: 2026-05-08
content_hash: ffe90544aa460fdc
---

# Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More

**Conference**: ACL 2025  
**arXiv**: [2503.10542](https://arxiv.org/abs/2503.10542)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: graph search, supervision adulteration, next-token prediction, subtask decomposition, shortcut learning, Clever Hans Cheat, planning capabilities  

## TL;DR

This paper demonstrates that the failure of decoder-only LMs on the path-star graph search task is not a fundamental limitation of the next-token prediction paradigm, but is caused by "supervision adulteration"—where excessive teacher-forcing supervision signals induce the model to learn a Clever Hans Cheat shortcut, preventing subtask decomposition. The task is shown to be learnable through six orthogonal methods, including token masking, ranking-into-the-future, scratchpad, and tree topologies.

## Background & Motivation

**Path-star Task**: A minimalist graph search task proposed by Bachmann & Nagarajan (2024)—a star graph has $D$ arms, each of length $M$. Given a starting node $s$ and a target node $t$, the model must generate the correct arm sequence from $s$ to $t$. The core challenge lies in selecting the correct leading node $l_t$.

**Surprising Failure**: Standard decoder-only LMs trained with teacher-forcing fail to exceed the random baseline of $1/D$ on this task. This failure has been used to argue that the next-token prediction paradigm has fundamental deficiencies for planning tasks.

**Subsequent Works**: This failure has prompted several works to propose alternative architectures (e.g., auxiliary autoencoders by Yin et al. 2024, bidirectional encoders by Hu et al. 2025a). However, these approaches modify the model architecture itself.

**Clever Hans Cheat (CHC)**: During teacher-forcing training, the model exploits the previous ground-truth token to perform a single-edge lookup, rather than actually learning to reconstruct the arm path backward from $t$. CHC absorbs almost all sequential supervision signals except for $l_t$, leaving the learning of the core planning subtask dependent on a single token.

**Core Insight—Supervision Adulteration**: The authors propose the concept of supervision adulteration: an adverse interaction between excessive or inappropriate supervision signals, which dilutes the learning signal of the target task with irrelevant shortcut learning. This is not a data overfitting issue, but an issue with how the task itself is constructed.

**Motivation for Refutation**: If standard methods (decoder-only + teacher-forcing + next-token prediction) can solve this task with minor modifications, it proves that the original claims were overly absolute and that the failure does not invalidate the paradigm itself.

## Method

### Overall Architecture

The authors formulate the issue as follows: the unlearnability of the path-star task stems from supervision adulteration preventing subtask decomposition. Graph search/arm reconstruction is recursively defined and inherently contains a decomposed structure, but the excessive supervision of teacher-forcing prompts the model to take shortcuts instead of learning this decomposition. Therefore, as long as a training method that induces subtask decomposition is designed, the task becomes learnable. The authors propose six orthogonal methods to validate this theory.

Experimental setup: decoder-only Transformer, 2 heads, 64-dimensional embeddings, 256-dimensional FFN, 8 layers, learning rate $5\times10^{-4}$, batch size 1024, online dataset (generating new samples on-the-fly to prevent overfitting), $|V|=|G|$, trained on 100M samples.

### Module 1: Token Masking

**Motivation**: To break the bad interaction in teacher-forcing between the previous ground-truth token and the current prediction, starting from the target-side input.

**Approach**: During training, tokens in the target sequence are randomly masked or replaced (scheduled sampling), supporting both uniform sampling and continuous span sampling modes. Masked positions cannot trigger the CHC single-edge lookup, forcing the model to perform multi-edge lookups.

**Decomposition Mechanism**: When $l_t$ is masked, the model is forced to learn the subtask of backward reasoning from $t$ (Figure 5c), which is precisely a subset of the core planning task. Masking not only prevents CHC but also directly induces subtask decomposition.

### Module 2: Ranking-into-the-Future (RITF)

**Motivation**: Redesigning the supervision signal from the perspective of the loss function—enabling the model to predict the distribution of future tokens rather than just the next token at each step.

**Approach**: At each time step $i$, a ranking objective $x_i \succ x_{i+1} \succ \dots \succ x_M$ is constructed using a pairwise hinge loss:

$$L_B = \sum_{i=1}^{M}\sum_{j=i}^{M}\sum_{k=j+1}^{M}\max(0, 1-(\sigma_i[j]-\sigma_i[k]))$$

Simultaneously, a constraint is added to ensure that tokens within the correct arm are ranked above tokens in other arms. Compared with Bag-of-Words (BoW) and label smoothing, RITF achieves the best performance.

**Decomposition Mechanism**: The multi-token loss inherently requires multi-edge lookups. Training the future distribution at each step forms nested subproblems ($P_{B,i+1}$ is a subproblem of $P_{B,i}$), producing dense decomposition supervision across the sequence.

### Module 3: Graph Topology Modification (Tree-Star) and Generalized Queries

**Tree-Star**: The arms during training are changed from paths to tree structures (split tree), where each split creates a sub-path-star task with $D'=2$. Evaluating on paths while training on trees reveals that, counterintuitively, training and evaluating on different distributions is effective because paths are "over-informative" graph structures.

**Generalized Querying (GST)**: In the query, a single node from $R_t$ is randomly sampled to replace the fixed target $t$, forcing the model to face sub-paths of varying lengths during training, which directly introduces decomposition.

**Universal Length Decomposition**: Mixing graphs with different values of $M$ during training naturally provides subtasks of different granularities.

### Loss & Training

All methods retain the standard teacher-forcing + next-token prediction paradigm (except RITF, which uses the future distribution loss). An online dataset (no fixed dataset) is utilized to avoid overfitting. Each set of experiments is executed with 5 different random seeds, reporting the Success Rate (SR, success with >95% sequence accuracy) and Arm-to-Baseline Ratio (ABB, exceeding the baseline by $100/D+10$%).

## Key Experimental Results

### Table 1: Success Rate of Different Methods under Various (D, M)

| Method | D=2,M=5 | D=3,M=5 | D=4,M=5 | D=5,M=5 | D=2,M=7 | D=3,M=7 |
|------|---------|---------|---------|---------|---------|---------|
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Causal-wise shuffle | ✓ | ✓ | Partial | Partial | ✓ | Partial |
| Token masking | ✓ | ✓ | Partial | ✗ | ✓ | Partial |
| RITF | ✓ | ✓ | ✓ | Partial | ✓ | ✓ |
| Split tree | ✓ | ✓ | ✓ | Partial | ✓ | Partial |
| GST (Generalized Querying) | ✓ | ✓ | Partial | — | ✓ | Partial |

### Table 2: Comparison of Scratchpad Methods

| SP Type | SP Accuracy | R_t Accuracy | Description |
|---------|----------|-----------|------|
| Reverse arm order | ~100% | High | trivial, no analytical value |
| BoW (Sorting) | Medium | Low-Medium | Sorting itself is hard to learn, limited decomposition effect |
| Graph Reconstruction (GR-SP) | Partially learned | Extremely low (4/80) | Can identify node sets but fails to complete arm reconstruction |

### Key Findings

1. **All six orthogonal methods render the task learnable**, which demonstrates the fragility of the "unlearnable" conclusion—minor modifications can break the barrier.
2. **Preventing CHC is not a necessary condition**: Split tree and universal length decomposition do not prevent CHC, yet the task remains learnable. The key lies in whether subtask decomposition is induced.
3. **RITF outperforms BoW and label smoothing**: This demonstrates that specifying ranking rules is more effective than specifying concrete token weights.
4. **The negative results of GR-SP are highly informative**: The model can correctly identify and sort leading/target node sets but fails to complete arm reconstruction. This indicates that the core difficulty is graph reconstruction rather than planning selection.
5. **Impact of causal constraints**: Causal-wise shuffle makes the task learnable, indicating that the causal constraints of decoders introduce additional difficulty.
6. **Scalability issues**: The performance of all methods degrades as $D$ or $M$ increases. The authors hypothesize that a stronger/more consistent decomposition structure (where subtasks are isomorphic to the main task) is required to resolve scalability.

## Highlights & Insights

- **The concept of "supervision adulteration" is novel and profound**: It explicates the implicit problems of teacher-forcing, unifying why more supervision can be detrimental. It is not about the volume of supervision, but because interactions between supervisions generate shortcuts that absorb useful signals.
- **Unified explanation of six orthogonal methods**: All methods, from different angles (input, loss, data, and topology), induce subtask decomposition, strongly supporting the unified theory.
- **Negative results are equally valuable**: The failure of GR-SP reveals that arm reconstruction (rather than path selection) is the core difficulty. The failure of BoW SP reveals that reverse solutions are not automatically discovered by the model, challenging the intuition that "models can easily find obvious solutions."
- **Bridging prior work**: This research provides a unified explanation that harmonizes the seemingly contradictory findings of Bachmann & Nagarajan (2024) and Saparov et al. (2025).

## Limitations & Future Work

1. **Scalability remains unresolved**: All methods fail when $D$ or $M$ becomes significantly larger. The paper only demonstrates learnability on small-scale graphs without providing a scalable solution.
2. **Only small models trained from scratch** are used (2 heads, 64 dimensions, 8 layers). This formulation is not validated on pre-trained large language models, where semantic information introduced during pre-training might alter the conclusions.
3. **Why subtask decomposition is necessary** remains an open question—the paper demonstrates empirical necessity but lacks a mathematical proof.
4. **The path-star task itself has limited representativeness**: The authors acknowledge that the task is not ideal as a benchmark for evaluating planning capabilities, and graph search does not represent general search problems.
5. **Practicality of RITF is unverified**: RITF is only tested on synthetic tasks and has not been extended to natural language tasks or larger-scale scenarios.

## Related Work & Insights

- **Origin of the path-star task**: Bachmann & Nagarajan (2024) "The Pitfalls of Next-Token Prediction" first proposed this task and claimed that decoder-only LMs cannot learn it.
- **Alternative architectures**: Yin et al. (2024) proposed auxiliary autoencoder planning tokens; Hu et al. (2025a) utilized bidirectional encoders; Ahn et al. (2025) proposed minimal architectural changes.
- **Learnability of graph search**: Saparov et al. (2025) achieved positive results on encoder-only models and general topologies, but did not test the decoder + path-star setting. Frydenlund (2024) proved expressiveness sufficiency, but learning remains an open problem.
- **Shortcut learning**: Du et al. (2023) reviewed shortcut learning in LLMs, and Bhattamishra et al. (2023) investigated the simplicity bias of Transformers learning sparse Boolean functions.
- **Multi-token prediction**: Cai et al. (2024) proposed Medusa multi-head speculative decoding. Although relevant to the future distribution learning idea in RITF, their objectives differ.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of "supervision adulteration" is original, and the unified explanation framework of subtask decomposition offers theoretical value.
- **Technical Depth**: ⭐⭐⭐⭐ — The six orthogonal methods are carefully designed, and the experimental analysis is thorough (especially the in-depth discussion on negative results).
- **Value**: ⭐⭐⭐ — Core findings are restricted to synthetic tasks, and the transfer path to real-world NLP scenarios is not yet clear.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous logical flow, excellent illustrations, and accurate, intuitive concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[ACL 2025\] CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision](codetool_enhancing_programmatic_tool_invocation_of_llms_via_process_supervision.md)
- [\[ICML 2026\] Why Are Linear RNNs More Parallelizable?](../../ICML2026/llm_nlp/why_are_linear_rnns_more_parallelizable.md)

</div>

<!-- RELATED:END -->
