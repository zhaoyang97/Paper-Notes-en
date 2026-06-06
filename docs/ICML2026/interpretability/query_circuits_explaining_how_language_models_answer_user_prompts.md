---
title: >-
  [Paper Note] Query Circuits: Explaining How Language Models Answer User Prompts
description: >-
  [ICML 2026][Interpretability][Query circuits] This paper proposes the **query circuit discovery** task—directly tracing sparse sub-networks within the original LLM to explain "why the model produced a specific output for…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Query circuits"
  - "mechanistic interpretability"
  - "circuit discovery"
  - "BoN sampling"
  - "normalized deviation faithfulness"
date: 2026-05-08
content_hash: a69e55bc42f465c8
---

# Query Circuits: Explaining How Language Models Answer User Prompts

**Conference**: ICML 2026  
**arXiv**: [2509.24808](https://arxiv.org/abs/2509.24808)  
**Code**: https://tony10101105.github.io/query-circuit/ (Project Page)  
**Area**: Interpretability / Mechanistic Interpretability / Circuit Discovery  
**Keywords**: Query circuits, mechanistic interpretability, circuit discovery, BoN sampling, normalized deviation faithfulness

## TL;DR
This paper proposes the **query circuit discovery** task—directly tracing sparse sub-networks within the original LLM to explain "why the model produced a specific output for a given input." It introduces the robust faithfulness metric NDF and the Best-of-N sampling algorithm, enabling circuits comprising only 1.3% of model edges to recover approximately 60% of single-item behavior on MMLU.

## Background & Motivation

**Background**: The mainstream path in mechanistic interpretability is **circuit discovery**—representing a Transformer as a directed graph of nodes (MLP / attention heads) and edges (residual rewrites) to identify sparse subgraphs that implement a specific capability. Representative works include ACDC, EAP, and EAP-IG, which primarily study capability circuits ($C_c$) for "toy" tasks like IOI and greater-than.

**Limitations of Prior Work**: Capability circuits only explain "how the model implements a class of algorithmic skills" (global explanation) and cannot answer "why the model gave this specific answer to this particular user input" (local explanation). Existing instance-oriented solutions like Circuit Tracing must rely on **surrogate models** such as SAEs or cross-layer transcoders (CLTs). However, surrogate reconstructions of LLM activations are often unfaithful and expensive to train, and circuits discovered on surrogates are defined on CLTs rather than the original model, potentially failing to correspond to the actual computing mechanism.

**Key Challenge**: There is a tension between faithfulness (in-place explanation) and analyzability (sparsity/readability). When directly searching for single-query circuits within an LLM, capability circuit scoring formulas suffer from (1) gradient noise and (2) ignoring combinatorial effects between edges. Furthermore, the commonly used metric NFS exhibits severe drift (extending $>1$ or $<0$) on general datasets like MMLU, making it impossible to monitor discovery progress.

**Goal**: (i) Formalize the in-place, instance-level query circuit discovery task; (ii) Design a circuit faithfulness evaluation metric that is stable on general data; (iii) Design discovery algorithms that find *sparse and faithful* circuits for a single query.

**Key Insight**: The authors observed a counter-intuitive phenomenon in IOI: EAP-IG fails to find faithful circuits on the original query $q$, but the circuits found on its **paraphrases** highly recover the behavior of $q$. This reinterprets circuit discovery as a **"lottery ticket"** problem: the original query and its paraphrased edge-scoring matrices $\{S, S_1,\dots,S_p\}$ act as perturbations of each other, where one of them is the "winning ticket."

**Core Idea**: Use **Sampling + Best-of-N (BoN)** to pick the most faithful query circuit from a set of paraphrases, and replace NFS with **NDF**—a symmetric, bounded metric insensitive to $L(M(q'))\neq 0$—for evaluation.

## Method

### Overall Architecture

The input consists of a target LLM $M$, a natural language query $q$, and an edge budget $N$. Process: (1) Use GPT-4o (or random IOI samples / operand permutations) to generate $p$ paraphrases of $q$: $\{q_1,\dots,q_p\}$; (2) Calculate edge importance score matrices $\{S, S_1,\dots,S_p\}$ for each query in $\{q, q_1,\dots,q_p\}$ using EAP-IG (discretization step $m=20$); (3) Greedily select $N$ edges using each scoring matrix to form candidate circuits, perform a single forward pass on $M$ to evaluate NDF for each, and select the one with the highest NDF as the query circuit $C_q$ for $q$. Subsequent versions like iBoN / BoN-CSM interpolate or re-rank existing $k$ circuits to construct new budgets $N$ without further LLM runs.

### Key Designs

1.  **Query Circuit Discovery**:
    - **Function**: For any natural query $q$ and edge budget $N$, find a sparse subset of edges $E_q \subset E$ in the original LLM $M$ such that the model's behavior on the *single query* is maximally recovered when only $E_q$ is retained.
    - **Mechanism**: The key difference from capability circuits (which select edges after averaging IE over a dataset $D$) is that **cross-sample averaging is no longer performed**. Edge importance $a_e$ is estimated on a single query via the IE formula $a_e = L(M(q\mid \mathrm{do}(e\leftarrow e'))) - L(M(q))$, where $q'$ is a corrupted query with key facts removed. Crucially, the circuit is *defined on the original LLM*; SAEs are only used post-hoc for labeling nodes and **do not participate in circuit construction**, thus removing dependency on surrogate faithfulness.
    - **Design Motivation**: Bridging the dichotomy of "capability circuit = global explanation, instance explanation = performed on surrogates"; providing auditable circuit-level answers for "why the model gave *this* answer to *this* input" in high-risk scenarios like healthcare or autonomous driving.

2.  **Normalized Deviation Faithfulness (NDF)**:
    - **Function**: Replaces NFS to measure the faithfulness of $C_q$ on $q$, requiring symmetry, boundedness, and monotonic monitoring of circuit scale.
    - **Mechanism**: Defined as $\mathrm{NDF}(C_q) = 1 - \min\!\big(\big|\frac{L(M(q)) - L(C_q(q))}{L(M(q)) - L(M(q'))}\big|, 1\big)$, where the circuit output deviation relative to $M(q)$ is normalized by the performance difference of $M$ between original and corrupted queries. Two key modifications: (i) Symmetry around $L(M(q))$—circuit outputs exceeding or falling short of $M(q)$ are penalized equally, avoiding the inflated $>1$ values of NFS when $C_q(q) > M(q)$; (ii) Clipping to $[0,1]$—avoiding explosive values of NFS when the performance difference is small ($L(M(q))\approx L(M(q'))$) or $L(M(q'))\neq 0$ (e.g., position bias in MCQ). NDF is derived from the integrated circuit-model distance (CMD) in the MIB benchmark.
    - **Design Motivation**: For three MMLU Marketing samples in Table 1, NFS gives uninterpretable values like $2.15 / 1.32 / -1.57$; NDF gives $0.00 / 0.68 / 0.00$, directly indicating which circuit actually restored model behavior and enabling monotonic visualizations.

3.  **Best-of-N Sampling and Accelerated Variants (BoN / iBoN / BoN-CSM)**:
    - **Function**: Stably forces "coarse scoring" methods like EAP-IG to produce faithful and sparse query circuits on a single query; supports Pareto curve scanning across different $N$ with zero additional forward overhead.
    - **Mechanism**: BoN treats $q$ and $p$ paraphrases as a set of perturbations for score matrix $S$, constructing $p+1$ candidate circuits and retaining the one with highest NDF as $C_q$ ($p=9$ in experiments). **iBoN** uses $k$ existing BoN circuits $\{E_1,\dots,E_k\}$ to construct a new budget $N$ by taking the nearest smaller circuit $E_i$ and filling remaining slots with high-scoring edges from a larger circuit $E_j$. **BoN-CSM** maintains a score/level matrix pair $(S, T)$ to track the score and index of the first circuit in which an edge appeared, selecting the top $N$ edges sorted by $T$ (prioritizing smaller circuits) then $S$. All three are built on the "lottery ticket" observation from Section 5.1.
    - **Design Motivation**: (1) Running EAP-IG directly on $q$ requires $\sim 100\text{k}$ edges (25.9%) to exceed a random baseline on MMLU Astronomy (Fig 2c), and increasing IG steps $m$ doesn't help because the bottleneck is combinatorial effects, not single-edge IE precision; (2) BoN reduces the edges needed to reach NDF$=0.6$ from $\sim 200\text{k}$ (51.7%) to $\sim 5\text{k}$ (1.3%) across tasks.

### Loss & Training

No training involved. All edge scores use integrated gradient approximation from EAP-IG ($m=20$): $a_e \approx (e - e')^\top \tfrac{1}{m}\sum_{k=1}^m \nabla_e M(z' + \tfrac{k}{m}(z-z'))$. Circuits are constructed via greedy selection of top $N$ edges. The evaluation metric is NDF.

## Key Experimental Results

### Main Results

Target LLMs: GPT-2 Small (32,491 edges) for IOI, Llama-3.2-1B-Instruct (386,713 edges) for others. Baselines: (i) Single-query EAP-IG, (ii) Averaging $a_e$ across original query + paraphrases. Metric: Dataset-averaged NDF.

| Dataset | Edge Budget / % | Single Query (EAP-IG) | BoN (Ours) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| MMLU (Avg. 9 categories) | 5k / 1.3% | ≪ 0.6 | ≈ 0.6 | Single-query needs ~200k (51.7%) edges for 0.6 |
| IOI | 1k / 3.1% | < 0.5 (per query) | Significantly higher | Capability circuit ≈ 0.65 at same budget |
| Arithmetic Add/Mul, ARC | Multiple | Consistently < BoN | Order of magnitude better efficiency | iBoN / BoN-CSM perform between BoN and baseline |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| BoN, $p=9$ | Highest NDF | Default configuration |
| BoN, $p$ from 1 to 9 | Monotonic NDF increase | Fig 7: More paraphrases improve performance with diminishing returns |
| Averaging baseline | Worse than Single Query | Averaging dilutes edges critical only to the original query |
| iBoN | Slightly below BoN | Interpolates between existing circuits; no extra forward pass |
| BoN-CSM | Slightly below BoN | Ranks by "small circuit priority + high score priority" |
| Increasing IG steps $m$ | No significant gain | Confirms bottleneck is combinatorial effects, not IE precision |
| Gender Bias SAE Ablation | Logit Deviation Reduction | Circuits with high NDF significantly reduce bias when associated SAE features are ablated ($p<0.0001, r=0.787$) |

### Key Findings

- **Sparsity and Faithfulness Coexist**: On MMLU, just 1.3% of edges can recover ~60% of single-item behavior, extending "input-dependent activation sparsity" to *circuit sparsity*.
- **Existence of Shared Sub-circuits**: BoN candidate circuits for different IOI queries share 66 edges with the capability circuit at $N=500$ (Fig 8), while single-query discovery misses 23 critical edges, refuting the "lucky guess" hypothesis for BoN.
- **Paraphrases as Winning Tickets**: When EAP-IG fails on the original query, paraphrases almost always yield faithful circuits. $S_i$ shares coarse patterns with $S$ but differs greatly in specific edge rankings.
- **High NDF Circuits are Actionable**: Ablating gender-related SAE features in the highest NDF circuits reduces bias metrics significantly more than in the worst circuits ($r \in [0.737, 0.836]$).

## Highlights & Insights

- **Circuit Discovery as a Lottery Ticket**: Reframing faithful circuit discovery from "improving score accuracy" to "sampling for the winning ticket among coarse scores" avoids the dead-end of tuning IG steps or attribution formulas.
- **NDF for General Monitoring**: Previous NFS instability made Pareto curve scanning nearly impossible. Symmetric and bounded modifications allow circuit evaluation to function as reliable infrastructure.
- **Decoupling In-place from SAEs**: Circuits are defined on original LLM edges; SAEs are optional post-hoc labels. This allows leveraging surrogate readability without being contaminated by surrogate unfaithfulness or training failures.
- **Transferable Techniques**: The BoN + paraphrase paradigm can extend beyond attribution patching to any attribution noise scenario. iBoN / BoN-CSM effectively *cache* circuit discovery, which is ideal for online monitoring.

## Limitations & Future Work

- **Scale of LLMs**: Only tested on GPT-2 Small (124M) and Llama-3.2-1B-Instruct. Whether the BoN overhead and edge traversal costs remain feasible at 7B+ scales needs further experimentation.
- **Dependency on Paraphrase Quality**: GPT-4o paraphrasing for MMLU/ARC might cause semantic drift for short or logic-sensitive queries.
- **Indirect Interpretability Evaluation**: 60% NDF measures logit-level faithfulness, but a gap remains between this and human-expected reasoning processes.
- **Manual $q'$ Construction**: Corruption rules ($q'$) differ by task type and lack a unified principle, affecting comparability.
- **Future Directions**: (i) Automated $q'$ construction; (ii) Combining BoN with surrogate-based Circuit Tracing; (iii) Exploring time-evolving circuits in multi-step reasoning (RAG/agent loops).

## Rating
- Novelty: ⭐⭐⭐⭐☆ Brings instance-level circuits back to the original LLM and bypasses attribution noise via BoN + paraphrase.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers IOI, Arithmetic, MMLU, ARC, and SAE bias ablation; limited to 1B scale and $p=9$ paraphrases.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear concept definitions and logical derivation (Lottery ticket metaphor). Figures 1, 5, and 8 directly support the core arguments.
- Value: ⭐⭐⭐⭐☆ NDF is a reusable infrastructure for circuit evaluation; BoN/paraphrase is a practical tool for the interpretability community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ICML 2026\] How Language Models Process Negation](how_language_models_process_negation.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)
- [\[ICML 2026\] How Language Models Process Negation](how_language_models_process_negation.md)
- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](../../ACL2025/interpretability/reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)

</div>

<!-- RELATED:END -->
