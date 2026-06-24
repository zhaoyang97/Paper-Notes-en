---
title: >-
  [Paper Note] Tokenisation is NP-Complete
description: >-
  Proves that two variants of the tokenisation problem—direct tokenisation and bottom-up tokenisation—are both NP-complete. This is achieved via a polynomial-time reduction from the max-2-SAT problem, implying that finding an efficient optimal tokenisation algorithm is impossible and thus justifying the use of approximation methods like BPE.
tags:

date: 2026-05-08
content_hash: eb0cc36609e6d5fc
---

# Tokenisation is NP-Complete

- **Conference**: ACL 2025
- **arXiv**: [2412.15210](https://arxiv.org/abs/2412.15210)
- **Code**: Not provided
- **Area**: Other
- **Keywords**: Tokenisation, NP-Completeness, BPE, Compression, Max-2-SAT Reduction

## TL;DR

Proves that two variants of the tokenisation problem—direct tokenisation and bottom-up tokenisation—are both NP-complete. This is achieved via a polynomial-time reduction from the max-2-SAT problem, implying that finding an efficient optimal tokenisation algorithm is impossible and thus justifying the use of approximation methods like BPE.

## Background & Motivation

- **Limitations of Prior Work**: Tokenisation is the very first and most fundamental step in NLP. While BPE and UnigramLM are the most widely used tokenisation algorithms, they are both greedy/heuristic approaches that do not guarantee finding the optimal vocabulary (i.e., one that maximises compression). Researchers have long remained uncertain whether efficient exact algorithms exist to replace these approximations. Suboptimal tokenisation choices propagate to downstream tasks, causing performance degradation (e.g., GPT-4 failing to correctly count the number of "r"s in "strawberry").

- **Key Challenge**: Although the optimization objective of tokenisation (maximizing compression) is clear, the computational complexity of finding the optimal vocabulary under a given vocabulary size constraint $K$ has remained undetermined. If it belongs to P, researchers should develop exact algorithms to replace BPE; if it is NP-hard, approximation algorithms are inevitable. This fundamental theoretical question directly dictates the research direction of tokenisation algorithms.

- **Goal**: (1) Prove that the direct tokenisation problem (finding the optimal vocabulary to maximize compression) is NP-complete; (2) Prove that the bottom-up tokenisation problem (finding the optimal sequence of merge operations) is also NP-complete; (3) Discuss the practical implications of these theoretical results.

- **Key Insight**: The authors formalize the tokenisation problem as a decision problem—"whether there exists a vocabulary/merge sequence that compresses the dataset to no more than $\delta$ symbols"—and construct a polynomial-time reduction from the well-known NP-hard problem max-2-SAT. Key Insight: The True/False assignment of variables in max-2-SAT can be mapped to the choice of "which subword to include in the vocabulary".

## Method

### Overall Architecture

This paper is a purely theoretical work. The core consists of two NP-completeness proofs, each divided into two parts: (1) proving that the problem is in NP (a given solution can be verified in polynomial time); (2) proving that the problem is NP-hard (reduction from max-2-SAT).

### Key Designs

1. **Formal Definition of Tokenisation**:

    - **Function**: Rigorously defines two variants of tokenisation.
    - **Mechanism**: The tokeniser is defined as a triplet $\langle \mathcal{S}, \text{tok}, \text{detok}\rangle$, where $\mathcal{S}$ is the vocabulary (containing all original characters plus $K$ additional subwords). **Direct tokenisation** finds the shortest subword sequence given a vocabulary (solvable in $O(|c|^2)$ via dynamic programming); **bottom-up tokenisation** sequentially merges adjacent subwords through a series of merge operations (which is how BPE operates). Both optimize for maximizing the compression $\mathfrak{G}(s)=-|s|$.
    - **Design Motivation**: Distinguishing these two variants is crucial because they correspond to different practical tokenisation algorithms (UnigramLM vs. BPE) and technically require distinct reduction constructions.

2. **NP-Completeness Proof of Direct Tokenisation (Reduction 1)**:

    - **Function**: Proves that searching for the optimal vocabulary is NP-hard.
    - **Mechanism**: Given an instance of max-2-SAT ($J$ variables, $I$ clauses, and threshold $\psi$), the proof constructs an alphabet $\Sigma = \{\circledcirc\} \cup \{x_j^T, x_j^F\}_{j=1}^J$ and three sets of strings $\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3$. The extremely high frequency $f$ of $\mathcal{D}_1$ forces the optimal vocabulary to select subwords of the form $\circledcirc x_j^T \circledcirc$ or $\circledcirc x_j^F \circledcirc$; $\mathcal{D}_2$ further forces selecting only one of them for each variable (corresponding to True/False assignments); $\mathcal{D}_3$ encodes the SAT clauses such that assignments satisfying more clauses yield better compression. The vocabulary size is set to $K=J$, and the target compression size is $\delta = (4f+3f')J + 5I - 2\psi$.
    - **Design Motivation**: The key technique in the reduction is using the frequency of data repetition (via the magnitude relationship $f \gg f' \gg 1$) to establish a "priority hierarchy", ensuring that the optimal solution must select subwords in a specific manner.

3. **NP-Completeness Proof of Bottom-Up Tokenisation (Reduction 2)**:

    - **Function**: Proves that finding the optimal merge sequence is NP-hard.
    - **Mechanism**: It shares a similar idea but is more complex, requiring the introduction of an additional character $\otimes$ and five sets of strings $\mathcal{D}_1$-$\mathcal{D}_5$. Due to additional constraints imposed by the ordered application of merge operations and their left-to-right processing, a more delicate construction is needed to ensure that merge choices align with SAT assignments. Here, $K=8J$ (requiring more merge operations).
    - **Design Motivation**: Bottom-up tokenisation is much closer to practically deployed BPE, making its NP-completeness proof theoretically more significant.

### NP Membership Proofs

- **Direct Tokenisation**: Given a vocabulary $\mathcal{S}$ as a certificate, verification is completed in polynomial time $O(|c|^2)$ using the PathPiece method.
- **Bottom-Up Tokenisation**: Given a merge sequence as a certificate, the merges are applied step-by-step with each step costing $O(|c|)$, tracking an overall polynomial time complexity of $O(|\mathcal{D}||c||m|)$.

## Key Experimental Results

### Summary of Theoretical Results

| Tokenisation Variant | NP-Completeness | Source of Reduction | Vocabulary Size Parameter | Key Construction |
|---------|----------|---------|-------------|---------|
| Direct Tokenisation | **Yes** | max-2-SAT | $K=J$ | 3 string groups + 2 frequency parameters |
| Bottom-Up Tokenisation | **Yes** | max-2-SAT | $K=8J$ | 5 string groups + 4 frequency parameters |
| Direct Tokenisation (non-repeating dataset) | **Yes** | §6.3 variant | — | Alphabet expansion |
| Bottom-Up Tokenisation (non-repeating dataset) | **Yes** | §6.3 variant | — | Alphabet expansion |

### Complexity Comparison of Practical Tokenisation Algorithms

| Algorithm | Is Optimal | Running Time | Description |
|------|---------|---------|------|
| BPE (Greedy) | No | $O(N \cdot K)$ | Bottom-up, selects the most frequent pair at each step |
| UnigramLM | No | Iterative optimization | Direct tokenisation, EM approximation |
| PathPiece | Optimal given vocabulary | $O(\|c\|^2)$ | Application phase of direct tokenisation |
| Optimal Vocabulary Search | Yes | **NP-hard** | Proven in this paper |

### Key Findings

- **Theoretical necessity of greedy algorithms like BPE is confirmed**: No polynomial-time exact optimal tokenisation algorithm exists (unless P=NP), verifying BPE and UnigramLM as reasonable research avenues for approximation.
- **Consistency with concurrent work**: Kozma & Voderholzer (2024) independently proved the APX-hardness of bottom-up tokenisation (a stronger result). The convergence of these two independent works bolsters the credibility of the findings.
- **Practical implications**: For applications such as tokenizer compression and multilingual fairness, future work should focus on improving the quality of approximation algorithms rather than pursuing exact solutions.

## Highlights & Insights

- **Fundamental theoretical contribution**: Achieves a definitive answer regarding computational complexity at the very foundation of NLP (tokenisation), defining theoretical boundaries for tokenisation research. Such explanations justifying "why approximate solutions suffice" are exceptionally valuable in empirical/experiment-driven NLP.
- **Clever reduction construction**: Creating a "priority hierarchy" through the magnitude difference of data repetitions ($f \gg f' \gg 1$) is a smart trick to strictly constrain the structure of the optimal solution. This approach can be adapted for other reductions in similar settings.
- **Unified treatment of both variants**: Simultaneously proves the NP-completeness of direct and bottom-up tokenisation, covering both dominant paradigms (UnigramLM and BPE).

## Limitations & Future Work

- Only considers compression as the optimization objective, leaving the complexity of other objectives (such as unigram log-probability or Rényi efficiency) unexplored.
- The proof relies on $K$ being part of the input; if $K$ is fixed to a constant (e.g., 32K), the problem might become tractable.
- Does not discuss the approximation ratio—while exact solutions are intractable, the approximation quality of BPE remains an open question.
- A purely theoretical work lacking empirical verification (e.g., comparing BPE against the exact optimal solution on small-scale instances).

## Related Work & Insights

- **vs BPE (Sennrich et al., 2016)**: BPE is a greedy implementation of bottom-up tokenisation. This paper proves that the problem being approximated is itself NP-hard.
- **vs UnigramLM (Kudo, 2018)**: UnigramLM corresponds to direct tokenisation and uses EM approximation. This paper proves that exact solving is NP-hard.
- **vs Kozma & Voderholzer (2024)**: Concurrent work proving the APX-hardness of bottom-up tokenisation (a stronger result); the two independent works validate each other.
- **vs Geh et al. (2024)**: Previous work proved the NP-completeness of LM-based tokenisation functions. This paper extends studies of NP-completeness to compression-based formulations.

## Rating

- **Novelty**: 9/10 — First to prove the NP-completeness of tokenisation optimization, filling a crucial theoretical gap.
- **Technical Depth**: 9/10 — Delicate reduction construction, rigorous proofs, and coverage of multiple variants.
- **Experimental Thoroughness**: 4/10 — Purely theoretical work, lacks experimental validation.
- **Clarity**: 7/10 — Contains heavy mathematical notation but is logically sound, posing a minor barrier for non-theoretical readers.
- **Overall Score**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Causal Estimation of Tokenisation Bias](causal_tokenisation_bias.md)
- [\[ICCV 2025\] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](../../ICCV2025/others/jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)
- [\[AAAI 2026\] The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](../../AAAI2026/others/the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)
- [\[ACL 2025\] Infogen: Generating Complex Statistical Infographics from Documents](infogen_generating_complex_statistical_infographics_from_documents.md)
- [\[ACL 2025\] What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation](what_matters_in_evaluating_book-length_stories_a_systematic_study_of_long_story_.md)

</div>

<!-- RELATED:END -->
