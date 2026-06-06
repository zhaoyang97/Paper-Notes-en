---
title: >-
  [Paper Note] Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning
description: >-
  [ICML 2026][LLM Pretraining][Symbolic Invariance] The authors modify the Transformer into a structure featuring "parallel embedding streams with shared weights for each interchangeable symbol + cross-stream aggregated at…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Symbolic Invariance"
  - "alpha-equivalence"
  - "parallel embedding streams"
  - "open-vocabulary generalization"
  - "LTL"
date: 2026-05-08
content_hash: f106d3feb13cfe74
---

# Names Don't Matter: Symbol-Invariant Transformer for Open-Vocabulary Learning

**Conference**: ICML 2026  
**arXiv**: [2601.23169](https://arxiv.org/abs/2601.23169)  
**Code**: https://bu-depend-lab.github.io/Symbol-Invariant-Transformer/ (Project Page)  
**Area**: LLM Pre-training / Transformer Architecture / Symbolic Reasoning / Open Vocabulary  
**Keywords**: Symbolic Invariance, alpha-equivalence, parallel embedding streams, open-vocabulary generalization, LTL

## TL;DR
The authors modify the Transformer into a structure featuring "parallel embedding streams with shared weights for each interchangeable symbol + cross-stream aggregated attention." This architecture-level design guarantees the output remains strictly invariant under variable renaming (alpha-equivalence) and allows the inclusion of new symbols not seen during training into the vocabulary at test time, outperforming baseline models and even GPT-5.2 on propositional logic and LTL witness generation tasks.

## Background & Motivation
**Background**: Applying Transformers to symbolic reasoning tasks (theorem proving, mathematical reasoning, LTL synthesis) has become a mainstream approach. Theoretically, Transformers have been proven capable of simulating any finite automaton. However, existing works rely on a **fixed vocabulary** for training and testing, treating "symbols" as ordinary discrete tokens and learning specific embeddings for them.

**Limitations of Prior Work**: Symbolic systems contain a special class of tokens—variable names, atomic propositions, and bound variables in $\lambda$-calculus—which are **interchangeable**. Renaming them should not alter semantics ($\lambda x.x+1$ is equivalent to $\lambda y.y+1$). However, models trained with fixed embedding tables overfit to specific names: LLMs see performance drops of up to 70% in code tasks under semantic-preserving renaming; models like DeepLTL fail as soon as an AP name unseen during training appears in the test set.

**Key Challenge**: There is an inherent conflict in the role of the embedding table—to make a model "distinguish between two different symbols," it must assign them different vectors. However, once a vector encodes "identity," renaming invariance is broken, and new symbols cannot be represented. Existing mitigations (such as using random vectors instead of learned embeddings, Işık et al. 2025) allow post-training vocabulary expansion, but the inherent randomness means **different seeds will produce different predictions for alpha-equivalent inputs**, lacking formal guarantees.

**Goal**: Design a Transformer architecture such that (i) the output for any renaming of interchangeable tokens is automatically equivalent; (ii) new interchangeable tokens outside the training vocabulary can be accepted at test time; (iii) the design does not rely on randomness, making invariance an "invariance by construction" hard guarantee.

**Key Insight**: Since alpha-equivalence is essentially "permutation invariance among $k$ interchangeable symbols," each interchangeable symbol can be split into an independent embedding stream. All streams share the same weights, and information across streams is fused using permutation-invariant operators (sum/mean). Consequently, **renaming only changes the order of the $k$ streams**, and since the operators are permutation-invariant, equivalence is naturally maintained.

**Core Idea**: Replace a single embedding table with $k$ parallel embedding streams sharing the same weights. Each stream observes the input from the "perspective of one interchangeable symbol." Use permutation-invariant aggregated attention for cross-stream communication, elevating alpha-equivalence from a training objective to an architectural guarantee.

## Method

### Overall Architecture
The vocabulary is first partitioned into two parts: the interchangeable part $\mathbb{V}_i$ (e.g., atomic propositions, variable names) and the fixed part $\mathbb{V}_n$ (e.g., logical operators, keywords). Suppose $k$ distinct interchangeable tokens appear in the input. The model replicates the same input sequence into $k$ "streams." Stream $i$ views the input from the perspective of the $i$-th interchangeable token: the position where this token appears is filled with an "actual" embedding, while positions of other interchangeable tokens are filled with a shared "placeholder" embedding, and fixed tokens remain unchanged. Thus, each of the $k$ streams is an isomorphic sequence where "the non-interchangeable part is identical, and the interchangeable part is disambiguated into actual/placeholder." These can be processed in parallel by the same Transformer weights. Each layer contains two types of attention: **per-stream** (self-attention within each stream) and **aggregated** (attention on a fused view created by averaging the $k$ streams, while restoring the correspond stream's real representation at interchangeable positions). The decoder also adds cross-stream cross-attention, with optional per-stream (stream $i$ attends to encoder stream $i$) or aggregated modes. Finally, in the projection head: fixed token logits use the average across the $k$ streams (equivalence across perspectives), while interchangeable token $i$ logits come from the $i$-th stream (its "exclusive" stream). Since all streams share the same weights, adding new interchangeable tokens only requires opening more streams, requiring **no new parameters and no retraining**.

### Key Designs

1.  **Parallel Embedding Streams + actual/placeholder Dual Embeddings (Stream Construction)**:
    *   Function: Opens a separate embedding stream for each interchangeable token in the input, shifting the responsibility of "distinguishing identity" from the embedding table to the stream index.
    *   Mechanism: When stream $i$ processes the sequence, positions belonging to token $i$ are filled with the actual embedding; other interchangeable token positions are filled with a uniform placeholder embedding; fixed tokens are untouched. A binary mask tracks which token belongs to which position. Renaming $f$ mapping token $i$ to $j$ merely turns the original "stream $i$" into "stream $j$," while the tensors inside the streams remain identical. Since all streams share self-attention/FFN/LayerNorm weights, extra tokens in a new vocabulary only require **opening more streams** without unlearned parameters.
    *   Design Motivation: Traditional embedding tables encode "identity" into vectors, making renaming invariance and open vocabularies conflicting goals. By setting "identity = stream index," both goals can be achieved simultaneously.

2.  **Aggregated Attention**:
    *   Function: Allows $k$ streams to exchange information without breaking permutation invariance.
    *   Mechanism: The hidden states of the $k$ streams are averaged to create a fused view. At positions where interchangeable tokens occur, the actual hidden state from the corresponding stream $i$ is substituted back (restoring the "exclusive representation"). Self-attention is then performed on this fused view. Since average is permutation-symmetric ($\sum_i v_i = \sum_i v_{\pi(i)}$) and position-based restoration uses the "token-corresponding stream" rather than an "absolute stream index," the entire aggregation path remains invariant under renaming. Both Encoder and Decoder use a combination of per-stream and aggregated attention; the Decoder also adds cross-attention, where per-stream mode (stream $i$ attends to encoder stream $i$) is the default—proven by experiments to be the key for stream identity alignment.
    *   Design Motivation: Pure per-stream processing cannot perform relational reasoning (e.g., "$p \land q$" requires relating two propositions). Aggregated attention fills this gap, using symmetric operators to ensure relational reasoning is also alpha-invariant.

3.  **Formal Invariance Guarantee (Theorem 4.1)**:
    *   Function: Elevates "invariance to renaming" from an empirical property to a theorem.
    *   Mechanism: For any alpha-renaming $f$, the model $M$ satisfies $M(f(x)) = f(M(x))$, meaning $f^{-1}(\hat{y}') = \hat{y}$. The key observation is that when renaming $f$ maps token $i$ to $j$, "stream $i$" in the original computation becomes "stream $j$" in the new computation. Per-stream operators (weight-shared self-attention/FFN/LayerNorm) only depend on the stream input, independent of stream indexing. Aggregated operators use sums/averages plus position-based extraction of corresponding streams; sums are permutation invariant, and token-based extraction does not depend on absolute stream numbers. Thus, the entire network is strictly symmetric to stream permutations.
    *   Design Motivation: Previous random embedding methods (Işık et al. 2025) offered statistical invariance; different seeds could give different predictions for alpha-equivalent inputs. This work uses structural symmetry to turn invariance into a 0-1 hard guarantee.

### Loss & Training
The model employs the cosine loss from Işık et al. 2025 (normalizing both features and embeddings, with logits simplified to cosine similarity), specifically using AdaCos for adaptive scaling. Sequence length is treated as the batch dimension. The Encoder uses RoPE + tree position encoding (tailored for the tree structure of logical formulas). Beam search $k=3$ is used during decoding. To save costs, fixed token logits are averaged across all streams; interchangeable token $i$ logits are taken directly from stream $i$ to avoid dilution by cross-stream summation.

## Key Experimental Results

### Main Results
Two core tasks: **Propositional Logic Assignment Prediction** (PropRandom35) and **LTL Witness Generation** (LTLRandom35, DeepLTL benchmark), with correctness verified using pyaiger/spot. Three types of metrics: Correct (semantic accuracy), Exact (exact match with ground truth), Alpha-Covariance (consistency across alpha-equivalent inputs with 3/4/5 APs).

| Task | Training Setting | Method | Correct | Exact | $\alpha$-cov (5 AP) |
|------|---------|------|---------|-------|--------------|
| Prop Logic | Normal | Baseline | 95.62% | 57.94% | 76.02% |
| Prop Logic | Normal | Random Emb (Işık 2025) | 93.25% | 56.45% | 92.98% |
| Prop Logic | Normal | **Ours** | **98.03%** | **60.96%** | **100.0%** |
| Prop Logic | Reduced (80K) | Baseline | 63.26% | 29.31% | 53.31% |
| Prop Logic | Reduced | **Ours** | **70.43%** | **35.81%** | **100.0%** |
| Prop Logic | Pretrained | GPT-5.2 | 99.73% | 25.60% | 1.03% |
| LTL | Normal | Baseline | 98.23% | 83.23% | 91.80% |
| LTL | Normal | **Ours** | **98.24%** | 79.65% | **100.0%** |
| LTL | Pretrained | GPT-5.2 | 86.83% | 35.93% | 77.56% |

Highlights: Alpha-covariance is **100%** across all numbers of APs (verifying Theorem 4.1). Performance on LTL even **surpasses GPT-5.2** (98.24% vs 86.83%). GPT-5.2's $\alpha$-cov on 5 AP propositional logic is only 1.03%, showing LLMs struggle significantly with renaming invariance.

### Ablation Study (Propositional Logic)
Two-letter codes: first letter E/D/C = Encoder/Decoder/Cross-Attention; second letter P/A = per-stream/aggregated.

| Configuration | Heatmap Accuracy | Description |
|------|---------------|------|
| Best (EP-DP-EA-DA-CP) | 95.05% | Default recommended configuration |
| -CP+CA | 28.51% | **Catastrophic**: replacing per-stream cross-attention with aggregated; decoder fails to recognize corresponding encoder streams |
| -DP | 46.55% | Removing decoder per-stream → significant drop (failure in stream identity recognition) |
| -EA-DA | 72.35% | Removing both aggregated attention → loss of relational reasoning capability |
| -DA | 84.48% | Removing decoder aggregated → moderate drop |
| -EA | 92.47% | Removing encoder aggregated → slight drop (DA provides indirect compensation) |

### Key Findings
- **Per-stream cross-attention is the lifeline for identity alignment**: Switching to aggregated attention causes a 60+ percentage point drop, indicating that the decoder must know "which encoder stream the current token I am generating corresponds to."
- **Task dictates aggregated attention importance**: In propositional logic, relational reasoning (operators like implication/xor relating multiple APs) is the primary bottleneck, making aggregated attention highly contributory. In LTL, temporal reasoning is the main bottleneck rather than AP relations; removing DA even leads to a slight gain.
- **Large gap in alpha-covariance from baseline**: Baselines drop to 76% / 91.8% at 5 AP, while the proposed model remains at a constant 100%—this is an architectural guarantee, not a result of data or hyperparameters.
- **Pareto Improvement**: On renamed training sets, the proposed model outperforms the baseline on the original dataset, proving that this inductive bias is not just a robustness trick but helps the model learn structure.
- **Retrofitting Pretrained Models**: By mapping baseline token embeddings to actual/placeholder, 1 epoch of fine-tuning raised LTL heatmap performance from nothing to 85.91%, nearing the 84.13% of from-scratch training. This is a crucial path for LLM integration.
- **Controllable Overhead**: Theoretical complexity is $O(SL^2)$. For $S=10$, per-sample inference time is 3.38 ms → 5.13 ms, which is 4 orders of magnitude faster than GPT-5.2's 10–90 seconds/sample.

## Highlights & Insights
- **Invariance as an Architectural Primitive**: Equivalence class invariance (here, alpha-equivalence) is usually approximated via data augmentation or regularization. This paper addresses it directly using symmetric operators on group actions. This "group action = invariance" approach can potentially be migrated to other domains with symmetries (graph node renaming, set inputs, $k$-ary relation encoding).
- **Weight Sharing + Stream Indexing = Open Vocabulary**: Traditional models require changing embedding tables and retraining to expand the vocabulary. Here, adding a new token simply means opening another stream, with zero new parameters and zero retraining. This is an elegant paradigm for "infinite vocabularies" in symbolic systems—similar to how GNNs replace node ID embeddings with message passing.
- **Structural vs. Statistical Guarantees**: Previous random embedding work showed empirical gains but lacked 0-1 guarantees. This paper demonstrates a "formal guarantee + empirical performance + computational feasibility" trifecta. For safety/verification/formal reasoning scenarios, this 0-1 guarantee is far more important than a few points of accuracy.
- **Functional Decoupling of per-stream vs. aggregated**: The ablation clearly separates "stream identity recognition" (per-stream) from "cross-stream relational reasoning" (aggregated) and finds that their importance varies by task, providing a highly educational design analysis.
- **Path for Pretrained Model Retrofitting**: Allowing existing large models to gain alpha-invariance with minimal fine-tuning moves this technology from "must train from scratch" to "incremental improvement," which is key for application in code or math LLMs.

## Limitations & Future Work
- **Hard Upper Bound on Stream Count $S$**: Complexity is $O(SL^2)$ and memory is $O(SLd)$. $S \le 10$ has been validated, but program synthesis and theorem proving often involve hundreds of local variables, which would cause resource exhaustion. The authors suggest Top-K stream sparsification, but it must follow "input-symmetric" criteria (like position frequency) rather than token identity to avoid breaking invariance.
- **Inability to Generate "New Symbols" Outside training Vocabulary**: Streams are instantiated from the encoder input, meaning the model can only output interchangeable tokens that appeared in the input. Tasks requiring the "invention of new variable names" (constructive proof/code synthesis) are not yet supported; authors suggest maintaining a "future symbol pool" with reserved streams.
- **Relatively Restricted Tasks**: Experiments are limited to two toy-ish symbolic tasks (propositional logic and LTL). While the comparison against GPT-5.2 is compelling, proving the superiority of this method in real-world code, math, or theorem proving requires larger-scale experiments and integration with large models.
- **Task-Dependency of Aggregated Attention**: Ablations show DA is critical in propositional logic but harmful in LTL, meaning architectural hyperparameters (which A or P to enable) require task-level tuning rather than being purely plug-and-play.

## Related Work & Insights
- **vs. Random Embedding (Işık et al. 2025)**: They used random vectors to give each interchangeable token a non-learnable "identity code" + a shared learnable part. This statistically expands the vocabulary, but different seeds can yield different outputs for alpha-equivalent inputs. This paper uses structural symmetry for a 0-1 guarantee and improves accuracy on 5 AP logic (98.03% vs. 93.25%).
- **vs. Renamer (Ankner et al. 2023)**: Also sought provable invariance to variable renaming but did not consider vocabulary expansion (limited to a fixed vocabulary). This paper achieves both invariance and open vocabulary through weight sharing.
- **vs. Automated Reasoning GNNs (Olsák et al. 2019)**: GNNs in ATP also have invariance but only for graph structures and cannot handle seq2seq. This paper introduces that "permutation invariance" into an encoder-decoder Transformer for sequence inputs/outputs.
- **vs. Set/Permutation Invariant Transformers (Lee et al. 2019 Set Transformer / Xu et al. 2024)**: General permutation invariance treats the whole sequence as a set and discards order. This paper applies invariance only to the "interchangeable token subset," preserving order for fixed tokens—a more refined partial-invariance design.
- **vs. LLM (GPT-5.2)**: Generic LLMs have only 1.03% $\alpha$-cov on 5 AP propositional logic and 86.83% accuracy on LTL, taking 10–90 seconds per sample. The specialized model here achieves 98.24% on LTL with millisecond latency, illustrating that "for structural tasks, inductive bias is more important than scale."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elevates alpha-equivalence to architectural symmetry and combines it with open vocabulary; clean logic with formal guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ The two tasks are sufficient to illustrate the point (detailed ablation + heatmap + GPT-5.2 comparison + pretrained retrofitting), but lacks validation on large-scale symbolic tasks like code or mathematics.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear link from motivation → method → theorem → ablation. Concepts like per-stream/aggregated are consistently named and illustrated, making it easy to replicate.
- Value: ⭐⭐⭐⭐⭐ Provides a provably secure architectural primitive for formal verification, theorem proving, and symbolic reasoning, with a clear path to retrofitting existing pretrained models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] If open source is to win, it must go public](if_open_source_is_to_win_it_must_go_public.md)
- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](../../NeurIPS2025/llm_pretraining/learning_in_compact_spaces_with_approximately_normalized_transformer.md)
- [\[NeurIPS 2025\] Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](../../NeurIPS2025/llm_pretraining/born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)
- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](../../ICLR2026/llm_pretraining/lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[ICML 2026\] Focus and Dilution: The Multi-stage Learning Process of Attention](focus_and_dilution_the_multi-stage_learning_process_of_attention.md)

</div>

<!-- RELATED:END -->
