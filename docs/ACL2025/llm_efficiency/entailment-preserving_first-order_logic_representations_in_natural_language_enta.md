---
title: >-
  [Paper Note] Entailment-Preserving First-order Logic Representations in Natural Language Entailment
description: >-
  [ACL 2025][LLM Efficiency][First-order logic] This paper formally defines the Entailment-Preserving First-Order Logic Representation (EPF) task and a set of reference-free evaluation metrics (EPR family). It proposes an iterative learning-to-rank training method that optimizes the NL→FOL translation of T5 models via the BRIO loss, enabling the generated FOL representations to be verified for entailment relations by automated theorem provers. It improves EPR by 1.8–2.7% and EP…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "First-order logic"
  - "natural language entailment"
  - "semantic parsing"
  - "Learning-to-rank"
  - "theorem proving"
date: 2026-05-08
content_hash: 72265278c0eacf92
---

# Entailment-Preserving First-order Logic Representations in Natural Language Entailment

**Conference**: ACL 2025  
**arXiv**: [2502.16757](https://arxiv.org/abs/2502.16757)  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: First-order logic, natural language entailment, semantic parsing, Learning-to-rank, theorem proving

## TL;DR
This paper formally defines the Entailment-Preserving First-Order Logic Representation (EPF) task and a set of reference-free evaluation metrics (EPR family). It proposes an iterative learning-to-rank training method that optimizes the NL→FOL translation of T5 models via the BRIO loss, enabling the generated FOL representations to be verified for entailment relations by automated theorem provers. It improves EPR by 1.8–2.7% and EPR@16 by 17.4–20.6% across three datasets.

## Background & Motivation
- **Background**: First-Order Logic (FOL) is a classic representation of natural language semantics, allowing automated theorem provers to determine entailment relations in theory. NL→FOL translation has recently attracted attention due to the code generation capabilities of LLMs.
- **Limitations of Prior Work**:
    - Classic methods (such as translating via CCG/AMR to FOL) almost completely fail on RTE tasks (Bos 2014 reported only 1.9% recall).
    - LLMs (such as GPT-4o) perform exceptionally well on synthetic logical entailment tasks but poorly on natural language entailment.
    - The core problem is **arbitrariness**: LLMs generate inconsistent predicate names and arities for synonymous concepts, leading to theorem proving failures.
- **Key Challenge**: Natural language entailment is much broader than strict logical entailment ("a human reading P would infer that h is likely true"), and traditional FOL cannot capture this non-strict entailment.
- **Key Insight**: Do not rely on reference FOL representations; instead, directly use the execution results of theorem provers as optimization signals.
- **Core Idea**: Use learning-to-rank to teach the model to favor generating FOL representations that preserve entailment in combinations, thereby indirectly reducing arbitrariness.

## Method

### Overall Architecture
1. Base Model Training: Fine-tune T5-base with standard cross-entropy on MALLS (an NL-FOL parallel corpus) to obtain $S_0$.
2. Sampling: For each sentence in the training set, sample K FOL representations using beam search with $S_t$.
3. Evaluation: Use the Vampire theorem prover to check all possible premise-hypothesis FOL combinations and calculate the score for each output.
4. Training: Train the model using the BRIO loss for ranking outputs to obtain $S_{t+1}$.
5. Iterate 5 times to obtain $S_5$.

### Key Designs
1. **EPR Metric Family (Entailment-Preserving Rate)**:
    - **EPR**: For each premise-hypothesis pair, select the highest-probability FOL translation combination and check if entailment is preserved.
    - **EPR@K**: Allow K translations per sentence; success is achieved if any combination preserves entailment (a relaxed version).
    - **EPR@K-Oracle**: Choose one translation per sentence to globally optimize EPR (NP-complete, solved using Answer Set Programming / ASP).
    - Inequality relation: EPR = EPR@1 ≤ EPR@K-Oracle ≤ EPR@K
    - Completely reference-free: No gold FOL representations are needed; only NL entailment labels are required.
    - Spurious entailment prevention: Verification steps ensure the hypothesis does not introduce new predicates/constants, and the proof must utilize all premises.

2. **Scoring Function**:
    - Since the global EPR metric cannot be directly used for sentence-level training, a sentence-level score is defined.
    - The score of an FOL output $S(p)_j$ is equal to the number of entailment-preserving combinations that include it.
    - Syntax-invalid outputs receive a score of -1.
    - If a sentence appears in multiple premise-hypothesis pairs, its scores are accumulated.
    - Design Motivation: Maximizing the score naturally improves the global EPR.

3. **BRIO Learning-to-Rank Loss**:
    - For each input, its K outputs are sorted in descending order of their scores.
    - Goal: Ensure that the average token log-probability of higher-scoring outputs is greater than that of lower-scoring outputs.
    - $\mathcal{L}_{BRIO} = \sum_i \sum_j max(\hat{p}(y_j|x) - \hat{p}(y_i|x) + \Delta(j-i), 0)$
    - Incorporate cross-entropy regularization: $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{BRIO}$
    - $\Delta=0.01$, $\lambda=10$

4. **Iterative Training**:
    - Re-evaluate and train in each iteration using the current model's sampled results.
    - Analogous to online iterative strategies in RLHF (e.g., Iterative DPO).
    - 5 iterations, 20 epochs each.

### Loss & Training
- Initial Training: Fine-tune T5-base with standard cross-entropy on MALLS (34k NL-FOL pairs).
- Iterative Training: $\mathcal{L} = \mathcal{L}_{CE} + 10 \cdot \mathcal{L}_{BRIO}$
- Sample $K=16$ outputs using beam search.
- Perform fully automated evaluation using the Vampire theorem prover.

## Key Experimental Results

### Main Results

| Dataset | Metric | T5-Iter5 | GPT-4o | T5-Iter0 | CCG2Lambda |
|--------|------|------|----------|------|------|
| EntailmentBank | EPR | **7.4** | 2.9 | 5.6 | 0.0 |
| eQASC | EPR | **4.9** | 1.1 | 2.6 | 0.0 |
| e-SNLI | EPR | **4.3** | 1.5 | 0.1 | 0.0 |
| EntailmentBank | EPR@16 | **32.8** | 13.2 | 15.4 | - |
| eQASC | EPR@16 | **33.1** | 11.4 | 12.5 | - |
| e-SNLI | EPR@16 | **36.1** | 8.3 | 3.4 | - |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Number of Iterations (0→5) | Continuous EPR increase | EPR, EPR@16, and EPR@16-Oracle monotonically increase across all datasets. |
| Unique Predicate Names/Sent. | Significant decrease after Iter1 | Synonymous concepts are mapped to fewer predicate names, reducing arbitrariness. |
| Arity Entropy | Continuous decrease | Predicate arities tend to be consistent (e.g., CauseCycles converges from a mix of 2 and 3 arguments to 2 arguments consistently). |
| Cross-dataset Transfer | EPR@16 increase | Iter5 trained on dataset A also outperforms Iter0 on dataset B, showing transferability. |

### Key Findings
- Classic NL→FOL methods (e.g., CCG2Lambda, AMR2FOL) have EPRs close to 0 on multi-premise RTE, resulting in complete failure.
- Although GPT-4o performs strongly on synthetic logical reasoning, its EPR for FOL generation on natural language entailment is only 1-3%.
- Core Mechanism of Iterative Training: The BRIO loss only provides gradient signals when different outputs have different scores, yet the model generalizes well to unseen combinations.
- Arbitrariness analysis reveals a direct causal relationship with EPR improvement: rising predicate name consistency $\uparrow \rightarrow$ success rate $\uparrow$.
- EPR@16-Oracle is close to EPR@16 (with a gap of only 1.7 percentage points), suggesting the existence of a near-optimal single-selection strategy.
- Models trained on e-SNLI show the best cross-domain generalization, likely due to the large scale of the data and its broad semantic coverage.

## Highlights & Insights
- Ingenious design of the EPR metric: completely reference-free, allows arbitrary FOL structures, and indirectly evaluates semantic quality through execution results (theorem proving).
- Modeling NL→FOL translation as an execution-guided sequence generation problem, which is closely aligned with execution-guided methods in semantic parsing.
- The iterative learning-to-rank framework is highly generalizable and can be extended to other execution-guided generation tasks.
- Profound analysis of arbitrariness—which represents the fundamental barrier for LLMs in FOL generation—providing both quantitative metrics and solutions in this paper.

## Limitations & Future Work
- Even for the best T5-Iter5, the EPR is only 7.4% (on EntailmentBank), which is still far from the theoretical upper bound (EPR@16-Oracle $\approx 31\%$).
- The study uses T5-base (220M) and does not explore the potential of larger models.
- Datasets lack linguistically controlled minimal pairs, which might miss fine-grained semantic differences.
- EPR@K-Oracle evaluation is NP-complete, requiring approximations in practical computation.
- **Future Work**: (1) Utilize larger pre-trained models (e.g., T5-XXL or LLMs) as backbones; (2) Integrate classical linguistic knowledge to constrain the structures of generated FOL.

## Related Work & Insights
- The negative conclusion of Bos (2014) regarding the failure of FOL on single-premise RTE is extended to multi-premise scenarios.
- BRIO was originally designed for ranking training in summarization tasks; this paper innovatively applies it to execution-guided semantic parsing.
- The approach is conceptually similar to online iterative training strategies in RLHF (such as Iterative DPO and Self-Play).
- Pipelines like Logic-LM (LLM + FOL + Prover) are effective on synthetic data but fail to generalize; this paper provides quantitative evidence for this phenomenon.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formally defining the EPF task and EPR metrics is a community-level contribution, and the iterative learning-to-rank method is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple baselines, reasoning type analyses, and cross-domain analyses, though the absolute EPR values remain relatively low.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous problem definition, clear metric definitions, and step-by-step progressive analysis.
- Value: ⭐⭐⭐⭐ Provides a new research framework and benchmark for the long-standing challenge of "using FOL for natural language inference".

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Consistency-Preserving Contrastive Decoding for Faithful Document-Grounded Dialogue](consistency-preserving_contrastive_decoding_for_faithful_document-grounded_dial.md)
- [\[ICLR 2026\] Predicting LLM Output Length via Entropy-Guided Representations](../../ICLR2026/llm_efficiency/predicting_llm_output_length_via_entropy-guided_representations.md)
- [\[ICLR 2026\] Cartridges: Lightweight and General-Purpose Long Context Representations via Self-Study](../../ICLR2026/llm_efficiency/cartridges_lightweight_and_general-purpose_long_context_representations_via_self.md)
- [\[AAAI 2026\] Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity](../../AAAI2026/llm_efficiency/connectivity-guided_sparsification_of_2-fwl_gnns_preserving_full_expressivity_wi.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](../../ACL2026/llm_efficiency/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)

</div>

<!-- RELATED:END -->
