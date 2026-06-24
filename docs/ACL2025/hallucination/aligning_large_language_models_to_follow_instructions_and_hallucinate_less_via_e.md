---
title: >-
  [Paper Note] Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering
description: >-
  [ACL 2025][Hallucination Detection][Hallucination Mitigation] This paper proposes the NOVA framework, which filters knowledge-aligned, high-quality instruction data by measuring the LLM's familiarity with instructions via Internal Consistency Probing (ICP) and familiarity with target responses via Semantic Equivalence Identification (SEI). Fine-tuning LLaMA-3-8B with only 5% of selected data achieves an 8.6-point improvement on BioGEN and a 7.2-point improvement on FollowRAG…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hallucination Mitigation"
  - "Data Filtering"
  - "Instruction Tuning"
  - "Knowledge Alignment"
  - "Internal Consistency"
date: 2026-05-08
content_hash: a1c6a93acf7f9bea
---

# Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering

**Conference**: ACL 2025  
**arXiv**: [2502.07340](https://arxiv.org/abs/2502.07340)  
**Code**: [GitHub](https://github.com/S1s-Z/NOVA)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Mitigation, Data Filtering, Instruction Tuning, Knowledge Alignment, Internal Consistency

## TL;DR

This paper proposes the NOVA framework, which filters knowledge-aligned, high-quality instruction data by measuring the LLM's familiarity with instructions via Internal Consistency Probing (ICP) and familiarity with target responses via Semantic Equivalence Identification (SEI). Fine-tuning LLaMA-3-8B with only 5% of selected data achieves an 8.6-point improvement on BioGEN and a 7.2-point improvement on FollowRAG, while preserving instruction-following capability.

## Background & Motivation

**Background**: Instruction tuning is a crucial step for LLM alignment. However, studies show that fine-tuning LLMs on data containing unfamiliar knowledge encourages overconfidence and hallucination.

**Limitations of Prior Work**: (a) RL-based methods (e.g., FLAME-DPO) reduce hallucination through preference learning after instruction tuning, but deteriorate instruction-following capabilities and incur extra data and API costs; (b) Existing data filtering methods (e.g., IFD, CaR, Nuggets) focus solely on quality, selecting high-quality data that often contains expert-level knowledge unfamiliar to the LLM, which paradoxically exacerbates hallucination.

**Key Challenge**: High-quality instruction data often contains deeper expert knowledge (correctness $\uparrow$), which may not have been learned by the LLM during pre-training (familiarity $\downarrow$), thereby intensifying hallucination.

**Goal**: To simultaneously achieve "instruction following" and "hallucination reduction" during the instruction tuning stage by filtering data that is both high-quality and knowledge-aligned.

## Method

### Overall Architecture

NOVA = ICP (measuring instruction familiarity) + SEI (measuring response familiarity) + Quality RM (ensuring data quality). The final rank is computed as $(familiarity\_rank + quality\_rank) / 2$, and the top-$k$\% data is selected for fine-tuning.

### Key Designs

1. **Internal Consistency Probing (ICP)**:
    - **Function**: Measures the LLM's understanding of a given instruction $q$.
    - **Mechanism**: Generates $K$ responses for instruction $q$, and extracts the internal states of the last token from each response as sentence embeddings $E=[e_1,...,e_K]$. Assuming $E \sim \mathcal{N}(\mu, \Sigma)$, it calculates the differential entropy: $F_{ins}(q) = \frac{1}{2}\sum_{i=1}^d \lambda_i + G$, where $\lambda_i$ represents the eigenvalues of the covariance matrix $\Sigma$. Low entropy $\rightarrow$ consistent responses $\rightarrow$ high LLM familiarity with the instruction.
    - **Design Motivation**: Compared to surface-level metrics like perplexity or Rouge-L, the differential entropy of internal states captures more fine-grained semantic consistency information.

2. **Semantic Equivalence Identification (SEI)**:
    - **Function**: Measures the LLM's familiarity with the knowledge in the target response $r$.
    - **Mechanism**: (1) Employs an NLI model to perform bidirectional entailment detection on the $K$ generated responses, clustering semantically equivalent responses into $[c_1,...,c_M]$; (2) Applies a voting strategy to determine which cluster the target response $r$ belongs to; (3) Calculates $F_{res}(r) = k_{target}/\sum k_m$—a higher proportion of the target cluster in total responses indicates greater familiarity of the LLM with the target response content.
    - **Design Motivation**: Since target responses originate from human annotations or GPT-4, the LLM's internal states cannot effectively represent these external inputs. Thus, NLI-based semantic clustering and voting are utilized as an alternative.

3. **Expert-Aligned Quality Reward Model**:
    - **Function**: Trains a reward model using 3,751 expert-annotated preference data points to evaluate data quality.
    - **Mechanism**: The final score is computed as $R_{final}^{(i)} = \frac{1}{2}(R_{familiarity}^{(i)} + R_{quality}^{(i)})$, balancing familiarity and quality.
    - **Design Motivation**: When only considering familiarity (w/o Quality RM), the selected data significantly reduces hallucinations but drastically degrades instruction-following capabilities (MT-Bench drops from 64.6 to 48.6).

### Loss & Training

Experiments are conducted based on LLaMA-3-8B and LLaMA-3-70B in Alpaca (52K) and Alpaca-GPT4. Top 5%/10%/15% data is selected for SFT. The NLI model uses DeBERTa-v3.

## Key Experimental Results

### Main Results

LLaMA-3-8B, Alpaca-GPT4, 5% data selection:

| Method | BioGEN(FactScore)↑ | LongFact-Obj↑ | FollowRAG-Avg↑ | MT-Bench↑ |
|------|-------------------|---------------|----------------|-----------|
| Vanilla-100% | 41.9 | 84.7 | 38.1 | 64.3 |
| IFD-5% | 46.7 | 84.4 | 42.3 | 65.0 |
| Nuggets-5% | 47.2 | 87.0 | 41.5 | 66.2 |
| FLAME-DPO | 46.3 | 87.3 | 41.5 | 56.2 |
| **NOVA-5%** | **50.5** | **90.1** | **45.3** | **64.6** |

Improvements of NOVA relative to Vanilla-100%: BioGEN +8.6, LongFact +5.1, FollowRAG +7.2, MT-Bench +0.3.

### Ablation Study

Contributions of each component (LLaMA-3-8B, Alpaca-GPT4, 5%):

| Configuration | BioGEN↑ | MT-Bench↑ |
|------|---------|-----------|
| Full NOVA | 50.5 | 64.6 |
| -w/o ICP | 47.6 | 64.1 |
| -w/o SEI | 48.3 | 63.8 |
| -w/o Quality RM | 55.6 | **48.6** |
| -w/o ICP & SEI | 43.7 | 65.2 |

Comparison of alternative ICP methods:

| ICP Alternatives | BioGEN↑ | MT-Bench↑ |
|---------|---------|-----------|
| Internal States (NOVA) | **50.5** | **64.6** |
| Perplexities | 48.4 | 62.2 |
| Rouge-L | 47.9 | 61.5 |
| External Embedding Models | 49.8 | 63.9 |

### Key Findings

1. **Surpassing 100% full-data training with only 5% of the data**: This applies to both the hallucination and instruction-following dimensions.
2. **RL-based methods (FLAME-DPO, SELF-EVAL) severely compromise instruction-following while reducing hallucinations**: MT-Bench scores drop by 8.1 and 11.2, respectively.
3. **Pure quality-based data filtering can exacerbate hallucinations**: For instance, IFD increases the number of generated facts on LongFact (39.2 vs 32.0).
4. **Quality RM is key to maintaining instruction-following capabilities**: Without it, BioGEN achieves higher scores (55.6 vs 50.5) but MT-Bench collapses (48.6 vs 64.6).
5. **Internal states are more effective than external embeddings**: Internal states capture fine-grained information that might otherwise be lost during the decoding phase.
6. **Scalability to 70B**: NOVA-5%-70B achieves 60.9 (+7.2 Gain) on BioGEN.

## Highlights & Insights

- **Resolves a fundamental trade-off**: Co-optimizes two potentially conflicting objectives through data filtering without introducing extra RL stages.
- **Novelty of ICP**: Employs the differential entropy of LLM internal states to measure consistency, capturing minor semantic differences more effectively than surface-level metrics.
- **NLI+Voting design of SEI**: Cleverly addresses the challenge where target responses from external models cannot be effectively represented by internal states of the local LLM.
- **Quality RM as a balancer**: Shows that while pure familiarity-based filtering significantly mitigates hallucinations, it hurts instruction capabilities; the role of the reward model as a balancer is crucial.

## Limitations & Future Work

- Requires generating $K$ responses for each instruction, which increases offline data filtering latency (though it has zero impact on inference time).
- Applicable only to single-turn instruction data; multi-turn dialog scenarios remain unexplored.
- The Quality RM requires training on 3,751 expert preference data points, introducing additional data requirements.
- NLI models may yield inaccurate matches of semantic equivalence on long texts or domain-specific professional fields.

## Related Work & Insights

- **Gekhman et al. (2024)**: Provides theoretical foundations showing that fine-tuning LLMs on new knowledge encourages hallucinations.
- **FLAME (NeurIPS 2024)**: Offers RL-based hallucination mitigation, but compromises instruction-following capabilities.
- **Insight**: 'Data selection' can be more efficient than 'extra training stages (RL)' in solving alignment issues, as it prevents the problems at their root.

## Rating

- Novelty: ⭐⭐⭐⭐ The designs of ICP and SEI are novel, taking a unique perspective to resolve hallucinations through knowledge alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 3 hallucination benchmarks, 2 instruction benchmarks, comprehensive ablation studies, alternative comparisons, and human evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation and well-justified motivations, though heavily marked with mathematical symbols.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for LLM alignment research, providing a simple yet effective methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/hallucination/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)
- [\[ACL 2025\] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)

</div>

<!-- RELATED:END -->
