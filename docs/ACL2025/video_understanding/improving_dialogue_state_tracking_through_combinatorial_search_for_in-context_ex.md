---
title: >-
  [Paper Note] Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples
description: >-
  [ACL 2025][Video Understanding][Dialogue State Tracking] This paper proposes CombiSearch, a method that employs combinatorial scoring to select the optimal combination of in-context examples for Dialogue State Tracking (DST). It outperforms all baselines trained on 100% of the training data using only 5% of the data. Under oracle settings, its Joint Goal Accuracy (JGA) upper bound is 12% higher than traditional methods.
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Dialogue State Tracking"
  - "In-Context Learning"
  - "Combinatorial Search"
  - "Retriever Training"
  - "Few-Shot Learning"
date: 2026-05-08
content_hash: 09eed516e95e6c4d
---

# Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples

**Conference**: ACL 2025  
**arXiv**: [2506.00622](https://arxiv.org/abs/2506.00622)  
**Code**: [GitHub](https://github.com/holi-lab/combisearch)  
**Area**: Video Understanding  
**Keywords**: Dialogue State Tracking, In-Context Learning, Combinatorial Search, Retriever Training, Few-Shot Learning

## TL;DR

This paper proposes CombiSearch, a method that employs combinatorial scoring to select the optimal combination of in-context examples for Dialogue State Tracking (DST). It outperforms all baselines trained on 100% of the training data using only 5% of the data. Under oracle settings, its Joint Goal Accuracy (JGA) upper bound is 12% higher than traditional methods.

## Background & Motivation

**Background**: Dialogue State Tracking (DST) is a core task in task-oriented dialogue systems, aiming to track user intentions (slot-value pairs) throughout a conversation. Recently, LLMs have achieved DST without fine-tuning via in-context learning (ICL), but performance heavily depends on the quality of example selection.

**Limitations of Prior Work**: Existing preparation of training data for ICL example retrievers suffers from three major limitations:
   - Examples are scored independently, ignoring the **synergistic effects** when used in combination.
   - Retrieval relies solely on dialogue state similarity, ignoring the **linguistic features** of the dialogue itself (e.g., coreference resolution, dialogue style).
   - Ranking based on dialogue state similarity serves as **indirect supervision**, which does not directly optimize DST performance metrics.

**Key Challenge**: The quality of examples depends not only on the similarity of individual examples to the query but, more importantly, on the contribution of the example combination to the final DST performance. However, existing methods fail to capture this combinatorial effect.

**Goal**: To design an efficient combinatorial search method to generate high-quality training data for ICL retrievers, thereby directly optimizing DST performance.

**Key Insight**: Think of each example as a "team member" and measure its consistent contribution across different "teams" via randomized combinatorial sampling and JGA evaluation.

**Core Idea**: By randomly sampling combinations of examples and accumulating JGA scores, the CombiScore for each example can be calculated in linear time. This score is then used to train a retriever capable of selecting the most complementary combinations of examples.

## Method

### Overall Architecture

CombiSearch consists of three stages:
1. **Candidate Pool Construction**: Employs a hybrid retrieval using BM25 and SBERT to build a diverse candidate example pool.
2. **Combinatorial Scoring**: Randomly samples combinations of examples and accumulates the CombiScore for each example based on DST performance (JGA).
3. **Retriever Training**: Trains the retriever via InfoNCE loss using data sorted by CombiScore.

### Key Designs

#### 1. Diverse Candidate Pool Construction
- **Function**: Constructs a diverse pool of $N=100$ high-quality candidate examples for each query.
- **Mechanism**: 
    - Retrieve top-$N$ candidates using BM25 (capturing lexical/linguistic features such as word choice, coreference, and named entities) and SBERT (capturing semantic similarity), respectively.
    - Merge and deduplicate the retrieved candidates, then rerank them using a hybrid score: $\text{hybrid\_score} = \text{TF-IDF} \times \text{cos\_sim}$.
    - Retain the top-$N$ candidates to form the final candidate pool.
- **Design Motivation**: Relying solely on semantic retrieval overlooks crucial lexical and syntactic features. BM25 can effectively capture linguistic phenomena such as coreference resolution.

#### 2. Combinatorial Example Scoring (CombiScore)
- **Function**: Evaluates the contribution of each individual example when combined with other examples for DST.
- **Mechanism**: 
    - Randomly sample $k=10$ examples from the candidate pool $E$ to form a combination.
    - Run the DST model using this combination as in-context examples and compute the JGA.
    - If the JGA is 1, the CombiScore of each example in this combination is incremented by 1; if the JGA is 0, the score remains unchanged.
    - Repeat the sampling and evaluation process $M=3$ times to obtain the final cumulative CombiScore for each example.
- **Key Advantage**: The computational complexity is linear $O(N \cdot M)$ with respect to the number of examples, rather than undergoing exponential exhaustive search.
- **Design Motivation**: Exhaustively searching all $\binom{100}{10}$ combinations is intractable. Approximating via random sampling can effectively identify "good teammate" style examples.

#### 3. Retriever Training
- **Function**: Trains a dedicated ICL example retriever using the CombiScore dataset.
- **Mechanism**: 
    - Positive examples: The top-$|P|$ examples with the highest CombiScores.
    - Negative examples: The bottom-$B$ examples with the lowest CombiScores + $B-1$ randomly sampled examples from outside the candidate pool.
    - Train using the InfoNCE contrastive learning loss:
  
$$L(x, P, N) = \sum_{e^+ \in P} -\log \frac{\exp(\text{sim}(x, e^+))}{\sum_{e' \in N \cup \{e^+\}} \exp(\text{sim}(x, e'))}$$

- **Hybrid Retrieval during Inference**: Retrieve using both the trained retriever and BM25 simultaneously, then merge and rerank using the hybrid score.

### Loss & Training

- **Scoring Model**: Llama-3-8B-Instruct is used as the DST model during the CombiSearch phase.
- **Retriever**: Fine-tuned SBERT with InfoNCE loss.
- **Prompt Format**: Text-to-JSON (reduces formatting errors compared to Text-to-Python).
- **Key Parameters**: Candidate pool $N=100$, combination size $k=10$, evaluation count $M=3$.

## Key Experimental Results

### Main Results

Closed-source setting (gpt-3.5-turbo, MultiWOZ 2.4) — JGA scores:

| Method | 1% Data | 5% Data | 100% Data |
|------|---------|---------|-----------|
| IC-DST | 50.7 | 48.4 | 55.4 |
| SynthDST | 51.0 | 50.4 | 55.2 |
| RefPyDST | 44.9 | 52.3 | 58.0 |
| **CombiSearch** | **56.7** | **59.8** | **64.2** |

Open-source setting (Llama-3-8B, MultiWOZ 2.4):

| Retrieval Method | 1% | 5% | 100% |
|----------|-----|-----|------|
| RefPyDST | 47.7 | 50.6 | 55.5 |
| **CombiSearch** | **52.1** | **56.2** | **61.8** |

CombiSearch outperforms all baselines trained on 100% of the data using only 5% of the data, demonstrating a **20x data efficiency gain**.

### Ablation Study

Oracle setting (no retrieval errors, Llama-3-8B) — JGA upper bound:

| Scoring Method | 1% | 5% | 100% |
|----------|-----|-----|------|
| RefPyDST | 58.0 | 62.7 | 69.7 |
| Hybrid | 60.4 | 68.0 | 75.9 |
| **CombiSearch** | **68.4** | **75.1** | **82.7** |

Combinatorial scoring vs. Individual scoring (Oracle setting):

| Method | JGA | Calls/query |
|------|-----|---------------|
| Individual (100 candidates) | 79.9% | 100 |
| CombiSearch (M=3) | 82.7% | 30 |
| CombiSearch (M=9) | 85.3% | 90 |

Candidate pool construction ablation: BM25+SBERT scores 3-4% higher JGA than SBERT alone, and 14% higher than a random pool.

### Key Findings

1. CombiSearch achieves 59.8% JGA on MultiWOZ 2.4 with only 5% of the data, surpassing the performance of all baselines using 100% of the data.
2. The Oracle upper bound is 12% higher than traditional methods (82.7% vs. 69.7%), proving that existing retriever training data is severely sub-optimal.
3. Combinatorial scoring yields a 2.8% higher JGA than individual scoring while requiring only 1/3 of the computation.
4. In cross-domain transfer experiments on the SGD dataset, CombiSearch maintains a consistent advantage.
5. In coreference resolution scenarios, CombiSearch retrieves more examples containing coreferences (3.7 vs. 3.6/query), resulting in a ~2% higher JGA.

## Highlights & Insights

- **Extremely High Data Efficiency**: Exceeding 100% baselines with only 5% of the data yields a 20x data efficiency improvement, offering immense practical value.
- **Clever Combinatorial Approximation**: Converts an intractable combinatorial optimization problem into linear-time randomized sampling and scoring, grounded in intuitive "good teammate" heuristics.
- **Revealing Key Insights**: Identifies that training data for existing retrievers is sub-optimal (with a 12% upper-bound gap), pointing out a breakthrough direction in the DST field.
- **Text-to-JSON Prompt**: Reduces formatting errors compared to Text-to-Python, serving as a useful engineering contribution.

## Limitations & Future Work

1. The data construction stage of CombiSearch requires multiple LLM calls, resulting in longer wall-clock time compared to simpler methods.
2. Evaluated only on MultiWOZ and SGD, with a lack of experiments on more diverse dialogue datasets.
3. The sampling count of $M=3$ is relatively small, which might cause unstable estimation with larger candidate pools.
4. The hybrid retrieval strategy uses a simple product of scores; more optimal fusion methodologies could be explored.
5. The performance analysis comparing combinatorial search under the 3-shot setting versus the 10-shot setting is insufficient.

## Related Work & Insights

- **IC-DST** (Hu et al., 2022): The first ICL-based DST method, which ranks examples using dialogue state similarity.
- **RefPyDST** (King & Flanigan, 2023): Introduces Maximal Marginal Relevance (MMR) for diversifying example selection.
- **Se²** (Liu et al., 2024a): Greedy combinatorial search; CombiSearch achieves a 12% higher JGA and is more efficient.
- **SynthDST** (Kulkarni et al., 2024): Trains retrievers using synthetic data; CombiSearch outperforms it even with only 1% data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combinatorial scoring concept is novel, and the "good teammate" intuition is highly inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive experiments across multiple settings (closed-source, open-source, oracle, cross-domain, coreference).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Highly rigorous logic, clear charts, and a smooth query-solution-evaluation flow.
- **Value**: ⭐⭐⭐⭐ — The 20x data efficiency and 12% upper-bound improvement are highly significant for the DST field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] From Teacher to Student: Tracking Memorization Through Model Distillation](from_teacher_to_student_tracking_memorization_through_model_distillation.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](../../ECCV2024/video_understanding/finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[CVPR 2026\] Hypergraph-State Collaborative Reasoning for Multi-Object Tracking](../../CVPR2026/video_understanding/hypergraph-state_collaborative_reasoning_for_multi-object_tracking.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)

</div>

<!-- RELATED:END -->
