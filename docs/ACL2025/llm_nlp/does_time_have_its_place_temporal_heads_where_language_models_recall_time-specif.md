---
title: >-
  [Paper Note] Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information
description: >-
  [ACL 2025][LLM (Other)][Temporal Heads] Through EAP-IG circuit analysis, "Temporal Heads" specialized in processing time-conditional knowledge are discovered in Llama-2/Qwen/Phi-3. Ablating these heads selectively decreases the accuracy of temporal knowledge (by 3-9%) without affecting time-invariant knowledge or general QA. The study also demonstrates the feasibility of selective temporal knowledge editing by injecting temporal head activations.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Temporal Heads"
  - "Attention Heads"
  - "Knowledge Circuits"
  - "EAP-IG"
  - "Temporal Knowledge Editing"
date: 2026-05-08
content_hash: 130f6099c066b62d
---

# Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information

**Conference**: ACL 2025  
**arXiv**: [2502.14258](https://arxiv.org/abs/2502.14258)  
**Code**: [https://github.com/dmis-lab/TemporalHead](https://github.com/dmis-lab/TemporalHead)  
**Area**: LLM/NLP  
**Keywords**: Temporal Heads, Attention Heads, Knowledge Circuits, EAP-IG, Temporal Knowledge Editing

## TL;DR
Through EAP-IG circuit analysis, "Temporal Heads" specialized in processing time-conditional knowledge are discovered in Llama-2/Qwen/Phi-3. Ablating these heads selectively decreases the accuracy of temporal knowledge (by 3-9%) without affecting time-invariant knowledge or general QA. The study also demonstrates the feasibility of selective temporal knowledge editing by injecting temporal head activations.

## Background & Motivation

**Background**: LLMs store a vast amount of factual knowledge that changes over time (e.g., presidents, CEOs, sports team members in a specific year). Prior studies show that LLMs possess some temporal awareness, correctly generating time-dependent answers when prompted with "In 1999, [X] was a member of sports team".

**Limitations of Prior Work**: Knowledge editing methods (e.g., ROME, MEMIT) operate on MLP layers to perform global editing, making it difficult to precisely manipulate knowledge along the temporal dimension—such as editing "the CEO of 2023" potentially affecting "the CEO of 2024" unexpectedly. Existing works primarily study temporal knowledge through external evaluation, lacking an understanding of how the model internally encodes and distinguishes knowledge under different temporal conditions.

**Key Challenge**: LLMs appear to distinguish facts across different years, but which components in the model architecture are specifically responsible for this time-conditional knowledge retrieval? Is there a specialized "temporal processing unit"?

**Goal**: (a) Locate the specific attention heads that handle temporal knowledge in LLMs; (b) verify the selectivity of these heads—affecting only temporal knowledge without degrading general capabilities; (c) explore targeted temporal knowledge editing using these heads.

**Key Insight**: Extend Knowledge Circuit Analysis from static facts to the temporal dimension—constructing Temporal Knowledge Circuits (TKC) for the same (subject, relation) across different years $T_k$, and identifying attention heads that consistently reappear in all temporal circuits but are absent in time-invariant circuits.

**Core Idea**: Extend EAP-IG circuit analysis to the temporal dimension to discover and validate "Temporal Heads" specifically handling time-conditional knowledge in LLMs.

## Method

### Overall Architecture

Input: Time-conditional knowledge triples $(T_k, s, r) \to o_k$ (e.g., "In 2004, David Beckham was a member of → Real Madrid"). Output: Identified positions of Temporal Heads and their functional validation. The overall pipeline consists of four stages: (1) constructing Temporal Knowledge Circuits (TKC) for multiple years using EAP-IG; (2) comparing TKC with time-invariant Knowledge Circuits (KC) to find temporal-specific nodes (Temporal Heads); (3) validating the selectivity of temporal heads via ablation experiments; (4) achieving temporal knowledge editing through attention value injection.

### Key Designs

1. **Temporal Knowledge Circuit (TKC) Construction**

    - **Function**: Construct knowledge circuit subgraphs for each year $T_k$ and knowledge category.
    - **Mechanism**: Define a time-variant edge importance score $S(e_i, T_k) = \log p_G(o_k | s, r, T_k) - \log p_{G/e_i}(o_k | s, r, T_k)$, where $p_{G/e_i}$ is the prediction probability after ablating edge $e_i$. Retain edges with $S(e_i, T_k) > \tau$ to form the TKC.
    - **Design Motivation**: Traditional knowledge circuits only analyze static triples $(s, r, o)$, failing to capture temporal conditioning. Introducing the parameter $T_k$ allows isolating circuit components that specifically respond to temporal conditions.

2. **EAP-IG (Effective Attribution Pruning with Integrated Gradients) Circuit Discovery**

    - **Function**: Precisely quantify the contribution of each edge/node to the correct prediction.
    - **Mechanism**: Use TransformerLens to intercept model components, calculate Integrated Gradients attribution scores for each edge, and prune low-contribution edges. Perform forward passes on both clean and corrupted prompts to compare the differences.
    - **Design Motivation**: Compared to simple zero-out ablation, IG provides smoother attribution estimates, allowing for more accurate identification of critical edges. Use Circuit Reproduction Score (CRS) to evaluate how well the pruned circuit reproduces the behavior of the original model.

3. **Temporal Head Identification & Verification**

    - **Function**: Find temporal-specific attention heads from the intersection of multiple TKCs.
    - **Mechanism**: Attention heads that consistently appear across TKCs of all years (1999/2004/2009) and knowledge categories, but do not appear in time-invariant circuits, are identified as Temporal Heads. In Llama-2, these are a15.h0 and a18.h3; in Qwen1.5, a17.h15; in Phi-3, a10.h13.
    - **Design Motivation**: Heads appearing only in a single circuit might be noise; consistent appearance across years signals a true temporal processing mechanism.

4. **Ablation Inference**

    - **Function**: Set the output weights of temporal heads to zero and measure changes in model behavior.
    - **Mechanism**: After setting temporal head outputs to zero, recalculate the log-probabilities of all candidate objects $O$. Evaluate the changes in probability distributions of Target (correct answer) vs. Non-Target (answers for other years) using softmax normalization.
    - **Comparison with prior methods**: Instead of solely assessing the correctness of the final output, this method analyzes shifts across the entire probability distribution, allowing for a more fine-grained observation of the differential impact of temporal heads on knowledge from different years.

5. **Temporal Knowledge Editing**

    - **Function**: Modify incorrect temporal knowledge by injecting attention values of temporal heads.
    - **Mechanism**: Extract the activations of the temporal head $\mathbf{a}_{\text{src}}$ from `source_prompt` (time queries correctly answered by the model), average them across multiple sources, and inject them into the corresponding positions of the `target_prompt` (time queries incorrectly answered) scaled by a coefficient $\lambda$. Dynamic modification is achieved via forward hook mechanisms without altering model parameters.
    - **Design Motivation**: If temporal heads are indeed the key channels for temporal knowledge, manipulating their activations should enable targeted edits of temporal knowledge. Three injection intensities ($\lambda = 1, 3, 6$) are tested in the experiments.

### Loss & Training
No training is involved in this work. All experiments are conducted zero-shot on pre-trained models using greedy decoding.

## Key Experimental Results

### Main Results: Impact of Temporal Head Ablation on Different Knowledge Types

| Model | Temporal Head Position | Temporal Knowledge (%) | Time-invariant (%) | QA (F1) |
|------|-----------|------------|------------|---------|
| Llama-2-7b (baseline) | a15.h0, a18.h3 | 29.7 | 61.8 | 55.4 |
| Llama-2-7b (Ablated) | — | 25.6 ↓4.1 | 61.7 | 54.9 |
| Qwen1.5-7B (baseline) | a17.h15 | 22.4 | 62.7 | 49.7 |
| Qwen1.5-7B (Ablated) | — | 19.8 ↓2.6 | 62.6 | 49.5 |
| Phi-3-mini (baseline) | a10.h13 | 35.4 | 59.8 | 46.8 |
| Phi-3-mini (Ablated) | — | 26.0 ↓9.4 | 60.6 | 46.2 |

### Ablation Study: Circuit Reproduction Score and Robustness to Prompt Variants

| Knowledge Type | Category | Avg No. of Nodes | Avg No. of Edges | CRS |
|---------|------|----------|---------|-----|
| Temporal Knowledge | Sports (Nicolas Anelka) | 29 | 37 | 74.14 |
| Temporal Knowledge | Sports (David Beckham) | 43 | 80 | 39.53 |
| Temporal Knowledge | Presidents (Argentina) | 42 | 102 | 60.97 |
| Temporal Knowledge | Presidents (South Korea) | 46 | 110 | 65.55 |
| Temporal Knowledge | CEO (Hewlett-Packard) | 52 | 115 | 53.49 |
| Temporal Knowledge | Average | 42 | 87 | **54.56** |
| Time-invariant | Object Superclass | 43 | 56 | 44.47 |
| Time-invariant | Geometric Shape | 52 | 118 | 76.09 |
| Time-invariant | Roman Numerals | 43 | 135 | 95.70 |
| Time-invariant | Average | 54 | 110 | **67.33** |

| Prompt Format (Llama-2) | Baseline (%) | Ablation (%) | Decrease |
|------------------------|-------------|---------|------|
| "In XXXX, ..." (Base) | 29.7 | 25.6 | -4.1 |
| "In year XXXX, ..." (Variant) | 33.4 | 30.0 | -3.4 |
| "In XXXX, which ..." (QA) | 34.5 | 30.1 | -4.4 |

### Key Findings
- **Temporal heads exhibit high selectivity**: Ablating temporal heads drops temporal knowledge accuracy by 3–9%, while time-invariant knowledge and QA remain virtually unaffected (drop < 0.6 F1).
- **Different years are affected differently by ablation**: a18.h3 notably impacts 2004 knowledge but has a lesser effect on 2002; during transition boundaries (e.g., 2002 $\to$ 2003 presidential transitions), non-target probabilities rise substantially.
- **Temporal heads exist across models but at different locations**: Llama-2 has a15.h0/a18.h3, Qwen has a17.h15, and Phi-3 has a10.h13.
- **Backup temporal heads appear at lower thresholds (70–80%)**: E.g., a0.h15, a20.h17, a31.h25 provide redundant protection.
- **No distinct heads exist exclusively for time-invariant knowledge**: General knowledge heads are reused for temporal tasks, but temporal heads are not utilized for general tasks.
- **Textual aliases also activate temporal heads**: Replacing "In 2004" with "In the year the Summer Olympics were held in Athens" still activates temporal heads (CRS drops to 40.3 but heads remain identifiable).
- **Temporal knowledge editing is effective**: Injecting a18.h3 activations successfully corrects incorrect answers (e.g., from "Vladimir Putin" to "Dmitry Medvedev"), with a18.h3 achieving the highest edit success rate.

## Highlights & Insights
- **The "Selective Ablation" paradigm of temporal heads**: This is the first systematic identification of dedicated components for temporal knowledge processing in LLMs. Ablating temporal heads only "shuts down" the temporal knowledge channel while preserving other capabilities, demonstrating that functionally differentiated attention heads indeed exist within LLMs.
- **Minimal invasiveness of temporal editing**: Injecting the activation values of a single head via forward hooks modifies temporal knowledge without retraining or tuning parameters. This opens up a new path for precise temporal knowledge updates.
- **Transferable methodology**: The EAP-IG + cross-conditional circuit comparison paradigm can be directly transferred to other studies of conditional knowledge—such as geographical conditions ("In France, ..."), persona conditions ("As a doctor, ..."), simply by replacing the conditioning variable.

## Limitations & Future Work
- **Limited to 7B-scale models**: Only Llama-2-7B, Qwen1.5-7B, and Phi-3-mini are analyzed. The behavior of temporal heads in larger models (70B+) remains unknown.
- **EAP-IG does not natively support GQA**: It cannot directly analyze newer models like Llama-3 that use Grouped-Query Attention (though supplementary analysis shows temporal heads a18.h15/a23.h26 exist in Llama-3, the circuit quality is suboptimal).
- **Temporal definitions are restricted to years**: Only year conditions in the form of "In XXXX" are tested. More complex temporal reasoning ("before/after", "during the Cold War", seasons, or monthly granularity) is not covered.
- **Limited editing success rate**: The success rate of knowledge editing halves (8% to 4%) under prompt variations, indicating insufficient robustness. Integration with methods like ROME/MEMIT might improve outcomes.
- **Single placement of temporal conditions**: The temporal condition is always placed before the subject. Whether different placements (e.g., at the end of the sentence) affect temporal head activation is untested.

## Related Work & Insights
- **vs. Knowledge Neurons (Dai et al., 2022)**: They focus on MLP neurons that store knowledge, whereas this work focuses on temporal processing mechanisms in attention heads. The two are complementary—MLPs store factual content, and attention heads route temporal conditions.
- **vs. ROME/MEMIT (Meng et al., 2022)**: Knowledge editing methods operate on MLP layers for broad editing, while temporal head editing is more precise but confined to the temporal dimension.
- **vs. Retrieval Heads (Wu et al., 2024)**: They find that retrieval heads are responsible for copying information from context, whereas temporal heads route knowledge based on temporal conditions. Functional interactions between them may exist.
- **vs. Subject/Relation Heads (Chughtai et al., 2024)**: They identify subject and relation heads; temporal heads in this work can be viewed as another class of semantic heads. How these three types of heads coordinate to complete conditional knowledge retrieval is an intriguing future direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic discovery and validation of temporal heads in LLMs, with a transferable methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models + multiple knowledge types + ablation + prompt variants + alias tests + editing applications, though limited by model scale.
- Writing Quality: ⭐⭐⭐⭐ Clear analytical logic and rich charts, though some minor grammatical errors exist in the English writing.
- Value: ⭐⭐⭐⭐ Direct application prospects for temporal knowledge editing and LLM interpretability, though still some distance from practical adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)
- [\[ACL 2025\] DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [\[ACL 2025\] Nudging: Inference-time Alignment of LLMs via Guided Decoding](nudging_inference_time_alignment.md)
- [\[ACL 2025\] UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization](unseentimeqa_time-sensitive_question-answering_beyond_llms_memorization.md)

</div>

<!-- RELATED:END -->
