---
title: >-
  [Paper Note] Building Reliable Long-Form Generation via Hallucination Rejection Sampling
description: >-
  [ICML 2026][LLM Evaluation][Hallucination Mitigation] The SHARS framework is proposed to detect and reject hallucinated content sentence-by-sentence during inference…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Hallucination Mitigation"
  - "Inference-time Compute"
  - "Semantic Entropy"
  - "Rejection Sampling"
  - "Long-form Generation"
date: 2026-05-08
content_hash: 512aa845e0db9f56
---

# Building Reliable Long-Form Generation via Hallucination Rejection Sampling

**Conference**: ICML 2026  
**arXiv**: [2606.03628](https://arxiv.org/abs/2606.03628)  
**Code**: https://github.com/TreeLLi/hallucination-rejection-sampling  
**Area**: LLM Evaluation  
**Keywords**: Hallucination Mitigation, Inference-time Compute, Semantic Entropy, Rejection Sampling, Long-form Generation  

## TL;DR

The SHARS framework is proposed to detect and reject hallucinated content sentence-by-sentence during inference, only retaining verified factual segments for continuous generation. Combined with an improved semantic entropy detector, HalluSE, it improves factual precision by approximately 20–26% on FactScore while maintaining or even increasing the volume of factual information.

## Background & Motivation

**Background**: Large language models (LLMs) excel in open-ended long-form generation, but hallucination issues severely impact reliability. Existing mitigation methods are primarily divided into training-time methods (e.g., DPO preference optimization, FactAlign sentence-level rewards) and inference-time methods (e.g., DoLa contrastive decoding, RAG retrieval augmentation).

**Limitations of Prior Work**: Long-form generation suffers from a **hallucination snowballing effect**, where early errors propagate and amplify errors in subsequent outputs. Existing inference-time methods either require external knowledge bases (RAG) or intervene only at the token level (DoLa), failing to effectively block the sentence-by-sentence accumulation of errors.

**Key Challenge**: Open-ended questions often have an infinite set of valid information, but models only utilize a finite subset. If hallucinated content can be filtered out and the model guided to explore truthful content within the remaining information space, the chain of error propagation can be broken.

**Goal**: To design a general inference-time framework capable of (1) detecting and rejecting hallucinated content segment-wise, (2) continuing generation only on the basis of verified facts, and (3) functioning without reliance on external knowledge bases.

**Key Insight**: The authors observe that the inference-time compute scaling paradigm has not been fully explored regarding factuality, and users in high-risk scenarios are willing to trade more inference time for more reliable output.

**Core Idea**: Use segment-wise rejection sampling to filter hallucinations sentence-by-sentence, continuing generation only based on verified facts to block the hallucination snowballing effect at its source.

## Method

### Overall Architecture

The SHARS pipeline is as follows: given a user query $q$, the model generates text sentence-by-sentence → HalluSE, the hallucination detector, evaluates the factuality of each sentence → the system decides to retain, rewrite, or discard based on the detection result → generation continues based on the verified text. Termination occurs when an EOS is generated, the maximum token budget is reached, or $N$ consecutive samples are all identified as hallucinations.

### Key Designs

1. **Segment-wise Rejection Sampling**:

    - **Function**: Detects and filters hallucinated content sentence-by-sentence, retaining only factual segments for subsequent generation.
    - **Mechanism**: Unlike traditional best-of-N global rejection, SHARS performs rejection sampling dynamically for each sentence. For the current sentence, the detector decomposes it into a set of facts and determines the credibility of each. If the sentence is entirely hallucinated, it is discarded; if it contains a mix of facts and hallucinations, it is rewritten by the LLM to retain only verified factual claims (adopting a "restructuring based on positive examples" approach rather than "deleting negative content," as experiments show the former performs better on small-to-mid-sized models); if it is entirely factual, it is retained. When sampling a new sentence, a **Following strategy** is used—temporarily keeping the identified hallucinated sentence as context for the model to continue generation, leveraging the model's content planning capability to avoid repeat generation of similar knowledge, while ensuring the hallucinated sentence does not participate in semantic entropy calculation to prevent detection pollution.
    - **Design Motivation**: Sentence-level intervention blocks the hallucination snowballing effect at the earliest stage of error emergence, proving more efficient and fine-grained than global rejection.

2. **HalluSE Hallucination Detector**:

    - **Function**: A long-form hallucination detection method based on semantic entropy that addresses three key flaws of naive semantic entropy.
    - **Mechanism**: The process involves: (1) breaking down generated text into (entity, factual claim) pairs rather than just factual claims—solving the entity ambiguity probe problem in naive methods; (2) generating $Q$ probing questions for each fact, using an improved prompting strategy to ensure the expected answers are unambiguous; (3) sampling $A$ answers for each question and explicitly instructing the LLM to provide all valid answers—solving the issue of artificially high semantic entropy caused by multiple valid answers; (4) calculating the semantic entropy for each question and taking the average; if it exceeds a threshold $\theta$, it is judged as a hallucination. Semantic entropy is defined as $H_s = -\sum_i p(C_i) \log p(C_i)$, where $C_i$ represents semantic clusters and $p(C_i) = \sum_{y \in C_i} p(y)$.
    - **Design Motivation**: Naive semantic entropy methods suffer from entity probe ambiguity and false positives due to multiple valid answers, leading to insufficient detection precision in long-form scenarios.

3. **Dynamic Abstention**:

    - **Function**: Automatically refuses to answer when the model lacks reliable knowledge about a query, rather than fabricating content.
    - **Mechanism**: Generation is terminated when $N$ consecutive samples of new sentences are all judged as complete hallucinations. This abstention can occur at the start of generation (lack of knowledge about the entire question) or midway through (after the model has output the parts it is certain of). The trade-off between response rate and factual precision can be smoothly controlled by adjusting the detection threshold $\theta$.
    - **Design Motivation**: Rather than having the model generate unreliable content for a user to review, it is better for the model to actively identify its own knowledge boundaries and stop appropriately.

## Key Experimental Results

### Main Results

Constraint-free evaluation on the FactScore benchmark (Qwen3-32B):

| Method | Response Rate (%) | # Unsup. Facts | # Supp. Facts | Factual Precision (%) |
|------|-----------|-------------|-----------|------------|
| Greedy | 99.5 | 8.8 | 9.7 | 52.4 |
| DoLa | 95.6 | 9.3 | 8.2 | 53.1 |
| ChatProtect | 98.9 | 8.1 | 6.8 | 54.4 |
| Self-Endorse | 91.8 | 4.9 | 8.4 | 63.2 |
| **SHARS-Info** | **92.9** | **4.2** | **11.7** | **73.5** |
| **SHARS-Prec** | **82.4** | **3.1** | **11.1** | **78.4** |

FactualBio hallucination detection evaluation (Qwen3-32B, Major+Minor):

| Method | AUROC | AURAC | Acc@0.8 | Acc@0.9 |
|------|-------|-------|---------|---------|
| Self-Check | 57.6 | 69.3 | 73.5 | 73.5 |
| P(True) | 69.8 | 73.3 | 70.0 | 70.0 |
| Naive SE | 66.2 | 73.1 | 70.5 | 70.5 |
| **HalluSE** | **72.9** | **77.3** | **75.4** | **72.8** |

### Ablation Study

| Sampling Strategy | Rewrite | Response Rate (%) | Factual Precision (%) | Relative Latency |
|----------|------|-----------|------------|---------|
| Following | Yes | 91.8 | 69.4 | 1.00× |
| Temperature | Yes | 95.6 | 64.8 | 1.01× |
| Following | No | 54.4 | 73.5 | 1.60× |
| Temperature | No | 40.1 | 76.2 | 1.55× |

### Key Findings

- **Contributions from both the SHARS framework and the detector**: Even when replacing semantic entropy with naive token-level entropy (Ours-NE), factual precision reaches 70.1%, surpassing the strongest baseline Self-Endorse (63.2%), indicating that the framework's segment-wise rejection strategy is effective.
- **Complementary to training-time methods**: Applying SHARS on top of FactAlign boosts factual precision from 53.1% to 80.6% (without length constraints), showing that inference-time and training-time methods can synergize.
- **Effective for small models**: A +16–24% precision gain is achieved on Qwen3-4B, demonstrating that the method does not rely solely on strong instruction-following capabilities.
- **Rewriting is crucial for response rate**: Disabling rewriting causes the response rate to plunge from 91.8% to 54.4%, as mixed sentences are discarded entirely, leading to high abstention; the Following strategy outperforms the Temperature strategy in both supported fact count and precision.

## Highlights & Insights

- **A New Paradigm for Inference-time Factuality Scaling**: This work is the first to systematically demonstrate the scaling characteristics of inference-time compute in open-ended generation factuality—increasing inference compute within reasonable bounds consistently improves factual precision, with efficiency significantly outperforming methods like Self-Endorse (2–3× lower compute for the same precision).
- **Positive Example Rewriting vs. Negative Deletion**: It was discovered that having the LLM restructure sentences based on a list of verified facts is more effective than providing the original sentence with annotations to delete hallucinated parts. This finding is particularly significant for small-to-mid-sized models and offers insights for other LLM post-processing tasks.
- **Spontaneous Perception of Knowledge Boundaries**: The abstention mechanism does not require additional calibration; the model naturally stops after repeated sampling failures, essentially achieving a perception of its own parametric knowledge boundaries.

## Limitations & Future Work

- The method does not introduce external knowledge; if the model is completely ignorant of a topic, rejection sampling cannot produce new correct information and can only choose to abstain.
- Inference-time compute overhead remains high (approx. 10–50× Greedy); although better than baselines at the same precision, it remains challenging for latency-sensitive scenarios.
- Currently applied only to English factuality benchmarks; cross-lingual and non-factual hallucination (e.g., logical inconsistency) scenarios have not yet been validated.
- Potential improvements: (1) Combining with RAG to fill model knowledge gaps; (2) Distilling the detector into a lightweight probe to reduce overhead; (3) Exploring more efficient batch sentence-level detection methods.

## Related Work & Insights

- **Semantic Entropy** (Farquhar et al., 2024): The foundational method for HalluSE, which this paper improves via entity decomposition, prompt enhancement, and handling multiple valid answers for long-form scenarios.
- **FactAlign** (Huang & Chen, 2024): A training-time sentence-level factual reward method, which is orthogonally complementary to SHARS.
- **DoLa** (Chuang et al., 2024): Contrastive decoding between layers; provides fine-grained token-level intervention but cannot block sentence-level error propagation.
- **Self-Endorse** (Wang et al., 2024): A self-consistency verification method that offers high precision but comes with even greater computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)
- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](../../NeurIPS2025/llm_evaluation/keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](../../ICLR2026/llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)

</div>

<!-- RELATED:END -->
