---
title: >-
  [Paper Note] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation
description: >-
  [ACL 2025][Hallucination Detection][Long-Form Generation] This work systematically investigates reference-free hallucination detection in open-domain long-form generation, discovering that the internal states (probabilities/entropy) of LLMs are insufficient to reliably distinguish factual and hallucinated content. It proposes RATE-FT (Rationale and Auxiliary Task Enhanced Fine-Tuning), which enhances fine-tuning by incorporating reasoning rationales and auxiliary QA tasks…
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Long-Form Generation"
  - "Auxiliary Tasks"
  - "Fine-Tuning"
  - "Reference-Free Detection"
date: 2026-05-08
content_hash: 62a7b752b3fd5585
---

# Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation

**Conference**: ACL 2025  
**arXiv**: [2505.12265](https://arxiv.org/abs/2505.12265)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Detection, Long-Form Generation, Auxiliary Tasks, Fine-Tuning, Reference-Free Detection

## TL;DR

This work systematically investigates reference-free hallucination detection in open-domain long-form generation, discovering that the internal states (probabilities/entropy) of LLMs are insufficient to reliably distinguish factual and hallucinated content. It proposes RATE-FT (Rationale and Auxiliary Task Enhanced Fine-Tuning), which enhances fine-tuning by incorporating reasoning rationales and auxiliary QA tasks, achieving over 3% improvement on LongFact compared to standard fine-tuning.

## Background & Motivation

LLM hallucination (generating content inconsistent with factual truth) remains a core challenge. In **open-domain long-form generation**, this issue is particularly critical:

**Differences from Short-Form**: In short-form tasks (where outputs are only a few tokens), the internal states of the model (output probability, entropy) are often used to detect hallucinations. However, long-form responses can span hundreds or even thousands of tokens, requiring the integration of information across multiple knowledge domains.

**Limitations of Prior Work**: 
   - Restricted to specific domains (e.g., biography generation)
   - Rely on external fact-checking tools (e.g., Google Search), which are not always available or scalable

**Core Problem**: Can a hallucination detector be developed that relies solely on the model itself, without needing external tools?

The paper first demonstrates through empirical analysis that the internal states of LLMs **cannot reliably** (i.e., not better than random guessing) distinguish between factual and hallucinated claims in long-form scenarios. This stands in stark contrast to the findings of SelfCheckGPT in short-form scenarios, revealing the unique challenges of long-form hallucination detection.

## Method

### Overall Architecture

The research path is progressive:

1. **Prior Analysis**: Verify whether LLM internal states are sufficient.
2. **Systemic Comparison**: Evaluate three classes of methods: Prompting, Probing, and Fine-Tuning.
3. **Proposed RATE-FT**: Incorporate reasoning rationales and auxiliary QA tasks on top of fine-tuning.

### Key Designs

**Data Construction** (based on the LongFact dataset):
1. For each prompt, generate a long-form response using Llama-3-8B-Instruct (greedy decoding).
2. Use a model to decompose the response into atomized claims.
3. Evaluate the relevance of each claim to the prompt.
4. For relevant claims, generate multi-step Google Search queries and determine if search results support them.
5. Obtain a set of labeled claims ("factual" or "hallucinated") containing 2,394 factual and 223 hallucinated claims.

**Internal State Analysis** (Prior Experiments):
- Multiple internal state variants were tested:
    - Arithmetic/geometric mean probability and entropy of all tokens
    - Average of Top-K lowest probability / highest entropy tokens (K=1,3,5)
    - Average of Top-P% lowest probability / highest entropy tokens (P=5,10,15)
    - Probability and entropy of entity-only tokens
- **Conclusion**: All variants fail to reliably distinguish factual and hallucinated claims.
- **Reason Analysis**: In long-form text, probability/entropy reflects the model's confidence in the "expression style" of a claim, rather than its confidence in the "correctness" of the claim—different formulations of the same fact yield different confidence levels.

**Comparison of Three Classes of Existing Methods**:
1. **Prompting**: Direct prompting of the model to make judgments ($\text{Prompt}_\text{TF}$, $\text{Prompt}_\text{Prob}$, SelfCheckGPT).
2. **Probing**: Training an MLP classifier on a frozen LLM using contextualized embeddings.
3. **Fine-Tuning**: LoRA fine-tuning of the base LLM to enhance its capability to output True/False.

**Core Innovations of RATE-FT**:

**Introducing Reasoning Rationales (Rationale)**:
- Incorporate reasoning rationales collected during the data construction stage (why search results support/refute the claim) into the fine-tuning data.
- Adopt a "label-rationale" format: output the label first, followed by the explanation. This allows obtaining $P_\text{factual}$ during inference using only the first token, avoiding additional inference costs.

**Introducing Auxiliary QA Tasks**:
- Inspired by the cognitive principle of "consolidating knowledge through repetition from different perspectives."
- For each claim, use a model to generate questions regarding its key information.
- For factual claims: extract correct answers and explanations from the claim.
- For hallucinated claims: guide the model using rationales to generate correct answers and explanations.
- Jointly train by merging these QA samples with the original detection data.

### Loss & Training

- Perform LoRA fine-tuning using LLaMA-Factory.
- Split the training data into 70% training / 20% validation / 10% testing.
- Search for optimal hyperparameters and classification thresholds on the validation set.
- Evaluation metric: Balanced Accuracy (BAcc) = $\frac{1}{2}(\frac{TP}{TP+FN} + \frac{TN}{TN+FP})$

## Key Experimental Results

### Main Results

On the LongFact and Biography datasets, using Llama-3-8B-Instruct:

| Method | LongFact BAcc | Biography BAcc |
|---|---|---|
| $\text{Prompt}_\text{TF}$ | 69.9% | 72.3% |
| $\text{Prompt}_\text{Prob}$ | 53.4% | 56.3% |
| SelfCheckGPT | 69.1% | 71.9% |
| $\text{Prompt}_\text{CoT-TF}$ | 74.9% | 74.8% |
| Probing | 74.4% | 77.0% |
| Fine-Tuning | 76.1% | 78.2% |
| **RATE-FT** | **79.6%** | **80.9%** |

RATE-FT significantly outperforms all baseline methods on both datasets (p<0.01).

**OOD (Out-of-Distribution) Generalization**: Trained on LongFact and evaluated on Biography, Fine-Tuning achieves 74.7%, still outperforming other methods.

### Ablation Study

| Method | LongFact | Biography |
|---|---|---|
| Fine-Tuning | 76.1% | 78.2% |
| RATE-FT w.o. aux | 77.5% | 79.4% |
| RATE-FT w.o. rationale | 77.9% | 79.5% |
| RATE-FT (Full) | 79.6% | 80.9% |

Both components contribute to performance, with performance decreasing when either auxiliary tasks or reasoning rationales are removed.

**Auxiliary Tasks vs. Data Augmentation**:
- Augment data by paraphrasing original claims using GPT-4 ($\text{Fine-Tuning}_\text{para}$): 76.8%
- Half of the training data for RATE-FT ($\text{RATE-FT}_\text{half}$): 78.5%
- Conclusion: The performance gains primarily stem from the design of the auxiliary QA tasks rather than simple data quantity increase.

**Cross-Model Generalization** (on LongFact):

| Model | Fine-Tuning | RATE-FT |
|---|---|---|
| Llama-3.1-70B-Instruct | 80.6% | 83.8% |
| Mistral-7B-Instruct | 70.8% | 73.4% |
| Qwen2.5-7B-Instruct | 78.4% | 81.1% |

RATE-FT consistently outperforms the baseline across all models, demonstrating strong generalization.

### Key Findings

1. **LLM Internal States are Ineffective for Long-Form Text**: This is completely different from findings in short-form scenarios. The reason is that in long-form text, token probabilities reflect "expression confidence" rather than "factual confidence."
2. **Fine-Tuning > Probing > Prompting**: There is a clear hierarchy of methods for detection effectiveness.
3. **Auxiliary QA Tasks are an Effective Mechanism Independent of Data Augmentation**: Offering complementary learning perspectives is more effective than simply adding more isomorphic data.
4. **Uncertainty Integration**: By setting dual thresholds ($\alpha_\text{low}$, $\alpha_\text{high}$), uncertain claims are marked as "unknown" and delegated to external tools, further boosting BAcc-unknown to 85.0%.
5. **Robustness to Response Length**: RATE-FT consistently outperforms Fine-Tuning across different length intervals (<500, 500-1000, >1000 tokens).

## Highlights & Insights

- **Systematic Research Methodology**: Starting from internal state analysis, progressively ruling out ineffective methods, and eventually landing on the optimal path of fine-tuning + auxiliary tasks, showing a very clear research logic.
- **Cognitive-inspired Auxiliary QA Tasks**: Drawing inspiration from the human learning principle of "consolidating knowledge by repeating it in different contexts," the authors designed auxiliary tasks complementary to the main task, which is a simple yet effective innovation.
- **No External Tools Needed at Inference**: Guided by Google Search only during the training data construction stage, inference is fully self-contained, ensuring practical deployment feasibility.
- **Uncertainty Integration Framework**: The proposed hybrid pipeline of dual thresholds and external tools provides a flexible deployment option for practical scenarios.
- The innovation of the "label-rationale" format—learning reasoning during training while requiring only the first token at inference—cleverly integrates the benefits of CoT into fine-tuning without increasing inference overhead.

## Limitations & Future Work

- Focuses only on improving detector performance, without exploring how to leverage detector feedback as a reward signal to guide LLMs toward generating more factual content.
- Domain coverage of the benchmark dataset is still limited (LongFact with 38 domains + Biography); a larger-scale benchmark would enhance applicability.
- Training data construction relies on Google Search for labeling, with potential limitations in search result quality and coverage.
- The proportion of hallucinated claims in the data is naturally low (2394 factual vs. 223 hallucinated), which may affect the model's sensitivity to hallucinations.
- Does not address faithfulness hallucination, dealing only with factuality hallucination.

## Related Work & Insights

- **SelfCheckGPT** (Manakul et al., 2023): Discovered that LLM probabilities correlate with factuality, whereas this study refutes this conclusion in long-form scenarios.
- **F2** (Hu et al., 2024): Also integrates reasoning and auxiliary tasks, but aims to enhance response faithfulness rather than hallucination detection.
- **Wei et al., 2024**: Proposed the LongFact benchmark and verified it using Google Search. This study builds on top of it to develop a reference-free detection method.
- Insights for hallucination research: The hallucination mechanisms of long-form and short-form text are fundamentally different, and short-form methods cannot be simply transferred.
- The idea of auxiliary task learning can be generalized to other scenarios for enhancing LLM capabilities (such as consistency, safety).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Enhancing detection with auxiliary QA tasks is a novel and effective paradigm
- **Value**: ⭐⭐⭐⭐ — No external tools needed during inference, making it suitable for practical deployment
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — System comparisons, ablations, and cross-model/cross-dataset validation are highly comprehensive
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The research logic is clear and well-structured, progressing logically from phenomenon to method to validation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](../../ICML2026/hallucination/building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ACL 2025\] Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](localizing_and_mitigating_errors_in_long-form_question_answering.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ICLR 2026\] Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection](../../ICLR2026/hallucination/beyond_in-domain_detection_spikescore_for_cross-domain_hallucination_detection.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)

</div>

<!-- RELATED:END -->
