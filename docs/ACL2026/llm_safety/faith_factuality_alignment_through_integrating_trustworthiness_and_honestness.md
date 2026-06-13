---
title: >-
  [Paper Note] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness
description: >-
  [ACL 2026][LLM Safety][Factuality Alignment] This paper proposes the FAITH framework, which maps LLM uncertainty signals (consistency + semantic entropy) to naturally described Knowledge State Quadrants (Trustworthiness…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Factuality Alignment"
  - "Knowledge State Quadrants"
  - "Uncertainty Estimation"
  - "PPO"
  - "Retrieval Augmentation"
date: 2026-05-08
content_hash: 68425e45756210a7
---

# FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.10189](https://arxiv.org/abs/2604.10189)  
**Code**: [https://github.com/xndong/FAITH](https://github.com/xndong/FAITH)  
**Area**: Information Retrieval  
**Keywords**: Factuality Alignment, Knowledge State Quadrants, Uncertainty Estimation, PPO, Retrieval Augmentation

## TL;DR
This paper proposes the FAITH framework, which maps LLM uncertainty signals (consistency + semantic entropy) to naturally described Knowledge State Quadrants (Trustworthiness × Honestness). It designs a fine-grained reward function considering uncertainty for PPO training and utilizes a RAG module to correct potential errors, systematically improving the factual accuracy of LLMs.

## Background & Motivation

**Background**: LLMs may generate fluent but factually incorrect content (hallucinations), even when the model internally possesses the correct knowledge. This "know-tell gap" seriously undermines reliability. Recent work has attempted to introduce uncertainty signals during training to align factuality.

**Limitations of Prior Work**: (1) Existing methods directly insert numerical uncertainty scores into QA prompts (e.g., "Conf: 0.833"), which lacks semantic richness and is difficult for LLMs to utilize; (2) The use of binary reward functions (correct/incorrect) ignores answer confidence, potentially encouraging "guessing" behavior; (3) Neglect of external knowledge usage fails to correct potentially erroneous answers.

**Key Challenge**: Factual inconsistency in LLMs manifests as follows—the same question may yield correct or incorrect answers under different phrasings. The root cause is a lack of alignment between the model's knowledge possession state (whether it truly knows) and its answering behavior (whether it expresses honestly). Numerical uncertainty signals do not help the model understand its own knowledge boundaries.

**Goal**: Design a post-training framework that transforms uncertainty signals into semantically rich descriptions of knowledge states, enhancing factual accuracy and truthfulness through fine-grained rewards and external knowledge retrieval.

**Key Insight**: Categorize the LLM's knowledge state for each question into four quadrants—KH (Known and Honest), K¬H (Known but Dishonest), ¬KH (Unknown but Honest), and ¬K¬H (Unknown and Dishonest). These states are described in natural language and embedded into the training prompts.

**Core Idea**: Replace numerical uncertainty with natural language knowledge states $\rightarrow$ utilize fine-grained rewards considering both correctness and uncertainty $\rightarrow$ deploy a RAG module to correct weakly grounded answers.

## Method

### Overall Architecture
FAITH consists of three modules: (1) Data Augmentation—sampling multiple answers, estimating uncertainty, and mapping to knowledge state quadrants to enhance training data; (2) Model Training—Reference model SFT $\rightarrow$ Reward Model training $\rightarrow$ PPO policy optimization $\rightarrow$ RAG model training; (3) Inference—Knowledge state estimation $\rightarrow$ Policy model generation $\rightarrow$ RAG correction.

### Key Designs

1.  **Knowledge State Quadrant Mapping**:
    - **Function**: Transforms numerical uncertainty signals into semantically rich natural language knowledge state descriptions.
    - **Mechanism**: For each question $x_i$, $K=6$ answers are sampled to calculate consistency $\text{Consistency}(x_i) = \frac{1}{K}\sum \mathbb{1}\{y_i^k = \hat{y_i}\}$ and semantic entropy $SE(x_i) = -\sum_c p(c|x_i) \log p(c|x_i)$. Consistency $>0$ and $SE=0 \rightarrow$ KH; Consistency $>0$ and $SE \neq 0 \rightarrow$ K¬H; Consistency $=0$ and $SE=0 \rightarrow$ ¬KH; otherwise $\rightarrow$ ¬K¬H. State descriptions are embedded into training prompts using natural language.
    - **Design Motivation**: Raw numerical values ("Conf: 0.833") are opaque to LLMs—the model cannot infer whether it "knows" or "should answer" from them. Natural language descriptions provide semantically rich and interpretable guidance, allowing the model to better identify its knowledge boundaries.

2.  **Fine-grained Reward Function**:
    - **Function**: Provides more informative feedback than binary rewards by considering both answer correctness and model uncertainty.
    - **Mechanism**: $R_{\text{FAITH}} = R_{\text{correctness}} + R_{\text{uncertainty}}$, where the correctness reward is standard exact matching, and the uncertainty reward is assigned based on the knowledge state: KH $\rightarrow$ +2, K¬H $\rightarrow$ +1, ¬KH $\rightarrow$ -1, ¬K¬H $\rightarrow$ -2. The total reward ranges from -2 to +3.
    - **Design Motivation**: Binary rewards only focus on correctness and do not distinguish between "confidently correct" and "guessing correctly." Fine-grained rewards encourage the model to answer confidently when it knows the answer (KH $\rightarrow$ +2) and refuse honestly when uncertain (¬KH), while punishing incorrect answers generated without knowledge (¬K¬H $\rightarrow$ -2).

3.  **RAG Error Correction Module**:
    - **Function**: Uses external knowledge to correct weakly grounded answers produced by the policy model.
    - **Mechanism**: A vector database based on Wikipedia corpora is constructed. A RAG model is trained to retrieve relevant passages as context to correct potentially incorrect answers from the policy model. The RAG model runs after the policy model as a final factuality verification layer.
    - **Design Motivation**: Even after PPO training, the model may still provide incorrect answers for questions where internal knowledge is lacking (¬K states). RAG provides an external knowledge source to compensate for internal knowledge deficits.

### Loss & Training
Four-stage training: (1) SFT to train the reference model $\pi_\mu$; (2) Reward model training; (3) PPO optimization of the policy model; (4) RAG model training. Validated on Llama3-8B and Mistral-7B-v0.1. NQ-Open, SciQ, and TriviaQA are used for in-domain training, while PopQA is used for out-of-domain testing.

## Key Experimental Results

### Main Results (Llama3-8B)

| Method | In-domain Acc | In-domain Truthfulness | Out-of-domain Acc | Out-of-domain Truthfulness |
| :--- | :--- | :--- | :--- | :--- |
| Baselines Avg | Low | Low | Low | Low |
| UAlign | Medium | Medium | Medium | Medium |
| **Ours** (FAITH) | **74.26%** | **45.73%** | **67.99%** | **34.03%** |

### Ablation Study

| Configuration | Accuracy | Truthfulness | Description |
| :--- | :--- | :--- | :--- |
| Full FAITH | Optimal | Optimal | Complete framework |
| w/o Knowledge State (Numerical) | Decrease | Decrease | Advantage of natural language descriptions |
| w/o Uncertainty Reward (Binary) | Decrease | Decrease | Necessity of fine-grained rewards |
| w/o RAG | Decrease | Decrease | Contribution of external knowledge correction |

### Key Findings
- Natural language knowledge state descriptions consistently outperform numerical uncertainty signals across two models and four datasets.
- Fine-grained rewards provide better learning signals than binary rewards, reducing inflated performance from "guessing correctly."
- The RAG module contributes significantly to out-of-domain generalization by compensating for gaps in internal model knowledge.
- FAITH outperforms five strong baselines on both Llama3-8B and Mistral-7B, proving cross-model generalizability.

## Highlights & Insights
- **Semanticization of Knowledge State Quadrants**: Converting opaque uncertainty values into natural language descriptions like "Known and Honest/Dishonest" allows LLMs to understand their knowledge boundaries during training. This aligns better with LLM capabilities than learning the meaning of arbitrary values like "0.833."
- **Distinguishing "Confidently Correct" from "Guessed Correctly"**: Punishing low-confidence correct answers via uncertainty rewards reduces hollow performance gains. This is crucial for high-risk scenarios such as medical or legal applications.
- **Three-tier Factuality Assurance**: Knowledge state guidance (self-awareness) $\rightarrow$ fine-grained rewards (correct behavior) $\rightarrow$ RAG correction (external verification) forms a complete factuality assurance chain.

## Limitations & Future Work
- Knowledge state estimation requires sampling multiple answers (K=6) per question, incurring additional overhead during inference.
- The four-quadrant division might be oversimplified; actual knowledge states may be continuous rather than discrete.
- The RAG module relies on Wikipedia, which may lack coverage for the latest knowledge or specialized domains.
- The computational cost of PPO training is high, making it potentially unsuitable for extremely large-scale models.

## Related Work & Insights
- **vs. UAlign**: UAlign uses numerical uncertainty and binary rewards directly in the prompt. FAITH uses natural language states and fine-grained rewards to provide richer signals.
- **vs. R-Tuning**: R-Tuning teaches the model to refuse to answer when uncertain but does not distinguish fine-grained differences between knowledge possession and answering behavior.
- **vs. Self-Consistency**: Self-consistency detects uncertainty through multiple samplings; FAITH transforms this signal into a training signal rather than using it only for inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of knowledge state quadrants and fine-grained rewards is an insightful design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, two models, five baselines, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework description and complete formalization.
- **Value**: ⭐⭐⭐⭐ Provides a practical and interpretable method for LLM factuality alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Permutation-Consensus Listwise Judging for Robust Factuality Evaluation](permutation-consensus_listwise_judging_for_robust_factuality_evaluation.md)
- [\[ACL 2026\] CAP: Controllable Alignment Prompting for Unlearning in LLMs](cap_controllable_alignment_prompting_for_unlearning_in_llms.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ACL 2026\] Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding](context-fidelity_boosting_enhancing_faithful_generation_through_watermark-inspir.md)

</div>

<!-- RELATED:END -->
