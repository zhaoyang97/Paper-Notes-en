---
title: >-
  [Paper Note] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing
description: >-
  [ACL 2025][Knowledge Editing][LLM Detoxification] ToxEdit is proposed—a toxicity-aware knowledge editing method that detects harmful hidden states in the early layers of LLM forward propagation using an SVM classifier. Through a routing mechanism, harmful inputs are directed to edited FFN replicas, while harmless inputs follow the original FFN. This achieves nearly 98% detoxification success rate and 95% instruction-following retention (DL metric) on LLaMA3-8B/LLaMA2-7B/Mistr…
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "LLM Detoxification"
  - "Over-editing"
  - "Adaptive Routing"
  - "Toxicity Detection"
date: 2026-05-08
content_hash: eae308b07c180404
---

# ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing

**Conference**: ACL 2025  
**arXiv**: [2505.22298](https://arxiv.org/abs/2505.22298)  
**Code**: None  
**Area**: LLM Safety / Knowledge Editing  
**Keywords**: LLM Detoxification, Knowledge Editing, Over-editing, Adaptive Routing, Toxicity Detection

## TL;DR

ToxEdit is proposed—a toxicity-aware knowledge editing method that detects harmful hidden states in the early layers of LLM forward propagation using an SVM classifier. Through a routing mechanism, harmful inputs are directed to edited FFN replicas, while harmless inputs follow the original FFN. This achieves nearly 98% detoxification success rate and 95% instruction-following retention (DL metric) on LLaMA3-8B/LLaMA2-7B/Mistral-7B, resolving the key challenge of "detoxification vs. over-editing" in knowledge editing detoxification.

## Background & Motivation

**Background**: Although LLMs are aligned for safety through RLHF/DPO, they can still be induced to generate harmful content by malicious prompts and jailbreak attacks. Knowledge editing techniques can quickly modify a small number of parameters for detoxification without large-scale retraining, showing promising applications.

**Limitations of Prior Work**: (1) **Failure in entity localization**—Existing knowledge editing methods (such as ROME/MEMIT) rely on specific entity localization to edit specific areas, but adversarial inputs often lack explicit entities, making localization ineffective. (2) **Severe over-editing**—Edited models tend to reject legitimate queries (e.g., "how to aim more accurately with a slingshot"), violating the "helpfulness" principle. Methods like DINM produce meaningless character loops or reject normal requests after detoxification.

**Key Challenge**: There is a trade-off between detoxification capability and general capability preservation—more aggressive editing results in better detoxification but larger losses in general capability, and vice versa.

**Goal**: To achieve adaptive detoxification—rejecting responses to harmful inputs while maintaining original capabilities for normal inputs, without post-editing interference.

**Key Insight**: Leveraging the finding that hidden states in the early layers of LLMs already exhibit different patterns for toxic and non-toxic inputs, a binary classifier is used to detect toxicity, selectively routing the input through either the edited or original path.

**Core Idea**: Embed a "toxicity-detection -> routing-diversion" mechanism inside the LLM. Harmful inputs go through the edited FFN for detoxification, while harmless inputs go through the original FFN to preserve capability.

## Method

### Overall Architecture

ToxEdit consists of two modules: (1) a semantic profiling toxicity detection module, which uses an SVM in the early layers of the LLM to determine input toxicity; (2) an anti-toxic FFN module, which duplicates the $W^V$ matrix of the target FFN layer for detoxification editing and dynamically routes the input through either FFN using a router.

### Key Designs

1. **Semantic Profiling for Toxicity Detection**:

    - **Function**: Real-time detection of input toxicity during LLM forward propagation.
    - **Mechanism**: Formulate toxicity detection as a binary classification problem. Extract the hidden state $h_l^{(n)}$ at the last position of the $l$-th layer, and feed it into a linear SVM classifier to output +1 (harmful) or -1 (harmless): $R_l = classifier_\sigma(h_l^{(n)})$
    - Training Data: 4000 harmful prompts (malicious single-turn questions + constructed jailbreak prompts) + 2000 harmless prompts, constructed from the SafeEdit training set. A system safety prefix $S$ is prepended to each prompt.
    - Optimal Layer Selection: Iterate through all layers to train the classifier and select the layer $l'$ with the highest F1 score. Research shows that layers 10-15 perform best (F1 close to 1), suggesting that intermediate layers effectively associate toxic content with refusal intentions.
    - **Design Motivation**: LLMs have already learned to distinguish ethical concepts of harmful/harmless inputs during the pre-training stage, which is reflected in the distribution differences of hidden states in early layers.

2. **Anti-Toxic Feed-Forward Module**:

    - **Function**: Construct a dedicated FFN replica for detoxification, achieving adaptive detoxification through routing.
    - **Mechanism**:
        - Duplicate the $W_{l'}^V$ (the second MLP layer of the FFN) of the target layer as the editing replica.
        - Edit the replica for $T$ steps using harmful prompts $P$ paired with safe responses $Y_{safe}$, with the loss function: $\mathcal{L} = -\log P_{\mathcal{W}^t}(Y_{safe}|[S;P])$
        - Freeze all other LLM parameters and only update the replica.
        - Routing mechanism: If detected as harmful $\rightarrow h_{l'+1} = h_{l'}^{down} W_{l'}^{V*}$ (edited FFN); if detected as harmless $\rightarrow h_{l'+1} = h_{l'}^{down} W_{l'}^{V}$ (original FFN).
    - **Design Motivation**: Duplicating rather than directly modifying the original parameters ensures that harmless inputs are completely unaffected. The routing mechanism allows the detoxification module to focus solely on the detoxification task, without needing to constrain normal responses in the loss function as in DINM.

3. **SafeEdit Benchmark Enhancement**:

    - **Function**: Add instruction-following evaluation metrics.
    - **Mechanism**: Introduce a new Defense Locality (DL) metric: $DL = \mathbb{E}_{q_n \sim Q_n}\{Sim(f_{W'}([S;q_n]), f_W([S;q_n]))\}$, measuring the response consistency for harmless instructions before and after editing. Meanwhile, adjust the Fluency metric to evaluate the response fluency of safe requests using n-grams.
    - **Design Motivation**: The original SafeEdit only used QA and summarization to evaluate the preservation of general capabilities, but instruction-following is closest to the editing task and most likely to expose over-editing.

## Key Experimental Results

### Main Results (SafeEdit Test Set, 3 LLMs)

| Method | Model | DS(↑) | DG-Avg(↑) | DL(↑) | Fluency(↑) |
|------|------|-------|-----------|-------|------------|
| Vanilla | LLaMA3-8B | 14.82 | 32.97 | - | 7.89 |
| FT-L | LLaMA3-8B | 82.18 | 90.57 | 64.65 | 6.42 |
| DINM | LLaMA3-8B | 82.89 | 99.40 | 3.92 | 1.20 |
| **ToxEdit** | LLaMA3-8B | **97.78** | **98.55** | **95.36** | **8.07** |
| DINM | LLaMA2-7B | 96.02 | 86.74 | 13.55 | 3.43 |
| **ToxEdit** | LLaMA2-7B | **99.55** | **98.68** | **98.02** | **7.56** |
| DINM | Mistral-7B | 81.33 | 73.95 | 66.16 | 6.69 |
| **ToxEdit** | Mistral-7B | **91.63** | **97.96** | **94.62** | **7.22** |

### Ablation Study (LLaMA3-8B-Instruct)

| Configuration | DS | DG-Avg | DL | Note |
|------|-----|--------|-----|------|
| ToxEdit Full | 97.78 | 98.55 | 95.36 | Baseline |
| W/o Toxicity Detection | 98.13 | 99.29 | **6.71** (-88.65) | Detoxification increases slightly but DL collapses—proving routing is key to capability preservation |
| W/o System Safety Prefix | 81.31 | 88.39 | 74.79 | Safety prefix helps the model identify toxicity |
| W/o Jailbreak Sample Training | 95.55 | 84.93 | 78.79 | Jailbreak samples are crucial for generalization |

### Key Findings

- ToxEdit simultaneously achieves optimal performance in both detoxification capability and general capability preservation across all three LLMs—whereas DINM detoxifies well but has a DL of only 3.92% (almost complete loss of instruction-following capability).
- Removing the toxicity detection module causes DL to plummet from 95.36% to 6.71%—proving that adaptive routing is crucial to preventing over-editing.
- The classifier deployed at layers 10-15 performs best, reaching an F1 score close to 1 with only a small amount of training data.
- ToxEdit does not strongly depend on specific training datasets—switching to AdvBench/StrongReject for training still achieves nearly 100% detoxification.

## Highlights & Insights

- Adaptive routing is the core innovation—instead of a one-size-fits-all approach to editing the processing path for all inputs, paths are dynamically selected based on toxicity detection. This fundamentally solves the trade-off dilemma between "detoxification vs. over-editing".
- Using the LLM's own hidden states for toxicity detection (SVM classifier) is a simple and effective design—it leverages the ethical judgment capabilities already learned by the model during pre-training, without requiring an additional large model.
- Case studies visually demonstrate the issue: DINM rejects normal requests with "I'm sorry", and FT-L outputs meaningless character loops; whereas ToxEdit rejects malicious requests and provides helpful responses to normal queries.

## Limitations & Future Work

- The linear SVM classifier might be bypassed by carefully constructed adversarial examples—requiring more robust detection mechanisms.
- Only editing $W^V$ of a single FFN layer—joint editing across multiple layers could further improve detoxification generalization.
- The binary classification (harmful/harmless) assumption is too coarse—toxicity occurs on a continuous spectrum, and fine-grained routing might be more effective.
- Experiments were only validated on the SafeEdit benchmark—performance on other safety evaluation benchmarks (such as ToxiGen, RealToxicityPrompts) remains unknown.

## Related Work & Insights

- **vs. ROME/MEMIT**: Relying on entity localization is not suitable for adversarial inputs without entities; ToxEdit replaces it with toxicity semantic detection.
- **vs. DINM**: Direct editing leads to severe over-editing (DL of only 3.92%); ToxEdit protects normal inputs through a routing mechanism.
- **vs. RLHF/DPO**: Requires massive data and training resources; ToxEdit only needs a small number of editing steps and simple SVM training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The knowledge editing framework combining toxicity detection and adaptive routing is a novel and intuitive contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on 3 LLMs, with complete ablation studies, cross-dataset validation, and case analyses; the benchmark enhancement provides an extra contribution.
- **Writing Quality**: ⭐⭐⭐⭐ The problem definition is clear, the method illustration is intuitive, and the case comparison is compelling.
- **Value**: ⭐⭐⭐⭐⭐ It addresses the core pain point of knowledge editing for detoxification (over-editing), achieving an impressive improvement in DL from 3.92% to 95.36%.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A General Knowledge Injection Framework for ICD Coding](a_general_knowledge_injection_framework_for_icd_coding.md)
- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)
- [\[ACL 2025\] ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains](chainedit_propagating_ripple_effects_in_llm.md)

</div>

<!-- RELATED:END -->
