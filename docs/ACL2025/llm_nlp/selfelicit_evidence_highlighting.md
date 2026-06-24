---
title: >-
  [Paper Note] SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence
description: >-
  [ACL 2025][LLM (Other)][Attention Analysis] SelfElicit discovers that the attention scores in deep layers of LLMs naturally possess the capability to localize key evidence within the context (even when the model generates incorrect answers). Based on this finding, an inference-time context enhancement method is proposed: by generating only one extra token, it automatically identifies and highlights key evidence sentences to guide the model toward generating more accurate answ…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Attention Analysis"
  - "Evidence Localization"
  - "Context Highlighting"
  - "Inference-time Enhancement"
  - "Training-free"
date: 2026-05-08
content_hash: c4236e69b61ba476
---

# SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence

**Conference**: ACL 2025  
**arXiv**: [2502.08767](https://arxiv.org/abs/2502.08767)  
**Code**: [ZhiningLiu1998/SelfElicit](https://github.com/ZhiningLiu1998/SelfElicit)  
**Area**: Context Enhancement, QA  
**Keywords**: Attention Analysis, Evidence Localization, Context Highlighting, Inference-time Enhancement, Training-free  

## TL;DR

SelfElicit discovers that the attention scores in deep layers of LLMs naturally possess the capability to localize key evidence within the context (even when the model generates incorrect answers). Based on this finding, an inference-time context enhancement method is proposed: by generating only one extra token, it automatically identifies and highlights key evidence sentences to guide the model toward generating more accurate answers.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Key Challenge**: Insufficient utilization of contextual evidence: Although providing LLMs with context containing evidence can significantly improve answer quality, recent studies show that LLMs struggle to fully utilize key evidence when the context is noisy and contains irrelevant information, sometimes producing incorrect answers even when the evidence is clearly present in the input.

### Solution

**Goal**: **Background**: Limitations of existing methods: Existing improved prompting and decoding methods treat the entire context as a single entity, ignoring the fact that not all contextual information is equally important.

### Solution

**Proposed Approach**: **Core Finding**: By analyzing the attention distribution across layers when generating the first token in multiple LM families, the authors discover that **attention in deep layers is significantly higher for evidence sentences than for non-evidence sentences** (up to 6 times higher). Notably, this pattern consistently holds true regardless of whether the model's answer is correct or incorrect. This indicates that LLMs inherently possess the capability of evidence localization, which is simply underutilized.

## Method

### Overall Architecture

SelfElicit consists of two steps: (1) **Evidence Discovery**: Automatically locating key evidence sentences in the context using the deep-layer attention scores of the LM; (2) **Evidence Highlighting**: Highlighting the selected evidence sentences in the original context with text markers and modifying the prompt template to guide the model to focus on the highlighted content before regenerating the answer.

### Key Designs

1. **Sentence-level Attention Aggregation**: For $m$ contextual sentences in the input sequence, the sentence-level attention $\bar{a}_i^{(\ell)}$ (the average attention over all tokens within the sentence) is calculated for each layer $\ell$, indicating the relative importance of each sentence at every layer.
2. **Evidence Reading Layer Selection**: The last 50% of the layers are selected as the "Evidence Reading Layers" $\mathcal{L}_{ER}$. The sentence-level attention scores from these layers are aggregated to obtain the evidence score $e_i = \frac{1}{|\mathcal{L}_{ER}|}\sum_{\ell \in \mathcal{L}_{ER}} \bar{a}_i^{(\ell)}$.
3. **Threshold-based Evidence Selection**: A threshold parameter $\alpha \in [0,1]$ (default 0.5) is introduced to select sentences whose evidence scores exceed $\alpha$ times the maximum score: $\mathcal{S}_{SE} = \{s_i | e_i \geq \alpha \cdot \max(\mathbf{e})\}$.
4. **Text Marker Highlighting**: `<start_important>` and `<end_important>` tags are inserted around the selected evidence sentences, while the prompt template is updated to guide the model to focus on the highlighted information.

### Loss & Training

SelfElicit is a training-free, execution-time-only method that **requires no training** and involves no loss function. The only additional computational overhead is generating a single token to retrieve the attention scores.

## Experiments

### Main Results: 6 LMs × 4 QA Tasks

| Model | Method | HotpotQA EM | NewsQA EM | TQA EM | NQ EM | Inference Time (ms) |
|------|------|------------|----------|--------|------|-----------|
| Llama-3.1-8B | Base | 58.9 | 64.3 | 72.8 | 59.7 | 224.1 |
| | CoT | 60.4 | 64.9 | 74.4 | 59.6 | 224.8 |
| | FullElicit | 60.7 | 65.9 | 72.8 | 61.1 | 226.3 |
| | PromptElicit | 66.3 | 62.8 | 76.0 | 61.8 | 1672.0 |
| | **SelfElicit** | **68.5** | **66.9** | **79.4** | **64.0** | **264.1** |
| Llama-3.1-70B | Base | 71.8 | 66.7 | 78.0 | 59.3 | 1389.8 |
| | **SelfElicit** | — | — | — | — | — |

SelfElicit achieves the best or near-best EM and Token F1 across all model-dataset combinations, while incurring minimal inference time overhead (only about an 18% increase compared to Base).

### Ablation Study: Analysis of Design Choices

| Ablation Item | Impact |
|--------|------|
| Selection of Evidence Reading Layers | The last 50% of layers are consistently optimal, while the first 50% of layers perform poorly |
| Selection of Threshold α | α=0.5 consistently performs well across all models and tasks, showing strong robustness |
| Token-level vs. Sentence-level Highlighting | Sentence-level is semantically more complete and yields better results |
| Highlighting Method (Marker tags vs. Bolding vs. Deleting non-evidence) | Text markers perform the best |

### Key Findings

- **Deep attention naturally localizes evidence**: Across multiple model families including Llama, Mistral, and Qwen, the attention in deep layers is consistently and significantly higher for evidence sentences than for non-evidence sentences, even when the model answers incorrectly.
- **Extremely high efficiency**: It only requires generating 1 additional token to retrieve the attention scores, which is about 6 times faster than PromptElicit (which requires the LLM to extract evidence before answering).
- **Robustness to noise**: Even when the context is heavily noisy (with a large number of irrelevant passages), SelfElicit consistently localizes the evidence and improves performance.
- **Evidence discovery precision**: On HotpotQA, SelfElicit's evidence discovery accuracy (recall of supporting facts) exceeds 70% for most models.

## Highlights & Insights

- The core finding is highly inspiring: the deep attention of LMs inherently possesses evidence localization capabilities, independent of whether the final answer is correct or not.
- Extremely simple and efficient method: zero training, zero iterations, and only 1 extra token of overhead.
- Strong generalizability: consistently effective across 6 model families and 4 QA datasets.
- The systematic ablation analysis on highlighting methods provides valuable design guidance for future work.

## Limitations & Future Work

- Although the threshold $\alpha$ has a small impact on performance, it still needs to be preset, lacking complete adaptivity.
- The evaluation is mainly verified on open-domain QA; its effectiveness on other NLG tasks (summarization, dialogue) has not been explored.
- Evidence highlighting relies on the quality of sentence segmentation, and performance may degrade on structured or irregular texts.
- It assumes that relevant evidence is indeed contained within the context, making it inapplicable to scenarios completely lacking evidence (e.g., where external retrieval is required).

## Related Work & Insights

- **Context-Enhanced QA**: Context-Aware Decoding (Shi et al., 2024) enhances performance by treating the entire context as a single entity.
- **KV Cache Compression & Attention Analysis**: H2O (Li et al., 2024a) utilizes attention patterns for KV cache compression.
- **Retrieval-Augmented Generation (RAG)**: RAG methods focus on how to provide relevant evidence to LLMs, whereas SelfElicit focuses on how to make better use of the existing evidence.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IPO: Your Language Model is Secretly a Preference Classifier](ipo_your_language_model_is_secretly_a_preference_classifier.md)
- [\[ACL 2025\] ConceptCarve: Dynamic Realization of Evidence](conceptcarve_dynamic_realization_of_evidence.md)
- [\[ACL 2025\] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_braces_straightening.md)
- [\[ACL 2025\] Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)
- [\[ACL 2025\] Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)

</div>

<!-- RELATED:END -->
