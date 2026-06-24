---
title: >-
  [Paper Note] Personalized Text Generation with Contrastive Activation Steering
description: >-
  [ACL 2025][Text Generation][Personalized Generation] StyleVector is proposed as a training-free framework for personalized text generation. It extracts a "style vector" by contrasting the hidden layer activation differences between real user responses and style-free model generations. During inference, a simple linear activation intervention steers the LLM to generate text conforming to the user's writing style. It achieves an 8% relative improvement on the LaMP and LongLaMP…
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Personalized Generation"
  - "Style Vector"
  - "Contrastive Activation Steering"
  - "Training-free Framework"
  - "LoRA Alternative"
date: 2026-05-08
content_hash: 7b70a425fc5b21cc
---

# Personalized Text Generation with Contrastive Activation Steering

**Conference**: ACL 2025  
**arXiv**: [2503.05213](https://arxiv.org/abs/2503.05213)  
**Code**: Not released  
**Authors**: Jinghao Zhang, Yuting Liu, Wenjie Wang, Qiang Liu, Shu Wu, Liang Wang, Tat-Seng Chua  
**Affiliations**: Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; Northeastern University; University of Science and Technology of China; National University of Singapore  
**Area**: Personalized Text Generation / Activation Engineering  
**Keywords**: Personalized Generation, Style Vector, Contrastive Activation Steering, Training-free Framework, LoRA Alternative  

## TL;DR

StyleVector is proposed as a training-free framework for personalized text generation. It extracts a "style vector" by contrasting the hidden layer activation differences between real user responses and style-free model generations. During inference, a simple linear activation intervention steers the LLM to generate text conforming to the user's writing style. It achieves an 8% relative improvement on the LaMP and LongLaMP benchmarks while reducing storage requirements to 1/1700 of PEFT methods.

## Background & Motivation

**Background**: LLMs are "one-size-fits-all" systems optimized for the average user, failing to adapt to individual stylistic preferences. Personalized text generation aims to infer writing styles from a user's historical text to generate stylistically consistent outputs.

**Limitations of Prior Work**:
   - **RAG Methods**: Retrieve relevant historical texts as context. Limitations: (a) content semantics and style patterns are entangled — retrieval based on semantic matching leads to style dilution; (b) retrieval latency scales with the amount of history.
   - **PEFT Methods (e.g., LoRA)**: Train independent adapters for each user. Limitations: (a) style-content entanglement still exists; (b) requires storing independent parameter files per user (~17MB/user); (c) high latency in loading and merging LoRA adapters.

**Key Insight**: Research in activation engineering indicates that LLMs encode features and concepts as linear directions in the hidden activation space. Through contrastive analysis, user-specific writing styles can similarly be represented as directional vectors within the activation space.

## Method

### Overall Architecture (StyleVector)

Three-stage pipeline: **Generate Style-Free Response** $\rightarrow$ **Extract Style Vector** $\rightarrow$ **Activation Steering Generation**

### Stage 1: Generate Style-Free Response

Given user $u$'s historical interactions $P_u = \{(x_i, y_i)\}$, a general LLM $M_g$ is used to generate a style-free response $\hat{y}_i = M_g(x_i)$ for each input $x_i$.

- $y_i$ (User real response) = Content semantics + User style
- $\hat{y}_i$ (Model generated response) = Content semantics (without user style)
- $M_g$ can be any model (open-source or closed-source); experiments demonstrate that the method is robust to the choice of $M_g$.

### Stage 2: Extract Style Vector

The style vector is extracted by contrasting hidden layer activations. Let $h_\ell(r)$ be the hidden state of the last token when the $\ell$-th layer processes text $r$:

- **Positive Activation**: $a_{p,i}^{\ell} = h_{\ell}(x_i \oplus y_i)$ (concatenating input and user real response)
- **Negative Activation**: $a_{n,i}^{\ell} = h_{\ell}(x_i \oplus \hat{y}_i)$ (concatenating input and style-free response)

Three extraction functions $f(\cdot)$ are provided to compute the style vector $s_u^{\ell}$:

**1) Mean Difference**:
$$s_u^{\ell} = \frac{1}{|P_u|} \sum_{i=1}^{|P_u|} (a_{p,i}^{\ell} - a_{n,i}^{\ell})$$
The simplest and most direct method — computing the average difference direction of positive and negative activations.

**2) Logistic Regression**:
Employs logistic regression to find the optimal direction $w$ that separates positive and negative samples, and normalizes it to serve as the style vector: $s_u^{\ell} = w / \|w\|_2$

**3) PCA Method**:
Performs PCA on the difference vectors $\{\Delta_i\} \cup \{-\Delta_i\}$ and extracts the first principal component direction:
$$s_u^{\ell} = \arg\max_{v:\|v\|=1} \sum_{i=1}^{|P_u|} (\Delta_i^T v)^2$$

### Stage 3: Activation Steering Generation

During inference, the scaled style vector is added to the hidden layer activations:
$$h'_{\ell}(x)_t = h_{\ell}(x)_t + \alpha \cdot s_u^{\ell}$$

- $\alpha$ is the scaling factor controlling the steering intensity.
- Intervention is performed at each token position $t \ge |x|$ of the generated text (only intervening at a single layer $\ell$).
- $\ell$ and $\alpha$ are selected via the validation set.

### Efficiency Analysis

| Metric | RAG | PEFT | StyleVector |
|------|-----|------|-------------|
| Preprocessing / User | O(\|P_u\|)* | O(\|P_u\|) | O(\|P_u\|)* |
| Inference Latency / Query | O(\|P_u\|) | O(Load+Merge) | **O(1)** |
| Storage / User | O(\|P_u\|·D) | O(r·D·L) | **O(D)** |

*Denoted as "training-free", requiring only forward passes.

## Experiments

### Experimental Setup

- **Benchmarks**: LaMP (short-text personalization) + LongLaMP (long-text personalization)
- **Base Model**: LLaMA-2-7B-chat
- **Metrics**: ROUGE-L, METEOR
- **Baselines**: Non-personalized / BM25-RAG / Contriever-RAG / SFT-LoRA / DPO-LoRA

### Main Results

| Task | Metric | Non-personalized | BM25 | Contriever | SFT | DPO | **StyleVector** | Gain |
|------|------|----------|------|------------|-----|-----|----------------|------|
| Summarization | ROUGE-L | 0.206 | 0.202 | 0.204 | 0.204 | 0.202 | **0.206** | 0.2% |
| Topic Writing | ROUGE-L | 0.130 | 0.124 | 0.126 | 0.130 | 0.128 | **0.136** | 4.7% |
| Review Generation | ROUGE-L | 0.138 | 0.139 | 0.139 | 0.136 | 0.132 | **0.145** | 5.0% |
| Review Generation | METEOR | 0.161 | 0.166 | 0.166 | 0.157 | 0.145 | **0.180** | 11.8% |
| Academic Title | ROUGE-L | 0.109 | 0.091 | 0.092 | 0.110 | 0.105 | **0.137** | **25.8%** |
| Tweet Paraphrasing | ROUGE-L | 0.251 | 0.255 | 0.257 | 0.234 | 0.220 | **0.283** | **12.8%** |

**Key Findings**:
- StyleVector achieves the best or tied-best performance across all tasks, with an average ROUGE-L improvement of ~11% and METEOR improvement of ~8%.
- The improvements are most prominent in Academic Title (25.8%) and Tweet Paraphrasing (12.8%) tasks, where individual stylistic patterns are more pronounced.
- Both RAG and PEFT methods exhibit instability — RAG performs better in tasks with limited history, whereas PEFT excels in tasks with more history.

### Efficiency Comparison

| Metric | SFT | RAG | StyleVector |
|------|-----|-----|-------------|
| Preprocessing Time / User | 62-132s | 0.4-1.2s | 11-27s |
| Inference Latency / Query | 19-26s | 8-19s | **10-16s** |
| Storage / User | **17MB** | 0.1-0.8MB | **0.01MB** |

- Storage requirement is only **1/1700** of SFT (0.01MB vs. 17MB/user).
- Inference latency is comparable to RAG, but does not scale with the size of user history.

### Layer Analysis and Intensity Analysis

- **Intervention Layer Selection**: Mid-to-late layers (around layer 15 and beyond) are the most effective. Stylistic information is progressively refined during the forward pass and achieves maximum linear separability in the higher layers.
- **Intervention Intensity $\alpha$**: Positive values steer towards the user's style, while negative values steer away from it (falling below the non-personalized baseline). An excessively large $\alpha$ disrupts the generation process.

### Linear Probing Analysis

- All layers achieve AUC > 0.85, indicating that stylistic patterns are robustly encoded throughout the network.
- The AUC increases as the layer depth grows, consistent with the empirical findings on intervention layer selection.

### Case Study (user_310 News Title Generation)

- The style vector encodes user preferences: the top-5 matching tokens (":", "ips", "for", "What", "Need") reveal the user's habit of using subtitles and the "tips for" combination.
- The generated title "Keeping Your Teen Safe Online: Tips and Strategies for Parents" naturally incorporates 3 of these style tokens.
- **Significant Discrepancy Between Style and Semantic Rankings**: Semantically similar documents retrieved by RAG fail to provide sufficient style information, demonstrating the necessity of style-content decoupling.

### Style Transfer Experiments

When using GPT to rewrite the user's history into specific styles (such as "exclamatory tone + exclamation mark" or "removing colons and subtitles"), recalculating the style vector indeed steers the generation toward the corresponding style while maintaining semantic fidelity.

## Highlights & Insights

1. **Theoretical Contribution**: This work is the first to reveal that user-specific writing styles can be represented as linear directions in the LLM's activation space, bridging activation engineering and personalized generation.
2. **Extreme Storage Efficiency**: Each user requires only a single $D$-dimensional vector (~0.01MB), which is 1700 times smaller than LoRA.
3. **Completely Training-Free**: Requires only $2|P_u|$ forward passes (zero backpropagation).
4. **Style-Content Decoupling**: Decoupling is naturally achieved through contrastive analysis. The case study demonstrates the discrepancy between style and semantic rankings.
5. **$O(1)$ Inference Latency**: The intervention only requires a $D$-dimensional element-wise addition, which does not scale with the size of user history.

## Limitations & Future Work

1. The current contrastive method relies on the model's internal capability to separate style and content, which might not yield optimal decoupling.
2. A single vector representation might conflate multiple stylistic dimensions (lexical preference, syntactic structure, discourse patterns). Future work could leverage sparse combinations for finer-grained control.
3. Evaluation benchmarks assume domain homogeneity in user historical data, lacking evaluation for cross-domain style consistency.
4. Only validated on LLaMA-2-7B; generalization to larger or newer models remains to be verified.

## Related Work & Insights

- **Personalized Text Generation**: RAG paradigm (Zhang et al., 2023; Salemi & Zamani, 2024) $\rightarrow$ PEFT paradigm (per-user LoRA) $\rightarrow$ StyleVector (activation space vector)
- **Activation Engineering**: Zou et al. (2023) discovered the linear representation hypothesis $\rightarrow$ Turner et al. (2023) proposed Activation Addition $\rightarrow$ Rimsky et al. (2024) quality mean difference $\rightarrow$ Zhang et al. (2024a) truthfulness head $\rightarrow$ This work is the first to apply it to personalized writing style.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐⭐ — Applying activation engineering to personalized text generation is a completely new direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 6 tasks, comparison with multiple baselines, efficiency analysis, layer analysis, case study, and style transfer.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations and intuitive framework diagrams.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free + extremely low storage + $O(1)$ inference latency, deployment-friendly.
- **Limitations**: Only validated on a single base model, and cross-domain scenarios are not covered.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries](dpp_diverse_multidoc_summary.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)
- [\[ACL 2025\] Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)
- [\[ACL 2025\] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](dehumanizing_machines_anthropomorphic.md)
- [\[ACL 2025\] CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](cocolex_legal_text_gen.md)

</div>

<!-- RELATED:END -->
