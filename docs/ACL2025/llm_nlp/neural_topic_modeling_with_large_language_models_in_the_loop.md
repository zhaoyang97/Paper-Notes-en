---
title: >-
  [Paper Note] Neural Topic Modeling with Large Language Models in the Loop
description: >-
  [ACL2025][LLM (Other)][LLM-ITL] Proposed the LLM-ITL framework, which integrates LLMs in an "in-the-loop" manner into Neural Topic Model (NTM) training. Using an optimal transport-based topic alignment objective and a confidence weighting mechanism, it significantly improves topic interpretability while maintaining document representation quality and computational efficiency.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "LLM-ITL"
  - "Neural Topic Model"
  - "Optimal Transport"
  - "Topic Refinement"
  - "Confidence Weighting"
date: 2026-05-08
content_hash: b70f39ed1cd1be41
---

# Neural Topic Modeling with Large Language Models in the Loop

**Conference**: ACL2025  
**arXiv**: [2411.08534](https://arxiv.org/abs/2411.08534)  
**Code**: [GitHub](https://github.com/Xiaohao-Yang/LLM-ITL)  
**Area**: Topic Modeling / LLM Enhancement / Optimal Transport  
**Keywords**: LLM-ITL, Neural Topic Model, Optimal Transport, Topic Refinement, Confidence Weighting  

## TL;DR

Proposed the LLM-ITL framework, which integrates LLMs in an "in-the-loop" manner into Neural Topic Model (NTM) training. Using an optimal transport-based topic alignment objective and a confidence weighting mechanism, it significantly improves topic interpretability while maintaining document representation quality and computational efficiency.

## Background & Motivation

### Background
Topic modeling is a fundamental task in NLP, used to discover latent topic structures in text collections. Neural Topic Models (NTMs) leverage deep neural networks to model document-topic distributions, but the generated topic word lists often lack semantic coherence and interpretability.

### Limitations of Prior Work

**Pure NTM methods**: Topic words can be overly general, specific, or semantically vague, making them difficult for users to comprehend.

**Direct LLM modeling**: Invoking LLMs for every document in the corpus presents three major issues:
   - Inability to cover global topics of the corpus (focusing only on a single document at a time)
   - Poor performance in long-document multi-topic scenarios
   - Extremely high computational cost and poor scalability

**Post-processing methods**: Refining topic words with LLMs after training cannot optimize topic quality during the training process.

### Core Idea
Transforming the LLM from a "post-processing tool" or "dominant driver" into an "in-the-loop assistant." LLM feedback is introduced at the word level (rather than document level) during NTM training, significantly reducing computational costs.

## Method

### Overall Architecture

LLM-ITL consists of three key components: LLM topic suggestions, OT distance-based topic alignment, and confidence-weighted refinement.

1. The NTM component learns global topics and document representations.
2. Following a warm-up phase, the LLM suggests better topic words for each topic learned by the NTM.
3. An OT-based alignment objective drives NTM topics to align with LLM suggestions.
4. A confidence mechanism dynamically adjusts the LLM's influence.

### NTM Preliminaries

NTMs are based on the VAE framework, maximizing the ELBO:
$$\max_{\theta, \phi} \left(\mathbb{E}_{q_\theta(\mathbf{z}|\mathbf{x})}[\log p_\phi(\mathbf{x}|\mathbf{z})] - \text{KL}[q_\theta(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})]\right)$$

The $k$-th topic distribution is obtained by normalizing the $k$-th column of the decoder's weight matrix $\phi \in \mathbb{R}^{V \times K}$:
$$\mathbf{t}_k = \text{softmax}(\phi_{:,k})^T$$

The topic words $\mathbf{w}_k$ represent the top $N$ words with the highest weights in $\mathbf{t}_k$.

### 1. LLM Topic Suggestion

For the top word list $\mathbf{w}$ of each topic, CoT prompting is used to guide the LLM to generate:
- **Topic Label** $\mathbf{w}^l$: A concise summary of the topic
- **Refined Topic Words** $\mathbf{w}'$: Words that better represent the topic's semantics

$$\mathbf{s} = \theta^{\text{llm}}(\text{Prompt}(\mathbf{w}))$$

Refined words must be filtered to ensure they do not contain out-of-vocabulary (OOV) words. Chain-of-Thought prompting is used to guide the LLM through step-by-step reasoning, first identifying irrelevant words (intruders) before providing the label and suggested words.

### 2. Topic Alignment via Optimal Transport

**Why OT?** OT measures the semantic distance between two word distributions, accounting for word-to-word similarity (unlike simple KL divergence).

Given the original topic words $\mathbf{w}$ (with probability $\mathbf{t}$) and the LLM-refined words $\mathbf{w}'$ (with uniform distribution $\mathbf{u}$), the OT distance is:
$$d_{\text{OT}}(\mu(\mathbf{w}, \mathbf{t}), \mu(\mathbf{w}', \mathbf{u})) = \min_{\mathbf{P}} \sum_{i=1}^N \sum_{j=1}^M C_{i,j} P_{i,j}$$

The cost matrix $C$ is constructed using the cosine distance of pre-trained word embeddings (GloVe):
$$C_{i,j} = d_{\cos}(\mathbf{e}^{w_i}, \mathbf{e}^{w'_j})$$

By minimizing the OT distance, the topic word distribution learned by the NTM is aligned towards the direction of the LLM suggestions.

### 3. Confidence-Weighted Refinement

LLMs might hallucinate (providing irrelevant suggestions), requiring influence regulation based on confidence.

**Method 1: Label Token Probability (applicable to open-source LLMs)**
$$\text{Conf}(\mathbf{w}^l)^{\text{prob}} = \prod_{i=\text{sol}}^{\text{eol}} p(s_i | \mathbf{s}_{<i}, \mathbf{c})$$

**Method 2: Word Intrusion Confidence (applicable to all LLMs)**
$$\text{Conf}(\mathbf{w}^l)^{\text{intrusion}} = 1 - \frac{N^{\text{intruder}}}{N^{\mathbf{w}}}$$

A higher number of intruders identified by the LLM indicates a more incoherent original topic, resulting in a lower confidence score for the LLM annotation.

### Loss & Training

Combining the NTM loss and the confidence-weighted OT refinement loss, with warm-up:
$$\min_\Theta \left(\mathcal{L}^{\text{ntm}} + \gamma \cdot \mathbf{I}(t > T^{\text{refine}}) \cdot \mathcal{L}^{\text{refine}}\right)$$

where $\mathcal{L}^{\text{refine}} = \sum_{k=1}^K \text{Conf}(\mathbf{w}_k^l) \cdot d_{\text{OT}}(\mu(\mathbf{w}_k, \mathbf{t}_k), \mu(\mathbf{w}'_k, \mathbf{u}_k))$.

The warm-up phase allows the NTM to first learn stable corpus representations, avoiding over-reliance on LLM knowledge.

## Key Experimental Results

### Experimental Settings
- **Datasets**: 20Newsgroup (20News), Reuters-21578 (R8), DBpedia, AGNews
- **Number of Topics**: $K=50$ for long documents (20News, R8), and $K=25$ for short documents (DBpedia, AGNews)
- **LLM**: LLAMA3-8B-Instruct
- **OT Implementation**: GloVe word embeddings + POT package
- **Baselines**: Integration of 8 commonly used NTMs + LDA, BERTopic, TopicGPT
- **Evaluation Metrics**: $C_V$ (topic coherence), PN (normalized mutual information, document representation quality)
- **Hardware**: A single 80GB A100 GPU

### Main Results

| Baseline NTM | Original $C_V$ → +LLM-ITL | Gain |
|---|---|---|
| NVDM | 0.261 → 0.336 | ↑28.7% |
| PLDA | 0.368 → 0.525 | ↑42.7% |
| SCHOLAR | 0.479 → 0.591 | ↑23.4% |
| ETM | 0.491 → 0.578 | ↑17.7% |
| NSTM | 0.444 → 0.521 | ↑17.3% |
| CLNTM | 0.490 → 0.612 | ↑24.9% |
| WeTe | 0.495 → 0.583 | ↑17.8% |

Key Findings:
1. **Significant Improvement in Topic Coherence**: The $C_V$ of all 8 NTMs improved significantly after integrating LLM-ITL, with an average increase of approximately 24%.
2. **Preserved Document Representation Quality**: The PN metric remains largely unchanged (within ±3%), indicating that LLM refinement does not harm the quality of document representations.
3. **Outperforming LDA and BERTopic**: After integrating LLM-ITL, most NTMs outperform traditional methods in terms of $C_V$.
4. **Significantly Better than TopicGPT**: This demonstrates that the LLM-in-the-loop paradigm is superior to pure LLM-based topic modeling.
5. **Efficiency Advantage**: Word-level LLM invocations are far fewer than document-level ones (whereas TopicGPT requires invoking the LLM for every single document).

### Ablation Study

| Component | Effect of Removal |
|------|----------|
| OT Alignment | Significant drop in $C_V$, verifying the critical role of OT |
| Confidence Weighting | Quality degradation, especially on incoherent topics |
| Warm-up Phase | Drop in document representation quality, indicating warm-up prevents LLM bias |
| Different LLMs | LLAMA3-8B > LLAMA2-7B, indicating stronger LLMs provide better suggestions |

## Highlights & Insights

1. **Paradigm Innovation**: Shifts LLMs from being a "replacement" to an "assistant," introducing LLM feedback directly into the NTM training loop, thereby maintaining the efficiency advantages of NTMs.
2. **Incredible Design of OT Alignment**: Employs optimal transport to measure differences in topic word distributions, which captures word-level semantic relationships better than simple KL divergence.
3. **Hallucination Prevention via Confidence Mechanism**: Offers two confidence calculation methods suited for open-source and closed-source LLMs, demonstrating strong practical utility.
4. **Modular Design**: LLM-ITL can be easily plugged into any VAE-based NTM without modifying the underlying model architecture.
5. **Cost-Effective Word-Level Queries**: LLM queries are only made for the word lists of $K$ topics (instead of $N$ documents), keeping computational overhead highly manageable.

## Limitations & Future Work

1. Reliance on the quality of the LLM's CoT reasoning; weaker LLMs may produce lower-quality suggestions.
2. The duration of the warm-up phase and refinement intensity $\gamma$ require hyperparameter tuning.
3. OT computation can become a bottleneck when the vocabulary size is extremely large.
4. Evaluated only on English corpora, leaving multilingual scenarios to be explored.
5. GloVe word embeddings might be less precise compared to contextual embeddings.

## Related Work & Insights

- **Traditional Topic Models**: LDA and its hierarchical Bayesian extensions
- **Neural Topic Models**: VAE-based (NVDM, ETM, SCHOLAR), Clustering-based (BERTopic)
- **LLM-assisted Topic Modeling**: TopicGPT (pure LLM topic generation), ChatGPT-generated topic descriptions
- **Optimal Transport**: Word Mover's Distance, Sinkhorn distances
- **LLM Uncertainty Estimation**: Token probability calibration

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Evaluation | ⭐⭐⭐⭐ |

> This is a meticulously designed methodology paper. The core idea—leveraging LLMs to provide word-level feedback in the NTM training loop—is both efficient and effective. The three components (OT alignment objective, confidence weighting, and warm-up strategy) are closely interconnected. The experiments cover 8 NTM baselines across 4 datasets, fully validating the framework's universality and effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)
- [\[ACL 2025\] Information Locality as an Inductive Bias for Neural Language Models](information_locality_as_an_inductive_bias_for_neural_language_models.md)
- [\[ACL 2025\] Enhancing Spoken Discourse Modeling in Language Models Using Gestural Cues](enhancing_spoken_discourse_modeling_in_language_models_using_gestural_cues.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)

</div>

<!-- RELATED:END -->
