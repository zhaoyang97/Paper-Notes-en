---
title: >-
  [Paper Note] LLMs + Persona-Plug = Personalized LLMs
description: >-
  [ACL 2025][LLM (Other)][Personalized Generation] This paper proposes PPlug, which compresses user historical behavior into a single personalized embedding via a lightweight plug-and-play user embedder to guide LLMs in generating personalized outputs. PPlug significantly outpaces retrieval-based and fine-tuning-based baselines on the LaMP benchmark by up to 35.8%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Personalized Generation"
  - "User Modeling"
  - "Plug-and-play Methods"
  - "User Embeddings"
  - "Language Model Personalization"
date: 2026-05-08
content_hash: 011153f1d5e91064
---

# LLMs + Persona-Plug = Personalized LLMs

**Conference**: ACL 2025  
**arXiv**: [2409.11901](https://arxiv.org/abs/2409.11901)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Personalized Generation, User Modeling, Plug-and-play Methods, User Embeddings, Language Model Personalization

## TL;DR

This paper proposes PPlug, which compresses user historical behavior into a single personalized embedding via a lightweight plug-and-play user embedder to guide LLMs in generating personalized outputs. PPlug significantly outpaces retrieval-based and fine-tuning-based baselines on the LaMP benchmark by up to 35.8%.

## Background & Motivation

The current mainstream usage of LLMs is "one-size-fits-all": providing almost identical responses to the same input from different users, which ignores individual user differences. While personalized LLMs have emerged as an important research direction, existing approaches have respective limitations:

**Fine-tuning-based methods** (e.g., OPPU, per-user LoRA): Training separate LLMs for each individual user incurs extremely high training and inference costs, preventing large-scale deployment.

**Retrieval-based methods** (e.g., RAG-based): Retrieving relevant behaviors from user history as in-context demonstrations, which only focuses on a few history records most related to the current input, easily losing the user's overall style and preference.

**Core Problem**: How can one capture the user's overall style and preference without modifying the parameters of the LLM itself?

## Method

### Overall Architecture

PPlug consists of three core components:
1. **User Behavior Encoder**: Encodes each user historical behavior into a dense vector.
2. **Input-aware Personal Aggregator**: Weighted-aggregates all historical behavior vectors into a personalized embedding based on the current input.
3. **PPlug for LLM Personalization**: Appends the personalized embedding to the LLM input to guide generation.

### Key Designs

**User Behavior Encoder**: Uses an encoder model such as BGE-base to encode each historical behavior $h_i^u$ into a vector $\mathbf{h}_i^u = \text{Enc}^{\text{his}}(h_i^u)$. A small encoder is adopted instead of an LLM for two reasons: (1) bidirectional attention is better suited for capturing behavioral information; (2) the lightweight encoder improves efficiency (with only 220M parameters, accounting for 3.1% of a 7B LLM). The parameters of the historical behavior encoder are frozen and not trained, whereas only the input encoder is fine-tuned.

**Input-aware Personal Aggregator**: Unlike simply averaging all historical vectors, PPlug employs an attention mechanism to dynamically allocate weights:

$$w_i = \frac{\exp(\mathbf{x}^{u\top} \mathbf{h}_i^u)}{\sum_k \exp(\mathbf{x}^{u\top} \mathbf{h}_k^u)}$$

$$\mathbf{P}^u = \sum_i w_i \cdot \text{Proj}(\mathbf{h}_i^u)$$

where Proj is a 2-layer MLP that maps the encoder space to the LLM representation space. This design ensures that, when generating academic titles, the model automatically pays more attention to historical titles that align with the topic of the current abstract.

**LLM Personalization Generation**: The personalized embedding $\mathbf{P}^u$, the trainable instruction embedding $\mathbf{I}$, and the LLM embedding of the current input are concatenated and then fed into the LLM:

$$\mathbf{X}_i^u = [\mathbf{I}; \mathbf{P}^u; \text{Emb}_{\text{LLM}}(x^u); \text{Emb}_{\text{LLM}}(y_{<i}^u)]$$

Key Feature: The LLM parameters are completely frozen; only the instruction embedding $\mathbf{I}$, the input encoder, and the projector are trained.

### Loss & Training

Optimized end-to-end across all user data using the standard next token prediction loss:

$$\mathcal{L} = -\sum_u \sum_i \log p_{\text{LLM}}(y_i^u | \mathbf{X}_i^u)$$

Training Configuration: AdamW optimizer, learning rate 1e-4, warmup ratio 0.05, batch size 64, training for 2 epochs (LaMP-3 is trained for only 1 epoch due to its large data size). Beam search (beam size=4) is used during generation.

## Key Experimental Results

### Main Results

Performance on the 6 tasks of the LaMP benchmark (Table 1):

| Task | Best Baseline | PPlug | Relative Gain |
|------|---------|-------|---------|
| LaMP-1 Citation Resolution (Acc) | 0.682 | 0.680 | -0.3% |
| LaMP-2 Movie Tagging (Acc) | 0.416 | 0.565 | +35.8% |
| LaMP-2 Movie Tagging (F1) | 0.337 | 0.501 | +48.7% |
| LaMP-3 Product Rating (MAE↓) | 0.246 | 0.231 | +6.1% |
| LaMP-4 News Headline (R-1) | 0.207 | 0.216 | +4.3% |
| LaMP-5 Academic Title (R-1) | 0.480 | 0.487 | +1.5% |
| LaMP-7 Tweet Paraphrasing (R-1) | 0.468 | 0.536 | +14.5% |

The significant gains on LaMP-2 and LaMP-7 (35.8% and 14.5%) indicate that these two tasks depend more on the overall user style rather than individual history records.

### Ablation Study

**Impact of Input-Aware Attention**: When removed and replaced with simple averaging, performance drops but still outperforms the baselines. This indicates that overall behavior compression is effective even without differentiated weights.

**Impact of Instruction Embeddings**: When removed, performance drops slightly, showing that instruction embeddings help the LLM separate global task knowledge from user-specific patterns, though the primary contribution comes from the personalized embedding itself.

**Integration with Retrieval Schemes** (Table 3): PPlug + Retrieval yields further improvements across multiple tasks. PPlug provides coarse-grained user style, while retrieval provides fine-grained task-related context, making them highly complementary.

**Analysis of LLM and Encoder** (Table 2):
- Best results are obtained using FlanT5-XXL (11B) + BGE-base.
- Performance is positively correlated with the size of the LLM (FlanT5-XXL > Llama2 7B > FlanT5-XL 3B).
- Performance is comparable when using BGE-base vs. Contriever as the encoder, verifying the robustness of the method.

### Key Findings

1. Compressing all historical behaviors of a user into a single embedding vector is a more effective personalization strategy than selective retrieval.
2. Tasks requiring global style understanding (movie preferences, tweet style) benefit the most.
3. Coarse-grained user embeddings and fine-grained retrieval demonstrations are complementary.
4. PPlug can be trained end-to-end, which is cleaner and more efficient than reinforcement learning or knowledge distillation in ROPG/RSPG.

## Highlights & Insights

- **Elegant Design**: Adding only a single user embedding vector significantly improves personalization performance, without requiring any modifications to the LLM.
- **Plug-and-play Deployment Benefits**: Service providers only need to deploy a single LLM and provide different embedding inputs for different users, which greatly simplifies the hosting infrastructure.
- **End-to-end Optimization**: Compared to RL-feedback-based optimization of retrieval models, PPlug's direct gradient training is more stable and efficient.
- **Balancing Global vs. Local Preferences**: Input-aware attention enables the model to focus on history related to the current task while maintaining global preferences.

## Limitations & Future Work

- Experiments were only conducted on earlier models such as FlanT5 and Llama2; the effectiveness on stronger LLMs (e.g., GPT-4, Llama3) remains unverified.
- Using BGE-base (220M parameters) as the encoder may limit expressive capacity for extremely long or complex user histories.
- Only utilizes user historical behavior data, and does not extend to user attributes (e.g., age, geographic location).
- Compressing the personalized embedding into a single vector may lose information regarding multifaceted user preferences.
- The optimal combination strategy between coarse-grained embeddings and fine-grained retrieval warrants further exploration.
- The temporal dynamics of user preferences over time are not considered.

## Related Work & Insights

- Compared to fine-tuning-based methods (OPPU, per-user LoRA): PPlug is an efficient alternative as it avoids training separate models for each individual user.
- Compared to retrieval-based methods (ROPG, RSPG): PPlug integrates all user history rather than retrieving only a few most relevant ones, thereby capturing the overall style better.
- Promising future directions: multi-granularity personalized embeddings (combining coarse and fine), sequence-aware or temporal user modeling, and general-purpose user embeddings transferable across different tasks.

## Rating

- **Novelty**: 8/10 — The plug-and-play concept of compressing the entire user history into a single embedding is novel and practical.
- **Technical Depth**: 7/10 — The methodology is relatively straightforward but reasonably designed.
- **Experimental Thoroughness**: 8/10 — Comprehensive baseline comparisons, ablations, and LLM/encoder analyses.
- **Value**: 9/10 — Deployment-friendly, enabling a single model to serve multiple users.
- **Overall Rating**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)

</div>

<!-- RELATED:END -->
