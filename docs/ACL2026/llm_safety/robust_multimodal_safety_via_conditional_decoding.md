---
title: >-
  [Paper Note] Robust Multimodal Safety via Conditional Decoding
description: >-
  [ACL2026][LLM Safety][Multimodal Jailbreak Defense] This paper proposes the CASA conditional decoding framework, which requires multimodal models to predict a safety token before generating responses. It utilizes safety…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Multimodal Jailbreak Defense"
  - "Conditional Decoding"
  - "Safety Attention"
  - "Qwen2.5-Omni"
  - "CASA"
date: 2026-05-08
content_hash: 740ebacc18ff8915
---

# Robust Multimodal Safety via Conditional Decoding

**Conference**: ACL2026  
**arXiv**: [2604.00310](https://arxiv.org/abs/2604.00310)  
**Code**: No public code provided in the paper  
**Area**: Multimodal Safety / Speech Language Models / Safety Alignment  
**Keywords**: Multimodal Jailbreak Defense, Conditional Decoding, Safety Attention, Qwen2.5-Omni, CASA

## TL;DR
This paper proposes the CASA conditional decoding framework, which requires multimodal models to predict a safety token before generating responses. It utilizes safety attention to amplify malicious signals, reducing average attack success rates by over 97% across text, vision, and audio jailbreak benchmarks while maintaining multimodal capabilities for benign inputs.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) can simultaneously process text, images, and audio. However, safety alignment primarily stems from refusal training on the text side. When models integrate visual or speech encoders, cross-modal interactions can bypass existing safety boundaries, causing stable alignment behavior in text to degrade under multimodal inputs.

**Limitations of Prior Work**: The mainstream approach is supervised safety fine-tuning (SSFT), which fine-tunes models using malicious questions paired with refusals and benign questions paired with normal answers. This objective forces safety and utility to compete for the same generation target; enhancing refusal capabilities may lead to over-refusal and decreased performance on benign tasks. Furthermore, different modalities require additional safety data and hyperparameter searches.

**Key Challenge**: Models may internally distinguish between safe and unsafe inputs, but standard decoding does not explicitly invoke this internal judgment. Malicious prompts can induce models to bypass safety refusals by hiding intent in long contexts, images, or audio. Thus, the issue is not simply that the "model does not recognize danger," but that the "model does not stably perform safety discrimination before generation."

**Goal**: The authors aim to design a mechanism that does not rely on external classifiers, does not add independent safety heads, and does not require separate training for each modality. The mechanism compels the model to judge input safety first and then condition subsequent generation on that judgment, balancing robust defense and benign utility.

**Key Insight**: PCA on the final layer representations of Qwen2.5-Omni reveals separability between benign and malicious queries. Consequently, the authors transform safety judgment into the first token of the generation process and design a safety attention module to directly influence the logit of the safety token.

**Core Idea**: Safety judgment is converted from an implicit generation preference into an explicit binary token, conditioning the subsequent response on this safety token. Furthermore, safety attention calculated from internal model representations reinforces malicious signals, ensuring the model is blocked by a safety gate before a multimodal jailbreak occurs.

## Method
The design of CASA is straightforward: it does not wrap a detector around the model or train additional classification heads. Instead, it requires the original model to generate a safety label at the start of every response. This label is not final content for the user but a conditional variable controlling the subsequent generation trajectory. Another key component is the safety attention module, which only operates during the safety token prediction step, using prompt representations and safety query embeddings to calculate a malice weight that scales the safe/unsafe token logits.

### Overall Architecture
During training, CASA rewrites benign responses as `{C_safe, response}` and refusals for malicious questions as `{C_unsafe, refusal}`. Thus, the model no longer struggles directly between "outputting a normal response" and "outputting a refusal"; it first predicts the input state and generates appropriate text under that state.

During inference, the model is constrained to choose between safe and unsafe labels at the safety token step. The safety attention module calculates weights based on prompt hidden states; if the input resembles a malicious query, the unsafe token logit is increased; if it resembles a benign query, the safe token logit is increased. Once the safety token is generated, the subsequent response is naturally conditioned on it.

The experimental bases are Qwen2.5-Omni 3B and 7B. Training data includes approximately 6.2k malicious questions and 10k Alpaca benign questions. Evaluation covers text jailbreaking, visual jailbreaking, and audio spelling attacks, using Claude 3.7 as an LLM judge and 13 human annotators to verify safety and utility.

### Key Designs
1.  **Classify Before You Generate**:
    - **Function**: Explicitly places safety judgment before response generation to prevent the model from organizing content while simultaneously judging risk.
    - **Mechanism**: The training objective shifts from generating `y_resp` or `y_ref` to generating `{C_safe, y_resp}` or `{C_unsafe, y_ref}`. The response probability is the product of predicting the safety variable `P(y0=C|x)` and the subsequent tokens.
    - **Design Motivation**: SSFT's safety and utility goals often compete. Safety tokens convert these into serial decisions: first classify the context, then generate by category.

2.  **Safety Attention Module**:
    - **Function**: Amplifies malicious signals hidden in multimodal inputs during safety token prediction.
    - **Mechanism**: Uses prompt hidden states as key/value and safety embeddings from a frozen pretrained model as the query to compute weight `v_s`. The unsafe logit is scaled by `v_s`, and the safe logit by `1-v_s`. Stop-gradient prevents the module from interfering with prompt representations.
    - **Design Motivation**: Jailbreak inputs often hide intent in long contexts or audio/visual details; standard refusal training might only learn surface templates. Safety attention forces the model to focus on risk cues at critical time steps.

3.  **Constrained Decoding for Safety Tokens**:
    - **Function**: Prevents the model from bypassing the safety discrimination step.
    - **Mechanism**: During inference, tokens other than safe/unsafe are masked in the vocabulary at the safety token step, and logits are replaced by the learned scaling factors. Subsequent normal generation does not recompute safety attention.
    - **Design Motivation**: If allowed to generate freely, the model might skip safety labels or output other prefixes. Constrained decoding ensures safety judgment occurs with minimal computational overhead.

### Loss & Training
CASA continues the benign/malicious paired training of SSFT but adds the safety token to the target sequence. The parameter `β` in the training objective controls the weight of malicious refusals versus benign responses. Gradients for safety attention originate from the logit scaling term, training the attention parameters and the original MLLM. The authors use PEFT/LoRA to fine-tune Qwen2.5-Omni 3B and 7B without external detectors or modality-specific safety tuning.

## Key Experimental Results

### Main Results
The table displays the multimodal jailbreak Attack Success Rate (ASR), where lower is better. CASA significantly reduces ASR across text, vision, and audio attacks.

| Model | Safety Prompt | 3B JB-Prompt | 3B JBV-28k | 3B MM-SB | 3B AIAH | 7B JB-Prompt | 7B JBV-28k | 7B MM-SB | 7B AIAH |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pretrained | No | 42.3 | 36.8 | 37.7 | 81.3 | 33.5 | 37.9 | 38.1 | 64.2 |
| SSFT | No | 18.4 | 7.9 | 14.9 | 71.0 | 0.0 | 7.5 | 8.8 | 25.0 |
| Circuit Breaker | No | 0.9 | 3.9 | 5.1 | 2.3 | 0.3 | 5.7 | 5.4 | 24.4 |
| CASA | No | 0.0 | 4.6 | 9.2 | 2.3 | 0.0 | 0.7 | 9.0 | 1.1 |
| CASA | Yes | 0.0 | 1.4 | 1.2 | 0.0 | 0.9 | 0.0 | 0.2 | 0.6 |

### Ablation Study

| Configuration | JBV-28k ASR | MM-SB ASR | AIAH ASR | Description |
| :--- | :--- | :--- | :--- | :--- |
| CASA + Safety Attention + Safety Prompt | 1.4 | 1.2 | 0.0 | Complete config; near-total defense for vision and audio |
| CASA + Safety Attention, w/o Safety Prompt | 4.6 | 9.1 | 2.3 | Still significantly better than versions without attention |
| CASA w/o Safety Attention + Safety Prompt | 8.2 | 18.3 | 60.2 | Particularly vulnerable to audio spelling attacks |
| CASA w/o Safety Attention, w/o Safety Prompt | 13.2 | 26.8 | 61.9 | Shows safety tokens alone cannot cover all multimodal attacks |

### Key Findings
- In prefill attacks, the Pretrained model's ASR rose from 65.3 to 84.7 as prefill length increased. SSFT and Circuit Breaker fluctuated, while CASA maintained 0.0 ASR across 2, 4, 9, and 12 token prefills.
- In MME utility evaluations, CASA did not degrade multimodal capabilities; it achieved Perception 1621.23 and Cognition 530.71 on 3B, and Perception 1651.98 and Cognition 652.85 on 7B, outperforming Pretrained, SSFT, and Circuit Breaker.
- Consistency between human safety evaluation and the Claude judge was high: Cohen's κ was 0.79 for safety tasks; Krippendorff's α for human internal consistency was 0.60; Human-LLM judge consistency for utility was 0.68.
- Safety attention values approached 1 for malicious queries and 0 for benign queries during training, indicating the module learned interpretable risk gating signals.

## Highlights & Insights
- The core insight of CASA is clean: multimodal safety failures are not necessarily due to a total lack of risk awareness, but rather the failure to prioritize safety judgment in the generation process. Explicit safety tokens provide a low-cost yet powerful behavioral intervention.
- The method avoids the complexity of deploying external safety classifiers and the need for per-modality defense training. For industrial multimodal systems, this endogenous gating is easier to maintain than multiple external guards in series.
- Safety attention is computed only once at the safety token step, exploiting the phenomenon that refusal behavior is concentrated at the start of generation. This is efficient and aligns with mechanistic analyses of safety alignment.
- The utility results are notable: CASA outperforms SSFT and CB on MME, suggesting that decoupling safety and utility allows the model to gain defensive capabilities without sacrificing performance on normal queries.

## Limitations & Future Work
- Although the paper evaluates various text, visual, and audio jailbreaks, the authors acknowledge that more complex attacks may exist, particularly compositional, multi-turn, or in-context induction attacks.
- Safety Attention performs cross-attention over the entire prompt, which may become a computational bottleneck in long contexts; while computed only once, optimization is needed for ultra-long video, audio, or multi-document inputs.
- The safety scope primarily covers explicit malicious queries, with insufficient coverage of indirect risks where "surface-level safety" combined with context leads to harm.
- CASA identifies models where internal representations already contain separable safety signals; for weaker models, non-instruct models, or domains with poor representation separability, effectiveness may decrease.

## Related Work & Insights
- **vs SSFT**: SSFT learns refusal and normal responses via the same generation objective, leading to safety-utility conflicts. CASA uses safety judgment as the first conditional variable, reducing competition between the two goals.
- **vs Circuit Breaker**: Circuit Breaker is a strong defensive baseline but is unstable in certain utility and audio attacks. CASA's advantage lies in safety tokens and attention gating directly integrated into the decoding process.
- **vs External Safety Classifiers**: External classifiers require additional deployment and may miss internal cross-modal cues. CASA uses MLLM hidden states directly, staying closer to the model's actual generation path.
- **Insight**: Many alignment problems can be addressed through "explicit state variables before answering," such as factuality, permission, or privacy tokens. The key is conditioning subsequent generation on controllable states rather than filtering outputs post-hoc.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The idea of conditional safety tokens is simple and effective; safety attention solidifies the mechanism, though it still builds on SSFT and token-level gating.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across text, vision, audio, multiple attacks, utility, and human evaluation provide a complete evidence chain.
- **Writing Quality**: ⭐⭐⭐⭐☆ Methods are clearly explained with sufficient tabular information; some formula layouts are dense but do not hinder understanding.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for the safety deployment of multimodal models, especially for systems avoiding external classifiers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](safetyalfred_evaluating_safety-conscious_planning_of_multimodal_large_language_m.md)
- [\[ACL 2026\] PARASITE: Conditional System Prompt Poisoning to Hijack LLMs](parasite_conditional_system_prompt_poisoning_to_hijack_llms.md)
- [\[ACL 2026\] MUSE: A Run-Centric Platform for Multimodal Unified Safety Evaluation of Large Language Models](muse_a_run-centric_platform_for_multimodal_unified_safety_evaluation_of_large_la.md)
- [\[ACL 2026\] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)

</div>

<!-- RELATED:END -->
