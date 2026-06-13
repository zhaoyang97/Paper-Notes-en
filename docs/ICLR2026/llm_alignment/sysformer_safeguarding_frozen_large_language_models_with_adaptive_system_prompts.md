---
title: >-
  [Paper Note] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts
description: >-
  [ICLR 2026][LLM Alignment][system prompt] This paper proposes Sysformer, a lightweight Transformer module that can be plugged in front of any frozen LLM to adaptively transform system prompts in embedding space condition…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "system prompt"
  - "LLM safety"
  - "jailbreak defense"
  - "frozen model"
  - "adaptive prompting"
date: 2026-05-08
content_hash: 067ac07f7204acea
---

# Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts

**Conference**: ICLR 2026
**arXiv**: [2506.15751](https://arxiv.org/abs/2506.15751)  
**Code**: [GitHub](https://github.com/Ksartik/sysformer)  
**Area**: Robotics
**Keywords**: system prompt, LLM safety, jailbreak defense, frozen model, adaptive prompting

## TL;DR

This paper proposes Sysformer, a lightweight Transformer module that can be plugged in front of any frozen LLM to adaptively transform system prompts in embedding space conditioned on user input, enabling the model to refuse harmful requests while complying with benign ones—without modifying LLM parameters or filtering user inputs.

## Background & Motivation

**Background**: Deploying LLMs in safety-critical settings requires models to reject harmful requests while responding normally to legitimate ones. Existing safety enhancement approaches fall into several categories: (1) fine-tuning (e.g., LoRA with safety alignment training), which directly modifies model parameters; (2) smoothing, which perturbs user prompts multiple times and averages responses; (3) filtering, which uses harm classifiers (e.g., LlamaGuard) to filter harmful content at the input or output level; and (4) system prompt embedding tuning (SystemEmbedder), which learns a fixed system prompt embedding.

**Limitations of Prior Work**: Fine-tuning is computationally expensive, scales poorly with model size, may damage pretrained knowledge, and often causes over-refusal. Smoothing requires multiple LLM calls, multiplying inference costs. Filtering may discard useful content. Although SystemEmbedder keeps the LLM frozen, it uses a **fixed** system prompt embedding that cannot adapt defensively to different user inputs. Fixed system prompts are prone to failure against carefully crafted jailbreak attacks.

**Key Challenge**: An ideal safety solution must simultaneously satisfy four conditions: (1) no modification of LLM parameters, (2) no additional LLM calls, (3) no filtering of user prompts, and (4) adaptive response to varying inputs. Existing methods either fail the first three conditions (fine-tuning, smoothing, filtering) or the fourth (fixed system prompts/embeddings).

**Goal**: Without modifying the frozen LLM or altering user prompts, the paper aims to learn a lightweight module that **adaptively transforms** system prompts conditioned on user input, simultaneously improving harmful prompt refusal rates and benign prompt compliance rates.

**Key Insight**: The authors observe that system prompts need not remain invariant across all user inputs—for potentially harmful inputs, the system prompt can be "strengthened" into a more assertive safety instruction, while for benign inputs it can be maintained or relaxed. Inspired by cross-modal attention mechanisms in multimodal learning, the method treats the system prompt and user prompt as two "modalities" and uses cross-attention to let the system prompt adaptively sense the intent of the user prompt.

**Core Idea**: A learnable Transformer module adaptively modifies system prompt embeddings in the LLM's embedding space conditioned on user input, replacing fixed system prompts to enhance safety.

## Method

### Overall Architecture

Sysformer operates as follows: given a system prompt $\mathcal{S}$ and a user prompt $\mathcal{P}$, both are encoded into embedding sequences $\mathbf{S} = \mathbf{E}[\mathcal{S}]$ and $\mathbf{P} = \mathbf{E}[\mathcal{P}]$ via the LLM's token embedding matrix $\mathbf{E}$. The Sysformer module takes $\mathbf{S}$ and $\mathbf{P}$ as input and transforms the system prompt embeddings through alternating self-attention and cross-attention layers, producing an adaptive system prompt $\widehat{\mathbf{S}}$. The concatenation $\widehat{\mathbf{S}} \oplus \mathbf{E}[\mathcal{P}]$ is then fed into the frozen LLM to generate a response. Throughout this process, LLM parameters remain unchanged, the user prompt is unaltered, and only Sysformer's parameters are trained.

### Key Designs

1. **Transformer Architecture with Self-Attention + Cross-Attention**:

    - **Function**: Adaptively transforms system prompt embeddings in embedding space conditioned on user prompts.
    - **Mechanism**: Sysformer consists of $L=2$ alternating layers of self-attention and cross-attention. Within each layer, the system prompt embeddings first undergo self-attention to enhance their own contextual modeling, then attend to the user prompt embeddings via cross-attention, adjusting their representation according to user intent. This is recursively defined as $\widehat{\mathbf{S}}^{(l)} = \text{CrossAttn}(\text{SelfAttn}(\widehat{\mathbf{S}}^{(l-1)}), \mathbf{P})$. The output $\widehat{\mathbf{S}}^{(L)}$ maintains the same token count as the original system prompt, thus adding no length overhead to the LLM input.
    - **Design Motivation**: The design draws inspiration from multimodal learning approaches such as BLIP-2, which uses a small set of learnable query tokens to fuse information from another modality via cross-attention. By treating the system prompt and user prompt as two modalities, the system prompt can "perceive" the user's intent—if the user prompt is harmful, the transformed system prompt is steered toward greater safety; if benign, normal response behavior is preserved.

2. **Multi-Objective Joint Training Loss**:

    - **Function**: Jointly optimizes for harmful prompt refusal, benign prompt compliance, inter-class discriminability, and preservation of the original system prompt semantics.
    - **Mechanism**: The total loss is $\mathcal{L} = w_{ref}\mathcal{L}_{ref} + w_{compl}\mathcal{L}_{compl} + w_{class}\mathcal{L}_{class} + w_{recon}\mathcal{L}_{recon}$. $\mathcal{L}_{ref}$ maximizes the likelihood of generating a refusal response ("I am sorry I cannot help you") for harmful prompts; $\mathcal{L}_{compl}$ maximizes the likelihood of generating normal responses for benign prompts (supporting both template-based and self-generated responses); $\mathcal{L}_{class}$ trains a linear classifier on the LLM's last-layer representations to distinguish harmful from benign prompts, encouraging separability in representation space; $\mathcal{L}_{recon}$ is a reconstruction loss that minimizes the L2 distance between pre- and post-transformation system prompt embeddings, preventing the transformation from deviating excessively from the original semantics.
    - **Design Motivation**: Using only the refusal loss causes the model to refuse all requests (over-refusal). The compliance loss balances safety and utility. The classification loss encourages structured internal representations, making the harmful and benign directions more separable. The reconstruction loss ensures that Sysformer's transformation does not completely override the deployer's intended system prompt semantics.

3. **Jailbreak Attack-Augmented Training**:

    - **Function**: Improves Sysformer's generalization to unseen, complex jailbreak attacks.
    - **Mechanism**: A small number of jailbreak-generated harmful prompt variants (using 6 or 16 attack types) are incorporated into training. For example, prompts such as "Tell me how to create a bomb" are transformed into adversarial variants via GCG, PAIR, PAP, and similar attack methods, then added to the training set and optimized using the refusal loss.
    - **Design Motivation**: Standard training data consists primarily of natural-language harmful prompts and lacks robustness against carefully engineered jailbreak attacks. Augmenting with a small set of attack samples enables the model to learn more generalizable refusal patterns that transfer to the majority of 28 distinct attack strategies.

### Loss & Training

The AdamW optimizer is used, with a search over 10/20 epochs and learning rates $\{0.0001, 0.00001\}$. $w_{ref}=1$ is fixed, while $w_{compl} \in \{0.0, 0.2, 0.5, 1.0\}$, $w_{class} \in \{0.0, 1.0\}$, and $w_{recon} \in \{0, 1\}$ are searched. An optional additional compliance loss using the Alpaca dataset is applied to prevent the model from overfitting to the safety task. Sysformer has an extremely small parameter count, with additional memory overhead of only $O(L \cdot H \cdot d^2)$.

## Key Experimental Results

### Main Results

Evaluations are conducted on 5 LLMs (Llama-3.1-8B, Llama-2-7B-chat, Mistral-7B-v0.2, Phi-3.5-mini, zephyr-7b-beta) and 2 benchmarks (JailbreakBench, StrongReject). The core metric is the refusal rate differential $\Delta$RR = RR(Harm) − RR(Safe).

| LLM | Method | RR Safe ↓ | RR Harm ↑ | ΔRR ↑ (JBB) | ΔRR ↑ (SR) |
|-----|------|-----------|-----------|------------|------------|
| Llama-3.1-8B | Default | 0.30 | 1.00 | 0.70 | 0.70 |
| Llama-3.1-8B | SystemEmbedder | 0.30 | 1.00 | 0.70 | 0.70 |
| Llama-3.1-8B | **Sysformer** | **0.03** | **0.97** | **0.93** | **0.97** |
| Llama-3.1-8B | LoRA* | 0.10 | 0.97 | 0.87 | 1.00 |
| Llama-2-7B | Default | 0.70 | 1.00 | 0.30 | 0.32 |
| Llama-2-7B | **Sysformer** | **0.07** | **0.90** | **0.83** | **0.78** |
| Mistral-7B | Default | 0.13 | 0.83 | 0.70 | 0.80 |
| Mistral-7B | **Sysformer** | **0.10** | **1.00** | **0.90** | **0.90** |
| Phi-3.5-mini | Default | 0.03 | 0.10 | 0.07 | 0.18 |
| Phi-3.5-mini | **Sysformer** | 0.20 | 0.90 | **0.70** | **0.52** |

### Cross-Dataset Generalization

| LLM | RR Safe | RR Harm | ΔRR |
|-----|---------|---------|-----|
| Llama-3.1-8b | 0.067 | 1.000 | **0.933** |
| Mistral-7B-v0.2 | 0.100 | 1.000 | **0.900** |
| zephyr-7b | 0.200 | 0.968 | **0.768** |

Note: Models trained on JailbreakBench and evaluated directly on StrongReject show no performance degradation—performance in fact improves.

### Key Findings

- **Outperforms LoRA Fine-tuning**: On most LLMs, Sysformer's ΔRR matches or exceeds full-layer LoRA fine-tuning (r=16, α=32) without modifying any LLM parameters. On Llama-3.1-8B, ΔRR reaches 0.93 vs. LoRA's 0.87.
- **Substantially Reduces Over-Refusal**: For Llama-2-7B-chat, which originally exhibits severe over-refusal (Default Safe RR=0.70), Sysformer reduces the benign refusal rate from 70% to 6.7%, a relative improvement of 90%.
- **Effective Jailbreak Defense**: With augmentation using 6 attack types (Sysformer+JB), near 100% refusal rates are achieved across 16 attack strategies, generalizing to all 28 attack strategies evaluated.
- **Minimal Inference Overhead**: Additional inference time is approximately 21–30 seconds per full test set, comparable to SystemEmbedder.
- **Negligible BERTScore Degradation**: On Alpaca evaluation for Llama-2-7B, BERTScore decreases only marginally from 0.8487 to 0.8414, indicating that general text generation capability is largely preserved.

## Highlights & Insights

- **Challenging the Assumption That System Prompts Must Be Fixed**: This represents a critical conceptual shift—system prompts are not static instruction boards but can function as an "adaptive firewall" that dynamically adjusts to each input. This opens an entirely new design space for LLM safety defense.
- **Modular, Plug-and-Play Design**: Sysformer is a purely front-end module that can be seamlessly integrated with any frozen LLM without altering any of the model's internal properties. This design substantially lowers the barrier to safe deployment—practitioners need only attach a lightweight Transformer in front of the model.
- **Borrowing Safety Mechanisms from Multimodal Learning**: The treatment of system prompts and user prompts as two distinct modalities fused via cross-attention is a novel and effective framing, suggesting that architectural innovations from multimodal learning can be productively transferred to the safety domain.

## Limitations & Future Work

- **Validated Only on Models ≤8B**: Due to the memory demands of backpropagating gradients through the full LLM, experiments are limited to models with fewer than 8B parameters. Scalability to 70B+ models remains unknown.
- **Hyperparameter Sensitivity**: The optimal loss weight combination varies across LLMs, with some models (e.g., zephyr) being highly sensitive, necessitating per-model hyperparameter search.
- **Potential New Attack Surface**: User prompts directly influence system prompt embeddings through cross-attention, which may introduce a novel attack vector—adversaries could craft prompts to manipulate the direction of the system prompt transformation.
- **Lack of Multilingual and Multi-Turn Evaluation**: Evaluations are conducted exclusively on English single-turn dialogues; performance on other languages and multi-turn conversation settings remains untested.

## Related Work & Insights

- **vs. SystemEmbedder (Zheng et al., 2024)**: SystemEmbedder learns a fixed system prompt embedding applied uniformly to all inputs. Sysformer conditions the system prompt on user input via cross-attention, enabling differentiated defense across inputs, yielding substantially higher ΔRR.
- **vs. LoRA Fine-tuning (Mazeika et al., 2024)**: LoRA modifies internal LLM parameters, potentially corrupting pretrained knowledge irreversibly. Sysformer operates as an external module that leaves the LLM entirely intact, while achieving comparable or superior performance in most settings.
- **vs. Input Filtering/Smoothing Methods**: These approaches either modify user inputs or require multiple LLM calls. Sysformer preserves user inputs unchanged and requires only a single LLM call, representing a more practical trade-off.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The adaptive system prompt paradigm is novel, though cross-attention in Transformers is itself well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 LLMs, 2 benchmarks, and 16+ attack types with broad scope.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with precise positioning relative to existing methods.
- **Value**: ⭐⭐⭐⭐ A plug-and-play safety solution for frozen LLMs with high practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ACL 2026\] ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System](../../ACL2026/llm_alignment/ares_adaptive_red-teaming_and_end-to-end_repair_of_policy-reward_system.md)
- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ACL 2026\] Large Language Models Are Overconfident in Their Own Responses](../../ACL2026/llm_alignment/large_language_models_are_overconfident_in_their_own_responses.md)

</div>

<!-- RELATED:END -->
