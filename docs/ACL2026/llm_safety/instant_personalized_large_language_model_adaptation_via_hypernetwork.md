---
title: >-
  [Paper Note] Instant Personalized Large Language Model Adaptation via Hypernetwork
description: >-
  [ACL2026][LLM Safety][Personalized LLM] Profile-to-PEFT (P2P) uses a hypernetwork to directly map user profiles into personalized LoRA parameters. This avoids the need for OPPU to retrain an adapter for every user…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Personalized LLM"
  - "Hypernetwork"
  - "LoRA"
  - "PEFT"
  - "User Profile"
date: 2026-05-08
content_hash: 733a30e9b1974ab4
---

# Instant Personalized Large Language Model Adaptation via Hypernetwork

**Conference**: ACL2026  
**arXiv**: [2510.16282](https://arxiv.org/abs/2510.16282)  
**Code**: https://zhaoxuan.info/p2p.github.io/  
**Area**: LLM Personalization / Parameter-Efficient Fine-Tuning  
**Keywords**: Personalized LLM, Hypernetwork, LoRA, PEFT, User Profile  

## TL;DR
Profile-to-PEFT (P2P) uses a hypernetwork to directly map user profiles into personalized LoRA parameters. This avoids the need for OPPU to retrain an adapter for every user, achieving faster, more scalable LLM personalization that generalizes to unseen users.

## Background & Motivation
**Background**: LLM personalization primarily follows two routes. Prompt-based methods incorporate user history, retrieval results, or user profiles into the prompt to adapt the model in-context. PEFT-based methods encode user preferences into lightweight parameters, such as training a LoRA adapter for each user.

**Limitations of Prior Work**: Prompt-based methods expose user history to centralized LLMs and are susceptible to interference from irrelevant history. While one-PEFT-per-user (OPPU) methods are highly effective, the cost of training a separate adapter for every user is prohibitive for millions of users, real-time preference updates, or on-device deployment.

**Key Challenge**: Personalization requires "user-specific parameters," yet industrial-scale systems cannot perform repeated gradient updates for every user. An ideal solution should retain the advantages of PEFT-based parametric personalization while being able to generate user parameters as quickly as a single forward pass.

**Goal**: The authors aim to learn a universal mapping from user profiles to PEFT parameters. After seeing diverse users during the training phase, the model can perform instant adaptation for unseen users during deployment without per-user fine-tuning.

**Key Insight**: This paper applies a hypernetwork for user-level PEFT generation. User history is first organized into a natural language summary and retrieved relevant historical interactions, then encoded into an embedding. The hypernetwork generates LoRA matrices for specific layers and modules based on the user embedding, layer depth embedding, and module embedding.

**Core Idea**: Replace "training a LoRA for each user" with "training a network that generates LoRA," using a mapping function shared across users to instantly transform user profiles into personalized parameters.

## Method
The goal of P2P is to generate a set of personalized PEFT parameters for any user at deployment time. Unlike OPPU, which runs optimization on test user history, P2P only requires a single forward pass of the user profile through the hypernetwork. This encodes user preferences into parameters while avoiding the overhead of stuffing long histories into prompts for every call.

### Overall Architecture
The system first constructs a user profile. If a profile already exists in the dataset, it is used directly; otherwise, a base LLM generates a global preference summary from the user's history, and BM25 retrieves the top-k historical interactions relative to the current input. These are concatenated into a profile text. The profile text is encoded into a user embedding by a frozen sentence embedding model.

To inform the hypernetwork which layer and module it is generating parameters for, the user embedding is concatenated with learnable module embeddings and depth embeddings. This position-aware representation enters an MLP hypernetwork, which outputs a flattened LoRA parameter vector, reshaped into $A$ and $B$ matrices for each target module/layer. During training, the generated LoRA is inserted into the frozen base LLM, and the hypernetwork is optimized end-to-end using SFT loss on the user's subsequent interactions.

### Key Designs
1. **Direct Mapping from User Profile to LoRA**:
	- **Function**: Compresses natural language user preferences into personalized parameters that can be inserted into the model.
	- **Mechanism**: The user profile $p_u$ is encoded to $e_u$. The hypernetwork $f_\theta$ generates LoRA matrices $(A_u^{m,l},B_u^{m,l})$ for each layer and module. The full parameter set is denoted as $\Delta W_u=Gen_\theta(p_u)$.
	- **Design Motivation**: Prompt adaptation requires reading user history every time, and OPPU requires training for each user; direct parameter generation turns personalization into a constant-time forward pass overhead.

2. **Module/Layer Position-Aware Parameter Generation**:
	- **Function**: Enables the same user profile to generate different LoRA weights for different layers and projection modules.
	- **Mechanism**: For each target position $(m,l)$, the input $\phi_u^{m,l}=[e_u\|E_{mod}[m]\|E_{dep}[l]]$ is processed by the MLP to output positional LoRA parameters.
	- **Design Motivation**: LLM layers and modules (like q_proj/v_proj) serve different functions. Generating a single shared set of parameters from the user embedding would ignore internal positional differences in the model.

3. **Cross-User End-to-End Training for Generalization**:
	- **Function**: Allows the hypernetwork to learn universal patterns from profile semantics to adapter behavior.
	- **Mechanism**: The training objective is $\mathbb{E}_{u\sim\mathcal{U}}[\mathcal{L}_{SFT}(\Psi\oplus Gen_\theta(p_u),\mathcal{H}_u^{\ge t})]$, minimizing SFT loss on a user's future interactions using generated parameters.
	- **Design Motivation**: Personalization is not about memorizing training users but learning "what kind of user profile should generate what kind of adapter."

### Loss & Training
The authors use Qwen2.5-7B-Instruct as the primary base model and Qwen3-Emb-4B as the default embedding model. The LoRA rank is set to 8, targeting `q_proj` and `v_proj`. P2P is trained for 20,000 steps with a learning rate of $2\times10^{-5}$ and a batch size of 32. Each batch mixes 4 personalization tasks, sampled by the square root of the dataset size to increase task diversity. Inference uses greedy decoding with a temperature of 0. Experiments are also replicated on Qwen2.5-3B-Instruct in the appendix.

## Key Experimental Results

### Main Results
| Setup | Method | Class. Acc↑ | Class. F1↑ | Gen. R-1↑ | Gen. R-L↑ | Avg. Inf. Time ms↓ |
|------|------|-----------|----------|-----------|-----------|------------------|
| Random split | Base | 0.505 | 0.496 | 0.287 | 0.207 | 31.97 |
| Random split | PAG | 0.565 | 0.564 | 0.312 | 0.214 | 66.85 |
| Random split | Full History | 0.575 | 0.566 | 0.310 | 0.224 | 461.83 |
| Random split | OPPU | 0.568 | 0.557 | 0.301 | 0.221 | 35.82 |
| Random split | P2P | 0.580 | 0.566 | 0.322 | 0.244 | 39.98 |
| OOD split | Base | 0.532 | 0.525 | 0.294 | 0.211 | 20.52 |
| OOD split | PAG | 0.562 | 0.563 | 0.329 | 0.234 | 61.66 |
| OOD split | Full History | 0.575 | 0.567 | 0.334 | 0.246 | 392.97 |
| OOD split | OPPU | 0.528 | 0.507 | 0.305 | 0.226 | 26.78 |
| OOD split | P2P | 0.581 | 0.563 | 0.326 | 0.243 | 28.64 |

P2P achieves the highest average Classification Acc and Generation R-1/R-L in the random split, outperforming OPPU without per-user training. In the OOD split, P2P achieves the highest Classification Acc and generation metrics close to the strong prompt-based Full History baseline, while being over an order of magnitude faster than Full History.

### Ablation Study
| Configuration | Class. Acc↑ | Class. F1↑ | Gen. R-1↑ | Gen. R-L↑ | Rating MAE↓ | Rating RMSE↓ |
|------|-----------|----------|-----------|-----------|-------------|--------------|
| P2P Full | 0.581 | 0.562 | 0.326 | 0.243 | 0.258 | 0.583 |
| random user profile | 0.570 | 0.553 | 0.304 | 0.228 | 0.276 | 0.601 |
| shuffle user profile | 0.535 | 0.521 | 0.307 | 0.223 | 0.322 | 0.692 |
| user summary only | 0.562 | 0.545 | 0.313 | 0.240 | 0.304 | 0.584 |
| retrieved history only | 0.538 | 0.521 | 0.298 | 0.216 | 0.405 | 0.712 |
| full history only | 0.541 | 0.526 | 0.302 | 0.217 | 0.392 | 0.740 |

### Key Findings
- In LLM-as-a-Judge open generation evaluation, P2P achieves 2.21/2.15 (Random/OOD) on Personal Reddit and 2.03/1.65 on Empathetic Conversations, consistently higher than Base, PAG, and MT-LoRA.
- Deployment efficiency analysis shows that generating personalized parameters takes 20.44 seconds for OPPU LoRA and 18.78 seconds for OPPU Prompt Tuning; P2P takes only 0.57 seconds, approximately a 33x speedup compared to the fastest OPPU. The one-time training cost (27,167 seconds) amortizes after approximately 1,450 users.
- Embedding backbone ablation reveals that Qwen3-Emb-4B performs best. Interestingly, Qwen3-Emb-8B performed worse, suggesting that larger embedding models are not necessarily better for this task.
- Training user analysis shows that user diversity is more critical than user quantity; increasing cluster diversity improves OOD performance, whereas simply increasing the number of users yields diminishing returns.

## Highlights & Insights
- The paper extends hypernetworks from task-level adapter generation to user-level personalization, a natural but highly practical step. A user profile is essentially a "task description" where the granularity shifts from datasets to individuals.
- The value of P2P extends beyond speed. By removing user history from the prompt, it reduces the exposure of raw history to centralized models and avoids repetitive computation for long contexts.
- Ablations indicate that the user summary is the most critical personalization signal. "Retrieved history only" performed significantly worse, suggesting that future systems should prioritize long-term user profile construction over simple query-time retrieval.

## Limitations & Future Work
- The authors acknowledge that existing datasets typically cover only one task or single-platform behavior per user (e.g., a movie tagging task only includes movie tag preferences). Real-world users span multiple domains (search, writing, shopping, social); cross-task profile generation remains unvalidated.
- While the framework is theoretically compatible with Adapter, IA3, or prefix tuning, the experiments focus on LoRA. Different parameter formats may present varying generation difficulties or privacy risks.
- Privacy is not automatically solved. Generated PEFT parameters are compressed representations of user profiles and could potentially be reverse-engineered to recover sensitive preferences; storage and management of adapters by service providers require additional encryption and isolation.
- Full History remains slightly stronger in OOD generation metrics, indicating that reading the full context provides informational advantages for certain tasks. Future work could explore hybrid solutions combining P2P with lightweight retrieved prompts.

## Related Work & Insights
- **vs prompt-based personalization**: RAG/PAG/Full History do not require training user parameters but increase context length and expose history; P2P writes preferences into parameters, making inference lighter and more suitable for on-device or privacy-sensitive scenarios.
- **vs OPPU**: OPPU trains adapters directly on target user history, acting as an oracle but suffering from slow deployment; P2P matches or exceeds OPPU on several average metrics without specific training for test users.
- **vs HyperLoRA / Text-to-LoRA**: These methods are mostly oriented toward task-level few-shot examples or natural language task descriptions. P2P's inspiration is to treat the user profile as the adapter generation condition, shifting from task generalization to user generalization.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Using hypernetworks to generate PEFT is not entirely new, but its application to large-scale user-level personalization is highly focused.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers LaMP, LongLaMP, Personal Reddit, Empathetic Conversation, Random/OOD splits, efficiency, and multiple ablations.
- Writing Quality: ⭐⭐⭐⭐☆ Problem motivation and system diagrams are clear; conclusions are distinct despite the large number of tables.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for industrial-grade personalized LLMs, especially for on-device user adapter generation and real-time preference updates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)
- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)
- [\[ACL 2026\] DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack](dualguard_dual-stream_large_language_model_watermarking_defense_against_paraphra.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](../../ICML2026/llm_safety/differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[ICML 2026\] Forgetting is Not Erasing: A Survey of Reversibility in Large Language Model Machine Unlearning](../../ICML2026/llm_safety/unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)

</div>

<!-- RELATED:END -->
