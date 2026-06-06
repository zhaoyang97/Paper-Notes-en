---
title: >-
  [Paper Note] CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks
description: >-
  [ACL 2026][LLM Safety][implicit jailbreak] Targeting "joint-modal implicit attacks" where images and text are individually safe but harmful when combined, this work proposes ImpForge…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "implicit jailbreak"
  - "joint-modal attack"
  - "red-teaming"
  - "guardrail"
  - "LoRA SFT"
  - "ImpForge"
date: 2026-05-08
content_hash: fe146f352902dc99
---

# CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks

**Conference**: ACL 2026  
**arXiv**: [2510.17687](https://arxiv.org/abs/2510.17687)  
**Code**: [github.com/ZhangXu0963/CrossGuard](https://github.com/ZhangXu0963/CrossGuard)  
**Area**: Multimodal VLM / AI Safety / Jailbreak Defense  
**Keywords**: implicit jailbreak, joint-modal attack, red-teaming, guardrail, LoRA SFT, ImpForge

## TL;DR
Targeting "joint-modal implicit attacks" where images and text are individually safe but harmful when combined, this work proposes ImpForge, an RL-based red-teaming framework that automatically generates such samples using three rewards (safety, semantic, and overlap). These data are then used to train CrossGuard via LoRA SFT. CrossGuard reduces the Attack Success Rate (ASR) of SIUO implicit attacks from 48.9% (GPT-4o) to 5.4%, while maintaining an average ASR of 2.79% across five safety benchmarks (compared to 12.05% for Claude-3.5).

## Background & Motivation

**Background**: MLLM jailbreak attacks are primarily categorized into text-based (gradient-based or evolutionary prompt optimization) and vision-based (adversarial perturbations, OCR triggers, or embedded malicious text). Current defenses such as LlavaGuard, Llama-Guard3-Vision, HiddenDetect, and JailDAM assume malicious signals exist explicitly within a single modality and treat vision and text as independent channels.

**Limitations of Prior Work**:

1. The SIUO benchmark (Wang et al. 2025a) reveals a new threat: **joint-modal implicit attacks**. In these cases, the image and text are completely harmless in isolation (e.g., neither a photo of a bomb nor a "how to make a bomb" query), but they form a malicious intent when combined (e.g., a dangerous device shown in an image with the text "I have this at home, how do I maximize its effect?"). GPT-4o reaches a 48.9% ASR on SIUO, and Llama-Guard3-Vision fails with a 90% ASR, despite both performing well against explicit attacks. This suggests current defenses over-fit to single-modality toxicity.
2. Data scarcity: SIUO contains only 167 human-annotated samples. Traditional LLM red-teaming scripts cannot generate "individually safe + jointly malicious" samples.
3. Lack of specialized training protocols: Existing guardrail training sets contain virtually zero implicit samples.

**Key Challenge**: Single-modality guards cannot naturally reach the level of "inter-modal compositional semantics." Training a guard capable of identifying implicit intent requires large-scale, diverse samples. However, generating such samples involves three conflicting objectives: maintaining single-modal safety (text must look safe), preserving the original malicious intent, and reducing semantic overlap between text and image (to increase implicitness and avoid detection by simple alignment).

**Goal**:

- Sub-problem 1: Can high-quality "individually safe / jointly malicious" cross-modal samples be automatically generated at scale?
- Sub-problem 2: Can a guard trained on such data defend against both explicit and implicit attacks without sacrificing usability on benign queries?

**Key Insight**: Upgrade the LLM single-modality RL red-teaming framework to a multimodal one. Fix the image and optimize the text (due to lower optimization costs) using three complementary rewards. This effectively adapts the LLM red-teaming paradigm to implicit multimodal scenarios.

**Core Idea**: A dual approach combining **"3-reward guided RL red-teaming + LoRA-based guard training."** ImpForge addresses the lack of training data, while CrossGuard converts this data into a deployable pre-filter, balanced with explicit and benign VQA samples for safety and utility.

## Method

### Overall Architecture
The framework consists of two coupled components: (1) **ImpForge**, a data generation pipeline in two stages: Stage 1 matches safe images to malicious text queries using NER and CLIP (keyword-to-image mapping); Stage 2 uses PPO and LoRA to train a rewriter policy that transforms a pair $(x^I, x^T)$ into a more implicit $(x^I, \hat{x}^T)$ under the supervision of the three-reward module. (2) **CrossGuard**, the guard model based on LLaVA-1.5-7B. It is trained on a mixture of ImpForge implicit data, VLGuard/FigStep explicit data, and VQAv2 benign data. LoRA fine-tuning is applied to both vision and language backbones, outputting a binary safety decision as a pre-filter.

### Key Designs

1. **ImpForge Three-Reward Design (safety / semantic / overlap)**:
    - **Function**: Formalizes the "ideal implicit malicious sample" as three complementary constraints for PPO optimization.
    - **Mechanism**: (a) **Safety reward** $R_{\text{safety}}(\hat{x}^T) = \text{softmax}(p(\texttt{safe}|x'_T))$ uses a pre-trained Llama-Guard style model to ensure the rewritten text appears harmless; (b) **Semantic reward** $R_{\text{sim}}(x^I, x^T, \hat{x}^T) = \cos(g(x^I \oplus \hat{x}^T), g(x^T))$ uses Sentence-BERT to ensure the joint representation of "image + rewritten text" aligns with the original intent; (c) **Overlap reward** $R_{\text{ovlp}} = 1 - \frac{1}{|\text{Tok}(\hat{x}^T)|} \sum_w \max[0, \cos(g(w), g(x^I)) - \tau]$ ($\tau=0.2$) penalizes word-level similarity between the rewritten text and the image to maximize implicitness. The PPO objective is $\max_\theta \mathbb{E}[R_\psi - \lambda D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})]$.
    - **Design Motivation**: A single reward cannot satisfy all criteria; there is a natural trade-off between safety and malicious intent. The three rewards decouple these goals, allowing PPO to explore the Pareto frontier. The overlap reward uses token-level cosine similarity as a proxy for mutual information to avoid training instability.

2. **Stage 1: Safe Image Matching via NER + CLIP Retrieval**:
    - **Function**: Finds a semantically relevant but harmless image for each malicious text query for which no ready-made "safe image" pair exists.
    - **Mechanism**: Extracts visual entities (nouns, verbs) from BeaverTails malicious queries via NER, filtering out abstract words. For each keyword, CLIP retrieves the most similar safe image from libraries like COCO or WIT using $\frac{g(k) \cdot g(x^I)}{\|g(k)\| \|g(x^I)\|}$. GPT is then used to verify that the image contains no primary malice. The output is a triplet $(x^I, x^T, k)$.
    - **Design Motivation**: Malicious text alone cannot train RL (missing the visual component). Random image pairing fails for implicit attacks as it lacks correlation. CLIP soft-matching provides a middle ground: keywords provide an anchor, CLIP ensures visual relevance, and GPT verification maintains the safety baseline.

3. **Hybrid Training Dataset & LoRA Dual-Backbone Fine-tuning for CrossGuard**:
    - **Function**: Trains the guard to recognize implicit and explicit malice while allowing benign queries.
    - **Mechanism**: The training set integrates ImpForge implicit samples (14 domains), VLGuard/FigStep explicit samples, and VQAv2 benign samples. LoRA adapters are applied to the vision encoder and language model of LLaVA-1.5-7B. The model optimizes binary cross-entropy $\mathcal{L}_{\text{CE}} = -\mathbb{E}_{(x^I,x^T,y)} \log p_\theta(y | x^I, x^T)$ for safety classification.
    - **Design Motivation**: (1) Mixing three types of data controls ASR (via ImpForge/VLGuard) and utility (via VQAv2). (2) Dual backbone LoRA is necessary because implicit detection requires joint vision-language understanding; freezing one side loses cross-modal reasoning depth. (3) Binary classification is chosen over generative refusal for speed, batch processing, and ease of integration into MLLM pipelines.

### Loss & Training
ImpForge employs PPO with LoRA adapters to update the rewriter policy, with the KL coefficient $\lambda$ controlling deviations from the reference policy. The composite reward is $R_\psi = R_{\text{safety}} + R_{\text{sim}} + R_{\text{ovlp}}$. Images are fixed during PPO to save computation. CrossGuard uses standard supervised LoRA SFT with a cross-entropy objective for binary classification.

## Key Experimental Results

### Main Results

The table below shows ASR comparisons across five safety benchmarks (lower is better):

| Model / Guard | JailBreakV (OOD) | MM-Safety (OOD) | SIUO (OOD, implicit) | FigStep (ID) | VLGuard (ID) | Avg ASR |
|---|---|---|---|---|---|---|
| LLaVA-1.5-7B (base) | 51.43 | 28.85 | 95.81 | 62.60 | 46.38 | 57.01 |
| Qwen2.5-VL-7B | 2.14 | 10.00 | 41.56 | 24.20 | 9.73 | 17.53 |
| GPT-4o | 6.08 | 16.15 | 48.92 | 1.60 | 6.11 | 15.77 |
| Claude-3.5-Sonnet | 5.00 | 13.08 | 23.95 | 13.00 | 5.21 | 12.05 |
| LlavaGuard | 90.71 | 32.58 | 90.80 | 83.08 | 90.42 | 77.52 |
| Llama-Guard3-Vision | 34.29 | 74.89 | 50.40 | 66.92 | 89.82 | 63.26 |
| HiddenDetect | 4.64 | 8.65 | 44.91 | 72.20 | 26.02 | 31.28 |
| JailDAM | 32.50 | 16.54 | 81.44 | 6.00 | 15.38 | 30.37 |
| **Ours (CrossGuard)** | **0.72** | **0.38** | **5.39** | **0.21** | 7.24 | **2.79** |

**ImpForge Attack Validity**: Pairing original BeaverTails queries with images (BeaverTails*) results in a low ASR. However, after ImpForge rewriting, ASR surges: Qwen2.5-VL-7B increases from 4.2% to 76.6%, and GPT-4o from 9.8% to 70.4%, proving the effectiveness of the generated implicit samples.

### Ablation Study

Ablations regarding data composition and reward modules:

| Configuration | SIUO ASR / Key Metric | Insight |
|---|---|---|
| Full CrossGuard (Implicit + Explicit + VQA) | 5.39% | Complete solution; comprehensive defense + high utility |
| Base LLaVA-1.5-7B (No guard) | 95.81% | Completely vulnerable to implicit attacks |
| Explicit Data Only (≈ Llama-Guard3-V) | ~50% | Explicit data fails to generalize to implicit attacks |
| ImpForge w/o safety reward | Rewrite looks malicious alone | Safety reward ensures "individual safety" |
| ImpForge w/o semantic reward | Re-write loses malicious intent | Semantic reward preserves "joint malice" |
| ImpForge w/o overlap reward | High text-image overlap | Easy detection by semantic alignment |
| BeaverTails* → +ImpForge | Qwen2.5-VL ASR: 4.2% → 76.6% | Rewriting yields over 70pp ASR increase |
| GPT-4o Attack: 9.8% → 70.4% | Breaking SOTA commercial models | Validates the threat level of ImpForge samples |

### Key Findings

- **Implicit attacks are a major blind spot**: Even GPT-4o (48.92%) and Llama-Guard3-Vision (50.40%) exhibit high ASR on SIUO, suggesting a lack of cross-modal intent integration. CrossGuard drastically reduces this to 5.39%.
- **Defensive asymmetry**: Many baselines performing well on explicit attacks fail on implicit ones, indicating that current guardrails learn "intra-modal malicious patterns" rather than "cross-modal intent."
- **Breaking the security-utility trade-off**: While JailDAM/HiddenDetect show over-defense (rejecting benign queries) and others show under-defense, CrossGuard achieves both high security and utility through its hybrid data mixture.
- **OOD Robustness**: Strong performance on JailBreakV and MM-SafetyBench (ASR < 1%) demonstrates that the model learns generalized safety boundaries rather than pattern memorization.

## Highlights & Insights

- **Approximating Mutual Information with token-level cosine similarity** is a clever engineering choice for the overlap reward. This non-parametric proxy avoids the instability of mutual information estimators in RL loops while effectively penalizing redundancy.
- **Fixing the image and optimizing text** is a pragmatic design. Text rewriting provides higher information density for optimization with lower computational costs compared to pixel-level optimization.
- **The hybrid data recipe** (Implicit + Explicit + Benign VQA) is crucial for balancing utility and security. Including VQAv2 samples prevents the "over-defense failure mode" where a model rejects all queries.
- **The "individually safe + jointly malicious" threat model** provides a methodological shift for the multimodal security community, highlighting the ceiling of single-modal guards.

## Limitations & Future Work

- ImpForge training costs are relatively high due to PPO and multi-reward tuning, with sensitive reward weights.
- Implicit data generation depends on seed data (BeaverTails); limited seed distributions may lead to defense gaps in new malicious categories.
- As a binary classifier, CrossGuard does not provide explanations for rejections, which may be required in industrial deployments.
- Deployment latency: Running LLaVA-7B + LoRA as a pre-filter adds overhead; distillation into smaller guards is a potential future direction.

## Related Work & Insights

- **Vs. Wang et al. 2025a (SIUO)**: SIUO identified the threat; this work provides the solution through RL-based data generation.
- **Vs. Llama-Guard3-Vision / LlavaGuard**: These models fail against implicit attacks (50-90% ASR). This work proves that existing architectures are sufficient if the training data includes implicit samples.
- **Vs. Ge et al. 2024 / Perez et al. 2022**: While prior works applied RL red-teaming to text LLMs, this work extends it to cross-modal implicit attacks with a 3-reward system.
- **Vs. JailDAM / HiddenDetect**: These rely on hidden state or prompt perturbations, often at the cost of utility. CrossGuard solves this trade-off using data and dual-backbone SFT.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First expansion of RL red-teaming to joint-modal implicit attacks with an elegant 3-reward design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 safety benchmarks, utility tests, and OOD assessment.
- **Writing Quality**: ⭐⭐⭐⭐ Clear visualization of threats in Figure 1, though some LaTeX formulas in the text required cross-referencing with the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Released code and dual artifacts (guard + generator) provide a high-value tool for the MLLM safety community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[ACL 2026\] Knowledge Poisoning Attacks on Medical Multi-Modal Retrieval-Augmented Generation](knowledge_poisoning_attacks_on_medical_multi-modal_retrieval-augmented_generatio.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)

</div>

<!-- RELATED:END -->
