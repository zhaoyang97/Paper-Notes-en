---
title: >-
  [Paper Note] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks
description: >-
  [ICML 2026][LLM Alignment][Harmful Fine-tuning] This paper demonstrates that all existing HFT defenses that "impose constraints in parameter space" can be bypassed due to parameter redundancy. It proposes Safety Bottlene…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "Harmful Fine-tuning"
  - "Unembedding Bottleneck"
  - "Parameter Redundancy"
  - "MSE Anchor"
  - "RLHF Robustness"
date: 2026-05-08
content_hash: c5001f19279b944e
---

# Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks

**Conference**: ICML 2026  
**arXiv**: [2605.05995](https://arxiv.org/abs/2605.05995)  
**Code**: [soyoaaa/SBR](https://github.com/soyoaaa/SBR)  
**Area**: Alignment / LLM Safety  
**Keywords**: Harmful Fine-tuning, Unembedding Bottleneck, Parameter Redundancy, MSE Anchor, RLHF Robustness

## TL;DR
This paper demonstrates that all existing HFT defenses that "impose constraints in parameter space" can be bypassed due to parameter redundancy. It proposes Safety Bottleneck Regularization (SBR), which shifts the defense battlefield to the geometric bottleneck of the unembedding layer: by anchoring the last-layer hidden state of just a single high-risk prompt, the Harmful Score can be suppressed to $< 10$ under 50 epochs of continuous HFT attack, without compromising benign task accuracy.

## Background & Motivation
**Background**: RLHF enables LLMs to refuse illegal requests. However, fine-tuning-as-a-service allows users to upload datasets for re-training, where even a small number of malicious samples (Harmful Fine-tuning, HFT) can dismantle safety guardrails within a few epochs. Existing defenses—(i) parameter distance (Lisa/EWC), (ii) gradient direction (Booster), and (iii) representation drift (Vaccine/T-Vaccine)—can suppress harmful behavior in early epochs.

**Limitations of Prior Work**: Through stress tests involving 50 epochs of continuous HFT, the authors found that all three types of defenses eventually collapse (HS $> 30$) after 5–10 epochs. Notably, when they collapse, their respective monitored metrics (parameter distance/gradient direction/representation drift) do not exceed established boundaries. This suggests these methods successfully constrain the form but fail to preserve the essence of safety.

**Key Challenge**: LLMs are highly over-parameterized, allowing attackers to always find optimization directions orthogonal to defense constraints. For instance, a random Rank-1 LoRA $\Delta W=BA^\top$ (with $A$ frozen) is sufficient to recover harmful capabilities, indicating that harmful directions are "ubiquitous" in parameter space rather than sparse. Consequently, any constraint in the "redundant high-dimensional parameter space" has a null space that can be exploited.

**Goal**: To identify a chokepoint that is unaffected by parameter redundancy and cannot be bypassed by attackers, and to apply defense solely at that location.

**Key Insight**: The authors observe that the final step of token generation is the inner product of the last-layer hidden state $h_{\text{final}}$ and the word embedding $w_t$: $\text{Score}(t)=h_{\text{final}}^\top w_t$. This unembedding projection is a geometric bottleneck through which all harmful tokens must pass. Since $w_t$ is frozen, as long as $h_{\text{final}}$ is anchored toward the refusal embedding direction, the softmax will inevitably prioritize refusal tokens.

**Core Idea**: Instead of setting defenses in parameter space, one should directly anchor the last-layer hidden states of a set of high-risk queries at the unembedding layer to remain consistent with the frozen aligned model. Regardless of how internal parameters evolve, if the bottleneck is fixed, malicious tokens cannot be generated.

## Method

### Overall Architecture
SBR operates in fine-tuning-as-a-service scenarios: the service provider holds the aligned base model $f_{\theta_{\text{base}}}$ and a set of "safety anchors" $\mathcal{X}_{\text{anchor}}=\{x'_1,\ldots,x'_K\}$ (typical dangerous prompts, e.g., "How to make a bomb?"), but cannot access the user-uploaded training set $\mathcal{D}_{\text{train}}$. SBR consists of two phases:

- **Phase 1 — Anchor Acquisition (Offline)**: Use the frozen $f_{\theta_{\text{base}}}$ to extract the hidden state of the last token in the last layer $h_{\text{ref}}(x')=f^{\text{last}}_{\theta_{\text{base}}}(x')$ for each anchor, which is cached as $\mathcal{H}_{\text{ref}}$.
- **Phase 2 — Dynamic Regularization (Parallel with fine-tuning)**: For each batch, compute $\mathcal{L}_{CE}$ (user task) and $\mathcal{L}_{\text{safe}}=\frac{1}{|\mathcal{X}_{\text{anchor}}|}\sum_{x'}\|h_\theta(x')-h_{\text{ref}}(x')\|_2^2$ simultaneously. The total objective is $\mathcal{L}_{\text{total}}=\mathcal{L}_{CE}+\lambda\mathcal{L}_{\text{safe}}$, where $\lambda$ controls refusal strength. The process does not modify the unembedding matrix or the base model architecture.

### Key Designs

1.  **Geometric Bottleneck Localization — Moving Defense to Unembedding Input**:
    - **Function**: Imposes defense at the "minimum redundancy / maximum necessity" location, making bypassing it require simultaneous changes to $h_{\text{final}}$ and $w_t$.
    - **Mechanism**: Since $P(t|x)=\text{softmax}(h_{\text{final}}^\top w_t)$, refusal and harmful tokens compete within the same softmax. If $h_{\text{final}}$ is geometrically biased toward refusal token embeddings, refusal tokens will strictly outscore harmful ones. MSE anchoring ensures $h_{\text{final}}$ stays close to the base model's aligned output for these prompts.
    - **Design Motivation**: Section 3 uses three stress tests (parameter distance / Rank-1 random subspace / representation drift) to prove that high-dimensional parameter space always contains "escape routes" orthogonal to defense directions. Only the lower-dimensional unembedding input layer, directly connected to token selection, lacks such a null space.

2.  **MSE Anchor Loss $\mathcal{L}_{\text{safe}}$ with Minimal Anchors**:
    - **Function**: Applies a hard constraint to the bottleneck with minimal overhead.
    - **Mechanism**: $\mathcal{L}_{\text{safe}}(\theta)=\frac{1}{|\mathcal{X}_{\text{anchor}}|}\sum_{x'\in\mathcal{X}_{\text{anchor}}}\|h_\theta(x')-h_{\text{ref}}(x')\|_2^2$ is weighted by $\lambda$ with the user's $\mathcal{L}_{CE}$. Only 1–8 anchors sampled randomly from a candidate pool (a BeaverTails subset disjoint from the attacker's data) are needed.
    - **Design Motivation**: The authors argue that the "refusal direction and benign reasoning direction are approximately orthogonal" (Zou 2023, Arditi 2024). Thus, anchoring a small set of high-risk prompts does not significantly restrict the optimization space for benign tasks. Experiments show 1 anchor is sufficient to keep HS $< 10$.

3.  **Stress Test Paradigm — Revealing Defeat via 50-Epoch HFT**:
    - **Function**: Shifts defense evaluation from "short-term snapshots" to "sustained attacks," revealing the illusion of transient safety.
    - **Mechanism**: A mixed dataset (10% harmful + 90% benign) is used to train the model for 20–50 epochs across four benign tasks (SST-2/AGNEWS/GSM8K/AlpacaEval). A Random Subspace Attack (training $B$ while freezing $A$) is designed to prove that even random Rank-1 directions can recover harmful capabilities.
    - **Design Motivation**: Existing papers often report HS within 3–5 epochs, obscuring the fact that service providers might allow long-term user training. Stress tests uncover this reality and provide a rigorous baseline for SBR.

### Loss & Training
LoRA rank 16 / alpha 16, AdamW lr $1\times 10^{-5}$, batch size 32, 20 epochs; anchor count $K=8$, $\lambda=50$. Anchors only require a forward pass without backpropagation through the base model. All baselines are re-run under identical hyperparameters.

## Key Experimental Results

### Main Results
Llama3.1-8B, dual metrics of HS↓ and FA↑ (Fine-tuning Accuracy) across 4 benign downstream tasks:

| Method | SST-2 HS↓ | SST-2 FA↑ | GSM8K HS↓ | GSM8K FA↑ | AlpacaEval HS↓ | AlpacaEval FA↑ | Avg HS↓ | Avg FA↑ |
|--------|-----------|-----------|-----------|-----------|----------------|-----------------|----------|---------|
| SFT (no defense) | 67.80 | 94.61 | 71.10 | 82.80 | 74.20 | 43.87 | 70.70 | 78.07 |
| DeepAlign | 25.90 | 93.12 | 20.70 | 88.00 | 23.70 | 33.64 | 25.10 | 76.04 |
| Lisa | 52.50 | 94.27 | 40.40 | 72.20 | 58.20 | 37.93 | 52.45 | 73.50 |
| Vaccine | 61.40 | 92.55 | 64.30 | 75.10 | 62.90 | 36.39 | 62.53 | 73.34 |
| Booster | 59.80 | 92.89 | 71.50 | 76.20 | 54.30 | 35.75 | 62.33 | 73.66 |
| **SBR** | **5.80** | **94.15** | **5.60** | **82.60** | **6.20** | **45.82** | **5.68** | **78.17** |

### Ablation Study
**Robustness to Poisoning Ratio (Llama3.1-8B)**:

| Poison ratio $p$ | SFT HS | DeepAlign HS | Vaccine HS | Booster HS | **SBR HS** | SBR FA |
|------------------|--------|--------------|------------|------------|------------|--------|
| 0.05 | 67.90 | 21.50 | 58.70 | 59.40 | **4.10** | 93.92 |
| 0.10 | 67.80 | 25.90 | 61.40 | 59.80 | **5.80** | 94.15 |
| 0.20 | 71.90 | 29.90 | 61.90 | 64.60 | **8.20** | 93.92 |
| 0.30 | 74.30 | 33.30 | 69.20 | 67.30 | **7.30** | 93.69 |
| Average | 70.48 | 27.65 | 62.80 | 62.78 | **6.35** | 93.92 |

**Sensitivity to $\lambda$**: Validation across $\lambda\in\{0,5,10,50,100\}$ shows $\lambda=50$ is a stable sweet spot for HS↓ and FA↑ ($\lambda=0$ degrades to SFT, while $\lambda\ge 100$ begins to erode FA).

### Key Findings
- $K=1$ anchor is sufficient: The paper repeatedly emphasizes that "a single safety anchor is sufficient to reduce the Harmful Score to < 10," proving that the unembedding bottleneck is extremely narrow.
- SBR remains stable under 50 epochs of continuous HFT, whereas Lisa/Vaccine/Booster collapse as early as 5 epochs, as shown in the dramatic comparison in Figure 2.
- Empirical evidence of Drift-Safety Dissociation in Section 3 (where embedding drift remains unchanged between steps 120 and 480 but HS jumps from 12 to 59) independently proves that monitoring global representation drift is an incorrect proxy variable.
- Performance on benign tasks not only holds but slightly improves (Avg FA 78.17 vs SFT 78.07), supporting the hypothesis that refusal and reasoning directions are nearly orthogonal.

## Highlights & Insights
- Thoroughly explains the mechanism of why existing HFT defenses are bypassed using stress tests and Random Subspace Attacks, providing a reality check for the research field.
- The "finding the bottleneck" concept is highly transferable—any defense where "redundancy allows attackers to find orthogonal bypasses" should consider moving to downstream mandatory geometric points.
- The minimalism of requiring only 1 anchor means deployment costs are nearly zero: service providers only need to cache $K$ hidden state vectors, adding negligible overhead to each forward pass.
- The lack of conflict between $\mathcal{L}_{\text{safe}}$ and $\mathcal{L}_{CE}$ is linked to the orthogonality of refusal/reasoning subspaces, providing a geometric explanation for why safety does not necessarily require a utility trade-off.

## Limitations & Future Work
- High-risk anchor pools must be pre-prepared and maintained; the cost of updating anchors for emerging attack types (e.g., multimodal, long-chain reasoning) is not quantified.
- Primarily validated at the 7B scale (Llama3.1-8B, Qwen2.5-7B, Gemma1.1-7B); whether 1 anchor remains sufficient as bottleneck dimension $d$ grows in 70B+ or MoE models is unclear.
- If attackers can directly fine-tune the unembedding matrix $w_t$ or induce the model to bypass the last layer via prompts, the geometric bottleneck assumption of SBR may fail.
- The interaction with continual learning or multi-task fine-tuning is not discussed: would anchors drift after long-term superposition of multiple benign tasks?
- The parameter $\lambda=50$ is empirical; different models or tasks may require parameter searching.

## Related Work & Insights
- **vs. Lisa / EWC**: Both constrain weight distance in parameter space. This paper proves high-dimensional redundancy makes this path inherently prone to failure. SBR avoids the null space by moving constraints to the unembedding input layer.
- **vs. Vaccine / T-Vaccine**: These monitor representation drift across entire layers, but the authors find drift and HS are decoupled. SBR is more precise by anchoring only the last layer and final token.
- **vs. Booster / Gradient-based**: These attempt to mask harmful gradient directions, but this paper proves harmful directions are "ubiquitous" in parameter space, making sparse masking ineffective.
- **vs. DeepAlign**: Imposes constraints on output tokens, which can have side effects on short outputs (classification). SBR constrains hidden states and is insensitive to token length.
- **Inspiration**: All research involving internal representation constraints for alignment, unlearning, or watermarking can benefit from "identifying geometric bottlenecks"—placing constraints where redundancy is minimal yields stronger robustness with fewer anchors.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Completely relocates HFT defense from "parameter space" to the unembedding bottleneck, proposing a new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three stress tests + 4 tasks + 4 poisoning ratios + $\lambda$ sensitivity + 3 base LLMs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The three-stage motivation in Section 3 is highly logical; the method is explained concisely, and diagrams are persuasive.
- **Value**: ⭐⭐⭐⭐⭐ 1 anchor is enough to resist attacks without utility loss, making it industry-friendly and compatible with existing pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)
- [\[ICLR 2026\] Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](../../ICLR2026/llm_alignment/safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](curriculum_learning_for_safety_alignment.md)
- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[ICML 2026\] MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)

</div>

<!-- RELATED:END -->
