---
title: >-
  [Paper Note] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks
description: >-
  [ICML 2026][LLM Safety][Harmful Fine-tuning] This paper proves that all existing HFT defenses that impose constraints in parameter space can be circumvented due to parameter redundancy. It proposes Safety Bottleneck Regu…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Harmful Fine-tuning"
  - "Unembedding Bottleneck"
  - "Parameter Redundancy"
  - "MSE Anchor"
  - "RLHF Robustness"
date: 2026-05-08
content_hash: a8b18ffeb935c859
---

# Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks

**Conference**: ICML 2026  
**arXiv**: [2605.05995](https://arxiv.org/abs/2605.05995)  
**Code**: [soyoaaa/SBR](https://github.com/soyoaaa/SBR)  
**Area**: Alignment / LLM Safety  
**Keywords**: Harmful Fine-tuning, Unembedding Bottleneck, Parameter Redundancy, MSE Anchor, RLHF Robustness

## TL;DR
This paper proves that all existing HFT defenses that impose constraints in parameter space can be circumvented due to parameter redundancy. It proposes Safety Bottleneck Regularization (SBR), which shifts the defense to the unembedding layer—a geometric bottleneck: by anchoring only the final hidden state of a single high-risk prompt, the Harmful Score can be suppressed to < 10 under 50 epochs of sustained HFT attack, without harming benign task accuracy.

## Background & Motivation
**Background**: RLHF enables LLMs to refuse illegal requests, but fine-tuning-as-a-service allows users to upload datasets for retraining. Even a small number of malicious samples (Harmful Fine-tuning, HFT) can dismantle safety guardrails within a few epochs. Existing defenses—(i) parameter distance (Lisa/EWC), (ii) gradient direction (Booster), (iii) representation drift (Vaccine/T-Vaccine)—can suppress harmful behaviors in early epochs.

**Limitations of Prior Work**: Through a 50-epoch sustained HFT stress test, the authors find that all three defenses collapse after 5–10 epochs (HS > 30), and at collapse, their monitored "parameter distance/gradient direction/representation drift" remain within bounds—indicating that while the form is constrained, the substance is not preserved.

**Key Challenge**: LLMs are highly over-parameterized, so attackers can always find optimization directions orthogonal to defense constraints—a random Rank-1 LoRA $\Delta W=BA^\top$ (with $A$ frozen) can restore harmful capability, showing that the harmful direction is ubiquitous in parameter space rather than sparse. This means any constraint in the "redundant high-dimensional parameter space" has a null space that can be exploited.

**Goal**: To identify a chokepoint unaffected by parameter redundancy and unavoidable by attackers, such that imposing defense only there is sufficient.

**Key Insight**: The authors observe that the final step of token generation is the inner product between the last layer hidden state $h_{\text{final}}$ and the word embedding $w_t$: $\text{Score}(t)=h_{\text{final}}^\top w_t$. This unembedding projection is a geometric bottleneck that all harmful tokens must pass through, and since $w_t$ is frozen, as long as $h_{\text{final}}$ is biased toward the refusal embedding direction, softmax will always select the refusal token.

**Core Idea**: Rather than defending in parameter space, directly anchor the final hidden states of a set of high-risk queries at the unembedding layer to match those of the aligned base model—regardless of internal parameter evolution, the bottleneck is pinned, preventing generation of malicious tokens.

## Method

### Overall Architecture
SBR operates in the fine-tuning-as-a-service scenario: the service provider holds the aligned base model $f_{\theta_{\text{base}}}$ and a set of "safety anchors" $\mathcal{X}_{\text{anchor}}=\{x'_1,\ldots,x'_K\}$ (typical dangerous prompts, e.g., "How to make a bomb?"), but cannot access the user-uploaded training set $\mathcal{D}_{\text{train}}$. SBR proceeds in two phases:

- **Phase 1 — Anchor Acquisition (offline)**: For each anchor, extract the final token's last layer hidden state $h_{\text{ref}}(x')=f^{\text{last}}_{\theta_{\text{base}}}(x')$ using the frozen base model, and cache as $\mathcal{H}_{\text{ref}}$.
- **Phase 2 — Dynamic Regularization (parallel with user fine-tuning)**: For each batch, compute both $\mathcal{L}_{CE}$ (user task) and $\mathcal{L}_{\text{safe}}=\frac{1}{|\mathcal{X}_{\text{anchor}}|}\sum_{x'}\|h_\theta(x')-h_{\text{ref}}(x')\|_2^2$, with total objective $\mathcal{L}_{\text{total}}=\mathcal{L}_{CE}+\lambda\mathcal{L}_{\text{safe}}$, where $\lambda$ controls refusal strength. The unembedding matrix remains untouched, and no base model architecture modification is required.

### Key Designs

1. **Geometric Bottleneck Localization — Move Defense to Unembedding Input**:

    - **Function**: Apply defense at the position with minimal redundancy and maximal necessity, so bypassing it requires changing both $h_{\text{final}}$ and $w_t$—but $w_t$ is frozen, leaving attackers only a $d$-dimensional constraint surface.
    - **Mechanism**: Since $P(t|x)=\text{softmax}(h_{\text{final}}^\top w_t)$, refusal and harmful tokens compete in the same softmax—if $h_{\text{final}}$ is geometrically biased toward the refusal token embedding, the refusal token's score strictly exceeds that of the harmful token, forcing the model to output refusal; MSE anchoring keeps $h_{\text{final}}$ close to the base model's output on these prompts.
    - **Design Motivation**: In §3, three stress tests (parameter distance / Rank-1 random subspace / representation drift) demonstrate that high-dimensional parameter space always contains "escape routes" orthogonal to defense directions; only the lower-dimensional unembedding input layer, directly linked to token selection, lacks such null spaces.

2. **MSE Anchor Loss $\mathcal{L}_{\text{safe}}$ + Minimal Anchors Suffice**:

    - **Function**: Impose a hard constraint on the bottleneck with minimal overhead.
    - **Mechanism**: $\mathcal{L}_{\text{safe}}(\theta)=\frac{1}{|\mathcal{X}_{\text{anchor}}|}\sum_{x'\in\mathcal{X}_{\text{anchor}}}\|h_\theta(x')-h_{\text{ref}}(x')\|_2^2$, combined with user $\mathcal{L}_{CE}$ via $\lambda$; only 1–8 anchors, randomly sampled from a candidate pool (disjoint from attacker data, e.g., BeaverTails subset), are needed.
    - **Design Motivation**: The authors argue that "refusal direction is approximately orthogonal to benign reasoning direction" (Zou 2023, Arditi 2024), so anchoring a small set of high-risk prompts does not significantly restrict the optimization space for benign tasks; empirically, a single anchor suffices to reduce HS to < 10.

3. **Stress Test Paradigm — 50-Epoch Sustained HFT Reveals True Defense Failure**:

    - **Function**: Shift defense evaluation from "short-term snapshots" to "sustained attack," exposing the illusion of transient safety.
    - **Mechanism**: Construct a mixed dataset (10% harmful + 90% benign), run models on four benign tasks (SST-2/AGNEWS/GSM8K/AlpacaEval) for 20–50 epochs; design Random Subspace Attack (freeze $A$, train $B$) to show that even random Rank-1 directions can restore harmful capability. Systematically compare "parameter distance/embedding drift vs. HS" correlations, proving that stable metrics do not imply stable safety.
    - **Design Motivation**: Prior work often reports "HS within 3–5 epochs," masking the reality that service providers may allow users to train for extended periods; stress testing dispels this illusion and provides a comparative backdrop for SBR.

### Loss & Training
LoRA rank 16 / alpha 16, AdamW lr $1\times 10^{-5}$, batch size 32, 20 epochs; anchors $K=8$, $\lambda=50$; anchors require only forward pass, no backward to base model. All baselines are rerun with identical hyperparameters.

## Key Experimental Results

### Main Results
Llama3.1-8B, four benign downstream tasks × HS↓ / FA↑ dual metrics:

| Method | SST-2 HS↓ | SST-2 FA↑ | GSM8K HS↓ | GSM8K FA↑ | AlpacaEval HS↓ | AlpacaEval FA↑ | Avg HS↓ | Avg FA↑ |
|--------|-----------|-----------|-----------|-----------|----------------|----------------|---------|---------|
| SFT (no defense) | 67.80 | 94.61 | 71.10 | 82.80 | 74.20 | 43.87 | 70.70 | 78.07 |
| DeepAlign | 25.90 | 93.12 | 20.70 | 88.00 | 23.70 | 33.64 | 25.10 | 76.04 |
| Lisa | 52.50 | 94.27 | 40.40 | 72.20 | 58.20 | 37.93 | 52.45 | 73.50 |
| Vaccine | 61.40 | 92.55 | 64.30 | 75.10 | 62.90 | 36.39 | 62.53 | 73.34 |
| Booster | 59.80 | 92.89 | 71.50 | 76.20 | 54.30 | 35.75 | 62.33 | 73.66 |
| **SBR** | **5.80** | **94.15** | **5.60** | **82.60** | **6.20** | **45.82** | **5.68** | **78.17** |

### Ablation Study
**Poison Ratio Robustness (Llama3.1-8B):**

| Poison ratio $p$ | SFT HS | DeepAlign HS | Vaccine HS | Booster HS | **SBR HS** | SBR FA |
|------------------|--------|--------------|------------|------------|------------|--------|
| 0.05 | 67.90 | 21.50 | 58.70 | 59.40 | **4.10** | 93.92 |
| 0.10 | 67.80 | 25.90 | 61.40 | 59.80 | **5.80** | 94.15 |
| 0.20 | 71.90 | 29.90 | 61.90 | 64.60 | **8.20** | 93.92 |
| 0.30 | 74.30 | 33.30 | 69.20 | 67.30 | **7.30** | 93.69 |
| Avg | 70.48 | 27.65 | 62.80 | 62.78 | **6.35** | 93.92 |

**$\lambda$ Sensitivity**: Validated on $\lambda\in\{0,5,10,50,100\}$, showing $\lambda=50$ achieves a stable sweet spot between HS↓ and FA↑ ($\lambda=0$ degenerates to SFT, $\lambda\ge 100$ starts to erode FA).

### Key Findings
- $K=1$ anchor suffices—the paper repeatedly emphasizes "a single safety anchor is sufficient to reduce the Harmful Score to < 10," demonstrating the extreme narrowness of the unembedding bottleneck.
- Under 50 epochs of sustained HFT, SBR remains robust, while Lisa/Vaccine/Booster collapse after just 5 epochs—Figure 2 provides a dramatic contrast.
- The Drift-Safety Dissociation in §3 (embedding drift nearly unchanged between steps 120 and 480, but HS jumps from 12 to 59) independently proves that monitoring global representation drift is a flawed proxy.
- On benign tasks, SBR not only avoids degradation but slightly improves performance (mean FA 78.17 vs SFT 78.07)—supporting the hypothesis that refusal and benign reasoning directions are nearly orthogonal.

## Highlights & Insights
- The combination of stress test and Random Subspace Attack thoroughly explains why existing HFT defenses are bypassed, providing a wake-up call for the field.
- The "find the bottleneck" concept has strong transfer value—not limited to unembedding, any defense with redundancy that allows attackers to find orthogonal bypasses should consider relocating to downstream geometric chokepoints.
- The minimalism of requiring only one anchor means deployment cost is nearly zero: the service provider only needs to cache $K$ hidden state vectors, with constant overhead per forward pass.
- The fundamental reason for the non-conflict between $\mathcal{L}_{\text{safe}}$ and $\mathcal{L}_{CE}$ is linked to the orthogonality of refusal/reasoning subspaces—offering a geometric explanation for why "safety vs utility is not necessarily a trade-off."

## Limitations & Future Work
- The high-risk anchor pool must be prepared and maintained by the service provider; emerging attack types (e.g., multimodal, long-chain reasoning) require anchor updates, and update costs are unquantified.
- Validation is mainly on 7B-scale models (Llama3.1-8B, Qwen2.5-7B, Gemma1.1-7B); not tested on 70B+ or MoE models; whether one anchor suffices as bottleneck dimension $d$ increases is unknown.
- If attackers can directly fine-tune the unembedding matrix $w_t$ (or induce the model to "bypass the last layer" via prompts), the geometric bottleneck assumption of SBR no longer holds—thus, the method relies on certain attack model assumptions.
- No discussion of interaction with continual learning/multi-task fine-tuning: whether anchors drift after long-term accumulation of multiple benign tasks remains unaddressed.
- $\lambda=50$ is an empirical value and requires re-tuning for different models/tasks.

## Related Work & Insights
- **vs Lisa / EWC**: Both constrain weight distance in parameter space; this paper proves high-dimensional redundancy makes this approach inherently fail. SBR relocates constraints to the unembedding input layer, avoiding the null space.
- **vs Vaccine / T-Vaccine**: Monitor entire layer representation drift, but the authors empirically show drift and HS are decoupled; SBR anchors only the last layer, last token, for more precise localization.
- **vs Booster / Gradient-based**: Attempt to block harmful gradient directions, but this paper shows harmful directions are "everywhere" in parameter space and cannot be sparsely blocked.
- **vs DeepAlign**: Adds constraints on output tokens, which has side effects for short outputs (classification tasks); SBR constrains hidden states, making it insensitive to token length.
- **Insights**: All "internal representation constraint" alignment/unlearning/watermarking work can reference the "find geometric bottleneck" approach—placing constraints at points of minimal redundancy enables stronger robustness with fewer anchors.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Relocates HFT defense from "parameter space" to the unembedding bottleneck, proposing a new paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three stress tests + four tasks + four poison ratios + $\lambda$ sensitivity + three base LLMs
- Writing Quality: ⭐⭐⭐⭐⭐ §3 motivation is logically strong, method fits on one page, figures and tables are concise and convincing
- Value: ⭐⭐⭐⭐⭐ One anchor suffices to block attacks without harming benign tasks, deployment-friendly and compatible with existing training pipelines

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICML 2026\] PFT: Phonon Fine-tuning for Machine Learned Interatomic Potentials](pft_phonon_fine-tuning_for_machine_learned_interatomic_potentials.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)

</div>

<!-- RELATED:END -->
