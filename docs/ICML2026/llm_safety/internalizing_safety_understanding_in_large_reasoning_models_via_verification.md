---
title: >-
  [Paper Note] Internalizing Safety Understanding in Large Reasoning Models via Verification
description: >-
  [ICML 2026][LLM Safety][Safety Alignment] This paper argues that "being able to generate safe answers" ≠ "understanding safety," and proposes the SInternal framework: training large reasoning models solely to verify the…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Safety Alignment"
  - "Reasoning Models"
  - "Self-Verification"
  - "Jailbreak Defense"
  - "SFT Initialization"
date: 2026-05-08
content_hash: 63eb1b586e588e64
---

# Internalizing Safety Understanding in Large Reasoning Models via Verification

**Conference**: ICML 2026  
**arXiv**: [2605.08930](https://arxiv.org/abs/2605.08930)  
**Code**: https://github.com/AlphaLab-USTC/SInternal (available)  
**Area**: LLM Reasoning / Safety Alignment  
**Keywords**: Safety Alignment, Reasoning Models, Self-Verification, Jailbreak Defense, SFT Initialization

## TL;DR
This paper argues that "being able to generate safe answers" ≠ "understanding safety," and proposes the SInternal framework: training large reasoning models solely to verify the safety of their own generated answers. The resulting emergent internal safety understanding significantly suppresses jailbreak attacks (StrongREJECT ASR drops from 41% to 0.6%) and provides a better starting point for subsequent RL.

## Background & Motivation

**Background**: Large reasoning models (LRMs, e.g., DeepSeek-R1) with explicit CoT make final answers more dangerous. The current mainstream alignment paradigm is answer-centric: either SFT on expert-curated "safe trajectories," or RL using a safety verifier to score final answers.

**Limitations of Prior Work**: The authors conducted a simple experiment—having an aligned LRM judge "whether a candidate answer to a prompt is safe." The result is concerning: DeepSeek-R1-Distill-Qwen-7B, after SFT + RLVR, performs worse than random guessing on this binary classification task (see Figure 2). In other words, the model learns to "output answers that look safe," but does not truly understand why they are safe.

**Key Challenge**: Current alignment decouples "execution" and "judgment"—outsourcing judgment entirely to external guardrails like Llama Guard, while the generator only learns to mimic surface patterns. This leads to extreme vulnerability to unseen jailbreaks: as long as an attacker hijacks the chain of thought with a compliant CoT, the model can be tricked into believing "this prompt is safe" and produce harmful answers.

**Goal**: Enable the model to internalize the ability to judge "why this answer is unsafe," rather than just learning "how to refuse."

**Key Insight**: "Being able to judge" is a stronger prerequisite for "being able to execute"—if the model can truly verify whether an answer violates the safety spec, it naturally knows what kind of answers should be produced. Thus, the training objective is flipped from "generating safe answers" to "verifying whether its own generated answers are safe."

**Core Idea**: Train LRMs solely with verification SFT to evaluate their own outputs. The emergent internal safety understanding both suppresses jailbreaks and provides a more robust foundation for subsequent RL.

## Method

### Overall Architecture
SInternal consists of two steps: (1) Data construction—For each safety-related prompt $\mathbf{x}$, the initial policy $\pi_\theta$ samples $N=8$ answers $(\mathbf{z}_k,\mathbf{y}_k)$. Claude-4-Sonnet, as the expert, evaluates each $\mathbf{y}_k$ according to the safety spec $\mathcal{S}$, producing a verification trajectory $\mathbf{c}_k=(\mathbf{z}_{{\rm ver},k},\mathbf{v}_k)$ containing critical reasoning and a binary judgment. (2) SFT optimization—The objective is to predict $\mathbf{c}$ given $(\mathcal{S},\mathbf{x},\mathbf{y})$, with loss $\mathcal{L}_{\rm SInternal}=-\mathbb{E}\log\pi_\theta(\mathbf{c}|\mathcal{S},\mathbf{x},\mathbf{y})$. Optionally, GRPO RLVR can be run on top of SInternal for further alignment of generation behavior.

### Key Designs

1. **Verifying Self-Generated Rather Than External Answers**:

    - **Function**: Uses the model's own sampled answers (including potentially unsafe ones) as verification training targets, ensuring verification ability matches the model's actual distribution.
    - **Mechanism**: For each harmful prompt, sample $N=8$ answers, retaining prompts containing both safe and unsafe answers, and select a pair for contrastive samples; for benign prompts, keep one sample. The training set contains about 6000 samples. Ablation between Self-Exp (own trajectories) and Other-Exp (trajectories from other models) shows self-generated is consistently better.
    - **Design Motivation**: Using other models' answers teaches verification about "mistakes others make," which mismatches the model's own distribution; having the model verify its own common mistakes aligns the safety boundary with its own behavior distribution.

2. **Expert Critique + Binary Judgment Dual-Component Trajectory**:

    - **Function**: Translates external safety specs into learnable "analysis + judgment" natural language trajectories, forcing the model to perform explicit reasoning during verification.
    - **Mechanism**: Claude-4-Sonnet, as expert, takes spec $\mathcal{S}$, prompt $\mathbf{x}$, and answer $\mathbf{y}_k$, and outputs: (a) critique reasoning $\mathbf{z}_{\rm ver}$ with detailed analysis of potential violations; (b) binary judgment $\mathbf{v}$ (safe/unsafe). Ablation shows critique mainly enables generalization to unseen jailbreaks (removing critique increases Fortress ASR from 19.2% to 46.8%), while judgment mainly stabilizes in-domain performance (removing judgment increases StrongREJECT ASR from 0.6% to 7.3%).
    - **Design Motivation**: A single binary label provides too little information, leading the model to learn only surface patterns; explicit critique offers reasoning supervision on "why unsafe," forcing the model to learn underlying safety concepts rather than memorizing refusal templates.

3. **SInternal as Initialization for Subsequent RL**:

    - **Function**: After SInternal SFT, GRPO RLVR is applied to build a more robust alignment foundation than standard SFT.
    - **Mechanism**: The reward function uses $r=\mathcal{V}_{\rm safe}$ for harmful prompts, and $r=\mathcal{V}_{\rm safe}(1-\mathcal{V}_{\rm refuse})$ for benign prompts to avoid over-refusal; Qwen3-Guard serves as verifier. GRPO optimizes $\hat{A}_i=(r_i-\bar{r})/(\sigma_r+\epsilon)$. RL initialized from SInternal is the only one able to defend against HCoT (the strongest LRM-specific jailbreak); other baseline RLs fail.
    - **Design Motivation**: Standard SFT only pushes the model toward "looking safe," making it hard to stably interpret reward signals during RL; SInternal equips the model with "why" understanding, enabling more stable RL fine-tuning.

### Loss & Training

Stage 1: Standard SFT cross-entropy $\mathcal{L}_{\rm SInternal}=-\mathbb{E}_{(\mathbf{x},\mathbf{y},\mathbf{c})\sim\mathcal{D}_{\rm ver}}\log\pi_\theta(\mathbf{c}|\mathcal{S},\mathbf{x},\mathbf{y})$, with about 6000 training samples, trained for 2 epochs using LoRA (rank=16, $\alpha=32$), learning rate $2\times10^{-4}$.  
Stage 2: GRPO, rollout batch of 64 prompts × $n=8$, actor learning rate $10^{-6}$, KL penalty off, and 3k DAPO math problems retained to preserve reasoning ability.

## Key Experimental Results

### Main Results
Three LRMs (DS-Qwen-7B / DS-Llama-8B / DS-Qwen-14B) × 9 benchmarks (3 safety, 1 overrefusal, 2 reasoning), with baselines including SafeChain and STAR-1.

| Configuration | StrongREJECT (ASR↓) | Fortress (ASR↓) | WildJailbreak (ASR↓) | HCoT (ASR↓) | XSTest (CR↑) | AIME (↑) |
|--------|------|------|------|------|------|------|
| DS-14B Base | 41.2 | 52.6 | 44.4 | 100.0 | 95.6 | 86.7 |
| DS-14B + SafeChain SFT | 24.9 | 48.2 | 45.2 | 100.0 | 99.6 | 83.3 |
| DS-14B + STAR-1 SFT | 0.6 | 28.2 | 18.4 | 100.0 | 94.0 | 83.3 |
| **DS-14B + SInternal SFT** | **0.6** | **19.2** | **6.8** | 90.0 | 98.0 | 86.7 |
| DS-14B + STAR-1 + GRPO | 0.0 | 7.8 | 3.6 | 98.0 | 96.0 | 80.0 |
| **DS-14B + SInternal + GRPO** | **0.0** | **5.2** | **0.4** | **62.0** | 99.2 | 80.0 |

### Ablation Study

| Configuration | StrongREJECT | Fortress | WildJailbreak | Notes |
|------|------|------|------|------|
| Full SInternal | 0.6 | 19.2 | 6.8 | Full version |
| w/o critique | 2.9 | 46.8 | 22.4 | Remove reasoning, keep only binary judgment |
| w/o judgment | 7.3 | 18.8 | 7.6 | Remove binary judgment, keep only critique |
| Self-Exp (DS-7B) | 7.0 | 22.6 | 21.6 | Verify self-sampled trajectories |
| Other-Exp (DS-7B) | 9.6 | 27.4 | 27.6 | Use trajectories sampled by DS-8B |

### Key Findings
- **Verification training transfers to generation**: SFT only on the verification task leads to a dramatic drop in ASR on generation tasks—indicating that "learning to verify" implicitly includes the ability to "generate safe answers."
- **Generalizes to unseen jailbreaks**: SInternal may not always be first on in-domain StrongREJECT (0.6 vs STAR-1 0.6 tie), but consistently leads on OOD Fortress and LRM-specific HCoT/Trotter, indicating concept learning rather than pattern memorization.
- **Emergent proactive verification**: Using GPT-4o to detect spontaneous safety verification in CoT, SInternal triggers at 50.4% vs Base 16.0% / STAR-1 28.4%, and after triggering, the conditional safe rate is 99.2%.
- **High data efficiency**: SInternal achieves or surpasses baseline full-data performance using only 50% of the SFT baseline data.
- **Preserves reasoning ability**: SInternal does not lose points on MATH/AIME, proving that safety alignment does not sacrifice reasoning.

## Highlights & Insights
- The conceptual flip that "verification is a necessary prerequisite for generation" deserves serious attention from the alignment community and can be extended to other alignment dimensions such as helpfulness and honesty.
- Constructing contrastive pairs (one safe, one unsafe) from the model's own answers automates DPO-style preference data generation, eliminating manual annotation.
- The division of labor—critique for generalization, judgment for in-domain stability—is intriguing and can inspire future safety dataset designs to include both "reasoning + label" components.
- Only SInternal+GRPO can defend against CoT-hijack attacks like HCoT, indicating that "the model truly understanding the consequences of final behavior" is key to resisting CoT manipulation.

## Limitations & Future Work
- Current verification is only performed post-generation, not extended to "dynamic self-verification during generation"—an obvious open direction.
- Verification ability is still weaker than generation: sometimes the model can generate a safe answer but misjudge during verification; the gap is not fully closed.
- Reliance on Claude-4-Sonnet as the expert for generating critiques may amplify biases if the expert itself is biased.
- Experiments are all on the DeepSeek-R1-Distill series, not validated on o1 / Claude Thinking or other closed-source LRMs.
- HCoT ASR remains at 62% after 14B+GRPO, far from complete defense.

## Related Work & Insights
- **vs SafeChain (Jiang et al. 2025)**: SafeChain distills long CoT safety reasoning but remains answer-centric; this paper shows that training only on verification generalizes better (Fortress ASR 19.2 vs 48.2).
- **vs STAR-1 (Wang et al. 2025)**: STAR-1 also uses deliberate reasoning over safety specs, but its training objective is to directly generate safe answers; this paper flips to a verification objective, yielding more stable performance.
- **vs Llama Guard / Qwen3-Guard external guardrails**: External guardrails outsource judgment, so the model only learns surface imitation; this paper demonstrates that internalizing judgment is essential for robust safety.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Flipping the training objective from "generation" to "verification" is a true conceptual shift, strongly supported by experiments
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 models × 9 benchmarks + self/other sampling ablation + critique/judgment split + spec replacement + data efficiency, very comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, but some formula formatting (reward function with align block) is messy
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for LRM safety alignment, code is open source, and can be directly reused by the alignment community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](../../ACL2026/llm_safety/autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](../../ACL2026/llm_safety/reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](../../ACL2026/llm_safety/when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](../../ACL2026/llm_safety/how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)

</div>

<!-- RELATED:END -->
