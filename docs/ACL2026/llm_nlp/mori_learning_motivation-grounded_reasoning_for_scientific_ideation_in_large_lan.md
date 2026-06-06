---
title: >-
  [Paper Note] MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Scientific Ideation] Explicitly models scientific ideation as a two-stage conditional reasoning task of 「context → motivation → reasoning → method」. Based on an SFT cold start…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Scientific Ideation"
  - "Motivation-grounded Reasoning"
  - "GRPO"
  - "Entropy-Aware Information Gain"
  - "Contrastive Semantic Gain"
date: 2026-05-08
content_hash: ca6ec18f67b7fe92
---

# MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2603.19044](https://arxiv.org/abs/2603.19044)  
**Code**: See paper GitHub (mentioned as available, no specific URL provided)  
**Area**: Scientific Ideation / LLM Reasoning / RL for Reasoning  
**Keywords**: Scientific Ideation, Motivation-grounded Reasoning, GRPO, Entropy-Aware Information Gain, Contrastive Semantic Gain  

## TL;DR
Explicitly models scientific ideation as a two-stage conditional reasoning task of 「context → motivation → reasoning → method」. Based on an SFT cold start, a 14B model is trained using GRPO with two novel verifiable rewards (**Entropy-Aware Information Gain EAIG** + **Contrastive Semantic Gain CSG**). It outperforms GPT-4o, Claude-3.5-Sonnet, and agentic frameworks like AI-Scientist-V2 on held-out ICLR/NeurIPS test sets.

## Background & Motivation

**Background**: LLMs are evolving from chatbots to "scientific assistants/autonomous researchers," with scientific ideation (input research context, output new methods) regarded as the most upstream task. Existing solutions primarily rely on **agentic pipelines**—AI-Scientist-V2, ResearchAgent, and VirSci use multi-agent debate, tree search, or peer review to simulate human research workflows.

**Limitations of Prior Work**: (1) These agentic frameworks essentially use heuristic scaffolding to "patch" base LLMs into researchers without improving the model's intrinsic scientific reasoning capability, often resulting in **superficial conceptual recombinations** lacking technical depth; (2) Large-scale human evaluations (Si et al. 2024 / Kumar et al. 2025) confirm that while base LLMs generate "new ideas," they lack implementation details and feasibility; (3) Sutton’s Bitter Lesson warns that external scaffolding is unsustainable; reasoning must be internalized.

**Key Challenge**: Scientific ideation has no "standard answer"—the same context can lead to multiple valid Method sections, which vary significantly in literal wording. Consequently, RL training faces three difficulties: (a) Absence of a deterministic verifier (unlike math/code with unit tests); (b) High noise in rewards based on reference probability or perplexity; (c) LLM-as-a-judge rewards are expensive and prone to reward hacking.

**Goal**: Abandon agentic scaffolding and instead **use RL to internalize scientific reasoning** into the model—enabling it to explicitly learn the chain of thought from "research motivation to method."

**Key Insight**: High-quality method sections contain "unpredictable, high-information" tokens that carry the "hard knowledge" of innovation (e.g., specific algorithms/parameters), while boilerplate like "we propose to" provides near-zero information. Furthermore, a method should move the model **from the context toward the ground-truth method** in semantic space, rather than simply restating the context.

**Core Idea**: Explicitly decompose ideation into two stages $x \to m \to (z, y)$, and use two **complementary verifiable rewards** for RL—EAIG (measuring if reasoning truly improves prediction on high-entropy tokens of the ground-truth method) and CSG (using a contrastive baseline to measure if the generated method truly "advances" in semantic space), supplemented by length anchoring and format constraints to prevent reward hacking.

## Method

### Overall Architecture
MoRI is a three-stage pipeline (Figure 2 + 7):
(1) **Data Construction**: Extract context $x$, motivation $m$, and de-symbolized method $y^*$ from ICLR 2024-2025 papers. Use "posterior reconstruction"—let Qwen3-235B-Thinking generate initial reasoning $z_{\text{init}}$ from context alone, then let Qwen3-235B-Instruct rewrite $z_{\text{init}}$ to align with ground-truth $z$ given $(x, y^*)$. Reverse-engineer 4,000 SFT samples + 2,000 RL prompt samples.
(2) **SFT Cold Start**: Train DeepSeek-R1-Distilled-Qwen-14B on two tasks—motivation generation ($x \to m$) and method generation ($x \to (z, y)$).
(3) **Motivation-grounded RL**: Use GRPO with token-level loss and clip-higher optimization. Each prompt is formed as $q = x \oplus m$, rolling out $G=16$ trajectories $(z_i, y_i)$. Calculate group-normalized advantage based on the composite reward.
**Inference**: Generate motivation $m$ first, then reason and output the method conditioned on $x \oplus m$.

### Key Designs

1. **Entropy-Aware Information Gain EAIG (Micro-level Technical Depth Reward)**:
    - **Function**: Measures whether "reasoning makes the model predict more accurately" specifically on the "hard" tokens of the ground-truth method, focusing rewards on technical details rather than boilerplate.
    - **Mechanism**: First, use a fixed SFT model via teacher-forcing to calculate entropy at each position $H_t = -\sum_v \pi_{\text{sft}}(v \mid x, m, y^*_{<t}) \log \pi_{\text{sft}}(\cdot)$. Select the top-25% highest-entropy tokens to form a mask $\mathcal{M}_t$ (34.2% of technical terms are selected vs. 14.5% of common words). Calculate per-token gain $g_t(z) = \log \pi_\theta(y^*_t \mid x, m, z, y^*_{<t}) - \log \pi_{\text{sft}}(y^*_t \mid x, m, y^*_{<t})$, resulting in $\Delta_{IG}(z) = \frac{1}{\sum \mathcal{M}_t} \sum_t \mathcal{M}_t \cdot g_t(z)$.
    - **Design Motivation**: Solves the core difficulty of no deterministic verifiers in open tasks—ignoring full method probability (noisy, bias toward brevity) and focusing on relative improvement on "hard tokens." Unlike Wang et al. 2025b, MoRI applies the mask to the **ground-truth method** rather than the reasoning trace, fundamentally suppressing reward hacking.

2. **Contrastive Semantic Gain CSG (Macro-level Logical Direction Reward)**:
    - **Function**: Ensures the generated method moves toward the ground-truth solution space rather than just piling up details.
    - **Mechanism**: Use Qwen3-Embedding-8B embeddings $\mathbf{E}(\cdot)$ to calculate $S_{gen} = \cos(\mathbf{E}(\hat{y}), \mathbf{E}(y^*))$ and a **counterfactual baseline** $S_{base} = \cos(\mathbf{E}(x \oplus m), \mathbf{E}(y^*))$ (the similarity gained by restating the input). Define gain as $\Delta_{sem} = S_{gen} - S_{base}$. $\Delta_{sem} > 0$ implies the model pushed the semantic focus from the problem space to the solution space.
    - **Design Motivation**: Using $\cos(\hat{y}, y^*)$ alone rewards lazy strategies like paraphrasing the context. By introducing $S_{base}$ as a contrastive baseline, the model must make a "real semantic jump" to receive a positive reward.

3. **Length Anchoring + Format Constraints (Preventing Reward Collapse)**:
    - **Function**: Prevents GRPO from inducing reward hacking such as "reasoning chain shrinking" or "entropy injection" under high-variance EAIG.
    - **Mechanism**: Use a length modulation factor $\alpha(z) = \min(1, 1 - \lambda \frac{L_{anchor} - |z|}{L_{anchor}})$. Rewards are discounted when $|z| < L_{anchor}$, forcing the model to maintain reasoning depth. Format indicator $\mathds{1}[\text{valid}]$ requires CoT to be non-empty, ≥ 1000 characters, and **void of `##`/`###`** (preventing smuggling method content into reasoning). Final reward $R_{\text{total}} = \alpha(z) \cdot \mathds{1}[\text{valid}] \cdot (w_e f_{\text{step}}(\Delta_{IG}) + w_s f_{\text{step}}(\Delta_{sem}))$. Optimal weights: $w_s = 0.7, w_e = 0.3$.
    - **Design Motivation**: Theoretical analysis in Appendix F shows GRPO group normalization implicitly favors low-variance strategies; since long reasoning chains have higher variance, they are suppresses. Length anchoring provides a positive gradient to offset this bias.

### Loss & Training
GRPO with token-mean loss, $\varepsilon_{\text{low}}/\varepsilon_{\text{high}} = 0.2/0.28$ (clip-higher), KL coefficient 0.001, rollout $G=16$, lr $5 \times 10^{-7}$ with 10% warmup cosine, global batch 8, temperature 1.0, max prompt/response 5000 each. Reward step-shaping quantifies continuous gains into 4 levels to resist noise: EAIG thresholds $[1.0, 1.5, 2.0]$, CSG thresholds $[0.01, 0.05, 0.1]$, reward levels $[0, 0.5, 0.8, 1.0]$.

## Key Experimental Results

### Main Results
Tested on 83 ICLR 2025 in-domain and 67 NeurIPS 2025 OOD papers (Metrics: Novelty / Tech Rigor / Feasibility / Mean, scale 1-5, Gemini-2.5-Pro RAG LLM-judge + 3 PhD human evaluators):

| Model / Framework | ICLR Mean | NeurIPS OOD Mean | Combined Mean | Δ vs MoRI |
|------------|-----------|------------------|---------------|-----------|
| GPT-4o | 2.69 | 2.77 | 2.74 | +16.1% |
| Claude-3.5-Sonnet | 3.09 | 3.13 | 3.11 | +2.3% |
| AI-Scientist-V2† (Sonnet, ICLR only) | 3.14 | – | – | +1.6% |
| AI-Scientist-V2 (GPT-4o) | 2.71 | 2.48 | 2.60 | +22.3% |
| ResearchAgent | 2.60 | 2.37 | 2.50 | +27.2% |
| VirSci | 2.25 | 2.23 | 2.24 | +42.0% |
| Full-SFT (x → m → y) | 2.99 | 2.85 | 2.93 | +8.5% |
| **Ours (MoRI)** | **3.19** | **3.15** | **3.18** | — |
| Ground Truth (oracle) | 3.58 | 3.59 | 3.59 | – |

On ICLR, MoRI's Feasibility (3.11) is 10.3% higher than Claude-3.5-Sonnet (2.82), and Rigor is 2.9% higher. The only dimension where Claude Sonnet (3.39) and AI-Scientist-V2†(Sonnet) (3.45) lead is Novelty. Human-LLM correlation Pearson $r = 0.715$ ($p < 0.001$), with 95% bootstrap CI [3.11, 3.25] showing no overlap with any agentic baseline.

### Ablation Study

| Configuration | Novelty | Rigor | Feas. | Mean | Phenomenon |
|------|---------|-------|-------|------|------|
| Full-SFT (direct $x \to y$) | 2.80 | 2.69 | 2.81 | 2.75 | No explicit motivation modeling |
| Full-SFT (two-stage $x \to m \to y$) | 2.85 | 2.76 | 2.94 | 2.85 | +0.10, motivation conditioning helps |
| MoRI (full) | 3.30 | 3.00 | 3.16 | 3.15 | +0.30 vs two-stage SFT |
| EAIG only ($w_s=0, w_e=1$) | 2.68 | 2.22 | 2.63 | 2.51 | Reward collapse + length explosion |
| CSG only ($w_s=1, w_e=0$) | 3.16 | 2.93 | 3.06 | 3.05 | Stable but suboptimal |
| Balanced ($w_s=w_e=0.5$) | 3.32 | 2.96 | 2.98 | 3.09 | Near optimal |
| **Optimal ($w_s=0.7, w_e=0.3$)** | **3.34** | **3.04** | **3.07** | **3.15** | EAIG as garnish, CSG as primary |
| Top-50% entropy mask | 3.16 | 2.78 | 3.00 | 2.98 | Noisy tokens drag down performance |
| **Top-25% entropy mask** | **3.32** | **2.96** | 2.98 | **3.09** | +3.7%, purer tech token signal |
| Optimal w/o Length Anchor | 3.22 | 2.94 | 3.00 | 3.05 | -0.10 |
| **Optimal w/ Length Anchor** | **3.34** | **3.04** | **3.07** | **3.15** | +3.2% |

### Key Findings
- **EAIG alone causes explosion**: Training with only EAIG leads to CoT length explosion and gibberish content ("hacking entropy"), proving macro-level direction signals are essential.
- **CSG is the chassis, EAIG is the garnish**: $w_s : w_e = 7 : 3$ outperforms 5:5, indicating the "semantic direction" provided by CSG is the stable signal.
- **Strict entropy masking is required**: A top-50% mask includes common words, introducing noise and causing CoT decay; a top-25% mask ensures the model learns to "explain hard tokens" rather than "pad word count."
- **Length Anchoring is essential against GRPO's implicit short-bias**: Appendix F formalizes this—GRPO group normalization maximizes the implicit Sharpe ratio, favoring low variance (short CoT). Anchoring provides positive gradients to neutralize this.
- **Negligible OOD decay**: ICLR 3.19 → NeurIPS 3.15 ($\Delta = 0.04$ not significant), proving the model learns reasoning patterns rather than venue templates. CoT spontaneously exhibits target decomposition, self-criticism loops, and paradigm questioning.
- **Dominates agentic frameworks**: MoRI's 14B single model + internalized RL > AI-Scientist-V2 + GPT-4o multi-agent + tree search (3.18 vs 2.60, +22.3%), significant at Bonferroni-corrected $p < 0.001$.

## Highlights & Insights
- **"Motivation as RL condition" is a key conceptual decoupling**: Conditioning $(z, y) \sim \pi_\theta(\cdot | x, m)$ allows RL to focus on "probing to solution" reasoning rather than learning "questioning + solving" simultaneously—a general insight for RLVR in open tasks.
- **EAIG as a surrogate verifier via "relative likelihood gain on hard tokens" is brilliant**: Resolves the absence of deterministic verifiers in open generation. Anchoring the verifier to ground-truth rather than generated content structurally avoids reward hacking.
- **The CSG contrastive baseline $S_{base}$ is a critical detail**: Since $\cos(\hat{y}, y^*)$ rewards restating input, subtracting $S_{base}$ forces the model to produce semantic content "beyond the input" to gain a positive reward.
- **Theory-Experiment Loop**: The use of the Sharpe ratio analogy to explain GRPO's short-bias and the derivation of length anchoring corrections turns an empirical observation into an analyzable phenomenon.

## Limitations & Future Work
- **Limitations**: (1) Restricted to CS/ML; domain logic in Physics/Biology remains unverified; (2) Evaluation relies on LLM-judge + partial human assessment, lacking "experimental execution" ground truth; (3) Risk of misuse for mass-producing low-quality papers.
- **Additional Insights**: (1) Still trails Claude-3.5-Sonnet in Novelty—MoRI is "stable/practical" but lacks "creative leaps," as the ground-truth anchor pulls the distribution toward the known; (2) "Posterior reconstruction" introduces hindsight bias, as generated $z_i$ traces are "cleaner" than real research processes; (3) Fixed SFT reference distribution for EAIG may become "outdated" during training.
- **Improvements**: Update the EAIG reference distribution online using the previous checkpoint; introduce retrieval-augmented contexts to detach Novelty from fixed ground-truth; replace posterior reconstruction with motivation extraction from real introductions to reduce hindsight bias.

## Related Work & Insights
- **vs Agentic Frameworks (AI-Scientist-V2, etc.)**: These rely on scaffolding without optimizing the model. MoRI internalizes reasoning via RL in a 14B model, outperforming them (+22.3%) with single inference.
- **vs Verifier-free RL (NOVER, RLPR)**: These use intrinsic signals (e.g., reference probability) as rewards, but are noisy for multi-solution tasks. MoRI’s EAIG + CSG provide more granular signals.
- **vs Rubrics-as-Rewards (Dr. Tulu)**: MoRI is model-based (embeddings + logprobs), making gradient estimation much cheaper than LLM-judge rubrics.
- **vs Beyond 80/20 (Wang et al. 2025b)**: While Wang filters high-entropy tokens in reasoning, MoRI applies this to the **ground-truth target**, fundamentally cutting off reward hacking paths.

## Rating
- Novelty: ⭐⭐⭐⭐ "Motivation-grounded RL + EAIG + CSG" is the first end-to-end RL solution for open scientific ideation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual-venue testing, 6 baselines, 4 ablation types, training dynamics curves, and rigorous statistical checking.
- Writing Quality: ⭐⭐⭐⭐ Theoretical analysis of CoT shortening in Appendix F is excellent; the motivation section is slightly verbose.
- Value: ⭐⭐⭐⭐ Establish a reproducible paradigm for "RL + LLM as research assistant." The EAIG/CSG philosophy is transferable to any open-ended LLM RL task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)

</div>

<!-- RELATED:END -->
