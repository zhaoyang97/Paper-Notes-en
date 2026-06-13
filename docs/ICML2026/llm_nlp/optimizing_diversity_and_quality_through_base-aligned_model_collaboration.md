---
title: >-
  [Paper Note] Optimizing Diversity and Quality through Base-Aligned Model Collaboration
description: >-
  [ICML 2026][LLM/NLP][diversity-quality trade-off] The authors propose BACO, an inference-time token-level routing framework: switching between an "unaligned base model" and an "aligned instruct model" token-by-token duri…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "diversity-quality trade-off"
  - "inference-time collaboration"
  - "token-level routing"
  - "alignment"
  - "open-ended generation"
date: 2026-05-08
content_hash: f874965a1faedbb6
---

# Optimizing Diversity and Quality through Base-Aligned Model Collaboration

**Conference**: ICML 2026  
**arXiv**: [2511.05650](https://arxiv.org/abs/2511.05650)  
**Code**: Yes (Project page + repository open-sourced)  
**Area**: LLM / NLP  
**Keywords**: diversity-quality trade-off, inference-time collaboration, token-level routing, alignment, open-ended generation

## TL;DR
The authors propose BACO, an inference-time token-level routing framework: switching between an "unaligned base model" and an "aligned instruct model" token-by-token during a single decoding pass. By using logit uncertainty and content word signals to decide which model to trust, BACO simultaneously achieves the diversity of the base model and the quality of the aligned model without retraining or multiple sampling. The best router achieves a 21.3% joint diversity-quality improvement over the strongest baseline.

## Background & Motivation

**Background**: Alignment (SFT + RLHF/DPO) has significantly enhanced LLMs in instruction following, safety, and reward scores, making it the default state for deployed models. However, when repeatedly sampling for the same prompt, aligned models tend to collapse into a few "template responses" (e.g., repeatedly suggesting "Maui, Hawaii" for summer travel destinations in the US).

**Limitations of Prior Work**: Existing efforts to mitigate diversity collapse follow two main paths. Training-side methods (e.g., diverse RLHF, diversity regularization) require retraining, which alters the alignment distribution and may sacrifice safety or helpfulness. Inference-side methods involve high-temperature/diverse beam search, in-context resampling, paraphrase prompting, or back-translation, most of which require multiple decodings or long-range planning and often trade quality for diversity.

**Key Challenge**: A structural trade-off exists in the single-model paradigm—the alignment process itself reduces the entropy of the next-token distribution (mode collapse), concentrating probability mass on a few high-quality tokens. Empirical comparisons in the paper show that while Llama-3-8B has 3.15× the diversity of Llama-3-8B-Instruct on a WildChat subset, its quality is 5.95× lower, showing no Pareto dominance for either side.

**Goal**: To develop a method that elevates the overall Pareto frontier on the diversity-quality plane within a single decoding pass without retraining, allowing users to adjust the operating point on demand.

**Key Insight**: The authors capitalize on the "superficial alignment" phenomenon—base and aligned models show highly consistent predictions for most tokens. Discrepancies are primarily concentrated in stylistic/functional tokens (punctuation, newlines, function words) and a few high-uncertainty "semantic forks." Since major differences only occur at specific positions, the model only needs to switch at these locations.

**Core Idea**: Treat the base model as the "source of diversity" and the aligned model as the "source of quality." A lightweight router dynamically selects one model at the token level during decoding, transforming the single-model trade-off into dual-model collaboration.

## Method

### Overall Architecture
BACO formalizes generation as:
$$P_{\text{BACO}}(y_t|c_t) = w_{\text{base}} \cdot P_{\text{base}}(y_t|c_t;\theta_{\text{base}}) + (1-w_{\text{base}}) \cdot P_{\text{aligned}}(y_t|c_t;\theta_{\text{aligned}})$$
where $c_t = [x, y_{<t}]$ and $w_{\text{base}} \in \{0,1\}$ is a hard choice provided by the router (not a mixture weight; each token belongs entirely to one model). The pipeline is concise: parallel forward passes for both models → router evaluates current step signals → select base or aligned → sample token → append to context. To prevent gibberish resulting from inconsistent tokenizers, the authors constrain switching to word boundaries. This process requires only one decoding pass and no fine-tuning or prompt engineering, allowing it to be applied to any existing "base + aligned" pair at no cost.

### Key Designs

1.  **Logit-based Routing (model-centric signals)**:
    - **Function**: Uses the base model's own prediction uncertainty to determine if the current position is a "semantic fork." If so, it routes to the base model for diversity; otherwise, it routes to the aligned model for quality.
    - **Mechanism**: Two representative variants—BACO-P routes to base when the base model's top-1 probability $\max_{y_t} P_{\text{base}}(y_t|\cdot) < \gamma$; BACO-H routes to base when the base model's prediction entropy $H_{\text{base}}(Y_t|\cdot) = -\sum_{y_t} P_{\text{base}}(y_t|\cdot)\log P_{\text{base}}(y_t|\cdot) > \gamma$. The threshold $\gamma$ acts similarly to temperature: increasing $\gamma$ favors the base model (more diversity), while decreasing it favors the aligned model (higher quality).
    - **Design Motivation**: High-uncertainty positions indicate multiple reasonable continuations; forcing alignment here wastes the "diversification budget." Conversely, at low-uncertainty positions, the base and aligned models likely agree, making switching unnecessary.

2.  **Content-based Routing (language-centric signals)**:
    - **Function**: Uses the linguistic/semantic role of a token rather than its probability to decide model assignment. Specifically, "stylistic tokens" are reserved for the aligned model, while "content words" are assigned to the base model.
    - **Mechanism**: BACO-PUNC forces the aligned model if the top-1 prediction is a punctuation/formatting token (e.g., `\n`, period) to maintain format consistency; BACO-FC uses the aligned model for function words (e.g., "and", "if", "the") to maintain discourse cohesion. These signals do not require logits and are applicable to black-box models.
    - **Design Motivation**: Drawing from linguistic observations, content words (nouns, verbs, descriptive terms) drive the perception of "diversity," whereas style/function words are where base and aligned models differ most but readers care least. Assigning the latter to the aligned model stabilizes style without sacrificing diversity.

3.  **Combined Routing + Controllable Thresholds (strongest version in practice)**:
    - **Function**: Chains logit and content signals by priority, allowing continuous movement along the Pareto frontier on the diversity-quality plane via a single threshold $\gamma$.
    - **Mechanism**: Combinations like BACO-P-PUNC, BACO-P-FC, and BACO-H-PUNC first use content rules (PUNC/FC) to lock in "must-be-aligned" tokens. The remaining tokens fall back to logit rules. Adjusting $\gamma$ sweeps a curve from low-diversity/high-quality to high-diversity/medium-quality, providing a "control knob" for applications.
    - **Design Motivation**: Individually, logit signals favor the aligned model when certain, while content signals assign style/function words to alignment. Their combination maintains coherence while allowing the base model to provide diversity at true semantic forks, approaching the Pareto frontier more closely than any single strategy.

### Loss & Training
No training is required. All routers use parameter-free heuristics. The only continuous hyperparameter is the threshold $\gamma$, which serves as a user-facing "diversity temperature" requiring neither calibration nor learning. The paper explicitly leaves learned routers for future work, noting that diversity is inherently multi-dimensional (lexical/semantic/discourse) and a single scalar loss might cause objective conflicts and training instability.

## Key Experimental Results

### Main Results
Evaluation sets: NoveltyBench (instruction following), WildChat (dialogue), Narrative-Discourse (creative writing). Model pairs: Llama-3-8B/Instruct, Olmo2-7B/Instruct. Metrics: 11 Diversity × 2 Quality = 22 diversity-quality subspaces, aggregated using **Coverage** (Area Under the Curve, measuring the trade-off region) and **Dominance** (percentage of the global Pareto frontier occupied).

| Method | Lexical Cov. | Lexical Dom. | Semantic Cov. | Semantic Dom. | Overall Cov. | Overall Dom. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Base | 0.098 | 12.7% | 0.098 | 16.0% | 0.098 | 14.3% |
| Aligned | 0.269 | 49.0% | 0.104 | 29.2% | 0.186 | 39.0% |
| Nudging (collaborative baseline) | 0.276 | 9.3% | 0.247 | 9.9% | 0.261 | 9.6% |
| Prompting (Best) | — | 2.7% | — | 2.2% | — | 2.4% |
| Ensemble (Best) | — | 1.1% | — | 1.9% | — | 1.5% |
| **BACO (Best)** | **0.445** | 24.9% | **0.360** | **40.5%** | **0.403** | 32.7% |

Coverage improved by 0.142 (+30% reachable area) relative to the strongest baseline, with an overall joint diversity-quality gain of 21.3%. Semantic Dominance increased to 40.5%, meaning nearly half of the Pareto optimal points are uniquely occupied by BACO.

### Ablation Study (Different routers on NoveltyBench)
| Router | Lexical Cov. | Lexical Dom. | Semantic Cov. | Semantic Dom. | Overall Cov. | Overall Dom. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| -RAND (Random switching) | 0.493 | 26.3% | 0.409 | 17.0% | 0.451 | 21.7% |
| -JUDGE (External model judge) | 0.302 | 2.6% | 0.254 | 0.6% | 0.278 | 1.6% |
| -P (Top-1 probability only) | 0.433 | 4.8% | 0.397 | 8.5% | 0.415 | 6.7% |
| -FC (Function word only) | 0.419 | 3.2% | 0.382 | 4.7% | 0.401 | 4.0% |
| **-P-PUNC (Best combination)** | **0.495** | **30.7%** | **0.452** | **31.3%** | **0.474** | **31.0%** |
| -H-PUNC | 0.466 | 16.4% | 0.427 | 18.6% | 0.446 | 17.5% |
| -P-FC | 0.435 | 16.0% | 0.406 | 19.2% | 0.421 | 17.6% |

### Key Findings
- Combined strategies (-P-PUNC, -H-PUNC, -P-FC) significantly outperform single strategies, proving that logit and content signals are complementary.
- -RAND performs surprisingly well on lexical metrics but reaches only 17% semantic Dominance, indicating that "blind switching" creates surface-level variety but lacks true semantic diversity, which requires router guidance.
- -JUDGE (using an external LLM as a token-level judge) performed the worst and slowest. The authors use this to demonstrate that the task does not require complex discriminators; heuristic signals are sufficient.
- On verifiable tasks (IFEval, GSM8K), BACO increases diversity while maintaining quality/accuracy, showing that gains are not artifacts of open-ended evaluation.
- Human evaluation aligns with automatic metrics: diversity improvements were perceived by human judges without significant quality degradation.

## Highlights & Insights
- Upgrading "model collaboration" from selecting one model per request to token-level hard switching within word boundaries is a clean and engineering-friendly approach. It can be applied out-of-the-box to any open-source base + instruct pair with zero training cost.
- The "superficial alignment" hypothesis is effectively utilized: since most tokens are identical between the two models, decisions are only needed at a few points of divergence. This allows the router to remain simple (a few rules) rather than a learned model.
- The evaluation methodology evolves from "single-point scores" to "Coverage + Dominance + 11×2 subspaces," treating "controllability" as a first-class citizen. This approach is more robust than previous single-metric evaluations like NoveltyBench.
- Content signals (PUNC/FC) are applicable to black-box models. This implies that even for API-only aligned models, BACO can be replicated by routing structure tokens to the API and others to an open-source base model.

## Limitations & Future Work
- Requires holding both base and aligned weights, doubling deployment memory. The paper does not provide quantized or KV-reuse versions, which may be challenging for memory-constrained scenarios.
- Strictly depends on the homology between the base and aligned models. If they come from different families (e.g., different tokenizers or pre-training data), word-boundary switching and the "superficial alignment" assumption may fail.
- Evaluation focuses on English open-ended generation, with limited coverage of code, long-chain reasoning, or multilingual tasks. Only sanity checks were performed on IFEval/GSM8K.
- The threshold $\gamma$ still requires manual tuning per user/task. An automatic mechanism for setting $\gamma$ based on prompt embeddings or task categories is a logical next step.
- Although learned routers were left for future work, designing learning signals for multi-dimensional diversity is an open problem that might require multi-objective RL.

## Related Work & Insights
- **vs Nudging (Fei et al., 2025)**: Both use "superficial alignment" for collaboration, but Nudging injects aligned tokens into base decoding to improve base quality. BACO does the opposite—injecting base into aligned to restore diversity—and uses a controllable router to sweep the entire Pareto frontier.
- **vs Training-side Diversity (diverse RLHF / DivPO)**: These require retraining the alignment phase, potentially harming safety. BACO is strictly inference-time, leaves weights untouched, and preserves safety attributes.
- **vs Decoding Diversification (Temperature, Diverse Beam Search, Contrastive Decoding)**: These operate within a single model's distribution. BACO leverages **two distinct distributions** as diversity sources, providing a structural solution to mode collapse.
- **vs Prompting (in-context resampling / paraphrase)**: These often require multiple decodings or longer contexts, increasing overhead. BACO completes in a single pass, making it more wall-clock friendly.

## Rating
- Novelty: ⭐⭐⭐⭐ Using base+aligned collaboration to "reverse-engineer superficial alignment" is a fresh perspective; the router is simple, but the shift toward diversity is a notable contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 trade-off subspaces + long text + multiple model pairs + human evaluation + verifiable task cross-checks; the coverage is top-tier for ICML.
- Writing Quality: ⭐⭐⭐⭐ Clear conceptual diagrams and effectively uses "diversity-quality plane + Coverage/Dominance" language. The method section is concise; the appendix is heavily utilized.
- Value: ⭐⭐⭐⭐ Relieves mode collapse without training. Immediately applicable for diversity-first scenarios like creative writing and dialogue. Provides a reusable multi-objective evaluation protocol.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards](../../ICLR2026/llm_nlp/optimas_optimizing_compound_ai_systems_with_globally_aligned_local_rewards.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICML 2026\] "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal](ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ICML 2026\] On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation](on_the_limits_of_llm_adaptability_impact_of_model-internalized_priors_on_annotat.md)

</div>

<!-- RELATED:END -->
