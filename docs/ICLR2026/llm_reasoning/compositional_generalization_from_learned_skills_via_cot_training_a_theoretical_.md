---
title: >-
  [Paper Note] Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning
description: >-
  [ICLR 2026][Reasoning][Compositional Generalization] This paper demonstrates through information-theoretic generalization bounds and interpretability analysis that the core mechanism of CoT training is **compositional generalization**: models learn to systematically combine simple learned skills to solve novel complex problems. This is internalized as a two-stage compositional reasoning circuit that extracts intermediate results at shallower layers…
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "Compositional Generalization"
  - "Chain-of-Thought Training"
  - "Information-theoretic Generalization Bound"
  - "Reasoning Circuit"
  - "OOD Generalization"
date: 2026-05-08
content_hash: f69c1a76a1e1e4e8
---

# Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning

**Conference**: ICLR 2026  
**arXiv**: [2502.04667](https://arxiv.org/abs/2502.04667)  
**Code**: [https://github.com/chen123CtrlS/T-CotMechanism](https://github.com/chen123CtrlS/T-CotMechanism)  
**Area**: AI Safety / LLM Reasoning  
**Keywords**: Compositional Generalization, Chain-of-Thought Training, Information-theoretic Generalization Bound, Reasoning Circuit, OOD Generalization

## TL;DR
This paper demonstrates through information-theoretic generalization bounds and interpretability analysis that the core mechanism of CoT training is **compositional generalization**: models learn to systematically combine simple learned skills to solve novel complex problems. This is internalized as a two-stage compositional reasoning circuit that extracts intermediate results at shallower layers, freeing deeper layers to focus on subsequent reasoning steps.

## Background & Motivation

**Background**: CoT training (e.g., long-CoT cold start in DeepSeek-R1, RFT in OpenAI o1) has became a core paradigm for enhancing LLM reasoning. However, the **mechanism** by which it strengthens generalization remains unclear.

**Limitations of Prior Work**:
   - Previous theoretical analyses focus on how CoT improves the expressivity or computational complexity class of Transformers but do not explain how capabilities emerge during training.
   - Models trained without CoT exhibit a "compositional gap" in ID (In-Distribution) generalization—they know all basic facts but cannot combine them.
   - Crucially, non-CoT models fail to achieve OOD (Out-of-Distribution) generalization (new compositional patterns) entirely.

**Key Challenge**: Why does CoT training allow models to extend from ID generalization to OOD generalization? Does the model only learn "what to think" (the correct answer) or "how to think" (the reasoning method)?

**Goal**
   - (Q1) Does CoT training improve both ID and OOD generalization? What is the theoretical principle?
   - (Q2) How is this generalization capability implemented internally within the model?

**Key Insight**: Decompose CoT as $P(Y|X) = \sum_C P(Y|X,C) \cdot P(C|X)$, where $C$ represents the reasoning chain. CoT training explicitly learns $P(C|X)$ and $P(Y|X,C)$, while non-CoT training only learns $P(Y|X)$.

**Core Idea**: CoT training teaches the model "how to think" by decomposing complex problems into combinations of simple learned skills ($P(C|X)$ + $P(Y|X,C)$), thereby pulling OOD problems closer to the ID distribution to achieve systematic generalization.

## Method

### Overall Architecture
The paper does not propose a new model but answers a mechanistic question: why can CoT training push a model from "memorizing seen combinations" to "handling unseen new combinations"? The authors develop two parallel lines of analysis. The **Theoretical Line** decomposes CoT into $P(Y|X)=\sum_C P(Y|X,C)\cdot P(C|X)$ and derives an information-theoretic generalization bound. This bound splits test error into ID and OOD components, explaining why CoT training suppresses the OOD term. This framework further analyzes the robustness of the mechanism when erroneous steps are present in the training chains. The **Structural Line** uses an 8-layer GPT-2 on a two-hop reasoning task $(e_1,r_1,r_2)\to e_3$, employing Logit Lens and causal tracing to examine how CoT training alters internal computational paths. These lines converge on a single conclusion: CoT training learns "how to think" (decomposing hard problems into learned skills) rather than "what to think" (rote memorization of answers).

> As this is a theoretical and interpretability analysis rather than a multi-stage processing pipeline, no architecture diagram is provided; the key designs below correspond to the two analytical lines mentioned above.

### Key Designs

**1. Information-theoretic Generalization Bound: Decomposing and Proving OOD Error Reduction**

To explain why CoT enables OOD generalization, a metric that distinguishes between ID and OOD error is required. The authors derive a generalization error upper bound proportional to:

$$\sqrt{\tfrac{1}{N}\big[(1-\alpha)D_{KL}(P_{test}^{ID}\,\|\,P_{train}) + \alpha\,D_{KL}(P_{test}^{OOD}\,\|\,P_{train})\big]}$$

The ID term is easy to suppress as the test and training sets share compositional patterns, making $D_{KL}\to 0$. The difficulty lies in the OOD term. In non-CoT training, the model only learns $P(Y|X)$ and effectively holds a uniform prior over the reasoning chain $C$. When encountering new combinations, this $D_{KL}$ is large, preventing OOD error reduction. With CoT training, this term decomposes further into $D_{KL}(P_{test}^{OOD}(C|X)\,\|\,P_{train}(C|X)) + \mathbb{E}[D_{KL}(P_{test}^{OOD}(Y|X,C)\,\|\,P_{train}(Y|X,C))]$. Since the training data already contains the simple skills forming these new combinations (i.e., how to chain $P(C|X)$ and calculate each step $P(Y|X,C)$), both terms can be minimized. Consequently, OOD problems are "pulled back" into the ID distribution spanned by learned skills (Theorem 1 & 2).

**2. Two-Stage Compositional Reasoning Circuit: Shallow Extraction of Intermediate Results**

Theory suggests CoT learns compositional ability; how does it manifest internally? Using Logit Lens and causal tracing on an 8-layer GPT-2, the authors find that CoT training solidifies a clear **two-stage circuit**: the first stage in shallow layers (Layer 0 to $l$) extracts the bridge entity $e_2$ from inputs $e_1,r_1,r_2$ and stores it in state $E[l,r_2]$; the second stage in deep layers (Layer $l$ to 8) uses $e_2$ to perform the second hop for the final answer $e_3$. Interestingly, CoT training shifts the boundary layer $l$ forward: $e_2$ is extracted at Layer 3 in the ID setting, whereas non-CoT training delays this to Layer 5. Earlier extraction leaves more layers for the second hop, effectively increasing the model's "effective depth."

**3. Noise Robustness Analysis: Compositional Generalization with Error-Prone Chains**

Real-world CoT data often contains errors; can this mechanism withstand them? The authors injected noise at a ratio $\xi$ into intermediate steps of the training data (scanning $\xi\in\{0.05,0.2,0.4,0.6,0.8\}$ for the second hop). Compositional generalization persists as long as noise remains moderate: at $\xi<0.2$, the model is nearly unaffected, preserving both ID and OOD generalization. As $\xi$ increases, generalization performance for both decreases monotonically, and the error bound rises accordingly (consistent with Theorem 3). Mechanistically, since noise is only in the second hop, the first-hop circuit is still learned cleanly, explaining why large-scale CoT data with flaws remains effective in practice.

### Loss & Training
The primary difference lies in whether to explicitly model the bridge entity $e_2$. Non-CoT training targets only the final answer: $\mathcal{L}=\mathbb{E}[\ell(e_3,\mathcal{M}(e_1,r_1,r_2))]$. CoT training predicts both the bridge entity and the final answer: $\mathcal{L}=\mathbb{E}[\ell(e_3,\mathcal{M}(e_1,r_1,r_2,\hat{e}_2))+\ell(e_2,\mathcal{M}(e_1,r_1,r_2))]$. The latter term forces the two-stage "solve $e_2$ then solve $e_3$" structure into the model. Training utilizes autoregressive next-token prediction.

## Key Experimental Results

### Main Results (Controlled environment, 2000 entities × 200 relations)

| Method | ID Accuracy | OOD Accuracy | Convergence Steps |
|------|----------|----------|---------|
| Non-CoT (grokking) | ~100% (delayed) | ~0% | >1M steps |
| **Ours (CoT)** | **~100%** | **~90%+** | **~4000 steps** |

### Ablation Study (λ = 2-hop/1-hop data ratio, CoT training)

| λ Value | OOD Generalization Speed | Final OOD Accuracy |
|------|-------------|--------------|
| 0.001 (minimal 2-hop) | Fastest | ~85% |
| 0.9 | Fast | ~90% |
| 7.2 | Medium | ~95% |
| 12.6 | Slow | ~95% |

### Key Findings
- **CoT training accelerates convergence by 250x**: ~4000 steps vs >1M steps (Non-CoT).
- **CoT enables OOD compositional generalization**: Non-CoT models remain at 0% OOD accuracy even after millions of steps; CoT models reach 90%+ in ~4000 steps.
- **Sparse 2-hop data accelerates OOD generalization**: Surprisingly, $\lambda=0.001$ yields faster OOD generalization—paralleling findings in OpenAI o1's RFT (reasoning emerges with minimal fine-tuning data).
- **Shallow extraction of intermediate results**: CoT training allows $e_2$ extraction at layer 3 (vs layer 5 for Non-CoT), freeing layers for subsequent reasoning.
- **Two-layer Transformers suffice**: The compositional circuit can fully emerge in a model with as few as 2 layers under CoT training.
- **Noise Robustness**: Generalization is largely unaffected at $\xi < 0.2$, explaining the tolerance for errors in real-world CoT datasets.

## Highlights & Insights
- **Formalization of "how to think vs what to think"**: The advantage of CoT is precisely defined as decomposing $P(Y|X)$ into $P(C|X) \cdot P(Y|X,C)$, where the model learns the reasoning process rather than the direct answer.
- **Shallow Extraction Discovery**: This finding provides a significant addition to Transformer reasoning theory—CoT training teaches the model to use depth more efficiently by distributing reasoning steps across layers, increasing "effective depth."
- **Counter-intuitive Data Scarcity Finding**: The observation that less 2-hop data can speed up generalization has strong practical implications for RFT/SFT—a small amount of high-quality CoT data may trigger reasoning capabilities faster than massive datasets.

## Limitations & Future Work
- Primarily validated on synthetic data (entity-relation); real-world NLP task verification is limited (found in appendix).
- Analysis is restricted to two-hop reasoning; compositional circuits for multi-hop (>3 steps) scenarios remain to be studied.
- The information-theoretic bound is an upper bound and may not be tight.
- Behavior under Reinforcement Learning fine-tuning (GRPO/PPO) was not analyzed.
- Identifying and defining "simple learned skills" in real-world tasks remains an open problem.

## Related Work & Insights
- **vs Wang et al. [102]**: Also studied compositional circuits in Transformers but found systematic circuits only in ID settings; this paper proves CoT training extends them to OOD.
- **vs Feng et al. [17]**: They proved CoT increases effective depth from an expressivity standpoint; this paper complements that by showing "how" it is implemented via generalization and internal structures.
- **vs COCONUT/CoT2**: These works explore continuous reasoning spaces; this paper's compositional generalization theory serves as a foundation for understanding such methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to formalize CoT generalization via dual perspectives of information theory and internal circuits.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments are systematic, but real-world task validation is less extensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are perfectly interwoven with clear argumentation.
- Value: ⭐⭐⭐⭐⭐ Milestone for the theoretical understanding of CoT training and practical guidance for RFT/SFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](analytica_soft_propositional_reasoning_for_robust_and_scalable_llm-driven_analys.md)
- [\[ICLR 2026\] TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)
- [\[ICLR 2026\] STAT: Skill-Targeted Adaptive Training](stat_skill-targeted_adaptive_training.md)
- [\[ICLR 2026\] RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems](rlad_training_llms_to_discover_abstractions_for_solving_reasoning_problems.md)
- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)

</div>

<!-- RELATED:END -->
