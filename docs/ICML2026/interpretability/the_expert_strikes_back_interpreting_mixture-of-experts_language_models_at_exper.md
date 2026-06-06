---
title: >-
  [Paper Note] The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level
description: >-
  [ICML 2026][Interpretability][Mixture-of-Experts] This paper systematically compares the polysemanticity of MoE expert neurons versus dense FFN neurons using $k$-sparse probing. It finds that MoE neurons naturally approa…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Mixture-of-Experts"
  - "Polysemanticity"
  - "Sparse Routing"
  - "Automated Interpretability"
  - "Expert Specialization"
date: 2026-05-08
content_hash: 50e1ff84685e2cf5
---

# The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level

**Conference**: ICML 2026  
**arXiv**: [2604.02178](https://arxiv.org/abs/2604.02178)  
**Code**: https://github.com/jerryy33/MoE_analysis  
**Area**: Mechanistic Interpretability / MoE Large Models  
**Keywords**: Mixture-of-Experts, Polysemanticity, Sparse Routing, Automated Interpretability, Expert Specialization

## TL;DR
This paper systematically compares the polysemanticity of MoE expert neurons versus dense FFN neurons using $k$-sparse probing. It finds that MoE neurons naturally approach monosemanticity under the pressure of sparse routing. Consequently, the analysis unit is elevated from "neurons" to "entire experts." The authors use LLMs to automatically assign natural language labels to hundreds of experts, validated through causal trigger experiments. The final conclusion is that experts are neither broad domain specialists nor token-level processors, but rather "fine-grained task experts."

## Background & Motivation
**Background**: MoE has become the de facto standard for scaling LLMs (e.g., Gemini 2.5, DeepSeek-V3, Qwen3, ERNIE-4.5), where each token activates only a small fraction of the total parameters. Simultaneously, interpretability research for dense models primarily relies on post-hoc sparse coding like Sparse Autoencoders (SAEs), which require separate training for every layer and incur massive computational costs.

**Limitations of Prior Work**: Neurons in dense FFNs are highly **polysemantic**: a single neuron responds to multiple unrelated concepts due to superposition—networks represent far more than $d$ features using approximately orthogonal directions in a $d$-dimensional space. This makes it nearly impossible to understand a neuron's function by simple inspection.

**Key Challenge**: MoE introduces sparsity at the architectural level, yet the academic community still interprets MoEs as if they were dense models (by examining individual neurons or training SAEs), failing to exploit structural dividends. Furthermore, there is a conflict between two schools of thought regarding expert specialization: one claiming experts divide labor by broad domains (e.g., biology, code) and the other claiming they follow syntax/token patterns.

**Goal**: (1) Quantify whether MoE expert neurons are truly more monosemantic than dense FFN neurons; (2) If so, determine if the analysis unit can be elevated to the expert level to scale LLM interpretation without SAEs; (3) Use these tools to arbitrate the debate on expert specialization.

**Key Insight**: Chaudhari et al. (2025) observed in toy models that "sparser routing leads to weaker superposition." The authors migrate this observation to production-scale LLMs. They also introduce the "residual stream contribution norm $g_i(x)\|E_i(x)\|_2$" as a key signal to measure true expert activity.

**Core Idea**: Architectural sparse routing pushes both **individual neurons** and **entire experts** toward monosemanticity. Thus, MoE models can be directly interpreted via natural language at the "expert level" without the need for expensive neuron-level decomposition.

## Method

### Overall Architecture
The paper does not train new models but performs a three-stage analysis on 12 public MoE/dense models (OLMoE-1B-7B, Mixtral-8x7B, Qwen3-30B-A3B, ERNIE-4.5-21B-A3B, OLMo-7B, etc.):

1.  **Neuron-level Probing**: Conducts $k$-sparse probing on 58 concepts (POS, LaTeX, code, natural language) to compare the separability of MoE expert intermediate activations $\mathbf{h}=\mathrm{Swish}(W_{\text{gate}}x)\odot W_{\text{up}}x$ against dense FFNs at the same position, matched by active-parameter counts.
2.  **Expert-level Automated Labeling**: Uses the residual stream contribution norm to select the top-20 activating sequences from The Pile along with the top-3 tokens promoted by the Logit Lens for each expert. An explainer LLM (Gemini 3 Flash Preview) generates a one-sentence description, and a scorer LLM calculates the $F_1$ score on 10 positive and 10 negative samples.
3.  **Quantitative Specialization Measurement**: Defines "model-native domains" using $k$-means clustering on the unembedding matrix ($k\in\{10,50,100,1000,5000\}$). Jensen-Shannon Divergence (JSD) measures each expert's deviation from the layer average, reporting Routing Specialization (distribution of incoming tokens) and Functional Specialization (distribution of tokens promoted via Logit Lens).

### Key Designs

1.  **$k$-sparse probing + best-layer protocol**:
    - **Function**: Trains L2-regularized logistic regression on MoE expert activation vectors $\mathbf{h}$ and dense FFN activation vectors, using only the top-$k$ dimensions as features to see if concepts can be binary classified.
    - **Mechanism**: For each concept, top-$k$ neurons are selected by $a_j=|\mathbb{E}[h_j\mid y=1]-\mathbb{E}[h_j\mid y=0]|$, where $k\in\{1,2,4,8,16,32,64\}$. In MoEs, only tokens routed to the target expert are retained. The best layer/expert is identified for each concept to avoid layer-selection bias.
    - **Design Motivation**: A high $F_1$ at $k=1$ is direct evidence that a concept is "pinned" to a single neuron. This converts "monosemanticity" from a qualitative into a comparable metric while controlling for active-parameter count.

2.  **Automated Expert Labeling Pipeline based on Residual Stream Contribution**:
    - **Function**: Uses an explainer LLM to write natural language labels for each MoE expert and a scorer LLM to verify the label's discriminative power ($F_1$) on held-out samples.
    - **Mechanism**: Activation selection ignores raw router weights $g_i(x)$ or absolute neuron values alone. Instead, it measures how large a vector the expert writes to the residual stream: $\mathrm{score}(s,E_i)=\max_{x\in s}\;g_i(x)\,\|E_i(x)\|_2$. This score, combined with Logit Lens tokens, is fed to the explainer.
    - **Design Motivation**: The only channel for a transformer component to affect the output is via the update vector on the residual stream; thus, $\|E_i(x)\|_2$ represents "causally significant" activity.

3.  **Trigger-Target Causal Attribution + JSD Specialization Metrics**:
    - **Function**: (a) Verifies if the expert labels correspond to their causal effects; (b) Quantifies the granularity of expert specialization.
    - **Mechanism**: (a) Given a label, the LLM synthesizes "trigger" words (to activate the expert) and "target" words (which the expert should promote). Direct Logit Attribution (DLA) ranks all experts in the layer. (b) JSD is calculated for Routing/Functional distributions across model-native domains defined by unembedding clusters, using a Random Expert Baseline to correct for noise.
    - **Design Motivation**: Trigger-target experiments turn "the label is descriptive" into a verifiable causal hypothesis. Sweeping $k$ in JSD arbitrates the "broad domain vs. task" debate—broad domain experts would show peak JSD at $k=10$, but results favor $k=5000$.

### Loss & Training
The paper **does not train any LLMs**. The only trained components are the logistic regression probes ($L_2$ regularization, 75/25 train-test split, fitted per concept/layer/expert). All interpretation processes depend on forward passes of existing checkpoints and external explainer/scorer LLMs.

## Key Experimental Results

### Main Results: Monosemanticity of MoE vs. Dense
Comparison of best-layer $F_1$ in $k$-sparse probes:

| Setting | MoE performance at $k=1$ | Dense performance at $k=1$ | Key Observation |
|---------|--------------------------|----------------------------|-----------------|
| Matched active-param (12 models) | Near optimal, low variance | Significantly lower than MoE | Gap is largest at $k=1$, meaning MoE concepts are often pinned to single neurons |
| OLMo family comparison | OLMoE-1B-7B near upper bound | OLMo-7B (7× active) is polysemantic | Sparse routing explains monosemanticity better than raw capacity |
| $N_A/N$ slices | Qwen3-30B-A3B ($N_A/N\approx0.06$) cleanest | Mixtral-8x7B ($N_A/N=0.25$) noticeably dirtier | Sparser routing leads to stronger monosemanticity |

### Key Findings
- **Sparsity is the key, not parameter count**: OLMoE-1B-7B (1B active) outperforms OLMo-7B (7B active) in monosemanticity, showing the interpretability dividend comes from architectural sparse routing. This implies the industry trend toward "more total experts + fewer active experts" naturally makes models more transparent.
- **Experts are fine-grained task specialists**: JSD is much higher at $k=5000$ than $k=10$. Qualitative taxonomy supports this: OLMoE-L1-E57 handles chemical/biological suffixes; ERNIE-L15-E0 handles completion after coordinating conjunctions; OLMoE-L15-E17 specifically closes LaTeX brackets `}}`.
- **Layer-wise functional division**: Early layers handle morphology/tokenization; middle layers handle syntactic cohesion and domain knowledge; deep layers handle structural validity and format constraints.

## Highlights & Insights
- **Residual stream contribution norm is an ingenious activity metric**: Previous methods for selecting activating samples looked at router weights or neuron magnitudes, neither of which directly correspond to causal influence. $g_i(x)\|E_i(x)\|_2$ measures the "only channel" for influence in a transformer.
- **The $k$-sweep of JSD + unembedding $k$-means** is the cleanest criterion for the specialization debate: Instead of human-defined domain labels, it uses the model's own output space structure to define domains and lets data arbitrate the granularity.
- **"Modular monosemanticity"**: The combination of single neurons becoming more monosemantic due to sparse routing and the router feeding homogeneous tokens to the same expert ensures the "entire expert" is readable.
- **Transferable trend**: If the trend of "sparsity $\uparrow$ $\to$ monosemanticity $\uparrow$" holds, next-generation MoE models may be naturally readable, providing huge benefits for alignment and safety auditing.

## Limitations & Future Work
- **Omission of largest-scale MoEs**: Due to GPU memory constraints, models like DeepSeek-V3 or Llama-4-MoE were not included; the authors only extrapolate trends.
- **Inter-expert superposition**: While intra-expert polysemanticity decreases, superposition across multiple experts may still exist, threatening the idea of an expert as a "sufficiently interpretable unit."
- **Labeling risks**: Natural language labels can hide prompt dependencies, dataset artifacts, or expert interactions. This same precision could be used by malicious actors to bypass safety mechanisms.
- **Future Directions**: Treating experts as sub-circuit nodes for circuit tracing on MoEs; using expert-level interventions for precise editing without touching all parameters.

## Related Work & Insights
- **vs. Sparse Autoencoder (Bricken et al., 2023)**: SAEs train sparse dictionaries post-hoc for every layer at high cost. This paper proves MoE models can use "experts" as natural sparse code units, **saving SAE training costs**.
- **vs. Chaudhari et al. (2025) toy models**: While Chaudhari proposed that sparse routing weakens superposition in toy models, this paper **scales it to production LLMs** with quantified monosemanticity and causal attribution.
- **vs. Specialization Schools**: Instead of picking sides between broad domains (Muennighoff et al., 2025) and token-level patterns (Xue et al., 2024), this work provides an objective JSD vs. $k$-sweep metric to show that experts are task-level specialists.
- **vs. Geva et al. (2021) "FFN as key-value memory"**: The paper treats the "value" as a human-readable natural language label, adding semantic annotations to the key-value memory perspective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Scaling sparse-superposition theories and proving the feasibility of expert-level automated interpretability.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive cross-family comparisons and three-tier validation (probing, labeling, causal triggers).
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and natural progression; good coordination between math and figures.
- **Value**: ⭐⭐⭐⭐⭐ Provides a cost-effective alternative to SAEs for MoE interpretability and arbitrates a long-standing debate on expert specialization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](../../CVPR2026/interpretability/ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)
- [\[NeurIPS 2025\] AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation](../../NeurIPS2025/interpretability/agentiql_an_agent-inspired_multi-expert_framework_for_text-to-sql_generation.md)
- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](../../ACL2026/interpretability/meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)

</div>

<!-- RELATED:END -->
