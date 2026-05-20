---
title: >-
  [Paper Note] ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models
description: >-
  [ICML 2026][LLM Reasoning][abductive reasoning] ANCHOR employs "bottom-up abduction + hierarchical clustering" to construct a dense factor space…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "abductive reasoning"
  - "Bayesian inference"
  - "LLM uncertainty"
  - "causal Bayesian network"
  - "hierarchical factor space"
date: 2026-05-08
content_hash: c7f17a92601fa718
---

# ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.10328](https://arxiv.org/abs/2605.10328)  
**Code**: Not released  
**Area**: LLM reasoning / probabilistic inference / causal Bayesian networks  
**Keywords**: abductive reasoning, Bayesian inference, LLM uncertainty, causal Bayesian network, hierarchical factor space

## TL;DR
ANCHOR employs "bottom-up abduction + hierarchical clustering" to construct a dense factor space, retrieves a sparse set of relevant factors for downstream conditions via coarse-to-fine search, and aggregates posteriors using both Naïve Bayes and a latent-variable causal Bayesian network constructed on-the-fly by an LLM. This approach significantly reduces "unknown" predictions and improves probability calibration in high-risk LLM decision scenarios.

## Background & Motivation

**Background**: In high-risk decision-making such as emergency response and infrastructure planning, reliable conditional probability estimates $P(O_i|C)$ from LLMs are required. Mainstream approaches (e.g., BIRD) use a two-stage "abduction + Bayesian" process—LLMs first generate a discrete factor set $F=\{F_1,\dots,F_N\}$ and their values from scenario Scen, then Naïve Bayes marginalizes $P(O_i|C)=\sum_f P(O_i|f)\prod_j P(f_j|C)$.

**Limitations of Prior Work**: There is a dilemma—(a) Forward abduction tends to generate a sparse factor space, causing downstream condition $u$ to map to zero factors, resulting in "unknown" outputs; (b) Forcibly expanding the factor set introduces noise and spurious correlations (e.g., "cold weather" and "wearing thick clothes" are highly correlated), violating the conditional independence assumption of Naïve Bayes.

**Key Challenge**: There is a trade-off between coverage of the factor space (avoiding unknowns) and independence (avoiding spurious correlations); additionally, LLM-provided numerical confidences are often overconfident and uninterpretable, making them unsuitable as direct probabilities.

**Goal**: (1) Construct a factor space that is both dense and structured, balancing coverage and noise; (2) Design a reliable "condition→relevant factors" retrieval mechanism; (3) Explicitly model latent variable dependencies among factors during probabilistic inference to mitigate the distortion of the Naïve Bayes independence assumption.

**Key Insight**: Reverse the traditional "top-down abduction" to **bottom-up abduction**—first freely generate a large number of supporting/opposing sentences, then extract factors, and finally organize them into a two-level hierarchy using clustering and LLM-based topic naming; also, use the LLM to infer latent variable structures online to build a query-specific causal Bayesian network (CBN) tailored to the current condition $u$.

**Core Idea**: Implement an end-to-end four-stage pipeline: "abduction → factor extraction → retrieval → probability aggregation," assigning each stage to the LLM's strengths (generation/extraction/naming/causal discovery/flexible priors), while delegating probabilistic computation to the lightweight NB + CBN models, with final weighted fusion.

## Method

### Overall Architecture
Input: Scenario Scen + condition $u$ + two candidate hypotheses $O_1, O_2$.  
Pipeline:  
(1) **Factor Space Construction**: bottom-up iterative generation → MiniLM embedding → UMAP dimensionality reduction → HDBSCAN clustering → LLM topic naming → form two-level hierarchy $\tilde{F}$;  
(2) **Context-Aware Mapping**: perform cluster-level KNN coarse retrieval + factor-level KNN fine retrieval on $\tilde{F}$ → self-consistency voting filter + reflection prompt for fine screening → obtain sparse factor set $F^*(u)$;  
(3) **Probabilistic Inference**: use LLM to flexibly output factor-level posteriors $\phi_f=P(O_1|f)$ and latent variable parameters → construct Naïve Bayes (output root→factors) and Causal Bayesian Network (output root→latent variables→factors);  
(4) **Aggregation**: weight and fuse the posteriors from both networks to obtain calibrated probabilities. When $|F^*(u)|=0$ or $\max_i P(O_i|C)<\tau$, actively abstain.

### Key Designs

1. **Bottom-up Abduction + Hierarchical Clustering for Dense Factor Space**:

    - **Function**: Reverses "structure first, then fill factors" to "massively generate factors, then merge into structure," alleviating the sparsity of forward abduction.
    - **Mechanism**: Start from the empty set $F^{(0)}=\emptyset$ and iterate $T_{max}$ rounds; in each round, (a) use few-shot prompts to let the LLM generate $b$ multi-perspective supporting/opposing sentences, (b) extract factors and merge into $F$, remove semantic duplicates, and check for convergence. Theoretically, the error rate of the factor set collected in each round is upper-bounded by $\exp(-2m(q-0.5)^2)$ under multiple self-consistency votes; after obtaining $F$, use MiniLM embedding → UMAP → HDBSCAN clustering (no need to preset $K$) → LLM assigns topics to each cluster (e.g., "Economic Feasibility") and prunes redundancy. Each factor is labeled as supports $O_1$ / supports $O_2$ / neutral, forming a two-level hierarchy $\tilde{F}$.
    - **Design Motivation**: Single-round forward abduction is limited by prompts and can only generate a few factors; free generation + post-structuring decouples "completeness" from "organization"—the structured output can be reused across multiple queries, avoiding redundant inference.

2. **Coarse-to-Fine Layered Retrieval + Self-Consistency Reflection Filtering**:

    - **Function**: Maps downstream condition $u$ to a high-precision, low-recall-bias subset of $\tilde{F}$.
    - **Mechanism**: Construct each cluster prototype embedding $\tilde{C}_j=\alpha\cdot e_{theme}+(1-\alpha)\cdot \frac{1}{|F_j|}\sum_{f\in F_j} e_f$ (mixing topic semantics and member mean); use KNN to select top-$K_1$ at the cluster level → top-$K_2$ at the factor level within each cluster → union as high-recall candidates $F_{cand}(u)$. Then, two-stage fine screening: (i) call the LLM $R$ times to select factors directly supported by $u$ from the candidates, each factor's vote count $v_f(u)=\sum_r \mathbf{1}[f\in m^{(r)}(u)]$, retain those above threshold $\gamma$ as $F_{vote}(u)$; (ii) use a reflection prompt to explicitly remove still unrelated factors, yielding $F^*(u)$.
    - **Design Motivation**: Brute-force retrieval in a dense factor space is computationally prohibitive; coarse filtering by semantic clusters followed by fine selection ensures millisecond-level response; the two-stage LLM fine screening uses "voting" and "reflection" prompts to suppress hallucinations and recall noise, complementing each other.

3. **NB + Latent Variable CBN Dual-Network Flexible Parameters + Posterior Aggregation**:

    - **Function**: While retaining the simplicity of NB, explicitly models latent variable dependencies among factors, outputting better-calibrated probabilities.
    - **Mechanism**: (a) **NB Model**: Root node Outcome ($O_1 / O_2$) connects to each factor $f_j$; query the LLM for $\phi_f=P(O_1|f)$, approximate $P(f|O_1)\approx\phi_f, P(f|O_2)\approx 1-\phi_f$ using symmetric priors. (b) **CBN Model**: LLM acts as a causal discovery engine, given the factor list, outputs a set of latent variables $L=\{L_1,\dots,L_k\}$ and their associated factor groups; the graph structure is Outcome → $L_i$ → corresponding $f_j$. The LLM further outputs $P(L_i=1|O_k)$, $P(f_j|L_i,O_k)$, etc. (c) **Aggregation**: Both networks infer $P^{NB}(O_i|C)$ and $P^{CBN}(O_i|C)$, which are then weighted and fused for the final estimate.
    - **Design Motivation**: Pure NB assumes factor independence (ignoring strong intra-category correlations like "economic factors"); CBN's explicit latent variables serve as common parents for factors, absorbing intra-class correlations without increasing data requirements—LLM-provided flexible priors are naturally suitable for "unsupervised decision" scenarios. NB and CBN have different bias directions, and their fusion provides complementary denoising.

### Loss & Training
ANCHOR does not require training any neural networks; all parameters are flexibly obtained via the LLM. Main hyperparameters include: factor clustering $K_1, K_2$, cluster prototype weighting $\alpha$, self-consistency query count $R$, voting threshold $\gamma$, abstain threshold $\tau$, iteration limit $T_{max}$, target factor count $N_{target}$, and NB-CBN fusion weight. Experiments use GPT-4 series / Qwen, with all prompt templates in the appendix.

## Key Experimental Results

### Main Results
The authors claim ANCHOR achieves SOTA on the same preference-based pairwise evaluation benchmark as BIRD (multiple LLM-driven decision tasks). Representative metrics (summarized from abstract and main text; specific values in Appendix D):

| Method | "unknown" prediction rate↓ | Human preference alignment↑ | Inference time↓ | Token usage↓ |
|--------|---------------------------|----------------------------|----------------|--------------|
| Direct LLM estimation | Low | Low (overconfident) | Low | Low |
| BIRD (forward abduction + NB) | High (factor sparsity) | Medium | Medium | Medium |
| BIRD + expanded factor set | Medium | Medium-low (noisy) | High | High |
| **ANCHOR (full)** | **Significantly reduced** | **SOTA** | **Significantly reduced** | **Significantly reduced** |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---------------|------------|----------------|
| Bottom-up factor space + NB only | "unknown" rate greatly reduced vs BIRD, but probabilities biased | Dense factor coverage solves sparsity |
| Add layered retrieval (no voting/reflection) | High recall but low precision in factor set | Retrieval alone insufficient, needs fine screening |
| Add self-consistency voting | Precision recovers | Voting removes occasional noisy factors |
| Add reflection prompt | Further removes residual irrelevant factors | Two-stage fine screening is complementary |
| Pure NB inference | Biased on highly correlated factors | Independence assumption fails |
| Pure CBN inference | Structure unstable, prone to overparameterization | Single network sensitive to latent variable mismatch |
| **NB + CBN weighted** | Best calibration | Complementary denoising |

### Key Findings
- Simultaneously reducing unknowns and inference cost is ANCHOR's most important engineering contribution—once a structured factor space is constructed, it can be reused across multiple downstream queries, with single-query retrieval + inference requiring only $O(K_1 K_2)$ LLM calls, greatly reducing token usage compared to BIRD.
- Self-consistency voting count $R$ is sensitive to the recall-precision trade-off; introducing the reflection prompt is more effective than simply increasing $R$, indicating that LLM "structured criticism" provides more information than "repeated sampling."
- Latent variables are inferred online by the LLM rather than learned globally, so each query has a customized CBN structure, avoiding the "cross-scenario latent variable structure mismatch" problem.

## Highlights & Insights
- **Clear division of roles**: Generation, extraction, naming, causal discovery, and flexible parameterization—tasks at which LLMs excel—are handled by the LLM; probabilistic computation is handled by the NB+CBN graphical models. This "probabilistic engine + LLM knowledge base" division is a good paradigm for all scenarios where LLMs replace expert knowledge.
- **Reusable structure vs one-off inference**: The factor hierarchy $\tilde{F}$ only needs to be constructed once; downstream queries repeatedly retrieve from it—thus, "expensive LLM inference" is amortized into "cheap vector retrieval," which is highly practical from an engineering perspective.
- **Abstain as a first-class citizen**: Explicitly treats "unknown" as a normal output (not an error); in high-risk scenarios, "better not to answer than to answer incorrectly" is more responsible than forcing a number.
- **Query-level on-the-fly latent variable construction**: Traditional causal inference requires stable structure; this work allows CBNs to vary by query, effectively enabling "on-demand causal inference." This idea can be extended to dialogue systems and medical decision-making.

## Limitations & Future Work
- All parameters depend on LLM flexibility; if the LLM's conditional probabilities $\phi_f$ are systematically biased (overconfident / reflecting training corpus bias), the entire framework will be biased; there is a lack of independent validation for $\phi_f$ calibration.
- CBN structures generated online by the LLM lack formal rationality checks, risking "hallucinated latent variables"; the paper does not provide a fallback mechanism for erroneous causal graphs from the LLM.
- Convergence of bottom-up abduction is controlled by $T_{max}$ and target factor count $N_{target}$; although geometric convergence is proven, actual quality depends on LLM diversity; rare scenarios may still be sparse.
- The evaluation benchmark relies on preference-based pairwise comparisons, with no ground-truth probabilities, making it difficult to judge whether ANCHOR's numerical outputs are truly calibrated (only "alignment with human preference" as a proxy metric).
- NB+CBN fusion weights require manual specification, lacking an adaptive scheme.

## Related Work & Insights
- **vs BIRD (Feng et al. 2025)**: BIRD uses forward abduction + single NB, prone to sparsity and independence violations; ANCHOR reverses abduction direction, adds hierarchy and CBN, addressing both issues.
- **vs CoT / ToT / Belief Graph**: Chain/tree-of-thought and belief graphs are reactive decompositions, redone for each query; ANCHOR is proactive, pre-building a reusable factor space, yielding higher efficiency and stability.
- **vs Graph RAG / Hierarchical RAG**: Traditional structured RAG indexes existing documents; ANCHOR generates knowledge sources (factors) from scratch and organizes them, making it more suitable for decision scenarios lacking domain documents.
- **vs LLM internal uncertainty methods (verbalized confidence / sampling)**: Directly asking LLMs "how confident are you" is unreliable; ANCHOR externalizes uncertainty via explicit probabilistic graphs, enhancing interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of bottom-up abduction, on-the-fly CBN construction, and NB-CBN fusion is an organic integration; each component has precedents, but the combination is novel.
- Experimental Thoroughness: ⭐⭐⭐ Main and ablation tables are fairly complete in the appendix; however, lacks calibration comparison with ground-truth probabilities and large-scale cross-domain generalization tests.
- Writing Quality: ⭐⭐⭐⭐ The motivation → pain points → pipeline derivation is clear, with formulas and flowcharts well-coordinated.
- Value: ⭐⭐⭐⭐ For high-risk LLM decision-making, simultaneously achieving "reduced unknowns + calibration + cost reduction" is highly practical for engineering deployment.

## Related Papers

- [\[ICML 2026\] Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)
- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)
- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)
- [\[NeurIPS 2025\] Curriculum Abductive Learning](../../NeurIPS2025/llm_reasoning/curriculum_abductive_learning.md)

</div>

<!-- RELATED:END -->
