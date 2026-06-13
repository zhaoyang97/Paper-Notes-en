---
title: >-
  [Paper Note] Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization
description: >-
  [ACL2026][LLM Alignment][Topological Data Analysis] This paper frames LLM alignment as a "semantic trajectory" shaping problem in the hidden space. It employs 0-dimensional persistent homology during the SFT stage to ext…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Topological Data Analysis"
  - "Persistent Homology"
  - "SFT"
  - "DPO"
  - "Representation Trajectory"
date: 2026-05-08
content_hash: e10b1b1256d63e94
---

# Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization

**Conference**: ACL2026  
**arXiv**: [2605.07172](https://arxiv.org/abs/2605.07172)  
**Code**: To be confirmed  
**Area**: LLM Alignment  
**Keywords**: Topological Data Analysis, Persistent Homology, SFT, DPO, Representation Trajectory

## TL;DR
This paper frames LLM alignment as a "semantic trajectory" shaping problem in the hidden space. It employs 0-dimensional persistent homology during the SFT stage to extract prompt-answer topological bridges and integrates Trajectory Topology Loss (TTL). In the DPO stage, it incorporates Topological Preference Optimization (TPO) using topic-specific preference directions. The method consistently outperforms non-topological baselines in reward, win rate, and harmlessness metrics on UltraChat and HH-RLHF.

## Background & Motivation
**Background**: Current LLM alignment typically involves Supervised Fine-Tuning (SFT) followed by RLHF or DPO. SFT focuses on optimizing token-level likelihood, while DPO/RLHF optimizes preference scores or pairwise rankings. While effective, these training signals mostly operate at the local sample or scalar level.

**Limitations of Prior Work**: Existing objective functions rarely impose direct constraints on how internal representations transition from user prompts to answers, nor do they address the "improvement direction" from rejected to chosen responses in the hidden space. Models may mimic preference data at the output level without learning stable, transferable paths within the hidden space.

**Key Challenge**: Alignment essentially aims to guide the model towards "more helpful, safer, and instruction-compliant" directions. However, standard objectives only provide local preference information for individual tokens or answer pairs, failing to explicitly leverage the overall geometric structure of a batch of samples in the representation space. A gap exists between local likelihood and the global semantic manifold.

**Goal**: The authors aim to investigate whether the hidden states of prompts, answers, chosen, and rejected responses can be treated as point clouds, utilizing Topological Data Analysis (TDA) to extract stable cross-cluster connections to regularize the model's semantic trajectories.

**Key Insight**: 0-dimensional persistent homology tracks the merging of connected components in a point cloud as the distance threshold increases. "Death edges" across labels act similarly to critical bridges in a Minimum Spanning Forest. The authors argue that these topological bridges reflect the global contact between the prompt manifold and the answer manifold more effectively than random, gold-standard, or kNN pairings.

**Core Idea**: Use topological bridges to replace arbitrary local pairings, ensuring that hidden states move along topologically consistent paths in addition to generating correct outputs.

## Method

### Overall Architecture
The method consists of two training stages. The first stage is SFT + Trajectory Topology Loss (TTL): For each batch, the model computes the mean-pooled last-layer hidden states of prompt tokens, teacher-forced answer tokens, and the mean input embeddings of gold answer tokens. A point cloud of $2B$ points is formed from prompt and gold answer representations. 0-dimensional persistent homology is applied to extract death edges connecting prompt and answer points, forming prompt-answer bridges. The model's actual prompt-to-answer trajectories are aligned with these bridge directions.

The second stage is DPO + Topological Preference Optimization (TPO): HH-RLHF prompts are clustered into topics offline to construct positive and negative templates. Topic-specific preference vectors are generated using a sentence transformer. During DPO training, the difference between the mean-pooled hidden states of chosen and rejected responses is calculated at an intermediate layer. A small projection matrix maps the topic vectors into the model's hidden space, and a cosine loss aligns the "rejected-to-chosen" semantic improvement direction with these mapped vectors. The final loss is a combination of the DPO loss and a dynamically weighted TPO loss.

### Key Designs
1.  **Trajectory Topology Loss (TTL) in SFT**:
    - **Function**: Organizes prompt and gold answer representations into a point cloud to extract cross-label topological bridges as additional trajectory supervision.
    - **Mechanism**: Euclidean distances between points are calculated, and connected components are merged using Union-Find in ascending order of distance. "Death edges" connecting different labels (prompt and gold answer) are identified as bridges. The TTL is the cosine loss between the model's trajectories ($h_i^{model}-h_i^{prompt}$) and these bridge directions: $L_{topo}=mean(1-cos(v_{topo}, v_{model}))$.
    - **Design Motivation**: Gold-standard pairing only considers individual samples, kNN focuses on local neighbors, and random pairing introduces noise. Persistent homology bridges originate from global connectivity, filtering out local accidental connections for more stable trajectory regularization.

2.  **Topological Preference Optimization (TPO) in DPO**:
    - **Function**: Explicitly constrains the hidden-space improvement direction during preference optimization, ensuring the chosen response is not just more probable but also moves in a topic-relevant "better answer" direction.
    - **Mechanism**: Prompts are embedded via sentence transformers and clustered using MiniBatch KMeans, followed by topic labeling via a strong model. Preference vectors are constructed using templates like "helpful/harmless" vs. "harmful/unhelpful." During training, the normalized hidden difference $\Delta h=LN(h^{ch})-LN(h^{rj})$ is aligned with the topic vector $u_t$ mapped via projection $P$ by optimizing $1-cos(\Delta h, Pu_t)$.
    - **Design Motivation**: Preference directions vary across topics (e.g., safety advice vs. knowledge Q&A). Topic-aware vectors provide finer granularity than a single global preference vector.

3.  **Dynamic Weighting and Fully Topological Variant**:
    - **Function**: Prevents the TPO auxiliary term from dominating the DPO main objective and tests the benefit of topological structures during preference optimization.
    - **Mechanism**: An EMA tracks the magnitudes of DPO and TPO losses to set $\lambda_{dyn}$ dynamically. A Topo-TPO variant treats chosen/rejected hidden states as a point cloud, using 0D persistent homology to extract rejected-to-chosen bridges for alignment with preference vectors.
    - **Design Motivation**: Fixed weights are sensitive to training phases; the fully topological variant explores whether gains come from topic vectors or the batch structure itself.

### Loss & Training
The total objective for SFT is $L_{SFT}=L_{CE}+\lambda_{topo}L_{topo}$, with an optimal $\lambda_{topo} \approx 0.2$. The total objective for DPO is $L_{total}=L_{DPO}+\lambda_{dyn}L_{TPO}$. Experiments use Qwen2.5-7B-Instruct with LoRA (rank 16). Persistent homology is implemented on CPU using Union-Find and pairwise distances. Cross-backbone validation was performed on Llama-3-8B-Instruct.

## Key Experimental Results

### Main Results

| Stage / Dataset | Method | Metric | Baseline | Ours | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SFT / UltraChat | Base SFT vs. SFT + TTL | RM / IFEval / Toxicity | 64.2 / 68.5 / 0.45 | 67.8 / 71.8 / 0.38 | RM +3.6, IFEval +3.3 |
| SFT zero-shot / HH-RLHF | Base SFT vs. SFT + TTL | RM / Help / Toxicity | 62.1 / 45.2 / 0.48 | 65.4 / 49.8 / 0.41 | Helpfulness +4.6 |
| DPO / HH-RLHF | DPO vs. DPO + TPO | RewardBench / Win Rate / MT-Bench / Harmless | 84.5 / 52.1% / 8.65 / 90.2% | 87.2 / 55.4% / 8.81 / 93.5% | Win & Harmlessness $\uparrow$ |
| DPO / HH-RLHF | DPO vs. DPO + Topo-TPO | RewardBench / Win Rate / MT-Bench / Harmless | 84.5 / 52.1% / 8.65 / 90.2% | 87.4 / 55.6% / 8.80 / 94.1% | Max Harmlessness gain |

### Ablation Study

| Configuration | RM | Win | IFEval | Toxicity | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| No TTL | 64.2 | - | 68.5 | 0.45 | Pure CE SFT |
| Random Pair | 64.6 | 50.8% | 68.9 | 0.44 | Minimal gain |
| All Pairs (no PH) | 66.1 | 53.2% | 69.8 | 0.41 | Effective but unstable |
| kNN Bridge | 66.8 | 55.6% | 70.5 | 0.40 | Better than per-sample |
| PH Bridge (Ours) | 67.8 | 58.4% | 71.8 | 0.38 | Best performance |

| TPO Configuration | RewardBench | AlpacaEval | Harmless | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| DPO | 84.5 | 52.1% | 90.2% | Standard DPO |
| + Global Cosine | 85.1 | 52.8% | 90.5% | Small manual gain |
| + Learned Global Vec. | 85.8 | 53.5% | 91.2% | Coarse global direction |
| + TPO (no dyn) | 86.3 | 54.2% | 91.8% | Topic vectors are better |
| + TPO (Ours) | 87.2 | 55.4% | 93.5% | Dynamic weight stability |

### Key Findings
- The benefits of TTL stem from the global connectivity structure of persistent homology bridges rather than the cosine loss itself; PH Bridges outperform Random, All Pairs, and kNN.
- TPO's effectiveness is rooted in being topic-aware. Segmenting preference directions by prompt topic is superior to using a single global vector.
- Topo-TPO shows slightly higher harmlessness than standard TPO, suggesting preference optimization also benefits from the global structure of chosen/rejected point clouds.
- Topological weights cannot be increased indefinitely; at $\lambda_{topo}=0.4$, RM and toxicity performance degrades, indicating trajectory regularization should support rather than replace language modeling.

## Highlights & Insights
- The most compelling aspect is shifting alignment from "output preference" to "hidden state trajectory." This provides a unified view for SFT and DPO.
- The use of 0D persistent homology is efficient, requiring only Union-Find and pairwise distances without expensive high-dimensional topological features. It acts as a global structure-aware matcher.
- Topic-aware preference vectors in TPO could be transferred to other tasks like factuality, politeness, or code execution by defining attribute-specific directions in the hidden space.
- The ablation studies clearly distinguish the contributions of topological vs. non-topological, topic-aware vs. global, and fixed vs. dynamic weighting.

## Limitations & Future Work
- The study centers on Qwen2.5-7B, Llama-3-8B, UltraChat, and HH-RLHF. Its stability in larger models, long-chain reasoning, or multi-turn agent scenarios remains untested.
- 0D persistent homology relies on in-batch point clouds. Batch composition and embedding layer choice can affect the bridges. Future work could explore cross-batch memory banks.
- TPO depends on offline clustering quality and template construction. Learning directions from human feedback residuals or automatically discovered concepts may be more robust.
- Evaluation relies on standard benchmarks and RM scores; more human validation of trajectory interpretability and stress testing in red-teaming scenarios is needed.

## Related Work & Insights
- **vs. Standard SFT / RLHF / DPO**: While standard methods optimize token likelihoods or rankings, this work constrains hidden-space trajectories, leveraging representation geometry at the cost of hypers like layer selection and weights.
- **vs. Representation Geometry Analysis**: Unlike prior works that use geometry solely for analysis, this paper converts geometric structures into training signals for alignment.
- **vs. TDA Regularization**: Traditional TDA regularization is often used for classification boundaries; this work innovatively applies 0D persistent homology to the SFT and DPO phases of LLM alignment.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Introducing persistent homology for hidden-trajectory regularization in LLM alignment is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Comprehensive main experiments and ablations, though model scale and safety scenarios could be further expanded.
- Writing Quality: ⭐⭐⭐⭐☆ Clear logic and well-supported tables, though some implementation details are relegated to the appendix.
- Value: ⭐⭐⭐⭐☆ Provides a "trajectory shaping" perspective for alignment, valuable for future hidden-space control and interpretable alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](../../ICLR2026/llm_alignment/safedpo_preference_optimization_safety.md)
- [\[ACL 2026\] Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards](teaching_llm_to_be_persuasive_reward-enhanced_policy_optimization_for_alignment_.md)
- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](../../ICLR2026/llm_alignment/towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)
- [\[ACL 2026\] Large Language Models Are Overconfident in Their Own Responses](large_language_models_are_overconfident_in_their_own_responses.md)

</div>

<!-- RELATED:END -->
