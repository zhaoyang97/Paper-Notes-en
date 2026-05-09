---
title: >-
  [Paper Note] Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation
description: >-
  [CVPR2026][LLM Alignment][mesh generation] This paper proposes Mesh-Pro, the first asynchronous online reinforcement learning framework for 3D quadrilateral mesh generation. Its core algorithm, ARPO (Advantage-guided Ranking Preference Optimization), combines the Plackett-Luce ranking model with advantage-function weighting to achieve simultaneous improvements in efficiency (3.75× faster than offline DPO) and generalization, attaining state-of-the-art generation quality for both artist-style and dense meshes.
tags:
  - CVPR2026
  - LLM Alignment
  - mesh generation
  - reinforcement-learning
  - preference optimization
  - artist-style mesh
  - quadrilateral mesh
  - online RL
date: 2026-05-08
content_hash: 6b41f0b1c9fdc7d0
---

# Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation

**Conference**: CVPR2026  
**arXiv**: [2603.00526](https://arxiv.org/abs/2603.00526)  
**Code**: To be confirmed  
**Area**: LLM Alignment  
**Keywords**: mesh generation, reinforcement-learning, preference optimization, artist-style mesh, quadrilateral mesh, online RL

## TL;DR
This paper proposes Mesh-Pro, the first asynchronous online reinforcement learning framework for 3D quadrilateral mesh generation. Its core algorithm, ARPO (Advantage-guided Ranking Preference Optimization), combines the Plackett-Luce ranking model with advantage-function weighting to achieve simultaneous improvements in efficiency (3.75× faster than offline DPO) and generalization, attaining state-of-the-art generation quality for both artist-style and dense meshes.

## Background & Motivation
3D mesh generation is one of the core tasks in computer graphics. In recent years, autoregressive transformer-based methods (e.g., MeshGPT, MeshAnything) have modeled mesh generation as a sequence generation problem and achieved significant progress. Nevertheless, producing meshes that qualify as "artist-style"—clean topology, well-organized edge flow, and high quadrilateral ratio—remains a challenge.

Reinforcement learning (RL) has been shown to effectively improve the output quality of generative models, yet applying RL to 3D mesh generation faces unique difficulties:

1. **Limitations of offline DPO**: Existing work (e.g., MeshAnything V2) uses offline DPO to align mesh generation quality. Offline DPO relies on pre-collected preference data pairs, but the output space of mesh generation is enormous (vertex coordinates + face topology), making it difficult for pre-collected preference data to cover sufficient diversity, leading to poor generalization.
2. **Low training efficiency**: Offline DPO requires first generating a large number of candidate meshes, annotating preferences manually or automatically, and then training the model—this "generate–annotate–train" loop is very time-consuming.
3. **Mesh-specific evaluation challenges**: Unlike text or images, mesh quality evaluation must account for geometric integrity (broken faces) and topological quality (quadrilateral ratio, edge-flow direction), making standard reward design difficult.
4. **Resource overhead**: 3D mesh models typically have large parameter counts (1B+), and the computational cost of sampling and policy updates in online RL is substantial.

The core motivation is: **Can an efficient online RL framework be designed that leverages real-time generated samples for policy optimization while avoiding the coverage deficiency of offline DPO?**

## Core Problem
How to design an efficient online preference optimization algorithm for 3D mesh generation models that improves generalization and training efficiency while ensuring convergence stability?

## Method

### Overall Architecture: Asynchronous Online RL

Mesh-Pro adopts an asynchronous architecture that decouples generation (rollout) from training (update):

- **Rollout Workers**: Multiple GPUs perform mesh sampling in parallel; each worker independently samples $N$ candidate meshes based on the current policy.
- **Reward Evaluator**: Computes reward scores for generated meshes (based on the Ray-based reward described below).
- **Trainer**: Asynchronously draws reward-annotated samples from the rollout buffer for policy updates.

The key efficiency gain lies in **pipelining** rollout and training—while the trainer processes the current batch, rollout workers are already generating the next round of samples. Compared to the serial "generate completely → annotate → train" pipeline of offline DPO, the asynchronous architecture delivers a **3.75×** training speedup.

### ARPO: Advantage-guided Ranking Preference Optimization

ARPO is the paper's core algorithmic contribution, integrating two key ideas:

#### 1. Plackett-Luce Ranking Model
Given $N$ candidate meshes $\{y_1, \ldots, y_N\}$ generated under the same input condition and their rewards $\{r_1, \ldots, r_N\}$, the candidates are sorted by reward in descending order to obtain a permutation $\sigma$. The Plackett-Luce model defines a probability distribution over rankings:

$$P(\sigma | \theta) = \prod_{i=1}^{N} \frac{\exp(\log \pi_\theta(y_{\sigma(i)}))}{\sum_{j=i}^{N} \exp(\log \pi_\theta(y_{\sigma(j)}))}$$

where $\pi_\theta(y)$ denotes the probability of the policy model generating $y$. The optimization objective is to maximize the ranking likelihood ordered by reward.

Compared to DPO, which can only handle pairwise preferences (one preferred + one rejected), the Plackett-Luce model exploits the ranking information of all $N$ candidates simultaneously, making more complete use of available information.

#### 2. Advantage Function Weighting
To further improve optimization efficiency, ARPO introduces advantage function weighting on the ranking gradients. The advantage value of the $i$-th candidate is defined as:

$$A_i = r_{\sigma(i)} - \bar{r}, \quad \bar{r} = \frac{1}{N}\sum_{j=1}^N r_j$$

The final ARPO loss is:

$$\mathcal{L}_{\text{ARPO}} = -\sum_{i=1}^{N} A_i \cdot \log P_i(\sigma | \theta) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where $P_i$ is the $i$-th term of the Plackett-Luce probability and $\beta$ controls the KL regularization strength. The intuition behind advantage weighting is to allocate larger gradient signals to samples that are clearly above or below the mean reward, while reducing gradient updates for samples with small margins, thereby reducing noise.

### Diagonal-aware Mixed Tri-Quad Tokenization
To address the overly strict constraints of pure quadrilateral mesh tokenization, a mixed tri/quad tokenization scheme is proposed:

- Quadrilateral faces are split into two triangular faces via a diagonal edge, with the diagonal edge shared.
- Diagonal-aware tokens are introduced to represent the presence and orientation of the diagonal, enabling the decoder to distinguish between "genuine triangular faces" and "halves of a quadrilateral face" during generation.
- This mixed representation balances the topological quality of quadrilaterals with the flexibility of triangles.

### Ray-based Reward
A geometric integrity reward based on ray casting is designed:

- Rays are cast through the generated mesh from multiple directions, and entry/exit intersection points along each ray are detected.
- If the entry/exit parity of a ray is inconsistent (e.g., enters but does not exit), the mesh is flagged as geometrically broken.
- The reward score is computed by combining the quadrilateral ratio, face count, vertex distribution, and other metrics.
- This reward can be computed automatically without manual annotation.

## Key Experimental Results

### Main Results

| Method | FID ↓ | Broken Ratio ↓ | Quad Ratio ↑ | Edge Quality ↑ | User Study ↑ |
|--------|-------|----------------|--------------|----------------|--------------|
| MeshGPT | 38.7 | 12.3% | 0% (pure tri) | 0.72 | 2.1/5 |
| MeshAnything | 31.2 | 8.1% | 68.2% | 0.78 | 3.2/5 |
| MeshAnything V2 (DPO) | 27.5 | 5.4% | 74.5% | 0.83 | 3.6/5 |
| **Mesh-Pro (ARPO)** | **23.1** | **2.1%** | **82.3%** | **0.89** | **4.3/5** |

### Efficiency Comparison

| Method | Training Mode | Training Time | GPUs | Relative Speedup |
|--------|--------------|---------------|------|-----------------|
| MeshAnything V2 (offline DPO) | Offline | ~3.75 days | 64 | 1× |
| **Mesh-Pro (async ARPO)** | Async online | **~1 day** | 64 | **3.75×** |

### Ablation Study
- **ARPO vs. DPO**: Under the same number of training steps, ARPO achieves a broken ratio 3.3% lower and a quad ratio 7.8% higher than DPO.
- **Effect of advantage weighting**: Removing advantage weighting increases the broken ratio by 1.5%, demonstrating the effectiveness of gradient signal filtering.
- **Ranking ($N=4$) vs. pairwise ($N=2$)**: Using 4-candidate ranking improves quad ratio by 2.1% over pairwise comparison.
- **Async vs. sync**: The asynchronous architecture is approximately 2× faster in wall-clock time compared to synchronous online RL.

## Highlights & Insights
- **First online RL framework for mesh generation**: A paradigm shift from offline DPO to asynchronous online RL, opening a new direction for RL-based optimization in 3D generation.
- **Elegant ARPO algorithm design**: The combination of the Plackett-Luce ranking model and advantage weighting balances information utilization efficiency with gradient stability.
- **Significant training efficiency improvement**: The 3.75× speedup does not rely on algorithmic tricks or approximations but stems purely from the architectural asynchronous pipeline design.
- **Extremely low geometric broken ratio**: A broken ratio of 2.1% is far below competing methods, demonstrating the effectiveness of the ray-based reward and ARPO in optimizing geometric quality.
- **Mixed tokenization**: The diagonal-aware design is a clever engineering contribution.

## Limitations & Future Work
1. **High model scale requirements**: The configuration of 1.1B parameters and 64 GPUs presents a high barrier to entry; applicability in small-scale settings remains to be explored.
2. **Scalability of reward design**: The ray-based reward primarily evaluates geometric integrity; modeling higher-level aesthetic properties (e.g., edge-loop quality, flow direction) still has room for improvement.
3. **Limited to closed meshes**: Ray parity detection assumes meshes are closed manifolds and is not fully applicable to open meshes (e.g., planes, cloth).
4. **Integration with 3D reconstruction pipelines**: The current evaluation covers only independently generated quality; effectiveness as a post-processing step in 3D reconstruction pipelines has not been validated.
5. **Stability of asynchronous training**: A version lag exists between the rollout policy and the trainer policy, which may introduce staleness issues during prolonged training.

## Rating
- Novelty: ⭐⭐⭐⭐ First online RL framework for mesh generation; ARPO algorithm is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers quantitative evaluation, user studies, efficiency comparisons, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with complete technical details.
- Value: ⭐⭐⭐⭐ Advances the intersection of 3D generation and RL alignment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](../../ICLR2026/llm_alignment/token-importance_guided_direct_preference_optimization.md)
- [\[ICLR 2026\] No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](../../ICLR2026/llm_alignment/no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)
- [\[ICLR 2026\] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](../../ICLR2026/llm_alignment/dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](../../ICLR2026/llm_alignment/swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)
- [\[AAAI 2026\] EPO: Diverse and Realistic Protein Ensemble Generation via Energy Preference Optimization](../../AAAI2026/llm_alignment/epo_diverse_and_realistic_protein_ensemble_generation_via_energy_preference_opti.md)

<!-- RELATED:END -->
