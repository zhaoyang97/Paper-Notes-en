---
title: >-
  [Paper Note] DEI: Diversity in Evolutionary Inference for Quality-Diversity Search
description: >-
  [ICML 2026][LLM Evaluation][Quality-Diversity] This paper proposes DEI, which treats multiple **LLMs from different families** as heterogeneous mutation operators distributed across different nodes. It utilizes fully asy…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Quality-Diversity"
  - "MAP-Elites"
  - "Heterogeneous LLM Ensemble"
  - "Asynchronous Gossip"
  - "Program Synthesis"
date: 2026-05-08
content_hash: 1cf00ec7949391dc
---

# DEI: Diversity in Evolutionary Inference for Quality-Diversity Search

**Conference**: ICML 2026  
**arXiv**: [2605.27130](https://arxiv.org/abs/2605.27130)  
**Code**: Extended based on the SakanaAI/drq open-source implementation, promised to be released with the paper  
**Area**: Optimization / Evolutionary Algorithms / Quality-Diversity / LLM as Operator  
**Keywords**: Quality-Diversity, MAP-Elites, Heterogeneous LLM Ensemble, Asynchronous Gossip, Program Synthesis  

## TL;DR
This paper proposes DEI, which treats multiple **LLMs from different families** as heterogeneous mutation operators distributed across different nodes. It utilizes fully asynchronous gossip to broadcast champions from each round, creating cross-model adversarial pressure. In the Core War program synthesis task, using the same total compute, DEI achieves a +124% gain in QD-Score and +28% in archive coverage compared to a single node.

## Background & Motivation

- **Background**: Replacing manual genetic operators with LLMs has become mainstream. Methods like FunSearch, Evolution through Large Models, and OPRO allow LLMs to perform "mutations" by reading old solutions plus prompts to output superior variants. The MAP-Elites/QD framework maintains an archive grid based on "Behavioral Characteristics (BC)," aiming to find a set of solutions that are both high-quality and diverse. Digital Red Queen (DRQ, kumar2026) further uses the champion of each round as an opponent for the next, creating Red Queen co-evolutionary pressure.
- **Limitations of Prior Work**: Current "distributed LLM search" methods merely replicate the same model across $N$ workers, relying on sampling temperature for diversity. This is equivalent to parallelizing the same generation distribution $N$ titles. Any solutions systematically avoided by that model (e.g., certain Redcode templates GPT-style models disfavor) remain as voids in the archive. Neither FunSearch nor AlphaEvolve addresses this; while AlphaEvolve uses two models, they are different sizes from the same family, aimed at saving compute rather than seeking diversity.
- **Key Challenge**: Quality-Diversity aims to "cover the entire behavioral space," but every LLM has its own deep-seated inductive biases (writing preferences from training corpora and alignment). **Samples from a single distribution can never fill the union of multiple distributions.** Parallelism can only scale compute, not the coverage of priors.
- **Goal**: Demonstrate that under a fixed **total LLM call budget**, "QD search using a heterogeneous model ensemble" is strictly superior to "parallel QD search using the same model" and superior to a single node. Simultaneously provide an engineering implementation that tolerates latency differences between models (a local 35B Qwen might be 10× slower than a cloud frontier model).
- **Key Insight**: Treat "model identity" itself as a source of diversity for QD. Each node runs an LLM from a different family, allowing them to occupy the niches they are best at within the behavioral space. Champions are injected into each other's opponent pools and archives via gossip, forming a "cross-model Red Queen."
- **Core Idea**: Upgrade homogeneous parallel computation to **parallel cognition**. Diversity originates from the prior differences of different models rather than just temperature sampling.

## Method
DEI retains the single-node pipeline of DRQ (code unchanged) and wraps it in an asynchronous communication layer, allowing $N$ nodes running different LLMs to feed each other champions. The following explains "what a node does in one round" and "how nodes communicate."

### Overall Architecture
Each node is a dual entity consisting of an (Async Communication Layer + Local DRQ Optimizer). The local optimizer is MAP-Elites: it maintains a 2D archive where behavioral axes are **TSP** (warrior code length × average survival time) and **MC** (proportion of core memory addresses touched during battle). Each cell stores the warrior with the highest current fitness. Each round consists of $T$ LLM calls, with 10% generating warriors from scratch and 90% sampling an elite from the archive for the LLM to mutate. Fitness is calculated using the formula $f(w_i, \mathcal{O}) = \sum_{\tau \in \mathcal{T}} \frac{N}{|\mathcal{T}|} \frac{A^i_\tau}{\sum_{o \in \mathcal{O}} A^o_\tau}$ in the MARS simulator against the opponent set $\mathcal{O}$. New solutions replace archive entries if they fill an empty cell or refresh the highest score in a cell. At the end of each round, a champion $\hat w_r = \arg\max_{w \in \mathcal{A}_r} f(w, \mathcal{O}_r)$ is selected and broadcast to all peers via gossip. Received peer champions are both added to the local opponent pool (creating cross-model adversarial pressure) and seeded into the local archive's empty cells (transferring cross-model diversity). Three experimental conditions—Solo (1 node), Homo Ensemble (4 nodes of the same model), and Diverse Ensemble (4 heterogeneous models: GPT-5.4-mini / Claude Sonnet 4.6 / GPT-5.2 / Claude Haiku 4.5)—share the same **total** call budget (per-node quota scaled by $1/N$) to ensure a fair comparison.

### Key Designs

1.  **Heterogeneous LLMs as Mutation Operators + Cross-Model Red Queen**:
    - **Function**: Each node is permanently bound to an LLM from a different family as the MAP-Elites mutation/generation operator. Each round's champion enters the peer's opponent pool $\mathcal{O}_i \leftarrow \mathcal{O}_i \cup \mathcal{R}$, forcing each node's fitness evaluation to contend with "strategies it could never write itself."
    - **Mechanism**: Fitness in the formula $f(w_i, \mathcal{O})$ explicitly depends on the opponent set $\mathcal{O}$. Thus, when a fortress warrior written by Claude—utilizing Claude's preferred patterns—suddenly appears in a GPT node's archive as an opponent, GPT must evolve solutions to counter it to gain high scores. This is the multi-distribution adversarial pressure that single-node self-play cannot achieve. Simultaneously, niche novelty $\eta = \mathbb{E}[\mathbf{1}[\mathbf{bc}(w) \notin \mathcal{A}_i^{(r-1)}]]$ measures the proportion of received champions falling into empty archive cells. In experiments, diverse $\eta \approx 0.45$ is much higher than homo $\eta \in [0.09, 0.35]$, directly proving that heterogeneous models indeed "cover each other's blind spots."
    - **Design Motivation**: Traditional parallel EAs assume operators are fixed and workers generate diversity via sampling noise. This paper elevates "model identity" to a first-order diversity source for QD—something neither the "same-family size pairing" of AlphaEvolve nor the "single-model multi-worker" approach of FunSearch achieved.

2.  **Fully Asynchronous Gossip Champion Sharing**:
    - **Function**: Allows nodes with 10× latency differences, like a local MLX Qwen-35B (~10s/call) and a cloud frontier (~2s/call), to coexist in the same ensemble without the slowest node dragging others down.
    - **Mechanism**: The "end-of-round sync barrier" is abandoned for non-blocking all-gather. Each node publishes its champion to all peers immediately upon selection and drains its receiving buffer at the start of the next round. Fast nodes do not wait for slow ones; slow nodes might receive champions from several rounds prior, but because the QD archive is inherently "cumulative," late champions can still fill empty cells or refresh elites without becoming obsolete. The underlying layer uses a Yggdrasil overlay to assign stable IPv6 addresses to each node for NAT traversal.
    - **Design Motivation**: Synchronization would cause frontier models to waste compute while waiting for a local 35B model, making the addition of a slow node actually decrease throughput. This would force engineering toward homogeneous hardware, contradicting the goal of "democratizing heterogeneous model collaboration." The asynchronous design ensures that "adding a slow laptop node" is strictly additive.

3.  **Compute-Equivalent Three-Condition Comparison Protocol**:
    - **Function**: Eliminates the major alternative explanation that "Diverse wins only because it uses more compute," turning "whether diversity brings gains" into a falsifiable experimental design.
    - **Mechanism**: The three conditions (Solo / Homo / Diverse) share the same **total** LLM call budget—e.g., Solo 250 iters/round vs. 4 nodes × 62 iters/round ≈ 248 calls—ensuring per-node quotas are scaled strictly by $1/N$. Two sets of metrics are reported: (a) local archive champion generality (win rate against human-written warrior set $\mathcal{H}$) and niche novelty for each node, and (b) merged archive (best per cell across nodes) QD-Score and coverage. The merged results at equal compute serve as the most critical comparison.
    - **Design Motivation**: The QD literature is most susceptible to the criticism: "Did you just spend more compute?" By locking the total budget and examining both individual and merged archives, "compute gain" and "diversity gain" are clearly decoupled.

### Loss & Training
Gradient-free training. All LLM calls are performed at inference time. Archive update rules follow standard MAP-Elites replacement logic. Each node has $T$ calls per round ($T \approx 62$ for 4 nodes; $T = 250$ for solo). MARS configuration: core 8000 instructions, 80,000 cycles max per match, 20 matches per warrior pair. Two prompt templates (initialization/mutation) are reused from the original DRQ repository.

## Key Experimental Results

### Main Results
All conditions share the total LLM call budget. Fitness is evaluated against the round champion pool in the MARS simulator. Final generality reports the win rate against a set of human-written warriors $\mathcal{H}$.

| Model / Condition | Peak Generality | Niche Novelty η | Remarks |
| :--- | :--- | :--- | :--- |
| Diverse Ensemble (Claude Sonnet 4.6) | **0.850 ± 0.087** | 0.483 ± 0.120 | Best overall, highest η |
| Homo Ensemble (Claude Sonnet 4.6) | 0.825 ± 0.106 | 0.348 ± 0.039 | Parallel homo already better than solo |
| Solo DRQ (Claude Sonnet 4.6) | 0.775 ± 0.035 | — | Single-node baseline |
| Diverse Ensemble (GPT-5.4-mini) | 0.767 ± 0.076 | 0.422 ± 0.072 | Diverse > Homo > Solo |
| Homo Ensemble (GPT-5.4-mini) | 0.725 ± 0.029 | 0.119 ± 0.013 | η much lower than diverse |
| Diverse Ensemble (Claude Haiku 4.5) | **0.700 ± 0.050** | 0.443 ± 0.132 | Solo 0.650, Homo only 0.538 |

Merged archive (Equal compute comparison, final round):

| Condition | Coverage | QD-Score | Gain vs Solo |
| :--- | :--- | :--- | :--- |
| Solo | 63.0% | 20.46 | Baseline |
| Homo merged | 59.0% | 29.85 | coverage -4pt, QD +46% |
| **Diverse merged** | **80.6%** | **45.90** | **coverage +28%, QD +124%** |

### Ablation Study
The "ablation" in the paper is equivalent to "degrading diverse to homo or solo"—this is included in the main tables. Supplementary observations:

| Comparison | Key Metric | Description |
| :--- | :--- | :--- |
| Diverse vs Homo (4 models × 4 nodes) | Generality | Diverse wins on all 4/4 models |
| Diverse vs Homo (merged QD-Score) | 45.90 vs 29.85 (+54%) | Under same compute, diversity is the main gain |
| Diverse vs Homo (merged Coverage) | 80.6% vs 59.0% (+22pt) | Coverage gain **only** appears in diverse condition |
| Niche novelty η (Homo → Diverse) | 0.09–0.35 → 0.42–0.48 | Proportion of champions falling into empty cells rises significantly |

### Key Findings
- **Homogeneous parallelism only boosts QD-Score, not coverage**: Homo merged coverage (59%) is even lower than Solo (63%), indicating that 4 homogeneous nodes are highly redundant in the behavioral space. Coverage only crosses 80% when heterogeneous priors are introduced.
- **Small models benefit most from diversity**: Claude Haiku 4.5 goes from 0.650 in solo to 0.700 in diverse. GPT-5.2 goes from 0.650 to 0.767. Weak models "piggyback" on the champions of strong models to reach niches they cannot generate themselves, providing evidence for "democratization" (mixing local small models with cloud frontier models).
- **Async gossip makes slow nodes a pure gain**: Because there is no barrier, adding a local MLX node strictly increases coverage without dragging down frontier node throughput. This is the engineering prerequisite for heterogeneous schemes to scale in mixed-hardware environments.

## Highlights & Insights
- **"Model identity" as a first-order QD diversity source**: Traditionally, diversity sources are temperature/sampling noise/BC dimension design. This paper is the first to elevate "which LLM is used" to a diversity dimension on par with BC—a very general perspective applicable to any LLM-driven search.
- **Clean compute-equivalent protocol**: Applying the "$N$ times parallel means $1/N$ quota" constraint while reporting both individual and merged metrics clearly decouples compute gains from diversity gains. This design is a reusable trick for any multi-agent/multi-model work.
- **Niche novelty metric directly quantifies complementarity**: $\eta = \Pr[\mathbf{bc}(w_{\text{peer}}) \notin \mathcal{A}_i]$ is a simple measurement that turns "how surprised model A is by model B's champion" into an observable signal. It is an inexpensive diagnostic for determining if two generators are truly complementary.
- **Async gossip with Yggdrasil overlay**: Provides a production-ready NAT traversal solution for mixing local open-weight models and cloud frontier models, a key engineering contribution for collaborative search.

## Limitations & Future Work
- Verified only in the Core War domain. Redcode has a highly regular structure with clear BC axes (TSP/MC). Whether this replicates in tasks where BC is hard to define or fitness evaluation is expensive (e.g., end-to-end robot control) is unverified.
- Model allocation is **static**: Each node is permanently bound to one LLM without dynamic scheduling (e.g., giving more compute to models with higher recent contributions). This is a space for "adaptive topology / heterogeneous BC axes" mentioned in future work.
- Lacks systematic ablation on "how many models are enough" or "how to select the optimal model combination." Niche novelty $\eta$ varies by 4× between different pairs, suggesting combination selection is its own optimization problem.
- No comparison with other combined diversity methods: For example, Homo with high temperature, Quality-Diversity through AI Feedback, or DARLING. Whether heterogeneous ensembles are orthogonal to these methods remains an open question.
- "Champion sharing" is a single communication protocol; more fine-grained transfers (entire archive subsets? mini-archives with BC annotations?) were not explored.

## Related Work & Insights
- **vs FunSearch (Romera-Paredes 2023)**: Similar large-scale parallel LLM evolution, but FunSearch uses the same model for the entire pool, relying on temperature; DEI switches model families to theoretically cover wider program distributions.
- **vs AlphaEvolve (Novikov 2025)**: Also uses LLM ensembles but pairs large/small models from the same family to save compute; DEI pairs cross-family models (Claude + GPT) to capture different priors for coverage.
- **vs DRQ (Kumar 2026)**: This work is a direct superset extension of DRQ—moving from single node to multi-node + cross-model competition. Red Queen pressure is upgraded from intra-model self-play to inter-model adversarial.
- **vs Island-model EA (Cantu-Paz 2001)**: The classic island model uses the same operator with periodic migration between sub-populations; DEI is equivalent to each island running a different operator with asynchronous pub-sub migration of champions.
- **vs Multi-agent debate (Liang 2024) / DARLING (Li 2025)**: These works introduce diversity in RL post-training or inference to improve reasoning; DEI applies the same intuition to the QD search phase.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to treat "heterogeneous LLM family" as a first-order QD diversity source with a strict controlled comparison.
- **Experimental Thoroughness**: ⭐⭐⭐ The compute-equivalent protocol is clean, but it is limited to the Core War domain and fixed combinations.
- **Writing Quality**: ⭐⭐⭐⭐ The argumentation chain (motivation—hypothesis—equivalent comparison—metric layering) is very clear.
- **Value**: ⭐⭐⭐⭐ Provides a reproducible engineering paradigm for collaborative search mixing local and cloud models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Soft Quality-Diversity Optimization](../../ICLR2026/llm_evaluation/soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/llm_evaluation/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ICML 2026\] BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback](bespoke_benchmark_for_search-augmented_large_language_model_personalization_via_.md)
- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](../../NeurIPS2025/llm_evaluation/efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[ICML 2026\] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay](politicsbench_benchmarking_political_values_in_large_language_models_with_multi-.md)

</div>

<!-- RELATED:END -->
