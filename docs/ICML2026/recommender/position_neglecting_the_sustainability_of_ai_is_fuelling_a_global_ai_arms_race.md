---
title: >-
  [Paper Note] Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race
description: >-
  [ICML 2026][Recommender Systems][Sustainable AI] Utilizing Karl Marx's "base-superstructure" framework, this position paper argues that current discussions on "sustainable AI" are dominated by environmental dimensions wh…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Sustainable AI"
  - "Climate-aware"
  - "Resource-aware"
  - "AI arms race"
  - "CARAML"
date: 2026-05-08
content_hash: 5738bc95e18bc1c4
---

# Position: Neglecting the Sustainability of AI is Fuelling a Global AI Arms Race

**Conference**: ICML 2026  
**arXiv**: [2502.20016](https://arxiv.org/abs/2502.20016)  
**Code**: None  
**Area**: AI Safety / Sustainable AI / AI Governance  
**Keywords**: Sustainable AI, Climate-aware, Resource-aware, AI arms race, CARAML

## TL;DR
Utilizing Karl Marx's "base-superstructure" framework, this position paper argues that current discussions on "sustainable AI" are dominated by environmental dimensions while neglecting economic and social dimensions. It calls for simultaneously elevate the **climate-aware** and **resource-aware** axes and proposes the CARAML five-layer action framework (Individual / Community / Industry / Government / Global) to curb the escalating "global AI arms race."

## Background & Motivation

**Background**: Sustainability has historically consisted of three pillars: environment, economy, and society. However, the discourse on "sustainable AI" focuses almost exclusively on the environment (carbon emissions, water consumption, e-waste), leaving "democratization / accessibility" issues to a separate group of researchers, with little dialogue between the two.

**Limitations of Prior Work**: (i) Promoting green datacenters while requiring LMICs to use low-carbon energy—despite their lack of basic electricity—exacerbates a "double penalty." (ii) Efficiency gains, such as those from DeepSeek, which reduce GPU time by an order of magnitude, trigger the Jevons paradox where the explosion of tokens in "reasoning models" leads to a net increase in energy consumption. (iii) Top500 supercomputing and AI investments are concentrated in the US, CN, and EU; in 2024, US private AI investment alone reached \$109.1 billion—exceeding the GDP of 134 countries—making "AI sovereignty" slogans nearly hollow for nations lacking material foundations.

**Key Challenge**: Compressing "sustainable AI" solely into "environmental sustainability" creates structural tension with "social sustainability"—pursuing climate awareness alone may exclude resource-poor nations, while pursuing resource awareness (increasing accessibility) drives compute expansion and rebound emissions.

**Goal**: (i) Redefine sustainable AI as being simultaneously high on both "climate-aware" and "resource-aware" axes; (ii) use a historical materialist base-superstructure model to explain how the "GPU-rich vs. GPU-poor" divide is reproduced by existing systems; (iii) provide the CARAML framework to distribute action obligations across five levels along the "agency $\times$ scope" axes.

**Key Insight**: The authors treat AI as **embodied infrastructure** in the sense of Crawford (2021)—not as abstract algorithms, but as a material aggregate of minerals, labor, electricity, and capital. The material base determines the superstructure of policy and cultural narratives; without changing the base, the narrative remains fixed.

**Core Idea**: Sustainable AI = a Quadrant high in both **climate-aware $\times$ resource-aware** dimensions; any solution that elevates only one axis (green AI / inclusive AI / SOTA AI / edge AI) collapses back into an arms race.

## Method

The "method" of this position paper consists of four argumentative components: the formalization of sustainable AI, two counter-examples of tension, the base-superstructure explanation, and the CARAML action framework.

### Overall Architecture

Section 2 decomposes sustainable AI into environmental, economic, and social pillars, introducing the "climate-aware vs. resource-aware" axes and using two counter-examples (low-carbon datacenters and efficiency rebounds) to prove that focusing on one axis degrades the whole. Section 3 uses the base-superstructure model to explain why the GPU-rich vs. GPU-poor divide persists. Section 4 presents the CARAML framework. Section 5 responds to counterarguments such as "emissions will decrease automatically" and "distilled models have solved the problem."

### Key Designs

1.  **Climate-aware $\times$ Resource-aware 2D Quadrant**:
    - **Function**: Places the vague concept of "sustainable AI" into a categorizable, debatable coordinate system, forcing any solution to declare its quadrant.
    - **Mechanism**: The horizontal axis is resource awareness (decoloniality, accessibility, opposition to resource concentration), and the vertical axis is climate awareness (carbon/water/e-waste). SOTA AI is low on both; green AI is high climate/low resource; inclusive AI is the reverse. Only the upper-right quadrant is sustainable AI. Edge AI is marked with a question mark due to difficult-to-define embodied carbon.
    - **Design Motivation**: To make explicit the implicit disagreements between the "democratization camp" and the "green AI camp"—pursuing a single axis inevitably leads to trade-offs.

2.  **Base-Superstructure Model for AI System Reproduction**:
    - **Function**: Explains why "individual effort + technical efficiency" cannot automatically dissolve the AI arms race—policies, narratives, and education in the superstructure lock in material monopolies in the base.
    - **Mechanism**: The "base" of AI is defined as compute + data + capital + labor + knowledge commodities; the "superstructure" consists of AI policies, regulations, research norms, and media narratives. Material monopolies (NVIDIA + a few hyperscalers) shape discourses like AI sovereignty and export controls, which in turn solidify the material order.
    - **Design Motivation**: Draws a historical analogy to colonial expansion—colonial capital and racial hierarchies were mutually reinforcing base and superstructure. This paper argues contemporary AI repeats this colonial structure in the GPU era.

3.  **CARAML Action Framework (Agency $\times$ Scope)**:
    - **Function**: Translates the goal of "simultaneously pulling climate and resource axes" into five specific levels from individual to global, matching levers of control at each level.
    - **Mechanism**: At each level, at least one executable action is provided—Individuals: replace pure accuracy with perf-per-resource; ML Community: pre-register large-scale experiments to eliminate waste; Industry: self-impose carbon caps; Government: integrate AI impact assessments into regulation; Global: redefine the "Right to Compute" as the "Right to Sustainable Compute."
    - **Design Motivation**: Uses the "agency $\times$ scope" diagram to prove that only multi-layer coordination can expand the sphere of influence.

### Loss & Training

Position paper; no model or loss. The evidentiary basis includes TOP500 (2025) compute distribution, Investment data from Maslej et al. (2025), author nationality distributions from ICML/ICLR/NeurIPS 2006–2025, and renewable data from IRENA/IEA.

## Key Experimental Results

### Main Results: Concentration of Global AI Compute, Investment, and Academic Participation

| Dimension | Data | Source |
| :--- | :--- | :--- |
| TOP500 Geo-concentration | Primarily US/CN/EU; Africa largely absent | TOP500 (2025) |
| 2024 US Private AI Investment | \$109.1B ($\approx 12 \times$ China, $\approx 24 \times$ UK) | Maslej et al. (2025) |
| Investment vs. National GDP | Exceeds 2024 GDP of 134 countries | Ibid. |
| LLM Leaderboard Composition | Majority of entries from corporate, non-academic entities | Moutawwakil & Pierrard (2023) |
| Compute Doubling Period | Frontier systems doubling every 3.4 months since 2012 | Sevilla et al. (2022) |
| ICML/ICLR/NeurIPS Authors (2006–2025) | Significant LMIC underrepresentation; gap widening | World Bank (2024) + Ours |

### Case Study: Training Cost of LLaMA-3.1 (405B) by Geography

| Training Location | GPU Hours | Estimated Consumption | Estimated Emissions |
| :--- | :--- | :--- | :--- |
| Generic | 30.84M H100-h | 21.5 GWh | — |
| US Grid | Ibid. | 21.5 GWh | 8,930 tCO2e |
| Sweden Grid | Ibid. | 21.5 GWh | 750 tCO2e |
| India Grid | Ibid. | 21.5 GWh | 14,737 tCO2e |

### Key Findings
- **DeepSeek Rebound**: DeepSeek-V3 used only 2.8M H800-h, an order of magnitude less than LLaMA-3, yet popularity surged 2800%, fueling "reasoning models" where an 80-token response requires $\sim 800$ latent thinking tokens, negating efficiency gains.
- **Low-Carbon Paradox**: LMICs with the weakest clean energy supplies are pressured to use renewables, suffering a "double penalty."
- **Carbon Credit Failure**: Probst et al. (2024) indicates carbon crediting projects rarely deliver promised reductions; "carbon-neutral training" claims are largely unsubstantiated.
- **Distilled models do not solve the problem**: Small models (Phi, QLoRA) amortize the embodied carbon of larger models, while manufacturing remains concentrated among a few actors.
- **IEA Projections**: By 2030, AI could consume 945 TWh, roughly equal to Japan's annual electricity consumption.

## Highlights & Insights
- The **2D Quadrant** makes the tension between "democratization" and "climate" camps visible, serving as a practical tool for reviewing research or funding.
- Integrating **NVIDIA's monopoly, export controls, and AI sovereignty** into the base-superstructure explanation provides theoretical grounding for why individual efforts are often neutralized by structures.
- The **CARAML "agency $\times$ scope" framework** offers a transferable template for multi-layered governance.
- Explicitly addressing the **rebound effect** as a counter-position acknowledges that efficiency does not equal net reduction—a rare honesty in sustainability literature.

## Limitations & Future Work
- Lacks reproducible code or a database; CARAML recommendations are largely qualitative without providing emission reduction elasticities for specific actions.
- The base-superstructure framework is a macro-narrative; the individual agency section feels strategically thin compared to structural forces.
- Relies on third-party economic reports without sensitivity analysis.
- The critique of AI sovereignty is sharp, but the specific path for LMICs to build their own sustainable compute is underdeveloped.

## Related Work & Insights
- **vs. Van Wynsberghe (2021)**: Accepts the three-pillar definition but formalizes it into a debatable 2D quadrant.
- **vs. Schwartz et al. (2020) Green AI**: Notes that "Green AI" falls into the top-left quadrant (high climate, low resource) and is not synonymous with sustainable AI.
- **vs. Patterson et al. (2022)**: Rebuts the claim that AI emissions will naturally plateau using Google’s 2023 data (+48% vs 2019 baseline).
- **vs. Ahmed & Wahed (2020)**: Extends the "compute divide" into a full base-superstructure explanation and intervening CARAML framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The integration of the 2D quadrant, base-superstructure explanation, and CARAML is unique in AI governance.
- Experimental Thoroughness: ⭐⭐⭐ Primarily relies on third-party evidence, but case studies (DeepSeek rebound, LLaMA carbon) are compelling.
- Writing Quality: ⭐⭐⭐⭐ Effectively embeds Marxist theory into an ICML-style paper while maintaining clarity.
- Value: ⭐⭐⭐⭐⭐ Expands the "sustainable AI" discourse from a single environmental axis to a dual social-environmental framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Stop Preaching and Start Practising Data Frugality for Responsible Development of AI](position_stop_preaching_and_start_practising_data_frugality_for_responsible_deve.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[AAAI 2026\] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](../../AAAI2026/recommender/moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[NeurIPS 2025\] NeurIPS Should Lead Scientific Consensus on AI Policy](../../NeurIPS2025/recommender/neurips_should_lead_scientific_consensus_on_ai_policy.md)

</div>

<!-- RELATED:END -->
