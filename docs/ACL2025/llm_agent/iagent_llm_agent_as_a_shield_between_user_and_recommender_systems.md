---
title: >-
  [Paper Note] iAgent: LLM Agent as a Shield between User and Recommender Systems
description: >-
  [ACL 2025][LLM Agent][User-Agent-Platform Paradigm] This paper proposes a user-agent-platform three-tier paradigm that inserts an LLM Agent as a protective layer between the user and the recommender system. It achieves personalized recommendation through instruction parsing, knowledge acquisition, reranking, and dynamic user profiling, yielding an average performance gain of 16.6% across four datasets while effectively mitigating the echo chamber effect and unfairness issues…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "User-Agent-Platform Paradigm"
  - "Recommender Systems"
  - "Information Protection Layer"
  - "Dynamic Memory"
  - "Echo Chamber Effect"
date: 2026-05-08
content_hash: 5ac61af8d9b5a4f3
---

# iAgent: LLM Agent as a Shield between User and Recommender Systems

**Conference**: ACL 2025  
**arXiv**: [2502.14662](https://arxiv.org/abs/2502.14662)  
**Code**: [https://github.com/agiresearch/iAgent](https://github.com/agiresearch/iAgent)  
**Area**: LLM Agent / Recommender Systems  
**Keywords**: User-Agent-Platform Paradigm, Recommender Systems, Information Protection Layer, Dynamic Memory, Echo Chamber Effect

## TL;DR
This paper proposes a user-agent-platform three-tier paradigm that inserts an LLM Agent as a protective layer between the user and the recommender system. It achieves personalized recommendation through instruction parsing, knowledge acquisition, reranking, and dynamic user profiling, yielding an average performance gain of 16.6% across four datasets while effectively mitigating the echo chamber effect and unfairness issues for inactive users.

## Background & Motivation

**Background**: Traditional recommender systems operate under a "user-platform" binary paradigm, where the platform's recommendation algorithms directly control the content presented to the user. Most models optimize for commercial objectives (e.g., click-through rate, conversion rate), leaving users to passively receive algorithmic recommendations.

**Limitations of Prior Work**:
   - Users lack control over recommendation results and cannot actively express their actual needs.
   - Recommendation models are trained on aggregated data of all users, meaning active users dominate collaborative learning, while the preferences of inactive users are neglected.
   - Recommending homogeneous content continuously leads to the echo chamber effect, narrowing the information users are exposed to.
   - Platforms may manipulate user behavior via algorithms (e.g., inserting advertisements, recommending high-profit products).

**Key Challenge**: Recommender platforms optimize for global commercial metrics, whereas users require personalized recommendations based on individual intent—there is a fundamental conflict of interest between the two. Existing conversational recommender systems (CRS) and Agent simulation methods still optimize on the platform side, failing to protect user interests fundamentally.

**Goal**: (1) How can users actively control recommendation results through natural language instructions? (2) How can independent modeling be done for each user to avoid cross-user interference? (3) How can the echo chamber effect and unfairness among users be mitigated simultaneously?

**Key Insight**: Inspired by the concept of a "personal secretary", an LLM Agent is introduced to act as a buffer layer between the user and the platform. Once the Agent understands user intent, it reranks the platform's recommended items without modifying the platform's underlying algorithms. This approach incurs low deployment costs and is non-intrusive to the platform.

**Core Idea**: Deploy an independent LLM Agent on the user side, shifting the core execution of personalized recommendation from the platform to the Agent.

## Method

### Overall Architecture
The inputs to the system are the user's natural language instructions (e.g., "I want to find a book about XX") and the initial recommendation list returned by the platform. The output is a reranked recommendation list based on user intent. The system consists of two versions: the basic version (iAgent) and the enhanced version with dynamic memory (i2Agent).

### Key Designs

1. **InstructRec Dataset Construction**:

    - **Function**: Starting from existing recommendation datasets like Amazon, Goodreads, and Yelp, it utilizes GPT-4o-mini to generate natural language instructions based on user reviews and randomized personas.
    - **Mechanism**: Free-text instructions are generated for each interaction record to simulate scenarios where users actively express their needs. An Instruction Cleaner is designed to filter out data leakage—if the LLM can infer the target item directly from the instruction, the instruction is excluded or downweighted.
    - **Design Motivation**: Existing recommendation datasets lack user instructions, and the conversational format of CRS is too restrictive to fully express high-level user demands.

2. **iAgent Base Architecture (Parser + Reranker + Self-Reflection)**:

    - **Function**: Parse user instructions $\rightarrow$ Extract internal/external knowledge $\rightarrow$ Rerank platform recommendation list $\rightarrow$ Validate via self-reflection.
    - **Mechanism**: The Parser $M_p$ receives the user instruction $X_I$, generates internal knowledge $X_{IK}$ and keywords $X_{KW}$, and determines whether to invoke external tools to obtain $X_{EK}$. The Reranker $M_r$ synthesizes all knowledge and the user history $X_{SU}$ to rerank the candidate list: $\mathcal{R}^* \leftarrow M_r(X_{IK} \| X_{EK} \| X_{SU} \| X_{Item} \| P_{tr})$.
    - **Design Motivation**: User instructions contain both explicit demands and implicit preferences; the Parser acts as a domain expert to extract deep intentions. Self-reflection prevents LLM hallucinations by comparing the consistency of the lists before and after reranking (experiments show it reduces the hallucination rate by over 20 times).

3. **i2Agent Dynamic Memory Mechanism (Profile Generator + Dynamic Extractor)**:

    - **Function**: Build and continuously update user profiles based on historical user feedback, and dynamically extract relevant interests according to the current instruction.
    - **Mechanism**: The Profile Generator mimics neural network training—it takes positive/negative sample pairs, lets the model make recommendations, compares them with ground-truth interactions, and updates the profile $\mathcal{F}^T$ using feedback. The Dynamic Extractor, resembling an attention mechanism, extracts relevant interest dynamics $X_{DU}$ from the static history $X_{SU}$ and the dynamic profile $\mathcal{F}^T$ in accordance with the current instruction.
    - **Design Motivation**: The user history in iAgent is static and cannot capture how interests evolve over time. The key innovation of i2Agent is that each user's memory is maintained independently, unaffected by other users' behaviors—fundamentally avoiding the issue of collaborative learning being dominated by active users.

### Loss & Training
- The entire system relies on in-context learning of the LLMs and does not require additional training of a recommendation model.
- The Profile Generator iteratively optimizes the user profile via feedback (similar to online learning).
- During evaluation, the initial ranking is obtained from the recommendation platform (consisting of 9 randomly sampled negative items and 1 positive item), and the Agent reranks this candidate list.

## Key Experimental Results

### Main Results

| Dataset | Metric | EasyRec (SOTA) | iAgent | i2Agent | Gain |
|--------|------|---------------|--------|---------|------|
| Amazon Book | HR@1 | 30.70% | 31.89% | **35.11%** | +14.4% |
| Amazon MovieTV | HR@1 | 34.96% | 38.19% | **46.43%** | +32.8% |
| Goodreads | HR@1 | 13.94% | 23.56% | **30.97%** | +122% |
| Yelp | HR@1 | 32.41% | 37.40% | **39.22%** | +21.0% |
| **Average** | All Metrics | - | - | - | **+16.6%** |

### Echo Chamber Mitigation

| Dataset | Metric | EasyRec | i2Agent | Gain |
|--------|------|---------|---------|------|
| Amazon Book | FR@1 (Ad filtering) | 68.41% | **77.15%** | +8.7% |
| Amazon Book | P-HR@3 (Diversity) | 59.28% | **64.70%** | +5.4% |
| Yelp | FR@1 | 76.45% | **87.69%** | +11.2% |
| Yelp | P-HR@3 | 61.05% | **64.48%** | +3.4% |

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Active vs. Inactive Users | HR@1 for inactive users increased from 32.93% $\rightarrow$ 37.92% (+5%), narrowing the performance gap. |
| Self-Reflection | The hallucination rate was reduced by over 20 times. i2Agent had the highest hallucination rate due to longer context but remained manageable. |
| Reranking Ratio | Reranking occurred almost every time at the Top@1/3/5 positions, indicating the Agent is constantly personalizing recommendations. |

### Key Findings
- **Dynamic memory is most beneficial to inactive users**: Independent user profiles are free from interference by active users' behaviors, effectively narrowing the performance gap between active and inactive users.
- **Significant echo chamber mitigation**: Guided by user intentions during reranking, i2Agent can effectively identify and downweight cross-domain promotional items, leading to overall improvements in diversity metrics.
- **Instruction-aware knowledge acquisition is critical**: The domain knowledge extracted by the Parser helps the Reranker comprehend high-level user preferences.

## Highlights & Insights
- **Utility of the "User-Side Agent" paradigm**: By inserting a protective layer on the user's side without modifying platform algorithms, this approach mimics hiring a personal shopping counselor in the physical world. It is easy to deploy, non-intrusive to platforms, and ready to use immediately.
- **Dynamic memory as individualized online learning**: The iterative update mechanism of the Profile Generator essentially uses LLMs to simulate online learning, enabling an independent learning trajectory for each individual user.
- **Necessity of Self-Reflection**: LLMs are prone to hallucinating (e.g., outputting items not present in the candidate list) during reranking. The self-reflection mechanism successfully resolves this issue via simple set consistency checks.

## Limitations & Future Work
- **Validated only in recommendation scenarios**: It remains unknown whether the user-side Agent paradigm can scale to other user-platform interaction scenarios, such as search engines or social media feeds.
- **High LLM inference cost**: Every recommendation requires multiple LLM calls (Parser + Reranker + Self-Reflection + Profile Generator + Dynamic Extractor), which can introduce latency and expenses that limit large-scale deployment.
- **Realism of instruction data**: Instructions in InstructRec are generated by GPT-4o-mini based on reviews, which may not fully reflect how real users express their intents.
- **Evaluations restricted to English**: The effectiveness of instructions in other languages remains to be tested.

## Related Work & Insights
- **vs. Conversational Recommender Systems (CRS)**: CRS is restricted in conversation format and optimized on the platform side, whereas iAgent allows users to fully express their needs in free-form text.
- **vs. AgentCF**: AgentCF uses agents to simulate user behaviors to optimize platform-side recommendation models, retaining a platform perspective; iAgent runs independently on the user side.
- **vs. EasyRec**: EasyRec aligns collaborative filtering signals via pre-training but remains limited by global optimization; the individualized design of iAgent fundamentally resolves this limitation.

## Rating
- Novelty: ⭐⭐⭐⭐ The user-side Agent + dynamic memory paradigm is highly novel, though the underlying core technique (LLM reranking) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on four datasets, combined with echo chamber/diversity/activity group analyses and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and intuitive description of the three-tier paradigm.
- Value: ⭐⭐⭐⭐ Holds direct application value for the fairness of recommender systems and user protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)
- [\[NeurIPS 2025\] TAI3: Testing Agent Integrity in Interpreting User Intent](../../NeurIPS2025/llm_agent/tai3_testing_agent_integrity_in_interpreting_user_intent.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](multi_agent_dialect_bias_privacy_qa.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)

</div>

<!-- RELATED:END -->
