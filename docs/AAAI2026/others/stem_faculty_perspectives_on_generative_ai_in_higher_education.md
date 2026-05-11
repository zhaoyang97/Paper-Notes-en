---
title: >-
  [Paper Note] STEM Faculty Perspectives on Generative AI in Higher Education
description: >-
  [AAAI 2026][Generative AI] Through focus group research with 29 STEM faculty at a large public university in the United States, this study reveals how instructors integrate GenAI into teaching…
tags:
  - "AAAI 2026"
  - "Generative AI"
  - "Higher Education"
  - "STEM Pedagogy"
  - "Focus Groups"
  - "Teaching Strategies"
date: 2026-05-08
content_hash: c869a83213d79a94
---

# STEM Faculty Perspectives on Generative AI in Higher Education

**Conference**: AAAI 2026
**arXiv**: [2603.04001](https://arxiv.org/abs/2603.04001)
**Code**: None
**Area**: AI Education / Social Science
**Keywords**: Generative AI, Higher Education, STEM Pedagogy, Focus Groups, Teaching Strategies

## TL;DR

Through focus group research with 29 STEM faculty at a large public university in the United States, this study reveals how instructors integrate GenAI into teaching, the observed benefits and challenges for student learning, and the institutional support required. A key finding is that GenAI shifts faculty labor from content creation to expert review and may obscure students' underlying competency gaps.

## Background & Motivation

1. **Background**: Adoption of GenAI tools in higher education has been largely student-driven, forcing faculty into a reactive stance. Some instructors have begun using GenAI for grading, course design, and content generation, while others remain cautious.
2. **Limitations of Prior Work**: (a) Academic integrity faces serious challenges, as AI-generated content cannot be reliably detected by existing tools; (b) over-reliance on GenAI may erode critical thinking and problem-solving skills; (c) existing research offers limited insight into how STEM faculty collectively interpret and negotiate the role of GenAI.
3. **Key Challenge**: GenAI increases assignment submission rates but may mask students' insufficient understanding of core concepts. Faculty face a dilemma between "banning it and falling behind" versus "permitting it and potentially harming learning."
4. **Goal**: To understand, from a faculty perspective, the current state, benefits, challenges, and support needs associated with GenAI in STEM higher education.
5. **Key Insight**: A qualitative focus group study comprising 7 sessions with 29 faculty members, 90 minutes per session, using semi-structured discussion protocols.
6. **Core Finding in One Sentence**: Effective GenAI integration requires rethinking assessment, pedagogy, and institutional governance—not merely technological adoption.

## Method

### Overall Architecture

A qualitative research design using focus group methodology. Three core research questions: (RQ1) How do faculty integrate GenAI into course design and learning activities? (RQ2) What benefits and challenges to student learning do faculty observe? (RQ3) What institutional resources and policies are needed to support effective GenAI adoption?

### Key Findings

1. **Shift in Faculty Labor (RQ1)**

    - Findings: 93% of faculty use GenAI; the most common applications include generating quiz/assessment items, creating assignment scenarios, developing rubrics, improving assignment instructions, and summarizing feedback.
    - Key Insight: GenAI does not reduce faculty workload; rather, it shifts labor from **content creation to expert review**—faculty spend more time checking, refining, and validating AI-generated content. "The efficiency gain is an illusion."
    - Pedagogical Applications: In CS courses, students first use GenAI to generate code snippets and then integrate them; in chemistry courses, GenAI generates Python data visualization code; "dual-solution tasks" require students to compare AI-generated and human-generated solutions.

2. **Illusion of Student Competence (RQ2)**

    - Benefits: Higher on-time submission rates, particularly in CS courses where GenAI helps students overcome technical barriers; GenAI acts as a "personal teaching assistant" providing immediate help, especially beneficial for working students.
    - Challenges: **Competence illusion**—higher submission rates but students cannot debug AI-generated code because they do not understand the underlying logic; over-reliance on GenAI may circumvent critical thinking processes.
    - Assessment Responses: Some faculty **return to traditional assessments** (paper-and-pencil exams, oral examinations), while simultaneously **designing new AI-integrated assignments** (requiring students to compare and critique AI outputs)—a dual-strategy approach.

3. **Institutional Support Needs (RQ3)**

    - Training Needs: (a) Workshops on GenAI fundamentals (understanding LLM mechanisms); (b) prompt engineering training; (c) task-specific training (e.g., using AI to create rubrics).
    - Resource Needs: Shared prompt libraries, case study repositories, dedicated AI consulting teams, and communities of practice (e.g., "AI Squares").
    - Policy Needs: **Cross-course consistency**—students are confused by inconsistent GenAI usage rules across courses. Department-level guidelines and cross-course coordination are needed. However, faculty also cautioned against **premature permanent institutional changes**, given that the impact of GenAI on higher education remains highly uncertain.
    - Curricular Reform: Proposals include a mandatory AI literacy course for all students and a rethinking of the differential role of GenAI in lower-division versus upper-division courses.

### Research Methodology

Seven focus group sessions with 29 participants (11 CS, 4 engineering, 4 psychology, and others), conducted remotely via Zoom. A Qualtrics demographic questionnaire and 8 open-ended questions were used. Data analysis involved Zoom auto-transcription, Google NotebookLM-assisted thematic analysis, and manual verification.

## Key Experimental Results

### Main Results

| Dimension | Data |
|-----------|------|
| Number of Participants | 29 STEM faculty |
| Proportion Using GenAI | 93% (27/29) |
| Usage Frequency Distribution | Occasionally 41%, Frequently 31%, Rarely 21%, Never 7% |
| AI Familiarity (Very/Extremely Familiar) | 72% (AI), 73% (GenAI) |

### Usage Scenario Distribution

| Use Case | Number of Faculty |
|----------|-------------------|
| Classroom discussion | 5 |
| Assessment/practice question generation | 4 |
| Student feedback | 2 |
| Student research | 1 |
| Automated grading | 1 |

### Key Findings

- **Labor Shift, Not Labor Reduction**: Faculty spend less time creating from scratch but more time reviewing and validating—net workload may remain unchanged.
- **Higher Submission Rates but Questionable Depth of Understanding**: More students complete assignments, but many cannot explain or debug AI-generated solutions.
- **Unreliable Detection Tools**: Faculty broadly agree that current GenAI detection tools produce false positives and fail to identify sophisticated usage.
- **"AI-proof" Assignments Are Unsustainable**: Some faculty have abandoned attempts to design tasks that GenAI cannot complete.
- **Dual-Strategy Assessment**: Simultaneous return to traditional examinations alongside design of critically-oriented AI-integrated assignments.

## Highlights & Insights

- **"Content Creation → Expert Review" Labor Shift Model**: This precisely characterizes the actual impact of GenAI on faculty work, contrasting with the optimistic narrative that "AI reduces workload."
- **"Competence Illusion" Concept**: High submission rates mask low comprehension—current assessment methods cannot distinguish genuine student ability from AI-assisted surface performance.
- **"University-Level Guardrails + Department-Level Autonomy" Policy Framework**: A pragmatic recommendation for balancing consistency with flexibility.

## Limitations & Future Work

- Participants are drawn from a single university's STEM college; the sample of 29 is small, with CS faculty over-represented at 38%.
- No complementary study from the student perspective.
- Social desirability bias inherent in focus groups—faculty may be reluctant to acknowledge heavy AI use in front of colleagues.
- Disciplinary differences outside STEM are not addressed.

## Related Work & Insights

- **vs. Student-Perspective Studies**: A large body of research examines how students use GenAI; this paper fills a gap by adopting the faculty perspective.
- **vs. Policy Documents**: Policy documents typically propose idealized frameworks, whereas this paper provides grounded experiences and needs from frontline instructors.
- The findings have direct reference value for AI education policymakers.

## Rating

- Novelty: ⭐⭐⭐ The focus group methodology is established rather than novel; the primary contribution is empirical insight
- Experimental Thoroughness: ⭐⭐⭐ 29 participants are sufficient for qualitative research but limit generalizability
- Writing Quality: ⭐⭐⭐⭐ Well-organized with systematic presentation of findings
- Value: ⭐⭐⭐⭐ Directly applicable to educational policy-making

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging the Skills Gap: A Course Model for Modern Generative AI Education](bridging_the_skills_gap_a_course_model_for_modern_generative_ai_education.md)
- [\[AAAI 2026\] Higher-Order Responsibility](higher-order_responsibility.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [\[AAAI 2026\] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)

</div>

<!-- RELATED:END -->
