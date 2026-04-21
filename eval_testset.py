"""
eval_testset.py
───────────────
Curated test set for the V3 evaluation harness.

Why this exists:
    With N=5 live user sessions we cannot make stable claims about whether V3
    is better than V2. A curated test set of 10 hand-written answers lets us
    make *deterministic*, reproducible comparisons: the same input always
    produces the same retrieval, so any difference we observe is attributable
    to the method, not to noise.

Selection criteria:
    - Cover the four main skill buckets in the knowledge base: conflict,
      leadership/initiative, technical depth, stakeholder management.
    - Include all three quality levels (strong / medium / weak) to test
      whether the system correctly differentiates them.
    - Mix behavioural and technical questions to test classification +
      retrieval across question types.
    - Include at least two adversarial cases where TF-IDF is expected to fail
      (semantic rephrasing that doesn't share vocabulary with the KB).

Ground-truth "expected retrievals":
    For each test case, we record which KB documents would ideally surface
    in the top-3 strong results. This lets the harness compute a simple
    recall@3 metric per retrieval mode.
"""

TEST_SET = [

    # ─────────────────────────────────────────────────────────────────────
    # 1 · BEHAVIOURAL · Conflict Resolution · STRONG answer
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "conflict_strong",
        "question": "Tell me about a time you handled a disagreement between two team members.",
        "answer": (
            "Two engineers on my team disagreed about whether to refactor our auth "
            "module before a launch. I scheduled a 45-minute session where each "
            "presented their approach with explicit trade-offs on timeline, risk, "
            "and long-term maintainability. We ended up shipping the patch first "
            "and scheduling the refactor for the next sprint. Both engineers "
            "signed off and we launched on time."
        ),
        "job_title": "Engineering Manager",
        "job_description": "Lead a team of 6 engineers. Strong facilitation and stakeholder management skills required.",
        "question_type": "Behavioral",
        "expected_skill": "conflict resolution",
        "expected_quality_tier": "strong",
        "notes": "Baseline strong answer with clear vocab overlap to KB — both retrievers should do well.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 2 · BEHAVIOURAL · Conflict Resolution · WEAK answer
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "conflict_weak",
        "question": "Tell me about a time you handled a disagreement between two team members.",
        "answer": (
            "I try to stay out of conflicts because I think it's better when the "
            "manager handles it. My teammates usually work things out themselves."
        ),
        "job_title": "Engineering Manager",
        "job_description": "Lead a team of 6 engineers. Strong facilitation and stakeholder management skills required.",
        "question_type": "Behavioral",
        "expected_skill": "conflict resolution",
        "expected_quality_tier": "weak",
        "notes": "Tests whether the weak-example retrieval surfaces a relevant 'what not to do' example.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 3 · ADVERSARIAL · Conflict without the word "conflict"
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "conflict_semantic_rephrase",
        "question": "Describe a situation where you had to bring opposing viewpoints together.",
        "answer": (
            "Our design and engineering leads had incompatible views on whether to "
            "prioritise pixel-perfect fidelity or shipping velocity. I facilitated a "
            "workshop where we mapped each feature to a 2x2 of user impact vs build "
            "cost, and we converged on a phased plan. Both sides felt heard and "
            "the roadmap shipped on schedule."
        ),
        "job_title": "Product Manager",
        "job_description": "Cross-functional product leadership. Strong stakeholder alignment and facilitation skills.",
        "question_type": "Behavioral",
        "expected_skill": "conflict resolution",
        "expected_quality_tier": "strong",
        "notes": (
            "ADVERSARIAL: the answer is about conflict resolution but never uses "
            "the word 'conflict' or 'disagreement'. TF-IDF is expected to miss "
            "the best KB matches; semantic is expected to find them."
        ),
    },

    # ─────────────────────────────────────────────────────────────────────
    # 4 · BEHAVIOURAL · Leadership / Initiative · STRONG
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "leadership_strong",
        "question": "Give me an example of when you took initiative without being asked.",
        "answer": (
            "I noticed our bug triage process was taking us 3+ days to acknowledge "
            "customer reports. Nobody asked me to fix it, but I built a lightweight "
            "SLA dashboard in a weekend that flagged tickets aging past 24 hours. "
            "Within a month our median acknowledgement time dropped from 72 to 8 hours."
        ),
        "job_title": "Senior Software Engineer",
        "job_description": "Autonomous engineer who drives improvements proactively.",
        "question_type": "Behavioral",
        "expected_skill": "process improvement",
        "expected_quality_tier": "strong",
        "notes": "Clear STAR structure with concrete metrics.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 5 · BEHAVIOURAL · Leadership · WEAK (generic claims)
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "leadership_weak",
        "question": "Give me an example of when you took initiative without being asked.",
        "answer": (
            "I am a very proactive and driven person and I always go the extra mile "
            "for my team. People often say I'm a natural leader."
        ),
        "job_title": "Senior Software Engineer",
        "job_description": "Autonomous engineer who drives improvements proactively.",
        "question_type": "Behavioral",
        "expected_skill": "leadership",
        "expected_quality_tier": "weak",
        "notes": "Generic claims, zero evidence. Should trigger the 'vague language' weak KB example.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 6 · TECHNICAL · Data / SQL · MEDIUM
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "technical_sql_medium",
        "question": "How would you identify duplicate customer records in a large table?",
        "answer": (
            "I would write a SQL query using GROUP BY on the columns that should "
            "be unique like email or phone, and filter with HAVING COUNT(*) > 1. "
            "Then I'd manually review the duplicates before deleting."
        ),
        "job_title": "Data Analyst",
        "job_description": "SQL proficiency, analytical thinking, data quality experience.",
        "question_type": "Technical",
        "expected_skill": "sql",
        "expected_quality_tier": "medium",
        "notes": "Correct approach but shallow — no discussion of fuzzy matches, edge cases, or scale.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 7 · BEHAVIOURAL · Stakeholder · STRONG
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "stakeholder_strong",
        "question": "Tell me about a time you had to influence someone more senior than you.",
        "answer": (
            "Our VP wanted to invest $200k in a feature I believed wouldn't move "
            "our north-star metric. I built a one-page analysis comparing projected "
            "vs historical lift of similar features, and proposed a $20k A/B test "
            "first. He agreed to the smaller test, results confirmed my hypothesis, "
            "and we reallocated the budget to higher-ROI work."
        ),
        "job_title": "Product Manager",
        "job_description": "Influence senior stakeholders with data-driven arguments.",
        "question_type": "Behavioral",
        "expected_skill": "stakeholder alignment",
        "expected_quality_tier": "strong",
        "notes": "Good data-driven influence narrative.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 8 · ADVERSARIAL · Failure without the word "failure"
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "failure_semantic_rephrase",
        "question": "Describe a project that didn't go as planned.",
        "answer": (
            "I led a six-month migration of our analytics pipeline. Halfway through "
            "we realised the new tool couldn't handle our event volume at peak. "
            "I paused the migration, wrote a retrospective, and proposed a hybrid "
            "architecture instead. We delivered a month late but the hybrid approach "
            "is still running today."
        ),
        "job_title": "Senior Data Engineer",
        "job_description": "Ownership mindset, learning from setbacks, technical judgement.",
        "question_type": "Behavioral",
        "expected_skill": "learning from failure",
        "expected_quality_tier": "strong",
        "notes": (
            "ADVERSARIAL: the answer is about handling failure but uses softer "
            "framing ('didn't go as planned', 'realised'). Tests whether semantic "
            "retrieval catches the failure/learning pattern without keyword matches."
        ),
    },

    # ─────────────────────────────────────────────────────────────────────
    # 9 · BEHAVIOURAL · Teamwork · WEAK (empty)
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "teamwork_very_short",
        "question": "How do you approach working with a new team?",
        "answer": (
            "I think teamwork is really important and I always try to collaborate "
            "and communicate well."
        ),
        "job_title": "Software Engineer",
        "job_description": "Collaborative engineering culture, pair programming, code reviews.",
        "question_type": "Behavioral",
        "expected_skill": "teamwork",
        "expected_quality_tier": "weak",
        "notes": "Empty platitudes — tests whether the scorer correctly flags 'nothing concrete said'.",
    },

    # ─────────────────────────────────────────────────────────────────────
    # 10 · TECHNICAL · System Design · STRONG
    # ─────────────────────────────────────────────────────────────────────
    {
        "id": "technical_system_design_strong",
        "question": "How would you design a notification system that handles millions of users?",
        "answer": (
            "I'd separate concerns into three layers: an event producer that writes "
            "to a queue like Kafka, a fan-out worker pool that reads the queue and "
            "looks up user preferences, and a delivery layer per channel (push, "
            "email, SMS). For scale I'd shard by user_id, use batching at the "
            "delivery layer, and add a dead-letter queue for failed deliveries. "
            "The trickiest part is idempotency: I'd assign each notification a "
            "deterministic ID so retries don't double-send."
        ),
        "job_title": "Staff Software Engineer",
        "job_description": "Large-scale distributed systems. Strong system design skills.",
        "question_type": "Technical",
        "expected_skill": "system design",
        "expected_quality_tier": "strong",
        "notes": "Full system-design narrative with scale, edge cases, and trade-offs.",
    },
]


def get_test_set():
    """Public accessor — mirrors the get_knowledge_base() pattern."""
    return TEST_SET


def get_test_case(case_id: str):
    """Fetch a single test case by id."""
    for case in TEST_SET:
        if case["id"] == case_id:
            return case
    raise KeyError(f"No test case with id '{case_id}'")
