from blueprints.memory import Budget, MemoryStore, build_context, consolidate


class Clock:
    def __init__(self):
        self.t = 1_000.0

    def __call__(self):
        self.t += 1
        return self.t


def test_changed_fact_invalidates_instead_of_overwriting():
    m = MemoryStore(clock=Clock())
    m.add_fact("u1", "project", "deadline", "Friday")
    m.add_fact("u1", "project", "deadline", "Monday")
    current = m.facts("u1")
    history = m.facts("u1", include_history=True)
    assert [f.value for f in current] == ["Monday"]
    assert [f.value for f in history] == ["Friday", "Monday"]
    assert history[0].valid_until is not None


def test_same_fact_twice_is_a_noop():
    m = MemoryStore(clock=Clock())
    a = m.add_fact("u1", "user", "prefers", "weekly reports")
    b = m.add_fact("u1", "user", "prefers", "weekly reports")
    assert a.id == b.id and len(m.facts("u1")) == 1


def test_users_never_see_each_others_memory():
    m = MemoryStore(clock=Clock())
    m.add_fact("alice", "alice", "works on", "billing service")
    m.log("alice", "user", "the billing service is down again")
    assert m.search_facts("bob", "billing") == []
    assert m.search_episodes("bob", "billing") == []


def test_forget_user_clears_every_tier_and_index():
    m = MemoryStore(clock=Clock())
    m.pin("u1", "name", "Sam")
    m.log("u1", "user", "remember my allergy to peanuts")
    m.add_fact("u1", "Sam", "is allergic to", "peanuts")
    m.set_summary("u1", "Sam mentioned an allergy.", 1)
    m.forget_user("u1")
    assert m.pinned("u1") == {} and m.recent("u1", 10) == [] and m.facts("u1", True) == []
    assert m.search_facts("u1", "peanuts") == [] and m.search_episodes("u1", "peanuts") == []
    assert m.summary("u1") == ("", 0)


def test_context_stays_within_budget_and_keeps_newest_turns():
    m = MemoryStore(clock=Clock())
    for i in range(40):
        m.log("u1", "user", f"message number {i} " + "filler " * 40)
    ctx = build_context(m, "u1", "anything", Budget(window=300, window_turns=8))
    assert ctx.recent, "recent window should not be empty"
    assert ctx.recent[-1].startswith("[user] message number 39")
    assert sum(len(r.split()) for r in ctx.recent) / 0.75 <= 300


def test_consolidate_extracts_facts_and_rolls_summary():
    m = MemoryStore(clock=Clock())
    for i in range(12):
        m.log("u1", "user", f"turn {i}")
    m.log("u1", "user", "our launch moved to March")

    def extract(summary, turns):
        return [("launch", "date", "March")] if any("March" in t for t in turns) else []

    def summarise(summary, turns):
        return (summary + " " + f"{len(turns)} older turns.").strip()

    consolidate(m, "u1", extract, summarise, window_turns=8)
    assert [f.value for f in m.facts("u1")] == ["March"]
    text, through = m.summary("u1")
    assert "older turns" in text and through > 0
    ctx = build_context(m, "u1", "when is the launch?")
    assert any("March" in f for f in ctx.facts)


def test_expire_unused_retires_stale_facts():
    clock = Clock()
    m = MemoryStore(clock=clock)
    m.add_fact("u1", "user", "uses", "Python 3.8")
    clock.t += 10_000
    assert m.expire_unused("u1", older_than_seconds=5_000) == 1
    assert m.facts("u1") == [] and len(m.facts("u1", include_history=True)) == 1
