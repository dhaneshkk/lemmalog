//! Coreference links as retractable memory facts (SPEC-7.1-O8) and the
//! gated second tier (SPEC-7.1-O9).

use lemmalog::agent::{AgentMemory, CandidateFact};
use lemmalog::extract::coref::{Gate, PairScorer};
use lemmalog::extract::{FixtureBackend, GlinerExtractor, LazyExtractor, Mention, NameLinker};

type Memory = AgentMemory<LazyExtractor>;

fn memory(backend: FixtureBackend, scorer: Option<Box<dyn PairScorer>>) -> Memory {
    let mut x = GlinerExtractor::new(backend, NameLinker).with_coref_facts(true);
    if let Some(s) = scorer {
        x = x.with_pair_scorer(s, Gate::default());
    }
    AgentMemory::new(LazyExtractor::ready("gliner: fixture", Box::new(x)), "").unwrap()
}

fn current(m: &Memory, subj: &str) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = m
        .engine
        .relation_keys("current")
        .into_iter()
        .filter(|k| m.engine.interner.display(&k[0]) == subj)
        .map(|k| (m.engine.interner.display(&k[1]).to_string(), m.engine.interner.display(&k[2]).to_string()))
        .collect();
    v.sort();
    v
}

fn coref_rows(m: &Memory) -> Vec<String> {
    let mut v: Vec<String> = m
        .engine
        .relation_keys("coref")
        .into_iter()
        .map(|k| k.iter().map(|x| m.engine.interner.display(x).to_string()).collect::<Vec<_>>().join(" "))
        .collect();
    v.sort();
    v
}

// obligation: SPEC-7.1-O8
#[test]
fn a_coref_link_is_a_fact_and_identity_is_derived() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .relation("He", "person", "works at", "Initech", "organization", 0.9);
    let mut m = memory(backend, None);
    m.observe_text("Bob joined in 2019. He works at Initech.", 10, None);
    m.maintain(10);
    // coref(mention, antecedent, conf per mille, episode)
    assert_eq!(coref_rows(&m), vec!["m:ep1:1:0-2 m:ep1:0:0-3 1000 ep1".to_string()]);
    // the fact is stored under the mention node, and derived for Bob
    assert_eq!(current(&m, "Bob"), vec![("works_at".into(), "Initech".into())]);
    assert!(current(&m, "m:ep1:1:0-2").is_empty(), "a resolved mention node is not shown as itself");
    let why = m.engine.why("current", &[m.engine.interner.lookup("Bob").map(lemmalog::intern::Value::Sym).unwrap(),
        m.engine.interner.lookup("works_at").map(lemmalog::intern::Value::Sym).unwrap(),
        m.engine.interner.lookup("Initech").map(lemmalog::intern::Value::Sym).unwrap()]);
    assert!(why.contains("refers") && why.contains("coref"), "{why}");
}

// obligation: SPEC-7.1-O8
#[test]
fn retracting_a_link_withdraws_what_it_derived_and_reopens_what_it_closed() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .relation("Bob", "person", "works at", "Acme", "organization", 0.9)
        .relation("He", "person", "works at", "Initech", "organization", 0.9);
    let mut m = memory(backend, None);
    m.observe_text("Bob works at Acme.", 10, None);
    m.maintain(10);
    // works_at is exclusive: "He" (= Bob) at Initech supersedes Acme
    m.observe_text("Bob was promoted. He works at Initech.", 20, None);
    m.maintain(20);
    assert_eq!(current(&m, "Bob"), vec![("works_at".into(), "Initech".into())]);
    // the link was wrong: retract it
    assert_eq!(m.retract_coref("m:ep2:1:0-2"), 1);
    m.maintain(21);
    assert_eq!(current(&m, "Bob"), vec![("works_at".into(), "Acme".into())], "Acme reopened, Initech withdrawn");
    assert_eq!(current(&m, "m:ep2:1:0-2"), vec![("works_at".into(), "Initech".into())], "the mention keeps its own fact");
}

// obligation: SPEC-7.1-O8
#[test]
fn a_plain_fact_supersedes_through_a_link_too() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .relation("He", "person", "works at", "Initech", "organization", 0.9)
        .relation("Bob", "person", "works at", "Globex", "organization", 0.9);
    let mut m = memory(backend, None);
    m.observe_text("Bob joined. He works at Initech.", 10, None);
    m.maintain(10);
    m.observe_text("Bob works at Globex.", 20, None);
    m.maintain(20);
    assert_eq!(current(&m, "Bob"), vec![("works_at".into(), "Globex".into())]);
}

/// Scores by a fixed table: (anaphor, candidate) -> score.
struct Table(Vec<(&'static str, &'static str, f32)>);

impl PairScorer for Table {
    fn name(&self) -> &str {
        "table"
    }
    fn score(&mut self, _s: &[String], a: &(usize, Mention), c: &[(usize, Mention)]) -> Result<Vec<f32>, String> {
        Ok(c.iter().map(|(_, m)| self.0.iter().find(|(x, y, _)| *x == a.1.text && *y == m.text).map_or(0.0, |t| t.2)).collect())
    }
}

// obligation: SPEC-7.1-O9
#[test]
fn tier_two_resolves_only_abstentions_through_the_gate_and_its_score_is_the_confidence() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .mention("Dana", "person")
        .relation("He", "person", "works at", "Acme", "organization", 0.9);
    let scorer = Table(vec![("He", "Bob", 0.9), ("He", "Dana", 0.3)]);
    let mut m = memory(backend, Some(Box::new(scorer)));
    m.observe_text("Bob met Dana. He works at Acme.", 10, None);
    m.maintain(10);
    // the sieve abstained (two people); tier 2 passed the gate at 0.9
    assert_eq!(coref_rows(&m), vec!["m:ep1:1:0-2 m:ep1:0:0-3 900 ep1".to_string()]);
    assert_eq!(current(&m, "Bob"), vec![("works_at".into(), "Acme".into())]);
    // the derived fact carries the link's confidence (product t-norm)
    let bob = m.engine.interner.lookup("Bob").map(lemmalog::intern::Value::Sym).unwrap();
    let conf = m.engine.query("current", &[Some(bob), None, None]).into_iter().next().unwrap().1.conf;
    assert!(conf <= 0.9 + 1e-9 && conf > 0.5, "{conf}");
}

// obligation: SPEC-7.1-O9
#[test]
fn a_closed_gate_leaves_the_pronoun_unresolved() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .mention("Dana", "person")
        .relation("He", "person", "works at", "Acme", "organization", 0.9);
    let scorer = Table(vec![("He", "Bob", 0.9), ("He", "Dana", 0.85)]);
    let mut m = memory(backend, Some(Box::new(scorer)));
    let r = m.observe_text("Bob met Dana. He works at Acme.", 10, None);
    m.maintain(10);
    assert_eq!(r.report.added, 0);
    assert!(coref_rows(&m).is_empty());
}

#[test]
fn line_protocol_can_state_links_directly() {
    // an agent (or a person) can assert and later retract a link by hand
    let mut m: AgentMemory<LazyExtractor> = AgentMemory::new(LazyExtractor::off("test"), "").unwrap();
    let facts = vec![
        CandidateFact { subj: "m:x:0:0-3".into(), pred: "mention_of".into(), obj: "Dana".into(), confidence: 1.0 },
        CandidateFact { subj: "m:x:1:0-3".into(), pred: "coref".into(), obj: "m:x:0:0-3".into(), confidence: 0.8 },
        CandidateFact { subj: "m:x:1:0-3".into(), pred: "likes".into(), obj: "tea".into(), confidence: 0.9 },
    ];
    let text = facts.iter().map(|c| format!("{} --{}[{}]--> {}", c.subj, c.pred, c.confidence, c.obj)).collect::<Vec<_>>().join("\n");
    m.observe_text(&text, 5, None);
    m.maintain(5);
    assert_eq!(current(&m, "Dana"), vec![("likes".into(), "tea".into())]);
}
