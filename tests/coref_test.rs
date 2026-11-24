//! Coreference in extraction (SPEC-7.1-O7): pronouns and short names are
//! linked as the earlier name they refer to, by the Datalog sieves in
//! `extract::coref`; a pronoun the sieves cannot resolve yields no fact.

use lemmalog::agent::{CandidateFact, Episode, Extractor};
use lemmalog::extract::{FixtureBackend, GlinerExtractor, NameLinker, RejectReason, TableLinker};

fn ep(text: &str) -> Episode {
    Episode { id: text.into(), text: text.into(), ts: 0, speaker: None }
}

fn triples(f: &[CandidateFact]) -> Vec<(&str, &str, &str)> {
    f.iter().map(|c| (c.subj.as_str(), c.pred.as_str(), c.obj.as_str())).collect()
}

fn reasons<B: lemmalog::extract::RelationBackend, L: lemmalog::extract::EntityLinker>(x: &GlinerExtractor<B, L>) -> Vec<String> {
    x.rejected().iter().map(|r| format!("{:?}", r.reason)).collect()
}

// obligation: SPEC-7.1-O7
#[test]
fn a_pronoun_takes_its_one_antecedent() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .mention("Acme", "organization")
        .relation("He", "person", "works at", "Initech", "organization", 0.9)
        .relation("It", "organization", "located in", "Austin", "location", 0.9);
    let mut x = GlinerExtractor::new(backend, NameLinker);
    let f = x.extract(&ep("Bob left Acme in May. He works at Initech. It is located in Austin."));
    // "He" -> Bob; "It" -> Initech, the nearest organisation (same sentence
    // set: Initech in the sentence before, Acme two back)
    assert_eq!(triples(&f), vec![("Bob", "works_at", "Initech"), ("Initech", "located_in", "Austin")]);
}

// obligation: SPEC-7.1-O7
#[test]
fn an_ambiguous_or_unanchored_pronoun_yields_no_fact() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .mention("Dana", "person")
        .relation("He", "person", "works at", "Acme", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, NameLinker);
    assert!(x.extract(&ep("Bob met Dana. He works at Acme.")).is_empty());
    assert_eq!(reasons(&x), vec![r#"Coreference("'He' could refer to bob or dana")"#.to_string()]);
    assert!(x.extract(&ep("He works at Acme.")).is_empty());
    assert_eq!(reasons(&x).last().unwrap(), r#"Coreference("'He' has no earlier mention to refer to")"#);
}

// obligation: SPEC-7.1-O7
#[test]
fn short_names_join_their_full_name() {
    let backend = FixtureBackend::new()
        .mention("Dana Smith", "person")
        .mention("International Business Machines", "organization")
        .relation("Smith", "person", "works at", "IBM", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, NameLinker);
    let f = x.extract(&ep("Dana Smith spoke at International Business Machines. Smith works at IBM."));
    assert_eq!(triples(&f), vec![("Dana Smith", "works_at", "International Business Machines")]);
}

// obligation: SPEC-7.1-O7
#[test]
fn resolved_pronouns_link_through_the_linker_and_keep_negation() {
    let linker = TableLinker::new()
        .entity("person:bob", Some("person"), &["Bob", "Robert"])
        .entity("org:acme", Some("organization"), &["Acme"])
        .entity("org:initech", Some("organization"), &["Initech"]);
    let backend = FixtureBackend::new()
        .mention("Robert", "person")
        .relation("He", "person", "works at", "Acme", "organization", 0.9)
        .relation("He", "person", "works at", "Initech", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, linker);
    // the antecedent is linked, in its own sentence: Robert -> person:bob
    let f = x.extract(&ep("Robert joined in 2019. He works at Acme. He no longer works at Initech."));
    assert_eq!(triples(&f), vec![("person:bob", "works_at", "org:acme")]);
    assert!(reasons(&x).contains(&"Negated".to_string()), "{:?}", reasons(&x));
}

// obligation: SPEC-7.1-O7
#[test]
fn first_person_is_still_the_speaker() {
    let backend = FixtureBackend::new()
        .mention("Bob", "person")
        .relation("I", "person", "works at", "Acme", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, NameLinker);
    let mut e = ep("Bob is my manager. I work at Acme.");
    e.speaker = Some("Dana".into());
    assert_eq!(triples(&x.extract(&e)), vec![("Dana", "works_at", "Acme")]);
}
