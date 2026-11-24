//! Multi-pass sieve coreference, as Datalog rules on lemmalog's own engine.
//!
//! After Raghunathan et al. (2010) and Lee et al. (2013): sieves run from
//! the most precise to the least, and a mention resolved by an earlier
//! sieve is not reconsidered by a later one. Without a parser, the sieves
//! are the ones mention text and labels support:
//!
//! 1. **exact**: the same name, earlier, same label ("Acme" ... "Acme").
//! 2. **acronym**: an earlier multi-word name whose capitalised initials
//!    spell this mention ("International Business Machines" ... "IBM").
//! 3. **head**: a one-word name equal to the first or last word of an
//!    earlier multi-word name with the same label ("Dana Smith" ...
//!    "Smith", "Acme Corp" ... "Acme"), never a generic word ("Corp").
//! 4. **pronoun**: "he"/"she" take a person, "it" an organisation, team,
//!    place or thing, "they" a person, organisation or team, from the nearest sentence (this one up to
//!    the pronoun, or one of the two before) that has a candidate at all.
//!
//! Precision first, as in the rest of the extractor: a mention resolves only
//! when its sieve's candidates are **one entity**. "Bob met Dana. He works
//! at Acme." leaves "he" ambiguous (no gender is guessed from names), and
//! "Dana Smith and John Smith ... Smith" leaves "Smith" ambiguous; both are
//! reported, and no fact is extracted for them. Antecedents always precede
//! the mention (no cataphora).
//!
//! Rust only computes lexical features (lower-cased text, first and last
//! word, initials, pronoun class); every decision is a rule in
//! [`SIEVE_RULES`], so each link has a `why` proof from the engine.

use std::collections::HashMap;

use super::pipeline::Mention;
use crate::eval::{Ann, Engine};
use crate::intern::Value;

/// The sieve program. EDB, from [`resolve`]:
///
/// ```text
/// mention(M, Sentence, Position, Label)   Position = Sentence * 1_000_000 + byte start
/// name(M)  pron(M, Class)                 Class: "he" | "she" | "it" | "they"
/// norm(M, Text)  ntok(M, N)  first_tok(M, T)  last_tok(M, T)  acro(M, Initials)
/// ```
pub const SIEVE_RULES: &str = r#"
# which labels a pronoun class can take
compat("he", "person").
compat("she", "person").
compat("they", "person").
compat("they", "organization").
compat("they", "team").
compat("it", "organization").
compat("it", "team").
compat("it", "location").
compat("it", "thing").
compat("it", "food").
compat("it", "activity").
# words that are never a head on their own
generic("inc"). generic("corp"). generic("corporation"). generic("ltd").
generic("llc"). generic("co"). generic("company"). generic("group").
generic("the"). generic("of"). generic("and"). generic("mr"). generic("mrs").
generic("ms"). generic("dr"). generic("jr"). generic("sr").

# ---- sieve 1: exact name match
exact(M, A) :- name(M), norm(M, T), norm(A, T), name(A), mention(M, _, P, L),
               mention(A, _, Q, L), Q < P.
# ---- sieve 2: acronym
acronym(M, A) :- name(M), norm(M, T), acro(A, T), name(A), mention(M, _, P, L),
                 mention(A, _, Q, L), Q < P.
# ---- sieve 3: head word (first or last word of a longer earlier name)
head(M, A) :- ntok(M, 1), norm(M, T), !generic(T), last_tok(A, T), ntok(A, N), N > 1,
              mention(M, _, P, L), mention(A, _, Q, L), Q < P.
head(M, A) :- ntok(M, 1), norm(M, T), !generic(T), first_tok(A, T), ntok(A, N), N > 1,
              mention(M, _, P, L), mention(A, _, Q, L), Q < P.

# a later sieve only sees mentions no earlier sieve resolved
has_exact(M) :- exact(M, _).
has_acronym(M) :- acronym(M, _).
cand(M, A, "exact") :- exact(M, A).
cand(M, A, "acronym") :- acronym(M, A), !has_exact(M).
cand(M, A, "head") :- head(M, A), !has_exact(M), !has_acronym(M).

# one entity among the candidates, then the nearest of them
name_keys(M, count(K)) :- cand(M, A, _), norm(A, K).
name_near(M, max(Q)) :- cand(M, A, _), mention(A, _, Q, _).
name_ante(M, A, S) :- name_near(M, Q), cand(M, A, S), mention(A, _, Q, _), name_keys(M, 1).
name_ambiguous(M, K) :- name_keys(M, N), N > 1, cand(M, A, _), norm(A, K).

# a name's cluster: follow antecedents back to an unresolved name
resolved(M) :- name_ante(M, _, _).
root(M, M) :- name(M), !resolved(M).
root(M, R) :- name_ante(M, A, _), root(A, R).

# ---- sieve 4: pronouns
pron_cand(X, A) :- pron(X, C), mention(X, S, P, _), compat(C, L), mention(A, T, Q, L),
                   Q < P, T >= S - 2, name(A).
pron_scope(X, max(T)) :- pron_cand(X, A), mention(A, T, _, _).
pron_in(X, A) :- pron_cand(X, A), mention(A, T, _, _), pron_scope(X, T).
pron_key(X, K) :- pron_in(X, A), root(A, R), norm(R, K).
pron_keys(X, count(K)) :- pron_key(X, K).
pron_near(X, max(Q)) :- pron_in(X, A), mention(A, _, Q, _).
pron_ante(X, A) :- pron_near(X, Q), pron_in(X, A), mention(A, _, Q, _), pron_keys(X, 1).
pron_ambiguous(X, K) :- pron_keys(X, N), N > 1, pron_key(X, K).
has_pron_cand(X) :- pron_cand(X, _).
pron_unresolved(X) :- pron(X, _), !has_pron_cand(X).

# ---- result: the cluster root each mention refers to, and the sieve
coref(M, R, S) :- name_ante(M, A, S), root(A, R).
coref(X, R, "pronoun") :- pron_ante(X, A), root(A, R).
"#;

/// Third-person pronouns the pronoun sieve resolves, with their class.
/// First-person ones are the speaker, linked elsewhere.
pub fn pronoun_class(text: &str) -> Option<&'static str> {
    match text.to_lowercase().as_str() {
        "he" | "him" | "his" | "himself" => Some("he"),
        "she" | "her" | "hers" | "herself" => Some("she"),
        "it" | "its" | "itself" => Some("it"),
        "they" | "them" | "their" | "theirs" | "themselves" => Some("they"),
        _ => None,
    }
}

/// A mention's place: (sentence index, byte start, byte end).
pub type Place = (usize, usize, usize);

/// What the sieves decided for one mention.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// It refers to `antecedent` (a name, the root of its cluster) in
    /// sentence `sentence`; `sieve` says which sieve decided, `why` is the
    /// engine's proof.
    Resolved { sentence: usize, antecedent: Mention, sieve: String, why: String },
    /// Its sieve's candidates are more than one entity (their names).
    Ambiguous(Vec<String>),
    /// A pronoun with no compatible earlier mention in range.
    NoAntecedent,
}

/// Name normalisation for the sieves: lower case, whitespace collapsed,
/// possessive and edge punctuation removed.
fn norm(text: &str) -> String {
    let t = super::link::NameLinker::canonical(text).unwrap_or_else(|| text.trim().to_string());
    t.to_lowercase()
}

fn words(text: &str) -> Vec<String> {
    norm(text)
        .split(|c: char| c.is_whitespace() || c == '-')
        .filter(|w| !w.is_empty())
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Initials of the capitalised words ("International Business Machines" ->
/// "ibm"), for names of two or more such words.
fn initials(text: &str) -> Option<String> {
    let caps: Vec<char> = text
        .split_whitespace()
        .filter_map(|w| w.chars().next())
        .filter(|c| c.is_uppercase())
        .collect();
    (caps.len() >= 2).then(|| caps.iter().flat_map(|c| c.to_lowercase()).collect())
}

/// Run the sieves over an episode's mentions, given in order as (sentence
/// index, mention). Returns an outcome for every pronoun, and for every
/// name a sieve resolved or found ambiguous; names absent from the map
/// stand for themselves.
pub fn resolve(mentions: &[(usize, Mention)]) -> HashMap<Place, Outcome> {
    let mut out = HashMap::new();
    // one entry per distinct place, first label wins
    let mut seen: HashMap<Place, usize> = HashMap::new();
    let mut ms: Vec<(usize, Mention)> = Vec::new();
    for (s, m) in mentions {
        let place = (*s, m.start, m.end);
        if !seen.contains_key(&place) && m.start < m.end {
            seen.insert(place, ms.len());
            ms.push((*s, m.clone()));
        }
    }
    let first_person = |t: &str| matches!(t.to_lowercase().as_str(), "i" | "me" | "my" | "mine" | "myself");
    // Only mentions some sieve could use go to the engine (its cost grows
    // with the facts), and none at all when no sieve can fire. A name takes
    // part if it shares a sieve key (its text, one of its end words, its
    // initials) with another name, or lies within a pronoun's window; a
    // pronoun always does. This keeps every candidate pair and every
    // pronoun window, so the rules decide exactly as on all mentions.
    let keys: Vec<Vec<String>> = ms
        .iter()
        .map(|(_, m)| {
            let mut k = vec![norm(&m.text)];
            let w = words(&m.text);
            if w.len() > 1 {
                k.extend(w.first().cloned());
                k.extend(w.last().cloned());
            }
            k.extend(initials(&m.text));
            k
        })
        .collect();
    let mut key_count: HashMap<&str, usize> = HashMap::new();
    for (i, (_, m)) in ms.iter().enumerate() {
        if pronoun_class(&m.text).is_none() && !first_person(&m.text) {
            let mut uniq: Vec<&str> = keys[i].iter().map(String::as_str).collect();
            uniq.sort();
            uniq.dedup();
            for k in uniq {
                *key_count.entry(k).or_default() += 1;
            }
        }
    }
    let pron_sentences: Vec<usize> =
        ms.iter().filter(|(_, m)| pronoun_class(&m.text).is_some()).map(|(s, _)| *s).collect();
    let takes_part = |i: usize| {
        let (s, m) = &ms[i];
        if first_person(&m.text) {
            return false;
        }
        if pronoun_class(&m.text).is_some() {
            return true;
        }
        keys[i].iter().any(|k| key_count.get(k.as_str()).copied().unwrap_or(0) > 1)
            || pron_sentences.iter().any(|p| *s <= *p && *s + 2 >= *p)
    };
    let part: Vec<usize> = (0..ms.len()).filter(|&i| takes_part(i)).collect();
    if part.is_empty() {
        return out;
    }
    let mut e: Engine<Ann> = Engine::new();
    if e.install_program(SIEVE_RULES).is_err() {
        return out; // the program is a constant; tested below
    }
    for &id in &part {
        let (s, m) = &ms[id];
        let idv = Value::Int(id as i64);
        let pos = Value::Int(*s as i64 * 1_000_000 + m.start as i64);
        let label = e.sym(&m.label.to_lowercase());
        e.declare("mention", &[idv.clone(), Value::Int(*s as i64), pos, label], Ann::unit());
        let n = norm(&m.text);
        let nv = e.sym(&n);
        e.declare("norm", &[idv.clone(), nv], Ann::unit());
        match pronoun_class(&m.text) {
            Some(c) => {
                let cv = e.sym(c);
                e.declare("pron", &[idv.clone(), cv], Ann::unit());
            }
            None => {
                e.declare("name", &[idv.clone()], Ann::unit());
                let w = words(&m.text);
                e.declare("ntok", &[idv.clone(), Value::Int(w.len() as i64)], Ann::unit());
                if let (Some(f), Some(l)) = (w.first(), w.last()) {
                    let (fv, lv) = (e.sym(f), e.sym(l));
                    e.declare("first_tok", &[idv.clone(), fv], Ann::unit());
                    e.declare("last_tok", &[idv.clone(), lv], Ann::unit());
                }
                if let Some(a) = initials(&m.text) {
                    let av = e.sym(&a);
                    e.declare("acro", &[idv.clone(), av], Ann::unit());
                }
            }
        }
    }
    e.run();
    let int = |v: &Value| v.as_int().map(|i| i as usize);
    let place = |id: usize| -> Place {
        let (s, m) = &ms[id];
        (*s, m.start, m.end)
    };
    for k in e.relation_keys("coref") {
        let (Some(m), Some(r)) = (int(&k[0]), int(&k[1])) else { continue };
        let sieve = e.interner.display(&k[2]).trim_matches('"').to_string();
        let why = e.why("coref", &k);
        let (rs, rm) = &ms[r];
        out.insert(place(m), Outcome::Resolved { sentence: *rs, antecedent: rm.clone(), sieve, why });
    }
    let mut ambiguous: HashMap<usize, Vec<String>> = HashMap::new();
    for pred in ["pron_ambiguous", "name_ambiguous"] {
        for k in e.relation_keys(pred) {
            if let Some(m) = int(&k[0]) {
                ambiguous.entry(m).or_default().push(e.interner.display(&k[1]).trim_matches('"').to_string());
            }
        }
    }
    for (m, mut names) in ambiguous {
        names.sort();
        names.dedup();
        out.entry(place(m)).or_insert(Outcome::Ambiguous(names));
    }
    for k in e.relation_keys("pron_unresolved") {
        if let Some(m) = int(&k[0]) {
            out.entry(place(m)).or_insert(Outcome::NoAntecedent);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(text: &str, label: &str, sentence: &str) -> Mention {
        let start = sentence.find(text).unwrap_or_else(|| panic!("{text} not in {sentence}"));
        Mention { text: text.into(), label: label.into(), start, end: start + text.len(), score: 0.9 }
    }

    /// (sentences, mentions as (sentence index, text, label))
    fn run(sentences: &[&str], mentions: &[(usize, &str, &str)]) -> HashMap<Place, Outcome> {
        let ms: Vec<(usize, Mention)> = mentions.iter().map(|(s, t, l)| (*s, m(t, l, sentences[*s]))).collect();
        resolve(&ms)
    }

    fn antecedent(o: &HashMap<Place, Outcome>, sentences: &[&str], s: usize, text: &str) -> Option<(String, String)> {
        let start = sentences[s].find(text)?;
        match o.get(&(s, start, start + text.len()))? {
            Outcome::Resolved { antecedent, sieve, .. } => Some((antecedent.text.clone(), sieve.clone())),
            _ => None,
        }
    }

    #[test]
    fn the_sieve_program_installs() {
        let mut e: Engine<Ann> = Engine::new();
        e.install_program(SIEVE_RULES).expect("sieve rules parse and stratify");
    }

    #[test]
    fn pronoun_takes_the_one_compatible_entity() {
        let s = ["Bob joined Acme in 2020.", "He works on the search team."];
        let o = run(&s, &[(0, "Bob", "person"), (0, "Acme", "organization"), (1, "He", "person")]);
        assert_eq!(antecedent(&o, &s, 1, "He"), Some(("Bob".into(), "pronoun".into())));
    }

    #[test]
    fn two_people_make_he_ambiguous() {
        let s = ["Bob met Dana.", "He works at Acme."];
        let o = run(&s, &[(0, "Bob", "person"), (0, "Dana", "person"), (1, "He", "person"), (1, "Acme", "organization")]);
        let start = s[1].find("He").unwrap();
        assert_eq!(o.get(&(1, start, start + 2)), Some(&Outcome::Ambiguous(vec!["bob".into(), "dana".into()])));
    }

    #[test]
    fn it_takes_an_organisation_not_a_person() {
        let s = ["Dana founded Acme.", "It is based in Austin."];
        let o = run(&s, &[(0, "Dana", "person"), (0, "Acme", "organization"), (1, "It", "organization"), (1, "Austin", "location")]);
        assert_eq!(antecedent(&o, &s, 1, "It"), Some(("Acme".into(), "pronoun".into())));
    }

    #[test]
    fn nearest_sentence_with_a_candidate_wins() {
        // Bob is two sentences back, Priya one: "she" -> Priya, never Bob
        let s = ["Bob runs Initech.", "Priya joined last week.", "She works at Acme."];
        let o = run(&s, &[(0, "Bob", "person"), (0, "Initech", "organization"), (1, "Priya", "person"), (2, "She", "person"), (2, "Acme", "organization")]);
        assert_eq!(antecedent(&o, &s, 2, "She"), Some(("Priya".into(), "pronoun".into())));
    }

    #[test]
    fn no_candidate_in_range_is_no_antecedent() {
        let s = ["Bob runs Initech.", "Nothing happened.", "Still nothing.", "Then nothing.", "He left."];
        let o = run(&s, &[(0, "Bob", "person"), (0, "Initech", "organization"), (4, "He", "person")]);
        let start = s[4].find("He").unwrap();
        assert_eq!(o.get(&(4, start, start + 2)), Some(&Outcome::NoAntecedent));
    }

    #[test]
    fn head_and_exact_names_join_one_cluster_and_pronouns_follow_it() {
        let s = ["Dana Smith joined Acme Corp.", "Smith reports to Bob.", "Acme hired her in May."];
        let o = run(
            &s,
            &[
                (0, "Dana Smith", "person"),
                (0, "Acme Corp", "organization"),
                (1, "Smith", "person"),
                (1, "Bob", "person"),
                (2, "Acme", "organization"),
                (2, "her", "person"),
            ],
        );
        assert_eq!(antecedent(&o, &s, 1, "Smith"), Some(("Dana Smith".into(), "head".into())));
        assert_eq!(antecedent(&o, &s, 2, "Acme"), Some(("Acme Corp".into(), "head".into())));
        // "her": in range, Smith (= Dana Smith) and Bob are two entities
        let start = s[2].find("her").unwrap();
        assert!(matches!(o.get(&(2, start, start + 3)), Some(Outcome::Ambiguous(_))));
    }

    #[test]
    fn a_cluster_counts_as_one_entity_for_pronouns() {
        let s = ["Dana Smith joined Acme.", "Smith leads search.", "She reports to the CTO."];
        let o = run(&s, &[(0, "Dana Smith", "person"), (0, "Acme", "organization"), (1, "Smith", "person"), (2, "She", "person")]);
        assert_eq!(antecedent(&o, &s, 2, "She"), Some(("Dana Smith".into(), "pronoun".into())));
    }

    #[test]
    fn acronym_and_ambiguous_heads() {
        let s = ["Dana joined International Business Machines.", "IBM is in Armonk.", "Dana Smith and John Smith met.", "Smith left."];
        let o = run(
            &s,
            &[
                (0, "Dana", "person"),
                (0, "International Business Machines", "organization"),
                (1, "IBM", "organization"),
                (1, "Armonk", "location"),
                (2, "Dana Smith", "person"),
                (2, "John Smith", "person"),
                (3, "Smith", "person"),
            ],
        );
        assert_eq!(antecedent(&o, &s, 1, "IBM"), Some(("International Business Machines".into(), "acronym".into())));
        let start = s[3].find("Smith").unwrap();
        assert_eq!(o.get(&(3, start, start + 5)), Some(&Outcome::Ambiguous(vec!["dana smith".into(), "john smith".into()])));
    }

    #[test]
    fn exact_beats_head_and_generic_words_never_head() {
        let s = ["Acme Corp hired Bob.", "Corp is a word.", "Acme Labs opened.", "Acme grew."];
        let o = run(
            &s,
            &[(0, "Acme Corp", "organization"), (0, "Bob", "person"), (1, "Corp", "organization"), (2, "Acme Labs", "organization"), (3, "Acme", "organization")],
        );
        assert_eq!(antecedent(&o, &s, 1, "Corp"), None, "generic word");
        // "Acme" heads both Acme Corp and Acme Labs: two entities
        let start = s[3].find("Acme").unwrap();
        assert!(matches!(o.get(&(3, start, start + 4)), Some(Outcome::Ambiguous(_))));
    }

    #[test]
    fn antecedents_precede_and_first_person_is_left_to_the_speaker() {
        let s = ["He said Bob works at Acme.", "I agree."];
        let o = run(&s, &[(0, "He", "person"), (0, "Bob", "person"), (0, "Acme", "organization"), (1, "I", "person")]);
        let start = s[0].find("He").unwrap();
        assert_eq!(o.get(&(0, start, start + 2)), Some(&Outcome::NoAntecedent), "no cataphora");
        assert!(!o.contains_key(&(1, 0, 1)));
    }

    #[test]
    fn resolutions_carry_a_proof() {
        let s = ["Bob joined Acme.", "He leads search."];
        let o = run(&s, &[(0, "Bob", "person"), (0, "Acme", "organization"), (1, "He", "person")]);
        let Some(Outcome::Resolved { why, .. }) = o.get(&(1, 0, 2)) else { panic!("{o:?}") };
        assert!(why.contains("coref") && why.contains("pron_ante"), "{why}");
    }
}

