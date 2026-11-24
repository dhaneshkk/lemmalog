//! `GlinerExtractor`: extraction without an LLM.
//!
//! ```text
//! episode text ── sentences ──▶ RelationBackend (gline-rs: NER + relations)
//!                                  │ mentions + raw relations, byte spans
//!                                  ▼
//!                               NegEx ── negated? ──▶ rejected (Negated)
//!                                  ▼
//!                               EntityLinker (GLinker) ── unlinked? ──▶ rejected
//!                                  ▼
//!                               CandidateFact(subj_id, pred, obj_id, confidence)
//! ```
//!
//! The backend and the linker are traits, so the gate's tests run with
//! [`FixtureBackend`] and [`TableLinker`](super::link::TableLinker) and need no
//! ONNX model or sidecar. The real backend is
//! [`GlineBackend`](super::gline::GlineBackend) (feature `gliner`).

use super::link::{EntityLinker, Link};
use super::negex::NegEx;
use crate::agent::{entity_token_problem, CandidateFact, Episode, Extractor};
use std::collections::{BTreeMap, HashMap};

/// An entity mention: text, GLiNER label and byte span in its sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct Mention {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

/// A relation as the model reports it, before negation and linking.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRelation {
    pub subject: Mention,
    /// The model's relation label, e.g. "works at".
    pub label: String,
    pub object: Mention,
    pub score: f32,
}

/// What the backend found in one sentence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SentenceAnalysis {
    pub mentions: Vec<Mention>,
    pub relations: Vec<RawRelation>,
    /// Relations the model reported whose subject or object is not one of
    /// this sentence's mentions: (subject text, label, object text).
    /// Reported as [`RejectReason::Unresolved`], never dropped silently.
    pub unresolved: Vec<(String, String, String)>,
}

/// A relation as a model reports it before it is tied to mention spans:
/// GLiNER multitask gives the subject as text only, and the object as text
/// plus a span that may be relative to the model's prompt, not the sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRelation {
    pub subject: String,
    pub label: String,
    pub object: String,
    /// The object's byte span in the sentence, if known. Ignored unless it
    /// lies in the sentence and covers exactly `object`.
    pub object_span: Option<(usize, usize)>,
    pub score: f32,
}

/// Tie model relations to this sentence's mentions. The object is the
/// mention at `object_span` when that span is valid, otherwise the mention
/// with the object's text nearest the subject; the subject is the mention
/// with the subject's text nearest the object (never the object itself).
/// A relation with no matching subject or object mention is unresolved.
pub fn resolve_relations(sentence: &str, mentions: Vec<Mention>, rels: &[ModelRelation]) -> SentenceAnalysis {
    let mut out = SentenceAnalysis { mentions, ..Default::default() };
    let dist = |a: &Mention, b: &Mention| (a.start as i64 - b.start as i64).abs();
    for r in rels {
        let hinted = r
            .object_span
            .filter(|&(a, b)| sentence.get(a..b) == Some(r.object.as_str()))
            .and_then(|(a, b)| out.mentions.iter().find(|m| m.start == a && m.end == b));
        let objects: Vec<&Mention> = match hinted {
            Some(m) => vec![m],
            None => out.mentions.iter().filter(|m| m.text == r.object).collect(),
        };
        let best = objects
            .iter()
            .flat_map(|o| {
                out.mentions
                    .iter()
                    .filter(|s| s.text == r.subject && (s.start, s.end) != (o.start, o.end))
                    .map(move |s| (dist(s, o), s.start, o.start, s, *o))
            })
            .min_by_key(|(d, ss, os, _, _)| (*d, *ss, *os));
        match best {
            Some((_, _, _, s, o)) => out.relations.push(RawRelation {
                subject: s.clone(),
                label: r.label.clone(),
                object: o.clone(),
                score: r.score,
            }),
            None => out.unresolved.push((r.subject.clone(), r.label.clone(), r.object.clone())),
        }
    }
    out
}

/// NER + relation extraction over a batch of sentences.
pub trait RelationBackend {
    /// One analysis per input sentence, same order.
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String>;
}

impl<B: RelationBackend + ?Sized> RelationBackend for &mut B {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        (**self).analyse(sentences)
    }
}

/// Remembers each sentence's analysis, so re-running extraction over the
/// same text (a threshold sweep, a second linker) calls the model once.
pub struct CachedBackend<B: RelationBackend> {
    pub inner: B,
    cache: HashMap<String, SentenceAnalysis>,
    /// Insertion order, for evicting the oldest entries.
    order: std::collections::VecDeque<String>,
    /// At most this many sentences stay cached (oldest evicted first).
    /// Default [`CachedBackend::DEFAULT_CAPACITY`].
    pub capacity: usize,
    /// Sentences sent to `inner`.
    pub misses: usize,
}

impl<B: RelationBackend> CachedBackend<B> {
    pub const DEFAULT_CAPACITY: usize = 100_000;

    pub fn new(inner: B) -> Self {
        CachedBackend { inner, cache: HashMap::new(), order: Default::default(), capacity: Self::DEFAULT_CAPACITY, misses: 0 }
    }

    /// Forget every cached analysis (after changing how `inner` analyses).
    pub fn clear(&mut self) {
        self.cache.clear();
        self.order.clear();
    }

    /// Sentences currently cached.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// The cached model output for a sentence, if it was analysed.
    pub fn analysis(&self, sentence: &str) -> Option<&SentenceAnalysis> {
        self.cache.get(sentence)
    }
}

impl<B: RelationBackend> RelationBackend for CachedBackend<B> {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        let mut todo: Vec<String> = Vec::new();
        let mut queued: std::collections::HashSet<&str> = Default::default();
        for s in sentences {
            if !self.cache.contains_key(s) && queued.insert(s.as_str()) {
                todo.push(s.clone());
            }
        }
        // answer from this call's results, so eviction below cannot drop a
        // sentence this call still needs
        // in sentence order, so eviction order is deterministic
        let mut fresh: Vec<(String, SentenceAnalysis)> = Vec::new();
        if !todo.is_empty() {
            let got = self.inner.analyse(&todo)?;
            if got.len() != todo.len() {
                return Err("backend returned the wrong number of analyses".into());
            }
            self.misses += todo.len();
            fresh = todo.into_iter().zip(got).collect();
        }
        let out = sentences
            .iter()
            .map(|s| fresh.iter().find(|(f, _)| f == s).map(|(_, a)| a).or_else(|| self.cache.get(s)).cloned().unwrap())
            .collect();
        for (s, a) in fresh {
            self.order.push_back(s.clone());
            self.cache.insert(s, a);
        }
        while self.cache.len() > self.capacity.max(1) {
            match self.order.pop_front() {
                Some(old) => {
                    self.cache.remove(&old);
                }
                None => break,
            }
        }
        Ok(out)
    }
}

/// Why a raw relation did not become a fact.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectReason {
    Negated,
    Unlinked(String),
    BadEntity(String),
    LowScore,
    UnknownRelation,
    /// The model named a subject or object that is not a mention of the
    /// sentence it was found in.
    Unresolved,
    /// A mention span is not a valid byte range of its sentence.
    BadSpan,
    /// The same sentence gave this fact in the other direction with stronger
    /// evidence (see SPEC-7.1-O4).
    Reversed,
    /// Subject and object linked to the same entity ("Bob reports to
    /// Robert" with both names linking to `person:bob`).
    SelfLoop,
    /// The relation is only possible, planned or conditional ("Bob might
    /// join Acme", "If Bob joins Acme"), not asserted (see SPEC-7.1-O6).
    Hedged,
    /// A pronoun the coreference sieves could not resolve: more than one
    /// candidate entity, or none in range (see SPEC-7.1-O7).
    Coreference(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rejection {
    pub episode: String,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub reason: RejectReason,
}

/// Map from the model's relation labels to lemmalog predicates, matching
/// the vocabulary of [`EXTRACTION_PROMPT`](crate::agent::EXTRACTION_PROMPT).
/// Each entry: (GLiNER label, predicate, subject labels, object labels,
/// inverse). `inverse` swaps subject and object: lemmalog reads
/// `X --manager--> Y` as "X reports to Y" (`reports_to(X,Y) :-
/// edge(X,"manager",Y,..)`), so "Dana is the manager of Bob" becomes
/// `Bob --manager--> Dana`.
pub const DEFAULT_RELATIONS: &[(&str, &str, &[&str], &[&str], bool)] = &[
    ("works at", "works_at", &["person"], &["organization"], false),
    ("reports to", "manager", &["person"], &["person"], false),
    ("manager of", "manager", &["person"], &["person"], true),
    ("likes", "likes", &["person"], &["thing", "food", "activity", "organization"], false),
    ("job title", "job_title", &["person"], &["job title"], false),
    ("member of", "member_of", &["person"], &["organization", "team"], false),
    ("located in", "located_in", &["person", "organization"], &["location"], false),
];

/// The GLiNER entity labels the default relation set needs.
pub fn default_entity_labels() -> Vec<String> {
    let mut v: Vec<String> = DEFAULT_RELATIONS
        .iter()
        .flat_map(|(_, _, s, o, _)| s.iter().chain(o.iter()))
        .map(|s| s.to_string())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Words that end in "." without ending a sentence when the next word is
/// capitalised (titles, initials-style abbreviations).
/// "st" is a title ("St. Louis") unless a capitalised word precedes it
/// ("Main St."), when it is the street abbreviation; "jr"/"sr" follow a name,
/// so they end a sentence when a capitalised word follows ("Bob Smith Jr.
/// Dana ..."), like the abbreviations below.
const TITLES: &[&str] = &["dr", "mr", "mrs", "ms", "prof", "rev", "gen", "col", "capt", "lt", "sgt", "hon"];
/// Abbreviations that end a sentence only if the next word is capitalised.
const ABBREVS: &[&str] = &[
    "inc", "corp", "ltd", "co", "llc", "plc", "bros", "dept", "univ", "vs", "etc", "e.g", "i.e",
    "approx", "no", "vol", "fig", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "u.s", "u.k", "a.m", "p.m",
    "jr", "sr", "ave", "rd", "blvd",
];

/// Split into sentences on `.`, `!`, `?` and newlines, keeping the
/// terminator with its sentence. Empty pieces are dropped. A run of
/// terminators ("...", "?!") is one boundary. A "." does not split inside a
/// number ("2.5"), after a title ("Dr. Smith"), or after an abbreviation or
/// single letter ("Acme Corp. in Austin", "the U.S. last year") unless the
/// next word starts with a capital letter. An ellipsis followed by a
/// lower-case word does not split either.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let push = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        cur.clear();
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            push(&mut cur, &mut out);
            i += 1;
            continue;
        }
        if !matches!(c, '.' | '!' | '?') {
            cur.push(c);
            i += 1;
            continue;
        }
        // the whole run of terminators
        let mut j = i;
        while j < chars.len() && matches!(chars[j], '.' | '!' | '?') {
            cur.push(chars[j]);
            j += 1;
        }
        let run = j - i;
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(j).copied();
        let next_word = chars[j..].iter().find(|c| !c.is_whitespace()).copied();
        let next_upper = next_word.map_or(true, |c| c.is_uppercase() || !c.is_alphabetic());
        // the word just before the terminator, lower-cased, keeping inner dots
        let word: String = {
            let w: String = cur[..cur.len() - run]
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '.')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            w.to_lowercase()
        };
        let only_dots = chars[i..j].iter().all(|c| *c == '.');
        // "Main St." (street) vs "St. Louis" (saint): the word before "St"
        let street = word == "st" && {
            let before = &cur[..cur.len() - run - word.len()];
            before.trim_end().rsplit(char::is_whitespace).next().and_then(|w| w.chars().next()).is_some_and(char::is_uppercase)
        };
        let keep = if run == 1 && c == '.' {
            let in_number = prev.map_or(false, |p| p.is_ascii_digit()) && next.map_or(false, |n| n.is_ascii_digit());
            let single_letter = word.len() == 1 && word.chars().all(char::is_alphabetic)
                || word.split('.').all(|p| p.chars().count() == 1 && p.chars().all(char::is_alphabetic));
            in_number
                || next.map_or(false, |n| !n.is_whitespace()) && next.map_or(false, |n| n.is_alphanumeric())
                || TITLES.contains(&word.as_str())
                || word == "st" && !street
                || (ABBREVS.contains(&word.as_str()) || single_letter || street) && !next_upper
        } else {
            // "..." before a lower-case word continues the sentence
            only_dots && run > 1 && !next_upper
        };
        if !keep {
            push(&mut cur, &mut out);
        }
        i = j;
    }
    push(&mut cur, &mut out);
    out
}

/// Is the relation's label written between its subject and its object, in
/// that order ("Sam [reports to] Priya", "Priya is the [manager of] Sam")?
/// This is the sentence's own evidence for the relation's direction; the
/// same label read the other way round ("Priya reports to Sam") has none.
pub fn worded_between(sentence: &str, r: &RawRelation) -> bool {
    r.subject.end <= r.object.start
        && sentence
            .get(r.subject.end..r.object.start)
            .map_or(false, |between| between.to_lowercase().contains(&r.label.to_lowercase()))
}

/// Drop a leading title ("Dr. Smith" -> "Smith", "Mrs Lee" -> "Lee") from
/// a mention before linking, keeping the span on the remaining name. A
/// mention that is only a title is returned unchanged.
pub fn strip_title(sentence: &str, m: &Mention) -> Mention {
    let text = sentence.get(m.start..m.end).unwrap_or(&m.text);
    let word: String = text.chars().take_while(|c| c.is_alphabetic()).collect();
    if word.is_empty() || !TITLES.contains(&word.to_lowercase().as_str()) {
        return m.clone();
    }
    let rest = &text[word.len()..];
    let rest_trim = rest.trim_start_matches('.').trim_start();
    if rest_trim.is_empty() || rest_trim.len() == rest.len() {
        // nothing after the title, or no separator ("Drake"): not a title
        return m.clone();
    }
    let cut = text.len() - rest_trim.len();
    Mention { text: rest_trim.to_string(), start: m.start + cut, ..m.clone() }
}

pub struct GlinerExtractor<B: RelationBackend, L: EntityLinker> {
    backend: B,
    linker: L,
    pub negex: NegEx,
    relations: HashMap<String, (String, bool)>,
    /// Relations scoring below this are dropped.
    pub min_score: f32,
    /// Second coreference tier (SPEC-7.1-O9): scores the pronouns the
    /// sieves abstain on; a resolution needs [`GlinerExtractor::coref_gate`].
    pub pair_scorer: Option<Box<dyn super::coref::PairScorer>>,
    pub coref_gate: super::coref::Gate,
    /// Emit each coref-resolved mention as a mention node with its own
    /// facts, plus `coref` and `mention_of` facts that tie it to its
    /// antecedent, instead of resolving it to the antecedent's id here
    /// (SPEC-7.1-O8): memory then derives identity by a rule, and a link
    /// can be retracted. Off by default.
    pub coref_facts: bool,
    /// episode id -> (fingerprint of text + speaker, facts)
    seen: HashMap<String, (u64, Vec<CandidateFact>)>,
    seen_order: std::collections::VecDeque<String>,
    /// At most this many episodes stay memoized (oldest evicted first).
    /// Default [`GlinerExtractor::DEFAULT_MEMO_CAPACITY`].
    pub memo_capacity: usize,
    /// `rejected()` and `errors()` keep at most this many of the most recent
    /// entries. Default [`GlinerExtractor::DEFAULT_LOG_CAPACITY`].
    pub log_capacity: usize,
    rejected: Vec<Rejection>,
    calls: usize,
    failures: usize,
    errors: Vec<(String, String)>,
}

impl<B: RelationBackend, L: EntityLinker> GlinerExtractor<B, L> {
    pub const DEFAULT_MEMO_CAPACITY: usize = 10_000;
    pub const DEFAULT_LOG_CAPACITY: usize = 10_000;

    pub fn new(backend: B, linker: L) -> Self {
        GlinerExtractor {
            backend,
            linker,
            negex: NegEx::default(),
            relations: DEFAULT_RELATIONS
                .iter()
                .map(|(l, p, _, _, inv)| (l.to_string(), (p.to_string(), *inv)))
                .collect(),
            min_score: 0.5,
            pair_scorer: None,
            coref_gate: super::coref::Gate::default(),
            coref_facts: false,
            seen: HashMap::new(),
            seen_order: Default::default(),
            memo_capacity: Self::DEFAULT_MEMO_CAPACITY,
            log_capacity: Self::DEFAULT_LOG_CAPACITY,
            rejected: Vec::new(),
            calls: 0,
            failures: 0,
            errors: Vec::new(),
        }
    }

    /// Relations the extractor saw but did not turn into facts.
    pub fn rejected(&self) -> &[Rejection] {
        &self.rejected
    }

    /// Backend failures, as (episode id, error message): each one is an
    /// episode that yielded no facts (a degraded turn).
    pub fn errors(&self) -> &[(String, String)] {
        &self.errors
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Use `scorer` as the second coreference tier, trusted through `gate`.
    pub fn with_pair_scorer(mut self, scorer: Box<dyn super::coref::PairScorer>, gate: super::coref::Gate) -> Self {
        self.pair_scorer = Some(scorer);
        self.coref_gate = gate;
        self
    }

    /// Emit coref links as facts (see [`GlinerExtractor::coref_facts`]).
    pub fn with_coref_facts(mut self, on: bool) -> Self {
        self.coref_facts = on;
        self
    }

    /// Keep only the newest `log_capacity` rejections and errors. Trims in
    /// batches (down to the capacity once a quarter over it), so the cost
    /// per entry stays constant.
    fn trim_logs(&mut self) {
        let cap = self.log_capacity.max(1);
        let slack = cap / 4;
        if self.rejected.len() > cap + slack {
            self.rejected.drain(..self.rejected.len() - cap);
        }
        if self.errors.len() > cap + slack {
            self.errors.drain(..self.errors.len() - cap);
        }
    }

    fn reject(&mut self, ep: &Episode, r: &RawRelation, reason: RejectReason) {
        self.rejected.push(Rejection {
            episode: ep.id.clone(),
            subject: r.subject.text.clone(),
            relation: r.label.clone(),
            object: r.object.text.clone(),
            reason,
        });
    }

    /// Run the coreference sieves over the episode's mentions: each
    /// sentence's mentions plus its relations' subjects and objects, with
    /// valid spans.
    fn coreference(
        &mut self,
        sentences: &[String],
        analyses: &[Option<SentenceAnalysis>],
    ) -> HashMap<super::coref::Place, super::coref::Outcome> {
        let mut ms: Vec<(usize, Mention)> = Vec::new();
        for (si, a) in analyses.iter().enumerate() {
            let Some(a) = a else { continue };
            let sentence = &sentences[si];
            let ok = |m: &Mention| m.start < m.end && sentence.get(m.start..m.end).is_some();
            for m in a.mentions.iter().chain(a.relations.iter().flat_map(|r| [&r.subject, &r.object])) {
                if ok(m) {
                    ms.push((si, Mention { text: sentence[m.start..m.end].to_string(), ..m.clone() }));
                }
            }
        }
        ms.sort_by_key(|(s, m)| (*s, m.start, m.end));
        let gate = self.coref_gate;
        super::coref::resolve_tiered(sentences, &ms, self.pair_scorer.as_deref_mut(), gate)
    }

    /// Is this mention a first-person pronoun ("I", "me", "my", "myself")?
    fn first_person(m: &Mention) -> bool {
        matches!(m.text.to_lowercase().as_str(), "i" | "me" | "my" | "myself")
    }

    /// Link the episode's speaker by name, once per episode: the name with
    /// no sentence context, so a span-scoring linker scores "Dana", not the
    /// word "I" it stands for.
    fn speaker_link(&mut self, ep: &Episode, cache: &mut Option<Link>) -> Link {
        if let Some(l) = cache {
            return l.clone();
        }
        let l = match &ep.speaker {
            Some(name) => {
                let m = Mention { text: name.clone(), label: "person".into(), start: 0, end: name.len(), score: 1.0 };
                self.linker.link(name, &[m]).into_iter().next().unwrap_or(Link::Unlinked)
            }
            None => Link::Unlinked,
        };
        *cache = Some(l.clone());
        l
    }

    fn run(&mut self, ep: &Episode) -> Vec<CandidateFact> {
        let sentences = split_sentences(&ep.text);
        if sentences.is_empty() {
            return Vec::new();
        }
        let reply = |r: Result<Vec<SentenceAnalysis>, String>, n: usize| match r {
            Ok(a) if a.len() == n => Ok(a),
            Ok(a) => Err(format!("backend returned {} analyses for {} sentences", a.len(), n)),
            Err(e) => Err(e),
        };
        let analyses: Vec<Option<SentenceAnalysis>> = match reply(self.backend.analyse(&sentences), sentences.len()) {
            Ok(a) => a.into_iter().map(Some).collect(),
            Err(e) => {
                // one failure per episode, however many sentences failed
                self.failures += 1;
                self.errors.push((ep.id.clone(), e));
                if sentences.len() == 1 {
                    self.trim_logs();
                    return Vec::new(); // degraded turn: no facts, no poison
                }
                // a bad sentence must not cost the episode its other
                // sentences: retry them one at a time
                let mut out = Vec::with_capacity(sentences.len());
                for (i, s) in sentences.iter().enumerate() {
                    match reply(self.backend.analyse(std::slice::from_ref(s)), 1) {
                        Ok(mut a) => out.push(a.pop()),
                        Err(e) => {
                            self.errors.push((ep.id.clone(), format!("sentence {}: {e}", i + 1)));
                            out.push(None);
                        }
                    }
                }
                out
            }
        };
        // (subj, pred, obj) -> best confidence; BTreeMap keeps output order stable
        let mut facts: BTreeMap<(String, String, String), f64> = BTreeMap::new();
        let mut speaker: Option<Link> = None;
        // coreference over the whole episode (SPEC-7.1-O7): every mention
        // a sieve resolves, or a pronoun it could not
        let coref = self.coreference(&sentences, &analyses);
        // antecedents linked once each, in their own sentence
        let mut antecedent_links: HashMap<super::coref::Place, Link> = HashMap::new();
        // coref-resolved mentions, for coref_facts: place -> node
        let mut nodes: HashMap<super::coref::Place, MentionNode> = HashMap::new();
        for (si, (sentence, a)) in sentences.iter().zip(analyses).enumerate() {
            let Some(a) = a else { continue };
            for (subject, relation, object) in &a.unresolved {
                self.rejected.push(Rejection {
                    episode: ep.id.clone(),
                    subject: subject.clone(),
                    relation: relation.clone(),
                    object: object.clone(),
                    reason: RejectReason::Unresolved,
                });
            }
            // a non-empty byte range of this sentence (`get` checks bounds
            // and char boundaries, and never panics)
            let valid = |m: &Mention| m.start < m.end && sentence.get(m.start..m.end).is_some();
            // NegEx reads the span and the linker reads the text: make them
            // the same, so negation and linking judge the same words
            let at_span = |m: &Mention| Mention { text: sentence[m.start..m.end].to_string(), ..m.clone() };
            // keep the relations that survive the cheap checks, then link
            // their mentions in one linker call per sentence
            let mut kept: Vec<(RawRelation, String, bool)> = Vec::new();
            for r in &a.relations {
                let Some((pred, inverse)) = self.relations.get(&r.label).cloned() else {
                    self.reject(ep, r, RejectReason::UnknownRelation);
                    continue;
                };
                if !valid(&r.subject) || !valid(&r.object) {
                    self.reject(ep, r, RejectReason::BadSpan);
                    continue;
                }
                // written so that a NaN score fails the check: `NaN < x` is
                // false, so `r.score < self.min_score` would let NaN through
                if !(r.score >= self.min_score) || !r.score.is_finite() {
                    self.reject(ep, r, RejectReason::LowScore);
                    continue;
                }
                let r = &RawRelation { subject: at_span(&r.subject), object: at_span(&r.object), ..r.clone() };
                let span = |m: &Mention| (m.start, m.end);
                let others: Vec<(usize, usize)> = a
                    .mentions
                    .iter()
                    .filter(|m| valid(m))
                    .map(span)
                    .filter(|s| *s != span(&r.subject) && *s != span(&r.object))
                    .collect();
                if self.negex.negated_with(sentence, span(&r.object), r.subject.start, &others)
                    || (pred == "likes" && super::negex::dislikes_between(sentence, span(&r.subject), span(&r.object)))
                {
                    self.reject(ep, r, RejectReason::Negated);
                    continue;
                }
                if self.negex.hedged_with(sentence, span(&r.object), r.subject.start, &others) {
                    self.reject(ep, r, RejectReason::Hedged);
                    continue;
                }
                kept.push((r.clone(), pred, inverse));
            }
            if kept.is_empty() {
                continue;
            }
            // first-person mentions link as the speaker (by name); the rest go
            // to the linker in one call, titles stripped ("Dr. Smith" -> "Smith")
            let all: Vec<&Mention> = kept.iter().flat_map(|(r, _, _)| [&r.subject, &r.object]).collect();
            let mut links: Vec<Link> = vec![Link::Unlinked; all.len()];
            let mut ask: Vec<(usize, Mention)> = Vec::new();
            for (i, m) in all.iter().enumerate() {
                if Self::first_person(m) {
                    links[i] = self.speaker_link(ep, &mut speaker);
                    continue;
                }
                match coref.get(&(si, m.start, m.end)) {
                    // linked as the name it refers to, in that name's sentence
                    Some(super::coref::Outcome::Resolved { sentence: rs, antecedent, conf, .. }) => {
                        let place = (*rs, antecedent.start, antecedent.end);
                        let l = match antecedent_links.get(&place) {
                            Some(l) => l.clone(),
                            None => {
                                let rsent = &sentences[*rs];
                                let l = self.linker.link(rsent, &[strip_title(rsent, antecedent)]).into_iter().next().unwrap_or(Link::Unlinked);
                                antecedent_links.insert(place, l.clone());
                                l
                            }
                        };
                        if let Link::Canonical { id, score } = &l {
                            nodes.insert(
                                (si, m.start, m.end),
                                MentionNode {
                                    id: mention_id(&ep.id, si, m.start, m.end),
                                    antecedent: mention_id(&ep.id, *rs, antecedent.start, antecedent.end),
                                    conf: *conf,
                                    entity: id.clone(),
                                    entity_score: *score,
                                },
                            );
                        }
                        links[i] = l;
                    }
                    // a pronoun the sieves could not resolve stays unlinked
                    // (reported below); an ambiguous name stands for itself
                    Some(_) if super::coref::pronoun_class(&m.text).is_some() => {}
                    _ => ask.push((i, strip_title(sentence, m))),
                }
            }
            if !ask.is_empty() {
                let ms: Vec<Mention> = ask.iter().map(|(_, m)| m.clone()).collect();
                for ((i, _), l) in ask.iter().zip(self.linker.link(sentence, &ms)) {
                    links[*i] = l;
                }
            }
            // this sentence's facts: triple -> (cue, relation score, best
            // confidence, relation)
            let mut found: BTreeMap<(String, String, String), (bool, f32, f64, RawRelation)> = BTreeMap::new();
            for (i, (r, pred, inverse)) in kept.iter().enumerate() {
                let unresolved = [&r.subject, &r.object].into_iter().find_map(|m| match coref.get(&(si, m.start, m.end)) {
                    Some(super::coref::Outcome::Ambiguous(names)) if super::coref::pronoun_class(&m.text).is_some() => {
                        Some(format!("'{}' could refer to {}", m.text, names.join(" or ")))
                    }
                    Some(super::coref::Outcome::NoAntecedent) => Some(format!("'{}' has no earlier mention to refer to", m.text)),
                    _ => None,
                });
                if let Some(why) = unresolved {
                    self.reject(ep, r, RejectReason::Coreference(why));
                    continue;
                }
                let (s, o) = (links.get(2 * i), links.get(2 * i + 1));
                let (Some(Link::Canonical { id: sid, score: ss }), Some(Link::Canonical { id: oid, score: os })) = (s, o) else {
                    let which = match s {
                        Some(Link::Canonical { .. }) => r.object.text.clone(),
                        _ => r.subject.text.clone(),
                    };
                    self.reject(ep, r, RejectReason::Unlinked(which));
                    continue;
                };
                if let Some(p) = entity_token_problem(sid).or_else(|| entity_token_problem(oid)) {
                    self.reject(ep, r, RejectReason::BadEntity(p));
                    continue;
                }
                // `f32::min` ignores a NaN argument, so a NaN link score would
                // silently drop out of the confidence: treat it as no link
                if !ss.is_finite() || !os.is_finite() {
                    self.reject(ep, r, RejectReason::LowScore);
                    continue;
                }
                if sid == oid {
                    self.reject(ep, r, RejectReason::SelfLoop);
                    continue;
                }
                let (sid, oid) = if *inverse { (oid, sid) } else { (sid, oid) };
                let conf = r.score.min(*ss).min(*os) as f64;
                let cue = worded_between(sentence, r);
                let key = (sid.clone(), pred.clone(), oid.clone());
                match found.get_mut(&key) {
                    Some(e) => {
                        e.2 = e.2.max(conf);
                        // scores are finite here (checked above); total_cmp
                        // keeps the order total, as the Lean model assumes
                        if cue.cmp(&e.0).then(r.score.total_cmp(&e.1)).is_gt() {
                            (e.0, e.1, e.3) = (cue, r.score, r.clone());
                        }
                    }
                    None => {
                        found.insert(key, (cue, r.score, conf, r.clone()));
                    }
                }
            }
            // one sentence never asserts a fact in both directions: keep the
            // one worded in the sentence, then the higher relation score, then
            // the smaller triple (a strict order, so exactly one survives).
            // Link scores are not compared: both directions link the same two
            // mentions, so they say nothing about direction.
            let losers: Vec<(String, String, String)> = found
                .iter()
                .filter(|((a, p, b), (c, f, _, _))| {
                    a != b
                        && found.get(&(b.clone(), p.clone(), a.clone())).map_or(false, |(c2, f2, _, _)| {
                            match c.cmp(c2).then(f.total_cmp(f2)) {
                                std::cmp::Ordering::Less => true,
                                std::cmp::Ordering::Equal => (a, b) > (b, a),
                                std::cmp::Ordering::Greater => false,
                            }
                        })
                })
                .map(|(k, _)| k.clone())
                .collect();
            for k in &losers {
                let (_, _, _, r) = found.remove(k).unwrap();
                self.reject(ep, &r, RejectReason::Reversed);
            }
            for ((subj, pred, obj), (_, _, conf, r)) in found {
                let (subj, obj) = if self.coref_facts {
                    // a coref-resolved side becomes its mention node, tied to
                    // the antecedent by `coref`, the antecedent to its entity
                    // by `mention_of`; memory derives the identity
                    let (sm, om) = if self.relations.get(&r.label).is_some_and(|(_, inv)| *inv) {
                        (&r.object, &r.subject)
                    } else {
                        (&r.subject, &r.object)
                    };
                    let mut side = |m: &Mention, id: String| match nodes.get(&(si, m.start, m.end)) {
                        Some(n) => {
                            let c = facts.entry((n.id.clone(), "coref".into(), n.antecedent.clone())).or_insert(0.0);
                            *c = c.max(n.conf);
                            let e = facts.entry((n.antecedent.clone(), "mention_of".into(), n.entity.clone())).or_insert(0.0);
                            *e = e.max(n.entity_score as f64);
                            n.id.clone()
                        }
                        None => id,
                    };
                    (side(sm, subj), side(om, obj))
                } else {
                    (subj, obj)
                };
                let e = facts.entry((subj, pred, obj)).or_insert(0.0);
                if conf > *e {
                    *e = conf;
                }
            }
        }
        self.trim_logs();
        facts
            .into_iter()
            .map(|((subj, pred, obj), confidence)| CandidateFact { subj, pred, obj, confidence })
            .collect()
    }
}

impl<B: RelationBackend, L: EntityLinker> Extractor for GlinerExtractor<B, L> {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        // an edited episode (same id, new text or speaker) is extracted again
        let print = fingerprint(episode);
        if let Some((p, cached)) = self.seen.get(&episode.id) {
            if *p == print {
                return cached.clone();
            }
        }
        self.calls += 1;
        let failures = self.failures;
        let out = self.run(episode);
        // memoize successes only: a failed turn (backend error) is retried
        // the next time the episode is extracted, instead of staying empty
        if self.failures == failures {
            if self.seen.insert(episode.id.clone(), (print, out.clone())).is_none() {
                self.seen_order.push_back(episode.id.clone());
            }
            while self.seen.len() > self.memo_capacity.max(1) {
                match self.seen_order.pop_front() {
                    Some(old) => {
                        self.seen.remove(&old);
                    }
                    None => break,
                }
            }
        }
        out
    }

    /// (model calls, failed calls)
    fn stats(&self) -> (usize, usize) {
        (self.calls, self.failures)
    }
}

/// A coref-resolved mention, for [`GlinerExtractor::coref_facts`].
#[derive(Debug, Clone)]
struct MentionNode {
    /// This mention's node id.
    id: String,
    /// Its antecedent's node id.
    antecedent: String,
    /// Confidence of the link (1 for a sieve, the gated score for tier 2).
    conf: f64,
    /// The antecedent's canonical id, and its link score.
    entity: String,
    entity_score: f32,
}

/// The node id of a mention: `m:<episode>:<sentence>:<start>-<end>`, with an
/// episode id that is long or has spaces replaced by its hash, so ids stay
/// short single tokens.
pub fn mention_id(episode: &str, sentence: usize, start: usize, end: usize) -> String {
    let ep = if episode.len() <= 24 && !episode.chars().any(|c| c.is_whitespace() || c == ':') {
        episode.to_string()
    } else {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        episode.hash(&mut h);
        format!("{:016x}", h.finish())
    };
    format!("m:{ep}:{sentence}:{start}-{end}")
}

/// What an episode's facts depend on besides its id: its text and speaker.
/// `DefaultHasher::new()` has fixed keys, so this is stable within and
/// across runs of one build.
fn fingerprint(ep: &Episode) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ep.text.hash(&mut h);
    ep.speaker.hash(&mut h);
    h.finish()
}

/// A backend that replays fixed analyses: for tests and offline replays.
/// It finds mentions by text, so fixtures stay readable:
/// `FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", 0.9)`.
#[derive(Debug, Clone, Default)]
pub struct FixtureBackend {
    rules: Vec<(String, String, String, String, String, f32)>,
    /// Standalone mentions: (text, label), reported at every whole-word
    /// occurrence, so a sentence can name an entity without a relation.
    mentions: Vec<(String, String)>,
    /// How many times `analyse` ran.
    pub calls: usize,
}

impl FixtureBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Report `subj --label--> obj` in every sentence containing both mentions
    /// (subject before object).
    pub fn relation(mut self, subj: &str, sl: &str, label: &str, obj: &str, ol: &str, score: f32) -> Self {
        self.rules.push((subj.into(), sl.into(), label.into(), obj.into(), ol.into(), score));
        self
    }

    /// Report `text` as a `label` mention wherever it occurs as a whole word
    /// ("Bob met Dana." names Bob and Dana with no relation between them).
    pub fn mention(mut self, text: &str, label: &str) -> Self {
        self.mentions.push((text.into(), label.into()));
        self
    }
}

/// Byte offsets of `word` in `s` as a whole word (not inside another word).
fn whole_word(s: &str, word: &str) -> Vec<usize> {
    let wordy = |c: char| c.is_alphanumeric() || c == '_';
    s.match_indices(word)
        .map(|(i, _)| i)
        .filter(|&i| {
            !s[..i].chars().next_back().is_some_and(wordy) && !s[i + word.len()..].chars().next().is_some_and(wordy)
        })
        .collect()
}

impl RelationBackend for FixtureBackend {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        self.calls += 1;
        Ok(sentences
            .iter()
            .map(|s| {
                let mut a = SentenceAnalysis::default();
                for (text, label) in &self.mentions {
                    for i in whole_word(s, text) {
                        a.mentions.push(Mention { text: text.clone(), label: label.clone(), start: i, end: i + text.len(), score: 0.9 });
                    }
                }
                for (subj, sl, label, obj, ol, score) in &self.rules {
                    let Some(si) = s.find(subj.as_str()) else { continue };
                    let Some(oi) = s[si + subj.len()..].find(obj.as_str()).map(|i| i + si + subj.len()) else {
                        continue;
                    };
                    let sm = Mention { text: subj.clone(), label: sl.clone(), start: si, end: si + subj.len(), score: *score };
                    let om = Mention { text: obj.clone(), label: ol.clone(), start: oi, end: oi + obj.len(), score: *score };
                    a.mentions.push(sm.clone());
                    a.mentions.push(om.clone());
                    a.relations.push(RawRelation { subject: sm, label: label.clone(), object: om, score: *score });
                }
                a
            })
            .collect())
    }
}
