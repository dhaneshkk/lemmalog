//! Mention-pair evaluation of coreference, per tier.
//!
//! Gold data is JSON Lines, one conversation per line:
//!
//! ```json
//! {"id": "c1",
//!  "sentences": ["Bob met Dana.", "He works at Acme."],
//!  "mentions": [{"s": 0, "text": "Bob", "label": "person"},
//!               {"s": 0, "text": "Dana", "label": "person"},
//!               {"s": 1, "text": "He", "label": "person"},
//!               {"s": 1, "text": "Acme", "label": "organization"}],
//!  "clusters": [["0:Bob", "1:He"]]}
//! ```
//!
//! A mention is referred to as `"<sentence>:<text>"`, or `"<sentence>:<text>#<k>"`
//! for the k-th occurrence (from 1) when the text repeats in that sentence.
//! `clusters` lists the mentions that refer to one entity; a mention in no
//! cluster is a singleton. First-person words are left to the speaker and not
//! scored.
//!
//! The report answers two questions:
//!
//! * per tier (each sieve, and the second tier if one is configured): how
//!   many mentions it resolved and how many of those were right;
//! * how often the sieves abstain on a pronoun, and how many of those
//!   abstentions had a gold antecedent (the most a second tier could add).
//!
//! The mention-pair score treats every ordered pair (earlier, later) of
//! scored mentions in a conversation as a binary decision, same entity or
//! not, gold against predicted clusters, and reports the confusion matrix
//! and MCC.

use std::collections::{BTreeMap, HashMap};

use super::coref::{pronoun_class, resolve_tiered, Gate, Outcome, PairScorer, Place};
use super::eval::Confusion;
use super::pipeline::Mention;

#[derive(Debug, Clone, PartialEq)]
pub struct GoldMention {
    pub s: usize,
    pub text: String,
    /// Default `person`.
    pub label: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoldConversation {
    pub id: String,
    pub sentences: Vec<String>,
    pub mentions: Vec<GoldMention>,
    pub clusters: Vec<Vec<String>>,
}

/// Parse gold JSON Lines (blank lines and `//` comment lines skipped).
pub fn parse_gold(jsonl: &str) -> Result<Vec<GoldConversation>, String> {
    use serde_json::Value;
    let strs = |v: &Value, what: &str| -> Result<Vec<String>, String> {
        v.as_array()
            .ok_or(format!("{what} is not a list"))?
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or(format!("{what} holds a non-string")))
            .collect()
    };
    let mut out = Vec::new();
    for (i, l) in jsonl.lines().enumerate() {
        if l.trim().is_empty() || l.trim_start().starts_with("//") {
            continue;
        }
        let err = |e: String| format!("line {}: {e}", i + 1);
        let v: Value = serde_json::from_str(l).map_err(|e| err(e.to_string()))?;
        let id = v["id"].as_str().ok_or_else(|| err("missing \"id\"".into()))?.to_string();
        let sentences = strs(&v["sentences"], "sentences").map_err(err)?;
        let mut mentions = Vec::new();
        for m in v["mentions"].as_array().ok_or_else(|| err("\"mentions\" is not a list".into()))? {
            mentions.push(GoldMention {
                s: m["s"].as_u64().ok_or_else(|| err("a mention has no sentence \"s\"".into()))? as usize,
                text: m["text"].as_str().ok_or_else(|| err("a mention has no \"text\"".into()))?.to_string(),
                label: m["label"].as_str().unwrap_or("person").to_string(),
            });
        }
        let clusters = match &v["clusters"] {
            Value::Null => Vec::new(),
            c => c
                .as_array()
                .ok_or_else(|| err("\"clusters\" is not a list".into()))?
                .iter()
                .map(|x| strs(x, "a cluster"))
                .collect::<Result<_, _>>()
                .map_err(err)?,
        };
        out.push(GoldConversation { id, sentences, mentions, clusters });
    }
    Ok(out)
}

/// One tier's tally.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TierScore {
    /// Mentions this tier resolved.
    pub resolved: usize,
    /// Of those, resolved to a mention of their own gold cluster.
    pub correct: usize,
}

impl TierScore {
    pub fn accuracy(&self) -> f64 {
        if self.resolved == 0 {
            0.0
        } else {
            self.correct as f64 / self.resolved as f64
        }
    }
}

/// The evaluation of one run over a gold set.
#[derive(Debug, Clone, Default)]
pub struct CorefReport {
    pub conversations: usize,
    /// Scored mentions (first person excluded).
    pub mentions: usize,
    pub pronouns: usize,
    /// Pronouns every tier left unresolved.
    pub abstained: usize,
    /// Abstained pronouns that do have a gold antecedent: what a second
    /// tier could still win.
    pub abstained_with_antecedent: usize,
    /// Abstentions by the sieves' reason ("ambiguous", "no antecedent").
    pub abstained_by_reason: BTreeMap<String, usize>,
    /// Per tier, keyed by sieve name ("exact", "acronym", "head",
    /// "pronoun", or a second tier's name).
    pub tiers: BTreeMap<String, TierScore>,
    /// Mention pairs: same entity or not, gold vs predicted.
    pub pairs: Confusion,
    /// Wrong resolutions, as (conversation, mention, predicted antecedent).
    pub errors: Vec<(String, String, String)>,
}

impl CorefReport {
    /// Share of pronouns every tier abstained on.
    pub fn abstention_rate(&self) -> f64 {
        if self.pronouns == 0 {
            0.0
        } else {
            self.abstained as f64 / self.pronouns as f64
        }
    }
}

impl std::fmt::Display for CorefReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "conversations {}, scored mentions {}, pronouns {}", self.conversations, self.mentions, self.pronouns)?;
        writeln!(f, "per tier:")?;
        for (t, s) in &self.tiers {
            writeln!(f, "  {t:<10} resolved {:>4}  correct {:>4}  accuracy {:.3}", s.resolved, s.correct, s.accuracy())?;
        }
        writeln!(
            f,
            "pronoun abstentions: {} of {} ({:.1}%), {} with a gold antecedent (second-tier headroom)",
            self.abstained,
            self.pronouns,
            100.0 * self.abstention_rate(),
            self.abstained_with_antecedent
        )?;
        for (r, n) in &self.abstained_by_reason {
            writeln!(f, "  {r}: {n}")?;
        }
        let c = &self.pairs;
        writeln!(
            f,
            "mention pairs: TP={} TN={} FP={} FN={}  precision={:.3} recall={:.3}  MCC={:.3}",
            c.tp,
            c.tn,
            c.fp,
            c.fn_,
            c.precision(),
            c.recall(),
            c.mcc()
        )?;
        for (conv, m, a) in self.errors.iter().take(20) {
            writeln!(f, "  wrong: [{conv}] {m} -> {a}")?;
        }
        Ok(())
    }
}

/// A gold mention reference "<s>:<text>" or "<s>:<text>#<k>".
fn place_of(conv: &GoldConversation, r: &str) -> Result<Place, String> {
    let (s, rest) = r.split_once(':').ok_or_else(|| format!("[{}] bad mention ref {r:?} (want \"<sentence>:<text>\")", conv.id))?;
    let s: usize = s.parse().map_err(|_| format!("[{}] bad sentence in {r:?}", conv.id))?;
    let (text, k) = match rest.rsplit_once('#') {
        Some((t, k)) if k.parse::<usize>().is_ok() => (t, k.parse::<usize>().unwrap()),
        _ => (rest, 1),
    };
    let sentence = conv.sentences.get(s).ok_or_else(|| format!("[{}] no sentence {s} for {r:?}", conv.id))?;
    let start = sentence
        .match_indices(text)
        .nth(k.saturating_sub(1))
        .map(|(i, _)| i)
        .ok_or_else(|| format!("[{}] {text:?} (occurrence {k}) not in sentence {s}: {sentence:?}", conv.id))?;
    Ok((s, start, start + text.len()))
}

fn first_person(t: &str) -> bool {
    matches!(t.to_lowercase().as_str(), "i" | "me" | "my" | "mine" | "myself")
}

/// Score the sieves (and `scorer` as a second tier, if given) on a gold set.
pub fn evaluate(
    gold: &[GoldConversation],
    mut scorer: Option<&mut (dyn PairScorer + 'static)>,
    gate: Gate,
) -> Result<CorefReport, String> {
    let mut rep = CorefReport::default();
    for conv in gold {
        rep.conversations += 1;
        // mentions, in the order the gold file names them (k-th occurrence
        // per sentence and text)
        let mut seen: HashMap<(usize, String), usize> = HashMap::new();
        let mut ms: Vec<(usize, Mention)> = Vec::new();
        for m in &conv.mentions {
            let k = seen.entry((m.s, m.text.clone())).or_insert(0);
            *k += 1;
            let place = place_of(conv, &format!("{}:{}#{}", m.s, m.text, k))?;
            ms.push((m.s, Mention { text: m.text.clone(), label: m.label.clone(), start: place.1, end: place.2, score: 1.0 }));
        }
        ms.sort_by_key(|(s, m)| (*s, m.start));
        // gold clusters
        let mut gold_id: HashMap<Place, usize> = HashMap::new();
        for (ci, c) in conv.clusters.iter().enumerate() {
            for r in c {
                gold_id.insert(place_of(conv, r)?, ci);
            }
        }
        let scored: Vec<Place> =
            ms.iter().filter(|(_, m)| !first_person(&m.text)).map(|(s, m)| (*s, m.start, m.end)).collect();
        let text_of: HashMap<Place, String> = ms.iter().map(|(s, m)| ((*s, m.start, m.end), format!("{s}:{}", m.text))).collect();
        let same_gold = |a: &Place, b: &Place| matches!((gold_id.get(a), gold_id.get(b)), (Some(x), Some(y)) if x == y);
        let out = resolve_tiered(&conv.sentences, &ms, scorer.as_deref_mut(), gate);
        // predicted clusters: each resolved mention joins its antecedent's
        let mut parent: HashMap<Place, Place> = HashMap::new();
        fn find(p: &HashMap<Place, Place>, x: Place) -> Place {
            let mut x = x;
            while let Some(&y) = p.get(&x) {
                if y == x {
                    break;
                }
                x = y;
            }
            x
        }
        for place in &scored {
            rep.mentions += 1;
            let is_pron = pronoun_class(&text_of[place][text_of[place].find(':').unwrap() + 1..]).is_some();
            if is_pron {
                rep.pronouns += 1;
            }
            match out.get(place) {
                Some(Outcome::Resolved { sentence, antecedent, sieve, .. }) => {
                    let a = (*sentence, antecedent.start, antecedent.end);
                    let ok = same_gold(place, &a);
                    let t = rep.tiers.entry(sieve.clone()).or_default();
                    t.resolved += 1;
                    if ok {
                        t.correct += 1;
                    } else {
                        rep.errors.push((conv.id.clone(), text_of[place].clone(), text_of.get(&a).cloned().unwrap_or_default()));
                    }
                    let (ra, rb) = (find(&parent, *place), find(&parent, a));
                    if ra != rb {
                        parent.insert(ra, rb);
                    }
                }
                Some(other) if is_pron => {
                    rep.abstained += 1;
                    let reason = if matches!(other, Outcome::Ambiguous(_)) { "ambiguous" } else { "no antecedent" };
                    *rep.abstained_by_reason.entry(reason.into()).or_default() += 1;
                    if scored.iter().any(|b| b < place && same_gold(b, place)) {
                        rep.abstained_with_antecedent += 1;
                    }
                }
                None if is_pron => {
                    rep.abstained += 1;
                    *rep.abstained_by_reason.entry("not reached".into()).or_default() += 1;
                }
                _ => {}
            }
        }
        // mention pairs
        for (j, b) in scored.iter().enumerate() {
            for a in &scored[..j] {
                let g = same_gold(a, b);
                let p = find(&parent, *a) == find(&parent, *b);
                match (g, p) {
                    (true, true) => rep.pairs.tp += 1,
                    (false, false) => rep.pairs.tn += 1,
                    (false, true) => rep.pairs.fp += 1,
                    (true, false) => rep.pairs.fn_ += 1,
                }
            }
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLD: &str = r#"
// a clear pronoun, an ambiguous one, a head-word name, an acronym
{"id":"a","sentences":["Bob joined Acme.","He works on search."],"mentions":[{"s":0,"text":"Bob"},{"s":0,"text":"Acme","label":"organization"},{"s":1,"text":"He"}],"clusters":[["0:Bob","1:He"]]}
{"id":"b","sentences":["Bob met Dana.","She works at Acme."],"mentions":[{"s":0,"text":"Bob"},{"s":0,"text":"Dana"},{"s":1,"text":"She"},{"s":1,"text":"Acme","label":"organization"}],"clusters":[["0:Dana","1:She"]]}
{"id":"c","sentences":["Dana Smith joined International Business Machines.","Smith likes IBM."],"mentions":[{"s":0,"text":"Dana Smith"},{"s":0,"text":"International Business Machines","label":"organization"},{"s":1,"text":"Smith"},{"s":1,"text":"IBM","label":"organization"}],"clusters":[["0:Dana Smith","1:Smith"],["0:International Business Machines","1:IBM"]]}
"#;

    #[test]
    fn reports_per_tier_accuracy_abstentions_and_pair_mcc() {
        let gold = parse_gold(GOLD).unwrap();
        let r = evaluate(&gold, None, Gate::default()).unwrap();
        assert_eq!((r.conversations, r.mentions, r.pronouns), (3, 11, 2));
        assert_eq!(r.tiers["pronoun"], TierScore { resolved: 1, correct: 1 });
        assert_eq!(r.tiers["head"], TierScore { resolved: 1, correct: 1 });
        assert_eq!(r.tiers["acronym"], TierScore { resolved: 1, correct: 1 });
        // "She" in b: Bob and Dana are both persons in range -> abstained,
        // and it does have a gold antecedent (Dana)
        assert_eq!((r.abstained, r.abstained_with_antecedent), (1, 1));
        assert_eq!(r.abstained_by_reason["ambiguous"], 1);
        // pairs: a 3, b 6, c 6 = 15; gold-coreferent 4, predicted 3 (all right)
        assert_eq!((r.pairs.tp, r.pairs.fp, r.pairs.fn_, r.pairs.tn), (3, 0, 1, 11));
        assert!(r.pairs.mcc() > 0.8 && r.pairs.mcc() < 1.0, "{}", r.pairs.mcc());
        assert!(r.to_string().contains("pronoun abstentions: 1 of 2"), "{r}");
    }

    #[test]
    fn a_second_tier_is_scored_as_its_own_tier() {
        struct DanaFirst;
        impl PairScorer for DanaFirst {
            fn name(&self) -> &str {
                "t2"
            }
            fn score(&mut self, _: &[String], _: &(usize, Mention), c: &[(usize, Mention)]) -> Result<Vec<f32>, String> {
                Ok(c.iter().map(|(_, m)| if m.text == "Dana" { 0.95 } else { 0.1 }).collect())
            }
        }
        let gold = parse_gold(GOLD).unwrap();
        let mut t = DanaFirst;
        let r = evaluate(&gold, Some(&mut t), Gate::default()).unwrap();
        assert_eq!(r.tiers["t2"], TierScore { resolved: 1, correct: 1 });
        assert_eq!(r.abstained, 0);
        assert_eq!((r.pairs.tp, r.pairs.fn_), (4, 0));
        assert!((r.pairs.mcc() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn wrong_resolutions_are_listed_and_count_against_the_tier() {
        // gold says "He" is Bob, but only Carl is in range: the sieve picks Carl
        let g = r#"{"id":"w","sentences":["Bob left.","x.","y.","Carl stayed.","He smiled."],"mentions":[{"s":0,"text":"Bob"},{"s":3,"text":"Carl"},{"s":4,"text":"He"}],"clusters":[["0:Bob","4:He"]]}"#;
        let r = evaluate(&parse_gold(g).unwrap(), None, Gate::default()).unwrap();
        assert_eq!(r.tiers["pronoun"], TierScore { resolved: 1, correct: 0 });
        assert_eq!(r.errors, vec![("w".into(), "4:He".into(), "3:Carl".into())]);
        assert_eq!((r.pairs.fp, r.pairs.fn_), (1, 1));
    }

    #[test]
    fn bad_references_are_errors_not_guesses() {
        let g = r#"{"id":"x","sentences":["Bob left."],"mentions":[{"s":0,"text":"Bob"}],"clusters":[["0:Bobby"]]}"#;
        assert!(evaluate(&parse_gold(g).unwrap(), None, Gate::default()).unwrap_err().contains("not in sentence"));
        assert!(parse_gold("{not json}").is_err());
    }
}
