//! Measure coreference on your own conversations: how often the Datalog
//! sieves abstain on a pronoun, how accurate each tier is, and the
//! mention-pair MCC, which tells you whether a second tier is worth adding.
//!
//! ```text
//! # 1. turn conversations into a labelling template (GLiNER finds mentions)
//! cargo run --release --example coref_eval -- label conversations.jsonl > gold.jsonl
//! #    conversations.jsonl: {"id": "...", "text": "..."} or
//! #                         {"id": "...", "turns": [{"speaker": "...", "text": "..."}]}
//! # 2. fill in "clusters" in gold.jsonl (see extract::coref_eval for the format)
//! # 3. score
//! cargo run --release --example coref_eval -- score gold.jsonl
//! ```
//!
//! `label` needs the GLiNER model (LEMMALOG_GLINER_DIR, see extract::setup)
//! and ONNX Runtime (ORT_DYLIB_PATH or ~/.lemmalog/lib). It leaves
//! "clusters" empty on purpose: pre-filling them with the sieves' answers
//! would anchor the labels to the system being measured. `--suggest` adds
//! the sieves' answers as a separate "suggested_clusters" field, which
//! `score` ignores. `score` needs no model: it runs the sieves on the gold
//! mentions.

use lemmalog::extract::coref::{resolve, Gate, Outcome};
use lemmalog::extract::coref_eval::{evaluate, parse_gold};
use lemmalog::extract::{split_sentences, Mention, RelationBackend};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("score") if args.len() >= 2 => score(&args[1], args.iter().any(|a| a == "--json")),
        Some("label") if args.len() >= 2 => label(&args[1], args.iter().any(|a| a == "--suggest")),
        _ => Err("usage: coref_eval score <gold.jsonl> [--json] | label <conversations.jsonl> [--suggest]".into()),
    };
    if let Err(e) = r {
        eprintln!("coref_eval: {e}");
        std::process::exit(1);
    }
}

fn score(path: &str, json: bool) -> Result<(), String> {
    let gold = parse_gold(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?)?;
    let report = evaluate(&gold, None, Gate::default())?;
    if json {
        let tiers: serde_json::Map<String, serde_json::Value> = report
            .tiers
            .iter()
            .map(|(k, t)| (k.clone(), serde_json::json!({"resolved": t.resolved, "correct": t.correct})))
            .collect();
        let c = &report.pairs;
        println!(
            "{}",
            serde_json::json!({
                "conversations": report.conversations, "mentions": report.mentions,
                "pronouns": report.pronouns, "abstained": report.abstained,
                "abstained_with_antecedent": report.abstained_with_antecedent,
                "abstained_by_reason": report.abstained_by_reason, "tiers": tiers,
                "pairs": {"tp": c.tp, "tn": c.tn, "fp": c.fp, "fn": c.fn_, "mcc": c.mcc()},
                "errors": report.errors.len(),
            })
        );
        return Ok(());
    }
    print!("{report}");
    if report.abstained_with_antecedent == 0 {
        println!("-> no abstention had a gold antecedent: a second tier has nothing to win here");
    } else {
        println!(
            "-> a second tier could resolve at most {} more pronoun(s) ({:.1}% of pronouns)",
            report.abstained_with_antecedent,
            100.0 * report.abstained_with_antecedent as f64 / report.pronouns.max(1) as f64
        );
    }
    Ok(())
}

fn label(path: &str, suggest: bool) -> Result<(), String> {
    use serde_json::{json, Value};
    let cfg = lemmalog::extract::GlinerConfig::from_env()?;
    let missing = cfg.missing();
    if !missing.is_empty() {
        return Err(format!("GLiNER model files not found: {} (set LEMMALOG_GLINER_DIR)", missing.join(", ")));
    }
    // as setup::build_gliner: ort reads ORT_DYLIB_PATH once, before the model loads
    if let (Some(p), None) = (&cfg.ort_dylib, std::env::var_os("ORT_DYLIB_PATH")) {
        std::env::set_var("ORT_DYLIB_PATH", p);
    }
    let mut backend = lemmalog::extract::gline::GlineBackend::new(&cfg.tokenizer, &cfg.model)?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let id = v["id"].as_str().map(str::to_string).unwrap_or_else(|| format!("c{}", i + 1));
        let body = match (&v["text"], v["turns"].as_array()) {
            (Value::String(t), _) => t.clone(),
            (_, Some(turns)) => turns
                .iter()
                .map(|t| t["text"].as_str().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => return Err(format!("line {}: need \"text\" or \"turns\"", i + 1)),
        };
        let sentences = split_sentences(&body);
        let analyses = backend.analyse(&sentences)?;
        let mut mentions: Vec<(usize, Mention)> = Vec::new();
        for (s, a) in analyses.iter().enumerate() {
            let mut ms: Vec<&Mention> = a.mentions.iter().chain(a.relations.iter().flat_map(|r| [&r.subject, &r.object])).collect();
            ms.sort_by_key(|m| (m.start, m.end));
            ms.dedup_by_key(|m| (m.start, m.end));
            mentions.extend(ms.into_iter().map(|m| (s, m.clone())));
        }
        let mut out = json!({
            "id": id,
            "sentences": sentences,
            "mentions": mentions.iter().map(|(s, m)| json!({"s": s, "text": m.text, "label": m.label})).collect::<Vec<_>>(),
            "clusters": [],
        });
        if suggest {
            let mut clusters: Vec<Vec<String>> = Vec::new();
            let r = |s: usize, m: &Mention| format!("{s}:{}", m.text);
            for ((s, start, end), o) in resolve(&mentions) {
                if let Outcome::Resolved { sentence, antecedent, .. } = o {
                    let m = mentions.iter().find(|(ms, mm)| *ms == s && mm.start == start && mm.end == end).map(|(_, m)| m);
                    if let Some(m) = m {
                        clusters.push(vec![r(sentence, &antecedent), r(s, m)]);
                    }
                }
            }
            clusters.sort();
            out["suggested_clusters"] = json!(clusters);
        }
        println!("{out}");
    }
    Ok(())
}
