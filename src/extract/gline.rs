//! [`GlineBackend`]: the real [`RelationBackend`], running a GLiNER
//! multitask model (token mode) through gline-rs. NER first, then relation
//! extraction over the NER spans, both on the same ONNX session.
//!
//! Model files are not shipped. Get them from Hugging Face, e.g.
//! `onnx-community/gliner-multitask-large-v0.5`:
//!
//! ```text
//! models/gliner-multitask-large-v0.5/tokenizer.json
//! models/gliner-multitask-large-v0.5/onnx/model.onnx
//! ```
//!
//! Two gline-rs behaviours shape this backend:
//!
//! * gline-rs runs relation extraction on a prompt, `"{PREFIX} {sentence}"`,
//!   and reports object spans against that prompt, not the sentence. The
//!   span is mapped back by [`RELATION_PROMPT_OFFSET`] and used only if it
//!   then covers the object's text exactly; otherwise the object is found by
//!   text ([`resolve_relations`]).
//! * gline-rs builds one relation label per (subject entity, relation) over
//!   the whole batch. The backend therefore runs one sentence per call, so
//!   labels never mix sentences and grow with one sentence's entities only,
//!   capped by [`GlineBackend::max_entities`].

use super::pipeline::{
    default_entity_labels, resolve_relations, Mention, ModelRelation, RelationBackend, SentenceAnalysis,
    DEFAULT_RELATIONS,
};
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::input::text::TextInput;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::output::relation::Relation;
use gliner::model::params::Parameters;
use gliner::model::pipeline::{relation::RelationPipeline, token::TokenPipeline};
use gliner::text::span::Span;
use gliner::text::splitter::RegexSplitter;
use gliner::text::tokenizer::HFTokenizer;
use orp::model::Model;
use orp::params::RuntimeParameters;

type Ner = TokenPipeline<RegexSplitter, HFTokenizer>;
type Rel = RelationPipeline<'static, RegexSplitter, HFTokenizer>;

/// gline-rs's relation prompt prefix (`PROMPT_PREFIX` in
/// `gline-rs/src/model/input/relation/mod.rs`), private to that crate.
pub const RELATION_PROMPT_PREFIX: &str = "Extract relationships between entities from the text: \n";
/// Bytes between the start of the relation prompt and the sentence:
/// gline-rs formats the prompt as `"{prefix} {text}"`. This is what the
/// pinned gline-rs does; [`relation_prompt_offset`] measures it from
/// gline-rs itself, and that is what mapping uses.
pub const RELATION_PROMPT_OFFSET: usize = RELATION_PROMPT_PREFIX.len() + 1;

/// The offset of the sentence inside the relation prompt, measured once from
/// the prompt gline-rs actually builds (`RelationInput::from_spans`), so a
/// change to gline-rs's prompt format moves object spans with it instead of
/// silently sending every object to the by-text fallback. Falls back to
/// [`RELATION_PROMPT_OFFSET`] if gline-rs's prompt does not end with the
/// sentence.
pub fn relation_prompt_offset() -> usize {
    static OFFSET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *OFFSET.get_or_init(|| {
        use gliner::model::input::relation::RelationInput;
        let text = "Bob works at Acme.";
        let spans = SpanOutput::new(vec![text.to_string()], vec![], vec![vec![]]);
        let input = RelationInput::from_spans(spans, &RelationSchema::new());
        input
            .prompts
            .first()
            .and_then(|p| p.strip_suffix(text).map(str::len))
            .unwrap_or(RELATION_PROMPT_OFFSET)
    })
}

/// Which ONNX Runtime execution provider the models run on, from
/// `LEMMALOG_ORT_EP`:
///
/// * `auto` (default): CUDA if the loaded ONNX Runtime has the CUDA provider
///   and it loads (CUDA 12 + cuDNN 9 present), else OpenVINO if it has the
///   OpenVINO provider, else CPU;
/// * `cuda`: CUDA on device `LEMMALOG_CUDA_DEVICE` (default 0); the model
///   load fails if CUDA cannot be used;
/// * `openvino`: OpenVINO on `LEMMALOG_OPENVINO_DEVICE` (default
///   `AUTO:GPU,CPU`: an Intel GPU if present, else CPU; `CPU`, `GPU`, `NPU`
///   pick one, and a device that is not present fails the load);
/// * `cpu`.
///
/// `cuda` and `openvino` need the matching Cargo feature (both are default).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionProvider {
    Auto { cuda_device: i32, openvino_device: String },
    Cuda { device: i32 },
    OpenVino { device: String },
    Cpu,
}

impl ExecutionProvider {
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let get = |k: &str| var(k).filter(|v| !v.trim().is_empty()).map(|v| v.trim().to_string());
        let cuda_device = match get("LEMMALOG_CUDA_DEVICE") {
            None => 0,
            Some(d) => d.parse().map_err(|_| format!("LEMMALOG_CUDA_DEVICE={d:?} is not a device number"))?,
        };
        let openvino_device = get("LEMMALOG_OPENVINO_DEVICE").unwrap_or_else(|| "AUTO:GPU,CPU".into());
        let ep = get("LEMMALOG_ORT_EP").unwrap_or_else(|| "auto".into()).to_lowercase();
        let missing = |f: &str| Err(format!("LEMMALOG_ORT_EP={ep}, but lemmalog was built without the `{f}` feature"));
        match ep.as_str() {
            "auto" => Ok(ExecutionProvider::Auto { cuda_device, openvino_device }),
            "cuda" if cfg!(feature = "cuda") => Ok(ExecutionProvider::Cuda { device: cuda_device }),
            "cuda" => missing("cuda"),
            "openvino" if cfg!(feature = "openvino") => Ok(ExecutionProvider::OpenVino { device: openvino_device }),
            "openvino" => missing("openvino"),
            "cpu" => Ok(ExecutionProvider::Cpu),
            other => Err(format!("unknown LEMMALOG_ORT_EP={other:?} (use auto, cuda, openvino or cpu)")),
        }
    }
}

/// The execution provider the models run on, once a model has loaded:
/// "CUDA (device 0)", "OpenVINO (AUTO:GPU,CPU)", "CPU", or "CPU (...why
/// no accelerator is used)".
pub fn active_execution_provider() -> Option<&'static str> {
    ACTIVE_EP.get().map(String::as_str)
}

static ACTIVE_EP: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// The ONNX Runtime environment must exist before a provider is queried
/// (otherwise: "Attempt to use DefaultLogger but none has been registered").
#[cfg(any(feature = "openvino", feature = "cuda"))]
fn init_ort() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = ort::init().with_name("lemmalog").commit();
    });
}

/// Can CUDA run here? Registering it on a throwaway session builder loads
/// the provider library and its CUDA/cuDNN dependencies, so this fails when
/// they are missing (the provider being compiled in is not enough).
#[cfg(feature = "cuda")]
fn cuda_usable(device: i32) -> Result<(), String> {
    use ort::execution_providers::CUDAExecutionProvider;
    init_ort();
    let ep = CUDAExecutionProvider::default().with_device_id(device).build().error_on_failure();
    ort::session::Session::builder()
        .and_then(|b| b.with_execution_providers([ep]))
        .map(|_| ())
        .map_err(|e| short_ort_error(&e.to_string()))
}

/// The useful part of an ONNX Runtime error: "Failed to load library ...
/// with error: libcudnn.so.9: cannot open shared object file" becomes the
/// part after "with error:"; otherwise the message without its source
/// location prefix.
#[cfg(feature = "cuda")]
fn short_ort_error(e: &str) -> String {
    let e = e.lines().next().unwrap_or_default();
    let e = match e.rsplit_once("with error: ") {
        Some((_, tail)) => tail,
        None => e.rsplit_once(" : ").map_or(e, |(_, t)| t),
    };
    e.chars().take(200).collect()
}

/// Does the loaded ONNX Runtime have the OpenVINO provider?
#[cfg(feature = "openvino")]
fn openvino_usable() -> Result<(), String> {
    use ort::execution_providers::{ExecutionProvider as _, OpenVINOExecutionProvider};
    init_ort();
    match OpenVINOExecutionProvider::default().is_available() {
        Ok(true) => Ok(()),
        Ok(false) => Err("this ONNX Runtime has no OpenVINO provider".into()),
        Err(e) => Err(e.to_string()),
    }
}

/// Runtime parameters for the models: the execution provider from the
/// environment (see [`ExecutionProvider`]). Which one is in use is reported
/// by [`active_execution_provider`] and on stderr.
fn runtime_parameters() -> Result<RuntimeParameters, String> {
    #[allow(unused_mut)]
    let mut why_not: Vec<String> = Vec::new();
    let (active, params) = match ExecutionProvider::from_vars(|k| std::env::var(k).ok())? {
        ExecutionProvider::Cpu => ("CPU".to_string(), RuntimeParameters::default()),
        #[cfg(feature = "cuda")]
        ExecutionProvider::Cuda { device } => {
            use ort::execution_providers::CUDAExecutionProvider;
            cuda_usable(device).map_err(|e| format!("LEMMALOG_ORT_EP=cuda, but CUDA cannot be used: {e}"))?;
            let ep = CUDAExecutionProvider::default().with_device_id(device).build().error_on_failure();
            (format!("CUDA (device {device})"), RuntimeParameters::default().with_execution_providers([ep]))
        }
        #[cfg(feature = "openvino")]
        ExecutionProvider::OpenVino { device } => {
            use ort::execution_providers::OpenVINOExecutionProvider;
            openvino_usable().map_err(|e| format!("LEMMALOG_ORT_EP=openvino, but OpenVINO cannot be used: {e}"))?;
            let ep = OpenVINOExecutionProvider::default().with_device_type(&device).build().error_on_failure();
            (format!("OpenVINO ({device})"), RuntimeParameters::default().with_execution_providers([ep]))
        }
        #[allow(unused_variables)]
        ExecutionProvider::Auto { cuda_device, openvino_device } => {
            #[allow(unused_mut)]
            let mut chosen: Option<(String, RuntimeParameters)> = None;
            #[cfg(feature = "cuda")]
            if chosen.is_none() {
                use ort::execution_providers::CUDAExecutionProvider;
                match cuda_usable(cuda_device) {
                    Ok(()) => {
                        let ep = CUDAExecutionProvider::default().with_device_id(cuda_device).build();
                        chosen = Some((format!("CUDA (device {cuda_device})"), RuntimeParameters::default().with_execution_providers([ep])));
                    }
                    Err(e) => why_not.push(format!("CUDA: {e}")),
                }
            }
            #[cfg(feature = "openvino")]
            if chosen.is_none() {
                use ort::execution_providers::OpenVINOExecutionProvider;
                match openvino_usable() {
                    Ok(()) => {
                        let ep = OpenVINOExecutionProvider::default().with_device_type(&openvino_device).build();
                        chosen = Some((format!("OpenVINO ({openvino_device})"), RuntimeParameters::default().with_execution_providers([ep])));
                    }
                    Err(e) => why_not.push(format!("OpenVINO: {e}")),
                }
            }
            chosen.unwrap_or_else(|| {
                let why = if why_not.is_empty() { String::new() } else { format!(" ({})", why_not.join("; ")) };
                (format!("CPU{why}"), RuntimeParameters::default())
            })
        }
        #[allow(unreachable_patterns)]
        other => unreachable!("from_vars rejects {other:?} without its feature"),
    };
    eprintln!("lemmalog: ONNX Runtime execution provider: {active}");
    let _ = ACTIVE_EP.set(active);
    Ok(params)
}

/// One model per model file for the whole process. The relation pipelines
/// borrow the model for `'static`, so a loaded model is never freed; sharing
/// it means building many backends (eval notebooks, tests) costs one model
/// per file, not one per backend.
fn shared_model(model_path: &str) -> Result<&'static Model, String> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static MODELS: OnceLock<Mutex<HashMap<String, &'static Model>>> = OnceLock::new();
    let mut models = MODELS.get_or_init(Default::default).lock().map_err(|_| "model cache poisoned".to_string())?;
    if let Some(m) = models.get(model_path) {
        return Ok(*m);
    }
    let m: &'static Model =
        Box::leak(Box::new(Model::new(model_path, runtime_parameters()?).map_err(|x| x.to_string())?));
    models.insert(model_path.to_string(), m);
    Ok(m)
}

/// Extra object labels for the relaxed schema: person-to-person relations
/// also accept an object GLiNER typed as an organization. GLiNER sometimes
/// types a short first name that way ("Priya is the manager of Sam" tags
/// "Sam" as an organization), and gline-rs then never offers the relation.
pub const RELAXED_OBJECTS: &[(&str, &str)] = &[("reports to", "organization"), ("manager of", "organization")];

/// Does any span's class appear as an allowed subject in the schema? If
/// not, gline-rs would build no relation labels and refuse the input.
pub fn has_subject_candidate(schema: &RelationSchema, classes: &[&str]) -> bool {
    schema.relations().values().any(|spec| classes.iter().any(|c| spec.allows_subject(c)))
}

pub struct GlineBackend {
    // The relation pipelines borrow their schemas and `Model::inference` ties
    // the pipeline lifetime to the model borrow, so both live for the
    // process: models are shared per file (`shared_model`), schemas (small)
    // are leaked per backend.
    model: &'static Model,
    ner: Ner,
    rel: Rel,
    schema: &'static RelationSchema,
    relaxed: Option<(Rel, &'static RelationSchema)>,
    params: Parameters,
    labels: Vec<String>,
    /// At most this many entity spans (highest score first) per sentence go
    /// into relation extraction, bounding the label prompt. Default 12.
    pub max_entities: usize,
    /// Add "I"/"me"/"myself" as `person` spans when NER missed them, so
    /// first-person relations reach the speaker. Default true.
    pub first_person: bool,
    /// Add third-person pronouns ("he", "she", "it", "they", ...) as spans
    /// when NER missed them, so their relations reach the coreference
    /// sieves (SPEC-7.1-O7), which resolve them or report them. Default true.
    pub third_person: bool,
    /// Use the relaxed schema ([`RELAXED_OBJECTS`]) for relation extraction.
    /// Off by default: it can let a real organization be a person's
    /// manager. Only [`GlineBackend::new`] builds a relaxed schema.
    pub relaxed_objects: bool,
    /// Entity spans dropped by `max_entities` so far.
    pub truncated: usize,
    /// Sentences whose relation step was skipped because no entity could be
    /// a subject (their mentions are still returned).
    pub no_subject: usize,
}

fn schema_from(relax: &[(&str, &str)]) -> RelationSchema {
    let mut schema = RelationSchema::new();
    for (label, _, subj, obj, _) in DEFAULT_RELATIONS {
        let mut objs: Vec<&str> = obj.to_vec();
        objs.extend(relax.iter().filter(|(l, _)| l == label).map(|(_, o)| *o));
        schema.push_with_allowed_labels(label, subj, &objs);
    }
    schema
}

impl GlineBackend {
    /// Load the model with the default relation set (see [`DEFAULT_RELATIONS`]),
    /// plus its relaxed variant (see [`GlineBackend::relaxed_objects`]).
    pub fn new(tokenizer_path: &str, model_path: &str) -> Result<Self, String> {
        let mut b = Self::with_schema(tokenizer_path, model_path, default_entity_labels(), schema_from(&[]))?;
        let relaxed: &'static RelationSchema = Box::leak(Box::new(schema_from(RELAXED_OBJECTS)));
        let rel = RelationPipeline::default(tokenizer_path, relaxed).map_err(|x| x.to_string())?;
        b.relaxed = Some((rel, relaxed));
        Ok(b)
    }

    pub fn with_schema(
        tokenizer_path: &str,
        model_path: &str,
        labels: Vec<String>,
        schema: RelationSchema,
    ) -> Result<Self, String> {
        let e = |x: Box<dyn std::error::Error + Send + Sync>| x.to_string();
        let model = shared_model(model_path)?;
        let schema: &'static RelationSchema = Box::leak(Box::new(schema));
        Ok(GlineBackend {
            model,
            ner: TokenPipeline::new(tokenizer_path).map_err(e)?,
            rel: RelationPipeline::default(tokenizer_path, schema).map_err(e)?,
            schema,
            relaxed: None,
            params: Parameters::default(),
            labels,
            max_entities: 12,
            first_person: true,
            third_person: true,
            relaxed_objects: false,
            truncated: 0,
            no_subject: 0,
        })
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.params = self.params.with_threshold(t);
        self
    }

    fn analyse_one(&mut self, sentence: &str) -> Result<SentenceAnalysis, String> {
        let e = |stage: &'static str| {
            move |x: Box<dyn std::error::Error + Send + Sync>| format!("{stage} failed on {sentence:?}: {x}")
        };
        let input = TextInput::new(vec![sentence.to_string()], self.labels.clone()).map_err(e("NER input"))?;
        let out: SpanOutput = self.model.inference(input, &self.ner, &self.params).map_err(e("NER"))?;
        let mut spans: Vec<Span> = out.spans.into_iter().next().unwrap_or_default();
        if self.first_person && self.labels.iter().any(|l| l == "person") {
            spans.extend(first_person_spans(sentence, &spans));
        }
        if self.third_person {
            spans.extend(third_person_spans(sentence, &spans, &self.labels));
        }
        spans.sort_by(|a, b| b.probability().total_cmp(&a.probability()));
        if spans.len() > self.max_entities {
            self.truncated += spans.len() - self.max_entities;
            spans.truncate(self.max_entities);
        }
        spans.sort_by_key(|s| s.offsets());
        let mentions: Vec<Mention> = spans.iter().map(mention).collect();
        let (rel, schema) = match (&self.relaxed, self.relaxed_objects) {
            (Some((r, s)), true) => (r, *s),
            _ => (&self.rel, self.schema),
        };
        let classes: Vec<&str> = spans.iter().map(|s| s.class()).collect();
        if !has_subject_candidate(schema, &classes) {
            // gline-rs would build no relation labels and reject the input
            self.no_subject += 1;
            return Ok(SentenceAnalysis { mentions, ..Default::default() });
        }
        let again = SpanOutput::new(vec![sentence.to_string()], out.entities, vec![spans]);
        let rels = self.model.inference(again, rel, &self.params).map_err(e("relation extraction"))?;
        let rels: Vec<Relation> = rels.relations.into_iter().flatten().collect();
        Ok(map_relations(sentence, mentions, &rels))
    }
}

fn mention(s: &Span) -> Mention {
    let (start, end) = s.offsets();
    Mention { text: s.text().to_string(), label: s.class().to_string(), start, end, score: s.probability() }
}

/// "I", "me" and "myself" (as whole words) not already covered by a span.
pub fn first_person_spans(sentence: &str, existing: &[Span]) -> Vec<Span> {
    super::negex::tokenize(sentence)
        .into_iter()
        .filter(|t| {
            let raw = &sentence[t.start..t.end];
            raw == "I" || t.lower == "me" || t.lower == "myself"
        })
        .filter(|t| existing.iter().all(|s| s.offsets().1 <= t.start || s.offsets().0 >= t.end))
        .map(|t| Span::new(0, t.start, t.end, sentence[t.start..t.end].to_string(), "person".into(), 1.0))
        .collect()
}

/// Third-person pronouns (as whole words, not already covered by a span),
/// labelled with a class GLiNER relations accept: "he"/"she"/"they" as
/// `person`, "it" as `organization`, when those labels are in use.
/// Possessives ("his", "its", "their") are left out: they qualify a noun
/// rather than name an entity.
pub fn third_person_spans(sentence: &str, existing: &[Span], labels: &[String]) -> Vec<Span> {
    let has = |l: &str| labels.iter().any(|x| x == l);
    super::negex::tokenize(sentence)
        .into_iter()
        .filter_map(|t| {
            let label = match t.lower.as_str() {
                "he" | "him" | "she" | "her" | "they" | "them" if has("person") => "person",
                "it" if has("organization") => "organization",
                _ => return None,
            };
            Some((t, label))
        })
        .filter(|(t, _)| existing.iter().all(|s| s.offsets().1 <= t.start || s.offsets().0 >= t.end))
        .map(|(t, label)| Span::new(0, t.start, t.end, sentence[t.start..t.end].to_string(), label.into(), 1.0))
        .collect()
}

/// Turn gline-rs relations for one sentence into a [`SentenceAnalysis`],
/// mapping prompt-relative object spans back to the sentence.
pub fn map_relations(sentence: &str, mentions: Vec<Mention>, rels: &[Relation]) -> SentenceAnalysis {
    let model: Vec<ModelRelation> = rels
        .iter()
        .map(|r| {
            let (a, b) = r.offsets();
            ModelRelation {
                subject: r.subject().to_string(),
                label: r.class().to_string(),
                object: r.object().to_string(),
                object_span: a
                    .checked_sub(relation_prompt_offset())
                    .zip(b.checked_sub(relation_prompt_offset())),
                score: r.probability(),
            }
        })
        .collect();
    resolve_relations(sentence, mentions, &model)
}

impl RelationBackend for GlineBackend {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        sentences.iter().map(|s| self.analyse_one(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn ort_errors_are_shortened_to_their_cause() {
        let e = "/ort-src/onnxruntime/core/session/provider_bridge_ort.cc:1952 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library /x/libonnxruntime_providers_cuda.so with error: libcublasLt.so.12: cannot open shared object file: No such file or directory";
        assert_eq!(short_ort_error(e), "libcublasLt.so.12: cannot open shared object file: No such file or directory");
        assert_eq!(short_ort_error("a.cc:1 f() [ONNXRuntimeError] : 1 : FAIL : CUDA device 3 not found"), "CUDA device 3 not found");
    }

    #[test]
    fn execution_provider_comes_from_the_environment() {
        let vars = |kv: &'static [(&'static str, &'static str)]| {
            move |k: &str| kv.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(
            ExecutionProvider::from_vars(vars(&[])).unwrap(),
            ExecutionProvider::Auto { cuda_device: 0, openvino_device: "AUTO:GPU,CPU".into() }
        );
        assert_eq!(
            ExecutionProvider::from_vars(vars(&[("LEMMALOG_ORT_EP", "Auto"), ("LEMMALOG_CUDA_DEVICE", "1"), ("LEMMALOG_OPENVINO_DEVICE", "NPU")])).unwrap(),
            ExecutionProvider::Auto { cuda_device: 1, openvino_device: "NPU".into() }
        );
        assert_eq!(ExecutionProvider::from_vars(vars(&[("LEMMALOG_ORT_EP", "cpu")])).unwrap(), ExecutionProvider::Cpu);
        if cfg!(feature = "cuda") {
            assert_eq!(ExecutionProvider::from_vars(vars(&[("LEMMALOG_ORT_EP", "CUDA")])).unwrap(), ExecutionProvider::Cuda { device: 0 });
        } else {
            assert!(ExecutionProvider::from_vars(vars(&[("LEMMALOG_ORT_EP", "cuda")])).is_err());
        }
        if cfg!(feature = "openvino") {
            assert_eq!(
                ExecutionProvider::from_vars(vars(&[("LEMMALOG_ORT_EP", "openvino")])).unwrap(),
                ExecutionProvider::OpenVino { device: "AUTO:GPU,CPU".into() }
            );
        }
        assert!(ExecutionProvider::from_vars(vars(&[("LEMMALOG_ORT_EP", "tensorrt")])).is_err());
        assert!(ExecutionProvider::from_vars(vars(&[("LEMMALOG_CUDA_DEVICE", "gpu0")])).is_err());
    }

    #[test]
    fn prompt_offset_matches_the_gline_rs_prompt() {
        // measured from gline-rs's own prompt builder, not from our format
        // string: if gline-rs changes its prompt, this fails
        assert_eq!(relation_prompt_offset(), RELATION_PROMPT_OFFSET);
        use gliner::model::input::relation::RelationInput;
        let text = "Dana is the manager of Bob.";
        let spans = SpanOutput::new(vec![text.to_string()], vec![], vec![vec![]]);
        let p = &RelationInput::from_spans(spans, &RelationSchema::new()).prompts[0];
        assert_eq!(&p[RELATION_PROMPT_OFFSET..], text);
    }

    /// What gline-rs's relation decoder emits: a span over the prompt whose
    /// class is "<subject> <> <relation>".
    fn rel(sentence: &str, subj: &str, label: &str, obj: &str, nth: usize) -> Relation {
        let at = sentence.match_indices(obj).nth(nth).unwrap().0;
        let prompt = format!("{RELATION_PROMPT_PREFIX} {sentence}");
        let (a, b) = (at + RELATION_PROMPT_OFFSET, at + RELATION_PROMPT_OFFSET + obj.len());
        assert_eq!(&prompt[a..b], obj, "offset constant matches the prompt format");
        Relation::from(Span::new(0, a, b, obj.into(), format!("{subj} <> {label}"), 0.9)).unwrap()
    }

    fn ms(sentence: &str, items: &[(&str, &str)]) -> Vec<Mention> {
        let mut out = Vec::new();
        for (text, label) in items {
            for (i, _) in sentence.match_indices(text) {
                out.push(Mention { text: (*text).into(), label: (*label).into(), start: i, end: i + text.len(), score: 0.9 });
            }
        }
        out
    }

    #[test]
    fn prompt_relative_object_spans_map_to_the_sentence() {
        let s = "Bob doesn't work at Acme.";
        let a = map_relations(s, ms(s, &[("Bob", "person"), ("Acme", "organization")]), &[rel(s, "Bob", "works at", "Acme", 0)]);
        let r = &a.relations[0];
        assert_eq!((&s[r.object.start..r.object.end], r.object.label.as_str()), ("Acme", "organization"));
        assert_eq!(&s[r.subject.start..r.subject.end], "Bob");
        assert!(a.unresolved.is_empty());
    }

    #[test]
    fn repeated_object_text_uses_the_mapped_span() {
        let s = "Acme sued Bob, and Bob works at Acme.";
        let a = map_relations(s, ms(s, &[("Bob", "person"), ("Acme", "organization")]), &[rel(s, "Bob", "works at", "Acme", 1)]);
        let r = &a.relations[0];
        assert_eq!(r.object.start, s.rfind("Acme").unwrap());
        assert_eq!(r.subject.start, s.rfind("Bob").unwrap(), "subject nearest the object");
    }

    #[test]
    fn subject_not_in_sentence_is_unresolved() {
        let s = "Bob works at Acme.";
        let a = map_relations(s, ms(s, &[("Bob", "person"), ("Acme", "organization")]), &[rel(s, "Dana", "works at", "Acme", 0)]);
        assert!(a.relations.is_empty());
        assert_eq!(a.unresolved, vec![("Dana".into(), "works at".into(), "Acme".into())]);
    }

    #[test]
    fn sentences_without_a_subject_candidate_are_detected() {
        let strict = schema_from(&[]);
        assert!(has_subject_candidate(&strict, &["organization", "location"]));
        assert!(has_subject_candidate(&strict, &["person"]));
        // e25 if "Acme" is not typed person/organization: nothing can be a subject
        assert!(!has_subject_candidate(&strict, &["location", "thing"]));
        assert!(!has_subject_candidate(&strict, &[]));
    }

    #[test]
    fn relaxed_schema_only_widens_person_to_person_objects() {
        let (strict, relaxed) = (schema_from(&[]), schema_from(RELAXED_OBJECTS));
        for (label, _, _, _, _) in DEFAULT_RELATIONS {
            let (s, r) = (&strict.relations()[*label], &relaxed.relations()[*label]);
            let widened = RELAXED_OBJECTS.iter().any(|(l, _)| l == label);
            assert_eq!(r.allows_object("organization") && !s.allows_object("organization"), widened, "{label}");
            assert!(r.allows_object("person") || !s.allows_object("person"), "{label}: relaxed keeps strict objects");
        }
    }

    #[test]
    fn third_person_spans_take_the_labels_in_use() {
        let s = "He said it hired her, and they agreed with his plan.";
        let labels: Vec<String> = ["person", "organization"].iter().map(|x| x.to_string()).collect();
        let got: Vec<(String, String)> =
            third_person_spans(s, &[], &labels).iter().map(|x| (x.text().to_string(), x.class().to_string())).collect();
        let want = [("He", "person"), ("it", "organization"), ("her", "person"), ("they", "person")];
        assert_eq!(got, want.map(|(a, b)| (a.to_string(), b.to_string())));
        // no "organization" label: "it" is not proposed
        assert_eq!(third_person_spans("It hired Bob.", &[], &["person".to_string()]).len(), 0);
    }

    #[test]
    fn first_person_spans_skip_covered_words() {
        let s = "I told Imogen about me and myself.";
        let got: Vec<_> = first_person_spans(s, &[]).iter().map(|x| x.text().to_string()).collect();
        assert_eq!(got, ["I", "me", "myself"]);
        let covered = [Span::new(0, 0, 1, "I".into(), "person".into(), 0.8)];
        assert_eq!(first_person_spans(s, &covered).len(), 2);
    }
}
