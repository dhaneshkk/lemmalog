//! LLM-free fact extraction: gline-rs (GLiNER NER + relation extraction),
//! NegEx negation detection and GLinker entity linking.
//!
//! See `spec/extraction.md` for the obligations this module must meet.

pub mod eval;
pub mod link;
pub mod negex;
pub mod coref;
pub mod pipeline;
pub mod setup;
#[cfg(feature = "gliner")]
pub mod gline;

pub use link::{CascadeLinker, EntityLinker, Link, NameLinker, TableLinker};
pub use setup::{default_extractor, GlinerConfig, LazyExtractor};
#[cfg(feature = "glinker")]
pub use link::GlinkerLinker;
pub use negex::NegEx;
pub use pipeline::{
    resolve_relations, split_sentences, strip_title, worded_between, CachedBackend, FixtureBackend, GlinerExtractor, Mention, ModelRelation,
    RawRelation, RejectReason, Rejection, RelationBackend, SentenceAnalysis, DEFAULT_RELATIONS,
};
