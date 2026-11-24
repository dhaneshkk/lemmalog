//! The default extractor: LLM-free extraction (gline-rs + NegEx + a linker),
//! configured from the environment and loaded on first use.
//!
//! ```text
//! LEMMALOG_EXTRACTOR        gliner (default) | off
//! LEMMALOG_GLINER_DIR       model dir with tokenizer.json and onnx/model.onnx
//!                           (default ~/.lemmalog/models/gliner-multitask-large-v0.5)
//! LEMMALOG_GLINKER_PIPELINE optional glinker-rs pipeline.json; without it,
//!                           entities link by name (NameLinker)
//! ORT_DYLIB_PATH            ONNX Runtime shared library, for the default
//!                           load-dynamic build (default
//!                           ~/.lemmalog/lib/libonnxruntime.so, .dylib on
//!                           macOS, if it exists; otherwise next to the
//!                           executable, then the system library path)
//! LEMMALOG_ORT_EP           auto (default: CUDA, then OpenVINO, then CPU,
//!                           whichever the loaded ONNX Runtime can run),
//!                           cuda, openvino or cpu; see gline::ExecutionProvider
//! LEMMALOG_CUDA_DEVICE      CUDA device number (default 0)
//! LEMMALOG_OPENVINO_DEVICE  AUTO:GPU,CPU (default), CPU, GPU, NPU, ...
//! ```
//!
//! The GLiNER backend needs the `gliner` feature (and `glinker` for a
//! GLinker pipeline). Built without it, or with the model missing, the
//! default extractor is *off*: line-protocol facts still work, free text is
//! not extracted, and [`LazyExtractor::status`] says why.

use crate::agent::{CandidateFact, Episode, Extractor};

/// Builds the real extractor on first use.
pub type ExtractorInit = Box<dyn FnOnce() -> Result<Box<dyn Extractor>, String>>;

/// An extractor that is built the first time it is needed, so a server
/// that only ever receives line-protocol facts never loads a model, and a
/// slow model load does not delay start-up. A failed build is reported by
/// [`error`](Self::error) and counted as a failure; nothing is extracted.
pub struct LazyExtractor {
    init: Option<ExtractorInit>,
    inner: Option<Box<dyn Extractor>>,
    error: Option<String>,
    status: String,
}

impl LazyExtractor {
    /// An extractor that extracts nothing; `why` is reported by `status()`.
    pub fn off(why: impl Into<String>) -> Self {
        LazyExtractor { init: None, inner: None, error: None, status: format!("off: {}", why.into()) }
    }

    /// Build with `init` on first use; `status` describes what it will be.
    pub fn new(status: impl Into<String>, init: ExtractorInit) -> Self {
        LazyExtractor { init: Some(init), inner: None, error: None, status: status.into() }
    }

    /// Wrap an extractor that is already built.
    pub fn ready(status: impl Into<String>, inner: Box<dyn Extractor>) -> Self {
        LazyExtractor { init: None, inner: Some(inner), error: None, status: status.into() }
    }

    /// Will this extractor (try to) extract free text?
    pub fn enabled(&self) -> bool {
        self.inner.is_some() || self.init.is_some()
    }

    /// Why building the extractor failed, if it did.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// One line on what this extractor is ("gliner: ...", "off: ...").
    pub fn status(&self) -> String {
        match &self.error {
            Some(e) => format!("failed: {e}"),
            None => self.status.clone(),
        }
    }

    fn force(&mut self) -> Option<&mut Box<dyn Extractor>> {
        if let Some(init) = self.init.take() {
            match init() {
                Ok(x) => self.inner = Some(x),
                Err(e) => self.error = Some(e),
            }
        }
        self.inner.as_mut()
    }
}

impl Extractor for LazyExtractor {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        self.force().map(|x| x.extract(episode)).unwrap_or_default()
    }

    fn stats(&self) -> (usize, usize) {
        let (calls, failures) = self.inner.as_ref().map_or((0, 0), |x| x.stats());
        (calls, failures + usize::from(self.error.is_some()))
    }
}

/// Where the default extractor's model files are.
#[derive(Debug, Clone, PartialEq)]
pub struct GlinerConfig {
    pub tokenizer: String,
    pub model: String,
    pub glinker_pipeline: Option<String>,
    /// The ONNX Runtime shared library to load (load-dynamic builds):
    /// ORT_DYLIB_PATH, else ~/.lemmalog/lib's copy if it exists, else `None`
    /// (ort then looks next to the executable and on the system path).
    pub ort_dylib: Option<String>,
}

/// The ONNX Runtime library file name on this platform.
pub const ORT_DYLIB_NAME: &str = if cfg!(target_os = "windows") {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

impl GlinerConfig {
    pub const DEFAULT_MODEL: &'static str = "gliner-multitask-large-v0.5";

    /// The configuration from the environment, or `Err(reason)` when the
    /// default extractor is switched off with `LEMMALOG_EXTRACTOR`.
    pub fn from_env() -> Result<GlinerConfig, String> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// [`from_env`](Self::from_env) over any variable lookup (for tests).
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Result<GlinerConfig, String> {
        match var("LEMMALOG_EXTRACTOR").unwrap_or_default().trim().to_lowercase().as_str() {
            "" | "gliner" => {}
            "off" | "none" | "protocol" => return Err("LEMMALOG_EXTRACTOR=off".into()),
            other => return Err(format!("unknown LEMMALOG_EXTRACTOR={other:?} (use gliner or off)")),
        }
        let dir = var("LEMMALOG_GLINER_DIR").filter(|d| !d.is_empty()).unwrap_or_else(|| {
            let home = var("HOME").unwrap_or_else(|| ".".into());
            format!("{home}/.lemmalog/models/{}", Self::DEFAULT_MODEL)
        });
        let home = var("HOME").unwrap_or_else(|| ".".into());
        let bundled = format!("{home}/.lemmalog/lib/{ORT_DYLIB_NAME}");
        let ort_dylib = var("ORT_DYLIB_PATH")
            .filter(|p| !p.is_empty())
            .or_else(|| std::path::Path::new(&bundled).exists().then_some(bundled));
        Ok(GlinerConfig {
            tokenizer: format!("{dir}/tokenizer.json"),
            model: format!("{dir}/onnx/model.onnx"),
            glinker_pipeline: var("LEMMALOG_GLINKER_PIPELINE").filter(|p| !p.is_empty()),
            ort_dylib,
        })
    }

    /// Files this configuration needs that do not exist. The ONNX Runtime
    /// library counts only when it was named: with no path, ort searches.
    pub fn missing(&self) -> Vec<String> {
        let dylib = if cfg!(feature = "gliner-load-dynamic") { self.ort_dylib.as_ref() } else { None };
        [Some(&self.tokenizer), Some(&self.model), self.glinker_pipeline.as_ref(), dylib]
            .into_iter()
            .flatten()
            .filter(|p| !std::path::Path::new(p.as_str()).exists())
            .cloned()
            .collect()
    }
}

/// The default extractor: GLiNER + NegEx + linker, built lazily, or off
/// (with the reason in `status()`) when it is switched off, not compiled in,
/// or its model files are missing.
pub fn default_extractor() -> LazyExtractor {
    let cfg = match GlinerConfig::from_env() {
        Ok(c) => c,
        Err(why) => return LazyExtractor::off(why),
    };
    if !cfg!(feature = "gliner") {
        return LazyExtractor::off("built without the `gliner` feature (rebuild with --features gliner,glinker)");
    }
    let missing = cfg.missing();
    if !missing.is_empty() {
        return LazyExtractor::off(format!("model files not found: {} (set LEMMALOG_GLINER_DIR)", missing.join(", ")));
    }
    let linker = cfg.glinker_pipeline.as_deref().unwrap_or("names");
    let runtime = if cfg!(feature = "gliner-load-dynamic") {
        format!(", onnxruntime: {}", cfg.ort_dylib.as_deref().unwrap_or("searched next to the binary, then the system path"))
    } else {
        String::new()
    };
    let status = format!("gliner: {} (linker: {linker}{runtime})", cfg.model);
    LazyExtractor::new(status, Box::new(move || build_gliner(&cfg)))
}

/// Load the GLiNER backend and the linker now.
#[cfg(feature = "gliner")]
pub fn build_gliner(cfg: &GlinerConfig) -> Result<Box<dyn Extractor>, String> {
    use super::gline::GlineBackend;
    use super::link::{EntityLinker, NameLinker};
    use super::GlinerExtractor;
    // ort reads ORT_DYLIB_PATH once, on first use: set it before the model
    // loads when the library was found in ~/.lemmalog/lib
    if let Some(p) = &cfg.ort_dylib {
        if std::env::var_os("ORT_DYLIB_PATH").is_none() {
            std::env::set_var("ORT_DYLIB_PATH", p);
        }
    }
    // ort panics (rather than returning an error) when it cannot load the
    // ONNX Runtime library; report that as a failed load, not a crash
    // (the default hook would print the panic and a backtrace to stderr
    // first; the message is kept in the error instead)
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let loaded = std::panic::catch_unwind(|| GlineBackend::new(&cfg.tokenizer, &cfg.model));
    std::panic::set_hook(hook);
    let backend = loaded
        .map_err(|p| {
            let why = p
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| p.downcast_ref::<&str>().copied())
                .unwrap_or("panic while loading the model");
            format!("could not load ONNX Runtime or the model: {why}")
        })??;
    let linker: Box<dyn EntityLinker> = match &cfg.glinker_pipeline {
        None => Box::new(NameLinker),
        #[cfg(feature = "glinker")]
        Some(p) => Box::new(super::link::GlinkerLinker::from_config_file(p)?),
        #[cfg(not(feature = "glinker"))]
        Some(_) => return Err("LEMMALOG_GLINKER_PIPELINE is set but lemmalog was built without the `glinker` feature".into()),
    };
    // memory derives coreference identity from link facts (SPEC-7.1-O8)
    Ok(Box::new(GlinerExtractor::new(backend, linker).with_coref_facts(true)))
}

#[cfg(not(feature = "gliner"))]
pub fn build_gliner(_cfg: &GlinerConfig) -> Result<Box<dyn Extractor>, String> {
    Err("lemmalog was built without the `gliner` feature".into())
}
