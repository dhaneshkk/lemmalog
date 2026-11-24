# 7. Fact extraction without an LLM

Lemmalog turns each conversation episode into candidate facts
`subject --predicate--> object` through the `Extractor` trait. The LLM
extractors (`LlmExtractor`, `LlmClientExtractor`) ask a chat model for the
facts. `GlinerExtractor` (`src/extract/`) replaces that call with three
deterministic stages:

1. **gline-rs** runs a GLiNER multitask model (ONNX, token mode) for named
   entities and relations, one sentence per call. gline-rs reports relation
   object spans against its prompt (`"{prefix} {sentence}"`); the backend
   maps them back to the sentence and ties subject and object to that
   sentence's NER spans (`resolve_relations`). Feature `gliner`.
2. **NegEx** drops relations whose object sits inside a negation scope.
3. **glinker-rs**, the Rust port of GLinker, links each mention to a canonical
   entity id in-process. It runs `Executor::link_with_mentions`, which skips L1
   because gline-rs already found the spans. Feature `glinker`. Tests use the
   deterministic `TableLinker`.

`GlinerExtractor` is the default extractor for free text
(`extract::setup::default_extractor`): `lemmalog_observe`, `lemmalog-cli
observe` and `lemmalog-bench ingest` use it, loading the model on first
use; the LLM extractors are opt-in (`LEMMALOG_EXTRACTOR=llm` in the
bench). With no entity database configured, the linker is `NameLinker`,
whose canonical id is the normalised name, so extracted facts join
line-protocol facts about the same entity.

The obligations below are what this change promises. Each one has
tagged Rust tests (`// obligation: <id>`) under `tests/`. `obligations.lock`
on `main` pins the obligations, the tests, the helpers they use and the data
files an obligation lists under `fixtures:`, so a later PR cannot weaken
them unnoticed (dropping hard sentences from a labelled set included).
CI runs `scripts/check-obligations.py`, which fails while the lock is
stale, and the approver re-locks with its `build` mode.

## 7.1 Obligations

```yaml
obligations:
  - id: SPEC-7.1-O1
    says: >
      A relation whose object is governed by a negation trigger produces no
      fact and is reported in rejected() with reason Negated. A trigger is a
      pre-negation trigger (including cessation words such as "left",
      "resigned", "former" and contractions typed without the apostrophe)
      that starts at or after the subject and ends within the window before
      the object; a post-negation trigger within the window after the
      object; or a sentential trigger ("it's not true that") anywhere before
      the object. A termination term ("but", ";", ".") between trigger and
      object ends its scope, and so does another entity mention between them
      unless it is coordinated with the object ("Acme, Initech or Globex").
      A pre-negation trigger inside a comma-enclosed aside after the subject
      ("Bob, who has no car, works at Acme") does not govern the object.
      "neither", "nor" or "not" directly before the subject negates the
      relation ("Neither Bob nor Dana works at Acme"; "Not Bob but Dana",
      for Bob). An adverbial termination term ("however", "now", "still")
      does not end a scope inside a comma-enclosed aside (", however,") or
      directly after a trigger, aside or not ("does not now work"). A
      relative pronoun ("which", "who") between object and post-trigger
      ends the post-trigger's scope ("works at Acme, which is not true of
      Dana"). "left" followed by a determiner and a noun that is not a job
      or an organisation ("left his laptop at Acme") is not cessation; "left
      his job", "left the company" are. A likes relation with a dislike
      verb ("dislikes", "hates", "can't stand") between subject and object
      is also rejected as Negated.
      Pseudo-triggers do not negate: phrases such as "not only" and
      "no doubt(s)", negated-positive idioms ("doesn't mind", "didn't
      hesitate") and double negatives ("never stopped", "hasn't left").
      On the labelled sets tests/negex_tuning.tsv and tests/negex_heldout.tsv
      NegEx keeps MCC at or above 0.90 and 0.85.
    kind: property
    risk: high
    scope:
      - src/extract/negex.rs::negated
      - src/extract/negex.rs::negated_with
      - src/extract/negex.rs::dislikes_between
      - src/extract/pipeline.rs::run
    fixtures:
      - tests/negex_tuning.tsv
      - tests/negex_heldout.tsv
  - id: SPEC-7.1-O2
    says: >
      Every emitted fact's subject and object are canonical ids returned by the
      EntityLinker for that mention, never the raw mention text. A relation
      with a mention that does not link produces no fact and is reported with
      reason Unlinked. A relation the model reports whose subject or object
      is not a mention of its sentence produces no fact and is reported with
      reason Unresolved. First-person mentions (I, me, my, myself), as
      subject or object, are linked as the episode's speaker: the speaker's
      name is linked once per episode, on its own, and pronouns never reach
      the linker. A leading title ("Dr.", "Mrs") is stripped from a mention
      before linking. TableLinker links an alias only when exactly one
      entity claims it for the mention's label; CascadeLinker links with its
      first linker and asks the second only for what the first left
      unlinked.
    kind: contract
    risk: high
    scope:
      - src/extract/pipeline.rs::run
      - src/extract/pipeline.rs::resolve_relations
      - src/extract/link.rs
      - src/extract/gline.rs
  - id: SPEC-7.1-O3
    says: >
      Extraction is deterministic and memoized, and calls no LLM. The
      backend runs at most once per episode (id, text and speaker) that
      it analysed successfully, and extracting the same episode again
      returns identical facts; an episode whose text or speaker changed
      under the same id is extracted again. A failed episode is not
      memoized: extracting it again retries the backend. A relation with
      a NaN or infinite relation or link score is rejected with reason
      LowScore. A fact's confidence is the minimum of its relation score
      and its two link scores. Output holds one fact per (subject,
      predicate, object), with the highest confidence, sorted by that
      triple. A backend error yields no facts for the sentences it
      covers, counts as one failure per episode in stats(), and keeps its
      message, with the episode id, in errors(); when a batch of several
      sentences fails, each sentence is retried alone, so one bad
      sentence costs only its own facts. The memo, rejected() and
      errors() keep at most memo_capacity episodes and log_capacity
      entries (newest kept), and CachedBackend at most capacity
      sentences. The GLiNER backend skips relation extraction for a
      sentence with no entity the schema allows as a subject (gline-rs
      rejects an empty label list) and still returns that sentence's
      mentions.
    kind: property
    risk: medium
    scope:
      - src/extract/pipeline.rs::extract
      - src/extract/pipeline.rs::run
      - src/extract/mod.rs
      - src/extract/gline.rs
  - id: SPEC-7.1-O4
    says: >
      Facts follow lemmalog's predicate direction. X --manager--> Y means X
      reports to Y: "reports to" relations keep their direction and "manager
      of" relations are inverted, so "Dana is the manager of Bob" and "Bob
      reports to Dana" both give person:bob --manager--> person:dana. One
      sentence never yields a fact in both directions: when it would, the
      fact whose relation label is written between its subject and its
      object ("Sam reports to Priya") wins, then the one with the higher
      relation score, then the smaller triple; the other is reported with
      reason Reversed. A relation whose subject and object link to the same
      entity produces no fact and is reported with reason SelfLoop. Link
      scores never decide the direction. Facts from different sentences are
      not compared.
    kind: contract
    risk: high
    scope:
      - src/extract/pipeline.rs::DEFAULT_RELATIONS
      - src/extract/pipeline.rs::run
  - id: SPEC-7.1-O5
    says: >
      Negation and linking only ever see mention spans that are non-empty,
      valid byte ranges of their own sentence, and a mention's text is the
      sentence text at its span, so negation and linking judge the same
      words. A relation with any other span produces no fact and is reported
      with reason BadSpan; no span makes extraction panic. Object spans from
      gline-rs are mapped back with the sentence offset measured from the
      prompt gline-rs builds. Sentences are not split inside numbers
      ("2.5"), after titles ("Dr.", "St. Louis"), or after abbreviations and
      initials ("Corp.", "U.S.", "Jr.", "Main St.") followed by a lower-case
      word, nor at an ellipsis followed by one; "Jr.", "Sr." and "Main St."
      followed by a capitalised word end the sentence.
    kind: property
    risk: high
    scope:
      - src/extract/pipeline.rs::run
      - src/extract/pipeline.rs::split_sentences
      - src/extract/gline.rs::map_relations
      - src/extract/gline.rs::relation_prompt_offset
      - src/extract/link.rs::GlinkerLinker
  - id: SPEC-7.1-O6
    says: >
      A relation that is only possible, planned or conditional produces no
      fact and is reported with reason Hedged. It is hedged when a modal or
      intent word ("might", "may", "will", "would", "plans to", "wants to",
      "is considering", "applied to") governs the object with the scope rules
      of a pre-negation trigger (after the subject, within the window, not
      past a termination term, another uncoordinated mention or out of a
      comma-enclosed aside), or a conditional ("if", "whether", "unless")
      precedes the object with no termination term between. "May" written
      with a capital past the first word is the month, not a modal. Negation
      is checked first, so a negated relation is reported as Negated.
    kind: property
    risk: high
    scope:
      - src/extract/negex.rs::hedged_with
      - src/extract/pipeline.rs::run
  - id: SPEC-7.1-O7
    says: >
      Third-person pronouns and short names are resolved within an episode
      by ordered Datalog sieves (extract::coref::SIEVE_RULES, run on the
      engine, so each link has a why proof): exact name match, acronym,
      head word (the first or last word of an earlier longer name with the
      same label, never a generic word such as "Corp"), then pronouns ("he",
      "she" take a person; "it" an organisation, team, place or thing;
      "they" a person, organisation or team) from the nearest sentence, up
      to two back, that has a compatible earlier mention. An antecedent
      always precedes its mention, and a mention resolves only when its
      sieve's candidates are one entity (a name and the names resolved to
      it count as one). A resolved mention is linked as its antecedent, in
      the antecedent's sentence. A relation whose pronoun has candidates
      that are more than one entity, or none, produces no fact and is
      reported with reason Coreference; an ambiguous short name stands for
      itself. First-person pronouns remain the speaker (SPEC-7.1-O2).
    kind: property
    risk: high
    scope:
      - src/extract/coref.rs::resolve
      - src/extract/coref.rs::SIEVE_RULES
      - src/extract/pipeline.rs::run
      - src/extract/gline.rs::third_person_spans
```

## 7.2 Out of scope

- Coreference for definite descriptions ("the company", "my manager"),
  gender agreement for "he"/"she" (two people in range stay ambiguous),
  and cataphora ("Before he joined Acme, Bob ...").
- Temporal negation ("used to work at") and recording when a relation ended.
  Cessation words ("left Acme", "former") only stop the stale fact from being
  asserted; the end time is a supersession, so it belongs to the bi-temporal
  layer rather than to extraction.
- Coreference for "we", "our" and other plural first-person forms.
- Recording hedged relations as possibilities (SPEC-7.1-O6 only keeps them
  out of the facts), and hedges expressed without a trigger word ("Bob's
  move to Acme is on the table").
- Accuracy targets for the ONNX models and the entity database. Measure those
  with MCC on a labelled set: `examples/extraction_eval.rs` over
  `benchmarks/extraction_gold.tsv` scores the whole pipeline, once with the
  gold alias table (GLiNER + NegEx alone) and once with glinker-rs, and
  `notebooks/extraction_eval.ipynb` runs it in Colab with the real models.
  They are not pass/fail checks for this gate.
