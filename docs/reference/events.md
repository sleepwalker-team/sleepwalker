# Shared annotation contract

`sleepwalker.datasets.events` provides validation and single-table interval operations for shared event preparation. Event-preparation callbacks live in `sleepwalker.training.callbacks`. Event names are extensible strings. This catalogue defines recognized meanings; it is not a closed enum and does not require every reader to provide every event or field.

Labelled datasets require an explicit ordered `annotation_labels` declaration. The optional `prepare_events(label_df, label_extra_df)` callback receives complete timestamped copies of both available reader tables once per recording, including optional metadata. `None` means identity processing; it does not disable loading. Return `(primary, alternate)` to retain the sources or `None` to reject the recording. Preserve alternate-source availability: an unavailable alternate remains `None`, while an available empty table remains a table. An empty prepared primary excludes the recording; exceptions and invalid schemas are errors.

The lifecycle is reader output validation against `supported_events()`, event preparation, validation against `annotation_labels`, core-column projection, `prepare_patient`, retained-table validation, then bounds/observed-label computation and window indexing. Label declarations never shrink to the observed union. `dataset.observed_labels` records the retained cohort observations separately. YAML training entries declare `output_classes` independently, in model/target order; mask labels need not be model outputs, and an explicit target fallback can supply a negative without an event interval.

`load_annotations=False` skips readers and annotation-only hooks and rejects configured annotation vocabularies/callbacks. `UnlabelledDataset.from_dataset(...)` copies the signal pipeline and omits annotation handling. Run metadata retains the labelled recipe while raw inference packages use the unlabelled template.

Every concrete reader declares `supported_events()` independently of patient selection or task mapping. Returned raw labels must belong to that declaration. A supported type can have no rows in a recording; this says nothing about whether it was scored. Measurement requirements belong to processing recipes: supporting desaturation does not guarantee an amplitude column or measurements for every event.

ISRUC exposes its annotators independently by default (`merge=False`). Its historical reader intersection remains an explicit `merge=True` source-selection choice. Unavailable alternate sidecars remain `None`; available empty tables retain their availability.

## Temporal core

Each row represents one event on the recording clock. `Label`, `Starttime` and `Endtime` are required, including on empty tables. Timestamps must have pandas datetime dtypes, no missing values, matching timezones and `Endtime > Starttime`. Intervals are half-open: `Starttime <= t < Endtime`. Readers decode source offsets, rollovers and change markers before returning this table. Overlapping and duplicate intervals are allowed; row order/index has no semantic meaning.

`validate_event_table(events, allowed_labels=..., context=...)` checks the core and an optional declared vocabulary. It never drops rows because optional metadata is missing. `validate_event_labels(...)` preserves declared order and rejects duplicates/empty strings. These checks do not infer which concepts were scored or which measurements a reader can provide.

## Recognized event names

| Name | Meaning |
| --- | --- |
| `wake` | Scored wakefulness. |
| `n1`, `n2`, `n3`, `rem` | Scored sleep stages. Source stage 4 belongs to n3 where the source convention supports that conversion. |
| `unscored` | An explicit unscored interval; never an implicit negative event. |
| `arousal` | Annotated arousal. |
| `obstructive_apnea`, `central_apnea`, `mixed_apnea` | Apnea with an explicit source subtype. |
| `apnea` | Apnea whose subtype is unspecified. |
| `hypopnea` | Annotated hypopnea before association or task relabeling. |
| `desaturation` | Annotated oxygen desaturation; does not imply measured amplitude is available. |
| `limb_movement` | Annotated limb movement. |
| `periodic_limb_movement` | Source explicitly identifies periodic limb movements. |
| `movement` | A general movement annotation or movement-coded stage; no limb specificity is inferred. |
| `rera` | Source explicitly identifies a respiratory effort-related arousal. |
| `periodic_breathing`, `cheyne_stokes` | Explicit source respiratory patterns, retained as distinct concepts. |
| `respiratory_artifact`, `spo2_artifact` | Source-annotated respiratory/oximetry artifacts. |
| `body_position_left`, `body_position_right`, `body_position_supine`, `body_position_prone`, `body_position_upright` | Explicit body-position intervals, expanded from change markers under the source convention when necessary. |
| `device_settings` | An explicit device-settings annotation; its absence does not establish non-use of a device. |
| `other` | An unclassified source event. Preserve its original source text for investigation. |

A reader can document additional source concepts without guessing physiology. Task labels such as `sleep`, `respiratory_event` and `hypopnea_unknown` belong to the transformed annotation vocabulary, not necessarily the reader's capabilities. Model output classes are separately declared; an implicit negative can be an output class without an annotation interval.

## Optional field catalogue

| Field | Type and units | Meaning and applicable events | Missing-value behavior |
| --- | --- | --- | --- |
| `desaturation_drop_pp` | Nullable numeric scalar, percentage points. | Measured SpO2 drop for `desaturation`. A fall from 96% to 92% is 4.0. | Omit the column when a recording's source cannot supply measurements; individual unavailable/irrelevant values are missing. Never fabricate zeros. |
| `SourceLabel` | Nullable string scalar, no unit. | Original source text for any event, retained for diagnostics/provenance. | Missing when unavailable. Shared physiological algorithms must not parse it. |

Consumers validate the optional fields they need and distinguish an absent column from a missing per-event value. Readers perform source-unit conversion and cheap extraction during annotation parsing. Waveform-derived measurements require an explicit processing operation. Add future fields here with type, units, meaning, applicable events and missing-value behavior. Duration is derived from the temporal core.

## Mapping and sleep policies

`sleepwalker.training.callbacks.map_events(label_df, label_extra_df, *, mapping, remove_unmapped=True)` applies the same dictionary to both available annotators independently. It is also exported through `sleepwalker.datasets.utils.map_events` for configuration use. It preserves metadata and returns independent tables. With `remove_unmapped=False`, unmatched labels retain their meaning. An unavailable alternate remains `None`; an empty available alternate remains an empty table. Mapping neither creates fallback intervals nor discovers class order.

`merge_event_intervals(events, labels=...)` unions overlapping intervals for each selected label. Touching endpoints remain separate by default. Optional scalar metadata survives only when all contributing rows agree, including missingness; conflicting values become missing rather than inheriting an arbitrary measurement. The result retains unselected events individually and sorts chronologically. Merged duration is always derived from the resulting endpoints.

`exclude_events_on_wake(events, labels=...)` removes entire selected events that overlap `wake`. An empty wake set removes nothing. `touching_counts_as_overlap=True` explicitly enables Ruhrland's historical endpoint policy in both helpers.

`sleepwalker.training.callbacks.prepare_sleep_events(...)` composes merging, wake exclusion and mapping in that order, separately for each annotator. Its parameters are `mapping`, `merge_labels`, `exclude_on_wake`, `touching_counts_as_overlap` and `remove_unmapped`. Ruhrland recipes configure these policies explicitly instead of applying them inside the spreadsheet reader. Both primary and alternate annotators now undergo the configured sequence; historical primary-only Ruhrland preprocessing is therefore a protocol difference.

SHHS/MrOS/ABC and the other NSRR/Profusion readers expose common physiological concepts and retain original labels. NSRR baseline/nadir and Profusion drop fields follow the [NSRR translator's documented meanings](https://github.com/nsrr/edf-editor-translator/wiki/Tutorial-2-EDF-Annotation-Translator), with drops expressed in percentage points. HSP extracts explicit `drop ... %` measurements before task mapping. Missing columns remain absent; missing measurements within present columns remain nullable. Unbounded HSP stage changes and instantaneous markers do not become invented intervals.

## Retained intervals

`EventIndex.events` retains an independent copy of the fully prepared DataFrame supplied to its constructor, preserving individual intervals, overlaps, duplicates, metadata, row order/index, datetime precision and timezone. Changing the caller's input table afterward does not change the retained table or target sampling. Treat the retained table as read-only and use `index.events.copy()` before modifying it for analysis, because the sampling arrays were built from it during construction. The selected recording bounds are on its containing `EDFFile.start_date` / `end_date`; clipping to prediction coverage belongs to the evaluator. Access neither rereads raw annotations nor repeats preparation callbacks.

The annotation contract itself is an ordinary DataFrame with the agreed labels and timestamps. `get_event_df()` decodes source annotations; mapping/filtering/trimming then produces the fully prepared DataFrame passed to `EventIndex`. The index stores that table directly alongside paired endpoint arrays for sampling many windows. There is no conversion method or interval reconstruction step.
