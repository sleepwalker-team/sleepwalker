# Metrics And Confusion Tables

This page documents the small metric/display layer used by the trainers.

## Metric Helpers

`sleepwalker.trainer.utils.metrics` currently provides:

- Cohen's kappa from a confusion matrix
- micro- and macro-averaged F1 from a confusion matrix

These helpers are used by both multiclass and multitask trainers.

## Display Helpers

`sleepwalker.trainer.utils.display` currently provides:

- `format_confusion_table(...)`
- `render_confusion_table_grid(...)`

These functions render confusion matrices and a small metric summary as aligned text blocks for logger output.

## Scope

This layer is intentionally small:

- metric computation from already-aggregated confusion matrices
- text formatting for logs

It is not a general reporting framework.
