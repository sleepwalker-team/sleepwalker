# Bounded signal reads

`GET /api/v1/sessions/{session_id}/signals/{signal_id}` passes `max_points`
to the signal reader. The EDF reader processes at most 65,536 source samples
per read, retains the first and last sample, and selects the strongest
absolute value in each interior bucket. Ties keep the first occurrence.
The selection matches the existing `downsample_extrema` algorithm.

Source values and timestamps for the whole requested recording are no longer
kept in memory. Working memory scales with one source chunk plus the output
budget, rather than with the total recording duration. `max_points` is
limited to 2–100,000. The response's `downsampled` flag compares the output
count with the original sample count, even when reduction happens in the
reader.

Finding bucket extrema still scans the requested samples, so an entire-night
query can remain disk/CPU intensive. This change bounds memory; it does not
introduce a cache or an index. When only two points are requested, the reader
can read the two boundaries directly.
