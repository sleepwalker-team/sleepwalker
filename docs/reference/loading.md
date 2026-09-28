# Dataset internals

Low-level objects that dataset classes build on. Most users work with the higher-level [Python API](api.md); these are documented for contributors implementing or debugging a dataset. `ChannelConfig` and `BaseDataset` are covered in the [Python API](api.md) and are not repeated here.

## `EDFFile`

::: sleepwalker.datasets.Basedataset.EDFFile

## `EventIndex`

::: sleepwalker.datasets.Basedataset.EventIndex

## `EDFCache` {#edf-cache}

::: sleepwalker.datasets.EDFCache.EDFCache
    options:
      show_root_heading: false
      members:
        - acquire
        - store_frame
        - attach
        - copy_window
        - close_local
        - close

### `EDFCacheEntry` {#edf-cache-entry}

::: sleepwalker.datasets.EDFCache.EDFCacheEntry
    options:
      show_root_heading: false
      members:
        - copy_window
