from torch.utils.data import DataLoader
from sleepwalker.datasets.Basedataset import batch_collate

from sleepwalker.utils import logger

def iterate_dataset(dataset, num_batches, batch_size = 128, preprocessors = None, device="cuda"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn= lambda x: batch_collate(x, ignore_list=["time", "patient", "target", "target_extra"]))

    cnt = 0
    assert len(loader) > 0

    logger.progress_start(total=min(num_batches, len(loader)), desc="Testing batches")
    for batch in loader:
        assert "data" in batch

        if preprocessors:
            x = batch["data"].to(device)
            for pre in preprocessors:
                pre.update(x)
                x = pre(x)

        assert "patient" in batch
        assert "time" in batch
        assert "target" in batch

        if dataset.has_extra_target():
            assert "target_extra" in batch

        cnt += 1
        logger.progress_advance(1)
        if cnt >= num_batches:
            break
            
    logger.progress_close()