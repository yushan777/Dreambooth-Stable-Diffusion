from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Callable, Dict, Optional

from typing import Union
from pathlib import Path
_PATH = Union[str, Path]

from ldm.pruner import prune_checkpoint
 
class PruningCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(
            self, 
            checkpoint: Dict[str, Any], 
            path: _PATH, 
            storage_options: Optional[Any] = None
        ) -> None:
        pruned_checkpoint = prune_checkpoint(checkpoint)
        TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)
