import torch
from mongoose_slide.slide_lib.simHash import SimHash


class Scheduler:
    """Cheap change detector used to gate expensive learnable-LSH work.

    Paper mapping:
        MONGOOSE (ICLR 2021), Section 3.1 "Slow Change" and Section 3.2
        "A Smart Scheduler for Learnable LSH Updates".

    Implementation note:
        The paper presents a general dynamic-maintenance data structure with
        theoretical guarantees. This repository uses a much smaller practical
        proxy: cache SimHash codes for the parameter matrix and trigger when
        enough codes change. It should therefore be described as an
        implementation inspired by the paper's scheduler, not as a literal
        transcription of Algorithm 1.

    Paper:
        https://openreview.net/forum?id=wWK7yXkULyh
    """

    def __init__(self, data: torch.Tensor, D: int, k: int = 1, l: int = 10, thresh: float = 0.01) -> None:
        self.thresh_hash = SimHash(D, k, l)
        self.hash_codes = self.thresh_hash.hash(data)
        self.thresh = thresh

    def detect_change(self, updated_data: torch.Tensor) -> bool:
        """Return True when the cached SimHash signature changes enough.

        The threshold is applied to the absolute difference between current
        and cached hash-code tensors. A True result also advances the cached
        signature. In the Reformer integration this return value gates whether
        triplet examples / learnable-hash updates are computed.

        Traceability test: MONGOOSE-SCHEDULER-CHANGE.\n\n        This is a code-level approximation of MONGOOSE Section 3.2's goal:
        avoid LSH maintenance when model parameters have changed too little to
        justify the update.
        """
        check = self.thresh_hash.hash(updated_data)
        distance = check - self.hash_codes
        if torch.sum(torch.abs(distance)) > self.thresh*distance.numel():
            self.hash_codes = check
            return True
        else:
            return False
