import joblib
from typing import Optional

from sequentia.utils.validation import Array

def _effective_n_jobs(n_jobs: int, lengths: Optional[Array[int]] = None) -> int:
    n_jobs_ = 1
    if lengths is not None:
        n_jobs_ = min(joblib.effective_n_jobs(n_jobs), len(lengths))
    return n_jobs_
