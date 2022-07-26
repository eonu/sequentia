import joblib
from typing import Optional

from .validation import Array

def effective_n_jobs(n_jobs: int, lengths: Optional[Array[int]] = None) -> int:
    n_jobs_ = 1
    if lengths is not None:
        n_jobs_ = min(joblib.effective_n_jobs(n_jobs), len(lengths))
    return n_jobs_
