"""
Contextual injection of "retrieved" sentences into BBQ prompts.

Two interchangeable strategies, both exposing the same interface:
    - RandomInjector:    samples CrowS-Pairs sentences at random (simulated RAG).
    - RetrieverInjector: asks a real retriever for the top-k sentences.

Both add an 'injected_context' column to the BBQ dataframe, which is then
placed in the prompt exactly as in `predict_row` of 5_optimize_coeff.py.
"""
import os
from typing import List, Optional, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_new import CROWS_AXIS_MAP, SEED

INJECTION_HEADER = "Retrieved Information:\n"


# ----------------------------------------------------------------------------
# Retriever interface (implemented in a separate file)
# ----------------------------------------------------------------------------
class Retriever(Protocol):
    """Minimal contract a retriever must satisfy.

    `retrieve` is mandatory. `retrieve_batch(queries, k, axis)` is optional:
    if present it is used instead of calling `retrieve` once per query.
    """

    def retrieve(self, query: str, k: int, axis: Optional[str] = None) -> List[str]:
        """Return at most k sentences (plain strings), best first."""
        ...


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def format_injected_block(sentences):
    """Build the text block placed before the BBQ question."""
    injected_str = INJECTION_HEADER
    for sent in sentences:
        injected_str += f"- {sent}\n"
    return injected_str + "\n"


def inject_crows_bias_to_df(bbq_df, crows_df, axis,
                            num_sentences=4, bias_ratio=0.5):
    """Random CrowS injection (moved verbatim from 5_optimize_coeff.py).

    Behavior is unchanged, so contexts are identical to the ones produced by
    script 5 for the same data, k, bias_ratio and SEED.
    """
    if num_sentences <= 0:
        bbq_df = bbq_df.copy()
        bbq_df['injected_context'] = ""  # no contextual injections
        return bbq_df

    crows_axis = CROWS_AXIS_MAP.get(axis, axis)

    crows_filtered = crows_df[crows_df['bias_type'] == crows_axis]
    if crows_filtered.empty:
        print(f"[WARNING] No CrowS '{crows_axis}' for BBQ '{axis}' axis. "
              f"Fallback on the full dataset.")
        crows_filtered = crows_df

    num_stereo = int(round(num_sentences * bias_ratio))
    num_anti = num_sentences - num_stereo
    rng = np.random.default_rng(SEED)  # random generator

    injected_contexts = []

    for _ in range(len(bbq_df)):
        sentences_to_inject = []

        if num_stereo > 0:
            stereo_rows = crows_filtered.sample(n=num_stereo, replace=True, random_state=rng)
            sentences_to_inject.extend(stereo_rows['sent_more'].tolist())

        if num_anti > 0:
            anti_rows = crows_filtered.sample(n=num_anti, replace=True, random_state=rng)
            sentences_to_inject.extend(anti_rows['sent_less'].tolist())

        order = rng.permutation(len(sentences_to_inject))
        sentences_to_inject = [sentences_to_inject[i] for i in order]

        injected_contexts.append(format_injected_block(sentences_to_inject))

    bbq_df = bbq_df.copy()
    bbq_df['injected_context'] = injected_contexts
    return bbq_df


# ----------------------------------------------------------------------------
# Injectors
# ----------------------------------------------------------------------------
class BaseInjector:
    mode = "base"
    cache_dir = None  # optional: set from outside (e.g. "../cache/<experiment>") to enable caching by name

    def __init__(self, k):
        self.k = k

    @property
    def subdir(self):
        """Folder name used to keep results of different injections apart."""
        raise NotImplementedError

    def describe(self):
        """Metadata to store in the result rows."""
        return {'injection_mode': self.mode, 'k_sentences': self.k}

    def _compute(self, df, axis):
        raise NotImplementedError

    def inject(self, df, axis, cache_name=None, cache_path=None):
        """Return a copy of `df` with the 'injected_context' column.

        Contexts are cached in a CSV, so they are computed only once per axis
        (they do not depend on the steering coefficient or on the model).
        The cache file is either `cache_path` (explicit) or, if `cache_name`
        and `self.cache_dir` are set:
            <cache_dir>/<subdir>/<axis>_<cache_name>.csv
        With neither, nothing is cached.
        """
        if cache_path is None and cache_name is not None and self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, self.subdir, f"{axis}_{cache_name}.csv")

        if cache_path is not None and os.path.exists(cache_path):
            cached = pd.read_csv(cache_path)
            if len(cached) != len(df):
                raise ValueError(f"Cache {cache_path} has {len(cached)} rows, "
                                 f"expected {len(df)}. Delete it and retry.")
            out = df.copy()
            # empty contexts are read back as NaN: restore empty strings
            out['injected_context'] = cached['injected_context'].fillna("").values
            return out

        out = self._compute(df, axis)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            out.to_csv(cache_path, index=False)
        return out


class RandomInjector(BaseInjector):
    """Simulated RAG: random CrowS sentences (same logic as script 5)."""
    mode = "random"

    def __init__(self, crows_df, k, bias_ratio):
        super().__init__(k)
        self.crows_df = crows_df
        self.bias_ratio = bias_ratio

    @property
    def subdir(self):
        return f"k-{self.k}_b-{self.bias_ratio}"

    def describe(self):
        info = super().describe()
        info['bias_ratio'] = self.bias_ratio
        return info

    def _compute(self, df, axis):
        return inject_crows_bias_to_df(df, self.crows_df, axis,
                                       num_sentences=self.k, bias_ratio=self.bias_ratio)


class RetrieverInjector(BaseInjector):
    """Real RAG: the top-k sentences come from a retriever."""
    mode = "retriever"

    def __init__(self, retriever, k, query_fields=("context", "question"), axis_aware=True):
        super().__init__(k)
        self.retriever = retriever
        self.query_fields = query_fields
        self.axis_aware = axis_aware  # pass the BBQ axis to the retriever (e.g. to filter by bias_type)

    @property
    def subdir(self):
        return f"k-{self.k}_retriever"

    def describe(self):
        info = super().describe()
        info['bias_ratio'] = None  # not controllable: decided by the retriever
        return info

    def build_query(self, row):
        """Query = BBQ context + question (answers are NOT included)."""
        return " ".join(str(row[f]) for f in self.query_fields).strip()

    def _compute(self, df, axis):
        queries = [self.build_query(row) for _, row in df.iterrows()]
        retrieval_axis = axis if self.axis_aware else None

        if hasattr(self.retriever, "retrieve_batch"):
            results = self.retriever.retrieve_batch(queries, self.k, retrieval_axis)
        else:
            results = [self.retriever.retrieve(q, self.k, retrieval_axis)
                       for q in tqdm(queries, desc=f"  Retrieving for {axis}", leave=False)]

        if len(results) != len(queries):
            raise ValueError(f"Retriever returned {len(results)} results for {len(queries)} queries.")

        # No retrieved sentence -> no header at all (empty context)
        contexts = [format_injected_block(sents[:self.k]) if sents else "" for sents in results]

        out = df.copy()
        out['injected_context'] = contexts
        return out


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def build_injector(mode, k, bias_ratio=0.5, crows_path=None, retriever=None):
    """Return an injector, or None when no injection is requested (k <= 0)."""
    if k <= 0 or mode == "none":
        return None

    if mode == "random":
        if crows_path is None:
            raise ValueError("Random injection requires crows_path.")
        return RandomInjector(pd.read_csv(crows_path), k, bias_ratio)

    if mode == "retriever":
        if retriever is None:
            raise ValueError("Retriever injection requires a retriever object.")
        return RetrieverInjector(retriever, k)

    raise ValueError(f"Unknown injection mode: '{mode}'")
