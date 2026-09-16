"""Context strategies operate on query/data only, never expected answers."""
import json
import re
from typing import List

from tokenpilot.core.planner import BreakEvenDetector, QualityContract
from tokenpilot.runtime.context import ContextSelection as Selection, optimize_records

STRATEGIES = ('full-history', 'sliding-window', 'retrieval-context', 'tokenpilot')
ABLATIONS = ('no-bypass', 'no-pruning', 'no-page-in')


def size(documents):
    return len(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode('utf-8'))


def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def lexical_retrieve(query, documents, top_k):
    query_terms = terms(query['key'])
    ranked = sorted(enumerate(documents), key=lambda pair: (
        -len(query_terms & terms(json.dumps(pair[1], sort_keys=True))), pair[0]))
    indices = {index for index, _ in ranked[:top_k]}
    return [doc for index, doc in enumerate(documents) if index in indices]


def select_context(query, documents, strategy, *, window_bytes=8192,
                   retrieval_k=8, ablation=None, detector=None):
    if strategy not in STRATEGIES:
        raise ValueError('unknown strategy')
    if ablation is not None and (strategy != 'tokenpilot' or ablation not in ABLATIONS):
        raise ValueError('invalid ablation')
    if window_bytes <= 0 or retrieval_k <= 0:
        raise ValueError('context allowances must be positive')
    detector = detector or BreakEvenDetector()
    if strategy == 'full-history':
        return Selection(list(documents), False, 'all records')
    if strategy == 'sliding-window':
        selected: List[dict] = []
        for doc in reversed(documents):
            if size([doc] + selected) > window_bytes:
                break
            selected.insert(0, doc)
        return Selection(selected, False, 'recent complete records within byte allowance')
    if strategy == 'retrieval-context':
        return Selection(lexical_retrieve(query, documents, retrieval_k), False,
                         'lexical overlap top-k; no learned embeddings or ground truth access')
    return optimize_records(query, documents, tenant='benchmark-public-fixtures',
                            contract=QualityContract((query['key'],)), detector=detector,
                            ablation=ablation)
