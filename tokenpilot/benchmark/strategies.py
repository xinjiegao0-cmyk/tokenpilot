"""Context strategies operate on query/data only, never expected answers."""
from dataclasses import dataclass
import json
import re
from typing import List

from tokenpilot.core.planner import BreakEvenDetector

STRATEGIES = ('full-history', 'sliding-window', 'retrieval-context', 'tokenpilot')
ABLATIONS = ('no-bypass', 'no-dedup', 'no-pruning', 'no-page-in')


def size(documents):
    return len(json.dumps(documents, ensure_ascii=False, sort_keys=True).encode('utf-8'))


def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower()))


@dataclass(frozen=True)
class Selection:
    documents: List[dict]
    bypassed: bool
    reason: str


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
    original_size = size(documents)
    if ablation != 'no-bypass' and original_size < detector.bypass_below_bytes:
        return Selection(list(documents), True, 'short context bypass')
    selected = list(documents)
    if ablation != 'no-pruning':
        # Structured-key relevance is lossless only for this declared record-query
        # contract. It is not a universal semantic relevance estimator.
        selected = [doc for doc in documents if doc['key'] == query['key']]
        if query['operation'] == 'join' and ablation != 'no-page-in':
            linked = {doc['value'] for doc in selected if isinstance(doc['value'], str)}
            selected += [doc for doc in documents if doc['key'] in linked]
    if ablation != 'no-dedup':
        # Only byte-identical records (including provenance id) can be deduplicated.
        # Two equal values from different sources may both be needed for sum/citations.
        seen = set()
        unique = []
        for doc in selected:
            fingerprint = json.dumps(doc, ensure_ascii=False, sort_keys=True)
            if fingerprint not in seen:
                unique.append(doc)
                seen.add(fingerprint)
        selected = unique
    if ablation != 'no-bypass' and detector.should_bypass(original_size, size(selected)):
        return Selection(list(documents), True, 'estimated reduction below configured floor')
    return Selection(selected, False, 'exact-key relevance with one-hop dependency page-in')
