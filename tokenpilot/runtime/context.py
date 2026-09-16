"""Reusable structured-record context compiler, independent of benchmark answers.

Only the declared lookup/latest/sum/one-hop join contract is supported. This is not
an unstructured natural-language relevance classifier.
"""
from dataclasses import dataclass
import json
from typing import Dict, List, Optional

from tokenpilot.core.planner import BreakEvenDetector, QualityContract
from tokenpilot.core.state import ArtifactPointer, ArtifactStore


def encoded_size(records):
    return len(json.dumps(records, ensure_ascii=False, sort_keys=True).encode('utf-8'))


@dataclass(frozen=True)
class ContextSelection:
    documents: List[dict]
    bypassed: bool
    reason: str


class RecordContext:
    """Raw records stay in an artifact store; the index holds spans only."""
    def __init__(self, tenant: str, records: List[dict]):
        self.tenant = tenant
        self.store = ArtifactStore()
        self.index: Dict[str, List[ArtifactPointer]] = {}
        # A record is an independently addressed artifact. This avoids repeatedly
        # hashing one giant blob when paging multiple small spans.
        for record in records:
            if not isinstance(record.get('key'), str) or not isinstance(record.get('id'), str):
                raise ValueError('record key and source id must be strings')
            raw = json.dumps(record, ensure_ascii=False, sort_keys=True)
            pointer = self.store.put(tenant, raw)
            self.index.setdefault(record['key'], []).append(pointer)

    def retrieve(self, key: str) -> List[dict]:
        return [json.loads(self.store.page_in(pointer, tenant=self.tenant,
                                             max_chars=pointer.end - pointer.start))
                for pointer in self.index.get(key, [])]


def optimize_records(query: dict, documents: List[dict], *, tenant: str,
                     contract: QualityContract, detector: Optional[BreakEvenDetector] = None,
                     ablation: Optional[str] = None) -> ContextSelection:
    if query.get('operation') not in {'lookup', 'latest', 'sum', 'join'}:
        raise ValueError('unsupported structured operation')
    if contract.required_keys != (query.get('key'),):
        raise ValueError('query does not match quality contract')
    if not tenant:
        raise ValueError('tenant required')
    if ablation not in {None, 'no-bypass', 'no-pruning', 'no-page-in'}:
        raise ValueError('unsupported ablation')
    ids = [record['id'] for record in documents]
    if len(ids) != len(set(ids)):
        raise ValueError('source ids must be unique; resolve duplicates before optimization')
    detector = detector or BreakEvenDetector()
    original_size = encoded_size(documents)
    if ablation != 'no-bypass' and original_size < detector.bypass_below_bytes:
        return ContextSelection(list(documents), True, 'short context bypass')
    selected = list(documents)
    if ablation != 'no-pruning':
        context = RecordContext(tenant, documents)
        selected = context.retrieve(query['key'])
        if query['operation'] == 'join' and ablation != 'no-page-in':
            linked = {record['value'] for record in selected if isinstance(record.get('value'), str)}
            # Stable order regardless of hash randomization or tenant process.
            for key in sorted(linked):
                selected.extend(context.retrieve(key))
    if ablation != 'no-bypass' and detector.should_bypass(original_size, encoded_size(selected)):
        return ContextSelection(list(documents), True, 'estimated reduction below configured floor')
    return ContextSelection(selected, False, 'exact-key relevance with one-hop artifact page-in')
