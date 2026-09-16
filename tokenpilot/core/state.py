"""Small semantic state with explicit provenance and tenant boundaries.

Eviction affects active context only. Invalidation is a separate, explicit event.
No untrusted source text is evaluated, imported, or executed.
"""
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Dict, Optional, Tuple


class Verification(str, Enum):
    UNVERIFIED = 'unverified'
    VERIFIED = 'verified'
    INVALIDATED = 'invalidated'


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    sha256: str
    start: int
    end: int

    def __post_init__(self):
        if not self.source_id or len(self.sha256) != 64:
            raise ValueError('source id and SHA256 required')
        try:
            int(self.sha256, 16)
        except ValueError:
            raise ValueError('invalid SHA256') from None
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end:
            raise ValueError('invalid source span')


@dataclass(frozen=True)
class Atom:
    atom_id: str
    tenant: str
    key: str
    value: str
    provenance: SourceSpan
    observed_at: float
    expires_at: Optional[float] = None
    verification: Verification = Verification.UNVERIFIED

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.atom_id, self.tenant, self.key)):
            raise ValueError('atom identity, tenant and key required')
        if not isinstance(self.value, str) or not isinstance(self.provenance, SourceSpan):
            raise ValueError('atom value and provenance required')
        if not isinstance(self.verification, Verification):
            raise ValueError('explicit verification state required')
        for timestamp in (self.observed_at, self.expires_at):
            if timestamp is not None and (type(timestamp) not in (int, float)
                                          or not math.isfinite(timestamp)):
                raise ValueError('finite timestamps required')
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError('expiry must follow observation')

    def reusable(self, *, tenant: str, now: float, source_hash: str) -> bool:
        return (math.isfinite(now) and self.tenant == tenant and self.verification == Verification.VERIFIED
                and now >= self.observed_at
                and (self.expires_at is None or now < self.expires_at)
                and source_hash == self.provenance.sha256)


@dataclass(frozen=True)
class ArtifactPointer:
    """Character span in content addressed storage; never a filesystem path."""
    tenant: str
    sha256: str
    start: int
    end: int


class ArtifactStore:
    def __init__(self):
        self._blobs: Dict[Tuple[str, str], str] = {}

    def put(self, tenant: str, text: str) -> ArtifactPointer:
        if not tenant:
            raise ValueError('tenant required')
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self._blobs[(tenant, digest)] = text
        return ArtifactPointer(tenant, digest, 0, len(text))

    def page_in(self, pointer: ArtifactPointer, *, tenant: str, max_chars: int) -> str:
        if tenant != pointer.tenant:
            raise PermissionError('tenant mismatch')
        text = self._blobs[(tenant, pointer.sha256)]
        if not 0 <= pointer.start <= pointer.end <= len(text):
            raise ValueError('invalid artifact span')
        if pointer.end - pointer.start > max_chars:
            raise ValueError('page exceeds context allowance')
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != pointer.sha256:
            raise ValueError('artifact integrity failure')
        return text[pointer.start:pointer.end]


@dataclass
class SemanticState:
    tenant: str
    atoms: Dict[str, Atom] = field(default_factory=dict)
    active_ids: set = field(default_factory=set)

    def add(self, atom: Atom):
        if atom.tenant != self.tenant:
            raise PermissionError('tenant mismatch')
        if atom.atom_id in self.atoms:
            raise ValueError('atom ids are immutable; create a revision')
        self.atoms[atom.atom_id] = atom
        self.active_ids.add(atom.atom_id)

    def evict(self, atom_id: str):
        # Losing task relevance is not evidence that a fact is invalid.
        self.active_ids.discard(atom_id)

    def reuse(self, key: str, *, now: float, source_hash: str) -> Optional[Atom]:
        candidates = [atom for atom in self.atoms.values()
                      if atom.key == key and atom.reusable(
                          tenant=self.tenant, now=now, source_hash=source_hash)]
        return max(candidates, key=lambda atom: atom.observed_at) if candidates else None
