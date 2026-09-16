from dataclasses import replace
import hashlib
import unittest

from tokenpilot.core.state import Atom, ArtifactStore, SemanticState, SourceSpan, Verification
from tokenpilot.core.planner import Action, BreakEvenDetector, HeuristicPlanner, PlannerState


class StateTests(unittest.TestCase):
    def atom(self):
        return Atom('a1', 'tenant-a', 'owner', 'lyra',
                    SourceSpan('doc', hashlib.sha256(b'lyra').hexdigest(), 0, 4),
                    10, 20, Verification.VERIFIED)

    def test_reuse_requires_tenant_freshness_hash_and_verification(self):
        atom = self.atom()
        kwargs = dict(tenant=atom.tenant, now=15, source_hash=atom.provenance.sha256)
        self.assertTrue(atom.reusable(**kwargs))
        for changes in ({'tenant': 'tenant-b'}, {'now': 20}, {'now': 9},
                        {'source_hash': '0'*64}, {'now': float('nan')}):
            self.assertFalse(atom.reusable(**dict(kwargs, **changes)))
        for verification in (Verification.INVALIDATED, Verification.UNVERIFIED):
            self.assertFalse(replace(atom, verification=verification).reusable(**kwargs))

    def test_eviction_keeps_provenance_and_reusable_data(self):
        state, atom = SemanticState('tenant-a'), self.atom()
        state.add(atom)
        state.evict(atom.atom_id)
        self.assertNotIn(atom.atom_id, state.active_ids)
        self.assertEqual(state.reuse('owner', now=15, source_hash=atom.provenance.sha256), atom)
        self.assertEqual(state.atoms[atom.atom_id].verification, Verification.VERIFIED)

    def test_cross_tenant_and_mutation_rejected(self):
        state = SemanticState('tenant-a')
        with self.assertRaises(PermissionError):
            state.add(replace(self.atom(), tenant='tenant-b'))
        state.add(self.atom())
        with self.assertRaises(ValueError):
            state.add(self.atom())

    def test_pointer_page_in_checks_tenant_span_and_budget(self):
        store = ArtifactStore()
        pointer = store.put('tenant-a', 'abcdef')
        self.assertEqual(store.page_in(replace(pointer, start=2, end=4), tenant='tenant-a', max_chars=2), 'cd')
        with self.assertRaises(PermissionError):
            store.page_in(pointer, tenant='tenant-b', max_chars=20)
        with self.assertRaises(ValueError):
            store.page_in(pointer, tenant='tenant-a', max_chars=3)
        with self.assertRaises(ValueError):
            store.page_in(replace(pointer, end=8), tenant='tenant-a', max_chars=20)

    def test_invalid_timestamps_and_spans(self):
        for changes in ({'expires_at': 10}, {'observed_at': float('nan')}, {'expires_at': float('inf')}):
            with self.assertRaises(ValueError):
                replace(self.atom(), **changes)
        with self.assertRaises(ValueError):
            SourceSpan('source', 'x'*64, 0, 1)


class PlannerTests(unittest.TestCase):
    def test_replans_one_action_at_a_time_and_stops(self):
        planner, state = HeuristicPlanner(), PlannerState(3000, 0)
        self.assertEqual(planner.next_action(state).action, Action.RETRIEVE)
        state.selection_ready = True
        self.assertEqual(planner.next_action(state).action, Action.CALL_MODEL)
        state.response_ready = True
        self.assertEqual(planner.next_action(state).action, Action.VERIFY)
        state.verification_done = True
        self.assertEqual(planner.next_action(state).action, Action.STOP)

    def test_failure_stops_without_retry(self):
        self.assertEqual(HeuristicPlanner().next_action(PlannerState(1, 1, failed=True)).action, Action.STOP)

    def test_short_and_low_gain_bypass(self):
        detector = BreakEvenDetector()
        self.assertTrue(detector.should_bypass(100, 5))
        self.assertTrue(detector.should_bypass(3000, 2900))
        self.assertFalse(detector.should_bypass(3000, 100))
