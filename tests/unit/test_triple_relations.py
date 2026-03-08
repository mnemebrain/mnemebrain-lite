"""Tests for Phase 5.2: inter-triple relations — TripleRelation model and RelationIndex."""

from uuid import uuid4

import pytest

from mnemebrain_core.triple_relations import (
    STAGE_2_RELATIONS,
    STAGE_3_RELATIONS,
    RelationIndex,
    TripleRelation,
    TripleRelationType,
)


def _uid():
    return uuid4()


def _rel(
    *,
    source: None | object = None,
    target: None | object = None,
    relation_type: TripleRelationType = TripleRelationType.ATTACKS,
    weight: float = 1.0,
    provenance: str = "",
    active: bool = True,
) -> TripleRelation:
    return TripleRelation(
        source_triple_id=source or _uid(),
        target_triple_id=target or _uid(),
        relation_type=relation_type,
        weight=weight,
        provenance=provenance,
        active=active,
    )


# ---------------------------------------------------------------------------
# TripleRelationType
# ---------------------------------------------------------------------------


class TestTripleRelationType:
    def test_stage_2_values(self):
        assert TripleRelationType.ATTACKS.value == "attacks"
        assert TripleRelationType.SUPPORTS.value == "supports"
        assert TripleRelationType.DEPENDS_ON.value == "depends_on"

    def test_stage_3_values(self):
        assert TripleRelationType.NARROWS.value == "narrows"
        assert TripleRelationType.OVERRIDES.value == "overrides"
        assert TripleRelationType.EXCEPTION_TO.value == "exception_to"
        assert TripleRelationType.DERIVED_FROM.value == "derived_from"

    def test_all_seven_variants_exist(self):
        assert len(TripleRelationType) == 7

    def test_stage_2_set_contains_core_three(self):
        assert STAGE_2_RELATIONS == {
            TripleRelationType.ATTACKS,
            TripleRelationType.SUPPORTS,
            TripleRelationType.DEPENDS_ON,
        }

    def test_stage_3_set_contains_extended_four(self):
        assert STAGE_3_RELATIONS == {
            TripleRelationType.NARROWS,
            TripleRelationType.OVERRIDES,
            TripleRelationType.EXCEPTION_TO,
            TripleRelationType.DERIVED_FROM,
        }

    def test_stage_sets_are_disjoint(self):
        assert STAGE_2_RELATIONS.isdisjoint(STAGE_3_RELATIONS)

    def test_stage_sets_cover_all_variants(self):
        assert STAGE_2_RELATIONS | STAGE_3_RELATIONS == set(TripleRelationType)


# ---------------------------------------------------------------------------
# TripleRelation model
# ---------------------------------------------------------------------------


class TestTripleRelation:
    def test_create_with_defaults(self):
        src, tgt = _uid(), _uid()
        rel = TripleRelation(
            source_triple_id=src,
            target_triple_id=tgt,
            relation_type=TripleRelationType.ATTACKS,
        )
        assert rel.source_triple_id == src
        assert rel.target_triple_id == tgt
        assert rel.relation_type == TripleRelationType.ATTACKS
        assert rel.weight == 1.0
        assert rel.provenance == ""
        assert rel.active is True
        assert rel.id is not None
        assert rel.created_at is not None

    def test_weight_bounds_valid(self):
        rel = _rel(weight=0.0)
        assert rel.weight == 0.0
        rel = _rel(weight=1.0)
        assert rel.weight == 1.0
        rel = _rel(weight=0.5)
        assert rel.weight == 0.5

    def test_weight_below_zero_rejected(self):
        with pytest.raises(ValueError):
            _rel(weight=-0.1)

    def test_weight_above_one_rejected(self):
        with pytest.raises(ValueError):
            _rel(weight=1.1)

    def test_self_relation_rejected(self):
        same_id = _uid()
        with pytest.raises(ValueError, match="must differ"):
            TripleRelation(
                source_triple_id=same_id,
                target_triple_id=same_id,
                relation_type=TripleRelationType.SUPPORTS,
            )

    def test_provenance_stored(self):
        rel = _rel(provenance="nli_conflict")
        assert rel.provenance == "nli_conflict"

    def test_serialization_roundtrip(self):
        rel = _rel(provenance="test")
        data = rel.model_dump()
        restored = TripleRelation.model_validate(data)
        assert restored.id == rel.id
        assert restored.relation_type == rel.relation_type
        assert restored.weight == rel.weight
        assert restored.provenance == rel.provenance

    def test_json_roundtrip(self):
        rel = _rel()
        json_str = rel.model_dump_json()
        restored = TripleRelation.model_validate_json(json_str)
        assert restored.id == rel.id


# ---------------------------------------------------------------------------
# RelationIndex — write operations
# ---------------------------------------------------------------------------


class TestRelationIndexWrite:
    def test_add_and_get(self):
        idx = RelationIndex()
        rel = _rel()
        idx.add(rel)
        assert idx.get(rel.id) is rel

    def test_get_unknown_returns_none(self):
        idx = RelationIndex()
        assert idx.get(_uid()) is None

    def test_add_many(self):
        idx = RelationIndex()
        rels = [_rel(), _rel(), _rel()]
        idx.add_many(rels)
        for r in rels:
            assert idx.get(r.id) is r

    def test_deactivate_returns_true_for_active(self):
        idx = RelationIndex()
        rel = _rel()
        idx.add(rel)
        assert idx.deactivate(rel.id) is True
        assert rel.active is False

    def test_deactivate_returns_false_for_already_inactive(self):
        idx = RelationIndex()
        rel = _rel(active=False)
        idx.add(rel)
        assert idx.deactivate(rel.id) is False

    def test_deactivate_returns_false_for_unknown(self):
        idx = RelationIndex()
        assert idx.deactivate(_uid()) is False

    def test_deactivate_by_triple_as_source(self):
        idx = RelationIndex()
        src = _uid()
        rel = _rel(source=src)
        idx.add(rel)
        count = idx.deactivate_by_triple(src)
        assert count == 1
        assert rel.active is False

    def test_deactivate_by_triple_as_target(self):
        idx = RelationIndex()
        tgt = _uid()
        rel = _rel(target=tgt)
        idx.add(rel)
        count = idx.deactivate_by_triple(tgt)
        assert count == 1
        assert rel.active is False

    def test_deactivate_by_triple_counts_both_directions(self):
        idx = RelationIndex()
        triple_id = _uid()
        other_a, other_b = _uid(), _uid()
        rel_out = TripleRelation(
            source_triple_id=triple_id,
            target_triple_id=other_a,
            relation_type=TripleRelationType.ATTACKS,
        )
        rel_in = TripleRelation(
            source_triple_id=other_b,
            target_triple_id=triple_id,
            relation_type=TripleRelationType.SUPPORTS,
        )
        idx.add_many([rel_out, rel_in])
        count = idx.deactivate_by_triple(triple_id)
        assert count == 2

    def test_deactivate_by_triple_no_double_count(self):
        """If a triple appears in both source and target indices, don't deactivate twice."""
        idx = RelationIndex()
        a, b = _uid(), _uid()
        rel = TripleRelation(
            source_triple_id=a,
            target_triple_id=b,
            relation_type=TripleRelationType.DEPENDS_ON,
        )
        idx.add(rel)
        # Deactivate by source
        count = idx.deactivate_by_triple(a)
        assert count == 1
        # Now try deactivating by target — already inactive
        count2 = idx.deactivate_by_triple(b)
        assert count2 == 0


# ---------------------------------------------------------------------------
# RelationIndex — read operations
# ---------------------------------------------------------------------------


class TestRelationIndexRead:
    def test_find_from_returns_outgoing(self):
        idx = RelationIndex()
        src = _uid()
        rel = _rel(source=src)
        idx.add(rel)
        results = idx.find_from(src)
        assert len(results) == 1
        assert results[0].id == rel.id

    def test_find_from_filters_by_type(self):
        idx = RelationIndex()
        src = _uid()
        r1 = _rel(source=src, relation_type=TripleRelationType.ATTACKS)
        r2 = _rel(source=src, relation_type=TripleRelationType.SUPPORTS)
        idx.add_many([r1, r2])
        results = idx.find_from(src, relation_type=TripleRelationType.ATTACKS)
        assert len(results) == 1
        assert results[0].relation_type == TripleRelationType.ATTACKS

    def test_find_from_excludes_inactive_by_default(self):
        idx = RelationIndex()
        src = _uid()
        rel = _rel(source=src, active=False)
        idx.add(rel)
        assert idx.find_from(src) == []

    def test_find_from_includes_inactive_when_asked(self):
        idx = RelationIndex()
        src = _uid()
        rel = _rel(source=src, active=False)
        idx.add(rel)
        results = idx.find_from(src, active_only=False)
        assert len(results) == 1

    def test_find_to_returns_incoming(self):
        idx = RelationIndex()
        tgt = _uid()
        rel = _rel(target=tgt)
        idx.add(rel)
        results = idx.find_to(tgt)
        assert len(results) == 1
        assert results[0].id == rel.id

    def test_find_to_filters_by_type(self):
        idx = RelationIndex()
        tgt = _uid()
        r1 = _rel(target=tgt, relation_type=TripleRelationType.DEPENDS_ON)
        r2 = _rel(target=tgt, relation_type=TripleRelationType.SUPPORTS)
        idx.add_many([r1, r2])
        results = idx.find_to(tgt, relation_type=TripleRelationType.DEPENDS_ON)
        assert len(results) == 1

    def test_find_to_excludes_inactive_by_default(self):
        idx = RelationIndex()
        tgt = _uid()
        rel = _rel(target=tgt, active=False)
        idx.add(rel)
        assert idx.find_to(tgt) == []

    def test_find_between_both_directions(self):
        idx = RelationIndex()
        a, b = _uid(), _uid()
        r_ab = TripleRelation(
            source_triple_id=a,
            target_triple_id=b,
            relation_type=TripleRelationType.ATTACKS,
        )
        r_ba = TripleRelation(
            source_triple_id=b,
            target_triple_id=a,
            relation_type=TripleRelationType.SUPPORTS,
        )
        idx.add_many([r_ab, r_ba])
        results = idx.find_between(a, b)
        assert len(results) == 2

    def test_find_between_excludes_inactive(self):
        idx = RelationIndex()
        a, b = _uid(), _uid()
        rel = TripleRelation(
            source_triple_id=a,
            target_triple_id=b,
            relation_type=TripleRelationType.ATTACKS,
            active=False,
        )
        idx.add(rel)
        assert idx.find_between(a, b) == []

    def test_find_between_includes_inactive_when_asked(self):
        idx = RelationIndex()
        a, b = _uid(), _uid()
        rel = TripleRelation(
            source_triple_id=a,
            target_triple_id=b,
            relation_type=TripleRelationType.ATTACKS,
            active=False,
        )
        idx.add(rel)
        results = idx.find_between(a, b, active_only=False)
        assert len(results) == 1

    def test_find_between_empty_for_unrelated(self):
        idx = RelationIndex()
        assert idx.find_between(_uid(), _uid()) == []

    def test_count_by_type(self):
        idx = RelationIndex()
        idx.add_many([
            _rel(relation_type=TripleRelationType.ATTACKS),
            _rel(relation_type=TripleRelationType.ATTACKS),
            _rel(relation_type=TripleRelationType.SUPPORTS),
        ])
        counts = idx.count_by_type()
        assert counts[TripleRelationType.ATTACKS] == 2
        assert counts[TripleRelationType.SUPPORTS] == 1

    def test_count_by_type_excludes_inactive(self):
        idx = RelationIndex()
        idx.add(_rel(active=False))
        assert idx.count_by_type() == {}

    def test_count_by_type_includes_inactive_when_asked(self):
        idx = RelationIndex()
        idx.add(_rel(relation_type=TripleRelationType.DEPENDS_ON, active=False))
        counts = idx.count_by_type(active_only=False)
        assert counts[TripleRelationType.DEPENDS_ON] == 1


# ---------------------------------------------------------------------------
# RelationIndex — edge cases
# ---------------------------------------------------------------------------


class TestRelationIndexEdgeCases:
    def test_find_from_with_dangling_id(self):
        """If secondary index has an ID not in primary store, skip it."""
        idx = RelationIndex()
        src = _uid()
        idx._by_source[src].append(_uid())  # dangling
        assert idx.find_from(src) == []

    def test_find_to_with_dangling_id(self):
        idx = RelationIndex()
        tgt = _uid()
        idx._by_target[tgt].append(_uid())  # dangling
        assert idx.find_to(tgt) == []

    def test_find_between_with_dangling_id(self):
        idx = RelationIndex()
        a, b = _uid(), _uid()
        idx._by_source[a].append(_uid())  # dangling
        assert idx.find_between(a, b) == []

    def test_find_between_with_dangling_id_in_reverse_direction(self):
        """Dangling ID in _by_source[triple_b] (reverse lookup) returns empty."""
        idx = RelationIndex()
        a, b = _uid(), _uid()
        idx._by_source[b].append(_uid())  # dangling in reverse direction
        assert idx.find_between(a, b) == []

    def test_find_between_excludes_inactive_in_reverse_direction(self):
        """Inactive relation from B→A is excluded by find_between with active_only=True."""
        idx = RelationIndex()
        a, b = _uid(), _uid()
        rel = _rel(source=b, target=a, active=False)
        idx.add(rel)
        assert idx.find_between(a, b, active_only=True) == []


class TestDeactivateByTripleSeenDedup:
    def test_deactivate_deduplicates_when_rel_in_both_indices(self):
        """If a relation ID appears in both _by_source and _by_target for the same
        triple_id, it should only be deactivated once (the `seen` set prevents double counting)."""
        idx = RelationIndex()
        t = _uid()
        # Create a normal relation where t is the source
        rel = _rel(source=t)
        idx.add(rel)
        # Manually also register this relation's ID under _by_target[t]
        # to simulate the edge case where the same rel_id appears in both indices
        idx._by_target[t].append(rel.id)
        count = idx.deactivate_by_triple(t)
        assert count == 1
        assert not rel.active
