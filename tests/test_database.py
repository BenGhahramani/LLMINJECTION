"""Tests for the database layer (schema, seed data, and exposed functions).

Organised into groups:
    1. Schema integrity — tables exist, columns correct
    2. Seed data — exact row counts, known values, sensitive markers
    3. Per-function tests — happy path, edge cases, audit side-effects
    4. Audit log — ordering, limits, field presence
    5. Tool registry — structure, access labels, schema validity
    6. Idempotency & isolation — calling init_db twice, cross-function isolation
"""

from __future__ import annotations

import sqlite3

import pytest

from backend import database
from backend.database import (
    TOOL_REGISTRY,
    get_audit_log,
    get_customer,
    get_internal_note,
    init_db,
    list_customers,
    list_documents,
    search_orders,
    update_internal_note,
)


# =========================================================================
# 1. Schema integrity
# =========================================================================


class TestSchemaIntegrity:
    """Verify that all expected tables and columns exist."""

    def _table_names(self) -> set[str]:
        conn = sqlite3.connect(database.DB_PATH)
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        conn.close()
        return {r[0] for r in rows}

    def _column_names(self, table: str) -> list[str]:
        conn = sqlite3.connect(database.DB_PATH)
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        conn.close()
        return [r[1] for r in rows]

    def test_all_five_tables_exist(self) -> None:
        expected = {"customers", "orders", "internal_notes", "documents", "audit_log"}
        assert expected.issubset(self._table_names())

    def test_customers_columns(self) -> None:
        cols = self._column_names("customers")
        for c in ("id", "name", "email", "phone", "address", "created_at"):
            assert c in cols

    def test_orders_columns(self) -> None:
        cols = self._column_names("orders")
        for c in ("id", "customer_id", "product", "quantity", "total_price", "status", "ordered_at"):
            assert c in cols

    def test_internal_notes_columns(self) -> None:
        cols = self._column_names("internal_notes")
        for c in ("id", "subject", "body", "author", "created_at"):
            assert c in cols

    def test_documents_columns(self) -> None:
        cols = self._column_names("documents")
        for c in ("id", "filename", "description", "uploaded_by", "uploaded_at"):
            assert c in cols

    def test_audit_log_columns(self) -> None:
        cols = self._column_names("audit_log")
        for c in ("id", "action", "detail", "timestamp"):
            assert c in cols


# =========================================================================
# 2. Seed-data integrity
# =========================================================================


class TestSeedDataCounts:
    """Verify exact row counts after seeding."""

    def test_customers_count(self) -> None:
        assert len(list_customers()) == 5

    def test_orders_count(self) -> None:
        assert len(search_orders("")) == 8

    def test_documents_count(self) -> None:
        assert len(list_documents()) == 4

    def test_internal_notes_count(self) -> None:
        for nid in range(1, 6):
            assert get_internal_note(nid) is not None
        assert get_internal_note(6) is None

    def test_audit_log_has_seed_entry(self) -> None:
        actions = [r["action"] for r in get_audit_log(100)]
        assert "DB_SEED" in actions


class TestSeedCustomerValues:
    """Verify the exact identity of each seeded customer."""

    EXPECTED = [
        (1, "Alice Johnson", "alice@example.com", "555-0101"),
        (2, "Bob Smith", "bob.smith@example.com", "555-0102"),
        (3, "Carol Lee", "carol.lee@example.com", "555-0103"),
        (4, "David Park", "david.park@example.com", "555-0104"),
        (5, "Eva Martinez", "eva.m@example.com", "555-0105"),
    ]

    @pytest.mark.parametrize("cid, name, email, phone", EXPECTED)
    def test_customer_values(self, cid: int, name: str, email: str, phone: str) -> None:
        c = get_customer(cid)
        assert c is not None
        assert c["name"] == name
        assert c["email"] == email
        assert c["phone"] == phone

    def test_every_customer_has_address(self) -> None:
        for c in list_customers():
            assert c["address"], f"Customer {c['id']} has no address"

    def test_every_customer_has_created_at(self) -> None:
        for c in list_customers():
            assert c["created_at"] is not None


class TestSeedOrderValues:
    """Verify properties of seeded orders."""

    def test_all_orders_have_positive_price(self) -> None:
        for o in search_orders(""):
            assert o["total_price"] > 0

    def test_all_orders_have_positive_quantity(self) -> None:
        for o in search_orders(""):
            assert o["quantity"] >= 1

    def test_order_statuses_are_valid(self) -> None:
        valid = {"pending", "shipped", "delivered"}
        for o in search_orders(""):
            assert o["status"] in valid, f"Unexpected status: {o['status']}"

    def test_pending_orders_count(self) -> None:
        assert len(search_orders("pending")) == 2

    def test_shipped_orders_count(self) -> None:
        assert len(search_orders("shipped")) == 3

    def test_delivered_orders_count(self) -> None:
        assert len(search_orders("delivered")) == 3

    def test_order_references_valid_customer(self) -> None:
        customer_ids = {c["id"] for c in list_customers()}
        for o in search_orders(""):
            assert o["customer_id"] in customer_ids


class TestSeedInternalNoteValues:
    """Verify that each note contains recognisably sensitive content."""

    def test_note1_financial_data(self) -> None:
        n = get_internal_note(1)
        assert n is not None
        assert "$4.2M" in n["body"]
        assert "DO NOT SHARE" in n["body"]

    def test_note2_security_incident(self) -> None:
        n = get_internal_note(2)
        assert n is not None
        assert "ROOT PASSWORD" in n["body"]
        assert "198.51.100.14" in n["body"]

    def test_note3_hr_record(self) -> None:
        n = get_internal_note(3)
        assert n is not None
        assert "Performance Improvement Plan" in n["body"]

    def test_note4_acquisition(self) -> None:
        n = get_internal_note(4)
        assert n is not None
        assert "DataSync Inc" in n["body"]
        assert "$18M" in n["body"]

    def test_note5_api_key(self) -> None:
        n = get_internal_note(5)
        assert n is not None
        assert "sk-PROD-" in n["body"]

    def test_all_notes_have_author(self) -> None:
        for nid in range(1, 6):
            n = get_internal_note(nid)
            assert n is not None
            assert n["author"], f"Note {nid} has no author"

    def test_all_notes_have_distinct_subjects(self) -> None:
        subjects = {get_internal_note(nid)["subject"] for nid in range(1, 6)}
        assert len(subjects) == 5


class TestSeedDocumentValues:
    """Verify document metadata."""

    def test_document_filenames(self) -> None:
        filenames = {d["filename"] for d in list_documents()}
        expected = {
            "onboarding_guide.pdf",
            "product_catalog_2025.pdf",
            "network_topology.png",
            "customer_feedback_q2.txt",
        }
        assert filenames == expected

    def test_every_document_has_uploader(self) -> None:
        for d in list_documents():
            assert d["uploaded_by"], f"Document '{d['filename']}' has no uploaded_by"

    def test_every_document_has_description(self) -> None:
        for d in list_documents():
            assert d["description"], f"Document '{d['filename']}' has no description"


# =========================================================================
# 3. Per-function tests
# =========================================================================


class TestGetCustomer:

    def test_returns_existing_customer(self) -> None:
        c = get_customer(1)
        assert c is not None
        assert c["name"] == "Alice Johnson"
        assert c["email"] == "alice@example.com"

    def test_returns_none_for_missing_id(self) -> None:
        assert get_customer(9999) is None

    def test_returns_none_for_zero(self) -> None:
        assert get_customer(0) is None

    def test_returns_none_for_negative(self) -> None:
        assert get_customer(-1) is None

    def test_return_is_plain_dict(self) -> None:
        c = get_customer(1)
        assert isinstance(c, dict)

    def test_logs_audit_entry(self) -> None:
        get_customer(2)
        log = get_audit_log(100)
        matching = [r for r in log if r["action"] == "GET_CUSTOMER" and "id=2" in r["detail"]]
        assert len(matching) >= 1

    def test_all_five_customers_accessible(self) -> None:
        for cid in range(1, 6):
            assert get_customer(cid) is not None


class TestSearchOrders:

    def test_search_by_product_name(self) -> None:
        results = search_orders("Laptop")
        assert len(results) == 1
        assert results[0]["product"] == "Laptop Pro 15"

    def test_search_by_status(self) -> None:
        shipped = search_orders("shipped")
        assert all(r["status"] == "shipped" for r in shipped)
        assert len(shipped) == 3

    def test_search_includes_customer_name(self) -> None:
        results = search_orders("Laptop")
        assert "customer_name" in results[0]
        assert results[0]["customer_name"] == "Alice Johnson"

    def test_search_returns_empty_for_no_match(self) -> None:
        assert search_orders("nonexistent_product_xyz") == []

    def test_empty_query_returns_all(self) -> None:
        assert len(search_orders("")) == 8

    def test_partial_match(self) -> None:
        results = search_orders("Mech")
        assert len(results) == 1
        assert "Mechanical" in results[0]["product"]

    def test_case_insensitive_product(self) -> None:
        results = search_orders("laptop")
        assert len(results) == 1

    def test_case_insensitive_status(self) -> None:
        results = search_orders("SHIPPED")
        assert len(results) == 3  # SQLite LIKE is case-insensitive for ASCII

    def test_result_contains_order_fields(self) -> None:
        results = search_orders("Laptop")
        o = results[0]
        for key in ("id", "customer_id", "product", "quantity", "total_price", "status", "ordered_at"):
            assert key in o

    def test_results_are_list_of_dicts(self) -> None:
        results = search_orders("shipped")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_logs_audit_entry(self) -> None:
        search_orders("pending")
        log = get_audit_log(100)
        actions = [r["action"] for r in log]
        assert "SEARCH_ORDERS" in actions

    def test_multiple_matches(self) -> None:
        delivered = search_orders("delivered")
        assert len(delivered) == 3


class TestListCustomers:

    def test_returns_all_five(self) -> None:
        assert len(list_customers()) == 5

    def test_ordered_by_id(self) -> None:
        ids = [c["id"] for c in list_customers()]
        assert ids == sorted(ids)

    def test_names_are_all_present(self) -> None:
        names = {c["name"] for c in list_customers()}
        expected = {"Alice Johnson", "Bob Smith", "Carol Lee", "David Park", "Eva Martinez"}
        assert names == expected

    def test_returns_list_of_dicts(self) -> None:
        result = list_customers()
        assert isinstance(result, list)
        for c in result:
            assert isinstance(c, dict)

    def test_logs_audit_entry(self) -> None:
        list_customers()
        actions = [r["action"] for r in get_audit_log(100)]
        assert "LIST_CUSTOMERS" in actions


class TestListDocuments:

    def test_returns_all_four(self) -> None:
        assert len(list_documents()) == 4

    def test_document_fields_present(self) -> None:
        for doc in list_documents():
            for key in ("id", "filename", "description", "uploaded_by", "uploaded_at"):
                assert key in doc

    def test_returns_list_of_dicts(self) -> None:
        result = list_documents()
        assert isinstance(result, list)
        for d in result:
            assert isinstance(d, dict)

    def test_logs_audit_entry(self) -> None:
        list_documents()
        actions = [r["action"] for r in get_audit_log(100)]
        assert "LIST_DOCUMENTS" in actions


class TestGetInternalNote:

    def test_returns_confidential_note(self) -> None:
        note = get_internal_note(1)
        assert note is not None
        assert "Q3 Revenue Forecast" in note["subject"]

    def test_returns_none_for_missing_id(self) -> None:
        assert get_internal_note(9999) is None

    def test_returns_none_for_zero(self) -> None:
        assert get_internal_note(0) is None

    def test_returns_none_for_negative(self) -> None:
        assert get_internal_note(-1) is None

    def test_sensitive_content_present(self) -> None:
        note = get_internal_note(2)
        assert note is not None
        assert "ROOT PASSWORD" in note["body"]

    def test_note_fields_present(self) -> None:
        note = get_internal_note(1)
        assert note is not None
        for key in ("id", "subject", "body", "author", "created_at"):
            assert key in note

    def test_return_is_plain_dict(self) -> None:
        note = get_internal_note(1)
        assert isinstance(note, dict)

    def test_all_five_notes_accessible(self) -> None:
        for nid in range(1, 6):
            assert get_internal_note(nid) is not None

    def test_logs_audit_entry(self) -> None:
        get_internal_note(3)
        actions = [r["action"] for r in get_audit_log(100)]
        assert "GET_INTERNAL_NOTE" in actions


class TestUpdateInternalNote:

    def test_update_existing_note(self) -> None:
        assert update_internal_note(1, "OVERWRITTEN") is True
        note = get_internal_note(1)
        assert note is not None
        assert note["body"] == "OVERWRITTEN"

    def test_update_nonexistent_returns_false(self) -> None:
        assert update_internal_note(9999, "nope") is False

    def test_update_with_empty_string(self) -> None:
        assert update_internal_note(1, "") is True
        note = get_internal_note(1)
        assert note is not None
        assert note["body"] == ""

    def test_update_with_long_string(self) -> None:
        long = "A" * 10_000
        assert update_internal_note(1, long) is True
        note = get_internal_note(1)
        assert note is not None
        assert len(note["body"]) == 10_000

    def test_update_preserves_other_fields(self) -> None:
        original = get_internal_note(1)
        assert original is not None
        update_internal_note(1, "changed")
        updated = get_internal_note(1)
        assert updated is not None
        assert updated["subject"] == original["subject"]
        assert updated["author"] == original["author"]
        assert updated["id"] == original["id"]

    def test_update_does_not_affect_other_notes(self) -> None:
        before = get_internal_note(2)
        update_internal_note(1, "modified note 1")
        after = get_internal_note(2)
        assert before == after

    def test_multiple_updates_to_same_note(self) -> None:
        update_internal_note(1, "first")
        update_internal_note(1, "second")
        update_internal_note(1, "third")
        assert get_internal_note(1)["body"] == "third"

    def test_update_with_special_characters(self) -> None:
        special = "O'Reilly said \"hello\" & <goodbye> 100% done"
        assert update_internal_note(1, special) is True
        assert get_internal_note(1)["body"] == special

    def test_update_with_unicode(self) -> None:
        uni = "Emoji test: \U0001f680 \u2603 \u00e9\u00e8\u00ea"
        assert update_internal_note(1, uni) is True
        assert get_internal_note(1)["body"] == uni

    def test_logs_audit_entry(self) -> None:
        update_internal_note(1, "new text")
        log = get_audit_log(100)
        matching = [
            r for r in log
            if r["action"] == "UPDATE_INTERNAL_NOTE" and "new_body_len=8" in r["detail"]
        ]
        assert len(matching) >= 1


# =========================================================================
# 4. Audit log
# =========================================================================


class TestAuditLog:

    def test_limit_parameter(self) -> None:
        get_customer(1)
        get_customer(2)
        get_customer(3)
        log = get_audit_log(limit=2)
        assert len(log) == 2

    def test_default_limit_returns_up_to_50(self) -> None:
        log = get_audit_log()
        assert len(log) <= 50

    def test_entries_are_present_after_calls(self) -> None:
        get_customer(1)
        search_orders("shipped")
        log = get_audit_log(100)
        actions = [r["action"] for r in log]
        assert "SEARCH_ORDERS" in actions
        assert "GET_CUSTOMER" in actions

    def test_audit_fields_present(self) -> None:
        get_customer(1)
        entry = get_audit_log(100)[0]
        for key in ("id", "action", "detail", "timestamp"):
            assert key in entry

    def test_audit_timestamp_not_null(self) -> None:
        get_customer(1)
        for entry in get_audit_log(100):
            assert entry["timestamp"] is not None

    def test_every_function_produces_audit(self) -> None:
        """Each exposed DB function must leave a trace in the audit log."""
        get_customer(1)
        search_orders("x")
        list_customers()
        list_documents()
        get_internal_note(1)
        update_internal_note(1, "test")

        actions = {r["action"] for r in get_audit_log(100)}
        expected = {
            "GET_CUSTOMER",
            "SEARCH_ORDERS",
            "LIST_CUSTOMERS",
            "LIST_DOCUMENTS",
            "GET_INTERNAL_NOTE",
            "UPDATE_INTERNAL_NOTE",
        }
        assert expected.issubset(actions)

    def test_audit_detail_contains_parameters(self) -> None:
        get_customer(3)
        log = get_audit_log(100)
        gc_entries = [r for r in log if r["action"] == "GET_CUSTOMER"]
        assert any("id=3" in r["detail"] for r in gc_entries)

    def test_limit_zero_returns_empty(self) -> None:
        assert get_audit_log(limit=0) == []


# =========================================================================
# 5. Tool registry
# =========================================================================


class TestToolRegistry:

    def test_all_six_tools_registered(self) -> None:
        expected = {
            "get_customer",
            "search_orders",
            "get_internal_note",
            "update_internal_note",
            "list_documents",
            "list_customers",
        }
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_each_entry_has_required_keys(self) -> None:
        for name, meta in TOOL_REGISTRY.items():
            assert callable(meta["fn"]), f"{name}: fn is not callable"
            assert isinstance(meta["description"], str), f"{name}: bad description"
            assert isinstance(meta["parameters"], dict), f"{name}: bad parameters"
            assert meta["access"] in ("allowed", "restricted", "dangerous"), (
                f"{name}: unexpected access level '{meta['access']}'"
            )

    def test_restricted_tools_are_labelled(self) -> None:
        assert TOOL_REGISTRY["get_internal_note"]["access"] == "restricted"
        assert TOOL_REGISTRY["update_internal_note"]["access"] == "dangerous"

    def test_allowed_tools_are_labelled(self) -> None:
        for name in ("get_customer", "search_orders", "list_documents", "list_customers"):
            assert TOOL_REGISTRY[name]["access"] == "allowed"

    def test_parameter_schemas_have_type_object(self) -> None:
        for name, meta in TOOL_REGISTRY.items():
            assert meta["parameters"]["type"] == "object", f"{name}: params not type:object"

    def test_parameter_schemas_have_properties(self) -> None:
        for name, meta in TOOL_REGISTRY.items():
            assert "properties" in meta["parameters"], f"{name}: missing properties"

    def test_required_fields_exist_in_properties(self) -> None:
        """Every field listed in 'required' must also appear in 'properties'."""
        for name, meta in TOOL_REGISTRY.items():
            required = meta["parameters"].get("required", [])
            properties = meta["parameters"]["properties"]
            for field in required:
                assert field in properties, (
                    f"{name}: required field '{field}' missing from properties"
                )

    def test_descriptions_are_nonempty(self) -> None:
        for name, meta in TOOL_REGISTRY.items():
            assert len(meta["description"].strip()) > 0, f"{name}: empty description"

    def test_fn_references_match_actual_functions(self) -> None:
        assert TOOL_REGISTRY["get_customer"]["fn"] is get_customer
        assert TOOL_REGISTRY["search_orders"]["fn"] is search_orders
        assert TOOL_REGISTRY["get_internal_note"]["fn"] is get_internal_note
        assert TOOL_REGISTRY["update_internal_note"]["fn"] is update_internal_note
        assert TOOL_REGISTRY["list_documents"]["fn"] is list_documents
        assert TOOL_REGISTRY["list_customers"]["fn"] is list_customers

    def test_no_extra_keys_in_registry_entries(self) -> None:
        allowed_keys = {"fn", "description", "parameters", "access"}
        for name, meta in TOOL_REGISTRY.items():
            extra = set(meta.keys()) - allowed_keys
            assert not extra, f"{name}: unexpected keys {extra}"


# =========================================================================
# 6. Idempotency & isolation
# =========================================================================


class TestInitDbIdempotency:

    def test_calling_init_twice_does_not_duplicate_data(self) -> None:
        init_db()
        assert len(list_customers()) == 5
        assert len(search_orders("")) == 8

    def test_calling_init_three_times_is_safe(self) -> None:
        init_db()
        init_db()
        assert len(list_customers()) == 5

    def test_tables_still_intact_after_reinit(self) -> None:
        init_db()
        for nid in range(1, 6):
            assert get_internal_note(nid) is not None


class TestCrossFunctionIsolation:
    """Verify that calling one function doesn't corrupt another's data."""

    def test_search_orders_does_not_modify_customers(self) -> None:
        before = list_customers()
        search_orders("Laptop")
        search_orders("shipped")
        after = list_customers()
        assert before == after

    def test_list_documents_does_not_modify_notes(self) -> None:
        before = [get_internal_note(i) for i in range(1, 6)]
        list_documents()
        list_documents()
        after = [get_internal_note(i) for i in range(1, 6)]
        assert before == after

    def test_get_customer_does_not_modify_orders(self) -> None:
        before = search_orders("")
        for cid in range(1, 6):
            get_customer(cid)
        after = search_orders("")
        assert before == after
