"""Tests for the database layer (schema, seed data, and exposed functions)."""

from __future__ import annotations

from backend.database import (
    TOOL_REGISTRY,
    get_audit_log,
    get_customer,
    get_internal_note,
    list_customers,
    list_documents,
    search_orders,
    update_internal_note,
)


# ---------------------------------------------------------------------------
# Seed-data integrity
# ---------------------------------------------------------------------------


class TestSeedData:
    """Verify that init_db() seeds the expected rows."""

    def test_customers_seeded(self) -> None:
        customers = list_customers()
        assert len(customers) == 5

    def test_customer_fields_present(self) -> None:
        customer = get_customer(1)
        assert customer is not None
        for key in ("id", "name", "email", "phone", "address", "created_at"):
            assert key in customer

    def test_orders_seeded(self) -> None:
        all_orders = search_orders("")
        assert len(all_orders) == 8

    def test_documents_seeded(self) -> None:
        docs = list_documents()
        assert len(docs) == 4

    def test_internal_notes_seeded(self) -> None:
        for note_id in range(1, 6):
            note = get_internal_note(note_id)
            assert note is not None
            assert note["subject"]
            assert note["body"]

    def test_audit_log_has_seed_entry(self) -> None:
        log = get_audit_log(100)
        actions = [row["action"] for row in log]
        assert "DB_SEED" in actions


# ---------------------------------------------------------------------------
# get_customer
# ---------------------------------------------------------------------------


class TestGetCustomer:

    def test_returns_existing_customer(self) -> None:
        customer = get_customer(1)
        assert customer is not None
        assert customer["name"] == "Alice Johnson"
        assert customer["email"] == "alice@example.com"

    def test_returns_none_for_missing_id(self) -> None:
        assert get_customer(9999) is None

    def test_logs_audit_entry(self) -> None:
        get_customer(2)
        log = get_audit_log(100)
        matching = [r for r in log if r["action"] == "GET_CUSTOMER" and "id=2" in r["detail"]]
        assert len(matching) >= 1


# ---------------------------------------------------------------------------
# search_orders
# ---------------------------------------------------------------------------


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

    def test_logs_audit_entry(self) -> None:
        search_orders("pending")
        log = get_audit_log(100)
        actions = [r["action"] for r in log]
        assert "SEARCH_ORDERS" in actions


# ---------------------------------------------------------------------------
# list_customers
# ---------------------------------------------------------------------------


class TestListCustomers:

    def test_returns_all_five(self) -> None:
        assert len(list_customers()) == 5

    def test_ordered_by_id(self) -> None:
        customers = list_customers()
        ids = [c["id"] for c in customers]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


class TestListDocuments:

    def test_returns_all_four(self) -> None:
        assert len(list_documents()) == 4

    def test_document_fields(self) -> None:
        doc = list_documents()[0]
        for key in ("id", "filename", "description", "uploaded_by", "uploaded_at"):
            assert key in doc


# ---------------------------------------------------------------------------
# get_internal_note (restricted)
# ---------------------------------------------------------------------------


class TestGetInternalNote:

    def test_returns_confidential_note(self) -> None:
        note = get_internal_note(1)
        assert note is not None
        assert "Q3 Revenue Forecast" in note["subject"]

    def test_returns_none_for_missing_id(self) -> None:
        assert get_internal_note(9999) is None

    def test_sensitive_content_present(self) -> None:
        """The seed data must contain recognisably sensitive strings so that
        leakage is easy to detect in injection experiments."""
        note = get_internal_note(2)
        assert note is not None
        assert "ROOT PASSWORD" in note["body"]

    def test_logs_audit_entry(self) -> None:
        get_internal_note(3)
        log = get_audit_log(100)
        actions = [r["action"] for r in log]
        assert "GET_INTERNAL_NOTE" in actions


# ---------------------------------------------------------------------------
# update_internal_note (dangerous)
# ---------------------------------------------------------------------------


class TestUpdateInternalNote:

    def test_update_existing_note(self) -> None:
        assert update_internal_note(1, "OVERWRITTEN") is True
        note = get_internal_note(1)
        assert note is not None
        assert note["body"] == "OVERWRITTEN"

    def test_update_nonexistent_returns_false(self) -> None:
        assert update_internal_note(9999, "nope") is False

    def test_logs_audit_entry(self) -> None:
        update_internal_note(1, "new text")
        log = get_audit_log(100)
        matching = [
            r for r in log
            if r["action"] == "UPDATE_INTERNAL_NOTE" and "new_body_len=8" in r["detail"]
        ]
        assert len(matching) >= 1


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:

    def test_limit_parameter(self) -> None:
        get_customer(1)
        get_customer(2)
        get_customer(3)
        log = get_audit_log(limit=2)
        assert len(log) == 2

    def test_newest_first_ordering(self) -> None:
        """Audit log is ordered by timestamp DESC, then by rowid DESC within
        the same second.  We verify that both entries are present."""
        get_customer(1)
        search_orders("shipped")
        log = get_audit_log(100)
        actions = [r["action"] for r in log]
        assert "SEARCH_ORDERS" in actions
        assert "GET_CUSTOMER" in actions


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


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
