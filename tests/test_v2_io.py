"""Tests for shared JSON artifact persistence."""

import json

from v2.io import is_valid_json_file, save_json


def test_is_valid_json_file_false_for_missing_and_empty(tmp_path):
    assert not is_valid_json_file(tmp_path / "nope.json")
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert not is_valid_json_file(empty)


def test_is_valid_json_file_false_for_corrupt(tmp_path):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text('{"id": 1', encoding="utf-8")
    assert not is_valid_json_file(corrupt)


def test_save_json_is_idempotent(tmp_path):
    path = tmp_path / "a.json"
    save_json({"x": 1}, path)
    save_json({"x": 2}, path)

    with path.open("r", encoding="utf-8") as file_obj:
        assert json.load(file_obj)["x"] == 1


def test_save_json_replaces_corrupt_artifact(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"broken"', encoding="utf-8")

    save_json({"x": 3}, path)

    with path.open("r", encoding="utf-8") as file_obj:
        assert json.load(file_obj)["x"] == 3