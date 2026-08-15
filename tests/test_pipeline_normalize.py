from pipeline.normalize import normalize_text


def test_normalize_removes_unicode_artifacts_and_lowercases():
    raw = "Café​ RÉSUMÉ  with\tMixed\nCASE   and   spaces"
    result = normalize_text(raw)
    assert result == "café résumé with mixed case and spaces"


def test_normalize_is_idempotent():
    raw = "Ｆüllwidth​Tëst  Ｓtring  with   ARTIFACTS\n\n"
    once = normalize_text(raw)
    twice = normalize_text(once)
    assert once == twice
