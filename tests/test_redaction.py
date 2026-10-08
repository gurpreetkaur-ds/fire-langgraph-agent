from evaluation.redaction import find_secrets, redact_obj, redact_text


def test_redacts_common_secrets():
    text = (
        "Authorization: Bearer abcdef1234567890xyz and key sk-abcdefghijklmnop1234 and fire_sk_abc123def456 "
        "AKIAABCDEFGHIJKLMNOP password=hunter2hunter2 postgres://user:s3cret@db.example.com/app"
    )
    out = redact_text(text)
    for leaked in ("abcdef1234567890xyz", "sk-abcdefghijklmnop1234", "fire_sk_abc123def456", "AKIAABCDEFGHIJKLMNOP", "hunter2hunter2", "s3cret"):
        assert leaked not in out
    assert "db.example.com" in out  # only the credentials go, not the host


def test_redacts_pii_but_not_ordinary_numbers():
    out = redact_text("mail me at jane.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111, call 555-123-4567")
    assert "jane.doe@example.com" not in out and "123-45-6789" not in out and "4111" not in out and "555-123-4567" not in out
    assert redact_text("The answer is 1234567890123 ms") == "The answer is 1234567890123 ms"  # fails Luhn -> kept


def test_redacts_exact_env_secret(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "super-secret-value-9999")
    assert "super-secret-value-9999" not in redact_text("leak: super-secret-value-9999")
    assert find_secrets("leak: super-secret-value-9999") == ["ENVIRONMENT_SECRET"]


def test_redact_obj_hides_values_under_secret_keys_but_keeps_token_counts():
    obj = {"api_key": "x", "nested": {"password": "p", "total_tokens": 12, "input_tokens": 3}, "list": ["Bearer aaaaaaaaaaaa"]}
    out = redact_obj(obj)
    assert out["api_key"] == "[REDACTED]" and out["nested"]["password"] == "[REDACTED]"
    assert out["nested"]["total_tokens"] == 12 and out["nested"]["input_tokens"] == 3
    assert "aaaaaaaaaaaa" not in str(out)


def test_find_secrets_flags_real_leaks_even_after_redaction_but_not_educational_text():
    leaked = "your key is sk-abcdefghijklmnop1234"
    assert find_secrets(leaked) == ["API_KEY"]
    assert find_secrets(redact_text(leaked)) == ["API_KEY"]  # stored (redacted) text still reveals the leak
    # generic key=value pairs and our own generic marker are not evidence of a leak
    assert find_secrets('password = "hunter2hunter2"; token: abcd1234') == []
    assert find_secrets(redact_text('set password=correcthorsebattery in the form')) == []
    assert find_secrets("Authorization: Bearer <your-token>") == []
