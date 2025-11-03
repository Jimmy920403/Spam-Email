from scripts.preprocess import preprocess_text


def test_preprocess_basic():
    s = "Hello, WORLD!! This is a test."
    out = preprocess_text(s)
    assert "hello" in out
    assert "world" in out
    assert "!!" not in out
