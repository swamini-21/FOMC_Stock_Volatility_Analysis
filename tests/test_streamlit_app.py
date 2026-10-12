from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "fed_policy_intelligence" / "app" / "streamlit_app.py"
FIXTURE = ROOT / "tests" / "fixtures" / "dashboard"


def _app(monkeypatch) -> AppTest:
    monkeypatch.setenv("FPI_DASHBOARD_DATA_DIR", str(FIXTURE))
    return AppTest.from_file(APP, default_timeout=10).run()


def test_streamlit_pages_render_from_cached_fixture(monkeypatch) -> None:
    app = _app(monkeypatch)

    assert not app.exception
    assert app.title[0].value == "Federal Reserve Policy Intelligence"
    assert [metric.value for metric in app.metric] == ["2", "2", "2024", "2024"]

    for page, heading in (
        ("Announcement explorer", "Announcement explorer"),
        ("Policy trends", "Historical policy trends"),
        ("Market reactions", "Market-reaction explorer"),
        ("Statistical results", "Statistical results"),
        ("Methodology", "Methodology and limitations"),
    ):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
        assert app.header[0].value == heading


def test_streamlit_reports_missing_cache_without_external_calls(monkeypatch) -> None:
    monkeypatch.setenv("FPI_DASHBOARD_DATA_DIR", str(ROOT / "tests" / "fixtures" / "missing"))

    app = AppTest.from_file(APP, default_timeout=10).run()

    assert not app.exception
    assert app.error
    assert "Dashboard inputs are missing" in app.error[0].value
