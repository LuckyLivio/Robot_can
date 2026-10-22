"""Read-only artifact/link checks; saves an exclusive delivery receipt, no simulation."""
import functools
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

RUN = Path(__file__).resolve().parent
ROOT = RUN.parent.parent


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.urls.append(value)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = RUN / "delivery_validation.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    status = json.loads((RUN / "delivery_status.json").read_text(encoding="utf-8"))
    documents = [ROOT / name for name in status["navigation_updated"]]
    documents += [RUN / name for name in ("REPORT.md", "METHODS.md", "index.html")]
    documents += [ROOT / "robot_can_v2/README.md"]
    errors, local_links = [], []
    for doc in documents:
        content = doc.read_text(encoding="utf-8")
        if doc.suffix == ".html":
            parser = Links()
            parser.feed(content)
            urls = parser.urls
        else:
            content = re.sub(r"```.*?```", "", content, flags=re.S)
            urls = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        for url in urls:
            url = url.strip("<>")
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (doc.parent / unquote(parsed.path)).resolve()
            local_links.append({"document": doc.relative_to(ROOT).as_posix(), "url": url})
            if not target.exists():
                errors.append(f"Missing local target: {doc}: {url}")

    manifest = json.loads((RUN / "figures/manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["outputs_sha256"].items():
        path = RUN / "figures" / name
        if not path.is_file() or sha(path) != expected:
            errors.append(f"Figure output hash mismatch: {name}")
    verification = {}
    for name in ("preservation_check.json", "confirmation_state_validation.json",
                 "analysis/confirmation_trace_verification.json", "analysis/confirmation_physics_replay.json",
                 "analysis/io_recovery_verification.json"):
        value = json.loads((RUN / name).read_text(encoding="utf-8"))
        verification[name] = {"passed": value.get("passed"), "sha256": sha(RUN / name)}
        if not value.get("passed"):
            errors.append(f"Verification not passed: {name}")
    assert status["confirmation_executions"] == 1680
    assert status["independent_confirmation_starts"] == 60
    assert status["research_executions"] == 2040
    assert status["registered_plus_duplicate_executions"] == 2100
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    resources = [ROOT / "index.html", RUN / "index.html", RUN / "REPORT.md",
                 RUN / "analysis/confirmation/summary.json"]
    resources += [RUN / "figures" / (name + ".png") for name in (
        "recovery_factorial_success", "recovery_factorial_effects",
        "observation_confirmation_success", "queue_budget_candidate_effects")]
    responses = []
    try:
        for path in resources:
            relative = path.relative_to(ROOT).as_posix()
            with urlopen(f"http://127.0.0.1:{server.server_port}/{relative}", timeout=10) as response:
                payload = response.read()
                responses.append({"resource": relative, "status": response.status, "bytes": len(payload)})
                if payload != path.read_bytes():
                    errors.append(f"HTTP bytes differ: {relative}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
    result = {
        "passed": not errors, "utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)), "errors": errors,
        "documents_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in documents},
        "local_links_checked": len(local_links), "http_responses": responses,
        "figure_bundles": len(manifest["figures"]),
        "visual_inspection_scope": ["recovery_factorial_success.png", "observation_confirmation_effects.png", "queue_budget_candidate_effects.png"],
        "visual_scope_note": "These three PNGs were inspected with view_image; HTTP/hash checks are not browser rendering or PDF layout verification.",
        "verification_receipts": verification,
        "tests_note": "58 unit/regression tests previously passed before the frozen confirmation; this script does not rerun them or simulate new episodes.",
    }
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "links": len(local_links), "http_resources": len(responses), "figure_bundles": len(manifest["figures"]), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
