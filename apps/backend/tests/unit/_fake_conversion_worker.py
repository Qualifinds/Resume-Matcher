"""Stand-in for ``python -m app.document_conversion`` in the process-supervision tests.

Same command line (``<input> <result>``); the behaviour comes from the environment so a test can
make the "converter" hang, crash or answer without needing a PDF that really breaks pdfminer:

    FAKE_CONVERSION_MODE   ok | hang | hold | crash | silent | garbage | validation | limit | other
    FAKE_CONVERSION_DIR    directory where it leaves ``started-<pid>`` and ``beat-<pid>`` files

``hang`` never ends (and heartbeats, so a test can prove it stopped beating once killed); ``hold``
heartbeats until ``<dir>/release`` exists and then answers ok.
"""

import json
import os
import sys
import time
from pathlib import Path

mode = os.environ.get("FAKE_CONVERSION_MODE", "ok")
control = (
    Path(os.environ["FAKE_CONVERSION_DIR"])
    if "FAKE_CONVERSION_DIR" in os.environ
    else None
)
result_path = Path(sys.argv[2])
pid = os.getpid()


def beat() -> None:
    if control is not None:
        (control / f"beat-{pid}").write_text(str(time.time()))


def reply(payload: dict[str, str]) -> None:
    result_path.write_text(json.dumps(payload), encoding="utf-8")


if control is not None:
    (control / f"started-{pid}").write_text(sys.argv[1])

if mode == "hang":
    while True:
        beat()
        time.sleep(0.05)
elif mode == "hold":
    while control is None or not (control / "release").exists():
        beat()
        time.sleep(0.05)
    reply({"status": "ok", "text": "fake text"})
elif mode == "crash":
    print("fake converter blew up", file=sys.stderr)
    sys.exit(3)
elif mode == "silent":
    pass
elif mode == "garbage":
    result_path.write_text("{not json", encoding="utf-8")
elif mode == "validation":
    reply(
        {
            "status": "error",
            "kind": "validation",
            "type": "DocumentValidationError",
            "message": "bad file",
        }
    )
elif mode == "limit":
    reply(
        {
            "status": "error",
            "kind": "resource_limit",
            "type": "DocumentResourceLimitError",
            "message": "too big",
        }
    )
elif mode == "other":
    reply(
        {
            "status": "error",
            "kind": "other",
            "type": "KeyError",
            "message": "converter exploded",
        }
    )
else:
    reply({"status": "ok", "text": "fake text"})
