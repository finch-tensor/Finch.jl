# SPDX-FileCopyrightText: 2026 Finch Developers
# SPDX-License-Identifier: MIT

"""Send one request to the persistent Finch server, with bounded waits."""

import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def request(command, args):
    fifo = os.environ.get("FINCH_SERVER_FIFO")
    if not fifo:
        raise RuntimeError("FINCH_SERVER_FIFO environment variable is not set")
    server_pid = os.environ.get("FINCH_SERVER_PID")
    timeout = float(os.environ.get("FINCH_SERVER_TIMEOUT", "300"))
    if not 0 < timeout < float("inf"):
        raise ValueError("FINCH_SERVER_TIMEOUT must be a finite positive number")
    deadline = time.monotonic() + timeout

    def check_server(stage):
        if server_pid:
            try:
                os.kill(int(server_pid), 0)
            except ProcessLookupError:
                raise RuntimeError(f"Finch server exited while {stage}") from None
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout:g}s while {stage}")

    with tempfile.TemporaryDirectory(prefix="finch-response-") as directory:
        response = Path(directory) / "response.json"
        payload = (json.dumps({
            "cmd": command, "args": args, "response": str(response),
        }) + "\n").encode()
        while True:
            check_server("connecting to Finch server")
            try:
                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
                time.sleep(0.01)
        try:
            # Long paths can make requests larger than one nonblocking write.
            remaining = memoryview(payload)
            while remaining:
                check_server("sending request to Finch server")
                try:
                    written = os.write(fd, remaining)
                    remaining = remaining[written:]
                except BlockingIOError:
                    time.sleep(0.01)
        finally:
            os.close(fd)

        while not response.exists():
            check_server(f"waiting for Finch server response to {command}")
            time.sleep(0.01)
        result = json.loads(response.read_text())
        if result.get("error"):
            print(result["error"], file=sys.stderr)
        return int(result.get("exit_code", 1))


if __name__ == "__main__":
    try:
        sys.exit(request(sys.argv[1], sys.argv[2:]))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
