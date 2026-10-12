# SPDX-FileCopyrightText: 2026 Finch Developers
# SPDX-License-Identifier: MIT

"""Regression tests for the compliance harness; no Julia packages required."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
COMMANDS = ("binsparse_to_npy", "npy_to_binsparse", "binsparse_to_binsparse")


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fifo = Path(self.directory.name) / "requests"
        os.mkfifo(self.fifo)
        self.env = {
            **os.environ,
            "FINCH_SERVER_FIFO": str(self.fifo),
            "FINCH_SERVER_PID": str(os.getpid()),
            "FINCH_SERVER_TIMEOUT": "0.3",
            "TMPDIR": self.directory.name,
        }

    def call(self, command=COMMANDS[0], *args):
        return subprocess.run(
            [str(HERE / command), *args], env=self.env,
            capture_output=True, text=True, timeout=5,
        )

    def start_server(self, code):
        server = subprocess.Popen([sys.executable, "-c", code, str(self.fifo)])

        def stop():
            if server.poll() is None:
                server.kill()
            server.wait(timeout=5)

        self.addCleanup(stop)
        self.env["FINCH_SERVER_PID"] = str(server.pid)
        self.env["FINCH_SERVER_TIMEOUT"] = "3"
        return server

    def test_commands_and_arguments(self):
        self.start_server('''
import json, pathlib, sys
with open(sys.argv[1]) as fifo:
    for line in fifo:
        request = json.loads(line)
        pathlib.Path(request["args"][0]).write_text(json.dumps(request))
        response = pathlib.Path(request["response"])
        temporary = response.with_suffix(".tmp")
        temporary.write_text('{"exit_code": 0}')
        temporary.replace(response)
''')
        # Keep the FIFO open across requests, as the real server does by reopening.
        fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(os.close, fd)
        for command in COMMANDS:
            with self.subTest(command=command):
                record = Path(self.directory.name) / "record.json"
                args = [str(record), 'path with spaces and "quotes"', "line\nbreak", "x" * 8192]
                result = self.call(command, *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                request = json.loads(record.read_text())
                self.assertEqual(request["cmd"], command)
                self.assertEqual(request["args"], args)
                self.assertFalse(Path(request["response"]).parent.exists())

    def test_server_error(self):
        self.start_server('''
import json, pathlib, sys
with open(sys.argv[1]) as fifo:
    request = json.loads(fifo.readline())
response = pathlib.Path(request["response"])
temporary = response.with_suffix(".tmp")
temporary.write_text('{"exit_code": 7, "error": "conversion failed"}')
temporary.replace(response)
''')
        result = self.call()
        self.assertEqual(result.returncode, 7)
        self.assertIn("conversion failed", result.stderr)

    def test_no_fifo_reader(self):
        result = self.call()
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)
        self.assertIn("connecting", result.stderr)

    def test_missing_response(self):
        fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(os.close, fd)
        result = self.call()
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)
        self.assertIn("waiting for Finch server response", result.stderr)

    def test_full_fifo(self):
        fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(os.close, fd)
        try:
            while True:
                os.write(fd, b"x" * os.fpathconf(fd, "PC_PIPE_BUF"))
        except BlockingIOError:
            pass
        result = self.call()
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)
        self.assertIn("sending request", result.stderr)

    def test_server_exit(self):
        server = self.start_server("pass")
        server.wait(timeout=5)
        result = self.call()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Finch server exited", result.stderr)


class RunnerTests(unittest.TestCase):
    def run_harness(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mock_bin = directory / "bin"
            mock_bin.mkdir()
            tests = directory / "tests"
            (tests / ".git").mkdir(parents=True)
            for command in ("git", "pixi", "julia"):
                mock = mock_bin / command
                mock.write_text(f"#!{sys.executable}\n" + '''
import os, pathlib, sys, time
if pathlib.Path(sys.argv[0]).name != "julia" or sys.argv[2] == "-e":
    sys.exit(0)
pathlib.Path(os.environ["PID_RECORD"]).write_text(str(os.getpid()))
if os.environ["SERVER_MODE"] == "crash":
    sys.exit(12)
if os.environ["SERVER_MODE"] == "ignore-shutdown":
    pathlib.Path(sys.argv[4]).touch()
time.sleep(60)
''')
                mock.chmod(0o755)
            record = directory / "server.pid"
            result = subprocess.run(
                ["bash", str(HERE / "run-binsparse-tests.sh")],
                env={
                    **os.environ,
                    "PATH": str(mock_bin) + os.pathsep + os.environ["PATH"],
                    "BINSPARSE_BUILD_DIR": str(directory / "build"),
                    "BINSPARSE_TESTS_DIR": str(tests),
                    "FINCH_SERVER_STARTUP_TIMEOUT": "1",
                    "SERVER_MODE": mode,
                    "PID_RECORD": str(record),
                    "TMPDIR": str(directory),
                },
                capture_output=True, text=True, timeout=15,
            )
            with self.assertRaises(ProcessLookupError):
                os.kill(int(record.read_text()), 0)
            self.assertEqual(list(directory.glob("tmp.*")), [])
            return result

    def test_startup_failure_does_not_hang_cleanup(self):
        result = self.run_harness("crash")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Finch server failed to start", result.stderr)

    def test_startup_timeout(self):
        result = self.run_harness("never-ready")
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out waiting for Finch server to start", result.stderr)

    def test_cleanup_with_unresponsive_server(self):
        result = self.run_harness("ignore-shutdown")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
