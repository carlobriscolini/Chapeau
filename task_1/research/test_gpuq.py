import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from research import gpuq


def entry(gpu=False, writes=(), reads=()):
    return {"gpu": gpu, "writes": list(writes), "reads": list(reads)}


class GpuQueueTest(unittest.TestCase):
    def test_conflicts(self):
        self.assertTrue(gpuq.conflicts(entry(gpu=True), entry(gpu=True)))
        self.assertFalse(gpuq.conflicts(entry(gpu=True), entry(reads=["/d"])))
        self.assertFalse(gpuq.conflicts(entry(reads=["/d"]), entry(reads=["/d"])))
        self.assertTrue(gpuq.conflicts(entry(writes=["/d"]), entry(reads=["/d"])))
        self.assertTrue(gpuq.conflicts(entry(reads=["/d"]), entry(writes=["/d"])))
        self.assertFalse(gpuq.conflicts(entry(writes=["/a"]), entry(writes=["/b"])))

    def test_writers_of_one_path_run_one_after_the_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); log = tmp / "log"
            env = {"WMLOC_QUEUE_DIR": str(tmp / "q"), "PATH": "/usr/bin:/bin"}
            job = f"echo start >> {log}; sleep 1; echo end >> {log}"
            cmd = [sys.executable, "-m", "research.gpuq", "run", "--no-gpu", "--writes", str(tmp / "d")]
            procs = []
            for i in range(2):
                procs.append(subprocess.Popen(cmd + ["--label", f"j{i}", "--", "sh", "-c", job], env=env))
                time.sleep(0.3)
            self.assertTrue(all(p.wait(30) == 0 for p in procs))
            self.assertEqual(log.read_text().split(), ["start", "end", "start", "end"])
            self.assertFalse(list((tmp / "q" / "entries").glob("*.json")))

    def test_failing_job_returns_its_code_and_leaves_no_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {"WMLOC_QUEUE_DIR": tmp, "PATH": "/usr/bin:/bin"}
            code = subprocess.call([sys.executable, "-m", "research.gpuq", "run", "--no-gpu", "--label", "f",
                                    "--", "sh", "-c", "exit 3"], env=env)
            self.assertEqual(code, 3)
            self.assertFalse(list((Path(tmp) / "entries").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
