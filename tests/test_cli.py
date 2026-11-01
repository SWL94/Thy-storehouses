import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_outputs_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = [Path(tmp) / 'a', Path(tmp) / 'b']
            for output in outputs:
                proc = subprocess.run([sys.executable, '-m', 'mm1', '--warmup', '20',
                                       '--duration', '500', '--replications', '4',
                                       '--seed', '42', '--output', str(output)],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn('mean_wait', proc.stdout)
                for name in ('comparison.png', 'replication_convergence.png'):
                    self.assertEqual((output / name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
                meta = json.loads((output / 'metadata.json').read_text())
                self.assertEqual(meta['root_seed'], 42)
                with (output / 'replications.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 4)
                with (output / 'summary.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 5)
            for name in ('replications.csv', 'summary.csv'):
                self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes())

    def test_unstable_cli_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [sys.executable, '-m', 'mm1', '--arrival-rate', '1.1',
                       '--warmup', '0', '--duration', '30', '--replications', '3',
                       '--output', tmp]
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn('allow-unstable', rejected.stderr)
            accepted = subprocess.run(command + ['--allow-unstable'], capture_output=True, text=True)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn('finite-horizon', accepted.stdout)
            with (Path(tmp) / 'summary.csv').open() as f:
                self.assertTrue(all(row['theory'] == '' for row in csv.DictReader(f)))
