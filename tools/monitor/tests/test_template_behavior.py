import re
import unittest
from pathlib import Path


class MonitorTemplateBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.template = (Path(__file__).resolve().parents[1] / "templates" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_running_view_uses_single_downsampled_realtime_chart(self):
        self.assertIn("function downsampleSeries", self.template)
        self.assertIn('canvas id="realtimeChart"', self.template)

        running_block = re.search(
            r"function renderRealtimeExperiment\(data\).*?renderRealtimeChart\(chartData\);",
            self.template,
            re.S,
        )
        self.assertIsNotNone(running_block)
        block = running_block.group(0)
        self.assertIn('canvas id="realtimeChart"', block)
        self.assertNotIn("renderCompletedCharts", block)


if __name__ == "__main__":
    unittest.main()
