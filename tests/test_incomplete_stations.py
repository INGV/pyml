import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import pyml


PROJECT = Path(__file__).resolve().parents[1]
CORRECTIONS = PROJECT / "dbona_magnitudes_stations_corrections_extended_mq.csv"


class IncompleteStationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corrections = pyml.read_table(CORRECTIONS)

    def setUp(self):
        self.request = json.loads(
            (PROJECT / "examples" / "input" / "test_period.json").read_text()
        )
        self.complete_request = copy.deepcopy(self.request)
        self.complete_request["data"]["amplitudes"] = [
            a for a in self.complete_request["data"]["amplitudes"]
            if a["sta"] != "AVL"
        ]

    def analyze(self, request):
        return pyml.analyze(request, self.corrections)

    def avl_channels(self, response):
        return [a for a in response["stationmagnitudes"] if a["sta"] == "AVL"]

    def avl_logs(self, response):
        return [a for a in response["log"]["stationmagnitudes"] if a["sta"] == "AVL"]

    def add_north_component(self, request):
        north = copy.deepcopy(request["data"]["amplitudes"][0])
        north["cha"] = "HNN"
        request["data"]["amplitudes"].append(north)
        return north

    def test_single_component_is_only_logged_for_both_orientations(self):
        baseline = self.analyze(self.complete_request)
        for orientation in ("E", "N"):
            for other_stations in (True, False):
                with self.subTest(orientation=orientation, other_stations=other_stations):
                    request = copy.deepcopy(self.request)
                    request["data"]["amplitudes"][0]["cha"] = "HN" + orientation
                    if not other_stations:
                        request["data"]["amplitudes"] = request["data"]["amplitudes"][:1]
                    response = self.analyze(request)
                    self.assertEqual(self.avl_channels(response), [])
                    if other_stations:
                        self.assertEqual(response["magnitudes"], baseline["magnitudes"])
                        self.assertEqual(response["stationmagnitudes"], baseline["stationmagnitudes"])
                        self.assertEqual(response["log"]["magnitude"], baseline["log"]["magnitude"])
                    else:
                        self.assertEqual(response["magnitudes"], {})
                        self.assertEqual(response["stationmagnitudes"], [])
                    self.assertEqual(self.avl_logs(response), [
                        {
                            "net": "IT", "sta": "AVL", "loc": "--", "band_inst": "HN",
                            "status": "critical",
                            "summary": "Station skipped due to missing channel",
                            "extended": "ML " + law,
                            "channels": [
                                {"orientation": ori,
                                 "status": "ok" if ori == orientation else "critical",
                                 "summary": "" if ori == orientation else "missing",
                                 "extended": ""}
                                for ori in ("N", "E")
                            ],
                        }
                        for law in ("HuttonBoore", "DiBona")
                    ])

    def test_complete_pair_with_a_zero_peak_is_retained(self):
        north = self.add_north_component(self.request)
        north["amp1"] = 0.0
        response = self.analyze(self.request)
        channels = self.avl_channels(response)
        self.assertEqual([a["cha"] for a in channels], ["HNN", "HNE"])
        self.assertEqual(channels[0]["amp1"], 0.0)
        self.assertEqual(self.avl_logs(response), [])
        for law in ("hb", "db"):
            self.assertEqual(response["magnitudes"][law]["totsta"], 5)
            for channel in channels:
                self.assertIsInstance(channel[law]["ml"], float)

    def test_unusable_partner_does_not_make_a_complete_pair(self):
        baseline = self.analyze(self.complete_request)
        for peaks in ((None, 0.0457), (False, 0.0457), (0.0, 0.0)):
            with self.subTest(peaks=peaks):
                request = copy.deepcopy(self.request)
                north = self.add_north_component(request)
                north["amp1"], north["amp2"] = peaks
                response = self.analyze(request)
                self.assertEqual(self.avl_channels(response), [])
                self.assertEqual(response["magnitudes"], baseline["magnitudes"])
                logs = self.avl_logs(response)
                self.assertEqual(len(logs), 2)
                for item in logs:
                    self.assertEqual(item["summary"], "Station skipped due to missing channel")
                    self.assertEqual(item["channels"][0]["status"], "critical")

    def test_components_from_different_groups_are_not_paired(self):
        baseline = self.analyze(self.complete_request)
        for field, value in (("cha", "HHN"), ("net", "XX"), ("loc", "01")):
            with self.subTest(field=field, value=value):
                request = copy.deepcopy(self.request)
                north = self.add_north_component(request)
                north[field] = value
                response = self.analyze(request)
                self.assertEqual(self.avl_channels(response), [])
                self.assertEqual(response["magnitudes"], baseline["magnitudes"])
                self.assertEqual(len(self.avl_logs(response)), 4)

    def test_complete_stations_with_zero_weight_are_retained(self):
        response = self.analyze(self.complete_request)
        channels = [a for a in response["stationmagnitudes"] if a["sta"] in ("FDMO", "MTRZ")]
        self.assertEqual(len(channels), 4)
        for channel in channels:
            for law in ("hb", "db"):
                self.assertIsInstance(channel[law]["ml"], float)
                self.assertEqual(channel[law]["w"], 0.0)

    def test_complete_pair_below_mindist_keeps_existing_output(self):
        self.add_north_component(self.request)
        for amplitude in self.request["data"]["amplitudes"]:
            if amplitude["sta"] == "AVL":
                amplitude["ipodistance(km)"] = 5.0
                amplitude["epidistance(km)"] = 4.0
        response = self.analyze(self.request)
        channels = self.avl_channels(response)
        self.assertEqual(len(channels), 2)
        for channel in channels:
            self.assertEqual(channel["hb"], {})
            self.assertEqual(channel["db"], {})
        logs = self.avl_logs(response)
        self.assertEqual(len(logs), 2)
        for item in logs:
            self.assertEqual(item["status"], "warning")
            self.assertIn("skipped due to ipodist", item["summary"])
            self.assertEqual(item["extended"], "Distance is 5.0 km")

    def test_json_cli_omits_incomplete_station(self):
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(self.request))), redirect_stdout(output):
            code = pyml.main([
                "--in_file_name", "-", "--in_file_format", "json",
                "--out_format", "json", "--dbona_corr", str(CORRECTIONS),
            ])
        self.assertEqual(code, 0)
        response = json.loads(output.getvalue())
        self.assertEqual(self.avl_channels(response), [])
        self.assertEqual(len(response["stationmagnitudes"]), 8)
        self.assertEqual(len(self.avl_logs(response)), 2)


if __name__ == "__main__":
    unittest.main()
