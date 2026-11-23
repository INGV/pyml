import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class StationMetadataTest(unittest.TestCase):
    def setUp(self):
        self.request = json.loads(
            (PROJECT / "tests" / "fixtures" / "cptp.json").read_text()
        )

    def run_pyml(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            input_path.write_text(json.dumps(self.request))
            result = subprocess.run(
                [
                    sys.executable, str(PROJECT / "pyml.py"),
                    "--in_file_name", str(input_path),
                    "--in_file_format", "json", "--out_format", "json",
                    "--dbona_corr", str(
                        PROJECT / "dbona_magnitudes_stations_corrections_extended_mq.csv"
                    ),
                ],
                cwd=directory, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def both_components(self, response):
        stations = response["stationmagnitudes"]
        self.assertEqual([station["cha"] for station in stations], ["HHN", "HHE"])
        return stations

    def test_zero_azimuth_is_a_number_on_both_components(self):
        response = self.run_pyml()
        for station in self.both_components(response):
            self.assertIsInstance(station["azimuth"], (int, float))
            self.assertNotIsInstance(station["azimuth"], bool)
            self.assertEqual(station["azimuth"], 0.0)

    def test_cptp_magnitudes_distances_and_readings_are_preserved(self):
        response = self.run_pyml()
        self.assertAlmostEqual(response["magnitudes"]["hb"]["ml"], 3.238701070990148, places=12)
        self.assertAlmostEqual(response["magnitudes"]["db"]["ml"], 3.326183568153756, places=12)
        for station, amplitude in zip(
            self.both_components(response), self.request["data"]["amplitudes"]
        ):
            self.assertAlmostEqual(station["ep_distance_km"], 147.57165046575133, places=10)
            self.assertAlmostEqual(station["orig_distance"], 148.02615749314674, places=10)
            for field in ("net", "sta", "cha", "loc", "time1", "time2"):
                self.assertEqual(station[field], amplitude[field], field)
            for field in ("amp1", "amp2", "lat", "lon", "elev"):
                self.assertAlmostEqual(station[field], amplitude[field], places=12, msg=field)
        self.assertEqual(response["log"]["magnitude"]["hb"]["summary"], "")

    def test_small_longitude_changes_preserve_nonzero_bearings(self):
        for longitude, expected in (
            (14.16250001, 3.3760522604080265e-7),
            (14.16249999, 359.9999996623948),
        ):
            with self.subTest(longitude=longitude):
                self.request["data"]["origin"]["lon"] = longitude
                for station in self.both_components(self.run_pyml()):
                    self.assertAlmostEqual(station["azimuth"], expected, places=10)

    def test_southward_bearing_keeps_the_station_to_event_convention(self):
        self.request["data"]["origin"]["lat"] = 36.737
        for station in self.both_components(self.run_pyml()):
            self.assertEqual(station["azimuth"], 180.0)

    def test_sea_level_station_is_retained(self):
        for amplitude in self.request["data"]["amplitudes"]:
            amplitude["elev"] = 0.0
        for station in self.both_components(self.run_pyml()):
            self.assertEqual(station["elev"], 0.0)
            self.assertEqual(station["azimuth"], 0.0)

    def test_zero_latitude_and_longitude_are_valid_coordinates(self):
        for latitude, longitude in ((0.0, 14.1625), (37.4075, 0.0), (0.0, 0.0)):
            with self.subTest(latitude=latitude, longitude=longitude):
                self.request["data"]["origin"].update(lat=latitude + 1.3295, lon=longitude)
                for amplitude in self.request["data"]["amplitudes"]:
                    amplitude.update(lat=latitude, lon=longitude)
                for station in self.both_components(self.run_pyml()):
                    self.assertEqual(station["lat"], latitude)
                    self.assertEqual(station["lon"], longitude)
                    self.assertEqual(station["azimuth"], 0.0)

    def test_negative_elevation_is_preserved(self):
        for amplitude in self.request["data"]["amplitudes"]:
            amplitude["elev"] = -20.0
        for station in self.both_components(self.run_pyml()):
            self.assertEqual(station["elev"], -20.0)

    def test_missing_coordinates_are_not_replaced_with_zero(self):
        original = copy.deepcopy(self.request)
        for field in ("lat", "lon", "elev"):
            for missing in (None, "absent"):
                with self.subTest(field=field, missing=missing):
                    self.request = copy.deepcopy(original)
                    for amplitude in self.request["data"]["amplitudes"]:
                        if missing == "absent":
                            amplitude.pop(field)
                        else:
                            amplitude[field] = None
                    self.assertEqual(self.run_pyml()["stationmagnitudes"], [])

    def test_uncomputed_azimuth_is_json_null(self):
        for amplitude in self.request["data"]["amplitudes"]:
            amplitude["ipodistance(km)"] = 148.02615749314674
            amplitude["epidistance(km)"] = 147.57165046575133
        for station in self.both_components(self.run_pyml()):
            self.assertIsNone(station["azimuth"])
            self.assertIsNone(station["ep_distance_delta"])


if __name__ == "__main__":
    unittest.main()
