from pathlib import Path
import sys
import pandas as pd

from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset


WINDOW_SIZE = 168


def prepare_dataframes(reference: pd.DataFrame, current: pd.DataFrame):
    if "date_to" in reference.columns:
        del reference["date_to"]

    if "date_to" in current.columns:
        del current["date_to"]

    common_columns = [col for col in reference.columns if col in current.columns]
    reference = reference[common_columns].copy()
    current = current[common_columns].copy()

    valid_columns = []
    skipped_columns = []

    for col in common_columns:
        reference_non_null = reference[col].notna().sum()
        current_non_null = current[col].notna().sum()

        if reference_non_null == 0 or current_non_null == 0:
            skipped_columns.append(col)
        else:
            valid_columns.append(col)

    reference = reference[valid_columns].copy()
    current = current[valid_columns].copy()

    return reference, current, skipped_columns


def all_tests_passed_from_result(result_dict: dict) -> bool:
    if "tests" not in result_dict:
        return True

    for test in result_dict["tests"]:
        if "status" in test and test["status"] != "SUCCESS":
            return False

    return True


def run_station_test(current_path: Path, reference_path: Path, report_path: Path) -> bool:
    current_full = pd.read_csv(current_path)

    if not reference_path.exists():
        print(f"Reference file not found. Copying from current data to {reference_path}.")
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        current_full.to_csv(reference_path, index=False)

    reference_full = pd.read_csv(reference_path)

    current_window = current_full.tail(WINDOW_SIZE).copy()
    reference_window = reference_full.tail(WINDOW_SIZE).copy()

    reference, current, skipped_columns = prepare_dataframes(reference_window, current_window)

    if skipped_columns:
        print(f"Skipping empty columns for {current_path.name}: {', '.join(skipped_columns)}")

    if reference.shape[1] == 0 or current.shape[1] == 0:
        print(f"No valid columns left for drift testing in {current_path.name}. Updating reference only.")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        current_full.to_csv(reference_path, index=False)
        return True

    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(),
        ],
        include_tests=True,
    )

    result = report.run(reference_data=reference, current_data=current)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(report_path))

    result_dict = result.dict()
    passed = all_tests_passed_from_result(result_dict)

    if passed:
        print(f"Data tests passed for {current_path.name}.")
        current_full.to_csv(reference_path, index=False)
        return True

    print(f"Data tests failed for {current_path.name}.")
    return False


def main():
    project_root = Path(__file__).resolve().parents[2]
    current_dir = project_root / "data" / "preprocessed" / "air"
    reference_dir = project_root / "data" / "reference" / "air"
    reports_dir = project_root / "reports" / "data_testing" / "air"

    if not current_dir.exists():
        print(f"Current data directory does not exist: {current_dir}")
        sys.exit(1)

    current_files = sorted(current_dir.glob("*.csv"))

    if not current_files:
        print(f"No CSV files found in {current_dir}")
        sys.exit(1)

    failed_stations = []

    for current_path in current_files:
        station_name = current_path.stem
        reference_path = reference_dir / f"{station_name}.csv"
        report_path = reports_dir / f"{station_name}.html"

        passed = run_station_test(
            current_path=current_path,
            reference_path=reference_path,
            report_path=report_path,
        )

        if not passed:
            failed_stations.append(station_name)

    if failed_stations:
        print("Data tests failed for stations:", ", ".join(failed_stations))
        sys.exit(1)

    print("Data tests passed for all stations.")
    sys.exit(0)


if __name__ == "__main__":
    main()