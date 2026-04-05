from pathlib import Path
import sys
import great_expectations as gx


def main():
    project_root = Path(__file__).resolve().parents[1]
    gx_root = project_root / "gx"
    data_root = project_root / "data"

    context = gx.get_context(context_root_dir=str(gx_root))

    datasource_name = "air_quality"
    data_asset_name = "air_quality_data"
    expectation_suite_name = "air_quality_suite"
    checkpoint_name = "air_quality_checkpoint"

    datasource = context.sources.add_or_update_pandas_filesystem(
        name=datasource_name,
        base_directory=str(data_root),
    )

    try:
        asset = datasource.get_asset(data_asset_name)
    except Exception:
        asset = datasource.add_csv_asset(
            name=data_asset_name,
            batching_regex=r"preprocessed/air/.*\.csv",
        )

    batch_request = asset.build_batch_request()

    batch_list = datasource.get_batch_list_from_batch_request(batch_request)

    if not batch_list:
        print("No batches found for validation.")
        sys.exit(1)

    context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name,
            }
        ],
    )

    checkpoint = context.get_checkpoint(checkpoint_name)
    checkpoint_result = checkpoint.run(run_id="air_quality_run")

    context.build_data_docs()

    if checkpoint_result["success"]:
        print("Validation passed for all stations!")
        sys.exit(0)
    else:
        print("Validation failed for at least one station!")
        sys.exit(1)


if __name__ == "__main__":
    main()