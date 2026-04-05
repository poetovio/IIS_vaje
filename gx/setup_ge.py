from pathlib import Path
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
        data_asset = datasource.get_asset(data_asset_name)
    except Exception:
        data_asset = datasource.add_csv_asset(
            name=data_asset_name,
            batching_regex=r"preprocessed/air/.*\.csv",
        )

    context.add_or_update_expectation_suite(
        expectation_suite_name=expectation_suite_name
    )

    batch_request = data_asset.build_batch_request()

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=expectation_suite_name,
    )

    validator.expect_table_columns_to_match_set(
        column_set=["date_to", "PM10", "PM2.5"]
    )

    validator.expect_column_values_to_not_be_null("date_to")
    validator.expect_column_values_to_be_unique("date_to")

    validator.expect_column_values_to_be_between(
        "PM10",
        min_value=0,
        max_value=300,
        mostly=1.0,
    )

    validator.expect_column_values_to_be_between(
        "PM2.5",
        min_value=0,
        max_value=300,
        mostly=1.0,
    )

    validator.expect_table_row_count_to_be_between(min_value=24)

    validator.save_expectation_suite(discard_failed_expectations=False)

    context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": expectation_suite_name,
            }
        ],
    )

    print("Great Expectations setup completed successfully.")


if __name__ == "__main__":
    main()