import sys
import great_expectations as gx


def main():
    context = gx.get_context()

    checkpoint_name = "air_quality_checkpoint"

    checkpoint = context.get_checkpoint(checkpoint_name)

    checkpoint_result = checkpoint.run(
        run_id="air_quality_run"
    )

    context.build_data_docs()

    if checkpoint_result["success"]:
        print("Validation passed for all stations!")
        sys.exit(0)
    else:
        print("Validation failed for at least one station!")
        sys.exit(1)


if __name__ == "__main__":
    main()