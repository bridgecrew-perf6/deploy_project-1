import argparse
import ml.train_model as train_model
import ml.eval_model as eval_model
import ml.clean_data as clean_data
import logging


def go(args):
    """
    Execute ml pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "clean data":
        logging.info("Basic cleaning procedure started")
        clean_data.execute_cleaning()

    if args.action == "all" or args.action == "train model":
        logging.info("Train model procedure started")
        train_model.train()

    if args.action == "all" or args.action == "evaluate model":
        logging.info("Evaluation procedure started")
        eval_model.evaluate_model()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=[
            "clean data",
            "train model",
            "evaluate model",
            "all"
        ],
        default="all",
        help="Pipeline action"
    )

    main_args = parser.parse_args()

    go(main_args)
