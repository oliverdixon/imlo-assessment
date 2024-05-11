from itertools import islice


def format_tuples(final_epoch: int) -> None:
    """
    Format the output of the ModelDriver for the purposes of PGFPlots/TikZ and print to stdout

    :param final_epoch: The index of the final epoch to include in the output
    """
    with open("data/losses-series.txt", "r") as file:
        dataset: list[tuple[int, float]]
        for dataset in list(map(eval, file.readlines())):
            for record in islice(dataset, 0, final_epoch + 1):
                print(f"({record[0]}, {record[1]:.0f}) ", end="")
            print("\n")


if __name__ == "__main__":
    format_tuples(118)
