import pandas as pd
import numpy as np
#for seasonal anlysis only 
def add_season_column(df):
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",

        3: "Summer",
        4: "Summer",
        5: "Summer",

        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",

        10: "Post-Monsoon",
        11: "Post-Monsoon"
    }

    df["season"] = df["month"].map(season_map)

    return df

def create_intermediate_summary(input_file, output_file,group_columns,chunksize=100_000,append=False,transform_fn=None):
    """we are going to process the file chunk by chunk so going to save the the summary of each chunk in intermediate file and then by processing it going to save in final file
    we have to make the intermediate file cause of chunking we cant directly do average otherwise it will get wrong """
    """"Creates chunk-level monthly summaries."""


    first_chunk = not append

    for chunk_number, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize),start=1):
        print(f"Processing Chunk {chunk_number}...")

        if transform_fn is not None:
            chunk = transform_fn(chunk)     #as we are adding the seasons so only for seasonal file we need this so while executing we are naming it with seasons


        chunk_summary = (
            chunk.groupby(group_columns)
            .agg(
                modal_sum=("modal_price", "sum"),
                modal_count=("modal_price", "count"),

                min_sum=("min_price", "sum"),
                min_count=("min_price", "count"),

                max_sum=("max_price", "sum"),
                max_count=("max_price", "count")
            )
            .reset_index()
        )

        chunk_summary.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print("Intermediate summary created successfully.")

#to convert the intermediate file data to data required to do analysis of commodity monthly prices

def create_final_summary(intermediate_file,output_file, group_columns):
    df = pd.read_csv(intermediate_file)
    monthly_summary = (df.groupby(group_columns,as_index=False)
        .agg(
            modal_sum=("modal_sum", "sum"),
            modal_count=("modal_count", "sum"),

            min_sum=("min_sum", "sum"),
            min_count=("min_count", "sum"),

            max_sum=("max_sum", "sum"),
            max_count=("max_count", "sum")
        )
    )

    monthly_summary["avg_modal_price"] = (
    monthly_summary["modal_sum"] /
    monthly_summary["modal_count"]
    )

    monthly_summary["avg_min_price"] = (
        monthly_summary["min_sum"] /
        monthly_summary["min_count"]
    )

    monthly_summary["avg_max_price"] = (
        monthly_summary["max_sum"] /
        monthly_summary["max_count"]
    )

    monthly_summary = monthly_summary[group_columns + ["avg_modal_price","avg_min_price","avg_max_price"]]
    monthly_summary.to_csv(output_file,index=False)

    print("final summary created successfully.")


#for commodity statistics analysis intermediate file 
def create_statistics_intermediate(
    input_file,
    output_file,
    group_columns,
    chunksize=100_000,
    append=False
):
    """
    Creates chunk-level statistics required to calculate
    mean, variance, standard deviation and coefficient of variation.

    For each chunk, it stores:
    - Count
    - Sum
    - Sum of Squares
    - Minimum
    - Maximum
    """

    first_chunk = not append

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_file, chunksize=chunksize),
        start=1
    ):

        print(f"Processing Chunk {chunk_number}...")

        # Create squared values
        chunk["modal_price_square"] = chunk["modal_price"] ** 2

        # Aggregate statistics for this chunk
        chunk_summary = (
            chunk.groupby(group_columns)
            .agg(
                modal_count=("modal_price", "count"),
                modal_sum=("modal_price", "sum"),
                modal_sum_of_squares=("modal_price_square", "sum"),
                modal_min=("modal_price", "min"),
                modal_max=("modal_price", "max")
            )
            .reset_index()
        )

        # Save chunk summary
        chunk_summary.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print("Statistics intermediate file created successfully.")

#Commodity statistics final file 
def create_statistics_summary(
    intermediate_file,
    output_file,
    group_columns
):
    """
    Combines chunk-level statistics and calculates
    Mean, Variance, Standard Deviation and
    Coefficient of Variation.
    """

    # Read intermediate file
    df = pd.read_csv(intermediate_file)

    # Merge statistics of all chunks
    statistics_summary = (
        df.groupby(group_columns, as_index=False)
        .agg(
            modal_count=("modal_count", "sum"),
            modal_sum=("modal_sum", "sum"),
            modal_sum_of_squares=("modal_sum_of_squares", "sum"),
            modal_min=("modal_min", "min"),
            modal_max=("modal_max", "max")
        )
    )

    # Mean
    statistics_summary["mean_price"] = (
        statistics_summary["modal_sum"] /
        statistics_summary["modal_count"]
    )

    # Population Variance
    statistics_summary["variance"] = (
        statistics_summary["modal_sum_of_squares"] /
        statistics_summary["modal_count"]
    ) - (
        statistics_summary["mean_price"] ** 2
    )

    # Small negative values can occur due to floating-point precision
    statistics_summary["variance"] = (
        statistics_summary["variance"]
        .clip(lower=0)     #if varience get very near to zero dur to floating point precision in computer it makes it negative so .clip makes it zero
    )

    # Standard Deviation
    statistics_summary["standard_deviation"] = np.sqrt(
        statistics_summary["variance"]
    )

    # Coefficient of Variation
    statistics_summary["coefficient_of_variation"] = (
        statistics_summary["standard_deviation"] /
        statistics_summary["mean_price"]
    ) * 100

    # Select final columns
    statistics_summary = statistics_summary[
        [
            "commodity",
            "year",
            "modal_count",
            "mean_price",
            "modal_min",
            "modal_max",
            "variance",
            "standard_deviation",
            "coefficient_of_variation",
        ]
    ]

    # Save final statistics
    statistics_summary.to_csv(
        output_file,
        index=False
    )

    print("Statistics summary created successfully.")