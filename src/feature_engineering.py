import pandas as pd
def create_monthly_intermediate_summary(input_file, output_file, chunksize=100_000,append=False):
    """we are going to process the file chunk by chunk so going to save the the summary of each chunk in intermediate file and then by processing it going to save in final file
    we have to make the intermediate file cause of chunking we cant directly do average otherwise it will get wrong """
    """"Creates chunk-level monthly summaries."""

    first_chunk = not append

    for chunk_number, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize),start=1):
        print(f"Processing Chunk {chunk_number}...")

        chunk_summary = (
            chunk.groupby(["commodity", "year", "month"])
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

def create_monthly_price_summary(intermediate_file,output_file):
    df = pd.read_csv(intermediate_file)
    monthly_summary = (df.groupby(["commodity", "year", "month"],as_index=False)
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

    monthly_summary = monthly_summary[["commodity","year","month","avg_modal_price","avg_min_price","avg_max_price"]]
    monthly_summary.to_csv(output_file,index=False)

    print("Monthly summary created successfully.")