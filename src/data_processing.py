import pandas as pd
TEXT_COLUMNS = [
    "state",
    "district",
    "market",
    "commodity",
    "variety",
    "grade"
]

PRICE_COLUMNS = [
    "min_price",
    "max_price",
    "modal_price"
]

def standardize_columns(df):
    df.columns = (df.columns.str.strip().str.lower().str.replace(' ', '_'))
    return df


def clean_text_columns(df, text_columns):
    for column in text_columns:
        df[column]=df[column].astype("string").str.strip()
        #.astype("string")Ensures that the column is treated as text.
        # .str.strip()Removes unnecessary spaces from the beginning and end
    return df

def clean_date_column(df,date_column):
    df[date_column]=pd.to_datetime(df[date_column],format="%Y-%m-%d",
                                                errors="coerce")#if error occurs in the value it replaces it with the NAN 
    return df

def clean_date_features(df,date_column):
    df["year"]=df[date_column].dt.year
    df["month"]=df[date_column].dt.month
    df["month_name"] = (df[date_column].dt.month_name())
    df["month"].head
    return df

def check_order_price(df):
    invalid_price_order = (
        (df["min_price"] > df["modal_price"]) |
        (df["modal_price"] > df["max_price"])
    )
    return invalid_price_order

def check_zero_prices(df,price_columns):
    zero_price_count = (df[price_columns] == 0).sum()
    return zero_price_count

def add_price_quality_flags(df, price_columns):
    
    df["invalid_price_order"] = (
        (df["min_price"] > df["modal_price"]) |
        (df["modal_price"] > df["max_price"])
    )
    
    df["has_negative_price"] = (
        df[price_columns] < 0
    ).any(axis=1)
    
    df["has_zero_price"] = (
        df[price_columns] == 0
    ).any(axis=1)
    
    return df

def clean_chunk(df):
    df = standardize_columns(df)
    df = clean_text_columns(df,TEXT_COLUMNS)
    df = clean_date_column(df,"arrival_date")
    df = clean_date_features(df,"arrival_date")
    df=add_price_quality_flags(df, PRICE_COLUMNS)
    return df

def validate_chunk(df):

    quality_report = {
        "rows": len(df),
        "missing_dates": df["arrival_date"].isna().sum(),
        "invalid_price_order": df["invalid_price_order"].sum(),
        "negative_price_rows": df["has_negative_price"].sum(),
        "zero_price_rows": df["has_zero_price"].sum()
    }

    return quality_report

def process_2025_file(
    input_file,
    output_file,
    chunksize=100_000
):
    
    quality_reports = []
    first_chunk = True
    
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_file,
            chunksize=chunksize
        ),
        start=1
    ):
        
        print(
            f"Processing chunk {chunk_number}..."
        )
        
        # 1. Clean the current chunk
        cleaned_chunk = clean_chunk(chunk)
        
        # 2. Validate the current chunk
        quality_report = validate_chunk(
            cleaned_chunk
        )
        
        # 3. Add chunk number
        quality_report["chunk"] = chunk_number
        
        quality_reports.append(
            quality_report
        )
        
        # 4. Write cleaned chunk to output
        cleaned_chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,  #means writing the column names 
            index=False
        )
        
        first_chunk = False
        
        print(
            f"Chunk {chunk_number} completed."
        )
    
    return pd.DataFrame(quality_reports)

