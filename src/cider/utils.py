from pydantic import BaseModel
from pyspark.sql import DataFrame as SparkDataFrame
from pandas import DataFrame as PandasDataFrame


def _validate_dataframe(
    df: SparkDataFrame | PandasDataFrame, required_schema: type[BaseModel]
) -> None:
    """
    Validate that the dataframe has the required schema.

    Args:
        df: Spark or Pandas dataframe to validate
        required_schema: Pydantic BaseModel schema that the dataframe must conform to

    Raises:
        ValueError: If any of the required columns are missing from the dataframe
    """
    df_columns = set(df.columns)
    required_columns = set(
        [k for k, field in required_schema.model_fields.items() if field.is_required()]
    )
    missing_columns = required_columns - df_columns
    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the dataframe: {missing_columns}"
        )
