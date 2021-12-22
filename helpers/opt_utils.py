from functools import reduce
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import lit
from typing import List


def generate_user_consent_list(data: List[SparkDataFrame], user_id_col: str, opt_in: bool) -> SparkDataFrame:
    """
    Generate table of user IDs and column specifying whether they have given their consent and should be included in
    the analysis

    Args:
        data: list of relevant datasets, e.g. CDRs and MoMo
        user_id_col: column containing user ID
        opt_in: whether the user's consent is set to true as default

    Returns: spark df with user consent

    """
    # Obtain all existing user IDs in the datasets
    user_dfs = []
    for df in data:
        user_dfs.append(df.select(user_id_col).distinct())
    users = reduce(SparkDataFrame.union, user_dfs)
    users = users.select(user_id_col).distinct()

    # Add default consent value
    if opt_in:
        users = users.withColumn('include', lit(True))
    else:
        users = users.withColumn('include', lit(False))

    return users
