from functools import reduce
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import lit
from typing import List


def generate_user_consent_list(data: List[SparkDataFrame], user_id_col: str, opt_in: bool) -> SparkDataFrame:
    user_dfs = []
    for df in data:
        user_dfs.append(df.select(user_id_col).distinct())

    users = reduce(SparkDataFrame.union, user_dfs)
    users = users.select(user_id_col).distinct()
    if opt_in:
        users = users.withColumn('include', lit(False))
    else:
        users = users.withColumn('include', lit(True))

    return users
