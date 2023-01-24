# Copyright ©2022-2023. The Regents of the University of California (Regents). 
# All Rights Reserved. 

# Permission to use, copy, modify, and distribute this software and its 
# documentation for educational, research, and not-for-profit purposes, without
# fee and without a signed licensing agreement, is hereby granted, provided that 
# the above copyright notice, this paragraph and the following two paragraphs 
# appear in all copies, modifications, and distributions. Contact The Office of
# Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, 
# CA 94720-1620, (510) 643-7201, otl@berkeley.edu, 
# http://ipira.berkeley.edu/industry-info for commercial licensing 
# opportunities.

# IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
# SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING
# OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS 
# BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED 
# HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE 
# MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

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
