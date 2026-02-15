# Copyright ©2022-2023. The Regents of the University of California
# (Regents). All Rights Reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met: 

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer. 

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the 
# documentation and/or other materials provided with the
# distribution. 

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import dask.dataframe as dd
from typing import List

# Type alias
DaskDataFrame = dd.DataFrame


def generate_user_consent_list(data: List[DaskDataFrame], user_id_col: str, opt_in: bool) -> DaskDataFrame:
    """
    Generate table of user IDs and column specifying whether they have given their consent and should be included in
    the analysis

    Args:
        data: list of relevant datasets, e.g. CDRs and MoMo
        user_id_col: column containing user ID
        opt_in: whether the user's consent is set to true as default

    Returns: 
        Dask df with user consent
    """
    # Obtain all existing user IDs in the datasets
    user_dfs = []
    for df in data:
        user_dfs.append(df[[user_id_col]].drop_duplicates())
    
    # Concatenate and get unique users
    users = dd.concat(user_dfs, axis=0)
    users = users.drop_duplicates()

    # Add default consent value
    users['include'] = opt_in

    return users
