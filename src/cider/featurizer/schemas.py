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


from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated
from enum import Enum


class DirectionOfTransactionEnum(str, Enum):
    """
    Enum for direction of transaction.
    """

    INCOMING = "incoming"
    OUTGOING = "outgoing"


class DataDiagnosticStatistics(BaseModel):
    """
    Schema for data diagnostic statistics.
    """

    model_config = ConfigDict(from_attributes=True)

    total_transactions: Annotated[
        int, Field(description="Total number of transactions in the dataset")
    ]
    num_unique_callers: Annotated[
        int, Field(description="Number of unique caller IDs in the dataset")
    ]
    num_unique_recipients: Annotated[
        int, Field(description="Number of unique recipient IDs in the dataset")
    ]
    num_days: Annotated[
        int, Field(description="Number of unique days covered in the dataset")
    ]


class AllowedPivotColumnsEnum(str, Enum):
    """
    Enum for allowed pivot columns.
    """

    IS_WEEKEND = "is_weekend"
    IS_DAYTIME = "is_daytime"
    TRANSACTION_TYPE = "transaction_type"
