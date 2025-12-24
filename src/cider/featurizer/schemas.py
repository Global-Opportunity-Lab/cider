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
from cider.schemas import (
    CallDataRecordData,
    MobileDataUsageData,
    MobileMoneyTransactionData,
    MobileMoneyTransactionType,
    RechargeData,
)
from datetime import date, datetime


class DirectionOfTransactionEnum(str, Enum):
    """
    Enum for direction of transaction.
    """

    INCOMING = "incoming"
    OUTGOING = "outgoing"


class AllowedPivotColumnsEnum(str, Enum):
    """
    Enum for allowed pivot columns.
    """

    IS_WEEKEND = "is_weekend"
    IS_DAYTIME = "is_daytime"
    TRANSACTION_TYPE = "transaction_type"


class StatsComputationMethodEnum(str, Enum):
    """
    Enum for statistics computation method.
    """

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STD = "std"
    MEDIAN = "median"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"


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


class CallDataRecordTagged(CallDataRecordData):
    """
    Schema for call data record with tagged conversations.
    Inherits from CallDataRecordData and adds additional fields for tagged conversation.
    """

    model_config = ConfigDict(from_attributes=True)

    day: Annotated[date, Field(description="Day of the month")]
    is_daytime: Annotated[
        bool, Field(description="Whether the call was made during daytime")
    ]
    is_weekend: Annotated[
        bool, Field(description="Whether the call was made on a weekend")
    ]
    direction_of_transaction: Annotated[
        DirectionOfTransactionEnum,
        Field(description="Direction of the transaction (incoming, outgoing, etc.)"),
    ]
    conversation: Annotated[
        datetime | None, Field(description="Timestamp of the conversation start")
    ]


class MobileDataUsageDataWithDay(MobileDataUsageData):
    """
    Schema for mobile data usage data with day information.
    Inherits from MobileDataUsageData and adds an additional field for day.
    """

    model_config = ConfigDict(from_attributes=True)

    day: Annotated[
        date, Field(description="Date without timestamp when the data usage occurred")
    ]


class MobileMoneyDataWithDay(MobileMoneyTransactionData):
    """
    Schema for mobile money transaction data with day information.
    Inherits from MobileMoneyTransactionData and adds an additional field for day.
    """

    model_config = ConfigDict(from_attributes=True)

    day: Annotated[
        date, Field(description="Date without timestamp when the transaction occurred")
    ]


class MobileMoneyDataWithDirection(BaseModel):
    """
    Schema for mobile money transaction data with direction of transaction.
    """

    model_config = ConfigDict(from_attributes=True)

    primary_id: Annotated[
        str,
        Field(
            description="Unique identifier for the primary account involved in the transaction"
        ),
    ]
    correspondent_id: Annotated[
        str,
        Field(
            description="Unique identifier for the correspondent account involved in the transaction"
        ),
    ]
    day: Annotated[
        date, Field(description="Date without timestamp when the transaction occurred")
    ]
    amount: Annotated[float, Field(description="Amount of the transaction")]
    balance_before: Annotated[
        float, Field(description="Balance before the transaction for primary account")
    ]
    balance_after: Annotated[
        float, Field(description="Balance after the transaction for primary account")
    ]
    transaction_type: Annotated[
        MobileMoneyTransactionType, Field(description="Type of the transaction")
    ]
    direction_of_transaction: Annotated[
        DirectionOfTransactionEnum,
        Field(description="Direction of the transaction (incoming or outgoing)"),
    ]


class RechargeDataWithDay(RechargeData):
    """
    Schema for recharge data with day information.
    Inherits from RechargeData and adds an additional field for day.
    """

    model_config = ConfigDict(from_attributes=True)

    day: Annotated[
        date, Field(description="Date without timestamp when the transaction occurred")
    ]


class AntennaDataGeometry(BaseModel):
    """
    Schema for antenna data with geometry information.
    """

    model_config = ConfigDict(from_attributes=True)

    caller_antenna_id: Annotated[
        str, Field(description="Unique identifier for the antenna")
    ]
    latitude: Annotated[float, Field(description="Latitude of the antenna location")]
    longitude: Annotated[float, Field(description="Longitude of the antenna location")]


class AntennaDataGeometryWithRegion(AntennaDataGeometry):
    """
    Schema for antenna data with region information.
    """

    model_config = ConfigDict(from_attributes=True)

    region: Annotated[str, Field(description="Region of the antenna location")]
