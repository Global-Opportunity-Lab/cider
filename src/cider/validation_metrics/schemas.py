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


from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Annotated, ClassVar
from enum import Enum


class ConsumptionColumn(str, Enum):
    GROUNDTRUTH = "groundtruth_consumption"
    PROXY = "proxy_consumption"


class ConsumptionData(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    household_id: Annotated[
        str, Field(description="Unique identifier for each household")
    ]
    groundtruth_consumption: Annotated[
        float, Field(description="Groundtruth consumption value")
    ]
    proxy_consumption: Annotated[
        float, Field(description="Proxy consumption value (estimated consumption)")
    ]
    weight: Annotated[
        float, Field(description="Weight of the household in the dataset")
    ]


class ConsumptionDataWithCharacteristic(ConsumptionData):
    model_config = ConfigDict(from_attributes=True)

    # Allowed values for characteristic -- this must be set before using the pydantic class
    # Useful if we want to run data checks before we start using the metric functions
    allowed_characteristic_values: ClassVar[set] = set()

    characteristic: Annotated[
        str | int, Field(description="Group identifier for fairness analysis")
    ]

    @field_validator("characteristic")
    @classmethod
    def validate_characteristic(cls, v):
        if not cls.allowed_characteristic_values:
            raise ValueError(
                "`allowed_characteristic_values` must be set before using ConsumptionDataWithCharacteristic"
            )
        if (
            cls.allowed_characteristic_values
            and v not in cls.allowed_characteristic_values
        ):
            raise ValueError(
                f"Characteristic value {v} not in allowed values: {cls.allowed_characteristic_values}"
            )
