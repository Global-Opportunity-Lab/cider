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

from typing import Callable, Optional
import hashids
from .datastore import DataStore
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

def __init__(self, datastore: DataStore, clean_folders = True, checker: Optional[Callable] = None):
    
    self.ds=datastore
    self.cfg = datastore.cfg
    self.outputs = self.cfg.path.working.directory_path / 'anonymized'
    
    make_dir(self.outputs, clean_folders)
    make_dir(self.outputs / 'outputs')
    
    data_type_map = {DataType.FEATURES: None}
    self.ds.load_data(data_type_map=data_type_map)
    
    with open(self.cfg.path.input_data.file_paths.anonymization_salt, r) as saltfile:
        salt = saltfile.read().strip() 
    
    self.hashcode = Hashids(salt=salt, min_length=16)
    self.checker = checker


def anonymize(self):
    
    # The featurizer writes caller id/phone number to the 'name' column
    column_to_anonymize = 'name'
    
    features = self.ds.features
    
    check_and_anonymize_ufunc = udf(_check_number_format_and_hash, BooleanType())
    
    features_with_anonymized_name_column = features.withColumn('name_anonymized', check_and_anonymize_ufunc('name'))
    

def _check_number_format_and_hash(raw):
    
    if not (isinstance(raw, str) and raw.isdigit()):
        return False
    
    if checker and not checker(raw):
        return False
    
    return _hash(raw)
    
    
def _check_number_format_long(
    raw, 
    required_prefix=None, 
    forbidden_prefix=None, 
    min_length=None, 
    max_length=None
):
    
    if not (
        isinstance(raw, str) 
        and raw.isdigit()
    ):
        return False
    
    if required_prefix and not raw.startswith(required_prefix):
        return False
    
    if forbidden_prefix and raw.startswith(forbidden_prefix):
        return False
    
    if min_length and len(str) < min_length:
        return False
    
    if max_length and len(str) > max_length:
        return False

def _hash(raw: str):
    
    return self.hashcode.encode(int(raw))