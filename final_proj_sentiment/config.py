from pydantic import BaseModel
from typing import Literal

class LogConfig(BaseModel):
    learning_rate: float
    strength: float
    num_iterations: int
    regularization: Literal["L1", "L2", "Elastic Net", "None"]

class SVMConfig(BaseModel):
    learning_rate: float
    strength: float
    num_iterations: int
    regularization: Literal["L1", "L2", "Elastic Net", "None"]

class NBConfig(BaseModel):
    learning_rate: float = 0.0  
    strength: float = 0.0     
    num_iterations: int = 1    
    regularization: Literal["L1", "L2", "Elastic Net", "None"] = "None"


