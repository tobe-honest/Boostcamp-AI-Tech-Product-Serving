import torch
import streamlit as st
from model_rating import NeuralCollaborativeFiltering
from utils import transform_image
import yaml
from typing import Tuple

# @st.cache
# @st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda parameter: parameter.data.numpy()})
@st.cache(hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
def load_model(args, data) -> NeuralCollaborativeFiltering:
    model = NeuralCollaborativeFiltering(args, data, True)
    
    return model
