import streamlit as st

import io
import os
import yaml
import argparse
import json
from io import StringIO
import pandas as pd
import torch

from PIL import Image

from predict_rating import load_model

from confirm_button_hack import cache_on_button_press
from data import context_data_split, context_data_loader, context_data_load
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'


def main():
    st.title("Book Rating Prediction Model")
    
# NCF_epoch_1_rmse_2.2396299296441464.pth

    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    
    if uploaded_files:
        l = []
        # import time
        # time.sleep(5)
        uploaded_files.sort(key=lambda x: x.name)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            s=str(bytes_data,'utf-8')
            data = StringIO(s) 
            df=pd.read_csv(data)
            l.append(df)
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        with open('args.txt', 'r') as f:
            args.__dict__ = json.load(f)
        # print(l[0])
        # print(l[0].columns)
        st.write("[START] LOAD DATA")
        data = context_data_load(args, l)
        st.write("[END] LOAD DATA")
        # st.dataframe(data['test'].style.highlight_max(axis=0))

        data = context_data_split(args, data)
        data = context_data_loader(args, data)
        st.write("[START] LOAD MODEL")
        model = load_model(args, data)
        st.write("[END] LOAD MODEL")
        
        with torch.no_grad():
            predicts = model.predict(data['test_dataloader'])
            submission = l[1]
            submission['rating'] = predicts
            # print(submission[:10])
            st.title("User's Expected Rating for book")
            st.dataframe(submission[['user_id', 'rating']][:10])
            
        # st.image(image, caption='Uploaded Image')
        # st.write("Classifying...")
        # _, y_hat = get_prediction(model, image_bytes)
        # label = config['classes'][y_hat.item()]
        
        # st.write(f'label is {label}')


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')