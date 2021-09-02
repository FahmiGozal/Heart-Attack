import streamlit as st
from PIL import Image # Required to show images
import pandas as pd

df = pd.read_csv('data/data.csv')

st.write(df.head())
