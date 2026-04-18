import numpy as np
from neuralnet import *
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="DigitDetective",page_icon='🔍',layout='wide')
st.title("DigitDetective")
st.space('medium')

"""## Draw the number"""

canvas_result = st_canvas(fill_color="#f22e1f",stroke_width=20,stroke_color="black",background_color="#fcfcfc")
