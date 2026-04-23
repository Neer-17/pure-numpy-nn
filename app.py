import numpy as np
from neuralnet import *
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="DigitDetective",page_icon='🔍',layout='wide')
st.title("DigitDetective")
st.space('medium')

def process_img(img):
    img = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_RGBA2GRAY) 
    ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    counters,hierarchy = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(counters)
    x,y,w,h = cv2.boundingRect(cnt)
    crop = thresh_img[y:y+h,x:x+w]
    pad_img = cv2.copyMakeBorder(crop,int(0.2*h),int(0.2*h),int(0.2*w),int(0.2*w),cv2.BORDER_CONSTANT,value=[0,0,0])
    img_array = cv2.resize(pad_img,(28,28),interpolation=cv2.INTER_AREA)
    fin_img = cv2.imwrite(f'output.jpg',img_array)
    norm_array = img_array/255
    return norm_array.flatten()
def guess(canvas_result):
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects")
        if len(objects) > 0:
            st.success("Drawing detected!")
            img = canvas_result.image_data
            array = process_img(img)
            n_net = Neural()
            n_net.load_weights()
            result = n_net.forward(array)
            result = result.round()
            # print(result)
            print(result)
            # st.write(result)
            # st.write(max(result))
        else:
            st.info("Canvas is currently empty.")
            pass
center = st.container(border=True,width='stretch',height='content')
with center:
    """## Draw a number"""

    canvas_result = st_canvas(fill_color="#f22e1f",stroke_width=20,stroke_color="white",background_color="#000000",width=400,height=400)
    guess(canvas_result)


