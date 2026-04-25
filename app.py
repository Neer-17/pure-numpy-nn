import cv2
import numpy as np
import streamlit as st
from neuralnet import Neural
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="DigitDetective",page_icon='🔍',layout='wide')
st.title("DigitDetective")
st.badge("numpy only - no frameworks")
st.space('medium')
@st.cache_resource
def load_model():
    """Loads the Neural network."""
    n_net = Neural()
    n_net.load_weights()
    return n_net

model = load_model()

def process_img(img):
    """Process the canvas result and returns a 1D array.\n
    img : drawing/canvas result"""
    img = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_RGBA2GRAY) 
    ret,thresh_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    counters,hierarchy = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(counters,key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    crop = thresh_img[y:y+h,x:x+w]
    pad_img = cv2.copyMakeBorder(crop,int(0.2*h),int(0.2*h),int(0.2*w),int(0.2*w),cv2.BORDER_CONSTANT,value=[0,0,0])
    img_array = cv2.resize(pad_img,(28,28),interpolation=cv2.INTER_AREA)
    _, thresh_img2 = cv2.threshold(img_array,127,255,cv2.THRESH_BINARY)
    norm_array = thresh_img2/255
    return norm_array.flatten()

def predict(model,array):
    """Predicts which number the given array is. """
    result = model.forward(array)
    predicted_digit = np.argmax(result)
    confidence = result
    return predicted_digit,confidence

if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0           

left_col, right_col = st.columns(2)
left_col = left_col.container(border=True, height="stretch", vertical_alignment="top")
with left_col:
    st.header("CANVAS")

    canvas_result = st_canvas(stroke_width=20,stroke_color="white",background_color="#000000",width=400,height=400,display_toolbar=False,key=f'canvas_{st.session_state.canvas_key}')
    button_cols = st.columns(2)
    with button_cols[1]:
        predict_btn = st.button("Predict", key="draw_predict")
    with button_cols[0]:
        clear_btn = st.button("Clear", key="draw_clear")    

    if predict_btn and len(canvas_result.json_data.get("objects")) > 0:
        array = process_img(canvas_result.image_data)
        digit, confidence = predict(model,array)
        st.session_state['prediction'] = [digit,confidence]
    
    if clear_btn:
        st.session_state.canvas_key += 1 
        if 'prediction' in st.session_state:
            del st.session_state['prediction']
        st.rerun()

right_col = right_col.container(border=True, height="stretch", vertical_alignment="top")
with right_col:
    st.header("PREDICTION")
    if 'prediction' in st.session_state :
        result = st.session_state['prediction']
        st.markdown(f'**<p style="font-size: 70px;">{result[0]}</p>**', unsafe_allow_html=True,text_alignment='center')
        data = result[1].tolist()
        data = data[0]
        for i,val in enumerate(data):
            col1,col2,col3 = st.columns(3)
            with col1:
                st.write(f"**{i}**")
            with col2:
                st.progress(value = val)
            with col3:
                st.write(f'{val:.2e}%')
        with col1:
            st.metric('Confidence',f'{(max(data)*100):.2f}')
        with col2:
            st.metric('Model','EMNIST')
        with col3:
            st.metric('Accuracy','97.8%')

    else:
        st.markdown("**Draw a digit to see the prediction.**")
    