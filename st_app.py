import streamlit as st
import os
from PIL import Image
import rubbish_classify
from load_css import local_css


def load_image(image_file):
	img = Image.open(image_file)
	return img


image_in = False
path = os.path.dirname(__file__)
point_path = path + '/point.txt'

with open(point_path, 'r') as f:
	point = int(f.read())

icon = load_image('VECO.ico')
st.set_page_config(
	page_title='VECO',
	page_icon=icon
)

local_css("style.css")
st.markdown("<h><span class='header bold highlight green' >VECO</span></h>", unsafe_allow_html=True)
st.markdown("<h><span class='green_text' >Go Hoi An, Go Eco</span></h>", unsafe_allow_html=True)

file = st.file_uploader('Classify Waste')
if file:
	point += 1
	with open(point_path, 'w') as f:
		f.write(str(point))

	file_details = {'File_Name': file.name, 'File_Type': file.type}
	# st.write(file_details)
	if file.type[:5] == 'image':
		image_in = True
		img = load_image(file)
		st.image(img, width=250)
		# Saving file
		with open(file.name, 'wb') as f:
			f.write(file.getbuffer())
		label = rubbish_classify.predict(file.name)
		st.markdown(f"<h><span class='subheader bold highlight green'>This is {label} waste</span></h>",
		            unsafe_allow_html=True)
		if os.path.exists(file.name):
			os.remove(file.name)
		else:
			pass
	else:
		st.write('Not an image file, try another one!')
else:
	image_in = False
st.sidebar.markdown(f"<h><span class='header bold highlight green' >Total points: {point}</span></h>",
                    unsafe_allow_html=True)
if image_in:
	col1, mid, col2 = st.sidebar.columns([1, 1, 15])
	with col1:
		st.image('VECO.ico', width=30)
	with col2:
		st.write('+1')
