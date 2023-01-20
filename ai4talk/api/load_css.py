import streamlit as st
import os
def local_css(file_name):
    dir=os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir, file_name)) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
