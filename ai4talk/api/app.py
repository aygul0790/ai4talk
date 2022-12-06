import streamlit as st
from ai4talk.ml_logic.model_base import transcript
from ai4talk.ml_logic.model import translate
from ai4talk.ml_logic.model import transcript
from ai4talk.ml_logic.utils import highlight_diffs,ret_high
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    name =uploaded_file.name
    result = transcript(name)
    trans_true =result["transcript true"]
    trans_model = result['transcript from model']

    st.write(result)
    st.write("transcript true -------- :  ","     ",trans_true[0])
    st.write('transcript from model-------------',trans_model[0])
    data_res = translate(result['transcript from model'][0])
    st.write("transcript from model-----------------",data_res)
    data_refs = translate(result['transcript true'][0])
    st.write(data_refs)
    st.write(ret_high)
