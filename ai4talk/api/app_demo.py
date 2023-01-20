import os

import streamlit as st
from streamlit_option_menu import option_menu

from ai4talk.ml_logic.model_tat import transcript_tat
from ai4talk.ml_logic.model_sah import transcript_sah
from ai4talk.ml_logic.utils import translate

from PIL import Image

from highlight import highlight_diffs
from load_css import local_css

#from google.cloud import translate
#from google.oauth2 import service_account
#from scipy.io.wavfile import write
#---------------------------------
# Create API client.
# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["gcp_service_account"])

#------------------------------------------------------------------------
st.set_page_config(
    page_title="Speech-to-Text Transcription App", page_icon=":lippen:", layout="wide"
)
original_title = '<p style="font-family:Courier; color: Green; font-weight: bold;font-size: 40px;">Automated Speech Recognition: Whisper </p>'
st.markdown(original_title, unsafe_allow_html=True)

#------------------------------------------------------------------------
original_title = '<p style="font-family:Courier; color: Green; font-weight: bold;font-size: 40px;">ASR</p>'
st.sidebar.markdown(original_title,unsafe_allow_html=True)
image = Image.open(os.path.join(os.getcwd(),'ai4talk',"api", 'asr.png'))

st.sidebar.image(image, caption='')
with st.sidebar:
    choose = option_menu("", ["About", "Tatar", "Sakha", "Contact"],
                         icons=['house', 'receipt', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#FAFAFA"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02AB21"},
    }
    )

profile = Image.open(os.path.join(os.getcwd(),'ai4talk',"api",'map.jpg'))
if choose == "About":
    col1, col2 = st.columns([0.5, 0.5])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:20px ;font-weight: bold; font-family: 'Courier'; color: #FF9633;}
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Audio to Text (Transcription to International Phonetic Alphabet) for Underrepresented Languages </p>', unsafe_allow_html=True)
   # st.write("Automated Speech Recognition for Underrepresented Languages")
    st.image(profile, width=700 )

elif choose == "Tatar":
    #Add a file uploader to allow users to upload their project plan file
    st.markdown(""" <style> .font {
    font-size:20px ; font-weight: bold; font-family: 'Courier'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your audio (Tatar language)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose your audio file", type=['mp3'], key="2", label_visibility='hidden')

    if uploaded_file is not None:
        name = uploaded_file.name
        audio_file = open(os.path.join(os.getcwd(),"processed_data","tat", name), 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        result = transcript_tat(uploaded_file.name)
        transcript_true = result["transcript true"]
        transcript_model = result['transcript from model']
        transcript_diff = highlight_diffs(transcript_true[0],transcript_model[0])

        translation_true = translate(transcript_true[0])
        translation_model = translate(transcript_model[0])
        translation_diff = highlight_diffs(translation_true,translation_model)

        local_css("style.css")

    #-----------------------------------------------
    #    """ """  project_id = "tatar-trans"
    #     assert project_id
    #     parent = f"projects/{project_id}"
    #     client= translate.TranslationServiceClient(credentials=credentials)
    #     sample_text_2 = translate(trans_model[0])
    #     target_language_code_2 = "en"
    #     response = client.translate_text(
    #     contents=[sample_text_2],
    #     target_language_code=target_language_code_2,
    #     parent=parent)
    #     for translation in response.translations:
    #         trans_text =translation.translated_text """

        col1, col2 = st.columns([5,5])
        col3, col4 = st.columns([5,5])
        col_space1,col_space2 = st.columns([5,5])
        col5, col6 = st.columns([5,5])
        col7, col8 = st.columns([5,5])

        with col1 :
            st.markdown("<span class='highlight red title'>IPA Transcript (True) </span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span class='highlight blue sub_fb '>{transcript_true[0]}</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("<span class='highlight red title'>IPA Transcript (Model)</span>", unsafe_allow_html=True)
        with col4:
            st.markdown("<span class='highlight blue sub_fb'>"+transcript_diff+"</span>", unsafe_allow_html=True)

        with col_space1:
            st.text(" ")
        with col_space2:
            st.text(" ")

        with col5:
            st.markdown("<span class='highlight red title'>Tatar Alphabet (True)</span>", unsafe_allow_html=True)
        with col6:
            st.markdown("<span class='highlight blue sub_fb'>"+translation_true+"</span>", unsafe_allow_html=True)
        with col7:
            st.markdown("<span class='highlight red title'>Tatar Alphabet (Model)</span>", unsafe_allow_html=True)
        with col8:
            st.markdown("<span class='highlight blue sub_fb'>"+translation_diff+"</span>", unsafe_allow_html=True)

        # with col7:
           # st.markdown("<span class='highlight red title'>Google Translate</span>", unsafe_allow_html=True)
        # with col8:
        #    st.markdown("<span class='highlight blue sub_fb'>"+trans_text+"</span>", unsafe_allow_html=True)

#------------------------------------------------------------------------

elif choose == "Sakha":
    #Add a file uploader to allow users to upload their project plan file
    st.markdown(""" <style> .font {
    font-size:20px ; font-weight: bold; font-family: 'Courier'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your audio (Sakha language)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose your audio file", type=['mp3'], key="2", label_visibility='hidden')

    if uploaded_file is not None:
        name = uploaded_file.name
        audio_file = open(os.path.join(os.getcwd(),"processed_data","sah", name), 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')
        result_2 = transcript_sah(uploaded_file.name)
        transcript_true_2 = result_2["transcript true"]
        transcript_model_2 = result_2['transcript from model']
        transcript_diff_2 = highlight_diffs(transcript_true_2[0],transcript_model_2[0])

        local_css("style.css")

    #-----------------------------------------------
    #    """ """  project_id = "tatar-trans"
    #     assert project_id
    #     parent = f"projects/{project_id}"
    #     client= translate.TranslationServiceClient(credentials=credentials)
    #     sample_text_2 = translate(trans_model[0])
    #     target_language_code_2 = "en"
    #     response = client.translate_text(
    #     contents=[sample_text_2],
    #     target_language_code=target_language_code_2,
    #     parent=parent)
    #     for translation in response.translations:
    #         trans_text =translation.translated_text """

        col1, col2 = st.columns([5,5])
        col3, col4 = st.columns([5,5])
        col_space1,col_space2 = st.columns([5,5])
        col5, col6 = st.columns([5,5])
        col7, col8 = st.columns([5,5])

        with col1 :
            st.markdown("<span class='highlight red title'>IPA Transcript (True) </span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span class='highlight blue sub_fb '>{transcript_true_2[0]}</span>", unsafe_allow_html=True)
        with col3:
            st.markdown("<span class='highlight red title'>IPA Transcript (Model)</span>", unsafe_allow_html=True)
        with col4:
            st.markdown("<span class='highlight blue sub_fb'>"+transcript_diff_2+"</span>", unsafe_allow_html=True)

  #  else :
  #      st.write('Please change folder')


elif choose == "Contact":
    #Add a file uploader to allow users to upload their project plan file
    st.markdown(""" <style> .font {
    font-size:35px ; font-weight: bold; font-family: 'Courier'; color: Green;}
    </style> """, unsafe_allow_html=True)

   # st.markdown('<p class="font">Team </p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font {
    font-size:25px ; font-weight: bold; font-family: 'Courier'; color: #FF9633;}
    </style> """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3,3,3])
    col4, col5, col6 = st.columns([3,3,3])
    col7, col8, col9 = st.columns([3,3,3])
    with col1 :
        col1.markdown("<span class='font'> Aygul </span>", unsafe_allow_html=True)
    with col2 :
        col2.markdown("<span class='font'> Margot </span>", unsafe_allow_html=True)
    with col3 :
        col3.markdown("<span class='font'> Dhanya </span>", unsafe_allow_html=True)

    st.markdown(""" <style> .font_1 {
    font-size:20px ; font-family: 'Courier'; color: black;}
    </style> """, unsafe_allow_html=True)
    with col4 :
        col4.markdown("<span class='font_1'> NLP Enthusiast </span>", unsafe_allow_html=True)
    with col5 :
        col5.markdown("<span class='font_1'> Linguist Expert </span>", unsafe_allow_html=True)
    with col6 :
        col6.markdown("<span class='font_1'> Programming Enthusiast </span>", unsafe_allow_html=True)
    with col7 :
        col7.markdown("<span class='font_1'> Github: aygul0790 </span>", unsafe_allow_html=True)
    with col8 :
        col8.markdown("<span class='font_1'> Github: margot95 </span>", unsafe_allow_html=True)
    with col9 :
        col9.markdown("<span class='font_1'> Github: Dhanya99Sanju </span>", unsafe_allow_html=True)

    # if st.button('Success') :
    #    st.balloons()
