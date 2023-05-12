
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from model.predictor import get_model
from model.mask import crop_img
import cv2
import os
import webbrowser
import test
import classification
import pickle
from keras.utils.layer_utils import count_params
import pandas as pd

from keras.applications.vgg16 import preprocess_input,VGG16
import tensorflow as tf

# setting page cofiguration including page icon, page name
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="collapsed"
    # menu_items={
    #     "Get Help": "#",
    #     "Report a bug": "#",
    #     "About": "Detecting brain tumors using different Artificial Intelligence techniques. Written by Priyanshu Agarwal",
    # },
)


# function to apply css effect to the page (cuatom CSs)
def set_css(css_path):
    """
    Set the CSS file to use.
    """
    css_file = open(css_path, "r").read()
    st.markdown('<style>{}</style'.format(css_file), unsafe_allow_html=True)
css_file = open('css/streamlit.css', "r").read()


# css applied here take effect on all pages from sidebar
st.markdown('<style>{}</style'.format(css_file), unsafe_allow_html=True)

    
# finction to add logo to page in sidebar
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo    
st.sidebar.image(add_logo(logo_path="images/logo.png", width=500, height=200)) 


# side bar Menu setup
with st.sidebar:
    selected = st.sidebar.selectbox("Menu",["Home","Detection Methods","Classification Methods","About & Contact"])


if selected == "Home":
    st.title("Brain Tumor Detector")
    css_file = open('css/streamlit.css', "r").read()
    st.markdown('<style>{}</style'.format(css_file), unsafe_allow_html=True)
    with st.container():
        st.image('https://scitechdaily.com/images/Brain-Signals-Rotating-Test.gif',width=None)
    
    b1=st.button("Check Now",use_container_width=True)

    col1, col2,col3 =st.columns(3)
    with col1:
        st.image("images/img1.jpg",use_column_width="always")
    with col2:
        st.image("images/img2.jpg",use_column_width="always")
    with col3:
        st.image("images/img3.jpg",use_column_width="auto")

    with col1:
        if st.button("What is Brain Tumor"):
            webbrowser.open_new_tab("https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084#:~:text=A%20brain%20tumor%20is%20a,tumors%20are%20cancerous%20(malignant).")
    with col2:
        if st.button("Brain Tumor and Brain Cancer"):
            webbrowser.open_new_tab("https://www.hopkinsmedicine.org/health/conditions-and-diseases/brain-tumor#:~:text=All%20brain%20cancers%20are%20tumors,distinct%20borders%20and%20rarely%20spread.")
    with col3:
        if st.button("Treatment Procedure"):
            webbrowser.open_new_tab("https://www.mayoclinic.org/diseases-conditions/brain-tumor/diagnosis-treatment/drc-20350088")


    with st.expander("How does this work :question: "):
        st.write('''
            :white_circle: This tool makes its prediction based on the prediction made by Deep Learning model Trained on over 3000 MRI Scanned images (Tumor + non-Tumor).\n
            :white_circle: Model has a Based accuracy of 98%.
        ''')
    with st.expander("Note :warning: "):
        st.error("Testing the Presence of a Brain Tumor online cannot be a suitable sunstitute to a test by an internist or neurologist! \n This tool gives you a breif idea of what could potentially be a neurologist's diagnosis.")

    def format_func(item):
        return item
    image_bytes = st.file_uploader(
    "Upload a brain MRI scan image", type=["png", "jpeg", "jpg"]
    )
    if image_bytes:
        array = np.frombuffer(image_bytes.read(), np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        st.write(
            """
                #### Brain MRI scan image
                """
        )
        st.image(image)
    b2=st.button("Analyze",key='analyze_2')


    # Detecting Brain Tumor
    if b1 or b2:
        flag=1
        # if image not found
        if not image_bytes:
            st.warning("Please select a Image first")
        else:
            # @st.cache
            def load_model():
                model, acc, loss = get_model(6)
                return model, acc, loss

            # prediction of Brain tumor through making method (Uses Model 1)
            with st.spinner(text="Analyzing..."):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img = crop_img(gray, image, None)
                # cv2.imwrite("temp.png", img)
                model, acc, loss = load_model()
                img_mask = image # crop_img(gray, image, None)
                gray_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_OTSU)[-1]
                try:
                    img = cv2.resize(img, (32, 32))
                    img = np.array([img])
                    prediction = model.predict(img)
                    
                    st.write(
                        """
                        #### Mask Threshold
                        """
                    )

                    st.image(cv2.resize(thresh, (128, 128)))

                    st.write("""#### Prediction""")
                    st.image(cv2.resize(img_mask, (128, 128)))
                    if prediction[0][0] == 1:
                        st.write(f"The sample has a tumor")
                    if prediction[0][0] == 0:
                        st.write(f"The sample has no tumor")
                        flag=0
                    st.write(f"Accuracy: {acc*100:.2f}%")
                except:
                    st.warning("Image size is not compatible with this model")
    
    # Classification of Brain Tumor only in case it is present
    if (b1 or b2) and flag==1:
        # if image not found
        if not image_bytes:
            st.warning("Please select a Image first")
        else:
            # classification through a Neural network based CNN model  (Uses model 2)
            with st.expander("Using Traditional CNN"):
                with st.spinner("Classifying..."):
                    model=pickle.load(open("classify/classification_model.sav",'rb'))
                    # st.image(image)
                    
                    img = image
                    img = cv2.resize(img,(150,150))
                    img_array = np.array(img)
                    img_array = img_array.reshape(1,150,150,3)
                    # return img_array.shape
                    a=model.predict(img_array)
                    indices = a.argmax()
                    labels = ['Glioma tumor','Meningioma tumor','No tumor','Pituitary tumor']
                    st.write(f':red[{labels[indices]}]')
                    #
                    # surity graph
                    #  
                    graph_side,empty_side=st.columns(2)
                    with graph_side:
                        cnn_surity_df=pd.DataFrame()
                        cnn_surity_df['Type']=["Glioma","Maningioma","No Tumor","Pituitory"]
                        cnn_surity_df["percentage"]=np.array(a).transpose() * 100
                        st.bar_chart(cnn_surity_df,x='Type',width=250)
                    # 
                    # might use it later
                    # /////////////////////////////////////////////////
                    # st.write(cnn_surity_df["percentage"][indices])
                    # /////////////////////////////////////////////////



                    # print(model.summary())
                    tab_0,tab_1, tab_2 =st.tabs([":blue[About Model]","Model Summary","Accuracy and Loss"])

                    # [model Summary including name and types of layer and other parameters]
                    with tab_1:
                        table=pd.DataFrame(columns=["Layer Name","Type","Shape","Params"])
                        for layer in model.layers:
                            # new_row={"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape,"Params":layer.count_params()}
                            # table = table.append(new_row, ignore_index=True)
                            new_row=[layer.name,  layer.__class__.__name__, str(layer.output_shape), layer.count_params()]
                            table.loc[len(table.index)]=new_row
                        st.dataframe(table,use_container_width=True)    
                        
                        st.write(f':green[Total Trainable Parameters: {count_params(model.trainable_weights)}]')
                        st.write(f':green[Total Non-Trainable Parameters: {count_params(model.non_trainable_weights)}]')
                        # st.code(model.summary())
                    
                    #[Accuracy and Loss Graphs]
                    with tab_2:
                        with open('classify/trainHistoryDict', "rb") as file_pi:
                            history = pickle.load(file_pi)
                        acc_col,temp1,loss_col,temp2=st.columns([5,1,5,1])
                        with acc_col:
                            model_acc = history['accuracy']
                            model_val_acc = history['val_accuracy']
                            epochs = range(len(model_acc))
                            dict={"Epochs":epochs,"Training Accuracy":model_acc,"Validation Accuracy":model_val_acc}
                            df=pd.DataFrame(dict)
                            st.line_chart(df,x="Epochs",y=["Training Accuracy","Validation Accuracy"],use_container_width=True)

                        with loss_col:
                            model_loss = history['loss']
                            model_val_loss = history['val_loss']
                            epochs = range(len(model_loss))
                            dict={"Epochs":epochs,"Training Loss":model_loss,"Validation Loss":model_val_loss}
                            df=pd.DataFrame(dict)
                            st.line_chart(df,x="Epochs",y=["Training Loss","Validation Loss"],use_container_width=True)

            # Classification using Transfer Leraning
            with st.expander("Using Transfer Learning"):
                with st.spinner("Classifying..."):
                    model_tl =tf.keras.saving.load_model("tl_classification/model_tl.h5")
                    # st.image(image)
                    # img_array=cv2.imread(image)
                    img_array=cv2.resize(image,(224, 224))
                    image_array = np.array(img_array)
                    # model_tl.predict(img_array)

                    def inverse_classes(num):
                        # st.write(num)
                        if num==0:
                            return 'Glioma Tumor'
                        elif num==1:
                            return 'Meningioma Tumor'
                        elif num==2:
                            return 'No Tumor'
                        else:
                            return 'Pituitary Tumor'
                        
                   
                    
                    temp_pred=model_tl.predict(np.reshape(img_array,(-1,224,224,3)))
                    # st.write(np.argmax(model_tl.predict(np.reshape(img_array,(-1,224,224,3))),axis=1))
                    pred_class=inverse_classes(np.argmax(temp_pred,axis=1))
                    st.write(f":red[{pred_class}]")
                    # 
                    # Surity graph
                    # 
                    graph_side,empty_side=st.columns(2)
                    with graph_side:
                        tl_surity_df=pd.DataFrame()
                        tl_surity_df['Type']=labels
                        tl_surity_df["percentage"]=np.array(temp_pred).transpose()*100
                        st.bar_chart(tl_surity_df,x='Type')
                    # 
                    # might use it later
                    # /////////////////////////////////////////////////
                    # (tl_surity_df["percentage"][np.argmax(temp_pred)])
                    # ////////////////////////////////////////////////

                    # 
                    tab_0,tab_1, tab_2 =st.tabs([":blue[About Model]","Model Summary","Accuracy and Loss"])
                    # [model Summary including name and types of layer and other parameters]
                    with tab_1:
                        table_tl=pd.DataFrame(columns=["Layer Name","Type","Shape","Params"])
                        for layer in model_tl.layers:
                            # new_row={"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape,"Params":layer.count_params()}
                            # table = table.append(new_row, ignore_index=True)
                            new_row=[layer.name,  layer.__class__.__name__, str(layer.output_shape), layer.count_params()]
                            table_tl.loc[len(table_tl.index)]=new_row
                        st.dataframe(table_tl,use_container_width=True)    
                        
                        st.write(f':green[Total Trainable Parameters: {count_params(model_tl.trainable_weights)}]')
                        st.write(f':green[Total Non-Trainable Parameters: {count_params(model_tl.non_trainable_weights)}]')
                        # st.code(model.summary())
                    
                    #[Accuracy and Loss Graphs]
                    with tab_2:
                        with open('tl_classification/history_tl', "rb") as file_pi:
                            history_tl = pickle.load(file_pi)
                        acc_col,temp1,loss_col,temp2=st.columns([5,1,5,1])
                        with acc_col:
                            model_acc = history_tl['accuracy']
                            model_val_acc = history_tl['val_accuracy']
                            epochs = range(len(model_acc))
                            dict={"Epochs":epochs,"Training Accuracy":model_acc,"Validation Accuracy":model_val_acc}
                            df=pd.DataFrame(dict)
                            st.line_chart(df,x="Epochs",y=["Training Accuracy","Validation Accuracy"],use_container_width=True)

                        with loss_col:
                            model_loss = history_tl['loss']
                            model_val_loss = history_tl['val_loss']
                            epochs = range(len(model_loss))
                            dict={"Epochs":epochs,"Training Loss":model_loss,"Validation Loss":model_val_loss}
                            df=pd.DataFrame(dict)
                            st.line_chart(df,x="Epochs",y=["Training Loss","Validation Loss"],use_container_width=True)
                        

if selected == "Detection Methods":
    st.title("History of AI in Brain Tumor Detection")
    test.main()


if selected == "Classification Methods":
    st.title("Study of Brain Tumor Classification")
    classification.main()


if selected =="About & Contact":
    import contact
    st.title("About")
    col1,col2,col3=st.columns([1,8,1])
    with col2:
        about_data='''
        This project offers a cutting-edge solution for brain tumor detection and classification. This website provides a comprehensive suite of tools to support medical professionals in accurately diagnosing brain tumors and providing the best possible care for their patients.

    One of the key features of the website is the ability to detect brain tumors with high accuracy using MRI images. It utilizes advanced machine learning algorithms to analyze medical imaging data and quickly identify potential tumors. This can significantly reduce the time and effort required for manual analysis, allowing doctors to focus on providing personalized treatment plans for their patients.

    In addition to brain tumor detection, this website offers a feature for classifying the type of tumor using state-of-the-art algorithms that can accurately identify the specific type of tumor present. This information is critical for determining the appropriate course of treatment.

    It also provides a comprehensive study and review of techniques used for brain tumor detection since 2010, offering access to the latest advances in brain tumor detection and diagnosis. Additionally, the interactive overview of the major steps involved in the studies can help individuals better understand the process of brain tumor detection and classification.

    Overall, this website represents a significant advance in the field of brain tumor detection and classification, utilizing MRI images to deliver high accuracy results. It provides a powerful tool for medical professionals to deliver personalized care to their patients and serves as an educational resource for those looking to learn more about this important area of research.
        
        '''
        st.write(about_data)
        
    with st.expander("Contact"):
        
        contact.main()