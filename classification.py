import streamlit as st
import webbrowser

def main():
    about_text = """
    Brain tumors can be classified into several types based on their histology, which refers to the appearance of the tumor cells under a microscope. The classification of brain tumors is important for determining treatment options and predicting outcomes.
    The different types of brain tumors that can be classified using AI include:

:red[Gliomas]:G Gliomas are tumors that originate in the glial cells of the brain. Gliomas can be classified into different types, including astrocytomas, oligodendrogliomas, and glioblastomas.

:red[Meningiomas]:a Meningiomas are tumors that arise from the meninges, which are the membranes that cover the brain and spinal cord.

:red[Pituitary adenomas]:P Pituitary adenomas are tumors that arise from the pituitary gland in the brain.

:red[Metastatic brain tumors]:M Metastatic brain tumors are tumors that have spread to the brain from other parts of the body.

Classifying brain tumors using AI can help clinicians make more accurate diagnoses and determine the best course of treatment for patients.
    """
    st.write(about_text)
    st.divider()
    st.subheader('Polpular Methods for Brain Tumor Classification')
    col_0,col_1,col_2 = st.columns([1,5,1])
    with col_0:
        pass
    with col_1:
        col1,col2=st.columns([2,1])
        with col1:
            st.subheader("Support Vector Machines (SVMs): ")
            st.write("SVMs are a type of machine learning algorithm that can be used for classification tasks. SVMs can be trained on features extracted from MRI images, such as the shape or texture of the tumor, and can learn to classify tumors into different types based on these features.")
        with col2:    
            st.image("https://miro.medium.com/v2/resize:fit:750/0*Zi-WEyWcYAyPK_Cl.gif",use_column_width="auto")
            if st.button("What is SVM :question: "):
                webbrowser.open("https://medium.com/swlh/the-support-vector-machine-basic-concept-a5106bd3cc5f")
        st.divider()
        col1,col2=st.columns([1,1])
        with col1:    
            st.subheader("Convolutional Neural Networks (CNNs)")
            st.write("CNNs are a type of deep learning algorithm that can be used to classify brain tumors based on MRI images. These models are trained on large datasets of MRI images and can learn to identify patterns in the images that correspond to different types of tumors. CNNs have been shown to achieve high accuracy in classifying brain tumors.")
        with col2:    
            st.image('https://miro.medium.com/v2/resize:fit:786/1*uQEWL_vd0Vfp5OwhywiveA.gif',use_column_width="auto")
            if st.button("What is CNN :question: "):
                webbrowser.open("https://medium.com/nybles/a-brief-guide-to-convolutional-neural-network-cnn-642f47e88ed4#:~:text=Convolutional%20Neural%20Network(CNN%20or,classification%2C%20object%20detection%2C%20etc.")
        st.divider()
        col1,col2=st.columns([2,1])
        with col1:
            st.subheader("Random Forests")
            st.write("Random forests are an ensemble learning algorithm that can be used for classification tasks. Random forests work by constructing multiple decision trees and combining their results to make a final prediction. Random forests can be trained on features extracted from MRI images to classify brain tumors into different types.")
        with col2:    
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*yoW30XVqAnKOA-7AArXqNg.gif",use_column_width="auto")
            if st.button("What is Random Forest :question:"):
                webbrowser.open("https://medium.com/x8-the-ai-community/random-forests-an-intuitive-understanding-13cece15ba88")
    with col_2:
        pass
        # st.subheader("Deep Radiomics")
        # st.write("Deep radiomics is a method that combines deep learning algorithms with radiomics features extracted from MRI images. Radiomics refers to the quantitative analysis of features extracted from medical images, such as texture or shape. Deep radiomics can be used to classify brain tumors based on these features, achieving high accuracy in some studies.")
      
