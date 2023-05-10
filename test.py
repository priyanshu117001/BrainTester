import streamlit as st


def main():
    about_text = """In the early 2000s, more sophisticated machine learning algorithms such as support vector machines (SVMs) and decision trees were developed and applied to brain tumor detection. These algorithms were able to achieve higher levels of accuracy than neural networks and paved the way for the development of more advanced AI techniques.

With the advent of deep learning in the mid-2010s, researchers began to explore the use of convolutional neural networks (CNNs) for brain tumor detection. CNNs are a type of deep learning algorithm that can learn hierarchical representations of images and identify features that are relevant to a particular task, such as tumor detection.

One of the early studies on using CNNs for brain tumor detection was published in 2016 in the Journal of Healthcare Engineering. In this study, the researchers used a 3D CNN to classify brain tumors based on their appearance in MRI images.

Since then, several studies have been published that explore the use of different types of AI algorithms for brain tumor detection, including not only CNNs but also other machine learning techniques such as random forests and support vector machines.

Overall, the use of AI for brain tumor detection has evolved significantly over the past few decades, and researchers continue to explore new techniques and approaches to improve the accuracy and efficiency of tumor detection using these methods.
    """
    st.write(about_text)
    
    tab1,tab2 = st.tabs(["Learn More: ",":red[Brief overview of different Methods]"])
    with tab2:
        with st.expander("Support Vector Machines (SVMs) (2010-2015):"):
            st.write('''
                SVMs were one of the earliest machine learning techniques applied to brain tumor detection. SVMs are a type of supervised learning algorithm that can classify data into one of two categories. In brain tumor detection, SVMs were used to classify MRI images as either tumor or non-tumor. Pros of SVMs include their ability to work with small datasets and their ability to handle high-dimensional data. However, SVMs are limited by their need for a well-defined set of features, and their inability to handle noisy or overlapping data.
            ''')
        with st.expander("Decision Trees (2010-2015):"):
            st.write('''
            Decision trees are another type of supervised learning algorithm that can be used for classification. In brain tumor detection, decision trees were used to analyze MRI images and identify patterns that could indicate the presence of a tumor. Pros of decision trees include their ability to handle noisy data and their interpretability. However, decision trees are limited by their tendency to overfit the data and their inability to handle high-dimensional data.
                ''')
        with st.expander("Convolutional Neural Networks (CNNs) (2015-present):"):
            st.write('''
            CNNs are a type of deep learning algorithm that can learn hierarchical representations of images and identify features that are relevant to a particular task, such as tumor detection. In brain tumor detection, CNNs have been used to analyze MRI images and detect tumors with high accuracy. Pros of CNNs include their ability to handle large and complex datasets, their ability to learn features automatically, and their high accuracy rates. However, CNNs are limited by their computational requirements and their tendency to overfit the data if not trained properly.
                ''')
        with st.expander("Recurrent Neural Networks (RNNs) (2016-present):"):
            st.write('''
            RNNs are a type of deep learning algorithm that can process sequences of data, such as time-series data or text. In brain tumor detection, RNNs have been used to analyze MRI images over time and detect changes that could indicate the presence of a tumor. Pros of RNNs include their ability to handle sequential data and their ability to learn long-term dependencies. However, RNNs are limited by their computational requirements and their tendency to suffer from the vanishing gradient problem.
                ''')
        with st.expander("Generative Adversarial Networks (GANs) (2018-present):"):
            st.write('''
            GANs are a type of deep learning algorithm that can generate new data that is similar to a given dataset. In brain tumor detection, GANs have been used to generate synthetic MRI images that can be used to augment existing datasets and improve the accuracy of AI algorithms. Pros of GANs include their ability to generate realistic data and their potential to overcome the problem of imbalanced datasets. However, GANs are limited by their tendency to generate biased or unrealistic data if not trained properly.
                ''')
