import streamlit as st


contact_form='''
<form action="https://formsubmit.co/priyanshu.agarwal117@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your Name" required >
     <input type="email" name="email" placeholder="Email" required>
     <textarea name="message" placeholder="Your Message Here"></textarea>
     <button type="submit">Send</button>
</form>
'''
linked_in='''
<a href="https://www.linkedin.com/in/priyanshu117001/">Linked In</a>
'''

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

# local_css("css/streamlit.css")

def main():
    st.caption("👋 Hi, I am Priyanshu Agarwal")
    st.markdown(linked_in,unsafe_allow_html=True)
    st.header(":mailbox: Get In Touch With Me!")
    st.markdown(contact_form,unsafe_allow_html=True)
    
