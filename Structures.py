import json
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

st.set_page_config(
    page_title="NeuralLens",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_model():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=300,
        temperature=0.2,
    )

    return ChatHuggingFace(llm=llm)

model = load_model()
st.markdown("""
<style>

.stApp{
background:#000000;
color:white;
}

h1,h2,h3,p,label,span{
color:white!important;
}

textarea{
background:#111111!important;
color:white!important;
border:1px solid #00f5c4!important;
}

.stButton button{
background:#00f5c4!important;
color:black!important;
font-weight:bold;
}

</style>
""",unsafe_allow_html=True)

st.title("🔍 NeuralLens")

st.write("AI Review Intelligence Platform")
review = st.text_area(
    "Paste Review",
    height=200,
    placeholder="Paste Instagram, Amazon, Hotel or Restaurant Review..."
)

analyze = st.button("Analyze Review")
if analyze:

    if review.strip()=="":

        st.warning("Please enter a review.")

    else:

        prompt=f"""
Analyze the following review.

Return ONLY valid JSON.

{{
"summary":"",
"sentiment":"",
"key_themes":[]
}}

Review:

{review}
"""

        try:

            response=model.invoke(prompt)

            content=response.content

            content=content.replace("```json","")
            content=content.replace("```","")

            result=json.loads(content)

            st.subheader("Sentiment")

            if result["sentiment"].lower()=="positive":
                st.success(result["sentiment"])
            else:
                st.error(result["sentiment"])

            st.subheader("Summary")
            st.write(result["summary"])

            st.subheader("Key Themes")

            for theme in result["key_themes"]:
                st.write("✅",theme)

            st.subheader("Raw JSON")

            st.json(result)

        except Exception as e:

            st.error(e)