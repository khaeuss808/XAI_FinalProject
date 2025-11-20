import streamlit as st
import pandas as pd

st.title("My First Streamlit App 🎉")
st.write("Hello, world! This is a simple Streamlit web app.")

# Text input
name = st.text_input("What's your name?")

if name:
    st.success(f"Nice to meet you, {name}!")

# Slider
age = st.slider("How old are you?", 0, 100, 25)
st.write(f"Your age is: {age}")

# DataFrame example
df = pd.DataFrame(
    {
        "x": [1, 2, 3, 4],
        "y": [10, 20, 30, 40],
    }
)

st.subheader("Example Data")
st.dataframe(df)

st.subheader("Line Chart")
st.line_chart(df.set_index("x"))
