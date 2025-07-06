import streamlit as st
import tempfile
import predict  # make sure scripts/ has __init__.py

st.set_page_config(page_title="Grape Mealybug Classifier", page_icon="🍇")

st.title("🍇 Grape Mealybug Detection")
st.markdown("Upload a grape leaf image to detect the presence of mealybugs.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running prediction..."):
        try:
            label, confidence = predict.main(tmp_path)
            st.success(f"✅ Prediction: **{label}** ({confidence:.2%} confidence)")
        except Exception as e:
            st.error("🚫 Error during prediction")
            st.exception(e)
