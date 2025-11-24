import requests
import streamlit as st

API_BASE_URL = "https://hotdog-legs-api-28817706146.us-central1.run.app"
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

st.set_page_config(
    page_title="Hotdogs or Legs?",
    page_icon="🌭",
    layout="centered",
)

# Header and Intro Section

st.title("🌭 Hotdogs or Legs? 🦵")

st.markdown(
    """
### *The Eternal Question*

In a world rife with ambiguity, one mystery has confounded millennial meme culture for years:

**Is it hot dogs...or is it legs?**

As we usher in a new era of AI, we must ask ourselves:  
*“Can machines succeed where we have failed?”*

To explore this profound dilemma, which has haunted humanity for far too long, we proudly present the Turing test of our generation - 
a model designed to finally bring clarity, closure, and perhaps...peace.

**Behold: Hotdogs or Legs.**
"""
)
st.image(
    "https://thedrum-media.imgix.net//thedrum-prod/s3/news/tmp/798481/hotdoglegs.png?w=1280&ar=default&fit=crop&crop=faces&auto=format",
    caption="The ad that captured the meme that started it all.",
    use_container_width=True,
)

st.markdown(
    """
---

### ⚠️ Disclaimer

This model is, frankly (pun intended), **not very good**.  
It was trained on:

- A bunch of cooked and prepared hotdog photos (buns, ketchup, the works)  
- A bunch of legs in every sense of the word (human legs, chair legs, chicken legs, you name it)   

It will absolutely struggle with anything nuanced, artistic, weirdly lit, or cursed (it also hates wiki images specifically).  
This entire project is **just for fun** so please calibrate expectations accordingly.
I hope you relsih (pun intended once more, tysm) using this as much as I did making it.

---
"""
)

st.write("Paste an image URL below and an *attempt* will be made to classify it as **hotdog** or **legs**.")

with st.spinner("Checking backend health..."):
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            st.success("Backend API is online!")
        else:
            st.warning(f"Backend health check returned status {resp.status_code}")
    except Exception as exc:
        st.error(f"Could not reach backend API: {exc}")

st.markdown("---")

# Image Input and Prediction

image_url = st.text_input(
    "Image URL",
    help="Paste a direct link to an image of hotdogs or legs.",
)

if image_url:
    st.markdown("### Preview")
    st.image(image_url, caption="Input image", use_container_width=True)

if st.button("Classify") and image_url:
    with st.spinner("Calling model..."):
        try:
            payload = {"image_url": image_url}
            resp = requests.post(PREDICT_URL, json=payload, timeout=15)

            if resp.status_code != 200:
                st.error(f"API error {resp.status_code}: {resp.text}")
            else:
                data = resp.json()
                label = data.get("label")
                probs = data.get("probabilities", {})

                st.markdown("### Prediction")
                st.write(f"**Predicted label:** `{label}`")

                if probs:
                    st.markdown("### Class probabilities")
                    for cls, p in probs.items():
                        st.write(f"- **{cls}**: {p:.3f}")

        except Exception as exc:
            st.error(f"Request failed: {exc}")
