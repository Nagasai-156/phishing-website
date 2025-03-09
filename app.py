# Step 1: Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from urllib.parse import urlparse
import re
import requests
from bs4 import BeautifulSoup
from sklearn.exceptions import NotFittedError

# Step 2: Load the trained model and scaler with error handling
try:
    with open('phishing_model (1).pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'phishing_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {str(e)}")
    st.stop()

# Step 3: Define feature extraction function with robust defaults
def extract_features(url):
    """Extract features from the URL with fallback values for robustness."""
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname if parsed_url.hostname else ''
        path = parsed_url.path
        query = parsed_url.query

        # Basic feature extraction from URL
        features = {
            'length_url': len(url),
            'length_hostname': len(hostname) if hostname else 0,
            'ip': 1 if hostname and re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname) else 0,
            'nb_dots': url.count('.'),
            'nb_qm': url.count('?'),
            'nb_eq': url.count('='),
            'nb_slash': url.count('/'),
            'nb_www': 1 if 'www' in hostname.lower() else 0,
            'ratio_digits_url': sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
            'ratio_digits_host': sum(c.isdigit() for c in hostname) / len(hostname) if hostname else 0,
            'tld_in_subdomain': 1 if hostname and '.' in hostname.split('.')[0] else 0,
            'prefix_suffix': 1 if hostname and '-' in hostname else 0,
            'shortest_word_host': min([len(word) for word in hostname.split('.') if word], default=0) if hostname else 0,
            'longest_words_raw': max([len(word) for word in re.split(r'[/\-.?=&]', url) if word], default=0),
            'longest_word_path': max([len(word) for word in re.split(r'[/\-.?=&]', path) if word], default=0),
            'phish_hints': sum([url.lower().count(word) for word in ['login', 'secure', 'account', 'update', 'verify']]),
            'nb_hyperlinks': 0,  # Default, updated below if webpage is accessible
            'ratio_intHyperlinks': 0,  # Default
            'empty_title': 0,  # Default
            'domain_in_title': 0,  # Default
            'domain_age': -1,  # Default (unknown)
            'google_index': 0,  # Default
            'page_rank': 0  # Default
        }

        # Attempt to fetch webpage content for additional features
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a')
                features['nb_hyperlinks'] = len(links)
                if features['nb_hyperlinks'] > 0:
                    internal_links = sum(1 for a in links if urlparse(a.get('href', '')).netloc == parsed_url.netloc)
                    features['ratio_intHyperlinks'] = internal_links / features['nb_hyperlinks']
                title = soup.title.string if soup.title and soup.title.string else ''
                features['empty_title'] = 1 if not title.strip() else 0
                features['domain_in_title'] = 1 if hostname and hostname.lower() in title.lower() else 0
        except requests.RequestException:
            pass  # Use defaults if webpage fetch fails

        return features

    except Exception as e:
        st.warning(f"Feature extraction error: {str(e)}. Using default values.")
        return {key: 0 for key in [
            'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_qm', 'nb_eq', 'nb_slash',
            'nb_www', 'ratio_digits_url', 'ratio_digits_host', 'tld_in_subdomain',
            'prefix_suffix', 'shortest_word_host', 'longest_words_raw', 'longest_word_path',
            'phish_hints', 'nb_hyperlinks', 'ratio_intHyperlinks', 'empty_title',
            'domain_in_title', 'domain_age', 'google_index', 'page_rank'
        ]}

# Step 4: Prediction function with reasoning
def predict_phishing(url):
    """Predict if the URL is phishing or legitimate and provide reasons."""
    try:
        # Extract features
        features = extract_features(url)
        
        # Ensure feature order matches training
        feature_order = [
            'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_qm', 'nb_eq', 'nb_slash',
            'nb_www', 'ratio_digits_url', 'ratio_digits_host', 'tld_in_subdomain',
            'prefix_suffix', 'shortest_word_host', 'longest_words_raw', 'longest_word_path',
            'phish_hints', 'nb_hyperlinks', 'ratio_intHyperlinks', 'empty_title',
            'domain_in_title', 'domain_age', 'google_index', 'page_rank'
        ]
        feature_df = pd.DataFrame([features])[feature_order]
        
        # Scale features
        feature_scaled = scaler.transform(feature_df)
        
        # Predict
        prediction = model.predict(feature_scaled)
        confidence = model.predict_proba(feature_scaled)[0][prediction[0]]  # Confidence score
        result = 'Phishing' if prediction[0] == 1 else 'Legitimate'

        # Determine reasons based on key features
        reasons = []
        if result == 'Phishing':
            if features['ip'] == 1:
                reasons.append("Uses an IP address instead of a domain name, common in phishing.")
            if features['phish_hints'] > 0:
                reasons.append(f"Contains suspicious keywords like 'login' or 'secure' ({features['phish_hints']} instances).")
            if features['length_url'] > 70:
                reasons.append(f"URL is unusually long ({features['length_url']} characters).")
            if features['nb_www'] == 0:
                reasons.append("Lacks 'www', which is less common in legitimate sites.")
            if features['ratio_digits_url'] > 0.1:
                reasons.append(f"High ratio of digits in URL ({features['ratio_digits_url']:.2f}).")
            if features['prefix_suffix'] == 1:
                reasons.append("Hostname contains a hyphen, often used in phishing.")
            if not reasons:  # Fallback if no clear reason
                reasons.append("Combination of subtle features flagged by the model.")
        else:  # Legitimate
            if features['nb_www'] == 1:
                reasons.append("Includes 'www', typical for legitimate sites.")
            if features['length_url'] < 50:
                reasons.append(f"URL is short and simple ({features['length_url']} characters).")
            if features['phish_hints'] == 0:
                reasons.append("No suspicious keywords like 'login' or 'secure' found.")
            if features['ip'] == 0:
                reasons.append("Uses a proper domain name, not an IP address.")
            if features['nb_hyperlinks'] > 10:
                reasons.append(f"Contains many hyperlinks ({features['nb_hyperlinks']}), suggesting a content-rich site.")
            if features['domain_in_title'] == 1:
                reasons.append("Domain matches webpage title, indicating authenticity.")
            if not reasons:  # Fallback
                reasons.append("Typical features of a legitimate site detected.")

        return result, confidence, reasons
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, ["Unable to determine reasons due to an error."]

# Step 5: Custom CSS for single box display with dark text
st.markdown("""
    <style>
    .reason-box-phishing {
        background-color: #ffcccc;
        border: 1px solid #ff9999;
        border-radius: 5px;
        padding: 15px;
        margin-top: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        color: #333333;  /* Dark text */
    }
    .reason-box-legitimate {
        background-color: #ccffcc;
        border: 1px solid #99ff99;
        border-radius: 5px;
        padding: 15px;
        margin-top: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        color: #333333;  /* Dark text */
    }
    .reason-box-phishing ul, .reason-box-legitimate ul {
        margin: 0;
        padding-left: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Step 6: Streamlit UI with reasons in a single colored box
def main():
    st.title("Phishing URL Detector")
    st.write("A machine learning-based tool to classify URLs as Phishing or Legitimate.")
    
    # Input field for URL
    url_input = st.text_input("Enter URL (e.g., http://example.com):", "")
    
    # Predict button
    if st.button("Check URL"):
        if not url_input:
            st.warning("Please enter a URL to check.")
        elif not url_input.startswith(('http://', 'https://')):
            st.warning("Please include 'http://' or 'https://' in the URL.")
        else:
            with st.spinner("Analyzing URL..."):
                result, confidence, reasons = predict_phishing(url_input)
                if result == "Error":
                    st.error("An error occurred during prediction. Please try another URL.")
                    st.write("Reasons:", ", ".join(reasons))
                elif result == 'Phishing':
                    st.error(f"The URL '{url_input}' is predicted to be **Phishing**! (Confidence: {confidence:.2%})")
                    st.write("**Reasons for Phishing Classification:**")
                    reason_html = "<div class='reason-box-phishing'><ul>" + "".join([f"<li>{reason}</li>" for reason in reasons]) + "</ul></div>"
                    st.markdown(reason_html, unsafe_allow_html=True)
                else:
                    st.success(f"The URL '{url_input}' is predicted to be **Legitimate**! (Confidence: {confidence:.2%})")
                    st.write("**Reasons for Legitimate Classification:**")
                    reason_html = "<div class='reason-box-legitimate'><ul>" + "".join([f"<li>{reason}</li>" for reason in reasons]) + "</ul></div>"
                    st.markdown(reason_html, unsafe_allow_html=True)
    
    # Additional info for professors
    st.markdown("""
    ### How It Works
    - **Model**: LightGBM Classifier trained on 11,430 URLs with 87 features.
    - **Features Used**: URL structure (e.g., length, dots), hostname properties, and basic webpage content (if accessible).
    - **Accuracy**: Achieved ~97% on test data (varies with real-world URLs).
    - **Limitations**: Some features (e.g., domain age, Google index) require external APIs and are set to defaults here.
    """)

# Step 7: Run the app
if __name__ == "__main__":
    main()