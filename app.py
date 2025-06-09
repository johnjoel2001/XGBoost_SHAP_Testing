# import streamlit as st
# import json
# import pandas as pd
# import joblib
# import tempfile
# import os
# from pdf_processor import extract_structured_data  
# import numpy as np
# import lime.lime_tabular
# from dotenv import load_dotenv
# import google.generativeai as genai 

# load_dotenv()
# api_key = os.getenv("API_KEY")
# genai.configure(api_key=api_key)  # Fixed: Use configure instead of Client

# # Load the trained model
# model = joblib.load('logistic regression.pkl')

# st.title("Semen Analysis Prediction")

# # File uploader
# uploaded_file = st.file_uploader("Upload your semen analysis PDF", type=["pdf","jpg","png"])

# if uploaded_file:
#     # Get correct file extension (e.g., .pdf, .jpg)
#     file_suffix = os.path.splitext(uploaded_file.name)[1].lower()

#     # Save to a temp file with that suffix
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

#     # Extract structured data using backend (works for PDF or image)
#     st.info("Extracting data from file...")
#     try:
#         result = extract_structured_data(tmp_file_path)
#         extracted_data = result.model_dump()
#         # Validate critical fields (e.g., name)
#         # if not extracted_data.get("patient_info", {}).get("name"):
#         #     raise ValueError("❌ Invalid report: missing patient name. This is likely not a valid semen analysis report.")

#         # Save extracted JSON for reference
#         json_path = tmp_file_path.replace(".pdf", ".json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(extracted_data, f, indent=2, ensure_ascii=False)

#         st.success("Data extracted successfully!")
#         #st.json(extracted_data)  # Display extracted JSON

#         # Extract relevant features for prediction
#         volume = extracted_data['semen_analysis']['volume']['value']
#         concentration = extracted_data['semen_analysis']['concentration']['value']
#         motility = extracted_data['semen_analysis']['motility']['value']

#         # Create DataFrame for model input
#         input_data = pd.DataFrame({
#             'Volume': [volume],
#             'Concentration': [concentration],
#             'Motility': [motility]
#         })

#         # Make prediction
#         probability = model.predict_proba(input_data)[:, 1][0]

#         # Display the result
#         st.subheader("Fertility score")
#         st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

#         # Initialize session state for sidebar selection
#         if "selected_option" not in st.session_state:
#             st.session_state.selected_option = "0-40"  # Default option

#         st.sidebar.title("📊 Interpret Scores")

#         # **Display All Score Ranges and Explanations**
#         st.sidebar.subheader("0-40%: Low probability of fertility")
#         st.sidebar.write("A score in this range suggests, based on our training data, that the semen analysis results indicate infertility.")

#         st.sidebar.subheader("40-70%: Inconclusive Results")
#         st.sidebar.write("A score in this range is considered inconclusive. Based on our training data, you may have aspects that indicate fertility and others that suggest otherwise.")

#         st.sidebar.subheader("70-100%: High probability of fertility")
#         st.sidebar.write("A score in this range suggests a high chance of fertility. Based on our training data, your sperm shows strong signs of fertility.")

#         datadisplay = {
#             "Label Name": ["Volume", "Concentration", "Motility"],  
#             "Value": [volume, concentration, motility]  
#         }

#         df = pd.DataFrame(datadisplay)
#         df = df.set_index("Label Name")

#         # **Display Table in Streamlit**
#         st.table(df)

#         # **LIME EXPLANATION**

#         # **Generate Perturbed Training Data Using Percentage-Based Changes**
#         num_samples = 100  # Number of perturbed instances
#         perturbation_percentage = 0.10  # 10% increase/decrease

#         # Create perturbations by modifying input data within ±10%
#         training_data = np.array([
#             input_data.iloc[0].values * (1 + np.random.uniform(-perturbation_percentage, perturbation_percentage, len(input_data.columns)))
#             for _ in range(num_samples)
#         ])

#         # Ensure values stay realistic (e.g., no negative semen volume)
#         training_data = np.maximum(training_data, 0)

#         # **Create LIME Explainer**
#         explainer = lime.lime_tabular.LimeTabularExplainer(
#             training_data=training_data,
#             feature_names=input_data.columns.tolist(),
#             class_names=["Infertile", "Fertile"],
#             mode="classification"
#         )

#         # **Generate LIME Explanation**
#         exp = explainer.explain_instance(
#             input_data.iloc[0].values,  # Instance to explain
#             model.predict_proba,
#             num_features=len(input_data.columns)
#         )

#         # Convert LIME explanation to a text-friendly format
#         lime_explanation = []
#         for feature, importance in exp.as_list():
#             # Prepare feature names and their respective importance (positive or negative)
#             lime_explanation.append(f"Feature: {feature}, Importance: {importance:.4f}")

#         # Create a prompt with the LIME explanation
#         lime_explanation_text = "\n".join(lime_explanation)

#         # Prepare the prompt for Gemini
#         prompt = f"""
#         You are an expert medical assistant explaining the prediction of fertility based on semen analysis results.

#         The model predicts a fertility probability of {probability*100:.2f}%. Here are the factors that contributed to the prediction, based on the LIME explanation:
#         {lime_explanation_text}

#         Please generate an explanation for the user on how these factors affect fertility, in an easy-to-understand manner. Follow this EXACT format:

#         ### SUMMARY:
#         [Write a concise overall explanation of the fertility score here]

#         ### FEATURE ANALYSIS:
#         - **[Feature Name 1] (Impact: High/Medium/Low)**
#         [Explanation of how this feature affects fertility and the WHO standard for this feature]
#         - **[Feature Name 2] (Impact: High/Medium/Low)**
#         [Explanation of how this feature affects fertility and the WHO standard for this feature]
#         - **[Feature Name 3] (Impact: High/Medium/Low)**
#         [Explanation of how this feature affects fertility and the WHO standard for this feature]

#         ### RECOMMENDATION:
#         [Write a general recommendation based on the results]

#         It is crucial that you follow this exact format with these exact section headings.
#         """

#         model_id = "gemini-2.0-flash-exp"  # Updated model name
#         # Call Gemini API to generate the explanation
#         gemini_model = genai.GenerativeModel(model_id)  # Create model instance
#         response = gemini_model.generate_content(prompt)  # Generate content
    
#         # Display the explanation in Streamlit
#         st.subheader("Model Explanation (via Gemini)")
#         st.write(response.text)
       
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")

#     # Cleanup temp file
#     os.remove(tmp_file_path)

# footer = st.container()
# with footer:
#     st.caption("Disclaimer: This tool provides estimates based on available data and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Results are for informational purposes only. Always consult a qualified healthcare provider for personalized medical guidance.")
#     st.markdown("[Terms of Service](https://urldefense.com/v3/__https://herafertility.co/policies/terms-of-service__;!!OToaGQ!sCo8Bfe3ZcWHgL0DRadehsIiVtW7Zh26T4qsqdIuulceNTFNeURTzNTtqLiLP6BiE7EtGDPZLcCDQ8_09BKegDk$) | [Privacy Policy](https://urldefense.com/v3/__https://herafertility.co/policies/privacy-policy__;!!OToaGQ!sCo8Bfe3ZcWHgL0DRadehsIiVtW7Zh26T4qsqdIuulceNTFNeURTzNTtqLiLP6BiE7EtGDPZLcCDQ8_0-HKOJfA$)")




# import streamlit as st
# import json
# import pandas as pd
# import pickle  # Changed from joblib to pickle
# import tempfile
# import os
# from pdf_processor import extract_structured_data  
# import numpy as np
# import lime.lime_tabular
# from dotenv import load_dotenv
# import google.generativeai as genai 

# load_dotenv()
# api_key = os.getenv("API_KEY")
# genai.configure(api_key=api_key)  # Fixed: Use configure instead of Client

# # Load the trained XGBoost model
# with open('xgboost_fertility_model_20250609_180158.pkl', 'rb') as f:  # Changed to pickle loading
#     model_package = pickle.load(f)
#     model = model_package['model']  # Extract the actual model
#     feature_names = model_package['feature_names']  # Get feature names from package

# st.title("Semen Analysis Prediction")

# # File uploader
# uploaded_file = st.file_uploader("Upload your semen analysis PDF", type=["pdf","jpg","png"])

# if uploaded_file:
#     # Get correct file extension (e.g., .pdf, .jpg)
#     file_suffix = os.path.splitext(uploaded_file.name)[1].lower()

#     # Save to a temp file with that suffix
#     with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         tmp_file_path = tmp_file.name

#     # Extract structured data using backend (works for PDF or image)
#     st.info("Extracting data from file...")
#     try:
#         result = extract_structured_data(tmp_file_path)
#         extracted_data = result.model_dump()
#         # Validate critical fields (e.g., name)
#         # if not extracted_data.get("patient_info", {}).get("name"):
#         #     raise ValueError("❌ Invalid report: missing patient name. This is likely not a valid semen analysis report.")

#         # Save extracted JSON for reference
#         json_path = tmp_file_path.replace(".pdf", ".json")
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(extracted_data, f, indent=2, ensure_ascii=False)

#         st.success("Data extracted successfully!")
#         #st.json(extracted_data)  # Display extracted JSON

#         # Extract relevant features for prediction
#         volume = extracted_data['semen_analysis']['volume']['value']
#         concentration = extracted_data['semen_analysis']['concentration']['value']
#         motility = extracted_data['semen_analysis']['motility']['value']

#         # Create DataFrame for model input using exact feature names from training
#         input_data = pd.DataFrame({
#             feature_names[0]: [volume],      # Use actual feature names from model
#             feature_names[1]: [concentration],
#             feature_names[2]: [motility]
#         })

#         # Make prediction
#         probability = model.predict_proba(input_data)[:, 1][0]

#         # Display the result
#         st.subheader("Fertility score")
#         st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

#         # Initialize session state for sidebar selection
#         if "selected_option" not in st.session_state:
#             st.session_state.selected_option = "0-40"  # Default option

#         st.sidebar.title("📊 Interpret Scores")

#         # **Display All Score Ranges and Explanations**
#         st.sidebar.subheader("0-40%: Low probability of fertility")
#         st.sidebar.write("A score in this range suggests, based on our training data, that the semen analysis results indicate infertility.")

#         st.sidebar.subheader("40-70%: Inconclusive Results")
#         st.sidebar.write("A score in this range is considered inconclusive. Based on our training data, you may have aspects that indicate fertility and others that suggest otherwise.")

#         st.sidebar.subheader("70-100%: High probability of fertility")
#         st.sidebar.write("A score in this range suggests a high chance of fertility. Based on our training data, your sperm shows strong signs of fertility.")

#         datadisplay = {
#             "Label Name": ["Volume", "Concentration", "Motility"],  
#             "Value": [volume, concentration, motility]  
#         }

#         df = pd.DataFrame(datadisplay)
#         df = df.set_index("Label Name")

#         # **Display Table in Streamlit**
#         st.table(df)

#         # **LIME EXPLANATION**

#         # **Generate Perturbed Training Data Using Percentage-Based Changes**
#         num_samples = 100  # Number of perturbed instances
#         perturbation_percentage = 0.10  # 10% increase/decrease

#         # Create perturbations by modifying input data within ±10%
#         training_data = np.array([
#             input_data.iloc[0].values * (1 + np.random.uniform(-perturbation_percentage, perturbation_percentage, len(input_data.columns)))
#             for _ in range(num_samples)
#         ])

#         # Ensure values stay realistic (e.g., no negative semen volume)
#         training_data = np.maximum(training_data, 0)

#         # **Create LIME Explainer**
#         explainer = lime.lime_tabular.LimeTabularExplainer(
#             training_data=training_data,
#             feature_names=feature_names,  # Use feature names from model package
#             class_names=["Infertile", "Fertile"],
#             mode="classification"
#         )

#         # **Generate LIME Explanation**
#         exp = explainer.explain_instance(
#             input_data.iloc[0].values,  # Instance to explain
#             model.predict_proba,
#             num_features=len(input_data.columns)
#         )

#         # Convert LIME explanation to a text-friendly format
#         lime_explanation = []
#         for feature, importance in exp.as_list():
#             # Prepare feature names and their respective importance (positive or negative)
#             lime_explanation.append(f"Feature: {feature}, Importance: {importance:.4f}")

#         # Create a prompt with the LIME explanation
#         lime_explanation_text = "\n".join(lime_explanation)

#         # Prepare the prompt for Gemini
#         prompt = f"""
#         You are an expert medical assistant explaining the prediction of fertility based on semen analysis results.

#         The model predicts a fertility probability of {probability*100:.2f}%. Here are the factors that contributed to the prediction, based on the LIME explanation:
#         {lime_explanation_text}

#         Please generate an explanation for the user on how these factors affect fertility, in an easy-to-understand manner. Follow this EXACT format:

#         ### SUMMARY:
#         [Write a concise overall explanation of the fertility score here]

#         ### FEATURE ANALYSIS:
#         - **[Feature Name 1] (Impact: High/Medium/Low)**
#         [Explanation of how this feature affects fertility and the WHO standard for this feature]
#         - **[Feature Name 2] (Impact: High/Medium/Low)**
#         [Explanation of how this feature affects fertility and the WHO standard for this feature]
#         - **[Feature Name 3] (Impact: High/Medium/Low)**
#         [Explanation of how this feature affects fertility and the WHO standard for this feature]

#         ### RECOMMENDATION:
#         [Write a general recommendation based on the results]

#         It is crucial that you follow this exact format with these exact section headings.
#         """

#         model_id = "gemini-2.0-flash-exp"  # Updated model name
#         # Call Gemini API to generate the explanation
#         gemini_model = genai.GenerativeModel(model_id)  # Create model instance
#         response = gemini_model.generate_content(prompt)  # Generate content
    
#         # Display the explanation in Streamlit
#         st.subheader("Model Explanation (via Gemini)")
#         st.write(response.text)
       
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")

#     # Cleanup temp file
#     os.remove(tmp_file_path)

# footer = st.container()
# with footer:
#     st.caption("Disclaimer: This tool provides estimates based on available data and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Results are for informational purposes only. Always consult a qualified healthcare provider for personalized medical guidance.")
#     st.markdown("[Terms of Service](https://urldefense.com/v3/__https://herafertility.co/policies/terms-of-service__;!!OToaGQ!sCo8Bfe3ZcWHgL0DRadehsIiVtW7Zh26T4qsqdIuulceNTFNeURTzNTtqLiLP6BiE7EtGDPZLcCDQ8_09BKegDk$) | [Privacy Policy](https://urldefense.com/v3/__https://herafertility.co/policies/privacy-policy__;!!OToaGQ!sCo8Bfe3ZcWHgL0DRadehsIiVtW7Zh26T4qsqdIuulceNTFNeURTzNTtqLiLP6BiE7EtGDPZLcCDQ8_0-HKOJfA$)")

import streamlit as st
import json
import pandas as pd
import pickle
import tempfile
import os
from pdf_processor import extract_structured_data  
import numpy as np
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import google.generativeai as genai 

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Load the trained model
with open('xgboost_fertility_model_20250609_180158.pkl', 'rb') as f:
    model_package = pickle.load(f)
    model = model_package['model']
    feature_names = model_package['feature_names']

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

st.title("Semen Analysis Prediction")

uploaded_file = st.file_uploader("Upload your semen analysis PDF", type=["pdf", "jpg", "png"])

if uploaded_file:
    file_suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info("Extracting data from file...")

    try:
        result = extract_structured_data(tmp_file_path)
        extracted_data = result.model_dump()

        json_path = tmp_file_path.replace(".pdf", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)

        st.success("Data extracted successfully!")

        volume = extracted_data['semen_analysis']['volume']['value']
        concentration = extracted_data['semen_analysis']['concentration']['value']
        motility = extracted_data['semen_analysis']['motility']['value']

        input_data = pd.DataFrame({
            feature_names[0]: [volume],
            feature_names[1]: [concentration],
            feature_names[2]: [motility]
        })

        probability = model.predict_proba(input_data)[:, 1][0]

        st.subheader("Fertility score")
        st.markdown(f"<h1 style='text-align: center;font-size: 4em'>{probability*100:.2f}%</h1>", unsafe_allow_html=True)

        if "selected_option" not in st.session_state:
            st.session_state.selected_option = "0-40"

        st.sidebar.title("📊 Interpret Scores")
        st.sidebar.subheader("0-40%: Low probability of fertility")
        st.sidebar.write("A score in this range suggests infertility.")
        st.sidebar.subheader("40-70%: Inconclusive Results")
        st.sidebar.write("This score is borderline; further consultation recommended.")
        st.sidebar.subheader("70-100%: High probability of fertility")
        st.sidebar.write("Strong indicators of fertility based on this data.")

        df = pd.DataFrame({
            "Label Name": ["Volume", "Concentration", "Motility"],
            "Value": [volume, concentration, motility]
        }).set_index("Label Name")
        st.table(df)

        # SHAP Analysis
        st.subheader("Feature Importance Analysis (SHAP)")
        shap_values = explainer.shap_values(input_data)
        base_value = explainer.expected_value

        # Prepare data
        feature_impact = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data.iloc[0].values,
            'SHAP_Impact': shap_values[0]
        }).sort_values(by='SHAP_Impact')

        feature_impact["Abs_Impact"] = np.abs(feature_impact["SHAP_Impact"])
        colors = ['#ff4d4d' if val < 0 else '#28a745' for val in feature_impact['SHAP_Impact']]

            # Create cleaner SHAP bar plot without showing raw input values
        fig, ax = plt.subplots(figsize=(8, 5))

        # Color coding based on impact sign
        colors = ['#ff4d4d' if val < 0 else '#28a745' for val in feature_impact['SHAP_Impact']]

        # Plot SHAP bars
        bars = ax.barh(
            feature_impact['Feature'], 
            feature_impact['SHAP_Impact'], 
            color=colors, 
            edgecolor='black'
        )

        # Annotate each bar with SHAP impact only
        # Annotate each bar with SHAP impact value (always on right side)
        # Annotate each bar with SHAP impact value
        for bar, impact in zip(bars, feature_impact['SHAP_Impact']):
            x_offset = 0.08 if impact < 0 else 0  # shift only negative bars
            ax.text(
                bar.get_width() + x_offset,
                bar.get_y() + bar.get_height() / 2,
                f'{impact:+.2f}',
                va='center',
                ha='left',
                fontsize=10,
                fontweight='bold'
            )


        # Draw vertical axis at 0 and style plot
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel("SHAP Impact on Prediction")
        ax.set_title("Feature Contribution to Fertility Prediction")
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


        # Text explanation
        def get_impact_strength(abs_impact):
            if abs_impact > 0.15:
                return "Very Strong"
            elif abs_impact > 0.10:
                return "Strong"
            elif abs_impact > 0.05:
                return "Moderate"
            else:
                return "Mild"

        st.subheader("📊 DETAILED FEATURE ANALYSIS")
        st.write("=" * 50)

        positive_factors = feature_impact[feature_impact['SHAP_Impact'] > 0.01]
        negative_factors = feature_impact[feature_impact['SHAP_Impact'] < -0.01]

        if len(positive_factors) > 0:
            st.write("✅ **POSITIVE INFLUENCES (Supporting Fertility):**")
            for _, row in positive_factors.iterrows():
                st.write(f"💪 **{row['Feature']}**: {row['Value']:.3f}")
                st.write(f"   → {get_impact_strength(row['Abs_Impact'])} POSITIVE impact (+{row['SHAP_Impact']:.3f})")
                st.write("")

        if len(negative_factors) > 0:
            st.write("⚠️ **NEGATIVE INFLUENCES (Areas for Improvement):**")
            for _, row in negative_factors.iterrows():
                st.write(f"🎯 **{row['Feature']}**: {row['Value']:.3f}")
                st.write(f"   → {get_impact_strength(row['Abs_Impact'])} NEGATIVE impact ({row['SHAP_Impact']:.3f})")
                st.write(f"   → 💡 Consider improving this parameter if possible.")
                st.write("")

    except Exception as e:
        st.error(f"Error processing the file: {e}")

    os.remove(tmp_file_path)

# Footer
footer = st.container()
with footer:
    st.caption("Disclaimer: This tool provides estimates based on available data and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
    st.markdown("[Terms of Service](https://herafertility.co/policies/terms-of-service) | [Privacy Policy](https://herafertility.co/policies/privacy-policy)")
