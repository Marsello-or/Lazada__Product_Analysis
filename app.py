import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Models and Preprocessors ---
@st.cache_resource
def load_sales_model():
    return joblib.load('xgb_sales_pipeline_full.joblib')

@st.cache_resource
def load_segmentation_model_bundle():
    return joblib.load('kmeans_lazada_segmentation.joblib')

sales_pipeline = load_sales_model()
segmentation_bundle = load_segmentation_model_bundle()
kmeans_model = segmentation_bundle['model']
kmeans_scaler = segmentation_bundle['scaler']
kmeans_features = segmentation_bundle['features']
segment_names_map = segmentation_bundle['segment_names']

# --- 2. Feature Engineering Functions (re-implement from notebook) ---
def create_features_for_sales_prediction(df_input):
    data = df_input.copy()

    # Recalculate reviews_per_price
    data['reviews_per_price'] = data['reviews'] / (data['final_price'] + 1)

    # Recalculate seller_tier
    def assign_seller_tier(row):
        if row['lazmall'] == True and row['is_super_seller'] == True:
            return 'LazMall_Super'
        elif row['lazmall'] == True:
            return 'LaMall_only'
        elif row['is_super_seller'] == True:
            return 'Super_only'
        else:
            return 'regular'
    data['seller_tier'] = data.apply(assign_seller_tier, axis=1)

    # Recalculate operational_score
    data['operational_score'] = (data['seller_ship_on_time'] + data['seller_chat_response']) / 2.0
    return data

# --- Streamlit App Layout ---
st.title("Lazada Product Analytics Dashboard")
st.write("This application demonstrates the Sales Prediction and Product Segmentation Models.")

# Sidebar for navigation or global settings
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Sales Prediction", "Product Segmentation"])


# --- Sales Prediction Page ---
if page == "Sales Prediction":
    st.header("Predict Number of Products Sold")
    st.write("Enter product details to predict the number of units that will be sold.")

    with st.form("sales_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            reviews = st.number_input("Number of Product Reviews", min_value=0, value=100)
            discount_pct = st.slider("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            rating = st.slider("Product Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
            final_price = st.number_input("Final Product Price (IDR)", min_value=0.0, value=150000.0, step=1000.0)
        with col2:
            seller_ship_on_time = st.slider("Seller Ship On Time (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
            seller_chat_response = st.slider("Seller Chat Response (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
            is_super_seller = st.checkbox("Is Super Seller?", value=False)
            lazmall = st.checkbox("Is LazMall?", value=False)
            color_variant_count = st.number_input("Number of Color Variants", min_value=1, value=1)

        submitted = st.form_submit_button("Predict")

        if submitted:
            # Create DataFrame for prediction
            input_df = pd.DataFrame({
                'reviews': [reviews],
                'discount_pct': [discount_pct],
                'rating': [rating],
                'final_price': [final_price],
                'seller_ship_on_time': [seller_ship_on_time],
                'seller_chat_response': [seller_chat_response],
                'is_super_seller': [is_super_seller],
                'lazmall': [lazmall],
                'color_variant_count': [color_variant_count]
            })

            # Apply feature engineering
            processed_input_df = create_features_for_sales_prediction(input_df)

            # Select features for the model (matching X_train columns)
            # model_input_features = [col for col in sales_pipeline.named_steps['preprocesssor'].named_transformers_['num'].feature_names_in_] + \
            #                        [col for col in sales_pipeline.named_steps['preprocesssor'].named_transformers_['cat'].get_feature_names_out(sales_pipeline.named_steps['preprocesssor'].named_transformers_['cat'].feature_names_in_)]

            # Adjust for OneHotEncoder output if needed, assuming the pipeline handles it.
            # For simplicity, we directly use the original feature names here, assuming pipeline handles OHE internally.
            # X_predict = processed_input_df[sales_pipeline.named_steps['regressor'].feature_names_in_ if hasattr(sales_pipeline.named_steps['regressor'], 'feature_names_in_') else sales_pipeline.named_steps['preprocesssor'].get_feature_names_out()]

            try:
                # Ensure the order of columns matches the training data
                # This is a critical step as pipelines expect features in a specific order.
                # The feature_names_in_ attribute on the regressor step (XGBoost) might not be directly available or representative
                # of the *original* features before preprocessing. Instead, we should pass the DataFrame
                # to the full pipeline, and it will handle the preprocessing.

                prediction_log = sales_pipeline.predict(processed_input_df)
                predicted_sales = np.expm1(prediction_log)[0]
                st.success(f"Predicted Number of Products Sold: **{predicted_sales:.0f} units**")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.write("Please ensure all inputs are correct and the model has been loaded properly.")


# --- Product Segmentation Page ---
elif page == "Product Segmentation":
    st.header("Product Segmentation")
    st.write("Enter product metrics to identify its product segment.")

    with st.form("segmentation_form"):
        number_sold_seg = st.number_input("Number of Products Sold", min_value=0, value=500)
        final_price_seg = st.number_input("Final Product Price (IDR)", min_value=0.0, value=200000.0, step=1000.0)
        reviews_seg = st.number_input("Number of Product Reviews", min_value=0, value=50)

        submitted_seg = st.form_submit_button("Segment Product")

        if submitted_seg:
            # Calculate GMV for segmentation
            gmv_seg = number_sold_seg * final_price_seg

            input_df_seg = pd.DataFrame({
                'number_sold': [number_sold_seg],
                'gmv': [gmv_seg],
                'final_price': [final_price_seg],
                'reviews': [reviews_seg]
            })

            # Log transform and scale
            X_log_seg = np.log1p(input_df_seg[kmeans_features])
            X_scaled_seg = kmeans_scaler.transform(X_log_seg)

            # Predict cluster
            raw_cluster_id = kmeans_model.predict(X_scaled_seg)[0]
            mapped_cluster_id = segmentation_bundle['cluster_mapping'][raw_cluster_id]
            product_segment = segment_names_map[mapped_cluster_id]

            st.success(f"This product belongs to the segment: **{product_segment}**")
