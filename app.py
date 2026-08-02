import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Lazada Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# --- 1. Load Models and Preprocessors ---
@st.cache_resource
def load_sales_model():
    try:
        return joblib.load('model/xgb_sales_pipeline_full.joblib')
    except Exception as e:
        st.error(f"Failed to load the sales prediction model: {e}")
        return None

@st.cache_resource
def load_segmentation_model_bundle():
    try:
        return joblib.load('model/kmeans_lazada_segmentation.joblib')
    except Exception as e:
        st.error(f"Failed to load the segmentation model bundle: {e}")
        return None

sales_pipeline = load_sales_model()
segmentation_bundle = load_segmentation_model_bundle()

if segmentation_bundle:
    kmeans_model = segmentation_bundle['model']
    kmeans_scaler = segmentation_bundle['scaler']
    kmeans_features = segmentation_bundle['features']
    segment_names_map = segmentation_bundle['segment_names']
    cluster_mapping = segmentation_bundle.get('cluster_mapping', {})


# --- 2. Feature Engineering Functions ---
def create_features_for_sales_prediction(df_input):
    data = df_input.copy()

    # Recalculate reviews_per_price
    data['reviews_per_price'] = data['reviews'] / (data['final_price'] + 1)

    # Recalculate seller_tier 
    def assign_seller_tier(row):
        if row['lazmall'] and row['is_super_seller']:
            return 'LazMall_Super'
        elif row['lazmall']:
            return 'LazMall_only'
        elif row['is_super_seller']:
            return 'Super_only'
        else:
            return 'regular'
            
    data['seller_tier'] = data.apply(assign_seller_tier, axis=1)

    # Recalculate operational_score
    data['operational_score'] = (data['seller_ship_on_time'] + data['seller_chat_response']) / 2.0
    return data


# --- Streamlit App Layout ---
st.title("Lazada Product Analytics Dashboard")
st.write("This application predicts estimated product sales volume and segments the catalog using Machine Learning.")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Analytics Module:", ["Sales Prediction", "Product Segmentation"])


# --- Page 1: Sales Prediction ---
if page == "Sales Prediction":
    st.header("Sales Volume Prediction")
    st.write("Enter product specifications and store performance metrics to predict the estimated number of units sold.")

    with st.form("sales_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Attributes")
            reviews = st.number_input("Number of Product Reviews", min_value=0, value=100, step=1)
            discount_pct = st.slider("Discount Percentage (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            rating = st.slider("Product Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
            final_price = st.number_input("Final Product Price (IDR)", min_value=0.0, value=150000.0, step=5000.0)
            color_variant_count = st.number_input("Number of Color Variants", min_value=1, value=1, step=1)

        with col2:
            st.subheader("Seller Attributes (Operational)")
            seller_ship_on_time = st.slider("Seller Ship On Time (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
            seller_chat_response = st.slider("Seller Chat Response (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
            
            st.write("---")
            is_super_seller = st.checkbox("Is Super Seller?", value=False)
            lazmall = st.checkbox("Is LazMall?", value=False)

        submitted = st.form_submit_button("Calculate Sales Prediction")

    if submitted:
        if sales_pipeline is None:
            st.error("Sales Pipeline model not found. Please ensure the `.joblib` file is available in the `model/` directory.")
        else:
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

            try:
                prediction_log = sales_pipeline.predict(processed_input_df)
                # Reverse log transformation (log1p -> expm1)
                predicted_sales = np.expm1(prediction_log)[0]
                
                # Format visual output
                st.markdown("### Prediction Results")
                st.metric(
                    label="Estimated Total Product Sales", 
                    value=f"{max(0, int(round(predicted_sales))):,} Units"
                )
                
            except Exception as e:
                st.error(f"An error occurred during prediction processing: {e}")


# --- Page 2: Product Segmentation ---
elif page == "Product Segmentation":
    st.header("Product Clustering & Segmentation")
    st.write("Identify the product's segment based on its historical sales performance.")

    with st.form("segmentation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            number_sold_seg = st.number_input("Number of Products Sold", min_value=0, value=500, step=10)
            final_price_seg = st.number_input("Final Product Price (IDR)", min_value=0.0, value=200000.0, step=5000.0)
            
        with col2:
            reviews_seg = st.number_input("Number of Product Reviews", min_value=0, value=50, step=1)

        submitted_seg = st.form_submit_button("Analyze Segmentation")

    if submitted_seg:
        if segmentation_bundle is None:
            st.error("Segmentation model bundle not found. Please ensure the `.joblib` file is available in the `model/` directory.")
        else:
            # Calculate GMV for segmentation
            gmv_seg = number_sold_seg * final_price_seg

            input_df_seg = pd.DataFrame({
                'number_sold': [number_sold_seg],
                'gmv': [gmv_seg],
                'final_price': [final_price_seg],
                'reviews': [reviews_seg]
            })

            try:
                # Transform using log1p & standard scaler matching the training set pipeline
                X_log_seg = np.log1p(input_df_seg[kmeans_features])
                X_scaled_seg = kmeans_scaler.transform(X_log_seg)

                # Predict cluster & map cluster labels
                raw_cluster_id = kmeans_model.predict(X_scaled_seg)[0]
                mapped_cluster_id = cluster_mapping.get(raw_cluster_id, raw_cluster_id)
                product_segment = segment_names_map.get(mapped_cluster_id, f"Cluster {mapped_cluster_id}")

                st.markdown("### Segmentation Results")
                st.info(f"This product belongs to the segment: **{product_segment}**")
                
                # Display calculated GMV metric
                st.metric("Calculated GMV (Gross Merchandise Value)", f"IDR {gmv_seg:,.0f}")
                
            except Exception as e:
                st.error(f"An error occurred during segmentation processing: {e}")
