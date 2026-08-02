import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Lazada Product Analysis App")

# --- Contextual Introduction ---
st.markdown("# Lazada Product Performance & Segmentation App")
st.markdown("This application helps Lazada sellers and strategists maximize Gross Merchandise Value (GMV) and optimize inventory by providing sales predictions and product segmentation with actionable strategies.")
st.markdown("---")

# --- Helper function for feature engineering (must match training) ---
def create_features_for_prediction(data_dict):
    data = pd.DataFrame([data_dict])

    # Feature 1: Social proof per price (reviews_per_price)
    # Add 1 to price to avoid division by zero if price is 0
    data['reviews_per_price'] = data['reviews'] / (data['final_price'] + 1)

    # Feature 2: Combined seller tier status
    def assign_seller_tier(row):
        lazmall = row.get('lazmall', False)
        is_super_seller = row.get('is_super_seller', False)
        if lazmall and is_super_seller:
            return 'LazMall_Super'
        elif lazmall:
            return 'LazMall_only'  # Typo fixed here
        elif is_super_seller:
            return 'Super_only'
        else:
            return 'regular'

    data['seller_tier'] = data.apply(assign_seller_tier, axis=1)

    # Operational score (assuming seller_ship_on_time and seller_chat_response are 0-100% values)
    data['operational_score'] = (data['seller_ship_on_time'] + data['seller_chat_response']) / 2.0

    return data


# --- Load Models ---
@st.cache_resource
def load_sales_prediction_model():
    try:
        # Added 'model/' path directory
        model = joblib.load('model/xgb_sales_pipeline_full.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading Sales Prediction Model: {e}")
        return None

@st.cache_resource
def load_segmentation_model():
    try:
        # Added 'model/' path directory
        bundle = joblib.load('model/kmeans_lazada_segmentation.joblib')
        return bundle
    except Exception as e:
        st.error(f"Error loading Product Segmentation Model: {e}")
        return None

sales_model = load_sales_prediction_model()
segmentation_bundle = load_segmentation_model()

# Extract components from segmentation bundle
if segmentation_bundle:
    kmeans_model = segmentation_bundle['model']
    scaler_segmentation = segmentation_bundle['scaler']
    segmentation_features = segmentation_bundle['features']
    cluster_mapping = segmentation_bundle['cluster_mapping']
    segment_names = segmentation_bundle['segment_names']
else:
    kmeans_model = None
    scaler_segmentation = None
    segmentation_features = None
    cluster_mapping = None
    segment_names = None


# --- Layout: Two Columns for Prediction and Segmentation Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.header("Sales Prediction Engine")
    st.markdown("Enter product details to predict potential sales volume.")

    with st.form("sales_prediction_form"):
        st.subheader("Product Attributes")
        sp_final_price = st.number_input("Final Price (IDR)", min_value=0.0, value=100000.0, step=1000.0)
        sp_reviews = st.number_input("Number of Reviews", min_value=0, value=50, step=1)
        sp_rating = st.slider("Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
        sp_discount_pct = st.slider("Discount Percentage", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        sp_color_variant_count = st.number_input("Color Variant Count", min_value=1, value=1, step=1)

        st.subheader("Seller Performance")
        sp_seller_ratings = st.slider("Seller Rating (0-1)", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
        sp_seller_ship_on_time = st.slider("Seller Ship-On-Time (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
        sp_seller_chat_response = st.slider("Seller Chat Response (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
        sp_lazmall = st.checkbox("Is LazMall Seller?")
        sp_is_super_seller = st.checkbox("Is Super Seller?")

        predict_button = st.form_submit_button("Predict Sales")

        if predict_button and sales_model:
            input_data_dict = {
                'reviews': sp_reviews,
                'discount_pct': sp_discount_pct,
                'rating': sp_rating,
                'final_price': sp_final_price,
                'seller_ratings': sp_seller_ratings,
                'seller_ship_on_time': sp_seller_ship_on_time,
                'seller_chat_response': sp_seller_chat_response,
                'lazmall': sp_lazmall,
                'is_super_seller': sp_is_super_seller,
                'color_variant_count': sp_color_variant_count
            }
            try:
                processed_input = create_features_for_prediction(input_data_dict)

                model_features = [
                    'reviews',
                    'discount_pct',
                    'rating',
                    'final_price',
                    'reviews_per_price',
                    'operational_score',
                    'seller_tier',
                    'color_variant_count'
                ]
                
                for feature in model_features:
                    if feature not in processed_input.columns:
                        processed_input[feature] = 0 

                prediction_input = processed_input[model_features]

                # Make prediction
                log_prediction = sales_model.predict(prediction_input)[0]

                # Reverse transformation
                predicted_sales = np.expm1(log_prediction)

                st.metric(label="Predicted Sales (Units)", value=f"{int(round(predicted_sales)):,}")
                st.info("💡 **Note:** This model does not provide a time-series forecast (e.g., sales per month). Instead, it acts as a static evaluation tool to measure a product's potential to become a best-seller based on its current pricing strategy, social proof, and operational ecosystem badges.")

            except Exception as e:
                st.error(f"An error occurred during sales prediction: {e}")
        elif predict_button:
            st.warning("Sales prediction model not loaded. Please check the backend.")

with col2:
    st.header("🔍 Product Segmentation")
    st.markdown("Categorize products to prescribe operational strategies.")

    with st.form("segmentation_form"):
        st.subheader("Product Performance Inputs")
        seg_number_sold = st.number_input("Number Sold (Units)", min_value=0, value=100, step=1)
        seg_final_price = st.number_input("Final Price (IDR)", min_value=0.0, value=50000.0, step=1000.0)
        seg_reviews = st.number_input("Number of Reviews", min_value=0, value=20, step=1)

        segment_button = st.form_submit_button("Segment Product")

        if segment_button and kmeans_model and scaler_segmentation and segmentation_features:
            try:
                # Dynamic GMV Calculation
                seg_gmv = seg_number_sold * seg_final_price

                segment_input_data = pd.DataFrame([{
                    'number_sold': seg_number_sold,
                    'gmv': seg_gmv,
                    'final_price': seg_final_price,
                    'reviews': seg_reviews
                }])

                # Log transform and scale
                X_log_segment = np.log1p(segment_input_data[segmentation_features])
                X_scaled_segment = scaler_segmentation.transform(X_log_segment)

                # Predict raw cluster ID
                raw_cluster_id = kmeans_model.predict(X_scaled_segment)[0]

                # Map to sorted cluster ID and then to segment name
                sorted_cluster_id = cluster_mapping.get(raw_cluster_id, raw_cluster_id) 
                product_segment = segment_names.get(sorted_cluster_id, "Unknown Segment")

                st.subheader("Product Segment")
                if "Tier 1" in product_segment or "Tier 2" in product_segment:
                    st.success(product_segment)
                elif "Tier 3" in product_segment:
                    st.info(product_segment)
                elif "Tier 4" in product_segment:
                    st.warning(product_segment)
                elif "Tier 5" in product_segment:
                    st.error(product_segment)
                else:
                    st.write(product_segment)
                
                # Added GMV display
                st.metric("Calculated GMV", f"IDR {seg_gmv:,.0f}")

                st.subheader("Actionable Recommendations")
                # Restored rich text markdown and emojis
                recommendations = {
                    'Tier 1 (Star Products)': "🌟 **VIP Treatment:** Secure supply chain and FBL. Allocate prime homepage real estate and LazMall banners.",
                    'Tier 2 (High Potential)': "🚀 **Conversion Nudging:** Deploy targeted flash sales or subsidized vouchers to break the conversion bottleneck.",
                    'Tier 3 (Mid-Range)': "🛒 **Boost AOV:** Implement 'Buy 2 Get 1 Free' or minimum-spend free shipping thresholds to increase basket size.",
                    'Tier 4 (Slow Movers)': "⚠️ **Algorithm Demotion:** Deprioritize in search. Advise sellers to drop prices to clearance levels to improve conversion rates.",
                    'Tier 5 (Low Potential)': "🛑 **Stop the Bleed:** Revoke subsidized marketing. Force liquidation or return inventory to free up warehouse capacity."
                }
                st.markdown(recommendations.get(product_segment, "No specific recommendations for this segment."))

            except Exception as e:
                st.error(f"An error occurred during product segmentation: {e}")
        elif segment_button:
            st.warning("Product segmentation model not loaded. Please check the backend.")

st.markdown("---")
st.markdown("**Overall App Architecture Notes:**\n" +
            "- Page configuration is set to `wide` for a better UI experience.\n" +
            "- Model loading and prediction/segmentation logic are wrapped in `try-except` blocks for error handling.\n" +
            "- Contextual introduction is provided on the main page.")
