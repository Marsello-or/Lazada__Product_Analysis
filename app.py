import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="Lazada Analytics Dashboard", page_icon="📊")

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
            return 'LazMall_only'
        elif is_super_seller:
            return 'Super_only'
        else:
            return 'regular'

    data['seller_tier'] = data.apply(assign_seller_tier, axis=1)

    # Operational score
    data['operational_score'] = (data['seller_ship_on_time'] + data['seller_chat_response']) / 2.0

    return data


# --- Load Models ---
@st.cache_resource
def load_sales_prediction_model():
    try:
        model = joblib.load('model/xgb_sales_pipeline_full.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading Sales Prediction Model: {e}")
        return None

@st.cache_resource
def load_segmentation_model():
    try:
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


# --- Sidebar Navigation ---
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRuX7zq4jb2PTq70xxEdVE1B4eX38g5XKNnxs9z2Rayoxa2AZrKGDOq_ePP&s=10", width=250)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:", ["Dashboard Overview", "Sales Prediction", "Product Segmentation"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Objective:**\nMaximize GMV and optimize inventory through data-driven operational strategies.")


# --- Page 0: Dashboard Overview ---
if page == "Dashboard Overview":
    st.title("Lazada E-Commerce Intelligence Dashboard")
    st.markdown("---")
    st.markdown("Welcome to the **Lazada Product Performance & Segmentation App**.")
    st.markdown("""
    This platform acts as the final deployment phase of an end-to-end data analytics project, designed to empower e-commerce strategists and sellers. 
    
    ### Modules Available:
    *   **Sales Prediction Engine:** Evaluate a product's potential to become a best-seller based on its current pricing strategy, social proof, and operational ecosystem badges.
    *   **Product Segmentation:** Categorize products into business tiers using machine learning, automatically prescribing actionable strategies to optimize inventory and boost Gross Merchandise Value (GMV).
    
    **Please select a module from the sidebar to begin your analysis.**
    """)


# --- Page 1: Sales Prediction Engine ---
elif page == "Sales Prediction":
    st.title("Sales Prediction Engine")
    st.markdown("Enter product details below to measure its potential sales volume.")
    st.markdown("---")

    with st.form("sales_prediction_form"):
        # Utilizing columns inside the form for a cleaner, wider UI
        col_prod, col_seller = st.columns(2)
        
        with col_prod:
            st.subheader("Product Attributes")
            sp_final_price = st.number_input("Final Price (IDR)", min_value=0.0, value=100000.0, step=1000.0)
            sp_reviews = st.number_input("Number of Reviews", min_value=0, value=50, step=1)
            sp_rating = st.slider("Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
            sp_discount_pct = st.slider("Discount Percentage", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            sp_color_variant_count = st.number_input("Color Variant Count", min_value=1, value=1, step=1)

        with col_seller:
            st.subheader("Seller Performance")
            sp_seller_ratings = st.slider("Seller Rating (0-1)", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
            sp_seller_ship_on_time = st.slider("Seller Ship-On-Time (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
            sp_seller_chat_response = st.slider("Seller Chat Response (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
            st.markdown("<br>", unsafe_allow_html=True) # spacer
            sp_lazmall = st.checkbox("Is LazMall Seller?")
            sp_is_super_seller = st.checkbox("Is Super Seller?")

        st.markdown("---")
        predict_button = st.form_submit_button("Predict Sales Volume", use_container_width=True)

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
                'reviews', 'discount_pct', 'rating', 'final_price', 
                'reviews_per_price', 'operational_score', 'seller_tier', 'color_variant_count'
            ]
            
            for feature in model_features:
                if feature not in processed_input.columns:
                    processed_input[feature] = 0 

            prediction_input = processed_input[model_features]
            log_prediction = sales_model.predict(prediction_input)[0]
            predicted_sales = np.expm1(log_prediction)

            st.success("Analysis Complete!")
            st.metric(label="Estimated Sales Volume", value=f"{int(round(predicted_sales)):,} Units")
            st.info("💡 **Note:** This model does not provide a time-series forecast (e.g., sales per month). Instead, it acts as a static evaluation tool to measure a product's potential to become a best-seller based on its current pricing strategy, social proof, and operational ecosystem badges.")

        except Exception as e:
            st.error(f"An error occurred during sales prediction: {e}")
    elif predict_button:
        st.warning("Sales prediction model not loaded. Please check the backend.")


# --- Page 2: Product Segmentation ---
elif page == "Product Segmentation":
    st.title("Product Segmentation Analyzer")
    st.markdown("Categorize products based on historical performance to prescribe operational strategies.")
    st.markdown("---")

    with st.form("segmentation_form"):
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.subheader("Financial Metrics")
            seg_number_sold = st.number_input("Number Sold (Units)", min_value=0, value=100, step=1)
            seg_final_price = st.number_input("Final Price (IDR)", min_value=0.0, value=50000.0, step=1000.0)
            
        with col_metrics2:
            st.subheader("Engagement Metrics")
            seg_reviews = st.number_input("Number of Reviews", min_value=0, value=20, step=1)

        st.markdown("---")
        segment_button = st.form_submit_button("Analyze & Segment Product", use_container_width=True)

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
            sorted_cluster_id = cluster_mapping.get(raw_cluster_id, raw_cluster_id) 
            product_segment = segment_names.get(sorted_cluster_id, "Unknown Segment")

            st.markdown("### Segmentation Results")
            
            # Colored feedback block
            if "Tier 1" in product_segment or "Tier 2" in product_segment:
                st.success(f"**Identified Category:** {product_segment}")
            elif "Tier 3" in product_segment:
                st.info(f"**Identified Category:** {product_segment}")
            elif "Tier 4" in product_segment:
                st.warning(f"**Identified Category:** {product_segment}")
            elif "Tier 5" in product_segment:
                st.error(f"**Identified Category:** {product_segment}")
            else:
                st.write(f"**Identified Category:** {product_segment}")
            
            # GMV Display
            st.metric("Calculated GMV (Gross Merchandise Value)", f"IDR {seg_gmv:,.0f}")

            # Recommendations
            st.markdown("### Actionable Recommendations")
            recommendations = {
                'Tier 1 (Star Products)': "**VIP Treatment:** Secure supply chain and FBL. Allocate prime homepage real estate and LazMall banners.",
                'Tier 2 (High Potential)': "**Conversion Nudging:** Deploy targeted flash sales or subsidized vouchers to break the conversion bottleneck.",
                'Tier 3 (Mid-Range)': "**Boost AOV:** Implement 'Buy 2 Get 1 Free' or minimum-spend free shipping thresholds to increase basket size.",
                'Tier 4 (Slow Movers)': "**Algorithm Demotion:** Deprioritize in search. Advise sellers to drop prices to clearance levels to improve conversion rates.",
                'Tier 5 (Low Potential)': "**Stop the Bleed:** Revoke subsidized marketing. Force liquidation or return inventory to free up warehouse capacity."
            }
            
            st.write(recommendations.get(product_segment, "No specific recommendations for this segment."))

        except Exception as e:
            st.error(f"An error occurred during product segmentation: {e}")
    elif segment_button:
        st.warning("Product segmentation model not loaded. Please check the backend.")
