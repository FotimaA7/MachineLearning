# app.py - Bulgaria Rain Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="🌧️ Bulgaria Rain Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #1a1a1a;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .main-header .subtitle {
        opacity: 0.95;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .prediction-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .prediction-box.rain {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .prediction-box.no-rain {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .prediction-box h2 {
        font-size: 2rem;
        margin: 0 0 0.5rem 0;
        font-weight: 800;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .small-muted {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
    }
    
    .footer {
        color: #999;
        font-size: 0.85rem;
        margin-top: 2rem;
        text-align: center;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
    }
    
    hr {
        margin: 2rem 0;
        opacity: 0.1;
    }
    </style>
""", unsafe_allow_html=True)

# Get the directory where the app is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, 'models')

# Check if model files exist
def check_files():
    """Check if all required model files exist"""
    required_files = [
        os.path.join(MODELS_DIR, 'model.pkl'),
        os.path.join(MODELS_DIR, 'scaler.pkl'),
        os.path.join(MODELS_DIR, 'features.pkl'),
        os.path.join(MODELS_DIR, 'metrics.pkl')
    ]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(os.path.basename(file))
    
    return missing_files

# Load model function
@st.cache_resource
def load_model():
    """Load all saved model components"""
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODELS_DIR, 'features.pkl'))
        metrics = joblib.load(os.path.join(MODELS_DIR, 'metrics.pkl'))
        return model, scaler, features, metrics
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None, None

def create_gauge_chart(probability, title, color='#667eea'):
    """Create an enhanced gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': title, 'font': {'size': 20}},
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.7},
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, 33], 'color': '#ffebee'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'family': "Poppins"}
    )
    return fig

def create_weather_summary(temp, humidity, wind_kph, cloud, wind_degree):
    """Create a visual summary of weather parameters"""
    summary = f"""
    <div class="info-box">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;">
            <div>🌡️ <strong>Temperature:</strong> {temp}°C</div>
            <div>💧 <strong>Humidity:</strong> {humidity}%</div>
            <div>💨 <strong>Wind:</strong> {wind_kph} km/h</div>
            <div>☁️ <strong>Cloud:</strong> {cloud}%</div>
            <div>🧭 <strong>Direction:</strong> {wind_degree}°</div>
        </div>
    </div>
    """
    return summary

def run_batch_predictions(batch_df, model, scaler, features):
    """Run batch predictions and return results with proper percentage formatting"""
    try:
        if not set(features).issubset(set(batch_df.columns)):
            return None, f"CSV missing required columns. Expected: {', '.join(features)}"
        
        # Make predictions
        Xb = scaler.transform(batch_df[features])
        preds_proba = model.predict_proba(Xb)
        preds_class = model.predict(Xb)
        
        # Create a copy to add predictions
        results_df = batch_df.copy()
        
        # Add prediction columns with proper formatting
        results_df['no_rain_probability_%'] = (preds_proba[:, 0] * 100).round(2)
        results_df['rain_probability_%'] = (preds_proba[:, 1] * 100).round(2)
        results_df['prediction'] = preds_class
        results_df['prediction_label'] = results_df['prediction'].map({0: '☀️ No Rain', 1: '🌧️ Rain'})
        
        # Add confidence level
        max_prob = np.maximum(preds_proba[:, 0], preds_proba[:, 1])
        results_df['confidence_%'] = (max_prob * 100).round(2)
        
        # Add confidence category
        def get_confidence_category(conf):
            if conf >= 90:
                return 'Very High'
            elif conf >= 75:
                return 'High'
            elif conf >= 60:
                return 'Medium'
            else:
                return 'Low'
        
        results_df['confidence_level'] = results_df['confidence_%'].apply(get_confidence_category)
        
        return results_df, None
    except Exception as e:
        return None, str(e)

def main():
    # Main Header
    st.markdown("""
        <div class="main-header">
            <h1>🌧️ Bulgaria Rain Predictor</h1>
            <div class="subtitle">Advanced Weather Prediction using Naive Bayes Machine Learning</div>
        </div>
    """, unsafe_allow_html=True)

    # Check for model files
    missing_files = check_files()
    if missing_files:
        st.error("❌ Missing model files!")
        st.write("**Missing files:**")
        for file in missing_files:
            st.write(f"  • `{file}`")
        
        st.warning("""
        **Quick Fix:** Run the save cell in your `weather.ipynb` notebook to generate the model files.
        """)
        st.stop()

    # Load model
    model, scaler, features, metrics = load_model()
    
    if model is None:
        st.stop()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Settings & Tools")
        
        # Preset Weather Conditions
        st.markdown("### 🔄 Quick Weather Presets")
        preset_option = st.selectbox(
            "Select a preset:",
            ["Custom", "☀️ Typical Summer", "⛈️ Stormy", "❄️ Dry Cold"]
        )
        
        preset_data = {
            "☀️ Typical Summer": {'temp': 25.0, 'humidity': 70, 'wind': 10.0, 'cloud': 40, 'wind_deg': 150},
            "⛈️ Stormy": {'temp': 18.0, 'humidity': 92, 'wind': 28.0, 'cloud': 95, 'wind_deg': 230},
            "❄️ Dry Cold": {'temp': -2.0, 'humidity': 30, 'wind': 6.0, 'cloud': 10, 'wind_deg': 60},
        }
        
        if preset_option in preset_data:
            if st.button("✅ Apply Preset", use_container_width=True):
                st.session_state['preset'] = preset_data[preset_option]
                st.success(f"Preset applied: {preset_option}")
        
        st.divider()
        
        # Batch Predictions
        st.markdown("### 📊 Batch Predictions")
        st.write("Upload a CSV file with weather data to make bulk predictions.")
        
        upload = st.file_uploader("📁 Choose CSV file", type=['csv'], label_visibility="collapsed")
        if upload is not None:
            try:
                batch_df = pd.read_csv(upload)
                st.success(f"✅ Loaded {len(batch_df)} records")
                
                # Show preview
                with st.expander("👁️ Preview data"):
                    st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🚀 Run Predictions", use_container_width=True):
                    results_df, error = run_batch_predictions(batch_df, model, scaler, features)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success(f"✅ Predictions complete for {len(results_df)} records!")
                        
                        # Show statistics
                        pred_counts = results_df['prediction_label'].value_counts()
                        col1, col2 = st.columns(2)
                        with col1:
                            rain_count = pred_counts.get('🌧️ Rain', 0)
                            st.metric("🌧️ Rain Predictions", rain_count)
                        with col2:
                            no_rain_count = pred_counts.get('☀️ No Rain', 0)
                            st.metric("☀️ No Rain Predictions", no_rain_count)
                        
                        # Show average probabilities
                        st.markdown("### 📊 Prediction Statistics")
                        avg_rain_prob = results_df['rain_probability_%'].mean()
                        avg_confidence = results_df['confidence_%'].mean()
                        
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            st.metric("Avg Rain Probability", f"{avg_rain_prob:.2f}%")
                        with stat_col2:
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                        with stat_col3:
                            st.metric("Total Records", len(results_df))
                        
                        # Show confidence distribution
                        confidence_dist = results_df['confidence_level'].value_counts()
                        st.markdown("### 🎯 Confidence Distribution")
                        st.bar_chart(confidence_dist)
                        
                        # Download button
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv_data,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Show detailed results with formatting
                        with st.expander("📋 View Detailed Predictions"):
                            # Select columns to display
                            display_cols = features + ['rain_probability_%', 'no_rain_probability_%', 'prediction_label', 'confidence_%', 'confidence_level']
                            display_df = results_df[display_cols].copy()
                            
                            # Format the display
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                column_config={
                                    "rain_probability_%": st.column_config.NumberColumn(
                                        "🌧️ Rain %",
                                        format="%.2f%%"
                                    ),
                                    "no_rain_probability_%": st.column_config.NumberColumn(
                                        "☀️ No Rain %",
                                        format="%.2f%%"
                                    ),
                                    "confidence_%": st.column_config.NumberColumn(
                                        "Confidence %",
                                        format="%.2f%%"
                                    ),
                                }
                            )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Model Info
        st.markdown("### 📚 Model Information")
        if metrics:
            accuracy = metrics.get('accuracy', 0)
            auc = metrics.get('auc', 0)
            st.info(f"""
            **Model Performance:**
            - Accuracy: **{accuracy*100:.1f}%**
            - AUC Score: **{auc:.3f}**
            """)
        
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #999; font-size: 0.85rem;'>Built with ❤️ using Streamlit & Scikit-learn</div>", unsafe_allow_html=True)

    # Main Content - Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Make a Prediction")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🌦️ Weather Parameters</div>', unsafe_allow_html=True)
            
            with st.form("prediction_form"):
                # Get preset values
                p = st.session_state.get('preset', None)
                
                # Input sliders with better styling
                temp = st.slider(
                    "🌡️ Temperature (°C)",
                    min_value=-30.0,
                    max_value=50.0,
                    value=p['temp'] if p else 18.5,
                    step=0.5,
                    help='Air temperature in degrees Celsius'
                )
                
                humidity = st.slider(
                    "💧 Humidity (%)",
                    min_value=0,
                    max_value=100,
                    value=p['humidity'] if p else 72,
                    help='Relative humidity percentage'
                )
                
                wind_kph = st.slider(
                    "💨 Wind Speed (km/h)",
                    min_value=0.0,
                    max_value=100.0,
                    value=p['wind'] if p else 15.0,
                    step=0.5,
                    help='Wind speed in kilometers per hour'
                )
                
                cloud = st.slider(
                    "☁️ Cloud Cover (%)",
                    min_value=0,
                    max_value=100,
                    value=p['cloud'] if p else 65,
                    help='Cloud coverage percentage'
                )
                
                wind_degree = st.slider(
                    "🧭 Wind Direction (°)",
                    min_value=0,
                    max_value=360,
                    value=p['wind_deg'] if p else 120,
                    help='Wind direction in degrees (0°=North, 90°=East, 180°=South, 270°=West)'
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🔮 Make Prediction", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-title">📈 Prediction Results</div>', unsafe_allow_html=True)
            
            if submitted:
                # Prepare input data
                input_data = pd.DataFrame([{
                    'temperature_celsius': temp,
                    'humidity': humidity,
                    'wind_kph': wind_kph,
                    'cloud': cloud,
                    'wind_degree': wind_degree
                }])

                # Scale and predict
                input_scaled = scaler.transform(input_data[features])
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0]
                confidence = max(probability) * 100

                # Display main prediction
                if prediction == 1:
                    st.markdown("""
                        <div class="prediction-box rain">
                            <h2>🌧️ RAIN EXPECTED</h2>
                            <p class="small-muted">Get your umbrella ready!</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-box no-rain">
                            <h2>☀️ NO RAIN EXPECTED</h2>
                            <p class="small-muted">Beautiful weather ahead!</p>
                        </div>
                    """, unsafe_allow_html=True)

                # Show confidence
                st.markdown(f"### 📊 Confidence Level: **{confidence:.1f}%**")
                st.progress(confidence / 100, text=f"{confidence:.1f}%")
                
                # Probability visualization
                st.markdown("### 📉 Probability Breakdown")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(create_gauge_chart(probability[1], "Rain Probability", '#667eea'), use_container_width=True)
                with col_b:
                    st.plotly_chart(create_gauge_chart(probability[0], "No Rain Probability", '#764ba2'), use_container_width=True)

                # Weather summary
                st.markdown("### 🌡️ Input Summary")
                st.markdown(create_weather_summary(temp, humidity, wind_kph, cloud, wind_degree), unsafe_allow_html=True)

                # Save to history
                history = st.session_state.get('history', [])
                rec = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'temperature': temp,
                    'humidity': humidity,
                    'wind_kph': wind_kph,
                    'cloud': cloud,
                    'wind_degree': wind_degree,
                    'rain_probability': float(probability[1]),
                    'no_rain_probability': float(probability[0]),
                    'prediction': 'Rain' if prediction == 1 else 'No Rain'
                }
                history.insert(0, rec)
                st.session_state['history'] = history[:50]

                # Export options
                st.markdown("### 💾 Export Results")
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.download_button(
                        "📥 Download as JSON",
                        data=pd.DataFrame([rec]).to_json(orient='records'),
                        file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with col_exp2:
                    st.download_button(
                        "📥 Download as CSV",
                        data=pd.DataFrame([rec]).to_csv(index=False),
                        file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Show history
                with st.expander(f"📋 Prediction History ({len(st.session_state.get('history', []))})"):
                    history_df = pd.DataFrame(st.session_state.get('history', []))
                    st.dataframe(history_df, use_container_width=True)
                    
                    if st.button('🗑️ Clear History', use_container_width=True):
                        st.session_state['history'] = []
                        st.rerun()
            else:
                st.info("👆 Configure the weather parameters and click 'Make Prediction' to get started!")
    
    with tab2:
        st.markdown("### 📊 Model Performance Metrics")
        
        if metrics:
            # Metrics cards
            metric_cols = st.columns(4)
            
            metrics_data = [
                ("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%", "🎯"),
                ("AUC Score", f"{metrics.get('auc', 0):.3f}", "📈"),
                ("Precision (Rain)", f"{metrics['precision'][1]:.3f}", "🎲"),
                ("Recall (Rain)", f"{metrics['recall'][1]:.3f}", "🔍"),
            ]
            
            for col, (label, value, emoji) in zip(metric_cols, metrics_data):
                with col:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{emoji} {label}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📚 Model Details")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Algorithm:** Gaussian Naive Bayes  
            **Training Approach:** Stratified 80-20 split  
            **Feature Scaling:** StandardScaler  
            **Features Used:** 5 weather parameters
            """)
        with col2:
            st.markdown(f"""
            **Features:**
            - Temperature (°C)
            - Humidity (%)
            - Wind Speed (km/h)
            - Cloud Cover (%)
            - Wind Direction (°)
            """)
    
    with tab3:
        st.markdown("### ℹ️ About This Application")
        
        st.markdown("""
        ## 🌧️ Bulgaria Rain Predictor
        
        This application uses machine learning to predict rainfall in Bulgaria based on weather conditions.
        It's powered by a **Gaussian Naive Bayes** classifier trained on historical weather data.
        
        ### 🎯 How It Works
        1. Input current or forecasted weather parameters
        2. The model analyzes weather patterns
        3. Get a probability-based rain prediction
        
        ### 📊 Features
        - **Single Predictions:** Get instant forecasts for specific weather conditions
        - **Batch Processing:** Upload CSV files for bulk predictions
        - **Preset Options:** Quick presets for common weather scenarios
        - **Prediction History:** Track all your predictions
        - **Export Capabilities:** Download results as JSON or CSV
        
        ### 🔧 Technical Stack
        - **Framework:** Streamlit
        - **ML Library:** Scikit-learn
        - **Visualization:** Plotly
        - **Data:** Pandas & NumPy
        
        ### 📝 Input Parameters
        - **Temperature:** Range from -30°C to 50°C
        - **Humidity:** 0-100%
        - **Wind Speed:** 0-100 km/h
        - **Cloud Cover:** 0-100%
        - **Wind Direction:** 0-360° (Compass direction)
        
        ### 💡 Tips
        - Use Quick Presets for common scenarios
        - Check Prediction History to see patterns
        - Upload multiple weather scenarios at once with Batch Predictions
        - Download results for further analysis
        
        ### ⚠️ Disclaimer
        This is a demonstration model. For actual weather forecasting, please refer to official 
        meteorological services and weather forecasts.
        """)
        
        st.markdown("---")
        st.markdown("<div class='footer'>Created with ❤️ • Last Updated: February 2026</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()