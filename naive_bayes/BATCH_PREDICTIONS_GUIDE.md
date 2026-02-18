# 🌧️ Bulgaria Rain Predictor - Batch Predictions Guide

## 📋 How to Use Batch Predictions

### Step 1: Prepare Your CSV File
Your CSV file must contain **exactly these columns** (in any order):
- `temperature_celsius` - Temperature in Celsius (-30 to 50)
- `humidity` - Humidity percentage (0-100)
- `wind_kph` - Wind speed in km/h (0-100)
- `cloud` - Cloud cover percentage (0-100)
- `wind_degree` - Wind direction in degrees (0-360)

### Step 2: Upload the File
1. Open the Streamlit app
2. Go to **Sidebar → Batch Predictions**
3. Click **"📁 Choose CSV file"**
4. Select your CSV file

### Step 3: Run Predictions
1. Preview the data (optional)
2. Click **"🚀 Run Predictions"**
3. Wait for processing

### Step 4: View & Download Results
The results will include:
- ☀️ **no_rain_probability_%** - Probability of NO rain (0-100%)
- 🌧️ **rain_probability_%** - Probability of RAIN (0-100%)
- **prediction_label** - Final prediction (☀️ No Rain or 🌧️ Rain)
- **confidence_%** - Confidence level of prediction (0-100%)
- **confidence_level** - Category (Very High, High, Medium, Low)

### 📊 Output Format

Your downloaded CSV will contain all original columns plus these prediction columns:

```
temperature_celsius,humidity,wind_kph,cloud,wind_degree,no_rain_probability_%,rain_probability_%,prediction,prediction_label,confidence_%,confidence_level
15.2,65,12.5,45,120,35.48,64.52,1,🌧️ Rain,64.52,Medium
18.5,72,15.0,65,120,45.23,54.77,1,🌧️ Rain,54.77,Medium
```

### 📝 Example CSV Format

```csv
temperature_celsius,humidity,wind_kph,cloud,wind_degree
15.2,65,12.5,45,120
18.5,72,15.0,65,120
12.3,58,8.2,30,200
22.1,45,20.0,80,250
```

### ✅ Valid Ranges

| Column | Min | Max | Unit |
|--------|-----|-----|------|
| temperature_celsius | -30 | 50 | °C |
| humidity | 0 | 100 | % |
| wind_kph | 0 | 100 | km/h |
| cloud | 0 | 100 | % |
| wind_degree | 0 | 360 | ° |

### ⚠️ Common Issues

**Issue: "CSV missing required columns"**
- Solution: Ensure your CSV has exactly these column names (case-sensitive):
  - temperature_celsius
  - humidity
  - wind_kph
  - cloud
  - wind_degree

**Issue: Values out of range**
- Solution: Check that all values are within the valid ranges above

**Issue: Predictions seem incorrect**
- Solution: Verify all weather values are realistic for the scenario

### 📥 Testing with Sample File

Use the included `sample_batch.csv` to test the batch predictions:
1. Upload `sample_batch.csv` 
2. Preview to verify format
3. Click "Run Predictions"
4. View and download results

### 🎯 Output Interpretation

**Confidence Levels:**
- ✅ **Very High (90-100%)** - Reliable prediction
- 📊 **High (75-89%)** - Good confidence
- ⚠️ **Medium (60-74%)** - Moderate confidence
- ❌ **Low (< 60%)** - Low confidence, consider uncertain

**Rain Probability:**
- 0-25% = Very likely NO rain ☀️
- 25-50% = Probably NO rain 🌤️
- 50-75% = Probably rain 🌦️
- 75-100% = Very likely rain 🌧️

### 💡 Tips for Best Results

1. **Use realistic weather values** - Values that don't match real conditions may give uncertain predictions
2. **Batch process multiple scenarios** - Upload 10-100 rows at once for efficiency
3. **Check data quality** - Ensure no missing values in your CSV
4. **Verify predictions** - Compare with actual weather patterns

### 📚 More Information

For single predictions and other features, use the **"🎯 Predict"** tab in the main interface.
