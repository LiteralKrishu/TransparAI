# TransparAI - Procurement Transparency Dashboard

![SFLC.in](https://img.shields.io/badge/SFLC.in-Hackathon-blue)
![Open Source](https://img.shields.io/badge/Open--Source-Yes-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## 🎯 Project Overview

**TransparAI** is an open-source AI-powered dashboard that analyzes public procurement data to expose bias, inefficiency, and anomalies. Built for the SFLC.in Hackathon under the theme **"Defending Digital Rights through Open Source Innovation."**

The dashboard provides real-time transparency into government contracting processes using machine learning to detect suspicious patterns and promote accountability.

## 🚀 Features

### 🔍 Core Dashboard Features

- **📊 At-a-Glance Summary**
  - Total contracts, value, and vendor metrics
  - Real-time procurement overview
  - Key performance indicators

- **🚨 AI Anomaly Detection**
  - Isolation Forest machine learning algorithm
  - Detection of suspicious contract values
  - Interactive scatter plots for visualization

- **🏢 Vendor Concentration Analysis**
  - Market distribution analysis
  - Top vendor share percentages
  - Contract value distribution charts

- **⏱️ Procurement Efficiency Tracking**
  - Time-to-award analysis
  - Processing delay identification
  - Monthly trend analysis

### 🎨 Design Features

- **SFLC.in Brand Compliance**: Official colors (#1486c9) and Noto Sans typography
- **Responsive Design**: Works on desktop and mobile devices
- **Accessibility**: WCAG AA compliant with proper contrast ratios
- **Real-time Data**: Integration with Government e-Marketplace (GeM) APIs

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Quick Start

1. **Clone or download the project**
```bash
# If using git
git clone <repository-url>
cd TransparAI-Dashboard

# Or simply extract the project folder
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

Open your browser and navigate to http://localhost:8501

### Automated Setup (Windows/PowerShell)

**Run the setup script**
```powershell
.\run_dashboard.ps1
```

Or use the batch file
```cmd
.\run_dashboard.bat
```

These scripts will automatically:
- Check for Python installation
- Create a virtual environment
- Install all dependencies
- Launch the dashboard

## 📊 Data Sources

### Primary Sources
- Government e-Marketplace (GeM) API - Real-time contract data
- Open Government Data Platform India - Supplementary datasets

### Sample Data
- Generated realistic Indian procurement data for demonstration
- Includes contracts from major vendors and ministries
- Anomalies artificially introduced for testing detection algorithms

## 🏗️ Project Structure

```text
TransparAI-Dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── run_dashboard.ps1     # PowerShell launch script
├── run_dashboard.bat     # Windows batch launch script
├── .gitignore           # Git ignore rules
├── config/
│   └── settings.json    # Application configuration
├── data/
│   ├── sample_procurement_data.csv    # Generated sample data
│   └── sample_procurement_data.json   # JSON format sample data
├── src/
│   └── data_generator.py # Sample data generation utility
├── utils/
│   └── helpers.py       # Data processing and API utilities
├── tests/
│   └── test_dashboard.py # Test cases
├── models/              # Machine learning models
└── assets/             # Static assets (images, etc.)
```

## 🔧 Configuration

### Data Source Selection
The dashboard supports multiple data sources accessible via the sidebar:
- Real-time API: Fetches live data from government portals
- Sample Data: Uses generated dataset for demonstration

### Customization
Edit `config/settings.json` to modify:
- API endpoints
- Anomaly detection parameters
- Styling and colors
- Analysis thresholds

```json
{
    "api_endpoints": {
        "gem": "https://api.gem.gov.in/public/api/v1/contracts",
        "ogd": "https://api.data.gov.in/resource/"
    },
    "anomaly_detection": {
        "contamination": 0.1,
        "random_state": 42
    },
    "styling": {
        "primary_color": "#1486c9",
        "font_family": "Noto Sans"
    }
}
```

## 🤖 Machine Learning Features

### Anomaly Detection
- Algorithm: Isolation Forest from scikit-learn
- Features: Contract value and processing time
- Output: Binary classification of anomalous contracts
- Visualization: Interactive scatter plots with hover details

### Statistical Analysis
- Vendor concentration metrics
- Processing time distributions
- Temporal trend analysis
- Category-wise breakdowns

## 🎨 Design System

### Typography
- Primary Font: Noto Sans (Google Fonts)
- Headings: 700 weight
- Subheadings: 500 weight
- Body Text: 400 weight

### Color Palette
- Primary: #1486c9 (SFLC.in Blue)
- Secondary: #ff6b6b (Anomaly highlights)
- Background: #f8f9fa (Light gray)
- Text: #333333 (Dark gray)

### Layout Principles
- Clean, minimal design
- Ample white space
- Consistent spacing
- Mobile-responsive grid

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python -m pytest tests/test_dashboard.py -v
```

Tests include:
- Data processing validation
- Anomaly detection accuracy
- Metric calculation correctness

## 🔄 Data Generation

Generate new sample data:
```bash
python src/data_generator.py
```

This creates:
- `data/sample_procurement_data.csv`
- `data/sample_procurement_data.json`

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
The application can be deployed on:
- Streamlit Cloud
- Heroku
- AWS/Azure/GCP
- Any Python-compatible hosting service

## 🤝 Contributing

This project is built for the SFLC.in Hackathon. We welcome contributions under the open-source initiative.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

Open Source - SFLC.in Hackathon Project

## 🙏 Acknowledgments

- SFLC.in for the hackathon platform and guidance
- Government of India for open data initiatives
- Open source community for tools and libraries
- Contributors who help improve transparency in public procurement

## 📞 Support

For technical issues or questions:
- Check the troubleshooting tips in the application
- Verify your Python environment meets requirements
- Ensure internet connectivity for real-time data features

---

Built with ❤️ for Digital Rights Transparency

TransparAI - Defending Digital Rights through Open Source Innovation