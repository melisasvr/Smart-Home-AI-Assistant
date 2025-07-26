# 🤖 Smart Home AI Assistant
An advanced AI-powered smart home assistant that learns from user behavior patterns, provides proactive suggestions, and manages smart devices through machine learning algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Web Dashboard](https://img.shields.io/badge/Dashboard-HTML%2FJS%2FCSS-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features
### 🧠 Advanced AI Capabilities
- **Machine Learning Pattern Recognition**: Uses Random Forest and Gradient Boosting algorithms
- **Behavioral Profiling**: K-means clustering for user behavior analysis  
- **Predictive Suggestions**: Context-aware proactive recommendations
- **Satisfaction-Based Learning**: Adapts based on user feedback ratings
- **Real-time Context Awareness**: Environmental and situational analysis

### 🏠 Smart Home Integration
- **Multi-Device Control**: Coffee maker, lights, climate, music, security systems
- **Intelligent Automation**: Calendar integration and schedule-based actions
- **Environmental Sensing**: Temperature, lighting, occupancy detection
- **Energy Management**: Optimized device scheduling and usage patterns

### 📊 Analytics & Insights
- **Performance Metrics**: Accuracy tracking and learning effectiveness
- **Usage Analytics**: Device interaction patterns and preferences
- **Satisfaction Trends**: User happiness tracking over time
- **Behavioral Insights**: Peak activity hours and habit analysis

### 🌐 Interactive Web Dashboard
- **Real-time Monitoring**: Live device status and AI suggestions
- **Visual Analytics**: Charts and graphs for usage patterns
- **Manual Control**: Direct device management interface
- **Feedback System**: Rate suggestions and improve AI learning

## 🚀 Quick Start
### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required Python packages
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-home-ai-assistant.git
cd smart-home-ai-assistant
```

2. **Install dependencies**
```bash
pip install numpy pandas scikit-learn datetime dataclasses collections pickle
```

3. **Run the AI Assistant**
```bash
python smart_home_office_assistant.py
```

4. **Launch the Web Dashboard**
```bash
# Open smart_home_dashboard.html in your web browser
# Or serve it using a local server:
python -m http.server 8000
# Then visit: http://localhost:8000/smart_home_dashboard.html
```

## 📋 Requirements

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
datetime
dataclasses
collections
pickle
warnings
```

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for optimal ML performance)
- **Storage**: 100MB for data and models
- **Browser**: Modern web browser for dashboard (Chrome, Firefox, Safari, Edge)

## 🏗️ Project Structure

```
smart-home-ai-assistant/
├── smart_home_office_assistant.py  # Main AI assistant backend
├── smart_home_dashboard.html       # Interactive web dashboard
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── data/                          # Training data and models
│   ├── user_actions.pkl          # Stored user action history
│   ├── ml_models.pkl             # Trained ML models
│   └── device_states.json        # Device configuration
├── docs/                          # Documentation
│   ├── API.md                    # API documentation
│   ├── SETUP.md                  # Detailed setup guide
│   └── CONTRIBUTING.md           # Contribution guidelines
└── examples/                      # Example usage scripts
    ├── basic_usage.py            # Simple usage example
    └── advanced_scenarios.py     # Complex scenarios
```

## 🎯 Usage Examples
### Basic Usage

```python
from smart_home_office_assistant import SmartHomeAI

# Initialize the AI assistant
assistant = SmartHomeAI("Your_Name")

# Log user actions for learning
assistant.log_user_action("coffee", {"time": "morning"}, satisfaction=5)
assistant.log_user_action("lights", {"room": "office"}, satisfaction=4)

# Get AI-powered suggestions
suggestions = assistant.make_ai_suggestions()
for suggestion in suggestions:
    print(f"💡 {suggestion.message} (Confidence: {suggestion.confidence:.2f})")

# Respond to suggestions
assistant.respond_to_suggestion(0, 'accepted', satisfaction=5)

# View analytics
analytics = assistant.get_ai_analytics()
print(f"🎯 Predictive Accuracy: {analytics['predictive_accuracy']}%")
```

### Advanced Configuration

```python
# Customize user preferences
assistant.user_preferences.update({
    'coffee_preferences': {'times': [7, 9, 14], 'strength': 'strong'},
    'work_schedule': {'start': 8, 'end': 18, 'break_times': [12, 15]},
    'proactive_level': 0.8,  # High proactiveness
    'learning_rate': 0.9     # Fast adaptation
})

# Configure device states
assistant.device_states['climate']['temperature'] = 70
assistant.device_states['lights']['office']['brightness'] = 85

# Advanced ML training
for _ in range(50):
    # Simulate various user interactions
    assistant.log_user_action("meeting_prep", {"urgency": "high"}, satisfaction=5)
```

## 🔧 Configuration

### AI Settings

```python
# Adjust AI behavior in smart_home_office_assistant.py
PROACTIVE_LEVEL = 0.7      # How often AI makes suggestions (0-1)
LEARNING_RATE = 0.8        # How quickly AI adapts (0-1)
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for suggestions
ML_RETRAIN_FREQUENCY = 5   # Retrain after N new actions
```

### Device Configuration

```python
# Add new devices to device_states dictionary
device_states = {
    'new_device': {
        'status': False,
        'properties': {'setting1': 'value1', 'setting2': 'value2'}
    }
}
```

## 📊 Web Dashboard Features

### Main Sections

1. **📊 Overview**
   - AI performance metrics
   - Activity statistics
   - System status indicators
   - Recent action log

2. **💡 AI Suggestions**
   - Real-time proactive suggestions
   - Confidence ratings and reasoning
   - Accept/decline interface
   - Satisfaction feedback system

3. **🏠 Devices**
   - Visual device status grid
   - Manual device control
   - Real-time status updates
   - Device interaction logging

4. **📈 Analytics**
   - Usage pattern charts
   - Satisfaction trend graphs
   - Behavioral profile analysis
   - Performance metrics visualization

5. **🧠 ML Learning**
   - Machine learning status
   - Training progress indicators
   - Pattern recognition insights
   - Model accuracy metrics

### Dashboard Screenshots

*Note: Add actual screenshots of your dashboard here*

```
[Overview Section]     [AI Suggestions]     [Analytics Charts]
     📊                    💡                    📈
```

## 🤖 Machine Learning Details

### Algorithms Used
1. **Random Forest Classifier**
   - **Purpose**: Action type prediction
   - **Features**: Time, context, environmental data
   - **Accuracy**: ~87% typical performance

2. **Gradient Boosting Classifier**
   - **Purpose**: Timing prediction
   - **Features**: Historical patterns, user schedule
   - **Performance**: High precision for routine predictions

3. **K-Means Clustering**
   - **Purpose**: User behavior profiling
   - **Clusters**: 5 distinct behavior patterns
   - **Application**: Personalization and adaptation

### Feature Engineering

```python
# Key features extracted for ML models:
- Time features: hour, minute, day_of_week, is_weekend
- Context features: temperature, stress_level, energy_level
- Device states: current status of all smart devices
- Historical patterns: recent actions and frequencies
- Environmental data: weather, lighting, occupancy
```

### Model Training Process
1. **Data Collection**: User actions logged with context
2. **Feature Extraction**: Convert actions to numerical features
3. **Model Training**: Periodic retraining every 5 actions
4. **Validation**: Cross-validation and accuracy tracking
5. **Deployment**: Live prediction serving

## 🔌 API Integration

### Adding New Devices

```python
def add_custom_device(self, device_name, device_config):
    """Add a new smart device to the system"""
    self.device_states[device_name] = device_config
    print(f"✅ Added device: {device_name}")

# Usage
assistant.add_custom_device('smart_tv', {
    'status': False,
    'volume': 30,
    'channel': 1,
    'input': 'HDMI1'
})
```

### Custom Action Types

```python
def register_custom_action(self, action_type, handler_function):
    """Register custom action type with handler"""
    self.custom_actions[action_type] = handler_function

# Usage
def handle_music_action(context):
    # Custom music control logic
    pass

assistant.register_custom_action('music_mood', handle_music_action)
```

## 📈 Performance Optimization

### Memory Management
- **Action History**: Limited to 1000 recent actions
- **Suggestion History**: Limited to 200 recent suggestions
- **Model Data**: Periodic cleanup of old training data

### Speed Optimization
- **Lazy Loading**: Models trained only when needed
- **Caching**: Frequent calculations are cached for performance
- **Batch Processing**: Multiple actions processed together

### Scalability
- **Modular Design**: Easy to add new devices and features
- **Database Ready**: Can be extended with persistent storage
- **Multi-User**: Architecture supports multiple user profiles

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests (when test suite is added)
python -m pytest tests/
```

### Integration Tests
```bash
# Test AI suggestions
python tests/test_ai_suggestions.py

# Test device control
python tests/test_device_control.py
```

### Performance Tests
```bash
# Benchmark ML performance
python tests/benchmark_ml.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make your changes**
4. **Add tests for new functionality**
5. **Submit a pull request**

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for all functions
- Comment on complex algorithms

## 🐛 Troubleshooting

### Common Issues

**Issue**: ML models are not training
```python
# Solution: Check minimum data requirements
if len(self.feature_data) < 5:
    print("Need at least 5 actions for ML training")
```

**Issue**: Dashboard not loading
```
# Solution: Check the browser console for errors
# Ensure all CSS/JS files are properly linked
```

**Issue**: Device control not working
```python
# Solution: Verify device configuration
print(assistant.device_states)  # Check device status
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose ML training
assistant.pattern_learner.debug_mode = True
```

## 📚 Documentation

- **[API Reference](docs/API.md)**: Detailed API documentation
- **[Setup Guide](docs/SETUP.md)**: Step-by-step setup instructions
- **[Architecture Guide](docs/ARCHITECTURE.md)**: System design overview
- **[ML Guide](docs/MACHINE_LEARNING.md)**: Machine learning implementation details


## 📄 License
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- **Scikit-learn** for machine learning algorithms
- **Chart.js** for dashboard visualizations
- **Modern CSS** techniques for responsive design
- **Open source community** for inspiration and best practices

## 📞 Support
- **Issues**: [GitHub Issues](https://github.com/yourusername/smart-home-ai-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/smart-home-ai-assistant/discussions)

- **Made with ❤️ by Melisa Sever**

- *Building the future of intelligent home automation, one algorithm at a time.*
