# CollabIDS: Federated Learning for Intrusion Detection System with Interactive Web Dashboard

## Overview
CollabIDS is a comprehensive Intrusion Detection System that combines the power of Federated Learning with an intuitive web-based interface. This system enables collaborative, privacy-preserving network security solutions through the Flower framework and UNSW_NB15 dataset, while providing real-time attack prediction and monitoring capabilities through a modern Streamlit dashboard.

## 🌟 Key Features
- **🔐 Federated Learning**: Privacy-preserving distributed training across multiple organizations
- **🖥️ Interactive Web Dashboard**: Modern Streamlit-based interface for model management
- **🔍 Real-time Attack Prediction**: Upload and analyze network traffic data instantly
- **📊 Performance Monitoring**: Comprehensive metrics and visualization tools
- **🎯 Model Selection**: Support for multiple ML architectures (RNN, CNN, DNN)
- **📈 Data Visualization**: Advanced plotting and analysis capabilities
- **🛡️ Security-First Design**: Enterprise-grade privacy and security features

## Architecture

### System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                     Federated Learning Server                  │
│                        (Port 8080)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   FedAvg        │  │   Aggregation   │  │   Evaluation    │ │
│  │   Strategy      │  │   Coordinator   │  │   Manager       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────┬───────────────────┬───────────────────┬─────────┘
              │                   │                   │
         ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
         │Client 1 │         │Client 2 │         │Client N │
         │         │         │         │         │         │
         │ ┌─────┐ │         │ ┌─────┐ │         │ ┌─────┐ │
         │ │Local│ │         │ │Local│ │         │ │Local│ │
         │ │ NN  │ │         │ │ NN  │ │         │ │ NN  │ │
         │ │Model│ │         │ │Model│ │         │ │Model│ │
         │ └─────┘ │         │ └─────┘ │         │ └─────┘ │
         │ ┌─────┐ │         │ ┌─────┐ │         │ ┌─────┐ │
         │ │Local│ │         │ │Local│ │         │ │Local│ │
         │ │Data │ │         │ │Data │ │         │ │Data │ │
         │ └─────┘ │         │ └─────┘ │         │ └─────┘ │
         └─────────┘         └─────────┘         └─────────┘
```

### System Components

#### 1. **Web Dashboard** (`src/streamlit_app.py`)
- **Role**: Main entry point with secure authentication
- **Key Features**:
  - Modern cybersecurity-themed UI
  - Secure login system (Admin/Admin)
  - Background security imagery
  - Navigation to prediction system
- **Technologies**: Streamlit, HTML/CSS styling

#### 2. **Interactive Pages** (`pages/`)
- **Page 1** (`page1.py`): **Model Selection**
  - Choose between RNN, CNN, DNN architectures
  - Detailed model descriptions and use cases
  - Save and proceed functionality
- **Page 2** (`page2.py`): **Data Input**
  - Dataset upload and validation
  - Data preview and preprocessing
- **Page 3** (`page3.py`): **Model Training Control**
  - Training parameter configuration
  - Real-time training monitoring
- **Page 4** (`page4.py`): **Performance Metrics**
  - Accuracy, loss, and convergence analysis
  - Model comparison tools
- **Page 5** (`page5.py`): **Logs and Reports**
  - Training logs and audit trails
  - Comprehensive reporting system
- **Page 6** (`page6.py`): **Data Visualization**
  - Interactive charts and graphs
  - Feature analysis and correlation plots
- **Page 7** (`page7.py`): **Attack Prediction**
  - Real-time attack classification
  - Threat assessment dashboard

#### 3. **Prediction Engine** (`pages/predict.py`)
- **Role**: Core prediction functionality for uploaded data
- **Key Features**:
  - CSV file upload and validation (42 features expected)
  - Real-time preprocessing (encoding + normalization)
  - Global model loading and prediction
  - Color-coded results (✅ Benign / 🚨 Attack)
  - Comprehensive model information sidebar
- **Output**: Attack/Benign classification with confidence scores

#### 4. **Federated Learning Core** (`src/`)
- **Server Component** (`src/server.py`)
  - Central coordinator for federated learning process
  - FedAvg aggregation strategy implementation
  - Minimum 3 clients requirement
  - Weighted averaging of client metrics
  
- **Client Component** (`src/client.py`)
  - Distributed training nodes
  - Local data processing and model training
  - Secure communication with server
  
- **Data Processing** (`src/loader.py`)
  - UNSW_NB15 dataset management
  - Feature encoding and normalization
  - Train/test data splitting

#### 5. **Model Architecture Support**
```
Input Layer (42 features from UNSW_NB15)
    ↓
Dense Layer (100 neurons, ReLU activation)
    ↓
Layer Normalization
    ↓
Dense Layer (50 neurons, ReLU activation)
    ↓
Layer Normalization
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

### How Federated Learning Works in CollabIDS

#### 1. **Initialization Phase**
```python
# Server initialization
strategy = FedAvg(min_fit_clients=3, min_evaluate_clients=3)
server.start_server(server_address="0.0.0.0:8080", strategy=strategy)

# Client initialization
client = Client()  # Loads local data and creates model
```

#### 2. **Training Workflow (Per Round)**

**Step 1: Model Distribution**
- Server sends current global model weights to all connected clients
- Each client receives the same initial model state

**Step 2: Local Training**
```python
# On each client
def fit(self, parameters, config):
    self.model.set_weights(parameters)  # Update with global weights
    history = self.model.fit(self.X_train, self.Y_train, epochs=1, batch_size=64)
    return self.model.get_weights(), len(self.X_train), metrics
```

**Step 3: Model Aggregation**
- Clients send updated weights back to server
- Server performs weighted averaging based on dataset sizes:
```python
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics = {}
    for num_examples, m in metrics:
        for key, value in m.items():
            weighted_metrics[key] += (num_examples * value)
    return {k: v / total_examples for k, v in weighted_metrics.items()}
```

**Step 4: Global Model Update**
- Server creates new global model from aggregated weights
- Process repeats for next training round

#### 3. **Privacy Preservation**
- **Data Privacy**: Raw data never leaves client premises
- **Model Privacy**: Only model updates (gradients/weights) are shared
- **Secure Aggregation**: Server only sees aggregated results, not individual contributions

#### 4. **Evaluation Process**
```python
def evaluate(self, parameters, config):
    self.model.set_weights(parameters)  # Use global model
    loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
    return loss, len(self.X_test), {"accuracy": accuracy}
```

### UNSW_NB15 Dataset Integration

#### Dataset Characteristics
- **Features**: 42 network traffic features
- **Classes**: Binary classification (Normal vs Attack)
- **Preprocessing Steps**:
  1. Remove non-predictive columns (`id`, `attack_cat`)
  2. Encode categorical variables using LabelEncoder
  3. Normalize numerical features using MinMaxScaler
  4. Split into features (X) and labels (Y)

#### Feature Types
- **Numerical**: Duration, packet counts, byte counts, flow statistics
- **Categorical**: Protocol types, service types, connection states
- **Binary**: Various flags and indicators

### Communication Protocol
- **Framework**: Flower (gRPC-based communication)
- **Message Types**:
  - Parameter requests/responses
  - Fit instructions/results  
  - Evaluate instructions/results
- **Serialization**: NumPy arrays for model weights
- **Network**: TCP/IP over configurable ports

## 🌐 Web Dashboard Features

### 🔐 Authentication System
- **Secure Login**: Username/Password authentication
- **Session Management**: Persistent login sessions
- **Access Control**: Protected pages and navigation

### 📱 Interactive Interface
- **Modern UI**: Cybersecurity-themed design with dark mode
- **Responsive Layout**: Works on desktop and mobile devices
- **Real-time Updates**: Live model training and prediction results
- **Background Images**: Professional cybersecurity aesthetics

### 🎯 Prediction Workflow
1. **Upload CSV**: Drag and drop or browse for network traffic data
2. **Data Validation**: Automatic check for 42 required features
3. **Preprocessing**: Automatic encoding and normalization
4. **Model Inference**: Real-time prediction using trained global model
5. **Results Display**: Color-coded attack/benign classifications
6. **Download Reports**: Export prediction results

### 📊 Dashboard Pages Overview

#### **Model Selection Page**
- **RNN**: For sequential network traffic analysis
- **CNN**: For pattern recognition in network packets  
- **DNN**: For general-purpose intrusion detection
- **Interactive Descriptions**: Detailed explanations for each model type

#### **Data Input Page**
- **File Upload**: Support for CSV datasets
- **Data Preview**: Interactive data table with filtering
- **Validation**: Real-time data quality checks
- **Format Guidance**: Clear instructions for data preparation

#### **Training Control Page**
- **Parameter Tuning**: Adjust learning rates, epochs, batch sizes
- **Progress Monitoring**: Real-time training progress bars
- **Client Status**: Live view of connected federated clients
- **Training Logs**: Detailed training history and metrics

#### **Performance Metrics Page**
- **Accuracy Tracking**: Global and per-client accuracy metrics
- **Loss Visualization**: Training and validation loss curves
- **Convergence Analysis**: Model convergence speed and stability
- **Comparison Tools**: Compare different model architectures

#### **Logs and Reports Page**
- **Training Logs**: Complete audit trail of all training activities
- **Error Reports**: Debugging information and error tracking
- **Export Options**: Download logs in various formats
- **Search and Filter**: Find specific events or time periods

#### **Data Visualization Page**
- **Feature Analysis**: Interactive correlation matrices
- **Attack Distribution**: Visual breakdown of attack types
- **Training Progress**: Multi-dimensional performance plots
- **Custom Charts**: Create custom visualizations

#### **Attack Prediction Page**
- **Real-time Classification**: Instant attack/benign prediction
- **Confidence Scores**: Model confidence for each prediction
- **Batch Processing**: Handle multiple samples simultaneously
- **Alert System**: Visual and audio alerts for detected attacks

## Setup & Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.15.0
- Flower framework 1.6.0
- Streamlit (latest)
- UNSW_NB15 dataset

### Quick Start Guide

#### 1. Clone and Setup
```bash
git clone https://github.com/malek0501/CollabIDS.git
cd CollabIDS

# Install core dependencies
pip install -r src/requirements.txt

# Install additional web dependencies
pip install streamlit streamlit-extras pillow
```

#### 2. Dataset Preparation
```bash
# Download UNSW_NB15 dataset from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Place files in data/ directory:
mkdir -p data/
# Copy UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv to data/
```

#### 3. Launch Web Dashboard
```bash
# Start the main web application
streamlit run src/streamlit_app.py

# Default login credentials:
# Username: Admin
# Password: Admin
```

#### 4. Alternative: Command Line Usage
```bash
# Traditional federated learning mode
cd src/

# Start server
python server.py

# Start clients (in separate terminals)
python client.py  # Run at least 3 instances

# Or run simulation
python simulation.py
```

## Execution Options

### 1. 🖥️ Web Dashboard (Recommended)
```bash
# Launch interactive web interface
streamlit run src/streamlit_app.py

# Access at: http://localhost:8501
# Login: Admin / Admin
```
**Features:**
- **Model Selection**: Choose RNN, CNN, or DNN architectures
- **Data Management**: Upload and process datasets
- **Training Control**: Monitor and control federated learning
- **Real-time Prediction**: Upload CSV files for instant attack detection
- **Visualization**: Interactive plots and performance metrics

### 2. 🔍 Prediction Mode
```bash
# Direct prediction interface
streamlit run pages/predict.py
```
Upload CSV files with 42 features for real-time attack classification.

### 3. ⚙️ Federated Learning (CLI)
```bash
cd src/

# Start server
python server.py

# Start clients (separate terminals)
python client.py  # Minimum 3 clients required
```

### 4. 🤖 Automated Simulation
```bash
cd src/
python simulation.py
```

### 5. 🐳 Containerized Deployment
```bash
docker-compose up --build
```

## 📊 Model Visualization & Analysis
```bash
# Generate model architecture plots
cd src/
python plot.py

# Results saved to: imagesplot/
# Metrics saved to: metrics_comparison/
```

## Performance Metrics & Monitoring

### Key Performance Indicators
- **Global Model Accuracy**: Aggregated accuracy across all clients
- **Training Loss**: Binary crossentropy loss per training round
- **Convergence Rate**: Speed of model convergence across rounds
- **Client Participation**: Number of active clients per round

### Monitoring Dashboard
The system tracks and reports:
```python
# Example output after training
INFO : Run finished 5 rounds in 45.32s
INFO : History (loss, distributed): [(1, 0.543), (2, 0.421), ...]
INFO : History (accuracy, distributed): [(1, 0.876), (2, 0.912), ...]
After 5 rounds of training the accuracy is 92.4%
```

## Security & Privacy Features

### Privacy Guarantees
1. **Data Locality**: Training data never leaves client premises
2. **Model Privacy**: Only aggregated model updates are shared
3. **Differential Privacy**: Can be enhanced with noise injection
4. **Secure Aggregation**: Server cannot inspect individual client contributions

### Security Measures
- **Authentication**: Client-server authentication via Flower framework
- **Encrypted Communication**: gRPC with TLS support
- **Access Control**: Configurable client admission policies
- **Audit Trail**: Comprehensive logging of all training activities

## Configuration

### Server Configuration (`server.py`)
```python
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,           # Minimum clients for training
    min_evaluate_clients=3,      # Minimum clients for evaluation  
    min_available_clients=3,     # Minimum clients to start round
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
)

config = fl.server.ServerConfig(
    num_rounds=5,                # Number of training rounds
)
```

### Client Configuration (`client.py`)
```python
# Training parameters
epochs_per_round = 1
batch_size = 64
learning_rate = 0.01

# Model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(50, activation="relu"), 
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

### Web Application Configuration
```bash
# Streamlit configuration
STREAMLIT_SERVER_PORT=8501       # Web dashboard port
STREAMLIT_SERVER_ADDRESS=0.0.0.0 # Bind address

# Authentication
ADMIN_USERNAME=Admin             # Default admin username
ADMIN_PASSWORD=Admin             # Default admin password

# File paths
GLOBAL_MODEL_PATH=global_model.h5    # Trained model location
BACKGROUND_IMAGE=cyber.jpg           # Dashboard background
```

### Environment Variables
```bash
# Federated Learning configuration
SERVER_ADDRESS=127.0.0.1:8080    # FL server connection address
TF_CPP_MIN_LOG_LEVEL=2           # Reduce TensorFlow verbosity

# Docker configuration  
COMPOSE_PROJECT_NAME=collabids   # Docker compose project name

# Data directories
DATA_DIR=./data                  # Dataset location
MODEL_DIR=./model_checkpoints    # Model storage
PLOTS_DIR=./imagesplot          # Visualization output
```

### Dependencies
```bash
# Core ML dependencies (src/requirements.txt)
flwr==1.6.0                     # Federated learning framework
pandas==2.1.4                   # Data manipulation
scikit_learn==1.3.2             # Machine learning utilities
tensorflow==2.15.0              # Deep learning framework

# Web interface dependencies
streamlit>=1.28.0               # Web framework
streamlit-extras>=0.3.0         # Additional components
pillow>=9.0.0                   # Image processing
plotly>=5.0.0                   # Interactive plotting
```

## Project Structure
```
CollabIDS/
├── src/                          # Core federated learning components
│   ├── streamlit_app.py         # Main web application entry point
│   ├── server.py                # Federated learning server
│   ├── client.py                # Federated learning client
│   ├── loader.py                # Data processing and model loading
│   ├── simulation.py            # Automated FL simulation
│   ├── plot.py                  # Model visualization utilities
│   ├── navigation.py            # Web app navigation logic
│   ├── summary.py               # Training summaries and reports
│   ├── flp.py                   # Federated learning protocols
│   └── requirements.txt         # Core dependencies
├── pages/                        # Interactive web dashboard pages
│   ├── page1.py                 # Model Selection interface
│   ├── page2.py                 # Data Input management
│   ├── page3.py                 # Training Control dashboard
│   ├── page4.py                 # Performance Metrics viewer
│   ├── page5.py                 # Logs and Reports system
│   ├── page6.py                 # Data Visualization tools
│   ├── page7.py                 # Attack Prediction interface
│   └── predict.py               # Core prediction engine
├── data/                         # Dataset and processing files
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   ├── Sampled_Data.csv         # Processed sample data
│   └── *.txt                    # Processing logs
├── model_checkpoints/            # Trained model storage
├── metrics_comparison/           # Performance analysis results
├── imagesplot/                   # Generated visualization plots
├── .devcontainer/               # Development container config
└── README.md                    # Project documentation
```

## Deployment Scenarios

### 1. Research & Development
```bash
# Single machine simulation
python simulation.py
```
- Simulates 3 clients on one machine
- Ideal for algorithm testing and development
- Fast iteration cycles

### 2. Multi-Organization Deployment  
```bash
# Organization A (Server)
python server.py

# Organization B (Client)  
SERVER_ADDRESS=orgA.example.com:8080 python client.py

# Organization C (Client)
SERVER_ADDRESS=orgA.example.com:8080 python client.py
```
- Distributed across multiple organizations
- Real-world federated learning scenario
- Each organization keeps data private

### 3. Cloud-Native Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  server:
    build: .
    command: python server.py
    ports: ["8080:8080"]
  
  client1:
    build: .
    command: python client.py
    environment:
      - SERVER_ADDRESS=server:8080
```

### 4. Web-Based Production Deployment
```bash
# Production web server
streamlit run src/streamlit_app.py --server.port 80 --server.address 0.0.0.0

# With SSL/TLS
streamlit run src/streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
```

### 5. Edge Computing Deployment
- Deploy clients on edge devices (IoT, routers, firewalls)
- Central server in cloud or data center
- Web dashboard for remote monitoring and control
- Real-time intrusion detection at network edge

## Troubleshooting

### Common Issues

#### 1. Web Dashboard Issues
**Problem**: `streamlit: command not found`
```bash
# Install streamlit
pip install streamlit streamlit-extras

# Verify installation
streamlit --version
```

**Problem**: Dashboard not accessible
```bash
# Check if port 8501 is available
netstat -tlnp | grep 8501

# Run with different port
streamlit run src/streamlit_app.py --server.port 8502
```

**Problem**: Background images not loading
- Ensure `cyber.jpg` exists in the project root
- Check file permissions: `chmod 644 cyber.jpg`
- Verify image path in `streamlit_app.py`

#### 2. Model Loading Issues
**Problem**: `global_model.h5 not found`
```bash
# Train a model first using federated learning
cd src/
python simulation.py

# Or copy pre-trained model to root directory
cp model_checkpoints/global_model.h5 ./
```

#### 3. Prediction Errors
**Problem**: `Expected 42 features, but received X`
- Ensure uploaded CSV has exactly 42 feature columns
- Remove `id`, `label`, and `attack_cat` columns before upload
- Check feature names match UNSW_NB15 format

#### 4. Connection Errors
**Problem**: `Connection refused` or `Unable to connect to server`
```bash
# Check server status
netstat -tlnp | grep 8080

# Verify server is running
ps aux | grep server.py

# Check firewall rules
sudo ufw status
```

#### 2. Insufficient Clients
**Problem**: `Not enough clients available`
- Ensure at least 3 clients are connected
- Check client logs for connection issues
- Verify server configuration allows client connections

#### 3. Dataset Issues  
**Problem**: `FileNotFoundError` for dataset files
```bash
# Verify dataset files exist
ls -la data/
ls -la data/UNSW_NB15_*.csv

# Check file permissions
chmod 644 data/*.csv
```

#### 4. Memory Issues
**Problem**: `Out of memory` during training
- Reduce batch size in client.py
- Implement data sampling for large datasets
- Use gradient accumulation for large models

#### 5. Model Convergence Issues
**Problem**: Poor accuracy or slow convergence
- Increase number of training rounds
- Adjust learning rate (0.001 - 0.01)
- Check data quality and preprocessing
- Verify model architecture suitability

### Performance Optimization

#### 1. Network Optimization
- Use compression for model updates
- Implement asynchronous communication
- Optimize serialization/deserialization

#### 2. Computational Optimization  
- Use GPU acceleration when available
- Implement model pruning
- Use mixed-precision training

#### 3. Data Optimization
- Implement data caching
- Use efficient data loaders
- Optimize preprocessing pipelines

## Future Enhancements

### Short-term Goals
- [ ] **Enhanced Authentication**: Multi-factor authentication and role-based access
- [ ] **Real-time Monitoring**: Live federated learning progress tracking in web UI
- [ ] **Batch Prediction**: Upload multiple files for bulk attack detection
- [ ] **Model Comparison**: Side-by-side performance comparison in dashboard
- [ ] **Export Features**: Download training reports and prediction results
- [ ] **Mobile Optimization**: Responsive design improvements for mobile devices

### Medium-term Goals  
- [ ] **Advanced Web Features**: 
  - Interactive model architecture editor
  - Real-time collaboration tools for multiple users
  - Advanced data visualization with 3D plots
  - Automated report generation and scheduling
- [ ] **Enhanced FL Capabilities**:
  - Support for more aggregation strategies (FedProx, FedNova)
  - Dynamic client joining/leaving via web interface
  - Personalized datasets for each client
  - Automated hyperparameter tuning through UI
- [ ] **Integration & Deployment**:
  - MLOps pipeline integration
  - Docker containerization for web app
  - Cloud deployment templates (AWS, Azure, GCP)
  - API endpoints for external integration

### Long-term Vision
- [ ] **Enterprise Integration**:
  - SIEM system integration with web hooks
  - Enterprise SSO (LDAP, Active Directory)
  - Multi-tenant architecture for organizations
  - Advanced audit trails and compliance reporting
- [ ] **AI-Powered Features**:
  - AI-driven client selection strategies
  - Automated threat intelligence integration
  - Self-healing federated networks
  - Predictive maintenance for model performance
- [ ] **Security & Privacy**:
  - Blockchain-based model verification
  - Quantum-resistant cryptographic protocols
  - Zero-knowledge proof implementations
  - Homomorphic encryption for enhanced privacy

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
[MIT License](LICENSE)

## Acknowledgments
- [Flower Framework](https://github.com/adap/flower) for federated learning capabilities
- [UNSW_NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for providing the training data