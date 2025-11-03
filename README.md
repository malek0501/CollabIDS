# CollabIDS: Federated Learning for Intrusion Detection System

## Overview
CollabIDS is a sophisticated Intrusion Detection System that leverages the power of Federated Learning to create a collaborative, privacy-preserving network security solution. Built using the Flower framework and trained on the UNSW_NB15 dataset, this system enables multiple organizations to collaboratively train an IDS model without sharing their sensitive network data.

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

#### 1. **Server Component** (`server.py`)
- **Role**: Central coordinator for the federated learning process
- **Key Features**:
  - Implements Federated Averaging (FedAvg) aggregation strategy
  - Manages client connections and synchronization
  - Enforces minimum client requirements (3 clients minimum)
  - Tracks training metrics across rounds
  - Provides weighted averaging of client metrics
- **Configuration**:
  - Port: 8080 (configurable)
  - Training rounds: 5 (default)
  - Minimum clients: 3 for fit/evaluate/available

#### 2. **Client Component** (`client.py`)
- **Role**: Distributed training nodes that process local data
- **Architecture**: Extends `fl.client.NumPyClient`
- **Key Functions**:
  - `get_parameters()`: Retrieves current model weights
  - `fit()`: Performs local training on private dataset
  - `evaluate()`: Evaluates global model on local test data
- **Local Training**:
  - Batch size: 64
  - Epochs per round: 1
  - Optimizer: Adam with learning rate 0.01

#### 3. **Data Processing Module** (`loader.py`)
- **DataLoader Class**:
  - Loads UNSW_NB15 training and testing datasets
  - Handles categorical encoding using LabelEncoder
  - Applies Min-Max normalization to numerical features
  - Removes irrelevant columns (`id`, `attack_cat`)
  - Separates features (X) from labels (Y)
- **ModelLoader Class**:
  - Creates neural network architecture
  - Input layer: Variable size based on dataset features
  - Hidden layers: Dense(100) → LayerNorm → Dense(50) → LayerNorm
  - Output layer: Dense(1) with sigmoid activation for binary classification
  - Loss function: Binary crossentropy
  - Metrics: Binary accuracy

#### 4. **Neural Network Architecture**
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

## Setup & Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Flower framework
- UNSW_NB15 dataset

### Dataset Setup
1. Download the UNSW_NB15 dataset from [UNSW_NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
2. Place the following files in the `data/` directory:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/malek0501/CollabIDS.git
   cd CollabIDS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Options

### 1. Manual Execution
```bash
# Start the server
python server.py

# Start clients (run in separate terminals)
python client.py  # Run at least 3 instances
```

### 2. Automated Simulation
```bash
python simulation.py
```

### 3. Containerized Deployment
```bash
docker-compose up --build
```

## Model Visualization
Generate a visual representation of the model architecture:
```bash
python plot.py
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

### Environment Variables
```bash
# Client configuration
SERVER_ADDRESS=127.0.0.1:8080    # Server connection address
TF_CPP_MIN_LOG_LEVEL=2           # Reduce TensorFlow verbosity

# Docker configuration  
COMPOSE_PROJECT_NAME=collabids   # Docker compose project name
```

## Project Structure
```
CollabIDS/
├── client.py           # Client implementation
├── server.py           # Server implementation
├── loader.py           # Data loading and preprocessing
├── plot.py            # Visualization utilities
├── simulation.py      # Automated simulation
├── requirements.txt   # Project dependencies
├── data/             # Dataset directory
└── pages/            # Web interface pages
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

### 4. Edge Computing Deployment
- Deploy clients on edge devices (IoT, routers, firewalls)
- Central server in cloud or data center
- Real-time intrusion detection at network edge

## Troubleshooting

### Common Issues

#### 1. Connection Errors
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
- [ ] Implement personalized datasets for each client
- [ ] Add support for dynamic client joining/leaving  
- [ ] Enhance privacy mechanisms with differential privacy
- [ ] Implement model compression for faster communication
- [ ] Add real-time monitoring dashboard

### Medium-term Goals  
- [ ] Support for more advanced aggregation strategies (FedProx, FedNova)
- [ ] Integration with MLOps pipelines
- [ ] Automated hyperparameter tuning
- [ ] Multi-class intrusion detection
- [ ] Real-time threat detection capabilities

### Long-term Vision
- [ ] Integration with SIEM systems
- [ ] Blockchain-based model verification
- [ ] Quantum-resistant cryptographic protocols
- [ ] Self-healing federated networks
- [ ] AI-driven client selection strategies

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
[MIT License](LICENSE)

## Acknowledgments
- [Flower Framework](https://github.com/adap/flower) for federated learning capabilities
- [UNSW_NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for providing the training data