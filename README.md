# CollabIDS: Federated Learning for Intrusion Detection System

## Overview
CollabIDS is a sophisticated Intrusion Detection System that leverages the power of Federated Learning to create a collaborative, privacy-preserving network security solution. Built using the Flower framework and trained on the UNSW_NB15 dataset, this system enables multiple organizations to collaboratively train an IDS model without sharing their sensitive network data.

## Architecture

### System Components
1. **Server Component** (`server.py`)
   - Coordinates the federated learning process
   - Aggregates model updates from clients using FedAvg strategy
   - Manages training rounds and client synchronization
   - Tracks and reports global model performance

2. **Client Component** (`client.py`)
   - Handles local model training
   - Processes local dataset
   - Communicates with the central server
   - Implements model evaluation on local data

3. **Data Processing** (`loader.py`)
   - Manages UNSW_NB15 dataset preprocessing
   - Handles feature engineering and normalization
   - Splits data for training and testing

4. **Visualization** (`plot.py`)
   - Generates model architecture visualizations
   - Creates performance metrics plots
   - Provides insights into training progress

### Workflow
1. **Initialization Phase**
   - Server starts and waits for client connections
   - Clients initialize with local datasets
   - Initial model architecture is distributed

2. **Training Phase**
   - Server orchestrates training rounds
   - Each client trains on local data
   - Model updates are aggregated using FedAvg
   - Progress is tracked and logged

3. **Evaluation Phase**
   - Global model is evaluated on each client
   - Performance metrics are aggregated
   - Results are stored and visualized

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
   git clone https://github.com/[username]/CollabIDS.git
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

## Configuration

### Server Configuration
- Default port: 8080
- Minimum clients required: 3
- Training rounds: 5 (configurable)
- Aggregation strategy: Federated Averaging (FedAvg)

### Client Configuration
- Batch size: 32
- Local epochs: 5
- Learning rate: 0.001
- Model architecture: Neural Network with multiple dense layers

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

## Future Enhancements
- [ ] Implement personalized datasets for each client
- [ ] Add support for dynamic client joining/leaving
- [ ] Enhance privacy mechanisms
- [ ] Implement more advanced aggregation strategies
- [ ] Add real-time threat detection capabilities

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
[MIT License](LICENSE)

## Acknowledgments
- [Flower Framework](https://github.com/adap/flower) for federated learning capabilities
- [UNSW_NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for providing the training data