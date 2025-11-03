import os
import flwr as fl
from loader import DataLoader, ModelLoader
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Make tensorflow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Client(fl.client.NumPyClient):
    def __init__(self):
        data_loader = DataLoader("data/UNSW_NB15_training-set.csv", "data/UNSW_NB15_testing-set.csv")
        self.X_train, self.Y_train, self.X_test, self.Y_test = data_loader.get_data()
        self.model = ModelLoader.get_model(self.X_train.shape[1:])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, _):
        self.model.set_weights(parameters)

        # Define callbacks here
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            #♪ModelCheckpoint("best_model.h5", save_best_only=True)
        ]

        self.model.fit(
            self.X_train,
            self.Y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks
        )

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, _):
        # Set model weights
        self.model.set_weights(parameters)
    
        # Evaluate model to get loss and accuracy
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
    
        # Get predictions for other metrics
        y_pred = self.model.predict(self.X_test)
        y_pred_binary = (y_pred > 0.5).astype(int) 

        #print(f"\n Eval loss = {loss:.4f} on {len(self.X_test)} samples\n")
        report = classification_report(self.Y_test, y_pred_binary, output_dict=True)
        print(classification_report(self.Y_test, y_pred_binary))
    
        positive_class = "1.0" if "1.0" in report else "1"
        #print(f"loss for this client is : {loss}\n")
        return loss, len(self.X_test), {
            "accuracy": report["accuracy"],
            "precision": report[positive_class]["precision"],
            "recall": report[positive_class]["recall"],
            "f1": report[positive_class]["f1-score"]
        }



if __name__ == "__main__":
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_numpy_client(server_address=server_address, client=Client())
