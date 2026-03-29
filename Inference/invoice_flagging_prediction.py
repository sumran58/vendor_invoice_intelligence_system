import joblib
import pandas as pd

MODEL_PATH = r"C:\Users\windows\Freight_cost_prediction\Invoice_flagging\models\predict_flag_invoice.pkl"
def load_model(model_path: str = MODEL_PATH):
    """
    Load trained classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    input_df = pd.DataFrame(input_data)
    
    # Predicting the risk flag (0 or 1)
    input_df['Predicted_Flag'] = model.predict(input_df)
    return input_df

if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [100],
        "invoice_dollars": [5000],
        "Freight": [250],
        "total_item_quantity": [100],
        "total_item_dollars": [5000]
    }

    result = predict_invoice_flag(sample_data)
    print(result)