import joblib
import os
import pandas as pd


class SupplyChainBackend:
    def __init__(self):
        # Base directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.base_dir, "models")

        self.models = {}
        self.encoders = {}

        model_files = {
            "late": "late_delivery_model.pkl",
            "loss": "abs_loss_model.pkl",
            "margin": "margin_loss_model.pkl"
        }

        try:
            # Load encoders
            encoder_path = os.path.join(self.model_dir, "encoders.pkl")
            self.encoders = joblib.load(encoder_path)

            # Load ML models
            for key, filename in model_files.items():
                model_path = os.path.join(self.model_dir, filename)
                self.models[key] = joblib.load(model_path)

            print("✅ Models and encoders loaded successfully.")

        except FileNotFoundError as e:
            print(f"❌ Missing file: {e.filename}")
        except Exception as e:
            print(f"❌ Initialization error: {e}")

    def process_new_order(self, raw_data):

        if not self.models or not self.encoders:
            return {"status": "Error", "message": "Models not initialized properly."}

        try:
            input_df = pd.DataFrame([raw_data])

            # Encode categorical features
            for col, encoder in self.encoders.items():
                if col in input_df.columns:
                    input_df[col] = encoder.transform(input_df[col].astype(str))

            # -----------------------------
            # ML MODEL RISK PREDICTIONS
            # -----------------------------
            risks = {
                name: model.predict_proba(input_df)[0][1]
                for name, model in self.models.items()
            }

            # -----------------------------
            # RETURN RISK (Improved Rule-Based)
            # -----------------------------
            discount = float(raw_data.get("Order Item Discount", 0))
            price = float(raw_data.get("Product Price", 0))
            shipping_days = float(raw_data.get("Days for shipment (scheduled)", 0))

            return_risk_score = 0

            # Higher discount → more returns
            if discount > 0.15:
                return_risk_score += 0.3

            # Cheaper products → more returns
            if price < 50:
                return_risk_score += 0.2

            # Slow shipping increases dissatisfaction
            if shipping_days > 5:
                return_risk_score += 0.3

            # Financial loss risk from ML model
            if risks.get("loss", 0) > 0.4:
                return_risk_score += 0.2

            # Clamp between 0 and 1
            return_risk_score = min(return_risk_score, 1)

            risks["return"] = return_risk_score

            # -----------------------------
            # GENERATE RECOMMENDATIONS
            # -----------------------------
            solutions = self._get_prescriptive_advice(risks, raw_data)

            return {
                "status": "Success",
                "risk_scores": {k: f"{v:.2%}" for k, v in risks.items()},
                "solutions": solutions
            }

        except Exception as e:
            return {"status": "Error", "message": f"Prediction failed: {str(e)}"}

    def _get_prescriptive_advice(self, risks, original_data):

        advice = []

        if risks.get("late", 0) > 0.5:
            advice.append(
                f"LOGISTICS: High delay risk in {original_data.get('Order Region','Unknown')} region. Consider faster shipping."
            )

        if risks.get("loss", 0) > 0.5:
            advice.append(
                "FINANCIAL: Order may lead to negative profit. Review discount strategy."
            )

        if risks.get("margin", 0) > 0.5:
            advice.append(
                "PROFITABILITY: Shipping overhead may reduce margins. Consider batching shipments."
            )

        if risks.get("return", 0) > 0.5:
            advice.append(
                "RETURNS: High probability of return. Consider reducing discount or verifying order details."
            )

        if not advice:
            advice.append("Order appears optimized. No action required.")

        return advice


# -----------------------------
# LOCAL TEST
# -----------------------------
if __name__ == "__main__":

    backend = SupplyChainBackend()

    test_order = {
        "Type": "DEBIT",
        "Shipping Mode": "Standard Class",
        "Order Region": "Western Europe",
        "Days for shipment (scheduled)": 6,
        "Product Price": 40,
        "Order Item Discount": 0.25
    }

    result = backend.process_new_order(test_order)

    print("\n--- SYSTEM OUTPUT ---")
    print(result)