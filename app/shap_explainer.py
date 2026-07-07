import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path


class ThreatExplainer:
    """
    SHAP-based explainability layer for Tier 1 LightGBM predictions.
    
    Uses TreeExplainer — the fastest and most accurate SHAP method
    for tree-based models. Computes feature contributions for individual
    flagged packets so analysts can understand WHY a packet was classified
    as a specific threat type.
    """

    def __init__(self, artifacts_dir: Path):
        print("[SHAP] Loading LightGBM model for explanation...")
        self.lgbm = joblib.load(artifacts_dir / "tier1_lightgbm.pkl")
        self.label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")
        self.class_names = list(self.label_encoder.classes_)

        # TreeExplainer is optimized specifically for LightGBM/XGBoost/RF
        # It computes exact SHAP values rather than approximations
        print("[SHAP] Initializing TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.lgbm)
        print("[SHAP] Explainer ready.\n")

    def explain_packet(self, packet_features: pd.DataFrame, top_n: int = 5) -> dict:
        """
        Computes SHAP explanation for a single packet.
        """
        # Get LightGBM probability predictions
        probs = self.lgbm.predict(packet_features)
        predicted_idx = int(np.argmax(probs[0]))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probs[0][predicted_idx])

        # Compute SHAP values
        shap_values = self.explainer.shap_values(packet_features)

        # === FIX: Handle list vs 3D Array formats across SHAP versions ===
        if isinstance(shap_values, list):
            # Old SHAP format: list of length num_classes with arrays of shape (num_samples, num_features)
            class_shap = shap_values[predicted_idx][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Modern SHAP format: (num_samples, num_features, num_classes)
            class_shap = shap_values[0, :, predicted_idx]
        else:
            # Fallback uniform array format
            class_shap = shap_values[0]
        # =================================================================

        # Build feature contribution table
        feature_names = list(packet_features.columns)
        feature_values = packet_features.iloc[0].values

        contributions = []
        for i, (name, value, shap_val) in enumerate(
            zip(feature_names, feature_values, class_shap)
        ):
            contributions.append({
                "feature": name,
                "value": round(float(value), 4),
                "shap_contribution": round(float(shap_val), 4),
                "direction": "toward_attack" if shap_val > 0 else "toward_benign"
            })

        # Sort by absolute SHAP value — biggest contributors first
        contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
        top_features = contributions[:top_n]

        # Build all class probabilities dict
        all_class_probs = {
            self.class_names[i]: round(float(probs[0][i]), 4)
            for i in range(len(self.class_names))
        }

        # Human-readable verdict
        verdict = self._build_verdict(predicted_class, confidence, top_features)

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_class_probs": all_class_probs,
            "top_features": top_features,
            "verdict": verdict
        }
    
    def explain_batch(self, packets_df: pd.DataFrame, top_n: int = 5) -> list:
        """
        Explains multiple packets at once.
        """
        probs = self.lgbm.predict(packets_df)
        shap_values = self.explainer.shap_values(packets_df)

        results = []
        feature_names = list(packets_df.columns)

        for row_idx in range(len(packets_df)):
            predicted_idx = int(np.argmax(probs[row_idx]))
            predicted_class = self.class_names[predicted_idx]
            confidence = float(probs[row_idx][predicted_idx])

            # === FIX: Handle batch list vs 3D Array formats ===
            if isinstance(shap_values, list):
                class_shap = shap_values[predicted_idx][row_idx]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Pull [current_packet, all_features, target_class]
                class_shap = shap_values[row_idx, :, predicted_idx]
            else:
                class_shap = shap_values[row_idx]
            # ===================================================

            feature_values = packets_df.iloc[row_idx].values

            contributions = []
            for name, value, shap_val in zip(feature_names, feature_values, class_shap):
                contributions.append({
                    "feature": name,
                    "value": round(float(value), 4),
                    "shap_contribution": round(float(shap_val), 4),
                    "direction": "toward_attack" if shap_val > 0 else "toward_benign"
                })

            contributions.sort(
                key=lambda x: abs(x["shap_contribution"]), reverse=True
            )

            all_class_probs = {
                self.class_names[i]: round(float(probs[row_idx][i]), 4)
                for i in range(len(self.class_names))
            }

            verdict = self._build_verdict(
                predicted_class, confidence, contributions[:top_n]
            )

            results.append({
                "predicted_class": predicted_class,
                "confidence": round(confidence, 4),
                "all_class_probs": all_class_probs,
                "top_features": contributions[:top_n],
                "verdict": verdict
            })

        return results

    def _build_verdict(
        self, predicted_class: str, confidence: float, top_features: list
    ) -> str:
        """
        Builds a human-readable explanation string from SHAP values.
        This is what goes into the API response for the analyst to read.
        """
        if predicted_class == "BENIGN":
            return f"Traffic classified as normal with {confidence:.1%} confidence."

        # Find the top positive contributor (most attack-like feature)
        attack_drivers = [
            f for f in top_features if f["direction"] == "toward_attack"
        ]
        benign_drivers = [
            f for f in top_features if f["direction"] == "toward_benign"
        ]

        if not attack_drivers:
            return (
                f"Classified as {predicted_class} with {confidence:.1%} confidence. "
                f"No dominant attack features identified."
            )

        top_driver = attack_drivers[0]
        verdict = (
            f"Classified as {predicted_class} with {confidence:.1%} confidence. "
            f"Primary indicator: '{top_driver['feature']}' "
            f"(value={top_driver['value']}, "
            f"contribution={top_driver['shap_contribution']:+.3f})"
        )

        if len(attack_drivers) > 1:
            second = attack_drivers[1]
            verdict += (
                f", supported by '{second['feature']}' "
                f"(contribution={second['shap_contribution']:+.3f})"
            )

        if benign_drivers:
            top_benign = benign_drivers[0]
            verdict += (
                f". Mitigating factor: '{top_benign['feature']}' "
                f"suggests partial normal behavior "
                f"(contribution={top_benign['shap_contribution']:+.3f})"
            )

        verdict += "."
        return verdict




# ── SMOKE TEST ── #

if __name__ == "__main__":
    import sys
    from preprocess import preprocess_chronological, scale_features

    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent
    ARTIFACTS_DIR = ROOT_DIR / "artifacts"
    DATA_DIR = ROOT_DIR / "data" / "MachineLearningCVE"

    print("=" * 54)
    print("  SHAP EXPLAINER SMOKE TEST")
    print("=" * 54)

    # Load a small sample of test data
    print("\n[TEST] Loading test data sample...")
    X_train, X_test, y_train_raw, y_test_raw = preprocess_chronological(
        data_dir_path=DATA_DIR,
        sample_size=10000
    )
    _, X_test_scaled = scale_features(X_train, X_test)

    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    y_test = label_encoder.transform(y_test_raw)
    benign_idx = int(np.where(label_encoder.classes_ == "BENIGN")[0][0])

    # Initialize explainer
    explainer = ThreatExplainer(ARTIFACTS_DIR)

    # Find one attack packet to explain
    lgbm = joblib.load(ARTIFACTS_DIR / "tier1_lightgbm.pkl")
    probs = lgbm.predict(X_test_scaled)
    predictions = np.argmax(probs, axis=1)

    # Find a packet predicted as attack
    attack_mask = predictions != benign_idx
    attack_indices = np.where(attack_mask)[0]

    if len(attack_indices) == 0:
        print("[TEST] No attack packets found in sample. Try larger sample_size.")
        sys.exit(1)

    # Explain the first flagged packet
    test_idx = attack_indices[0]
    packet = X_test_scaled.iloc[[test_idx]]
    true_label = y_test_raw.iloc[test_idx]

    print(f"\n[TEST] Explaining packet #{test_idx}")
    print(f"  True label    : {true_label}")

    explanation = explainer.explain_packet(packet, top_n=5)

    print(f"\n  Predicted     : {explanation['predicted_class']}")
    print(f"  Confidence    : {explanation['confidence']:.1%}")
    print(f"\n  Verdict: {explanation['verdict']}")
    print(f"\n  Top 5 contributing features:")
    for i, feat in enumerate(explanation['top_features'], 1):
        direction = "↑ attack" if feat['direction'] == 'toward_attack' else "↓ benign"
        print(
            f"    {i}. {feat['feature']:<35} "
            f"value={feat['value']:>10.3f}  "
            f"SHAP={feat['shap_contribution']:>+8.4f}  {direction}"
        )

    print(f"\n  All class probabilities:")
    sorted_probs = sorted(
        explanation['all_class_probs'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for cls, prob in sorted_probs[:5]:
        bar = "#" * int(prob * 40)
        print(f"    {cls:<35} {prob:.4f}  {bar}")

    print("\n[SUCCESS] SHAP explainer is working correctly.")