import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def automate_preprocessing(input_path="../heart.csv", output_dir="heart_preprocessing"):

    print("=== Automate Preprocessing Started ===")

    # 1. Load Dataset
    df = pd.read_csv(input_path)
    print("Dataset Loaded:", df.shape)

    target = "HeartDisease"

    # 2. Tentukan Fitur

    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    binary_features = ['FastingBS']  # tidak discaling
    categorical_features = [
        'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
    ]

    X = df[numeric_features + binary_features + categorical_features]
    y = df[target]

    # 3. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    # fit → transform
    X_processed = preprocessor.fit_transform(X)

    # 4. Ambil nama fitur hasil encoding

    cat_encoded = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

    processed_cols = numeric_features + list(cat_encoded) + binary_features

    X_processed_df = pd.DataFrame(X_processed, columns=processed_cols)

    # 5. Train–Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Simpan ke folder output

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Files saved in:", output_dir)
    print(" - X_train.csv")
    print(" - X_test.csv")
    print(" - y_train.csv")
    print(" - y_test.csv")

    print("=== Automate Preprocessing Completed ===")


if __name__ == "__main__":
    automate_preprocessing()
