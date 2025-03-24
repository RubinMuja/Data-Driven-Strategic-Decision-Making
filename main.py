from src.load_data import load_users
from src.features import extract_features
from src.model import train_model
from src.evaluate import evaluate

from sklearn.model_selection import train_test_split


def main():
    print("\n📥 Step 1: Loading data...")
    df = load_users()
    print(f"✅ Loaded {len(df)} users.")

    print("\n🔍 Step 2: Checking available columns and labels...")
    print("Columns:", df.columns.tolist())

    if 'label' not in df.columns:
        print("❌ ERROR: 'label' column not found in the dataset.")
        return

    print("Unique label values:", df['label'].unique())

    # 👇 Adjust this line if your labels are numbers or different strings
    df = df[df['label'].isin(['bot', 'human', 'Bot', 'Human', 0, 1])]
    print(df['label'].unique())

    if len(df) == 0:
        print("❌ ERROR: No usable data after filtering by label.")
        return

    print("\n🎯 Step 3: Mapping labels to numbers...")
    label_map = {
        'bot': 1, 'Bot': 1, 1: 1,
        'human': 0, 'Human': 0, 0: 0
    }
    df['label'] = df['label'].map(label_map)

    print("\n🔨 Step 4: Extracting features...")
    X = extract_features(df)
    y = df['label']
    print("✅ Sample features:")
    print(X.head())

    if X.empty or y.empty:
        print("❌ ERROR: Feature matrix or labels are empty.")
        return

    print("\n🧪 Step 5: Splitting into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ValueError as e:
        print(f"❌ ERROR: Could not split the data — {e}")
        return

    print(f"✅ Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("\n🤖 Step 6: Training the model...")
    model = train_model(X_train, y_train)
    print("✅ Model trained.")

    print("\n📊 Step 7: Evaluating the model...")
    evaluate(model, X_test, y_test)


if __name__ == '__main__':
    main()
