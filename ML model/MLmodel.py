# MLmodel.py - Train and Save Iris Classification Model

# Import necessary libraries for machine learning and data handling
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm
from sklearn.datasets import load_iris              # Built-in Iris dataset
from sklearn.model_selection import train_test_split # For splitting data into train/test sets
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics
import joblib  # For saving and loading trained models

def train_iris_model():
    """
    Train a Random Forest model on the Iris dataset and save it to disk.
    
    This function performs the complete machine learning pipeline:
    1. Loads the Iris dataset
    2. Splits data into training and testing sets
    3. Trains a Random Forest classifier
    4. Evaluates model performance
    5. Saves the trained model for future use
    
    Returns:
        tuple: (trained_model, accuracy_score)
    """
   
    # Step 1: Load the Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    
    # Extract features (X) and target labels (y)
    X = iris.data    # Features: sepal_length, sepal_width, petal_length, petal_width (4 features)
    y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)
   
    # Step 2: Split the data into training and testing sets
    # Use 80% for training, 20% for testing
    # stratify=y ensures each class is proportionally represented in both sets
    # random_state=42 ensures reproducible results
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% of data for testing
        random_state=42,    # Seed for reproducibility
        stratify=y          # Maintain class distribution in splits
    )
   
    # Display information about the data split
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
   
    # Step 3: Create and configure the Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,  # Number of decision trees in the forest (more trees = better performance but slower)
        random_state=42,   # Seed for reproducible results
        max_depth=3        # Maximum depth of each tree (prevents overfitting)
    )
   
    # Step 4: Train the model on the training data
    # The model learns patterns from X_train to predict y_train
    model.fit(X_train, y_train)
   
    # Step 5: Make predictions on the test set
    # Use the trained model to predict species for unseen test data
    y_pred = model.predict(X_test)
   
    # Step 6: Evaluate model performance
    # Calculate overall accuracy (percentage of correct predictions)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
   
    # Print detailed classification report showing precision, recall, and F1-score for each class
    target_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
   
    # Step 7: Analyze feature importance
    # Shows which features (sepal/petal length/width) are most important for classification
    feature_names = iris.feature_names
    feature_importance = model.feature_importances_
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.3f}")
   
    # Step 8: Save the trained model to disk for future use
    print("\nSaving model to 'iris_model.pkl'...")
    joblib.dump(model, 'iris_model.pkl')  # Save the trained model
   
    # Also save the target names for easy reference when making predictions
    joblib.dump(target_names, 'iris_target_names.pkl')
   
    print("Model saved successfully!")
   
    # Step 9: Test model loading functionality
    # Verify that the saved model can be loaded and used for predictions
    print("\nTesting model loading...")
    loaded_model = joblib.load('iris_model.pkl')
    loaded_target_names = joblib.load('iris_target_names.pkl')
   
    # Make a sample prediction to verify everything works
    # These measurements are typical for a setosa flower
    sample_features = [[5.1, 3.5, 1.4, 0.2]]  # [sepal_length, sepal_width, petal_length, petal_width]
    prediction = loaded_model.predict(sample_features)
    predicted_species = loaded_target_names[prediction[0]]  # Convert numeric prediction to species name
   
    print(f"Sample prediction: {predicted_species}")
    print("Model loading test successful!")
   
    # Return the trained model and its accuracy for further use
    return model, accuracy

# Main execution block - runs only when script is executed directly (not imported)
if __name__ == "__main__":
    # Train the model and get results
    model, accuracy = train_iris_model()
   
    # Display final summary with emojis for better visual appeal
    print(f"\n🎉 Model training complete!")
    print(f"📊 Final accuracy: {accuracy*100:.1f}%")
    print(f"💾 Model saved as 'iris_model.pkl'")
    print(f"🚀 Ready for FastAPI deployment!")