
"""
Complete Machine Learning Model Implementation
Demonstrates classification and regression with multiple algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, mean_squared_error, r2_score)
from sklearn.datasets import make_classification, make_regression, load_iris
import warnings
warnings.filterwarnings('ignore')

class MLModelImplementation:
    """
    Complete Machine Learning Model Implementation Class
    Supports both classification and regression tasks
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def generate_sample_data(self, task_type='classification', n_samples=1000):
        """Generate sample data for demonstration"""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples, 
                n_features=10, 
                n_informative=5,
                n_redundant=2,
                n_clusters_per_class=1,
                random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        elif task_type == 'regression':
            X, y = make_regression(
                n_samples=n_samples,
                n_features=10,
                n_informative=5,
                noise=0.1,
                random_state=42
            )
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        return pd.DataFrame(X, columns=feature_names), pd.Series(y, name='target')
    
    def load_iris_data(self):
        """Load the classic Iris dataset"""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name='species')
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, scale_features=True):
        """Preprocess the data: split and scale"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['standard'] = scaler
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def train_classification_models(self, X_train, y_train):
        """Train multiple classification models"""
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        self.models['classification'] = trained_models
        return trained_models
    
    def train_regression_models(self, X_train, y_train):
        """Train multiple regression models"""
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVR': SVR(kernel='rbf')
        }
        
        # Train models
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            
        self.models['regression'] = trained_models
        return trained_models
    
    def evaluate_classification(self, models, X_test, y_test):
        """Evaluate classification models"""
        results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_prob,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
        
        self.results['classification'] = results
        return results
    
    def evaluate_regression(self, models, X_test, y_test):
        """Evaluate regression models"""
        results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"\n{name} Results:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
        
        self.results['regression'] = results
        return results
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("Performing hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def plot_results(self, task_type='classification'):
        """Plot model comparison results"""
        if task_type == 'classification' and 'classification' in self.results:
            results = self.results['classification']
            
            # Accuracy comparison
            accuracies = [results[model]['accuracy'] for model in results.keys()]
            model_names = list(results.keys())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.title('Classification Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
        elif task_type == 'regression' and 'regression' in self.results:
            results = self.results['regression']
            
            # R² comparison
            r2_scores = [results[model]['r2'] for model in results.keys()]
            model_names = list(results.keys())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.title('Regression Model R² Score Comparison')
            plt.ylabel('R² Score')
            
            # Add value labels on bars
            for bar, r2 in zip(bars, r2_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{r2:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
    
    def feature_importance_analysis(self, model_name='Random Forest', task_type='classification'):
        """Analyze feature importance for tree-based models"""
        if task_type in self.models and model_name in self.models[task_type]:
            model = self.models[task_type][model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importance - {model_name}')
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                plt.tight_layout()
                plt.show()
                
                # Print top features
                print(f"\nTop 5 Most Important Features ({model_name}):")
                for i in range(min(5, len(importances))):
                    idx = indices[i]
                    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def main():
    """Main function to demonstrate the ML implementation"""
    print("=== Machine Learning Model Implementation Demo ===\n")
    
    # Initialize the ML implementation
    ml = MLModelImplementation()
    
    # === CLASSIFICATION EXAMPLE ===
    print("1. CLASSIFICATION EXAMPLE")
    print("-" * 40)
    
    # Generate or load data
    print("Loading Iris dataset...")
    X, y = ml.load_iris_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = ml.preprocess_data(X, y, scale_features=True)
    
    # Train models
    classification_models = ml.train_classification_models(X_train, y_train)
    
    # Evaluate models
    classification_results = ml.evaluate_classification(classification_models, X_test, y_test)
    
    # Plot results
    ml.plot_results('classification')
    
    # Feature importance
    ml.feature_importance_analysis('Random Forest', 'classification')
    
    # === REGRESSION EXAMPLE ===
    print("\n\n2. REGRESSION EXAMPLE")
    print("-" * 40)
    
    # Generate regression data
    print("Generating synthetic regression dataset...")
    X_reg, y_reg = ml.generate_sample_data('regression', n_samples=800)
    print(f"Dataset shape: {X_reg.shape}")
    
    # Preprocess data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = ml.preprocess_data(
        X_reg, y_reg, scale_features=True
    )
    
    # Train models
    regression_models = ml.train_regression_models(X_train_reg, y_train_reg)
    
    # Evaluate models
    regression_results = ml.evaluate_regression(regression_models, X_test_reg, y_test_reg)
    
    # Plot results
    ml.plot_results('regression')
    
    # Feature importance
    ml.feature_importance_analysis('Random Forest', 'regression')
    
    # === HYPERPARAMETER TUNING EXAMPLE ===
    print("\n\n3. HYPERPARAMETER TUNING EXAMPLE")
    print("-" * 40)
    
    # Example with Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    best_rf = ml.hyperparameter_tuning(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        X_train, y_train
    )
    
    # Evaluate tuned model
    y_pred_tuned = best_rf.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    print(f"Tuned Random Forest Accuracy: {tuned_accuracy:.4f}")
    
    print("\n=== Demo Complete! ===")
    
    return ml


if __name__ == "__main__":
    # Run the demonstration
    ml_implementation = main()
    
    # Example of making predictions on new data
    print("\n4. MAKING PREDICTIONS ON NEW DATA")
    print("-" * 40)
    
    # Create some sample new data (using Iris features)
    new_data = np.array([[5.1, 3.5, 1.4, 0.2],  # Should be Setosa
                        [6.2, 2.8, 4.8, 1.8],   # Should be Virginica
                        [5.7, 2.8, 4.1, 1.3]])  # Should be Versicolor
    
    # Scale the new data using the fitted scaler
    if 'standard' in ml_implementation.scalers:
        new_data_scaled = ml_implementation.scalers['standard'].transform(new_data)
        
        # Make predictions with the best model
        best_model = ml_implementation.models['classification']['Random Forest']
        predictions = best_model.predict(new_data_scaled)
        probabilities = best_model.predict_proba(new_data_scaled)
        
        print("Predictions on new data:")
        iris_names = ['Setosa', 'Versicolor', 'Virginica']
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Sample {i+1}: Predicted class = {iris_names[pred]} "
                  f"(confidence: {max(prob):.3f})")


def plot_results(self, task_type='classification'):
    """Plot model comparison results"""
    if task_type == 'classification' and 'classification' in self.results:
        results = self.results['classification']
        
        # Accuracy comparison
        accuracies = [results[model]['accuracy'] for model in results.keys()]
        model_names = list(results.keys())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Classification Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('classification_accuracy_comparison.png')  # Save the figure
        plt.show()

