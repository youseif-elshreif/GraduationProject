# AI Track Documentation - IDS-AI System

## Table of Contents

1. [Overview](#overview)
2. [AI Architecture](#ai-architecture)
3. [Dataset & Feature Engineering](#dataset--feature-engineering)
4. [Model Development](#model-development)
5. [Training Pipeline](#training-pipeline)
6. [Model Evaluation](#model-evaluation)
7. [Real-time Inference](#real-time-inference)
8. [Model Optimization](#model-optimization)
9. [Continuous Learning](#continuous-learning)
10. [Performance Monitoring](#performance-monitoring)
11. [Model Deployment](#model-deployment)
12. [Security & Robustness](#security--robustness)
13. [Interpretability & Explainability](#interpretability--explainability)
14. [Research & Innovation](#research--innovation)

## Overview

The AI component is the core intelligence of the IDS-AI system, responsible for analyzing network flow features and accurately detecting various types of cyber attacks. The system employs state-of-the-art machine learning techniques to provide real-time threat detection with high accuracy and low false positive rates.

### Key Objectives

- **High Accuracy**: Achieve >95% accuracy in attack detection
- **Low False Positives**: Minimize false alarms to <2%
- **Real-time Processing**: Process network flows with <100ms latency
- **Scalability**: Handle thousands of flows per second
- **Adaptability**: Continuously learn from new attack patterns
- **Interpretability**: Provide explanations for detection decisions

### AI Capabilities

- **Multi-class Classification**: Detect various attack types (DDoS, Port Scan, Brute Force, etc.)
- **Anomaly Detection**: Identify previously unknown attack patterns
- **Feature Engineering**: Extract meaningful patterns from network data
- **Real-time Inference**: Process live network traffic streams
- **Model Updates**: Adapt to evolving threat landscapes

## AI Architecture

### System Architecture Overview

```python
# AI System Architecture
┌─────────────────────────────────────────────────────────────┐
│                   AI Processing Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Data Ingestion│  │ Preprocessing │  │ Feature Eng.  │   │
│  │   - Raw Flow  │─▶│  - Cleaning   │─▶│ - Extraction  │   │
│  │   - Streaming │  │  - Validation │  │ - Selection   │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Model Serving │  │ Inference     │  │ Post-process  │   │
│  │ - Load Model  │─▶│ - Prediction  │─▶│ - Confidence  │   │
│  │ - Versioning  │  │ - Probability │  │ - Explanation │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Model Training│  │ Evaluation    │  │ Deployment    │   │
│  │ - Supervised  │  │ - Metrics     │  │ - A/B Testing │   │
│  │ - Unsupervised│  │ - Validation  │  │ - Monitoring  │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

```python
# ai/core/architecture.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]

class BasePreprocessor(ABC):
    """Base class for data preprocessing"""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BasePreprocessor':
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class BaseModel(ABC):
    """Base class for AI models"""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        pass

class BaseEvaluator(ABC):
    """Base class for model evaluation"""

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_pred_proba: np.ndarray) -> Dict[str, float]:
        pass
```

## Dataset & Feature Engineering

### Network Flow Features (80 → 60)

```python
# ai/features/flow_features.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

class FlowFeatureExtractor:
    """Extract and process network flow features"""

    def __init__(self):
        self.feature_names = self._get_feature_names()
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None

    def _get_feature_names(self) -> List[str]:
        """Define all 80 network flow features"""
        return [
            # Duration Features
            'flow_duration', 'flow_bytes_s', 'flow_packets_s',

            # Forward Packet Features
            'tot_fwd_pkts', 'totlen_fwd_pkts', 'fwd_pkt_len_max',
            'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',

            # Backward Packet Features
            'tot_bwd_pkts', 'totlen_bwd_pkts', 'bwd_pkt_len_max',
            'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std',

            # Flow Inter-Arrival Time Features
            'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',

            # Forward Inter-Arrival Time Features
            'fwd_iat_tot', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',

            # Backward Inter-Arrival Time Features
            'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min',

            # Flag Features
            'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
            'fwd_header_len', 'bwd_header_len',

            # Packet Length Features
            'pkt_len_min', 'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',

            # Flag Count Features
            'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
            'ack_flag_cnt', 'urg_flag_cnt', 'cwe_flag_count', 'ece_flag_cnt',

            # Additional Flow Features
            'down_up_ratio', 'pkt_size_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg',
            'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'fwd_blk_rate_avg',
            'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'bwd_blk_rate_avg',
            'subflow_fwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts',

            # Window Size Features
            'init_win_bytes_forward', 'init_win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward',

            # Additional Statistical Features
            'active_mean', 'active_std', 'active_max', 'active_min',
            'idle_mean', 'idle_std', 'idle_max', 'idle_min'
        ]

    def extract_features(self, flow_data: Dict) -> np.ndarray:
        """Extract features from raw flow data"""
        features = np.zeros(len(self.feature_names))

        try:
            # Basic flow information
            features[0] = flow_data.get('flow_duration', 0)  # flow_duration

            # Calculate rates
            duration = max(flow_data.get('flow_duration', 1), 0.001)
            total_bytes = flow_data.get('total_bytes', 0)
            total_packets = flow_data.get('total_packets', 0)

            features[1] = total_bytes / duration  # flow_bytes_s
            features[2] = total_packets / duration  # flow_packets_s

            # Forward packet statistics
            fwd_packets = flow_data.get('fwd_packets', [])
            if fwd_packets:
                features[3] = len(fwd_packets)  # tot_fwd_pkts
                features[4] = sum(fwd_packets)  # totlen_fwd_pkts
                features[5] = max(fwd_packets)  # fwd_pkt_len_max
                features[6] = min(fwd_packets)  # fwd_pkt_len_min
                features[7] = np.mean(fwd_packets)  # fwd_pkt_len_mean
                features[8] = np.std(fwd_packets)  # fwd_pkt_len_std

            # Backward packet statistics
            bwd_packets = flow_data.get('bwd_packets', [])
            if bwd_packets:
                features[9] = len(bwd_packets)  # tot_bwd_pkts
                features[10] = sum(bwd_packets)  # totlen_bwd_pkts
                features[11] = max(bwd_packets)  # bwd_pkt_len_max
                features[12] = min(bwd_packets)  # bwd_pkt_len_min
                features[13] = np.mean(bwd_packets)  # bwd_pkt_len_mean
                features[14] = np.std(bwd_packets)  # bwd_pkt_len_std

            # Inter-arrival time features
            iat_times = flow_data.get('inter_arrival_times', [])
            if iat_times:
                features[15] = np.mean(iat_times)  # flow_iat_mean
                features[16] = np.std(iat_times)  # flow_iat_std
                features[17] = max(iat_times)  # flow_iat_max
                features[18] = min(iat_times)  # flow_iat_min

            # TCP flags
            tcp_flags = flow_data.get('tcp_flags', {})
            flag_names = ['fin', 'syn', 'rst', 'psh', 'ack', 'urg', 'cwe', 'ece']
            for i, flag in enumerate(flag_names):
                features[40 + i] = tcp_flags.get(flag, 0)

            # Additional derived features
            if total_packets > 0:
                features[48] = len(bwd_packets) / total_packets  # down_up_ratio
                features[49] = total_bytes / total_packets  # pkt_size_avg

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return features

    def select_features(self, X: pd.DataFrame, y: pd.DataFrame,
                       n_features: int = 60) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using statistical tests"""

        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]

        self.feature_selector = selector
        self.selected_features = selected_features

        return pd.DataFrame(X_selected, columns=selected_features), selected_features

    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess features with scaling and normalization"""

        if fit:
            # Use RobustScaler to handle outliers
            self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.feature_selector is None:
            return {}

        scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)

        importance_dict = {}
        for i, idx in enumerate(selected_indices):
            feature_name = self.feature_names[idx]
            importance_dict[feature_name] = scores[idx]

        return importance_dict
```

### Data Preprocessing Pipeline

```python
# ai/preprocessing/pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class DataPreprocessingPipeline:
    """Complete data preprocessing pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_extractor = FlowFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

        # Attack type mapping
        self.attack_mapping = {
            'BENIGN': 0,
            'DDoS': 1,
            'PortScan': 2,
            'BruteForce': 3,
            'WebAttack': 4,
            'Infiltration': 5,
            'Botnet': 6,
            'DoS': 7
        }

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load and validate dataset"""
        try:
            # Load dataset (CSV format expected)
            df = pd.read_csv(dataset_path)

            print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")

            # Basic validation
            if 'Label' not in df.columns:
                raise ValueError("Dataset must contain 'Label' column")

            # Handle missing values
            df = self._handle_missing_values(df)

            # Remove duplicates
            initial_size = len(df)
            df = df.drop_duplicates()
            print(f"Removed {initial_size - len(df)} duplicate rows")

            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset"""

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Found {missing_counts.sum()} missing values")

            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)

            # For numerical columns, use median imputation
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col != 'Label':
                    df[col] = df[col].fillna(df[col].median())

            # For categorical columns, use mode imputation
            categorical_cols = df.select_dtypes(include=[object]).columns
            for col in categorical_cols:
                if col != 'Label':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

        return df

    def prepare_features_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training"""

        # Separate features and labels
        if 'Label' in df.columns:
            y = df['Label']
            X = df.drop('Label', axis=1)
        else:
            raise ValueError("No 'Label' column found in dataset")

        # Encode labels
        if not self.is_fitted:
            y_encoded = self.label_encoder.fit_transform(y)
            self.is_fitted = True
        else:
            y_encoded = self.label_encoder.transform(y)

        # Select and preprocess features
        if not self.feature_extractor.selected_features:
            X_selected, selected_features = self.feature_extractor.select_features(X, y_encoded)
        else:
            X_selected = X[self.feature_extractor.selected_features]

        X_processed = self.feature_extractor.preprocess_features(X_selected, fit=not self.is_fitted)

        return X_processed, pd.Series(y_encoded)

    def split_dataset(self, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, Any]:
        """Split dataset into training, validation, and test sets"""

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: separate validation set from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )

        # Print dataset statistics
        print(f"Dataset split:")
        print(f"  Training: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

        # Check class distribution
        print(f"\nClass distribution in training set:")
        train_dist = pd.Series(y_train).value_counts().sort_index()
        for class_idx, count in train_dist.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"  {class_name}: {count} samples ({count/len(y_train)*100:.1f}%)")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def save_preprocessing_pipeline(self, path: str):
        """Save preprocessing pipeline"""
        pipeline_data = {
            'feature_extractor': self.feature_extractor,
            'label_encoder': self.label_encoder,
            'attack_mapping': self.attack_mapping,
            'config': self.config
        }

        joblib.dump(pipeline_data, path)
        print(f"Preprocessing pipeline saved to {path}")

    def load_preprocessing_pipeline(self, path: str):
        """Load preprocessing pipeline"""
        pipeline_data = joblib.load(path)

        self.feature_extractor = pipeline_data['feature_extractor']
        self.label_encoder = pipeline_data['label_encoder']
        self.attack_mapping = pipeline_data['attack_mapping']
        self.config = pipeline_data['config']
        self.is_fitted = True

        print(f"Preprocessing pipeline loaded from {path}")
```

## Model Development

### Ensemble Model Architecture

```python
# ai/models/ensemble_model.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class EnsembleIDSModel:
    """Ensemble model for intrusion detection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.weights = {}
        self.is_trained = False

        # Initialize base models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize base models with optimized hyperparameters"""

        # Random Forest - Good for feature importance and robustness
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

        # Gradient Boosting - Good for complex patterns
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )

        # Support Vector Machine - Good for high-dimensional data
        self.models['svm'] = SVC(
            C=10.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )

        # Multi-layer Perceptron - Good for non-linear patterns
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )

        # Logistic Regression - Good baseline and interpretability
        self.models['logistic'] = LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            multi_class='ovr'
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train ensemble model"""

        print("Training ensemble model...")
        training_results = {}

        # Train each base model
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")

            try:
                # Train model
                model.fit(X_train, y_train)

                # Evaluate on validation set
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)

                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, average='weighted'),
                    'recall': recall_score(y_val, y_pred, average='weighted'),
                    'f1': f1_score(y_val, y_pred, average='weighted')
                }

                training_results[model_name] = metrics
                print(f"  {model_name} validation accuracy: {metrics['accuracy']:.4f}")

            except Exception as e:
                print(f"  Error training {model_name}: {e}")
                # Remove failed model
                del self.models[model_name]

        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(training_results)

        self.is_trained = True
        print("Ensemble training completed")

        return training_results

    def _calculate_ensemble_weights(self, training_results: Dict[str, Dict[str, float]]):
        """Calculate weights for ensemble voting"""

        # Use F1 score for weight calculation
        f1_scores = {name: results['f1'] for name, results in training_results.items()}

        # Normalize weights
        total_f1 = sum(f1_scores.values())
        self.weights = {name: score / total_f1 for name, score in f1_scores.items()}

        print("Ensemble weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble voting"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict_proba(X)

        # Weighted ensemble voting
        ensemble_proba = np.zeros_like(predictions[list(predictions.keys())[0]])

        for model_name, proba in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_proba += weight * proba

        # Return class with highest probability
        return np.argmax(ensemble_proba, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict_proba(X)

        # Weighted ensemble voting
        ensemble_proba = np.zeros_like(predictions[list(predictions.keys())[0]])

        for model_name, proba in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_proba += weight * proba

        return ensemble_proba

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from tree-based models"""
        importance_dict = {}

        # Get importance from Random Forest
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            for i, importance in enumerate(rf_importance):
                feature_name = f"feature_{i}"
                importance_dict[feature_name] = importance

        return importance_dict

    def save_model(self, path: str):
        """Save trained ensemble model"""
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'config': self.config,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, path)
        print(f"Ensemble model saved to {path}")

    def load_model(self, path: str):
        """Load trained ensemble model"""
        model_data = joblib.load(path)

        self.models = model_data['models']
        self.weights = model_data['weights']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']

        print(f"Ensemble model loaded from {path}")
```

### Deep Learning Model

```python
# ai/models/deep_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class DeepIDSModel:
    """Deep learning model for intrusion detection"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None
        self.is_trained = False

    def build_model(self, input_shape: Tuple[int], num_classes: int):
        """Build deep neural network architecture"""

        model = keras.Sequential([
            # Input layer
            layers.Input(shape=input_shape),

            # First hidden layer with dropout
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Second hidden layer
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Third hidden layer
            layers.Dense(64, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),

            # Fourth hidden layer
            layers.Dense(32, activation='relu', name='dense_4'),
            layers.Dropout(0.1),

            # Output layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 256) -> Dict[str, Any]:
        """Train deep learning model"""

        if self.model is None:
            input_shape = (X_train.shape[1],)
            num_classes = len(np.unique(y_train))
            self.build_model(input_shape, num_classes)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True

        # Return training history
        return {
            'history': self.history.history,
            'final_accuracy': self.history.history['val_accuracy'][-1],
            'final_loss': self.history.history['val_loss'][-1]
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError("Model not trained")

        results = self.model.evaluate(X_test, y_test, verbose=0)

        return {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_precision': results[2],
            'test_recall': results[3]
        }

    def save_model(self, path: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(path)
            print(f"Deep learning model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        self.is_trained = True
        print(f"Deep learning model loaded from {path}")
```

## Training Pipeline

### Complete Training Pipeline

```python
# ai/training/pipeline.py
import os
import time
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingPipeline:
    """Complete ML training pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessing_pipeline = None
        self.model = None
        self.results = {}

    def run_training(self, dataset_path: str, model_type: str = 'ensemble') -> Dict[str, Any]:
        """Run complete training pipeline"""

        print("Starting IDS-AI training pipeline...")
        start_time = time.time()

        try:
            # Step 1: Data preprocessing
            print("\n1. Data Preprocessing")
            self.preprocessing_pipeline = DataPreprocessingPipeline(self.config)

            # Load dataset
            df = self.preprocessing_pipeline.load_dataset(dataset_path)

            # Prepare features and labels
            X, y = self.preprocessing_pipeline.prepare_features_labels(df)

            # Split dataset
            data_splits = self.preprocessing_pipeline.split_dataset(X, y)

            # Step 2: Model training
            print("\n2. Model Training")
            if model_type == 'ensemble':
                self.model = EnsembleIDSModel(self.config)
            elif model_type == 'deep':
                self.model = DeepIDSModel(self.config)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Train model
            training_results = self.model.train(
                data_splits['X_train'], data_splits['y_train'],
                data_splits['X_val'], data_splits['y_val']
            )

            # Step 3: Model evaluation
            print("\n3. Model Evaluation")
            evaluation_results = self._evaluate_model(data_splits)

            # Step 4: Save results
            print("\n4. Saving Results")
            self._save_training_results(training_results, evaluation_results)

            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time:.2f} seconds")

            return {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'training_time': total_time
            }

        except Exception as e:
            print(f"Training pipeline error: {e}")
            raise

    def _evaluate_model(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model evaluation"""

        results = {}

        # Evaluate on test set
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, average='weighted')
        results['recall'] = recall_score(y_test, y_pred, average='weighted')
        results['f1'] = f1_score(y_test, y_pred, average='weighted')

        # Per-class metrics
        results['classification_report'] = classification_report(
            y_test, y_pred,
            target_names=self.preprocessing_pipeline.label_encoder.classes_,
            output_dict=True
        )

        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            from sklearn.metrics import roc_auc_score
            results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])

        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test Precision: {results['precision']:.4f}")
        print(f"Test Recall: {results['recall']:.4f}")
        print(f"Test F1: {results['f1']:.4f}")

        return results

    def _save_training_results(self, training_results: Dict[str, Any],
                             evaluation_results: Dict[str, Any]):
        """Save training and evaluation results"""

        # Create results directory
        results_dir = self.config.get('results_dir', 'training_results')
        os.makedirs(results_dir, exist_ok=True)

        # Save preprocessing pipeline
        preprocessing_path = os.path.join(results_dir, 'preprocessing_pipeline.pkl')
        self.preprocessing_pipeline.save_preprocessing_pipeline(preprocessing_path)

        # Save trained model
        model_path = os.path.join(results_dir, 'trained_model.pkl')
        self.model.save_model(model_path)

        # Save results as JSON
        results_data = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'config': self.config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        results_path = os.path.join(results_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        # Generate visualizations
        self._generate_visualizations(evaluation_results, results_dir)

        print(f"Results saved to {results_dir}")

    def _generate_visualizations(self, evaluation_results: Dict[str, Any],
                               results_dir: str):
        """Generate training and evaluation visualizations"""

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(evaluation_results['confusion_matrix'])
        class_names = self.preprocessing_pipeline.label_encoder.classes_

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        plt.close()

        # Feature Importance (if available)
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
            if importance:
                plt.figure(figsize=(12, 8))
                features = list(importance.keys())[:20]  # Top 20 features
                values = [importance[f] for f in features]

                plt.barh(features, values)
                plt.title('Top 20 Feature Importance')
                plt.xlabel('Importance Score')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
                plt.close()

        # Class Distribution
        if 'classification_report' in evaluation_results:
            report = evaluation_results['classification_report']
            classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

            metrics = ['precision', 'recall', 'f1-score']

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, metric in enumerate(metrics):
                values = [report[cls][metric] for cls in classes]
                axes[i].bar(classes, values)
                axes[i].set_title(f'{metric.title()} by Class')
                axes[i].set_ylabel(metric.title())
                axes[i].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'class_metrics.png'))
            plt.close()
```

## Real-time Inference

### Inference Server

```python
# ai/inference/server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import asyncio
import time
import logging
from contextlib import asynccontextmanager

# Model and preprocessing imports
from ai.models.ensemble_model import EnsembleIDSModel
from ai.preprocessing.pipeline import DataPreprocessingPipeline

class FlowData(BaseModel):
    """Input schema for network flow data"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    features: Dict[str, float]
    timestamp: float

class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    flow_id: str
    prediction: str
    attack_type: str
    confidence: float
    risk_level: str
    processing_time_ms: float
    timestamp: float

class InferenceServer:
    """Real-time inference server for IDS-AI"""

    def __init__(self):
        self.model = None
        self.preprocessing_pipeline = None
        self.is_ready = False
        self.stats = {
            'total_requests': 0,
            'attack_detections': 0,
            'average_processing_time': 0.0,
            'last_request_time': 0.0
        }

        # Risk level mapping
        self.risk_levels = {
            'BENIGN': 'low',
            'DDoS': 'critical',
            'PortScan': 'medium',
            'BruteForce': 'high',
            'WebAttack': 'high',
            'Infiltration': 'critical',
            'Botnet': 'high',
            'DoS': 'critical'
        }

    async def load_model(self, model_path: str, preprocessing_path: str):
        """Load trained model and preprocessing pipeline"""
        try:
            print("Loading AI model and preprocessing pipeline...")

            # Load preprocessing pipeline
            self.preprocessing_pipeline = DataPreprocessingPipeline({})
            self.preprocessing_pipeline.load_preprocessing_pipeline(preprocessing_path)

            # Load trained model
            self.model = EnsembleIDSModel({})
            self.model.load_model(model_path)

            self.is_ready = True
            print("AI model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    async def predict_flow(self, flow_data: FlowData) -> PredictionResponse:
        """Make prediction for single network flow"""
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")

        start_time = time.time()

        try:
            # Extract features from flow data
            features_df = self._extract_features(flow_data)

            # Make prediction
            prediction_proba = self.model.predict_proba(features_df)
            prediction_class = np.argmax(prediction_proba[0])
            confidence = float(np.max(prediction_proba[0]))

            # Decode prediction
            attack_type = self.preprocessing_pipeline.label_encoder.inverse_transform([prediction_class])[0]

            # Determine if it's an attack
            is_attack = attack_type != 'BENIGN'
            prediction = 'attack' if is_attack else 'normal'

            # Get risk level
            risk_level = self.risk_levels.get(attack_type, 'medium')

            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, is_attack)

            return PredictionResponse(
                flow_id=flow_data.flow_id,
                prediction=prediction,
                attack_type=attack_type,
                confidence=confidence,
                risk_level=risk_level,
                processing_time_ms=processing_time,
                timestamp=time.time()
            )

        except Exception as e:
            print(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _extract_features(self, flow_data: FlowData) -> pd.DataFrame:
        """Extract features from flow data"""
        try:
            # Convert flow data to feature vector
            feature_vector = np.array([flow_data.features.get(feature, 0.0)
                                     for feature in self.preprocessing_pipeline.feature_extractor.selected_features])

            # Create DataFrame
            features_df = pd.DataFrame([feature_vector],
                                     columns=self.preprocessing_pipeline.feature_extractor.selected_features)

            # Apply preprocessing
            features_df = self.preprocessing_pipeline.feature_extractor.preprocess_features(features_df, fit=False)

            return features_df

        except Exception as e:
            print(f"Feature extraction error: {e}")
            raise

    def _update_stats(self, processing_time: float, is_attack: bool):
        """Update server statistics"""
        self.stats['total_requests'] += 1
        if is_attack:
            self.stats['attack_detections'] += 1

        # Update average processing time (exponential moving average)
        alpha = 0.1
        if self.stats['average_processing_time'] == 0:
            self.stats['average_processing_time'] = processing_time
        else:
            self.stats['average_processing_time'] = (
                alpha * processing_time +
                (1 - alpha) * self.stats['average_processing_time']
            )

        self.stats['last_request_time'] = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            **self.stats,
            'attack_rate': (self.stats['attack_detections'] / max(self.stats['total_requests'], 1)) * 100,
            'model_ready': self.is_ready
        }

# FastAPI app initialization
inference_server = InferenceServer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await inference_server.load_model(
        model_path="models/trained_model.pkl",
        preprocessing_path="models/preprocessing_pipeline.pkl"
    )
    yield
    # Shutdown
    print("Shutting down inference server")

app = FastAPI(
    title="IDS-AI Inference Server",
    description="Real-time network flow analysis and attack detection",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(flow_data: FlowData):
    """Predict attack for network flow"""
    return await inference_server.predict_flow(flow_data)

@app.post("/predict/batch")
async def predict_batch_endpoint(flows: List[FlowData]):
    """Batch prediction for multiple flows"""
    results = []
    for flow in flows:
        try:
            result = await inference_server.predict_flow(flow)
            results.append(result)
        except Exception as e:
            results.append({
                "flow_id": flow.flow_id,
                "error": str(e)
            })

    return {"predictions": results}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_server.is_ready else "not_ready",
        "timestamp": time.time()
    }

@app.get("/stats")
async def get_stats():
    """Get inference server statistics"""
    return inference_server.get_stats()

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not inference_server.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    return {
        "model_type": "ensemble",
        "features_count": len(inference_server.preprocessing_pipeline.feature_extractor.selected_features),
        "classes": inference_server.preprocessing_pipeline.label_encoder.classes_.tolist(),
        "risk_levels": inference_server.risk_levels
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### High-Performance Inference

```python
# ai/inference/batch_processor.py
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

class BatchInferenceProcessor:
    """High-performance batch processing for inference"""

    def __init__(self, model, preprocessing_pipeline,
                 batch_size: int = 100, max_workers: int = 4):
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline
        self.batch_size = batch_size
        self.max_workers = max_workers

        # Processing queues
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_queue = queue.Queue(maxsize=1000)

        # Worker threads
        self.workers = []
        self.running = False

        # Statistics
        self.stats = {
            'processed_flows': 0,
            'detected_attacks': 0,
            'batch_count': 0,
            'average_batch_time': 0.0,
            'queue_size': 0
        }

    def start_processing(self):
        """Start batch processing workers"""
        self.running = True

        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._batch_worker,
                name=f"BatchWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        print(f"Started {self.max_workers} batch processing workers")

    def submit_flow(self, flow_data: Dict[str, Any]) -> bool:
        """Submit flow for batch processing"""
        try:
            self.input_queue.put_nowait(flow_data)
            return True
        except queue.Full:
            return False

    def get_results(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Get processed results"""
        results = []
        end_time = time.time() + timeout

        while time.time() < end_time:
            try:
                result = self.output_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break

        return results

    def _batch_worker(self):
        """Batch processing worker thread"""
        while self.running:
            try:
                # Collect batch
                batch = self._collect_batch()

                if batch:
                    # Process batch
                    results = self._process_batch(batch)

                    # Put results in output queue
                    for result in results:
                        try:
                            self.output_queue.put_nowait(result)
                        except queue.Full:
                            # Drop result if queue is full
                            pass

                time.sleep(0.01)  # Small sleep to prevent busy waiting

            except Exception as e:
                print(f"Batch worker error: {e}")

    def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect flows for batch processing"""
        batch = []

        # Try to collect up to batch_size items
        for _ in range(self.batch_size):
            try:
                item = self.input_queue.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                break

        return batch

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of flows"""
        start_time = time.time()

        try:
            # Extract features for all flows in batch
            features_list = []
            flow_ids = []

            for flow_data in batch:
                features = self._extract_features_dict(flow_data['features'])
                features_list.append(features)
                flow_ids.append(flow_data['flow_id'])

            # Create DataFrame for batch processing
            features_df = pd.DataFrame(features_list)

            # Apply preprocessing
            features_df = self.preprocessing_pipeline.feature_extractor.preprocess_features(
                features_df, fit=False
            )

            # Batch prediction
            predictions_proba = self.model.predict_proba(features_df)
            predictions = np.argmax(predictions_proba, axis=1)

            # Process results
            results = []
            for i, (flow_id, pred_class, pred_proba) in enumerate(
                zip(flow_ids, predictions, predictions_proba)
            ):
                # Decode prediction
                attack_type = self.preprocessing_pipeline.label_encoder.inverse_transform([pred_class])[0]
                confidence = float(np.max(pred_proba))

                # Create result
                result = {
                    'flow_id': flow_id,
                    'prediction': 'attack' if attack_type != 'BENIGN' else 'normal',
                    'attack_type': attack_type,
                    'confidence': confidence,
                    'risk_level': self._get_risk_level(attack_type),
                    'timestamp': time.time()
                }

                results.append(result)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_batch_stats(len(batch), processing_time, results)

            return results

        except Exception as e:
            print(f"Batch processing error: {e}")
            return []

    def _extract_features_dict(self, features: Dict[str, float]) -> List[float]:
        """Extract features from dictionary"""
        return [features.get(feature, 0.0)
                for feature in self.preprocessing_pipeline.feature_extractor.selected_features]

    def _get_risk_level(self, attack_type: str) -> str:
        """Get risk level for attack type"""
        risk_mapping = {
            'BENIGN': 'low',
            'DDoS': 'critical',
            'PortScan': 'medium',
            'BruteForce': 'high',
            'WebAttack': 'high',
            'Infiltration': 'critical',
            'Botnet': 'high',
            'DoS': 'critical'
        }
        return risk_mapping.get(attack_type, 'medium')

    def _update_batch_stats(self, batch_size: int, processing_time: float,
                           results: List[Dict[str, Any]]):
        """Update batch processing statistics"""
        self.stats['processed_flows'] += batch_size
        self.stats['batch_count'] += 1

        # Count attacks
        attacks = sum(1 for r in results if r['prediction'] == 'attack')
        self.stats['detected_attacks'] += attacks

        # Update average batch time
        alpha = 0.1
        if self.stats['average_batch_time'] == 0:
            self.stats['average_batch_time'] = processing_time
        else:
            self.stats['average_batch_time'] = (
                alpha * processing_time +
                (1 - alpha) * self.stats['average_batch_time']
            )

        # Update queue size
        self.stats['queue_size'] = self.input_queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['processed_flows'] > 0:
            stats['attack_rate'] = (stats['detected_attacks'] / stats['processed_flows']) * 100
            stats['throughput_fps'] = stats['processed_flows'] / (stats['batch_count'] * stats['average_batch_time'])

        return stats

    def stop_processing(self):
        """Stop batch processing"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5.0)
```

This comprehensive AI track documentation provides detailed implementation guidance for all AI/ML components of the IDS-AI system, including model development, training pipelines, real-time inference, and performance optimization strategies. The documentation covers both traditional machine learning and deep learning approaches, with emphasis on practical implementation and deployment considerations.
