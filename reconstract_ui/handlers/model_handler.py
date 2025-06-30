"""
Model handling logic for the UAV health monitoring application
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class ModelHandler:
    """Handles model loading, validation, and inference"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for model handling"""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = {}
        if 'model_detection_completed' not in st.session_state:
            st.session_state.model_detection_completed = False
    
    def detect_model_type(self, filename):
        """Detect model type based on filename prefix"""
        filename_lower = filename.lower()
        if filename_lower.startswith('battery_'):
            return 'battery'
        elif filename_lower.startswith('flight_'):
            return 'flight'
        else:
            return 'unknown'
    
    def validate_model_data_compatibility(self, model_type, data_type):
        """Validate model and data compatibility"""
        if model_type == 'unknown' or data_type == 'unknown':
            return False, "模型类型或数据类型未知"
        
        if model_type != data_type:
            return False, f"模型类型({model_type})与数据类型({data_type})不匹配"
        
        return True, "兼容"
    
    def load_model(self, uploaded_model):
        """Load and validate model file"""
        if uploaded_model is None:
            st.error("请先选择模型文件")
            return
        
        with st.spinner('加载模型...'):
            try:
                file_extension = Path(uploaded_model.name).suffix.lower()
                model_type = self.detect_model_type(uploaded_model.name)
                
                model_info = {
                    "filename": uploaded_model.name,
                    "filesize": f"{uploaded_model.size / 1024:.2f} KB",
                    "filetype": file_extension,
                    "model_type": model_type
                }
                
                model_object = self._load_model_by_type(uploaded_model, file_extension, model_info)
                
                # Validate compatibility with loaded data
                if st.session_state.data_loaded:
                    is_compatible, compatibility_msg = self.validate_model_data_compatibility(
                        model_type, st.session_state.data_type
                    )
                    model_info["compatibility"] = compatibility_msg
                    
                    if not is_compatible:
                        st.error(f"模型与数据不兼容: {compatibility_msg}")
                        st.error("请确保模型文件名以正确的前缀开头（battery_ 或 flight_）")
                        return
                    else:
                        st.success(f"模型与数据兼容性验证通过: {compatibility_msg}")
                
                st.session_state.model = {
                    'name': uploaded_model.name,
                    'type': file_extension,
                    'model_type': model_type,
                    'object': model_object,
                    'data': uploaded_model.getvalue()
                }
                st.session_state.model_info = model_info
                st.session_state.model_loaded = True
                
                st.success(f"模型 {uploaded_model.name} 加载成功！")
                
            except Exception as e:
                st.error(f"模型加载失败: {str(e)}")
                st.session_state.model_loaded = False
                st.session_state.model = None
                st.session_state.model_info = {}
    
    def _load_model_by_type(self, uploaded_model, file_extension, model_info):
        """Load model based on file type"""
        model_object = None
        
        if file_extension in ['.pkl', '.joblib']:
            model_info["framework"] = "Scikit-learn"
            try:
                if file_extension == '.pkl':
                    model_object = pickle.loads(uploaded_model.getvalue())
                else:
                    model_object = joblib.loads(uploaded_model.getvalue())
            except Exception as e:
                st.warning(f"无法加载模型对象: {str(e)}，将使用模拟模式")
                model_object = None
        
        elif file_extension == '.h5':
            model_info["framework"] = "TensorFlow/Keras"
            model_object = uploaded_model.getvalue()
        
        elif file_extension in ['.pth', '.pt']:
            model_info["framework"] = "PyTorch"
            try:
                import io
                model_object = torch.load(io.BytesIO(uploaded_model.getvalue()), map_location='cpu')
            except Exception as e:
                st.warning(f"无法加载PyTorch模型: {str(e)}，将使用模拟模式")
                model_object = uploaded_model.getvalue()
        
        elif file_extension == '.onnx':
            model_info["framework"] = "ONNX"
            model_object = uploaded_model.getvalue()
        
        else:
            model_info["framework"] = "Unknown"
            model_object = uploaded_model.getvalue()
        
        return model_object
    
    def prepare_data_for_model(self, data, data_type):
        """Prepare data for model inference"""
        try:
            if 'original_samples' in data:
                # H5 data processing
                sequences = data['original_samples']
                sample_info = data['sample_info']
                n_samples = sample_info['n_samples']
                
                sequence_times = []
                for i in range(n_samples):
                    start_time = i * 30
                    end_time = (i + 1) * 30
                    sequence_times.append((start_time, end_time))
                
                scaler = self._get_scaler_for_data_type(data_type)
                sequences = self._scale_sequences(sequences, scaler)
                
                return sequences, sequence_times, scaler
            
            else:
                # CSV data processing
                return self._prepare_csv_data_for_model(data, data_type)
            
        except Exception as e:
            st.error(f"数据预处理失败: {str(e)}")
            return None
    
    def _get_scaler_for_data_type(self, data_type):
        """Get appropriate scaler for data type"""
        if data_type == 'battery':
            return StandardScaler()
        else:
            return MinMaxScaler()
    
    def _scale_sequences(self, sequences, scaler):
        """Scale sequence data"""
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = scaler.fit_transform(sequences_flat)
        return sequences_scaled.reshape(original_shape)
    
    def _prepare_csv_data_for_model(self, data, data_type):
        """Prepare CSV data for model inference"""
        time_data = data['time']
        features = data['features']
        
        # Get expected features for data type
        if data_type == 'battery':
            expected_features = ['Ecell_V', 'I_mA', 'EnergyCharge_W_h', 'QCharge_mA_h', 
                               'EnergyDischarge_W_h', 'QDischarge_mA_h', 'Temperature__C']
            expected_feature_count = 7
        elif data_type == 'flight':
            expected_features = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 
                               'velocity', 'acceleration', 'altitude']
            expected_feature_count = 9
        else:
            return None
        
        # Check feature completeness
        available_features = [f for f in expected_features if f in features]
        if len(available_features) < expected_feature_count:
            st.warning(f"数据特征不完整，期望{expected_feature_count}个特征，实际{len(available_features)}个")
            return None
        
        # Build feature matrix
        feature_matrix = np.column_stack([features[f] for f in available_features])
        
        # Split data into 30-second windows
        sequence_length = 30
        sequences = []
        sequence_times = []
        
        time_min, time_max = time_data.min(), time_data.max()
        current_time = time_min
        
        while current_time + sequence_length <= time_max:
            window_mask = (time_data >= current_time) & (time_data < current_time + sequence_length)
            window_data = feature_matrix[window_mask]
            
            if len(window_data) >= sequence_length:
                if len(window_data) > sequence_length:
                    indices = np.linspace(0, len(window_data)-1, sequence_length, dtype=int)
                    window_data = window_data[indices]
                
                sequences.append(window_data)
                sequence_times.append((current_time, current_time + sequence_length))
            
            current_time += sequence_length
        
        if not sequences:
            st.warning("无法创建30秒序列，数据可能不足")
            return None
        
        sequences = np.array(sequences)
        scaler = self._get_scaler_for_data_type(data_type)
        sequences = self._scale_sequences(sequences, scaler)
        
        st.info(f"使用CSV数据格式，从时间序列创建了{len(sequences)}个30秒样本")
        return sequences, sequence_times, scaler
    
    def perform_model_inference(self, sequences, model_info):
        """Perform model inference"""
        model_type = model_info.get('model_type', 'unknown')
        framework = model_info.get('framework', 'Unknown')
        model_object = st.session_state.model.get('object')
        
        if model_object is None:
            st.error("模型对象为空，请检查模型文件")
            return None, None
        
        if framework == "PyTorch":
            return self._pytorch_inference(sequences, model_object, model_type)
        else:
            st.error(f"框架 {framework} 暂不支持")
            return None, None
    
    def _pytorch_inference(self, sequences, model_object, model_type):
        """Perform PyTorch model inference"""
        try:
            if isinstance(model_object, dict):
                # State dict - need to initialize model architecture
                model_object = model_object.get('model_state_dict', model_object)
                model = self._initialize_model_architecture(model_type)
                if model is None:
                    return None, None
                
                try:
                    model.load_state_dict(model_object)
                    st.success(f"成功加载{model_type}模型权重")
                except Exception as e:
                    st.error(f"加载模型权重失败: {str(e)}")
                    return None, None
                
                model.eval()
                return self._run_pytorch_inference(model, sequences)
                
            elif hasattr(model_object, 'eval'):
                # Complete model object
                st.info("检测到完整的PyTorch模型对象")
                model_object.eval()
                return self._run_pytorch_inference(model_object, sequences)
            
            else:
                st.error(f"无法识别的PyTorch模型格式: {type(model_object)}")
                return None, None
                
        except Exception as e:
            st.error(f"PyTorch模型推理失败: {str(e)}")
            return None, None
    
    def _initialize_model_architecture(self, model_type):
        """Initialize model architecture based on type"""
        try:
            if model_type == 'battery':
                from src.models.battery_cnn_lstm import BatteryAnomalyNet
                config_dict = {
                    'sequence_length': 30, 'input_features': 7, 'num_classes': 2,
                    'cnn_channels': [16, 32, 64], 'lstm_hidden': 64,
                    'attention_heads': 2, 'classifier_hidden': [32], 'dropout_rate': 0.5
                }
                return BatteryAnomalyNet(config_dict)
            
            elif model_type == 'flight':
                from src.models.flight_cnn_lstm import FlightAnomalyNet
                model_dict = {
                    'num_classes': 2, 'cnn_channels': [96, 192, 384],
                    'lstm_hidden': 256, 'attention_heads': 8,
                    'classifier_hidden': [128, 64]
                }
                return FlightAnomalyNet(model_dict)
            
            else:
                st.error(f"未知的模型类型: {model_type}")
                return None
                
        except ImportError as e:
            st.error(f"无法导入模型类: {str(e)}")
            return None
    
    def _run_pytorch_inference(self, model, sequences):
        """Run PyTorch inference and return predictions"""
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sequences)
            outputs = model(input_tensor)
            
            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 2 and outputs.shape[1] == 2:
                    # Binary classification output
                    probabilities = torch.softmax(outputs, dim=1)
                    confidences = probabilities[:, 1].numpy()
                else:
                    confidences = torch.sigmoid(outputs).squeeze().numpy()
            else:
                confidences = outputs
            
            anomaly_threshold = 0.5
            anomaly_predictions = confidences > anomaly_threshold
            
            # Show model performance if ground truth available
            if 'original_labels' in st.session_state.data:
                self._show_model_performance(anomaly_predictions, confidences)
            
            return anomaly_predictions, confidences
    
    def _show_model_performance(self, anomaly_predictions, confidences):
        """Show model performance statistics"""
        true_labels = st.session_state.data['original_labels'][:len(anomaly_predictions)]
        true_anomaly_ratio = np.mean(true_labels)
        pred_anomaly_ratio = np.mean(anomaly_predictions)
        accuracy = np.mean(anomaly_predictions == true_labels)
        
        st.info(f"真实异常比例: {true_anomaly_ratio:.1%}，正常比例: {1-true_anomaly_ratio:.1%}")
        st.info(f"模型预测异常比例: {pred_anomaly_ratio:.1%}，正常比例: {1-pred_anomaly_ratio:.1%}")
        st.info(f"模型准确率: {accuracy:.1%}")
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, anomaly_predictions)
        st.write("混淆矩阵（真实/预测）: 0=正常, 1=异常")
        st.write(cm)
    
    def generate_anomaly_detection(self, time_data):
        """Generate anomaly detection results"""
        if not st.session_state.model_loaded or not st.session_state.data_loaded:
            st.error("模型或数据未加载")
            return []
        
        data = st.session_state.data
        data_type = st.session_state.data_type
        model_info = st.session_state.model_info
        
        # Prepare data
        prepared_data = self.prepare_data_for_model(data, data_type)
        if prepared_data is None:
            return []
        
        sequences, sequence_times, scaler = prepared_data
        
        # Perform inference
        anomaly_predictions, confidences = self.perform_model_inference(sequences, model_info)
        if anomaly_predictions is None:
            return []
        
        # Generate anomaly regions
        anomaly_regions = []
        for i, (is_anomaly, confidence) in enumerate(zip(anomaly_predictions, confidences)):
            if is_anomaly:
                start_time, end_time = sequence_times[i]
                anomaly_regions.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': float(confidence),
                    'sequence_id': i
                })
        
        return anomaly_regions