# UAV Battery Health Monitoring System - Refactored

This is a refactored version of the original `ui.py` file, organized into a clean, modular architecture.

## Architecture

```
reconstract_ui/
├── app.py                      # Main application entry point
├── components/                 # UI components
│   ├── __init__.py
│   ├── header.py              # Application header
│   ├── control_panel.py       # Control panel with monitoring mode, data selection, model loading
│   └── visualization.py       # Data visualization and charts
├── handlers/                  # Business logic handlers
│   ├── __init__.py
│   ├── data_handler.py        # Data loading and processing
│   └── model_handler.py       # Model loading and inference
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── styles.py              # CSS styles
│   └── session_manager.py     # Session state management
└── README.md                  # This file
```

## Key Improvements

### 1. **Separation of Concerns**
- **UI Components**: Pure UI rendering logic
- **Handlers**: Business logic for data and model operations
- **Utils**: Shared utilities and configuration

### 2. **Single Responsibility Principle**
- Each class has one clear responsibility
- Methods are focused and specific
- Easier to test and maintain

### 3. **Clean Dependencies**
- Clear dependency injection pattern
- Handlers are injected into components
- Reduced coupling between modules

### 4. **Maintainability**
- Smaller, focused files
- Clear module boundaries
- Easy to extend and modify

## Usage

To run the refactored application:

```python
from reconstract_ui.app import main
main()
```

Or run directly:
```bash
streamlit run reconstract_ui/app.py
```

## Components

### HeaderComponent
- Renders the application title and header
- Pure UI component with no business logic

### ControlPanelComponent
- Monitoring mode selection
- Data file upload and loading
- Model file upload and loading
- Anomaly detection trigger
- Depends on DataHandler and ModelHandler

### VisualizationComponent
- Data visualization charts
- Time range selection
- Anomaly region display
- Battery and flight data specific visualizations

### DataHandler
- H5 and CSV data loading
- Data type detection and validation
- Data processing for visualization
- Anomaly label extraction

### ModelHandler
- Model loading and validation
- Model-data compatibility checking
- Data preparation for inference
- PyTorch model inference
- Performance metrics calculation

## Session State Management

All session state is managed centrally through `SessionManager`, ensuring consistent initialization and reducing state-related bugs.

## Styling

CSS styles are extracted to a separate module for better organization and reusability.

## Original Functionality Preserved

All original functionality from `ui.py` is preserved:
- Real-time/offline monitoring modes
- H5 data file support
- Battery and flight data types
- Model loading and inference
- Visualization with anomaly detection
- Time range filtering
- Performance metrics display