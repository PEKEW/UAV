import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import time
from pathlib import Path
from plotly.subplots import make_subplots
from src.ui.app import BatteryAnalysisApp

def main():
    app = BatteryAnalysisApp()
    app.run()

if __name__ == "__main__":
    main() 