# Process Monitoring of Grinding Processes

![image](https://github.com/MoritzOster/data-science-24/assets/44369843/0c3e96c2-c27e-4f16-b8bf-160cb776fbff)

## Overview

This repository contains the code for the project "Process Monitoring of Grinding Processes," conducted by Group 1 of Data Science course 2024 at Saarland University. The project aims to monitor and analyze grinding processes to identify trends and detect anomalies, using data from sensors installed on a Kellenberger T25 grinding machine.

## Project Structure

The repository is organized as follows:

- **data/**
  - Contains parquet files representing the data and pickle files for the model and PCA
- **src/**
  - Contains the source code for data processing and model development
- **dashboard/**
  - Contains the source code related to the monitoring dashboard for grinding processes

## Data Description

The data collected for this project includes:

- **Grinding process recordings**: ~5300 recordings
- **Dressing process recordings**: 140 recordings
- **Anomalous grinding process recordings**: 30 recordings

Each recording consists of:

- **Metadata file** (.json)
- **Acoustic emission signal** (.parquet)
- **RMS value of spindle currents** (.parquet)

**Sample frequencies**:

- AE sensor: 2 MHz
- Current sensor: 100 kHz

## Installation

To install the dependencies, run the following command:

- pip install pandas numpy sklearn tpot scipy PyWavelets altair streamlit

## Contents

In the src folder:

- feature_extraction.py: This script handles the feature extraction process.
- data_split.py: This script contains functionality for splitting the data into a test and training set and upsampling the training set.
- preprocessing.py: This script handles the preprocessing of the extracted features
- genetic_programming.py: Implements genetic programming to determine the best model.
- evaluation.py: Contains functions to evaluate the baseline models.
- pipeline.py: Combines all steps into one script.
- prodetect.py: Implements the ProDetect model and provides functionality to evaluate new data points.

Run: streamlit run dashboard.py in the dashboard folder, to start the dashboard.
Starting the dashboard in the current development stage, runs a simulation of a live grinding process, from a measurement, saved in dashboard/2024.02.14_22.00.40_Grinding
