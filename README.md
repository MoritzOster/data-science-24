# Process Monitoring of Grinding Processes

## Overview

This repository contains the code for the project "Process Monitoring of Grinding Processes," conducted by Group 1 of Data Science course 2024 at Saarland University. The project aims to monitor and analyze grinding processes to identify trends and detect anomalies, using data from sensors installed on a Kellenberger T25 grinding machine.

## Project Structure

The repository is organized as follows:

- **data/**
  - Contains a small selection of the raw and processed data files
  - **grinding/**: Grinding process data
  - **dressing/**: Dressing process data
- **src/**
  - Contains the source code for data processing and model development
- **notebooks/**
  - Jupyter notebooks for exploratory data analysis
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

## Contents

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/grinding-process-monitoring.git
   cd grinding-process-monitoring
   ```
