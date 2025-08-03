# PITkit Numerical Analysis Toolkit

This directory contains the Python scripts used to test the predictions of Participatory Interface Theory (PIT) against observational data.

## Overview

The primary tool here is the **global fitter**, a program designed to test the **Coherence-Acceleration Hypothesis** by fitting the PIT model to the entire SPARC (Spitzer Photometry and Accurate Rotation Curves) galaxy sample. It determines the single, best-fit universal value for the acceleration constant, $a_0$.

## Requirements

* Python 3.9+
* For macOS, it's recommended to install Python via [Homebrew](https://brew.sh/).

## Setup and Installation

Follow these steps to set up your environment to run the analysis scripts.

### Step 1: Clone the Repository

If you haven't already, clone the main PITkit repository to your local machine.

### Step 2: Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies. Navigate to this directory (`pitkit_code/`) in your terminal and run the following commands:

```bash
# Create a virtual environment named 'pit_env'
python3 -m venv pit_env

# Activate the environment
source pit_env/bin/activate
```

Step 3: Install Dependencies

This project requires several scientific Python libraries.

First, create a file named requirements.txt in this directory and add the following lines to it:

```
numpy
scipy
pandas
astropy
matplotlib
```

Next, with your virtual environment still active, run the following command to install all the necessary packages:

```bash
pip install -r requirements.txt
```

Step 4: Download the Data

The scripts require the SPARC rotation curve data.

Download the dataset from the official source: http://astroweb.case.edu/SPARC/Rotmod_LTG.zip

In this directory, create a subdirectory named data/.

Unzip the downloaded file and place all _rotmod.dat files into the data/ directory.

Usage
Single-Galaxy Fitter (single_galaxy_fitter.py)

This script performs a two-parameter fit (for Υ⋆ and a_0) on a single galaxy, which is useful for diagnostics and testing.

To Run:

```bash
python single_galaxy_fitter.py
```

This will print the best-fit parameters for that galaxy to the terminal and display a plot of its rotation curve with the PIT model overlaid.

Global Fitter (global_fitter.py)

This is the main experimental script. It runs a global fit to find the single best universal a_0 across the entire SPARC sample, while allowing Υ⋆ to be a free parameter for each galaxy.

To Run:

```bash
python global_fitter.py
```

Warning: This script will take several minutes to complete as it performs thousands of individual fits. It will print its progress to the terminal. The final output will be the best-fit universal value for a_0 and the total chi-squared for the entire sample.

(Note from Bob: it only took about 15 seconds on my (somewhat older now) Mac M1).
