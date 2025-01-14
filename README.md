# K12b Calculator Coordinates on Parallel Coordinates

A Streamlit-based tool designed to visualize K12b calculator coordinates with **Euclidean distance filtering** and optional **K-Means clustering**. Unlike standard PCA plots, this tool offers a **component-wise perspective**, displaying all DNA components individually in a user-friendly parallel-coordinates format.

**Disclaimer:** This tool is intended as a **helpful visualization aid**. It does not guarantee scientific accuracy or serve as a replacement for rigorous scientific modeling.

## Description

This project loads K12b data (consisting of DNA components per individual) and displays it in a **parallel-coordinates** chart. You can filter results via:  
- Euclidean distance (STD) filter, to remove samples too far from a chosen reference  
- K-Means clustering, to color and group samples  

With parallel coordinates, you can **inspect each DNA component** (e.g., Gedrosia, Siberian, etc.) on its own—unlike in PCA, where multiple components are combined.

## Getting Started

### Dependencies

- Streamlit  
- Python 3.x  
- Plotly (for Python)  
- scikit-learn (for K-Means clustering)  
- Pandas  
- NumPy  

*(Tested on Windows 10 with Python 3.9.)*

### Installing

1. Clone or download this repository.  
2. Install all required libraries (for example):
   pip install streamlit pandas numpy scikit-learn
3. Prepare your K12b data file (k12b.txt) in the correct format. Make sure it has 13 columns: 1 label + 12 numeric (the 12 DNA components).

### Executing Program

1. Open a terminal in the project folder.  
2. Launch the Streamlit app with:
   streamlit run load_k12b.py
3. Load your k12b.txt data (or place it in the same directory).  
4. Adjust filters (STD threshold, min–max sliders, K-Means) as needed.  
5. Inspect parallel coordinates to see each DNA component dimension.

## Help

If you encounter issues or have questions, feel free to open an issue on GitHub.

## Authors

- [@MertD95](https://github.com/MertD95)  
- [LinkedIn](https://www.linkedin.com/in/mert-demirs%C3%BC-5942b8222/)

For additional support, please reach out or create an issue on GitHub.

## Version History

- 0.1  
  Initial release
