import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx,get_script_run_ctx
from subprocess import Popen

ctx = get_script_run_ctx()
process = Popen(['python','my_script.py'])
add_script_run_ctx(process,ctx)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import os
from typing import Dict, List, Tuple

# --- File Upload Setup ---
# Create a folder to store uploaded files and plots
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Data Validation Function ---
def validate_input_data(df: pd.DataFrame) -> bool:
    """Validate the input data format and content."""
    if 'Grade' not in df.columns and 'Score' not in df.columns:
        st.error("CSV must contain either 'Grade' or 'Score' column.")
        return False
    
    grade_col = 'Grade' if 'Grade' in df.columns else 'Score'
    
    if df[grade_col].isnull().any():
        st.error("Found missing values in grade/score column.")
        return False
        
    if not pd.to_numeric(df[grade_col], errors='coerce').notnull().all():
        st.error("All grades/scores must be numeric values.")
        return False
        
    if 'Student ID' in df.columns and df['Student ID'].duplicated().any():
        st.error("Found duplicate Student IDs.")
        return False
        
    return True

# --- Statistical Calculation Function ---
def calculate_statistics(grades: np.ndarray) -> Tuple[float, float, float]:
    """Calculate mean, variance, and skewness of grades."""
    if len(grades) == 0:
        return 0.0, 0.0, 0.0
    mean = np.mean(grades)
    std_dev = np.std(grades, ddof=1)
    skewness = skew(grades)
    return mean, std_dev, skewness

# --- Plotting Functions ---
def plot_distribution(grades: np.ndarray | List[str], title: str, is_categorical: bool = False) -> str:
    """Plot the grade distribution and save it as an image."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
    
    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4))  # Smaller graph size
    
    if is_categorical:
        # Categorical distribution
        grade_counts = pd.Series(grades).value_counts().sort_index()
        grade_counts.plot(kind="bar", color="#69b3a2", alpha=0.85)
        plt.xlabel("Grades")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        # Add annotations
        for i, value in enumerate(grade_counts.values):
            plt.text(i, value + 0.2, str(value), ha='center', fontsize=8)
    else:
        # Numerical distribution
        plt.hist(grades, bins=15, alpha=0.6, color="#69b3a2", edgecolor='black', label="Histogram")
        mean, std = np.mean(grades), np.std(grades)
        
        # Overlay normal distribution
        x = np.linspace(min(grades), max(grades), 100)
        plt.plot(x, norm.pdf(x, mean, std), color="#ff5733", label="Normal Distribution", linewidth=2)
        
        # Add annotations for mean and standard deviation
        plt.axvline(mean, color="#3333cc", linestyle="--", label=f"Mean: {mean:.2f}")
        plt.axvline(mean + std, color="#cc3333", linestyle="--", label=f"Mean + 1 Std: {mean + std:.2f}")
        plt.axvline(mean - std, color="#cc3333", linestyle="--", label=f"Mean - 1 Std: {mean - std:.2f}")
        
        plt.xlabel("Scores")
        plt.ylabel("Frequency")
    
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Save static plot
    plot_path = os.path.join(UPLOAD_FOLDER, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)  # Slightly lower DPI for smaller size
    plt.close()
    return plot_path

# --- Grading Functions ---
# Function for HEC Relative Grading
def hec_relative_grading(scores: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
    """Implement HEC relative grading system based on mean and standard deviation."""
    mean = np.mean(scores)
    std_dev = np.std(scores, ddof=1)
    
    # Define grade cutoffs based on the HEC policy
    grade_cutoffs = {
        'A*': mean + 2 * std_dev,
        'A': (mean + 1.5 * std_dev, mean + 2 * std_dev),
        'A-': (mean + std_dev, mean + 1.5 * std_dev),
        'B+': (mean + 0.5 * std_dev, mean + std_dev),
        'B': (mean, mean + 0.5 * std_dev),
        'B-': (mean - 0.5 * std_dev, mean),
        'C+': (mean - std_dev, mean - 0.5 * std_dev),
        'C': (mean - (4 / 3) * std_dev, mean - std_dev),
        'C-': (mean - (5 / 3) * std_dev, mean - (4 / 3) * std_dev),
        'D': (mean - 2 * std_dev, mean - (5 / 3) * std_dev),
        'F': mean - 2 * std_dev
    }
    
    # Assign grades
    grades = []
    for score in scores:
        if score >= grade_cutoffs['A*']:
            grades.append('A*')
        elif grade_cutoffs['A'][0] <= score < grade_cutoffs['A'][1]:
            grades.append('A')
        elif grade_cutoffs['A-'][0] <= score < grade_cutoffs['A-'][1]:
            grades.append('A-')
        elif grade_cutoffs['B+'][0] <= score < grade_cutoffs['B+'][1]:
            grades.append('B+')
        elif grade_cutoffs['B'][0] <= score < grade_cutoffs['B'][1]:
            grades.append('B')
        elif grade_cutoffs['B-'][0] <= score < grade_cutoffs['B-'][1]:
            grades.append('B-')
        elif grade_cutoffs['C+'][0] <= score < grade_cutoffs['C+'][1]:
            grades.append('C+')
        elif grade_cutoffs['C'][0] <= score < grade_cutoffs['C'][1]:
            grades.append('C')
        elif grade_cutoffs['C-'][0] <= score < grade_cutoffs['C-'][1]:
            grades.append('C-')
        elif grade_cutoffs['D'][0] <= score < grade_cutoffs['D'][1]:
            grades.append('D')
        else:
            grades.append('F')
    
    return grade_cutoffs, grades


# Function for Custom Relative Grading
def custom_relative_grading(scores: np.ndarray, distribution: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Implement custom relative grading based on desired grade distribution."""
    
    # Normalize the distribution to ensure it sums to 1
    total_percentage = sum(distribution.values())
    if total_percentage != 1.0:
        # Normalize the distribution if the sum is not 1
        distribution = {grade: perc / total_percentage for grade, perc in distribution.items()}
    
    # Sort scores in descending order
    sorted_scores = np.sort(scores)[::-1]
    total_students = len(scores)
    
    # Calculate cutoffs based on normalized distribution
    cumulative_students = 0
    grade_cutoffs = {}
    for grade, percentage in distribution.items():
        student_count = int(percentage * total_students)
        cumulative_students += student_count
        
        if cumulative_students >= len(sorted_scores):
            cutoff = sorted_scores[-1]
        else:
            index = min(cumulative_students, len(sorted_scores) - 1)
            cutoff = sorted_scores[index]
            
        grade_cutoffs[grade] = cutoff
    
    # Assign grades based on cutoffs
    grades = []
    for score in scores:
        assigned = False
        for grade, cutoff in sorted(grade_cutoffs.items(), key=lambda x: x[1], reverse=True):
            if score >= cutoff:
                grades.append(grade)
                assigned = True
                break
        if not assigned:
            grades.append(list(grade_cutoffs.keys())[-1])
    
    return grade_cutoffs, grades

# Function for HEC Absolute Grading
def hec_absolute_grading(scores: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
    """Implement HEC absolute grading system with fixed score boundaries."""
    grade_cutoffs = {
        'A': 85,
        'A-': 80,
        'B+': 75,
        'B': 71,
        'B-': 68,
        'C+': 64,
        'C': 61,
        'C-': 58,
        'D+': 54,
        'D': 50,
        'F': 0
    }
    # Assign grades
    grades = []
    for score in scores:
        if score >= grade_cutoffs['A']:
            grades.append('A')
        elif score >= grade_cutoffs['A-']:
            grades.append('A-')
        elif score >= grade_cutoffs['B+']:
            grades.append('B+')
        elif score >= grade_cutoffs['B']:
            grades.append('B')
        elif score >= grade_cutoffs['B-']:
            grades.append('B-')
        elif score >= grade_cutoffs['C+']:
            grades.append('C+')
        elif score >= grade_cutoffs['C']:
            grades.append('C')
        elif score >= grade_cutoffs['C-']:
            grades.append('C-')
        elif score >= grade_cutoffs['D+']:
            grades.append('D+')
        elif score >= grade_cutoffs['D']:
            grades.append('D')
        else:
            grades.append('F')
    
    return grade_cutoffs, grades

# Function for Custom Absolute Grading
def custom_absolute_grading(scores: np.ndarray, cutoffs: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Implement custom absolute grading based on user-defined cutoffs."""
    # Sort cutoffs by value in descending order
    sorted_cutoffs = dict(sorted(cutoffs.items(), key=lambda x: x[1], reverse=True))
    
    # Assign grades
    grades = []
    for score in scores:
        assigned = False
        for grade, cutoff in sorted_cutoffs.items():
            if score >= cutoff:
                grades.append(grade)
                assigned = True
                break
        if not assigned:
            grades.append('F')  # Default grade if below all cutoffs
    
    return sorted_cutoffs, grades

# --- Main Application Logic ---
st.set_page_config(page_title="Advanced Grading System", layout="wide")
st.title("Grading System in HEC accredited university")

password = st.text_input("Enter Password", type="password")
if password != "dark1234":
    st.warning("Please enter the correct password to access the system.")
    st.stop() 
    
# Select grading system
grading_system = st.radio(
    "Select Grading System",
    ["HEC Relative Grading", "Custom Relative Grading", "Absolute Grading"],
    help="Choose between relative grading systems based on class performance or absolute grading with fixed cutoffs"
)

# --- File Upload ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file with student marks", 
    type=["csv"],
    help="File should contain 'Student ID' and 'Grade' or 'Score' columns"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if validate_input_data(df):
        # Determine which column to use
        grade_col = 'Grade' if 'Grade' in df.columns else 'Score'
        scores = df[grade_col].to_numpy()
        
        # Display basic statistics
        st.header("2. Data Analysis")
        mean, std_dev, skewness = calculate_statistics(scores)
        st.write(f"Mean: {mean:.2f}, Standard Deviation: {std_dev:.2f}, Skewness: {skewness:.2f}")
        
        # Display distribution plot
        plot_path = plot_distribution(scores, "Original Score Distribution", is_categorical=False)
        st.image(plot_path, caption="Original Score Distribution")
        
        
       # Grading Logic
        if grading_system == "HEC Relative Grading":
            grade_cutoffs, grades = hec_relative_grading(scores)
            
            # Calculate and display grade distribution
            grade_counts = pd.Series(grades).value_counts().sort_index()
            
            hec_grade_distribution = pd.DataFrame({
                "Grade": grade_counts.index,
                "Students": grade_counts.values,
                "Percentage": [(count / len(grades) * 100) for count in grade_counts.values]
            })

            # Display HEC Relative Grade Distribution as a table
            st.subheader("HEC Relative Grade Distribution")
            st.dataframe(hec_grade_distribution.style.format({"Percentage": "{:.1f}%"}))
            
            # Show grade distribution plot
            plot_path = plot_distribution(grades, "HEC Relative Grade Distribution", is_categorical=True)
            st.image(plot_path, caption="HEC Relative Grade Distribution")
            
            # Display Grade Results
            st.header("3. Grading Results")
            df['Grade'] = grades
            if 'Student ID' in df.columns:
                df = df[['Student ID', grade_col, 'Grade']]
            else:
                df = df[[grade_col, 'Grade']]
                
        elif grading_system == "Custom Relative Grading":
            st.sidebar.header("Custom Relative Grading Distribution")
            distribution = {
                    "A": st.sidebar.slider("A", 0.0, 1.0, 0.15),
                    "A-": st.sidebar.slider("A-", 0.0, 1.0, 0.15),
                    "B+": st.sidebar.slider("B+", 0.0, 1.0, 0.15),
                    "B": st.sidebar.slider("B", 0.0, 1.0, 0.15),
                    "B-": st.sidebar.slider("B-", 0.0, 1.0, 0.15),
                    "C+": st.sidebar.slider("C+", 0.0, 1.0, 0.15),
                    "C": st.sidebar.slider("C", 0.0, 1.0, 0.15),
                    "C-": st.sidebar.slider("C-", 0.0, 1.0, 0.15),
                    "D+": st.sidebar.slider("D+", 0.0, 1.0, 0.05),
                    "D": st.sidebar.slider("D", 0.0, 1.0, 0.05),
                    "D-": st.sidebar.slider("D-", 0.0, 1.0, 0.05),
         }
            grade_cutoffs, grades = custom_relative_grading(scores, distribution)
            # Calculate and display grade distribution
            grade_counts = pd.Series(grades).value_counts().sort_index()
            # Show grade distribution plot
            plot_path = plot_distribution(grades, "Custom Relative Grade Distribution", is_categorical=True)
            st.image(plot_path, caption="Custom Relative Grade Distribution")
            grade_distribution = pd.DataFrame({
             "Grade": grade_counts.index,
             "Students": grade_counts.values,
            })
            # Display the table in Streamlit
            st.subheader("Custom Relative Grade Distribution")
            st.dataframe(grade_distribution)
            # Display Grade Results
            st.header("3. Grading Results")
            df['Grade'] = grades
            if 'Student ID' in df.columns:
                df = df[['Student ID', grade_col, 'Grade']]
            else:
                df = df[[grade_col, 'Grade']]
            

                
        elif grading_system == "Absolute Grading":
            st.sidebar.header("Custom Absolute Cutoffs")
            cutoffs = {
                "A": st.sidebar.slider("A Cutoff", 0, 100, 85),
                "A-": st.sidebar.slider("A- Cutoff", 0, 95, 80),
                "B+": st.sidebar.slider("B+ Cutoff", 0, 90, 75),
                "B": st.sidebar.slider("B Cutoff", 0, 85, 71),
                "B-": st.sidebar.slider("B- Cutoff", 0, 80, 68),
                "C+": st.sidebar.slider("C+ Cutoff", 0, 75, 64),
                "C": st.sidebar.slider("C Cutoff", 0, 70, 61),
                "C-": st.sidebar.slider("C- Cutoff", 0, 65, 58),
                "D+": st.sidebar.slider("D+ Cutoff", 0, 60, 54),
                "D": st.sidebar.slider("D Cutoff", 0, 50, 50),
            }
            
            # Get both HEC and custom grades
            hec_cutoffs, hec_grades = hec_absolute_grading(scores)
            custom_cutoffs, custom_grades = custom_absolute_grading(scores, cutoffs)
            
            # Show grade distribution plots
            plot_path = plot_distribution(hec_grades, "HEC Grade Distribution", is_categorical=True)
            st.image(plot_path, caption="HEC Grade Distribution")
            
            plot_path = plot_distribution(custom_grades, "Custom Grade Distribution", is_categorical=True)
            st.image(plot_path, caption="Custom Grade Distribution")
            
            # Calculate grade distributions
            hec_grade_counts = pd.Series(hec_grades).value_counts().sort_index()
            custom_grade_counts = pd.Series(custom_grades).value_counts().sort_index()
            
            # Display grade distribution statistics
    # Display grade distribution statistics in a tabular format
            st.subheader("HEC Grade Distribution")
                # Create a DataFrame for HEC grades
            hec_grade_df = pd.DataFrame({
            "Grade": hec_grade_counts.index,
            " Count": hec_grade_counts.values,
            "Percentage": [(count / len(hec_grades) * 100) for count in hec_grade_counts.values]
                })
            # Display DataFrame
            st.dataframe(hec_grade_df.style.format({"Percentage": "{:.1f}%"}))
            st.subheader("Custom Grade Distribution")
                # Create a DataFrame for Custom grades
            custom_grade_df = pd.DataFrame({
            "Grade": custom_grade_counts.index,
            "Count": custom_grade_counts.values,
            "Percentage": [(count / len(custom_grades) * 100) for count in custom_grade_counts.values]
                })
            # Display DataFrame
            st.dataframe(custom_grade_df.style.format({"Percentage": "{:.1f}%"}))

            # Display Grade Results
            st.header("3. Grading Results")
            df['HEC Grade'] = hec_grades
            df['Custom Grade'] = custom_grades
            
            # Reorder columns for absolute grading
            if 'Student ID' in df.columns:
                df = df[['Student ID', grade_col, 'HEC Grade', 'Custom Grade']]
            else:
                df = df[[grade_col, 'HEC Grade', 'Custom Grade']]
        st.dataframe(df)
        st.download_button(
            label="Download Graded Results....",
            data=df.to_csv(index=False),
            file_name="graded_results.csv",
            mime="text/csv")
