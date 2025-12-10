# Mehta2026_Nature_pipeline
Analysis code for "Growth of Light Seed Black Holes in the Early Universe" paper
This repository contains the full analysis pipeline used to generate all figures, source-data files, and diagnostics associated with our simulations of early–universe star formation, heavy–seed BH formation, and radiation–gas interactions.
The repository does not include simulation-generation code, only the analysis tools used to process outputs and generate the plots for the accompanying manuscript.

The scripts expect a directory structure such as:
base/
   ├── L13/
   ├── L14/
   ├── L15/
   ├── L15_BHFB/
   └── ... (simulation snapshots)

**RUN**
python3 NaturePaperScripts.py
Comment out the functions you want to analyze.

**FILES**

NaturePaperScripts.py: main, contains all the scripts to create the plots in the paper.
DataReader.py: Contains the reader classes for pickle binary files and sink particle binary files
GroupReader.py: Contains the classes for reading the halos and subhalos files with basic scripts.
Plotter.py: Contains the classes for plotting routines.
SinkParticles.py: Contains the classes to read the sink particles files and perform basic scripts.
Utilities.py: Contains the classes for the basic functions needed by all other scripts along with constants.
YT.py: Contains the classes for using the YT_Analysis for simulation plotting routine.


**OUTPUTS**

Mehta_Fig1.pdf
Mehta_Fig2.pdf
Mehta_Fig3.pdf
Mehta_Fig4.pdf
Mehta_ED_Fig1.jpg
Mehta_ED_Fig2.jpg
Mehta_ED_Fig3.jpg
Mehta_ED_Fig4.jpg
Mehta_ED_Fig5.jpg
Mehta_ED_Fig6.jpg
Mehta_ED_Fig7.jpg
Mehta_ED_Fig8.jpg
Mehta_ED_Fig9.jpg
Mehta_SourceData_Fig1.txt
Mehta_SourceData_Fig2.txt
Mehta_SourceData_Fig3.txt
Mehta_SourceData_Fig4.txt
Mehta_SourceData_ED_Fig1.txt
Mehta_SourceData_ED_Fig2.txt
Mehta_SourceData_ED_Fig3.txt
Mehta_SourceData_ED_Fig4.txt
Mehta_SourceData_ED_Fig5.txt
Mehta_SourceData_ED_Fig6.txt
Mehta_SourceData_ED_Fig7.txt
Mehta_SourceData_ED_Fig8.txt
Mehta_SourceData_ED_Fig9.txt
