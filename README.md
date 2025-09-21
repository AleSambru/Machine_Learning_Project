# Machine Learning con Applicazioni — Exam Project

This repository contains the materials for the exam project of the course *Machine Learning con Applicazioni* at the University of Milan.  

## Project Description  

We consider the [CalCOFI dataset](https://www.kaggle.com/datasets/sohier/calcofi), which contains oceanographic data from Southern California.  
The goal is to implement a regression algorithm with regularization (Ridge or LASSO) and a Mercer kernel.  

The dataset includes a large number of feature variables. We focus on selecting those representing the chemical–physical conditions of the ecosystem (e.g., temperature, salinity, depth, concentrations of chemical elements).  

The objective of this project is to:  
- Predict **salinity** and **temperature** as functions of the other variables.  
- Investigate potential correlations between salinity and temperature.  

Further information about the dataset can be found on the [official CalCOFI website](https://calcofi.org/data/oceanographic-data/bottle-database/).  

## Repository Structure  

- **cacofi/**  
  Dataset used for the project.  

- **Analisi_Dataset/**  
  Preliminary analysis and data filtering.  

- **ML_ASambruna_Project/**  
  Complete analysis of the project.  
