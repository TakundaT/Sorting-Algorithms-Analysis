# Sorting-Algorithms-Analysis
# Sorting Algorithm Performance Analysis

## Overview

This project evaluates the performance of five sorting algorithms—**Bubble Sort, Insertion Sort, Selection Sort, Merge Sort, and Quick Sort**—under different experimental conditions. The experiments measure the runtime of these algorithms on different types of input lists, generating performance visualizations for analysis.

---

## Sorting Algorithms Implemented

1. **Bubble Sort** - A simple comparison-based sorting algorithm.
2. **Insertion Sort** - A sorting algorithm that builds a sorted sequence one element at a time.
3. **Selection Sort** - A sorting algorithm that repeatedly selects the smallest element and moves it to the sorted portion of the list.
4. **Merge Sort** - A divide-and-conquer sorting algorithm with better performance on large datasets.
5. **Quick Sort** - A highly efficient sorting algorithm that works by partitioning elements around a pivot.

---

## Experiments

The project includes five experiments to analyze the runtime behavior of these sorting algorithms.

### **Experiment A** - Random List  
- Generates a list of **100 random numbers**.
- Measures runtime over **80 trials** for each sorting algorithm.

### **Experiment B** - Near-Sorted List  
- Generates a **75% sorted** list with **25% random swaps**.
- Measures runtime over **80 trials** for each sorting algorithm.

### **Experiment C** - Reversed List  
- Generates a **fully reversed list** of **100 elements**.
- Measures runtime over **80 trials** for each sorting algorithm.

### **Experiment D** - Varying List Sizes  
- Generates lists of **sizes 10, 20, 30, 40, and 50**.
- Measures runtime over **80 trials** for each sorting algorithm.

### **Experiment E** - Reduced Unique List  
- Generates a list of **unique random numbers** to test how the algorithms handle a dataset with no duplicates.
- Measures runtime over **80 trials** for each sorting algorithm.

---
## Results
- Quick sort was found to be fastest in all 5 experiments. Quicksorts continuosly partitions the array according to the pivot element, therefore reducing the effective search range at each level of the recursion tree, making it proportional to ~O(NlogN).
- Bubble Sort was found to be slowest in all 5 experiments. Bubble sort works by doing successive comparisons between adjacent values, making it ~O(N^2).
- Insertion Sort was particularly effective for near sorted lists as it performed even better than merge sort. Due to the fact that there are less inversions in a near sorted list, the insertion sort algorithm will not need to make nearly as many swaps or comparisons when decrementing through the unsorted part of the array.
## How to Run

### **Prerequisites**
- Python 3.x
- Required libraries: `random`, `time`, `matplotlib`, `numpy`, `datetime`, `os`

### **Execution**
Run the script to execute all experiments:

```bash
python sorting_experiments.py

I do recommend using an i7 processor or something similar if you plan on running the algorithm with larger inputs to give more accurate results.
