

import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import datetime 
import time
import os

# Utitilty functions - some are implemented, others you must implement yourself.

# function to plot the bar graph and average runtimes of N trials
# Please note that this function only plots the graph and does not save it
# To save the graphs you must use plot.save(). Refer to matplotlib documentation
# Function to create a directory if it doesn't exist



def draw_plot(run_arr, mean):
    x = np.arange(0, len(run_arr),1)
    fig=plt.figure(figsize=(20,8))
    plt.axhline(mean,color="red",linestyle="--",label="Avg")
    plt.xlabel("Iterations")
    plt.ylabel("Run time in ms order of 1e-6")
    plt.title("Run time for retrieval")
    plt.show()

# function to generate random list 
# @args : length = number of items 
#       : max_value maximum value
def create_random_list(length, max_value, item=None, item_index=None):
    random_list = [random.randint(0,max_value) for i in range(length)]
    if item!= None:
        random_list.insert(item_index,item)

    return random_list

# function to generate reversed list of a given size and with a given maximum value
def create_reverse_list(length, max_value, item=None, item_index=None):
    reversed_list = []
    random_list = [random.randint(0,max_value) for i in range(length)]
    if item!= None:
        random_list.insert(item_index,item)
    random_list.sort(reverse=True)
    for i in range(length):
        reversed_list.append(random_list[i])
    return reversed_list
    


# function to generate near sorted list of a given size and with a given maximum value
def create_near_sorted_list(length, max_value, item=None, item_index=None):
    near_sorted_list = []
    sorted_list = sorted(random.randint(0, max_value) for x in range(length))

    #this line may not even be relavant
    if item is not None and item_index is not None:
        sorted_list[item_index] = item

    # we want the list to be 75% sorted i.e 25% unsorted, so we take 25% of the list and then do k random swaps to unsort it
    num_unsorted = max(1, (25 * length) // 100)  
    for k in range(num_unsorted):
        index1 = random.randint(0, length - 1)
        index2 = random.randint(0, length - 1)
        sorted_list[index1], sorted_list[index2] = sorted_list[index2], sorted_list[index1]

    for i in range(length):
        near_sorted_list.append(sorted_list[i])
    return near_sorted_list

def reduced_unique_list(length, max_value, item=None, item_index=None):

    data = create_random_list(length, max_value, item, item_index)
    reduced_list = []
    

    for num in data: 
        # Ensure the number is unique by checking the reduced list
       if num not in reduced_list:
           reduced_list.append(num)

    return reduced_list



# Implementation of sorting algorithms
class BubbleSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]
    
    def sort(self):
        items = self.items[:]
        x = len(items)
        for outer in range(x):#we need two loops to loop through adjacent pairs, so one loops through one element
            for inner in range(0,x-outer-1): #the second loops through elems after it
                if items[inner] > items[inner + 1]: #compare adjacent elems
                    items[inner], items[inner + 1] = items[inner + 1], items[inner] #swap
        self.sorted_items = items
 

    def get_sorted(self,):
        return self.sorted_items

"""
class InsertionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items=[]

    def sort(self):
        items = self.items[:]
        for counter in range(1, len(items)):
            key = items[counter]
            x = counter - 1
            while x >=0 and key < items[x]:
                items[x+1] = items[x]
                x -= 1
            items[x+1] = key
            self.sorted_items = items         
       ### your implementation for insertion sort goes here 
    
    def get_sorted(self,):
        return self.sorted_items
"""
class InsertionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items = []
#decrementing element in unsorted part through sorted part of list
    def sort(self):
        items = self.items[:] 
        for counter in range(1, len(items)): 
            key = items[counter]
            n = counter - 1
            while n >= 0 and items[n] > key:  
                items[n + 1] = items[n]
                n -= 1 #decrementing through sorted part so long as elem is >, then we move on to the next elem in the sorted part of the array
            items[n + 1] = key   
        self.sorted_items = items

    def get_sorted(self):
        return self.sorted_items


   
class SelectionSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort

    def sort(self):
        count = len(self.items)
        
        for x in range(count):
            smallest = x  # Assume smallest element at x
            
            for y in range(x + 1, count):  
                if self.items[y] < self.items[smallest]:  # Find smallest element
                    smallest = y
            
            # Swap smallest element with index x
            self.items[x], self.items[smallest] = self.items[smallest], self.items[x]
        
    def get_sorted(self):
        self.sort()
        return self.items

class MergeSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items = []
#two subbarrays, pointers starting at indexes of both. This compares that value at each pointer and puts them in the final list accordingly
    def merge(self, left, right):
        i, j = 0, 0
        merged = []

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        # Append remaining elements from either list
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    def sort(self, items=None):
        if items is None:
            items = self.items

        if len(items) <= 1:
            return items

        mid = len(items) // 2 #divides list into 2 
        left = self.sort(items[:mid])
        right = self.sort(items[mid:])
        return self.merge(left, right)

    def get_sorted(self):
        if not self.sorted_items:
            self.sorted_items = self.sort()
        return self.sorted_items



    
class QuickSort:
    def __init__(self, items_to_sort):
        self.items = items_to_sort
        self.sorted_items = []

    def sort(self):
        a = self.items[:]
        random.shuffle(a)  # Shuffle to avoid worst-case where pivot is first/last elem
        self._sort(a, 0, len(a) - 1)
        self.sorted_items = a

    def _sort(self, a, lo, hi):
        if hi <= lo:  
            return
        piv_index = self._partition(a, lo, hi)
        self._sort(a, lo, piv_index - 1)  # Sort the left side
        self._sort(a, piv_index + 1, hi)  # Sort the right side

    def _partition(self, a, lo, hi):
        piv = a[lo]  # Choose pivot
        x, y = lo + 1, hi
        while True:
            while x <= hi and a[x] < piv:  
                x += 1
            while a[y] > piv:  
                y -= 1
            if x >= y:  # Pointers crossed
                break
            self._exchange(a, x, y)  # Swap elements
        self._exchange(a, lo, y)  # Place pivot in final position
        return y  # Return pivot index

    def _exchange(self, a, x, y):
        a[x], a[y] = a[y], a[x]  # Swap elements

    def get_sorted(self):
        if not self.sorted_items:
            self.sort()
        return self.sorted_items


"""
test_case = []
print("Test case array input: ",test_case)  
test_case2 = [42]
print("Test case 2 array input: ",test_case2)  
test_case3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
print("Test case 3 array input: ",test_case3) 
test_case4 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] 
print("Test case 4 array input: ",test_case4) 
test_case5 = create_near_sorted_list(10, 100)
print("Test case 5 array input: ",test_case5) 

# Test case 1
print("Test case 1 input:", test_case)
bubble_sort1 = BubbleSort(test_case)
bubble_sort1.sort()
print("After sorting by BubbleSort:", bubble_sort1.get_sorted())

insertion_sort1 = InsertionSort(test_case)
insertion_sort1.sort()
print("After sorting by InsertionSort:", insertion_sort1.get_sorted())

selection_sort1 = SelectionSort(test_case)
selection_sort1.sort()
print("After sorting by SelectionSort:", selection_sort1.get_sorted())

quick_sort1 = QuickSort(test_case)
quick_sort1.sort()
print("After sorting by QuickSort:", quick_sort1.get_sorted())

merge_sort1 = MergeSort(test_case)
sorted_merge1 = merge_sort1.sort()
print("After sorting by MergeSort:", sorted_merge1)

# Test case 2
print("Test case 2 input:", test_case2)
bubble_sort2 = BubbleSort(test_case2)
bubble_sort2.sort()
print("After sorting by BubbleSort:", bubble_sort2.get_sorted())

insertion_sort2 = InsertionSort(test_case2)
insertion_sort2.sort()
print("After sorting by InsertionSort:", insertion_sort2.get_sorted())

selection_sort2 = SelectionSort(test_case2)
selection_sort2.sort()
print("After sorting by SelectionSort:", selection_sort2.get_sorted())

quick_sort2 = QuickSort(test_case2)
quick_sort2.sort()
print("After sorting by QuickSort:", quick_sort2.get_sorted())

merge_sort2 = MergeSort(test_case2)
sorted_merge2 = merge_sort2.sort()
print("After sorting by MergeSort:", sorted_merge2)

# Test case 3
print("Test case 3 input:", test_case3)
bubble_sort3 = BubbleSort(test_case3)
bubble_sort3.sort()
print("After sorting by BubbleSort:", bubble_sort3.get_sorted())

insertion_sort3 = InsertionSort(test_case3)
insertion_sort3.sort()
print("After sorting by InsertionSort:", insertion_sort3.get_sorted())

selection_sort3 = SelectionSort(test_case3)
selection_sort3.sort()
print("After sorting by SelectionSort:", selection_sort3.get_sorted())

quick_sort3 = QuickSort(test_case3)
quick_sort3.sort()
print("After sorting by QuickSort:", quick_sort3.get_sorted())

merge_sort3 = MergeSort(test_case3)
sorted_merge3 = merge_sort3.sort()
print("After sorting by MergeSort:", sorted_merge3)

# Test case 4
print("Test case 4 input:", test_case4)
bubble_sort4 = BubbleSort(test_case4)
bubble_sort4.sort()
print("After sorting by BubbleSort:", bubble_sort4.get_sorted())

insertion_sort4 = InsertionSort(test_case4)
insertion_sort4.sort()
print("After sorting by InsertionSort:", insertion_sort4.get_sorted())

selection_sort4 = SelectionSort(test_case4)
selection_sort4.sort()
print("After sorting by SelectionSort:", selection_sort4.get_sorted())

quick_sort4 = QuickSort(test_case4)
quick_sort4.sort()
print("After sorting by QuickSort:", quick_sort4.get_sorted())

merge_sort4 = MergeSort(test_case4)
sorted_merge4 = merge_sort4.sort()
print("After sorting by MergeSort:", sorted_merge4)

# Test case 5
print("Test case 5 input:", test_case5)
bubble_sort5 = BubbleSort(test_case5)
bubble_sort5.sort()
print("After sorting by BubbleSort:", bubble_sort5.get_sorted())

insertion_sort5 = InsertionSort(test_case5)
insertion_sort5.sort()
print("After sorting by InsertionSort:", insertion_sort5.get_sorted())

selection_sort5 = SelectionSort(test_case5)
selection_sort5.sort()
print("After sorting by SelectionSort:", selection_sort5.get_sorted())

quick_sort5 = QuickSort(test_case5)
quick_sort5.sort()
print("After sorting by QuickSort:", quick_sort5.get_sorted())

merge_sort5 = MergeSort(test_case5)
sorted_merge5 = merge_sort5.sort()
print("After sorting by MergeSort:", sorted_merge5)
"""



# Function to measure the runtime of a sorting algorithm
def measure_runtime(algorithm_class, input_list, N):
    runtimes = []
    for _ in range(N):
        copy = input_list[:]
        sorter = algorithm_class(copy) 
        start_time = time.time()
        sorter.sort()
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return runtimes

# Function to plot the runtimes
def draw_plot(run_arr, mean, algorithm, experiment):
    x = np.arange(len(run_arr))
    fig = plt.figure(figsize=(20, 8))
    plt.plot(x, run_arr, label=f'{algorithm} Runtimes')
    plt.axhline(mean, color="red", linestyle="--", label="Mean Runtime")
    plt.xlabel("Trial")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime for {algorithm} - {experiment}")
    plt.legend()
    plt.savefig(f"{algorithm}_{experiment}.png")
    plt.close(fig)

# Function to measure the runtime of a sorting algorithm
def measure_runtime(algorithm_class, data, trials):
    runtimes = []
    for _ in range(trials):
        copy = data[:]
        sorter = algorithm_class(copy)
        start_time = time.time()
        sorter.sort()
        end_time = time.time()
        runtimes.append(end_time - start_time)
    return runtimes

# Function to plot the runtimes
def draw_plot(run_arr, mean, algorithm, experiment):
    x = np.arange(len(run_arr))
    fig = plt.figure(figsize=(12, 6))
    plt.bar(x, run_arr, color='purple', label=f'{algorithm} Runtimes')
    plt.axhline(mean, color="black", linestyle="--", label="Mean Runtime")
    plt.xlabel("Trial")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime for {algorithm} - {experiment}")
    plt.legend()
    plt.savefig(f"{algorithm}_{experiment}.png")
    plt.close(fig)

# Experiment A
def experiment_A():
    N = 80 #no.trials
    list_size = 150
    for algorithm in [BubbleSort, InsertionSort, SelectionSort, MergeSort, QuickSort]:
        input_list = create_random_list(list_size, 1000)
        alg_name = algorithm.__name__
        runtimes = measure_runtime(algorithm, input_list, N)
        mean_runtime = sum(runtimes) / len(runtimes)
        draw_plot(runtimes, mean_runtime, alg_name, "Experiment A")
    print("Experiment A is done")

# Experiment B
def experiment_B():
    N = 80 #no.trials
    list_size = 150
    for algorithm in [BubbleSort, InsertionSort, SelectionSort, MergeSort, QuickSort]:
        input_list = create_near_sorted_list(100, 1000)
        alg_name = algorithm.__name__
        runtimes = measure_runtime(algorithm, input_list, N)
        mean_runtime = sum(runtimes) / len(runtimes)
        draw_plot(runtimes, mean_runtime, alg_name, "Experiment B")
    print("Experiment B is done")

# Experiment C
def experiment_C():
    N = 80
    list_size = 150
    for algorithm in [BubbleSort, InsertionSort, SelectionSort, MergeSort, QuickSort]:
        input_list = create_reverse_list(list_size, 1000)
        alg_name = algorithm.__name__
        runtimes = measure_runtime(algorithm, input_list, N)
        mean_runtime = sum(runtimes) / len(runtimes)
        draw_plot(runtimes, mean_runtime, alg_name, "Experiment C")
    print("Experiment C is done")

# Experiment D
def experiment_D():
    N = 10  # Number of trials
    list_sizes = [10, 20, 40, 50, 100]  # Different list sizes
    max_value = 1000

    # Define sorting functions
    def bubble_sort(test_list):
        bubble_sort_instance = BubbleSort(test_list)
        bubble_sort_instance.sort()

    def insertion_sort(test_list):
        insertion_sort_instance = InsertionSort(test_list)
        insertion_sort_instance.sort()

    def selection_sort(test_list):
        selection_sort_instance = SelectionSort(test_list)
        selection_sort_instance.sort()

    def quick_sort(test_list):
        quick_sort_instance = QuickSort(test_list)
        quick_sort_instance.sort()

    def merge_sort(test_list):
        merge_sort_instance = MergeSort(test_list)
        merge_sort_instance.sort()

    # List of sorting algorithms
    sorting_algorithms = {
        "Bubble Sort": bubble_sort,
        "Insertion Sort": insertion_sort,
        "Selection Sort": selection_sort,
        "Quick Sort": quick_sort,
        "Merge Sort": merge_sort
    }

    # Dictionary to store aggregated runtimes
    avg_runtimes = {name: [] for name in sorting_algorithms}

    # Run experiments for each sorting algorithm
    for name, sort_func in sorting_algorithms.items():
        total_time = 0
        for size in list_sizes:
            for _ in range(N):
                test_list = create_random_list(size, max_value)
                start_time = time.time()
                sort_func(test_list)
                end_time = time.time()
                total_time += (end_time - start_time)
        avg_runtimes[name] = total_time / (N * len(list_sizes))  # Average over all sizes and no.trials

    # Plot the aggregated runtimes
    for name, runtime in avg_runtimes.items():
        plt.figure(figsize=(10, 6))
        plt.bar([name], [runtime], color='blue')
        plt.axhline(runtime, color='red', linestyle='--', label='Avg Runtime')
        plt.xlabel('Sorting Algorithm')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title(f'{name} - Average Execution Time for Experiment D')
        plt.legend()
        plt.tight_layout()
        save_path = f"Experiment_D_{name.replace(' ', '_')}.png"
        plt.savefig(save_path)
        print(f"Graph saved to: {os.path.abspath(save_path)}")
        plt.close()
        print("Experiment D is done")

# Experiment E
def experiment_E():
    N = 80
    list_size = 150
    for algorithm in [BubbleSort, InsertionSort, SelectionSort, MergeSort, QuickSort]:
        input_list = reduced_unique_list(list_size, 1000)
        alg_name = algorithm.__name__
        runtimes = measure_runtime(algorithm, input_list, N)
        mean_runtime = sum(runtimes) / len(runtimes)
        draw_plot(runtimes, mean_runtime, alg_name, "Experiment E")
    print('Experiment E is done')

# Run experiments
experiment_A()
experiment_B()
experiment_C()
experiment_D()
experiment_E()
