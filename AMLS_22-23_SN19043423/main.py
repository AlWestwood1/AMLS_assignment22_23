from pathlib import Path
ROOT = Path(__file__).parent #Root directory is the folder this file is placed in
from importlib.machinery import SourceFileLoader
#Import modules for each task
import A1.A1_main as A1
import A2.A2_main as A2
import B1.B1_cnn as B1
import B2.B2_cnn as B2


while True:
    try:
        sel = int(input("Please type the number of the program you would like to run (-1 to quit):\n 1. A1\n 2. A2\n 3. B1\n 4. B2\n"))
        if sel == -1:
            exit()

        elif sel == 1:
            print("Task A1 selected. Please allow a few minutes for the program to run.\n")
            A1.A1_main()

        elif sel == 2:
            print("Task A2 selected. Please allow a few minutes for the program to run.\n")
            A2.A2_main()
        
        elif sel == 3:
            print("Task B1 selected. Please allow a few minutes for the program to run.\n")
            B1.B1_main()

        elif sel == 4:
            print("Task B2 selected. Please allow a few minutes for the program to run.\n")
            B2.B2_main()
        
        else:
            print("Please enter a valid number\n")
            
    except ValueError:
        print("Please enter a valid number\n")

    

    
