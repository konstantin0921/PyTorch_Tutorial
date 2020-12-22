from IPython import display
import pandas as pd
from os import system, name
from time import sleep

# define our clear function
def clear():
    # for windows
    if name == "nt":
        _ = system("cls")
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


df_base = pd.DataFrame({"name": ["jonh"], "age": [22]})
display.display(df_base)


for i in range(5):

    df = pd.DataFrame({"name": [f"Tom{i}"], "age": [22]})
    df_base = pd.concat([df_base, df], axis=0)
    clear()

    display.display(df_base)
    sleep(2)
