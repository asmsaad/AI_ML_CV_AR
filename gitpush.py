

 
 
 
import os

def header(header):
    print('┌───────────────────────────────────────────────────┐') 
    print('│' + ' '*51 + '│') 
    print('└───────────────────────────────────────────────────┘') 



header('')
os.system("git status")
header('')
os.system("git add . ")
header('')
os.system("git status")
header('')
os.system('git commit -m "first commit22" ')
os.system("git push -u origin main")