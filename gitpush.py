

 
 
 
import os

def header(header):
    print('┌' + '─'*51 + '┐') 
    print('│' + header.center(51,' ') + '│') 
    print('└' + '─'*51 + '┘') 



header('Checking Changes')
os.system("git status")
header('Adding Changes')
os.system("git add . ")
header('Added Status')
os.system("git status")
header('UPLOADING...')
os.system('git commit -m "first commit22" ')
os.system("git push -u origin main")