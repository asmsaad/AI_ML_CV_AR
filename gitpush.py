

 
 
 
import os
from datetime import datetime

# Get current date and time
current_datetime = datetime.now()

# Format date and time
formatted_datetime = current_datetime.strftime("%d%b%Y_%I:%M%p")

print(formatted_datetime)


def header(header):
    print('┌' + '─'*51 + '┐') 
    print('│' + header.center(51,' ') + '│') 
    print('└' + '─'*51 + '┘') 



header(formatted_datetime)
header('Checking Changes')
os.system("git status")
header('Adding Changes')
os.system("git add . ")
header('Added Status')
os.system("git status")
header('UPLOADING...')
os.system('git commit -m "first commit22" ')
os.system("git push -u origin main")