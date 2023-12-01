import psutil
def is_process_running(name):
    for process in psutil.process_iter(['name']):
        if process.info['name'] == name:
            return True
    return False
if is_process_running('svm.py'):
    print('The script is running.')
else:
    print('The script is not running.')