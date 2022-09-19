:: Define Anaconda root location.
set conda_root=D:/Anaconda3/
:: Run conda activator.
call %conda_root%/Scripts/activate.bat %conda_root%

:: Activate local environment.
conda activate ./env