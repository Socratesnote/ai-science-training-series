:: Define Anaconda root location.
set conda_root=D:/Anaconda3/
:: Run conda activator.
call %conda_root%/Scripts/activate.bat %conda_root%

set env_name=BasicEnv

:: Create environment in Anaconda Envs, with specific name.
:: conda env create -f %env_name%.yml

:: Create environment in specific folder, without name.
set local_path=%~dp0
conda create --prefix %local_path%\env -c anaconda spyder jupyter matplotlib tensorflow -c pytorch pytorch torchvision torchaudio cudatoolkit=11.6 -c conda-forge 
:: In this setup, conda normally displays the entire path as the prefix. Adjust configuration to only show the folder name.
conda config --set env_prompt '({name})'