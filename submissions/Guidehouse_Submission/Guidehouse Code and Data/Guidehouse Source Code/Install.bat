SETLOCAL EnableDelayedExpansion
ECHO INSTALLING APPLICATION ENVIRONMENT...
:: Temporarily modify PATH variable to access Anaconda installation
:: Add all potential path directory's, depending on user install of Anaconda

:: Old Anaconda pathnames that would have been added to PATH
SET folder1=C:\Users\echo001\AppData\Local\Continuum\anaconda3
SET folder2=C:\Users\echo001\AppData\Local\Continuum\anaconda3\Library\mingw-w64\bin
SET folder3=C:\Users\echo001\AppData\Local\Continuum\anaconda3\Library\user\bin
SET folder4=C:\Users\echo001\AppData\Local\Continuum\anaconda3\Library\bin
SET folder5=C:\Users\echo001\AppData\Local\Continuum\anaconda3\Scripts
:: New Anaconda pathnames that would have been added to PATH
SET folder6=C:\Users\echo001\Anaconda3
SET folder7=C:\Users\echo001\Anaconda3\Library\mingw-w64\bin
SET folder8=C:\Users\echo001\Anaconda3\Library\bin
SET folder9=C:\Users\echo001\Anaconda3\Library\usr\bin
SET folder10=C:\Users\echo001\Anaconda3\Scripts

IF EXIST %folder1% (PATH !PATH!;!folder1!)
IF EXIST %folder2% (PATH !PATH!;!folder2!) 
IF EXIST %folder3% (PATH !PATH!;!folder3!) 
IF EXIST %folder4% (PATH !PATH!;!folder4!) 
IF EXIST %folder5% (PATH !PATH!;!folder5!) 
IF EXIST %folder6% (PATH !PATH!;!folder6!)
IF EXIST %folder7% (PATH !PATH!;!folder7!)
IF EXIST %folder8% (PATH !PATH!;!folder8!)
IF EXIST %folder9% (PATH !PATH!;!folder9!)
IF EXIST %folder10% (PATH !PATH!;!folder10!)

:: Activate anaconda base environment to access pip
ECHO ACTIVATING CONDA BASE ENVIRONMENT...
IF EXIST %folder5% (CALL %folder5%\activate.bat)
IF EXIST %folder10% (CALL %folder10%\activate.bat)
ECHO PYTHON AND PIP VERSIONS IN BASE ANACONDA ENVIRONMENT
CALL python --version
CALL pip --version

:: Install virtualenv if not already
ECHO INSTALLING VIRTUALENV...
CALL pip install virtualenv

:: Create new virtual environment for gsa_env and activate
ECHO CREATING NEW VIRTUAL ENVIRONMENT...
CALL virtualenv -p 3.7 gsa_env
CALL gsa_env\Scripts\activate.bat
ECHO VIRTUAL ENVIRONMENT PYTHON VERSION
CALL python --version

:: Install application environments
ECHO INSTALLING REQUIREMENTS...
CALL python -m pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
ECHO APPLICATION INSTALLATION COMPLETE
PAUSE