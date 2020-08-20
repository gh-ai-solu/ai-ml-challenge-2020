ECHO OFF
SETLOCAL EnableDelayedExpansion
ECHO Setting up environment...
:: Temporarily modify PATH variable to access Anaconda installation
:: Add all potential path directory's, depending on user install of Anaconda
:: Old Anaconda pathnames that could have been added to PATH during install
SET folder1=C:\Users\%username%\AppData\Local\Continuum\anaconda3
SET folder2=C:\Users\%username%\AppData\Local\Continuum\anaconda3\Library\mingw-w64\bin
SET folder3=C:\Users\%username%\AppData\Local\Continuum\anaconda3\Library\user\bin
SET folder4=C:\Users\%username%\AppData\Local\Continuum\anaconda3\Library\bin
SET folder5=C:\Users\%username%\AppData\Local\Continuum\anaconda3\Scripts
:: New Anaconda pathnames that could have been added to PATH during install
SET folder6=C:\Users\%username%\Anaconda3
SET folder7=C:\Users\%username%\Anaconda3\Library\mingw-w64\bin
SET folder8=C:\Users\%username%\Anaconda3\Library\bin
SET folder9=C:\Users\%username%\Anaconda3\Library\usr\bin
SET folder10=C:\Users\%username%\Anaconda3\Scripts

IF EXIST %folder1% (PATH !PATH!;%folder1%)
IF EXIST %folder2% (PATH !PATH!;%folder2%)
IF EXIST %folder3% (PATH !PATH!;%folder3%)
IF EXIST %folder4% (PATH !PATH!;%folder4%)
IF EXIST %folder5% (PATH !PATH!;%folder5%)
IF EXIST %folder6% (PATH !PATH!;%folder6%)
IF EXIST %folder7% (PATH !PATH!;%folder7%)
IF EXIST %folder8% (PATH !PATH!;%folder8%)
IF EXIST %folder9% (PATH !PATH!;%folder9%)
IF EXIST %folder10% (PATH !PATH!;%folder10%)

ECHO Starting Application...
ECHO DO NOT CLOSE THIS WINDOW OR APP WILL EXIT

CMD /k "cd "gsa_env\Scripts" & activate.bat & cd "..\..\" & streamlit run streamlit_ui.py"


