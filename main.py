from PySide6.QtWidgets import * 
from PySide6.QtCore import *
from PySide6.QtGui import *

import app
import sys
import os


if __name__ == "__main__":
    # Delete pycache image files
    try:
        os.remove("__pycache__/copy1.png")
        os.remove("__pycache__/copy2.png")
        os.remove("__pycache__/copy3.png")
        os.remove("__pycache__/disp.png")
    except:
        pass

    # Loading Screen
    App = QApplication(sys.argv)
    window = app.MainWindow()
    window.setStyleSheet('''
        #MainMenu {
            font-style: Arial;
            font-size: 10pt;
            background-color: #8cd1f7;;
        }
        
        #FileSelection {
            font-style: Arial;
            font-size: 9pt;
        }

        #FileMenu {
            background-color: #ffd773;
            selection-background-color: #ffe5b9;
            color: #634700;
        }

        #ModelMenu {
            background-color: #ffa06d;
            selection-background-color: #ffe1c3;
            color: #634700;
        }

        #PredictMenu {
            background-color: #a87bff;
            selection-background-color: #dfcaff;
            color: #634700;
        }

        #FileSubSelection {
            background-color: #c2de24;
            selection-background-color: #f2ffb5;
            color: #634700;
        }
    ''')
    
    sys.exit(App.exec())


