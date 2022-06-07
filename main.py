from PySide6.QtWidgets import * 
from PySide6.QtCore import *
from PySide6.QtGui import *
import app
import sys


if __name__ == "__main__":
    # Loading Screen
    App = QApplication(sys.argv)
    App.setStyleSheet('''
        #LabelTitle {
            font-size: 60px;
            color: #93deed;
        }

        #LabelDesc {
            font-size: 30px;
            color: #edfbff;
        }

        #LabelLoading {
            font-size: 30px;
            color: #e8e8eb;
        }

        QFrame {
            background-color: #2F4454;
            color: rgb(220, 220, 220);
        }

        QProgressBar {
            background-color: #366910;
            color: rgb(200, 200, 200);
            border-style: none;
            border-radius: 10px;
            text-align: center;
            font-size: 30px;
        }

        QProgressBar::chunk {
            border-radius: 10px;
            background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #b6ffb4, stop:1 #006e2b);
        }
    ''')
    
    window = app.MainWindow()
    
    splash = app.SplashScreen(window)
    splash.show()

    app.setStyleSheet("")
    
    sys.exit(App.exec())


