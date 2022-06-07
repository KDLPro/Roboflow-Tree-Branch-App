from PySide6.QtWidgets import * 
from PySide6.QtCore import *
from PySide6.QtGui import *
import roboflow
import time, ctypes


# Loading Screen
class SplashScreen(QWidget):
    def __init__(self, window):
        super().__init__()
        self.setWindowTitle('Loading Roboflow Test App...')
        self.setFixedSize(1100, 500)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Start 
        self.counter = 0
        self.n = 300 # total instance

        self.loadingFail = False

        # Initialize splash screen
        self.initUI()

        # Set icons
        self.setWindowIcon(QIcon('logo.png'))
        self.myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(self.myappid)

        # Initialize main window
        self.mainWindow = window

        self.timer = QTimer()
        self.timer.timeout.connect(self.loadApp)
        self.timer.start(30)

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.frame = QFrame()
        layout.addWidget(self.frame)

        # Initialize label title
        self.labelTitle = QLabel(self.frame)
        self.labelTitle.setObjectName('LabelTitle')

        # center label
        self.labelTitle.resize(self.width() - 10, 150)
        self.labelTitle.move(0, 40) # x, y
        self.labelTitle.setText('Roboflow Test App')
        self.labelTitle.setAlignment(Qt.AlignCenter)

        # Initialize label description
        self.labelDescription = QLabel(self.frame)
        self.labelDescription.resize(self.width() - 10, 50)
        self.labelDescription.move(0, self.labelTitle.height())
        self.labelDescription.setObjectName('LabelDesc')
        self.labelDescription.setText('<strong>Loading Roboflow Project...</strong>')
        self.labelDescription.setAlignment(Qt.AlignCenter)

        self.progressBar = QProgressBar(self.frame)
        self.progressBar.resize(self.width() - 200 - 10, 50)
        self.progressBar.move(100, self.labelDescription.y() + 130)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFormat('%p%')
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, self.n)
        self.progressBar.setValue(20)

        self.labelLoading = QLabel(self.frame)
        self.labelLoading.resize(self.width() - 10, 50)
        self.labelLoading.move(0, self.progressBar.y() + 70)
        self.labelLoading.setObjectName('LabelLoading')
        self.labelLoading.setAlignment(Qt.AlignCenter)

    def loadApp(self):
        if self.loadingFail == False:
            self.progressBar.setValue(self.counter)

            if self.counter == int(self.n * 0.2):
                self.loadProject()
            elif self.counter == int(self.n * 0.4):
                self.labelDescription.setText('<strong>Loading Roboflow model...</strong>')
            elif self.counter == int(self.n * 0.6):
                self.loadModel()
            elif self.counter == int(self.n * 0.4):
                self.labelDescription.setText('<strong>Loading app data...</strong>')
            elif self.counter >= self.n:
                self.timer.stop()
                self.close()

                self.mainWindow.show()

            self.counter += 3
        else:
            time.sleep(3)

            # Close the loading screen and the app
            self.close()

    def loadProject(self):
        try:
            RoboflowTest.initialize_roboflow_proj()
            self.progressBar.setValue(self.counter)
        except:   
            self.labelLoading.setText('Loading failed. Please connect to the Internet.')
            self.loadingFail = True

    def loadModel(self):
        try:
            RoboflowTest.initialize_roboflow_model()
            self.progressBar.setValue(self.counter)
        except:   
            self.labelLoading.setText('Loading failed. Please connect to the Internet.')
            self.loadingFail = True
        
class RoboflowTest():
    def initialize_roboflow_proj():
        global rf, project
        
        rf = roboflow.Roboflow(api_key = "6rPY4Pu8E3vgBRtt04Re")        
        project = rf.workspace("yolov5-transformer").project("obstructions-in-transformer-lines-1")

    def initialize_roboflow_model():
        global model
        model = project.version("1").model

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create thread manager
        self.thread_manager = QThreadPool()
  
        self.setWindowTitle("Transformer Obstructions App")

        # Set the geometry and alignment of window
        self.setGeometry(0, 0, 500, 500)
        center = QScreen.availableGeometry(QApplication.primaryScreen()).center()
        geo = self.frameGeometry()
        geo.moveCenter(center)
        self.move(geo.topLeft())

        # Set icons
        self.setWindowIcon(QIcon('logo.png'))
        self.myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(self.myappid)

        # Create menu
        self.menu = self.menuBar()
        self.menu.setFont(QFont("Arial", 10))
        self.menu.setStyleSheet("background-color: #8cd1f7;")

        self.open_img_but = QAction("Open Image", self)
        self.open_img_but.setFont(QFont("Arial", 9))

        self.save_img_but = QAction("Save Prediction to Image", self)
        self.save_img_but.setFont(QFont("Arial", 9))

        self.file_menu = self.menu.addMenu("&File")
        self.file_menu.setStyleSheet("""
                background-color: #ffd773;
                selection-background-color: #FFE5B9;
                color: #634700;
            """)
        self.file_menu.addAction(self.open_img_but)
        self.file_menu.addAction(self.save_img_but)

        self.centralwidget = QWidget(self)
        self.loading = QLabel("No image loaded...", self.centralwidget)
        self.loading.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.loading)

        self.open_img_but.triggered.connect(self.open_file)
        self.save_img_but.triggered.connect(self.save_file)

        
        # Hide the app first
        self.hide()

    def open_file(self):
        self.image = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                           "Image Files (*.png *.jpg *.bmp)")

        self.img_loc = str(self.image[0])
        
        self.thread_manager.start(self.do_predictions)

    def do_predictions(self):
        # Do predictions on the image
        self.prediction = model.predict(self.img_loc)
        self.prediction.save(output_path = "__pycache__/prediction.jpg")

    def save_file(self):
        self.target_image = QFileDialog.getSaveFileName(self, 'Save predictions', '',
                                           "Image Files (*.png *.jpg *.bmp)")

        self.target_img_loc = str(self.image[0])
        
        self.thread_manager.start(self.save_predictions)

    def save_predictions(self):
        # Do predictions on the image
        self.prediction = model.predict(self.img_loc)
        self.prediction.save(output_path = self.target_img_loc)




