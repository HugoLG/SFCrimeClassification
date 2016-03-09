from Tkinter import *
import ttk
from ttk import *
from calendarWidget import *
import googlemaps
import tkFileDialog
from preprocessing import *
from randomforest import *
from network import *
from algorithms import *
from time import gmtime, strftime
import json

initialValues = ['']
rfPool = [] #Pool returned by the Random Forest training method
clfNaiveBayes = None #Trained Naive Bayes Classifier

class NotebookDemo(ttk.Frame):

    def __init__(self, isapp=True):
        ttk.Frame.__init__(self)
        self.pack(fill=X)
        self.master.title('Crime Classification System')
        self.isapp = isapp
        self._create_widgets()

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Import Data", command=lambda: self.onImport())
        fileMenu.add_command(label="Exit", command=self.onExit)
        helpMenu = Menu(menubar)
        helpMenu.add_command(label="Help", command=self.onExit)
        menubar.add_cascade(label="File", menu=fileMenu)
        menubar.add_cascade(label="Help", menu=helpMenu)

    def _create_widgets(self):

        self._create_demo_panel()

    def _create_demo_panel(self):
        mainPanel = Frame(self)
        mainPanel.pack(side=TOP, fill=BOTH)
        # create the notebook
        nb = ttk.Notebook(mainPanel, name='main window')
        nb.enable_traversal()

        nb.pack(fill=X, padx=5, pady=5)
        self._create_main_tab(nb)
        self._create_predict_tab(nb)


    def _create_main_tab(self, nb):

        mainFrame = Frame(nb)

        trainFrame = Frame(mainFrame)
        trainFrame.pack(fill=X)
        trainFilenameLabel = Label(trainFrame, text="No File", width=15)
        trainFilenameLabel.pack(side=LEFT, padx=5, pady=5)
        importTrainButton = Button(trainFrame, text="Import File", command=lambda: self.onImport(trainFilenameLabel))
        importTrainButton.pack(side=LEFT, padx=5, pady=5)
        classifierValue = StringVar()
        trainClassifierComboBox = ttk.Combobox(trainFrame, state="readonly", textvariable=classifierValue)
        trainClassifierComboBox['values']=("Random Forest", "Neural Network", "Naive Bayes")
        #trainClassifierComboBox.current(0)
        trainClassifierComboBox.pack(padx=5, expand=True)
        trainButton = Button(trainFrame, text="Train classifier", command=lambda: self.train(trainFilenameLabel, trainClassifierComboBox, testClassifierComboBox))
        trainButton.pack(padx=5, pady=5)

        separateFrame = Frame(mainFrame)
        separateFrame.pack(fill=X)
        separateLabel = Label(separateFrame, text="---------------------------------------------------------------------------------------------")
        separateLabel.pack(padx=5, pady=5)

        testFrame = Frame(mainFrame)
        testFrame.pack(fill=X)
        testFilenameLabel = Label(testFrame, text="No File", width=15)
        testFilenameLabel.pack(side=LEFT, padx=5, pady=5)
        importTestButton = Button(testFrame, text="Import File", command=lambda: self.onImport(testFilenameLabel))
        importTestButton.pack(side=LEFT, padx=5, pady=5)
        testClassifierValue = StringVar()
        testClassifierComboBox = ttk.Combobox(testFrame, state="readonly", textvariable=testClassifierValue)
        testClassifierComboBox['values'] = initialValues #("Random Forest", "Neural Network", "Naive Bayes")
        #testClassifierComboBox.current(0)
        testClassifierComboBox.pack(padx=5, expand=True)
        testButton = Button(testFrame, text="Test classifier", command=lambda: self.train(trainFilenameLabel, trainClassifierComboBox))
        testButton.pack(padx=5, pady=5)


        nb.add(mainFrame, text="main")

    def _create_predict_tab(self, nb):
        mainFrame = Frame(nb)

        addressFrame = Frame(mainFrame)
        addressFrame.pack(fill=X)
        addressLabel = Label(addressFrame, text="Address", width=10)
        addressLabel.pack(side=LEFT, padx=5, pady=5)
        addressEntry = Entry(addressFrame)
        addressEntry.pack(fill=X, padx=5, expand=True)

        pdFrame = Frame(mainFrame)
        pdFrame.pack(fill=X)
        policeDistrictLabel = Label(pdFrame, text="P. District", width=12)
        policeDistrictLabel.pack(side=LEFT, padx=3, pady=5)
        pdDistrictValue = StringVar()
        pdDistrictComboBox = ttk.Combobox(pdFrame, textvariable=pdDistrictValue)
        pdDistrictComboBox['values']=("pd1", "pd2", "pd3")
        pdDistrictComboBox.current(0)
        pdDistrictComboBox.pack(padx=5, expand=True)

        dateFrame = Frame(mainFrame)
        dateFrame.pack(fill=X)
        dateLabel = Label(dateFrame, text="Choose Date of Crime", width=20)
        dateLabel.pack(side=LEFT, padx=5, pady=5)
        ttkcal = Calendar(dateFrame,firstweekday=calendar.SUNDAY)
        ttkcal.pack()
        #dateButton = Button(dateFrame, text="Choose Date", command=lambda: self.showDate(ttkcal))
        #dateButton.pack(padx=5, pady=5)

        """
        #Remove comments if you want to show labels for Latitude and Longitude
        geolocationFrame = Frame(mainFrame)
        geolocationFrame.pack(fill=X)
        latitudeLabel = Label(geolocationFrame, text="No Latitude", width=12)
        longitudeLabel = Label(geolocationFrame, text="No Longitude", width=12)
        latitudeLabel.pack(side=LEFT, padx=5, pady=5)
        longitudeLabel.pack(side=LEFT, padx=5, pady=5)
        getGeolocationButton = Button(geolocationFrame, text="Get Geolocation", command= lambda: self.getGeolocation(latitudeLabel,longitudeLabel, addressEntry))
        getGeolocationButton.pack(padx=5, pady=5)
        """

        predictFrame = Frame(mainFrame)
        predictFrame.pack(fill=X)
        instrucLabel = Label(predictFrame, text="Select classifier", width=20)
        instrucLabel.pack(side=LEFT, padx=5, pady=5)
        predictClassifierValue = StringVar()
        predictClassifierComboBox = ttk.Combobox(predictFrame, textvariable=predictClassifierValue)
        predictClassifierComboBox['values']=("Random Forest", "Neural Network", "Naive Bayes")
        predictClassifierComboBox.current(0)
        predictClassifierComboBox.pack(side=LEFT, padx=5, expand=True)
        predictButton = Button(predictFrame, text="Predict Category", command= lambda: self.onPredictValue(addressEntry,pdDistrictComboBox,ttkcal,predictClassifierComboBox.get()))
        predictButton.pack(padx=5, pady=5)

        nb.add(mainFrame, text="Predict")


    def onExit(self):
        self.quit()

    def modifyValuesList(self, trainClassifierComboBox, testClassifierComboBox):
        if str(trainClassifierComboBox.get()) not in initialValues:
            initialValues.append(trainClassifierComboBox.get())
            testClassifierComboBox['values'] = initialValues

    def train(self, trainFilenameLabel, trainClassifierComboBox, testClassifierComboBox):
        if trainFilenameLabel["text"] == "No File":
            print("no training file available")
        else:
            if trainClassifierComboBox.get() == "Random Forest":
                global rfPool
                rfPool, cfMatrix = randomForest(trainFilenameLabel["text"])
                print rfPool
                self.modifyValuesList(trainClassifierComboBox, testClassifierComboBox)

            elif trainClassifierComboBox.get() == "Naive Bayes":
                global clfNaiveBayes
                precision, recall, accuracy, clfNaiveBayes = NB_clf_system(trainFilenameLabel["text"])
                self.modifyValuesList(trainClassifierComboBox, testClassifierComboBox)
                print precision, recall, accuracy
            elif trainClassifierComboBox.get() == "Neural Network":
                train()
                print("missing to add this module")
            else:
                print("No classifier selected")

    def onImport(self, trainFilenameLabel):
        #"""
        ftypes = [('All files', '*'),('Python files', '*py')]
        openFileWindow = tkFileDialog.Open(self, filetypes = ftypes)
        filename = openFileWindow.show()
        if filename == "":
            trainFilenameLabel["text"] = "No File"
        else:
            filename = filename.split("/")
            trainFilenameLabel["text"] = filename[-1]

    def onPredictValue(self, addressEntry, pdDistrictComboBox, ttkcal, clf):

        if clf in initialValues:
            gmaps = googlemaps.Client(key='AIzaSyA9ygeOD-Hs6-FRGnUECmKfF4eZs7sZkzw')
            geocodeResult = gmaps.geocode(addressEntry.get())
            lat = geocodeResult[0]['geometry']['location']['lat']
            lng = geocodeResult[0]['geometry']['location']['lng']
            dataList = print_date(ttkcal)
            dataList.append(addressEntry.get())
            dataList.append(pdDistrictComboBox.get())

            dayOfTheWeekNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dayOfTheWeekValue = dayOfTheWeekNames[dataList[3]]
            hr = strftime("%H:%M:%S", gmtime())
            dateFileFormat = str(dataList[0])+"-"+str(dataList[1])+"-"+str(dataList[2])+" "+hr

            fOut = open("singlePredictionData.csv", 'w')
            fOut.write("Dates,Category,Descript,DayOfWeek,PdDistrict,Resolution,Address,X,Y\n")
            fOut.write(str(dateFileFormat)+",,,"+str(dayOfTheWeekValue)+","+str(pdDistrictComboBox.get())+",,"+str(addressEntry.get())+","+str(lat)+","+str(lng))
            fOut.close()

            preprocess("singlePredictionData.csv", False)

            singleValue = pd.read_csv("preprocessed_testing.csv")
            headers = singleValue.columns.values
            headers = np.delete(headers,0)

            dataSingleValue = singleValue[headers]
            #dataSingleValue.get_values()
            categNumRandomForest = randomForestPredicted(dataSingleValue, 39, rfPool)
            print dataSingleValue
            categNumNB = predict_instance(clfNaiveBayes, dataSingleValue)

            print categNumRandomForest, categNumNB


        else:
            print "classifier not trained"



    def getGeolocation(self, latitudeLabel, longitudeLabel, addressEntry):
        if addressEntry.get() == "":
            latitudeLabel.config(text="Lat changed")
            longitudeLabel.config(text="Long changed")
        else:
            latitudeLabel.config(text=addressEntry.get())
            longitudeLabel.config(text="entry long")



if __name__ == '__main__':

    NotebookDemo().mainloop()
