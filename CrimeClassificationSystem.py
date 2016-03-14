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
import tkMessageBox as mbox

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
        #fileMenu.add_command(label="Import Data", command=lambda: self.onImport())
        fileMenu.add_command(label="Exit", command=self.onExit)
        helpMenu = Menu(menubar)
        helpMenu.add_command(label="Help", command=self.onHelp)
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
        """
        bgImage = Tkinter.PhotoImage(file="cover.png")
        bgLabel = Label(mainFrame, image=bgImage)
        bgLabel.bgImage = bgImage
        bgLabel.pack()
        """

        trainFrame = Frame(mainFrame)
        trainFrame.pack(fill=X)
        trainFilenameLabel = Label(trainFrame, text="No File", width=15)
        trainFilenameLabel.pack(side=LEFT, padx=5, pady=5)
        importTrainButton = Button(trainFrame, text="Import File", command=lambda: self.onImport(trainFilenameLabel, True))
        importTrainButton.pack(side=LEFT, padx=5, pady=5)
        classifierValue = StringVar()
        trainClassifierComboBox = ttk.Combobox(trainFrame, state="readonly", textvariable=classifierValue)
        trainClassifierComboBox['values']=("Random Forest", "Neural Network", "Naive Bayes")
        #trainClassifierComboBox.current(0)
        trainClassifierComboBox.pack(padx=5, expand=True)
        trainButton = Button(trainFrame, text="Train classifier", command=lambda: self.train(trainFilenameLabel, trainClassifierComboBox, testClassifierComboBox, trainMetricsDisplay))
        trainButton.pack(padx=5, pady=5)

        trainMetricsFrame = Frame(mainFrame)
        trainMetricsFrame.pack(fill=X)
        trainMetricsDisplay = Listbox(trainMetricsFrame, height=3)
        trainMetricsDisplay.pack(padx=5, pady=5)
        #clearButton = Button(trainMetricsFrame, text="clear metrics", command=lambda: self.clearMetrics(trainMetricsDisplay))
        #clearButton.pack(padx=5, pady=5)

        separateFrame = Frame(mainFrame)
        separateFrame.pack(fill=X)
        separateLabel = Label(separateFrame, text="---------------------------------------------------------------------------------------------")
        separateLabel.pack(padx=5, pady=5)

        testFrame = Frame(mainFrame)
        testFrame.pack(fill=X)
        testFilenameLabel = Label(testFrame, text="No File", width=15)
        testFilenameLabel.pack(side=LEFT, padx=5, pady=5)
        importTestButton = Button(testFrame, text="Import File", command=lambda: self.onImport(testFilenameLabel, False))
        importTestButton.pack(side=LEFT, padx=5, pady=5)
        testClassifierValue = StringVar()
        testClassifierComboBox = ttk.Combobox(testFrame, state="readonly", textvariable=testClassifierValue)
        testClassifierComboBox['values'] = initialValues #("Random Forest", "Neural Network", "Naive Bayes")
        #testClassifierComboBox.current(0)
        testClassifierComboBox.pack(padx=5, expand=True)
        testButton = Button(testFrame, text="Test classifier", command=lambda: self.test(testFilenameLabel, testClassifierComboBox, outputFileNameEntry.get()))
        testButton.pack(padx=5, pady=5)
        outputFileLabel = Label(testFrame, text="Name of the output file: ")
        outputFileLabel.pack(side=LEFT, padx=5, pady=15)
        outputFileNameEntry = Entry(testFrame)
        outputFileNameEntry.pack(side=LEFT, padx=5, pady=15, expand=True)

        nb.add(mainFrame, text="main")

    def clearMetrics(self, trainMetricsDisplay):
        trainMetricsDisplay.delete(0,trainMetricsDisplay.size())

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
        pdDistrictComboBox['values']=("NORTHERN", "PARK", "INGLESIDE", "BAYVIEW", "RICHMOND", "CENTRAL", "TARAVAL", "TENDERLOIN", "MISSION", "SOUTHERN")
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

        timeFrame = Frame(mainFrame)
        timeFrame.pack(fill=X)
        timeLabel = Label(timeFrame, text="Time when crime was committed:")
        timeLabel.pack(side=LEFT, padx=5, pady=5)
        hourValue = StringVar()
        hourComboBox = ttk.Combobox(timeFrame, textvariable=hourValue, width=5)
        hourComboBox['values'] = ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23')
        hourComboBox.current(0)
        hourComboBox.pack(padx=5)

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
        predictButton = Button(predictFrame, text="Predict Category", command= lambda: self.onPredictValue(addressEntry,pdDistrictComboBox,ttkcal,predictClassifierComboBox.get(), outputDisplay))
        predictButton.pack(padx=5, pady=5)

        outputFrame = Frame(mainFrame)
        outputFrame.pack(fill=X)
        outputDisplay = Listbox(outputFrame, height=3)
        outputDisplay.pack(padx=5, pady=5)


        nb.add(mainFrame, text="Predict")


    def onExit(self):
        self.quit()

    def modifyValuesList(self, trainClassifierComboBox, testClassifierComboBox):
        if str(trainClassifierComboBox.get()) not in initialValues:
            initialValues.append(trainClassifierComboBox.get())
            testClassifierComboBox['values'] = initialValues

    def train(self, trainFilenameLabel, trainClassifierComboBox, testClassifierComboBox, trainMetricsDisplay):
        if trainFilenameLabel["text"] == "No File":
            print("no training file available")
        else:
            if trainClassifierComboBox.get() == "Random Forest":
                global rfPool
                rfPool, cfMatrix, rfPrecision, rfRecall, rfCorrectlyClassify = randomForest("preprocessed_data.csv")
                #print rfPool
                trainMetricsDisplay.insert(1,"precision:"+str(rfPrecision))
                trainMetricsDisplay.insert(2,"recall:"+str(rfRecall))
                trainMetricsDisplay.insert(3, "accuracy:"+str(rfCorrectlyClassify))
                self.modifyValuesList(trainClassifierComboBox, testClassifierComboBox)

            elif trainClassifierComboBox.get() == "Naive Bayes":
                global clfNaiveBayes
                precision, recall, accuracy, clfNaiveBayes = NB_clf_system("preprocessed_data.csv")
                self.modifyValuesList(trainClassifierComboBox, testClassifierComboBox)
                trainMetricsDisplay.insert(1,"precision:"+str(precision))
                trainMetricsDisplay.insert(2,"recall:"+str(recall))
                trainMetricsDisplay.insert(3, "accuracy:"+str(accuracy))
                print precision, recall, accuracy
            elif trainClassifierComboBox.get() == "Neural Network":
                train()
                print("missing to add this module")
            elif trainClassifierComboBox.get() == "KNN":
                print "missing to add this module"
            else:
                print("No classifier selected")

    def test(self, testFilenameLabel, testClassifierComboBox, fOutput):
        categoryList = [line.rstrip() for line in open('dictionary.txt')]
        if testFilenameLabel["text"] == "No File":
            print("no testing file available")
        else:
            if fOutput == "":
                mbox.showerror("Output File Name", "No output file name was typed in the entry field")
            else:
                if testClassifierComboBox.get() == "Random Forest":
                    testingData = pd.read_csv("preprocessed_testing.csv")
                    results = randomForestPredicted(testingData, 39, rfPool)
                    print(results)
                elif testClassifierComboBox.get() == "Naive Bayes":
                    testingData = pd.read_csv("preprocessed_testing.csv")
                    results = getPredictions(clfNaiveBayes, testingData)
                    print(results)
                elif testClassifierComboBox.get() == "Neural Network":
                    print("missing to add this module")
                elif testClassifierComboBox.get() == "KNN":
                    print("missing to add this module")
                else:
                    print("No classifier selected")

                resultFile = open(fOutput, 'w')
                for elem in results:
                    resultFile.write(categoryList[elem]+"\n")


    def onImport(self, trainFilenameLabel, isTraining):
        # isTraining is a boolean variable that indicates if the preprocess is for training or testing
        ftypes = [('All files', '*'),('Python files', '*py')]
        openFileWindow = tkFileDialog.Open(self, filetypes = ftypes)
        filename = openFileWindow.show()
        if filename == "":
            trainFilenameLabel["text"] = "No File"
        else:
            preprocess(filename, isTraining)
            mbox.showinfo("Information", "Preprocesssing file. When preprocess is done, file name will be shown in the file label")
            filename = filename.split("/")
            trainFilenameLabel["text"] = filename[-1]

    def onPredictValue(self, addressEntry, pdDistrictComboBox, ttkcal, clf, outputDisplay):

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
            fOut.write(str(dateFileFormat)+",,,"+str(dayOfTheWeekValue)+","+str(pdDistrictComboBox.get())+",,"+str(addressEntry.get())+","+str(lng)+","+str(lat))
            fOut.close()

            prepFile = open("svpFilePreprocessed.csv", 'w')
            prepFile.write("0,0,0,0,0,0,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,Friday,Monday,Saturday,Sunday,Thursday,Tuesday,Wednesday,BAYVIEW,CENTRAL,INGLESIDE,MISSION,NORTHERN,PARK,RICHMOND,SOUTHERN,TARAVAL,TENDERLOIN,X,Y\n")
            prepFile.write("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
            prepFile.close()

            preprocess("singlePredictionData.csv", False)
            singleValue = pd.read_csv("preprocessed_testing.csv")
            svpData = pd.read_csv("svpFilePreprocessed.csv")
            incompleteHeaders = singleValue.columns
            incompleteHeaders = incompleteHeaders.delete(0)
            completeHeaders = svpData.columns

            for elem in incompleteHeaders:
                svpData[elem][0] = singleValue[elem][0]


            if clf == "Random Forest":
                categNum = randomForestPredicted(svpData, 39, rfPool)
            elif clf == "Naive Bayes":
                categNum = predict_instance(clfNaiveBayes, svpData)
            elif clf == "Neural Network":
                categNum = [0]
                print "missing module"

            categoryList = [line.rstrip() for line in open('dictionary.txt')]
            outputDisplay.insert(1, categoryList[categNum[0]])

            #print categNumRandomForest, categNumNB


        else:
            mbox.showinfo("Warning", "Classifier has not been trained")


    def getGeolocation(self, latitudeLabel, longitudeLabel, addressEntry):
        if addressEntry.get() == "":
            latitudeLabel.config(text="Lat changed")
            longitudeLabel.config(text="Long changed")
        else:
            latitudeLabel.config(text=addressEntry.get())
            longitudeLabel.config(text="entry long")

    def onHelp(self):
        topLevelWindow = Toplevel()
        instrucLabel = Label(topLevelWindow, text="Instructions...")
        instrucLabel.pack()
        topLevelWindow.focus_force()

if __name__ == '__main__':

    NotebookDemo().mainloop()
