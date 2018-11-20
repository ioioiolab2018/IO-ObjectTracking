import sys
import algorithms
from tkinter import *


def main():
    root = Tk()
    root.title("Object tracking")
    frame = Frame(root, height=200, width=200)
    frame.pack(fill=None, expand=False)
    frame.place(relx=.5, rely=.4, anchor="c")
    entry = Entry(frame, width=30, bd=3)

    def getCam():
        adress = "0"
        adress = entry.get()
        if adress.isdigit():
            return int(adress)
        else:
            return adress

    def runExit():
        exit(0)

    def runColor():
        algorithms.color_run(getCam())
        print("Color")

    def runCsrt():
        algorithms.csrt_run(getCam())
        print("Csrt")

    def runFace():
        algorithms.face_run(getCam())
        print("Face")

    label = Label(frame, text="Camera address:")
    label.pack()

    entry.pack()
    entry.insert(0, "0")
    # entry.insert(0, "http://192.168.0.11:8080/mjpeg")
    colorButton = Button(frame, text="Color Tracking", command=runColor)
    csrtButton = Button(frame, text="Csrt Tracking", command=runCsrt)
    faceButton = Button(frame, text="Face Tracking", command=runFace)
    exitButton = Button(frame, text="Exit", command=runExit)
    colorButton.pack()
    csrtButton.pack()
    faceButton.pack()
    exitButton.pack()

    root.mainloop()


if __name__ == '__main__':
    main()
